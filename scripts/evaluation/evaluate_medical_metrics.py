"""Unified detailed medical evaluation for BUSI classification-capable models."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import MKMNet, MedNet, MedNetMultiTask, RCBAMMNet


BUSI_CLASSES = ("benign", "malignant", "normal")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SMOOTH = 1e-6


class BUSIEvaluationDataset(Dataset):
    def __init__(self, dataset_dir, image_size, split):
        self.image_size = image_size
        self.split = split
        self.samples = []
        for label, class_name in enumerate(BUSI_CLASSES):
            images_dir = dataset_dir / split / class_name / "images"
            if not images_dir.is_dir():
                raise FileNotFoundError(f"Missing {split} image directory: {images_dir}")
            images = sorted(
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise ValueError(f"No {split} images found in: {images_dir}")
            for image_path in images:
                mask_path = (
                    image_path.parent.parent
                    / "masks"
                    / f"{image_path.stem}_mask.png"
                )
                if class_name != "normal" and not mask_path.is_file():
                    raise FileNotFoundError(f"Missing lesion mask: {mask_path}")
                self.samples.append((image_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1)
        image_tensor = (image_tensor - MEAN) / STD

        if mask_path.is_file():
            with Image.open(mask_path) as source:
                mask = source.convert("L").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
            mask_array = (np.asarray(mask) > 0).astype(np.float32)
        else:
            mask_array = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        return image_tensor, label, torch.from_numpy(mask_array).unsqueeze(0), str(image_path)


def parse_bool(value):
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected true or false")
    return lowered == "true"


def build_model(args):
    if args.model == "mednet":
        return MedNet(num_classes=len(BUSI_CLASSES), use_cbam=args.cbam)
    if args.model == "multitask":
        return MedNetMultiTask(
            num_classes=len(BUSI_CLASSES),
            num_segmentation_classes=1,
            use_cbam=args.cbam,
        )
    if args.model == "mk_mnet":
        return MKMNet(
            num_classes=len(BUSI_CLASSES),
            deep_supervision=True,
            width_mult=args.width_mult,
        )
    if args.model == "r_cbam_mnet":
        return RCBAMMNet(num_classes=len(BUSI_CLASSES))
    raise ValueError(f"Unsupported model: {args.model}")


def load_state_dict(checkpoint, device):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def unpack_outputs(model_name, output):
    if model_name == "mednet":
        return output[0] if isinstance(output, tuple) else output, None
    if not isinstance(output, tuple) or len(output) < 2:
        raise RuntimeError(f"{model_name} did not return classification and segmentation logits")
    return output[0], output[1]


def confusion_counts(labels, predictions):
    matrix = np.zeros((len(BUSI_CLASSES), len(BUSI_CLASSES)), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(labels, predictions, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    matrix = confusion_counts(labels, predictions)
    total = int(matrix.sum())
    per_class = []

    for index, class_name in enumerate(BUSI_CLASSES):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        precision = safe_divide(tp, tp + fp)
        sensitivity = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        f1 = safe_divide(2 * precision * sensitivity, precision + sensitivity)
        binary_labels = (labels == index).astype(np.int64)
        try:
            auc = float(roc_auc_score(binary_labels, probabilities[:, index]))
        except ValueError:
            auc = float("nan")
        per_class.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": sensitivity,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1": f1,
                "auc": auc,
                "support": int(matrix[index, :].sum()),
            }
        )

    try:
        macro_auc = float(roc_auc_score(labels, probabilities, multi_class="ovr"))
    except ValueError:
        macro_auc = float("nan")
    summary = {
        "accuracy": safe_divide(np.trace(matrix), total),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "macro_auc_ovr": macro_auc,
    }
    validate_classification_metrics(matrix, per_class, summary, total)
    return summary, per_class, matrix


def validate_classification_metrics(matrix, per_class, summary, sample_count):
    if int(matrix.sum()) != sample_count:
        raise AssertionError("Confusion-matrix total does not match test-set size")
    if sum(row["support"] for row in per_class) != sample_count:
        raise AssertionError("Per-class support does not match test-set size")
    for index, row in enumerate(per_class):
        if not np.isclose(row["sensitivity"], row["recall"]):
            raise AssertionError("Sensitivity must equal recall")
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = sample_count - tp - fp - fn
        expected_specificity = safe_divide(tn, tn + fp)
        if not np.isclose(row["specificity"], expected_specificity):
            raise AssertionError("Specificity does not match one-vs-rest TN/(TN+FP)")
    accuracy = safe_divide(np.trace(matrix), matrix.sum())
    if not np.isclose(summary["accuracy"], accuracy):
        raise AssertionError("Summary accuracy does not match confusion matrix")


def normalized_confusion(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )
    populated = row_sums[:, 0] > 0
    if populated.any() and not np.allclose(normalized[populated].sum(axis=1), 1.0):
        raise AssertionError("Normalized confusion-matrix rows must sum to one")
    return normalized


def segmentation_per_sample(predictions, targets):
    predictions = predictions.reshape(len(predictions), -1).astype(bool)
    targets = targets.reshape(len(targets), -1).astype(bool)
    intersection = np.logical_and(predictions, targets).sum(axis=1)
    prediction_pixels = predictions.sum(axis=1)
    target_pixels = targets.sum(axis=1)
    union = np.logical_or(predictions, targets).sum(axis=1)
    dice = (2 * intersection + SMOOTH) / (prediction_pixels + target_pixels + SMOOTH)
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return dice, iou


def segmentation_metrics(labels, dice, iou):
    labels = np.asarray(labels)
    dice = np.asarray(dice)
    iou = np.asarray(iou)
    lesion = labels != BUSI_CLASSES.index("normal")
    summary = {
        "dice_all": float(dice.mean()),
        "iou_all": float(iou.mean()),
        "dice_lesion": float(dice[lesion].mean()),
        "iou_lesion": float(iou[lesion].mean()),
    }
    per_class = {}
    for index, class_name in enumerate(BUSI_CLASSES):
        selected = labels == index
        per_class[class_name] = {
            "dice": float(dice[selected].mean()),
            "iou": float(iou[selected].mean()),
        }
    return summary, per_class


def refine_predictions(class_predictions, masks, area_threshold):
    refined_classes = np.asarray(class_predictions).copy()
    refined_masks = np.asarray(masks).copy()
    areas = refined_masks.reshape(len(refined_masks), -1).mean(axis=1)
    normal_index = BUSI_CLASSES.index("normal")
    refined_classes[areas < area_threshold] = normal_index
    refined_masks[refined_classes == normal_index] = False
    return refined_classes, refined_masks, areas


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path, matrix, number_format):
    rows = []
    for index, class_name in enumerate(BUSI_CLASSES):
        row = {"true_class": class_name}
        for column, predicted_name in enumerate(BUSI_CLASSES):
            row[f"pred_{predicted_name}"] = number_format(matrix[index, column])
        rows.append(row)
    write_csv(path, ["true_class", *[f"pred_{name}" for name in BUSI_CLASSES]], rows)


def save_confusion_png(path, matrix, normalized, title):
    cell = 150
    left = 150
    top = 105
    canvas = Image.new("RGB", (left + cell * 3 + 20, top + cell * 3 + 45), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill="black")
    draw.text((left + cell, 38), "Predicted class", fill="black")
    draw.text((10, top + cell), "True class", fill="black")
    for index, name in enumerate(BUSI_CLASSES):
        draw.text((left + index * cell + 35, top - 28), name, fill="black")
        draw.text((40, top + index * cell + 60), name, fill="black")
    for row in range(3):
        for column in range(3):
            value = float(normalized[row, column])
            shade = int(255 - 180 * value)
            color = (shade, shade, 255)
            box = (
                left + column * cell,
                top + row * cell,
                left + (column + 1) * cell,
                top + (row + 1) * cell,
            )
            draw.rectangle(box, fill=color, outline="white", width=2)
            text = f"{int(matrix[row, column])}\n{value:.1%}"
            draw.multiline_text((box[0] + 48, box[1] + 52), text, fill="black", align="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BUSIEvaluationDataset(args.dataset_dir, args.image_size, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(args)
    model.load_state_dict(load_state_dict(args.checkpoint, device))
    model.to(device).eval()

    labels = []
    probabilities = []
    raw_classes = []
    raw_masks = []
    targets = []
    image_paths = []
    with torch.inference_mode():
        for images, batch_labels, batch_masks, batch_paths in loader:
            logits, segmentation_logits = unpack_outputs(
                args.model, model(images.to(device))
            )
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            raw_classes.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())
            targets.extend(batch_masks.numpy() >= 0.5)
            image_paths.extend(batch_paths)
            if segmentation_logits is not None:
                raw_masks.extend(
                    (torch.sigmoid(segmentation_logits).cpu().numpy() >= args.threshold)
                )

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    raw_classes = np.asarray(raw_classes)
    targets = np.asarray(targets)
    variants = {"raw": {"classes": raw_classes, "masks": None, "areas": None}}
    if raw_masks:
        raw_masks = np.asarray(raw_masks)
        variants["raw"]["masks"] = raw_masks
        refined_classes, refined_masks, areas = refine_predictions(
            raw_classes, raw_masks, args.area_threshold
        )
        variants["raw"]["areas"] = raw_masks.reshape(len(raw_masks), -1).mean(axis=1)
        variants["refined"] = {
            "classes": refined_classes,
            "masks": refined_masks,
            "areas": areas,
        }

    summary_rows = []
    per_class_rows = []
    segmentation_by_variant = {}
    for variant_name, variant in variants.items():
        classification, per_class, matrix = classification_metrics(
            labels, variant["classes"], probabilities
        )
        normalized = normalized_confusion(matrix)
        segmentation = {
            "dice_all": "",
            "iou_all": "",
            "dice_lesion": "",
            "iou_lesion": "",
        }
        segmentation_per_class = {}
        if variant["masks"] is not None:
            dice, iou = segmentation_per_sample(variant["masks"], targets)
            segmentation, segmentation_per_class = segmentation_metrics(labels, dice, iou)
            segmentation_by_variant[variant_name] = (dice, iou)
        summary_rows.append(
            {
                "variant": variant_name,
                "model": args.model,
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "samples": len(labels),
                **classification,
                **segmentation,
                "mask_threshold": args.threshold if variant["masks"] is not None else "",
                "area_threshold": args.area_threshold if variant_name == "refined" else "",
            }
        )
        for row in per_class:
            seg = segmentation_per_class.get(row["class"], {"dice": "", "iou": ""})
            per_class_rows.append(
                {"variant": variant_name, "model": args.model, "split": args.split, **row, **seg}
            )
        prefix = args.output_dir / f"confusion_matrix_{variant_name}"
        write_matrix_csv(prefix.with_name(prefix.name + "_counts.csv"), matrix, int)
        write_matrix_csv(
            prefix.with_name(prefix.name + "_normalized.csv"),
            normalized,
            lambda value: f"{value:.10f}",
        )
        save_confusion_png(
            prefix.with_suffix(".png"), matrix, normalized, f"{args.model} - {variant_name}"
        )

    summary_fields = list(summary_rows[0].keys())
    write_csv(args.output_dir / "summary.csv", summary_fields, summary_rows)
    per_class_fields = list(per_class_rows[0].keys())
    write_csv(args.output_dir / "per_class_metrics.csv", per_class_fields, per_class_rows)

    prediction_rows = []
    raw_dice_iou = segmentation_by_variant.get("raw", (None, None))
    refined_dice_iou = segmentation_by_variant.get("refined", (None, None))
    for index, image_path in enumerate(image_paths):
        row = {
            "image": image_path,
            "true_class": BUSI_CLASSES[int(labels[index])],
            "raw_prediction": BUSI_CLASSES[int(variants["raw"]["classes"][index])],
            "refined_prediction": "",
            **{f"prob_{name}": float(probabilities[index, class_index]) for class_index, name in enumerate(BUSI_CLASSES)},
            "predicted_mask_area": "",
            "raw_dice": "",
            "raw_iou": "",
            "refined_dice": "",
            "refined_iou": "",
        }
        if variants["raw"]["areas"] is not None:
            row["predicted_mask_area"] = float(variants["raw"]["areas"][index])
            row["raw_dice"] = float(raw_dice_iou[0][index])
            row["raw_iou"] = float(raw_dice_iou[1][index])
            row["refined_prediction"] = BUSI_CLASSES[int(variants["refined"]["classes"][index])]
            row["refined_dice"] = float(refined_dice_iou[0][index])
            row["refined_iou"] = float(refined_dice_iou[1][index])
        prediction_rows.append(row)
    write_csv(args.output_dir / "predictions.csv", list(prediction_rows[0].keys()), prediction_rows)

    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")
    print(f"Saved detailed medical evaluation to: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Detailed medical evaluation on BUSI.")
    parser.add_argument("--model", choices=("mednet", "multitask", "mk_mnet", "r_cbam_mnet"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--cbam", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--area-threshold", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
