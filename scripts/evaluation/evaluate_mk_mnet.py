"""Detailed evaluation for MK-MNet / DAMK-Net on BUSI.

Reports per-class and lesion-only segmentation metrics, plus classification
per-class precision/recall/F1, computed both WITHOUT and WITH the
prediction-refining (PR) coherence rule, so the effect of PR on the normal
class is visible directly.

Prediction-refining (adapted from Aumente-Maestro et al., CMPB 2025):
  (a) classification refinement: if the predicted tumour area is below a small
      threshold, force the class to "normal";
  (b) segmentation refinement: if the predicted class is "normal", empty the
      segmentation mask.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from model import MKMNet
from scripts.training.train_mk_mnet import (
    BUSI_CLASSES,
    BUSIMultiTaskDataset,
    build_transforms,
)


NORMAL_INDEX = BUSI_CLASSES.index("normal")
LESION_INDICES = [i for i, name in enumerate(BUSI_CLASSES) if name != "normal"]
SMOOTH = 1e-6


def prediction_refine(class_predictions, masks, area_threshold):
    """Enforce coherence between classification and segmentation.

    class_predictions: (B,) long tensor of predicted class indices.
    masks: (B, 1, H, W) binary float tensor of predicted masks.
    Returns refined (class_predictions, masks).
    """
    tumor_area = masks.flatten(start_dim=1).mean(dim=1)
    refined_classes = class_predictions.clone()
    refined_masks = masks.clone()

    refined_classes[tumor_area < area_threshold] = NORMAL_INDEX
    refined_masks[refined_classes == NORMAL_INDEX] = 0.0
    return refined_classes, refined_masks


def per_sample_segmentation(masks, ground_truth):
    predictions = masks.flatten(start_dim=1)
    targets = ground_truth.flatten(start_dim=1)
    intersection = (predictions * targets).sum(dim=1)
    prediction_pixels = predictions.sum(dim=1)
    mask_pixels = targets.sum(dim=1)
    union = prediction_pixels + mask_pixels - intersection

    dice = (2 * intersection + SMOOTH) / (prediction_pixels + mask_pixels + SMOOTH)
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return dice.cpu().numpy(), iou.cpu().numpy()


def summarize(labels, class_predictions, dice, iou):
    labels = np.asarray(labels)
    class_predictions = np.asarray(class_predictions)
    dice = np.asarray(dice)
    iou = np.asarray(iou)

    metrics = {
        "accuracy": float(np.mean(class_predictions == labels)),
        "dice_all": float(np.mean(dice)),
        "iou_all": float(np.mean(iou)),
    }

    lesion_mask = np.isin(labels, LESION_INDICES)
    metrics["dice_lesion"] = float(np.mean(dice[lesion_mask])) if lesion_mask.any() else float("nan")
    metrics["iou_lesion"] = float(np.mean(iou[lesion_mask])) if lesion_mask.any() else float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        class_predictions,
        labels=list(range(len(BUSI_CLASSES))),
        zero_division=0,
    )

    per_class = {}
    for index, name in enumerate(BUSI_CLASSES):
        class_mask = labels == index
        per_class[name] = {
            "dice": float(np.mean(dice[class_mask])) if class_mask.any() else float("nan"),
            "iou": float(np.mean(iou[class_mask])) if class_mask.any() else float("nan"),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(class_mask.sum()),
        }
    metrics["per_class"] = per_class
    return metrics


def build_run_dir(args):
    return (
        args.outputs_root
        / "busi"
        / "mk_mnet"
        / f"img_{args.image_size}"
        / f"width_{args.width_mult:g}"
        / f"oversample_{int(args.oversample)}"
        / f"lambda_{args.lambda_weight:g}"
        / f"lr_{args.learning_rate:g}"
        / f"weight_decay_{args.weight_decay:g}"
        / f"patience_{args.patience}"
    )


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = build_run_dir(args)
    checkpoint = args.checkpoint or run_dir / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    _, eval_transform = build_transforms(args.image_size)
    dataset = BUSIMultiTaskDataset(args.dataset_dir / "test", eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"test: {len(dataset)} image-label-mask samples")

    model = MKMNet(num_classes=len(BUSI_CLASSES), deep_supervision=True, width_mult=args.width_mult)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    labels = []
    probabilities = []
    raw = {"classes": [], "dice": [], "iou": []}
    refined = {"classes": [], "dice": [], "iou": []}

    with torch.inference_mode():
        for images, batch_labels, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            classification_logits, segmentation_logits = outputs[0], outputs[1]

            class_predictions = classification_logits.argmax(dim=1)
            predicted_masks = (torch.sigmoid(segmentation_logits) >= args.threshold).float()

            refined_classes, refined_masks = prediction_refine(
                class_predictions, predicted_masks, args.area_threshold
            )

            raw_dice, raw_iou = per_sample_segmentation(predicted_masks, masks)
            refined_dice, refined_iou = per_sample_segmentation(refined_masks, masks)

            labels.extend(batch_labels.numpy())
            probabilities.extend(torch.softmax(classification_logits, dim=1).cpu().numpy())
            raw["classes"].extend(class_predictions.cpu().numpy())
            raw["dice"].extend(raw_dice)
            raw["iou"].extend(raw_iou)
            refined["classes"].extend(refined_classes.cpu().numpy())
            refined["dice"].extend(refined_dice)
            refined["iou"].extend(refined_iou)

    raw_metrics = summarize(labels, raw["classes"], raw["dice"], raw["iou"])
    refined_metrics = summarize(labels, refined["classes"], refined["dice"], refined["iou"])

    try:
        auc = roc_auc_score(np.asarray(labels), np.asarray(probabilities), multi_class="ovr")
    except ValueError:
        auc = float("nan")
    raw_metrics["auc"] = auc
    refined_metrics["auc"] = auc

    print_report("WITHOUT prediction-refining", raw_metrics)
    print_report(f"WITH prediction-refining (area_threshold={args.area_threshold:g})", refined_metrics)

    output_path = args.output or run_dir / "metrics_detailed.csv"
    save_metrics(output_path, args, raw_metrics, refined_metrics)
    print(f"\nCheckpoint: {checkpoint}")
    print(f"Detailed metrics saved to: {output_path}")


def print_report(title, metrics):
    print(f"\n===== {title} =====")
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f} "
        f"| Dice(all): {metrics['dice_all']:.4f} | IoU(all): {metrics['iou_all']:.4f}"
    )
    print(
        f"Lesion-only  Dice: {metrics['dice_lesion']:.4f} | IoU: {metrics['iou_lesion']:.4f}  "
        f"(benign+malignant, comparable to MK-UNet/CMUNeXt)"
    )
    print(f"{'Class':<12}{'Dice':>8}{'IoU':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'N':>6}")
    for name in BUSI_CLASSES:
        row = metrics["per_class"][name]
        print(
            f"{name:<12}{row['dice']:>8.4f}{row['iou']:>8.4f}"
            f"{row['precision']:>8.4f}{row['recall']:>8.4f}{row['f1']:>8.4f}{row['support']:>6d}"
        )


def save_metrics(path, args, raw_metrics, refined_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Refining", "Scope", "Metric", "Benign", "Malignant", "Normal", "Overall", "Lesion_Only"]
        )
        for refining_name, metrics in (("raw", raw_metrics), ("refined", refined_metrics)):
            per_class = metrics["per_class"]
            for metric_name in ("dice", "iou"):
                writer.writerow(
                    [
                        refining_name,
                        "segmentation",
                        metric_name,
                        per_class["benign"][metric_name],
                        per_class["malignant"][metric_name],
                        per_class["normal"][metric_name],
                        metrics[f"{metric_name}_all"],
                        metrics[f"{metric_name}_lesion"],
                    ]
                )
            for metric_name in ("precision", "recall", "f1"):
                writer.writerow(
                    [
                        refining_name,
                        "classification",
                        metric_name,
                        per_class["benign"][metric_name],
                        per_class["malignant"][metric_name],
                        per_class["normal"][metric_name],
                        metrics["accuracy"] if metric_name == "f1" else "",
                        "",
                    ]
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Detailed per-class evaluation of MK-MNet on BUSI.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--oversample", action="store_true", help="Locate the oversampled-training checkpoint.")
    parser.add_argument("--lambda", dest="lambda_weight", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=0.005,
        help="Tumour-area fraction below which an image is refined to 'normal'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
