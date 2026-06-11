import argparse
import csv
import random
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import FocalLoss, RCBAMMNet


BUSI_CLASSES = ("benign", "malignant", "normal")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class BUSIMultiTaskDataset(Dataset):
    def __init__(self, split_dir, transform):
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(BUSI_CLASSES):
            images_dir = split_dir / class_name / "images"
            masks_dir = split_dir / class_name / "masks"

            if not images_dir.exists() or not masks_dir.exists():
                raise FileNotFoundError(
                    f"Missing images or masks directory for {class_name}: {split_dir}"
                )

            image_paths = sorted(
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                mask_path = masks_dir / f"{image_path.stem}_mask.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing mask: {mask_path}")
                self.samples.append((image_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = np.asarray(image.convert("RGB")).copy()
        with Image.open(mask_path) as mask:
            mask = np.asarray(mask.convert("L")).copy()

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()
        return image, label, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits).flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def should_stop(self, score):
        if self.best_score is None or score >= self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size):
    train_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=20,
                p=0.5,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    eval_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return train_transform, eval_transform


def build_loaders(dataset_dir, image_size, batch_size, num_workers):
    train_transform, eval_transform = build_transforms(image_size)
    transforms = {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }
    loaders = {}

    for split, transform in transforms.items():
        split_dir = dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing BUSI split: {split_dir}")

        dataset = BUSIMultiTaskDataset(split_dir, transform)
        print(f"{split}: {len(dataset)} image-label-mask samples")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def segmentation_scores(logits, masks):
    predictions = (torch.sigmoid(logits) >= 0.5).float().flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)
    intersection = (predictions * masks).sum(dim=1)
    prediction_pixels = predictions.sum(dim=1)
    mask_pixels = masks.sum(dim=1)
    union = prediction_pixels + mask_pixels - intersection
    smooth = 1e-6

    dice = (2 * intersection + smooth) / (
        prediction_pixels + mask_pixels + smooth
    )
    iou = (intersection + smooth) / (union + smooth)
    return dice.sum().item(), iou.sum().item()


def run_epoch(
    model,
    loader,
    classification_criterion,
    segmentation_criterion,
    segmentation_weight,
    device,
    optimizer=None,
    description=None,
    collect_probabilities=False,
):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "segmentation_loss": 0.0,
        "correct": 0,
        "dice": 0.0,
        "iou": 0.0,
        "images": 0,
    }
    all_labels = []
    all_probabilities = []

    with torch.set_grad_enabled(is_training):
        for images, labels, masks in tqdm(loader, desc=description, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            if is_training:
                optimizer.zero_grad()

            classification_logits, segmentation_logits = model(images)
            classification_loss = classification_criterion(
                classification_logits, labels
            )
            segmentation_loss = segmentation_criterion(segmentation_logits, masks)
            loss = classification_loss + segmentation_weight * segmentation_loss

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            dice, iou = segmentation_scores(segmentation_logits, masks)
            totals["loss"] += loss.item() * batch_size
            totals["classification_loss"] += classification_loss.item() * batch_size
            totals["segmentation_loss"] += segmentation_loss.item() * batch_size
            totals["correct"] += (
                classification_logits.argmax(dim=1) == labels
            ).sum().item()
            totals["dice"] += dice
            totals["iou"] += iou
            totals["images"] += batch_size

            if collect_probabilities:
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(
                    torch.softmax(classification_logits, dim=1).cpu().numpy()
                )

    count = totals["images"]
    metrics = {
        "loss": totals["loss"] / count,
        "classification_loss": totals["classification_loss"] / count,
        "segmentation_loss": totals["segmentation_loss"] / count,
        "accuracy": 100 * totals["correct"] / count,
        "dice": totals["dice"] / count,
        "iou": totals["iou"] / count,
    }

    if collect_probabilities:
        labels = np.asarray(all_labels)
        probabilities = np.asarray(all_probabilities)
        try:
            metrics["auc"] = roc_auc_score(labels, probabilities, multi_class="ovr")
        except ValueError:
            metrics["auc"] = float("nan")

    return metrics


def joint_score(metrics):
    return 0.5 * (metrics["accuracy"] / 100.0 + metrics["dice"])


def save_epoch_log(path, rows):
    columns = [
        "Epoch",
        "Train_Loss",
        "Val_Loss",
        "Train_Classification_Loss",
        "Val_Classification_Loss",
        "Train_Segmentation_Loss",
        "Val_Segmentation_Loss",
        "Train_Accuracy",
        "Val_Accuracy",
        "Train_Dice",
        "Val_Dice",
        "Train_IoU",
        "Val_IoU",
        "Val_Joint_Score",
    ]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def save_result(path, args, parameter_count, best_scores, test_metrics, duration):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerows(
            [
                ["Image_Size", args.image_size],
                ["Segmentation_Weight", args.segmentation_weight],
                ["Parameters", parameter_count],
                ["Best_Val_Accuracy", best_scores["accuracy"]],
                ["Best_Val_Dice", best_scores["dice"]],
                ["Best_Val_Joint_Score", best_scores["joint"]],
                ["Test_Accuracy", test_metrics["accuracy"]],
                ["Test_AUC", test_metrics["auc"]],
                ["Test_Dice", test_metrics["dice"]],
                ["Test_IoU", test_metrics["iou"]],
                ["Test_Loss", test_metrics["loss"]],
                ["Training_Time_Min", duration / 60],
            ]
        )


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = (
        args.output_dir
        / "busi"
        / "r_cbam_mnet"
        / f"img_{args.image_size}"
        / f"seg_weight_{args.segmentation_weight:g}"
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_loaders(
        args.dataset_dir,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    model = RCBAMMNet(num_classes=len(BUSI_CLASSES)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("Model: R-CBAM MNet")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Segmentation weight: {args.segmentation_weight}")

    class_weights = torch.tensor([1.0, 1.0, 1.0], device=device)
    classification_criterion = FocalLoss(gamma=2, alpha=class_weights)
    segmentation_criterion = SegmentationLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=args.patience)

    checkpoints = {
        "accuracy": output_dir / "best_classification.pt",
        "dice": output_dir / "best_segmentation.pt",
        "joint": output_dir / "best_joint.pt",
    }
    best_scores = {"accuracy": -1.0, "dice": -1.0, "joint": -1.0}
    epoch_rows = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            classification_criterion,
            segmentation_criterion,
            args.segmentation_weight,
            device,
            optimizer=optimizer,
            description=f"Train {epoch:03d}",
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            classification_criterion,
            segmentation_criterion,
            args.segmentation_weight,
            device,
            description=f"Val {epoch:03d}",
        )
        scheduler.step()
        val_joint = joint_score(val_metrics)

        epoch_rows.append(
            [
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                train_metrics["classification_loss"],
                val_metrics["classification_loss"],
                train_metrics["segmentation_loss"],
                val_metrics["segmentation_loss"],
                train_metrics["accuracy"],
                val_metrics["accuracy"],
                train_metrics["dice"],
                val_metrics["dice"],
                train_metrics["iou"],
                val_metrics["iou"],
                val_joint,
            ]
        )
        save_epoch_log(output_dir / "epoch_log.csv", epoch_rows)

        print(
            f"Epoch {epoch:03d} | Val Acc: {val_metrics['accuracy']:.2f}% "
            f"| Val Dice: {val_metrics['dice']:.4f} "
            f"| Joint: {val_joint:.4f}"
        )

        current_scores = {
            "accuracy": val_metrics["accuracy"],
            "dice": val_metrics["dice"],
            "joint": val_joint,
        }
        for name, score in current_scores.items():
            if score > best_scores[name]:
                best_scores[name] = score
                torch.save(model.state_dict(), checkpoints[name])

        if early_stopping.should_stop(val_joint):
            print(f"Early stopping at epoch {epoch}")
            break

    duration = time.time() - start_time
    model.load_state_dict(
        torch.load(checkpoints["joint"], map_location=device, weights_only=True)
    )
    test_metrics = run_epoch(
        model,
        loaders["test"],
        classification_criterion,
        segmentation_criterion,
        args.segmentation_weight,
        device,
        description="Test",
        collect_probabilities=True,
    )
    save_result(
        output_dir / "result.csv",
        args,
        parameter_count,
        best_scores,
        test_metrics,
        duration,
    )

    print(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test IoU: {test_metrics['iou']:.4f}")
    print(f"Outputs: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train R-CBAM MNet on BUSI.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--segmentation-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
