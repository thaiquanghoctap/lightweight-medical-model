import argparse
import csv
import random
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import FocalLoss, MedNetMultiTask


BUSI_CLASSES = ("benign", "malignant", "normal")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
NUM_CLASSES = len(BUSI_CLASSES)
ALPHA_WEIGHTS = [1.0, 1.0, 1.0]


class BUSIMultiTaskDataset(Dataset):
    def __init__(self, split_dir, transform):
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(BUSI_CLASSES):
            images_dir = split_dir / class_name / "images"
            masks_dir = split_dir / class_name / "masks"
            if not images_dir.exists():
                raise FileNotFoundError(f"Missing BUSI images directory: {images_dir}")
            if not masks_dir.exists():
                raise FileNotFoundError(f"Missing BUSI masks directory: {masks_dir}")

            image_paths = sorted(
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_paths:
                raise ValueError(f"No BUSI images found in: {images_dir}")

            for image_path in image_paths:
                mask_path = masks_dir / f"{image_path.stem}_mask.png"
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Missing mask for image {image_path.name}: {mask_path}"
                    )
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
                shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5
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
    split_transforms = {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }
    loaders = {}

    for split, transform in split_transforms.items():
        split_dir = dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing processed split: {split_dir}. Run preprocess.py first."
            )

        dataset = BUSIMultiTaskDataset(split_dir, transform)
        print(f"{split}: {len(dataset)} image-label-mask samples")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
        )

    return loaders


def calculate_segmentation_metrics(segmentation_logits, masks):
    predictions = (torch.sigmoid(segmentation_logits) > 0.5).float()
    predictions = predictions.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    intersection = (predictions * masks).sum(dim=1)
    prediction_pixels = predictions.sum(dim=1)
    mask_pixels = masks.sum(dim=1)
    union = prediction_pixels + mask_pixels - intersection
    smooth = 1e-6

    dice = (2 * intersection + smooth) / (prediction_pixels + mask_pixels + smooth)
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
    total_loss = 0.0
    total_classification_loss = 0.0
    total_segmentation_loss = 0.0
    total_correct = 0
    total_dice = 0.0
    total_iou = 0.0
    total_images = 0
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
            batch_dice, batch_iou = calculate_segmentation_metrics(
                segmentation_logits, masks
            )
            total_loss += loss.item() * batch_size
            total_classification_loss += classification_loss.item() * batch_size
            total_segmentation_loss += segmentation_loss.item() * batch_size
            total_correct += (
                classification_logits.argmax(dim=1) == labels
            ).sum().item()
            total_dice += batch_dice
            total_iou += batch_iou
            total_images += batch_size

            if collect_probabilities:
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(
                    torch.softmax(classification_logits, dim=1).cpu().numpy()
                )

    metrics = {
        "loss": total_loss / total_images,
        "classification_loss": total_classification_loss / total_images,
        "segmentation_loss": total_segmentation_loss / total_images,
        "accuracy": 100 * total_correct / total_images,
        "dice": total_dice / total_images,
        "iou": total_iou / total_images,
    }

    if collect_probabilities:
        labels = np.asarray(all_labels)
        probabilities = np.asarray(all_probabilities)
        try:
            metrics["auc"] = roc_auc_score(labels, probabilities, multi_class="ovr")
        except ValueError:
            metrics["auc"] = float("nan")

    return metrics


def write_epoch_log(log_path, rows):
    with log_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Epoch",
                "Train_Loss",
                "Val_Loss",
                "Train_Classification_Loss",
                "Val_Classification_Loss",
                "Train_Segmentation_Loss",
                "Val_Segmentation_Loss",
                "Train_Acc",
                "Val_Acc",
                "Train_Dice",
                "Val_Dice",
                "Train_IoU",
                "Val_IoU",
            ]
        )
        writer.writerows(rows)


def append_result(log_path, best_val_acc, test_metrics, duration, args):
    should_write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if should_write_header:
            writer.writerow(
                [
                    "Image_Size",
                    "Use_CBAM",
                    "Segmentation_Weight",
                    "Best_Val_Acc",
                    "Test_Acc",
                    "Test_AUC",
                    "Test_Dice",
                    "Test_IoU",
                    "Test_Loss",
                    "Test_Classification_Loss",
                    "Test_Segmentation_Loss",
                    "Train_Time_Min",
                ]
            )
        writer.writerow(
            [
                args.image_size,
                args.use_cbam,
                args.segmentation_weight,
                best_val_acc,
                test_metrics["accuracy"],
                test_metrics["auc"],
                test_metrics["dice"],
                test_metrics["iou"],
                test_metrics["loss"],
                test_metrics["classification_loss"],
                test_metrics["segmentation_loss"],
                duration / 60,
            ]
        )


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbam_name = "cbam" if args.use_cbam else "nocbam"
    output_dir = (
        args.output_dir
        / "busi"
        / "multi"
        / f"img_{args.image_size}"
        / cbam_name
        / f"seg_weight_{args.segmentation_weight:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset: busi multi-task")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"CBAM: {args.use_cbam}")
    print(f"Segmentation weight: {args.segmentation_weight}")

    loaders = build_loaders(
        args.dataset_dir,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    model = MedNetMultiTask(
        num_classes=NUM_CLASSES,
        num_segmentation_classes=1,
        use_cbam=args.use_cbam,
    ).to(device)
    alpha_weights = torch.tensor(ALPHA_WEIGHTS, device=device)
    classification_criterion = FocalLoss(gamma=2, alpha=alpha_weights)
    segmentation_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    early_stopping = EarlyStopping()

    checkpoint_path = output_dir / "best_model.pt"
    epoch_rows = []
    best_val_acc = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            classification_criterion,
            segmentation_criterion,
            args.segmentation_weight,
            device,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            classification_criterion,
            segmentation_criterion,
            args.segmentation_weight,
            device,
        )
        scheduler.step()
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
            ]
        )
        print(
            f"Epoch {epoch:03d} | Train Acc: {train_metrics['accuracy']:.2f}% "
            f"| Val Acc: {val_metrics['accuracy']:.2f}% "
            f"| Val Dice: {val_metrics['dice']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), checkpoint_path)

        if early_stopping.should_stop(val_metrics["accuracy"]):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    duration = time.time() - start_time
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = run_epoch(
        model,
        loaders["test"],
        classification_criterion,
        segmentation_criterion,
        args.segmentation_weight,
        device,
        description="Testing",
        collect_probabilities=True,
    )

    write_epoch_log(output_dir / "epoch_log.csv", epoch_rows)
    result_path = args.output_dir / "busi" / "multi" / "result.csv"
    append_result(result_path, best_val_acc, test_metrics, duration, args)

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test IoU: {test_metrics['iou']:.4f}")
    print(f"Outputs saved to: {output_dir}")
    print(f"Result appended to: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-task MedNet on BUSI.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cbam", choices=["true", "false"], default="true")
    parser.add_argument("--segmentation-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.use_cbam = args.cbam == "true"
    return args


if __name__ == "__main__":
    train(parse_args())
