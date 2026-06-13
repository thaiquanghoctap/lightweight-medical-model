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

from model import FocalLoss, MedNet


BUSI_CLASSES = ("benign", "malignant", "normal")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
NUM_CLASSES = len(BUSI_CLASSES)
ALPHA_WEIGHTS = [1.0, 1.0, 1.0]


class BUSIImageDataset(Dataset):
    def __init__(self, split_dir, transform):
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(BUSI_CLASSES):
            images_dir = split_dir / class_name / "images"
            if not images_dir.exists():
                raise FileNotFoundError(f"Missing BUSI images directory: {images_dir}")

            image_paths = sorted(
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_paths:
                raise ValueError(f"No BUSI images found in: {images_dir}")

            self.samples.extend((image_path, label) for image_path in image_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = np.asarray(image)

        image = self.transform(image=image)["image"]
        return image, label


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

        dataset = BUSIImageDataset(split_dir, transform)
        print(f"{split}: {len(dataset)} classification images")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
        )

    return loaders


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


def evaluate_test_set(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            total_loss += criterion(outputs, labels).item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    labels = np.asarray(all_labels)
    probs = np.asarray(all_probs)
    accuracy = 100 * np.mean(probs.argmax(axis=1) == labels)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return total_loss / len(loader), accuracy, auc


def write_epoch_log(log_path, rows):
    with log_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Train_Acc", "Val_Acc"])
        writer.writerows(rows)


def append_result(
    log_path,
    image_size,
    use_cbam,
    run,
    best_val_acc,
    test_loss,
    test_acc,
    test_auc,
    duration,
):
    should_write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if should_write_header:
            writer.writerow(
                [
                    "Image_Size",
                    "Use_CBAM",
                    "Run",
                    "Best_Val_Acc",
                    "Test_Acc",
                    "Test_AUC",
                    "Test_Loss",
                    "Train_Time_Min",
                ]
            )
        writer.writerow(
            [
                image_size,
                use_cbam,
                run,
                best_val_acc,
                test_acc,
                test_auc,
                test_loss,
                duration / 60,
            ]
        )


def train_one_run(args, device, image_size, use_cbam, run):
    cbam_name = "cbam" if use_cbam else "nocbam"
    output_dir = (
        args.output_dir
        / "busi"
        / "classification"
        / f"img_{image_size}"
        / cbam_name
        / f"run_{run}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n===== Image size: {image_size} | CBAM: {use_cbam} "
        f"| Run: {run}/{args.runs} ====="
    )
    set_seed(args.seed + run - 1)
    loaders = build_loaders(
        args.dataset_dir,
        image_size,
        args.batch_size,
        args.num_workers,
    )
    model = MedNet(num_classes=NUM_CLASSES, use_cbam=use_cbam).to(device)
    alpha_weights = torch.tensor(ALPHA_WEIGHTS, device=device)
    criterion = FocalLoss(gamma=2, alpha=alpha_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=args.patience)

    checkpoint_path = output_dir / "best_model.pt"
    epoch_rows = []
    best_val_acc = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        epoch_rows.append([epoch, train_loss, val_loss, train_acc, val_acc])
        print(
            f"Epoch {epoch:03d} | Train Acc: {train_acc:.2f}% "
            f"| Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        if early_stopping.should_stop(val_acc):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    duration = time.time() - start_time
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_acc, test_auc = evaluate_test_set(
        model, loaders["test"], criterion, device
    )

    write_epoch_log(output_dir / "epoch_log.csv", epoch_rows)
    result_path = args.output_dir / "busi" / "classification" / "result.csv"
    append_result(
        result_path,
        image_size,
        use_cbam,
        run,
        best_val_acc,
        test_loss,
        test_acc,
        test_auc,
        duration,
    )

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Outputs saved to: {output_dir}")
    print(f"Result appended to: {result_path}")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbam_values = {
        "true": [True],
        "false": [False],
        "both": [True, False],
    }[args.cbam]
    total_sessions = len(args.img_sizes) * len(cbam_values) * args.runs

    print("Dataset: busi")
    print(f"Device: {device}")
    print(f"Image sizes: {args.img_sizes}")
    print(f"CBAM modes: {cbam_values}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Total training sessions: {total_sessions}")

    for image_size in args.img_sizes:
        for use_cbam in cbam_values:
            for run in range(1, args.runs + 1):
                train_one_run(args, device, image_size, use_cbam, run)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedNet on BUSI images.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--img-sizes", type=int, nargs="+", default=[224])
    parser.add_argument("--cbam", choices=["true", "false", "both"], default="true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
