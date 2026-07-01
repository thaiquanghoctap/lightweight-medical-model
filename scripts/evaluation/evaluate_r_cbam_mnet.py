"""Detailed evaluation for R-CBAM MNet on BUSI (decoder-ablation model).

Mirrors evaluate_mk_mnet.py so the proposed DAMK-Net and this Inception/CBAM-RI
decoder variant are measured on the exact same protocol: per-class and
lesion-only segmentation metrics, classification per-class precision/recall/F1,
both without and with prediction-refining.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model import RCBAMMNet
from scripts.training.train_r_cbam_mnet import (
    BUSI_CLASSES,
    BUSIMultiTaskDataset,
    build_transforms,
)
from scripts.evaluation.evaluate_mk_mnet import (
    per_sample_segmentation,
    prediction_refine,
    print_report,
    save_metrics,
    summarize,
)


def build_run_dir(args):
    return (
        args.outputs_root
        / "busi"
        / "r_cbam_mnet"
        / f"img_{args.image_size}"
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

    model = RCBAMMNet(num_classes=len(BUSI_CLASSES))
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

            classification_logits, segmentation_logits = model(images)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Detailed per-class evaluation of R-CBAM MNet on BUSI.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
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
