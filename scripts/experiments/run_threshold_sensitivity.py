"""Run prediction-refining threshold sensitivity for a trained BUSI checkpoint."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def slug_threshold(value):
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(args, threshold):
    output_dir = args.output_root / args.split / f"area_{slug_threshold(threshold)}"
    command = [
        sys.executable,
        str(args.evaluator),
        "--model",
        args.model,
        "--checkpoint",
        str(args.checkpoint),
        "--dataset-dir",
        str(args.dataset_dir),
        "--output-dir",
        str(output_dir),
        "--split",
        args.split,
        "--image-size",
        str(args.image_size),
        "--width-mult",
        str(args.width_mult),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--threshold",
        str(args.mask_threshold),
        "--area-threshold",
        str(threshold),
    ]
    if args.resume and (output_dir / "summary.csv").is_file():
        return output_dir
    subprocess.run(command, check=True)
    return output_dir


def collect_metrics(output_dir, threshold):
    summary_rows = read_csv(output_dir / "summary.csv")
    per_class_rows = read_csv(output_dir / "per_class_metrics.csv")
    refined = next(row for row in summary_rows if row["variant"] == "refined")
    normal = next(
        row
        for row in per_class_rows
        if row["variant"] == "refined" and row["class"] == "normal"
    )
    malignant = next(
        row
        for row in per_class_rows
        if row["variant"] == "refined" and row["class"] == "malignant"
    )
    return {
        "area_threshold": threshold,
        "accuracy": refined["accuracy"],
        "macro_f1": refined["macro_f1"],
        "macro_auc_ovr": refined["macro_auc_ovr"],
        "dice_lesion": refined["dice_lesion"],
        "iou_lesion": refined["iou_lesion"],
        "dice_normal": normal["dice"],
        "normal_specificity": normal["specificity"],
        "malignant_sensitivity": malignant["sensitivity"],
        "malignant_specificity": malignant["specificity"],
    }


def maybe_plot(output_root, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    thresholds = [float(row["area_threshold"]) for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    dice_lesion = [float(row["dice_lesion"]) for row in rows]
    dice_normal = [float(row["dice_normal"]) for row in rows]
    malignant_sensitivity = [float(row["malignant_sensitivity"]) for row in rows]

    plt.figure(figsize=(7.2, 4.5))
    plt.semilogx(thresholds, accuracy, marker="o", label="Accuracy")
    plt.semilogx(thresholds, dice_lesion, marker="o", label="Lesion Dice")
    plt.semilogx(thresholds, dice_normal, marker="o", label="Normal Dice")
    plt.semilogx(
        thresholds,
        malignant_sensitivity,
        marker="o",
        label="Malignant sensitivity",
    )
    plt.xlabel("Prediction-refining area threshold")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.02)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "threshold_sensitivity.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep prediction-refining area thresholds for BUSI models."
    )
    parser.add_argument("--model", choices=("mk_mnet", "multitask", "r_cbam_mnet"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/threshold-sensitivity"))
    parser.add_argument("--evaluator", type=Path, default=Path("scripts/evaluation/evaluate_medical_metrics.py"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--area-thresholds",
        nargs="+",
        type=float,
        default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2],
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = []
    for threshold in args.area_thresholds:
        output_dir = run_evaluation(args, threshold)
        rows.append(collect_metrics(output_dir, threshold))

    fields = list(rows[0].keys())
    write_csv(args.output_root / "threshold_sensitivity.csv", fields, rows)
    maybe_plot(args.output_root, rows)
    print(f"Wrote {args.output_root / 'threshold_sensitivity.csv'}")
    if (args.output_root / "threshold_sensitivity.png").is_file():
        print(f"Wrote {args.output_root / 'threshold_sensitivity.png'}")


if __name__ == "__main__":
    main()
