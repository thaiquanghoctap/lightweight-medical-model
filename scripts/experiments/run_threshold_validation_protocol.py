"""Select prediction-refining tau on validation, then report once on test."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def score_row(row):
    return (
        float(row["macro_f1"]),
        float(row["accuracy"]),
        float(row["dice_lesion"]),
        float(row["dice_normal"]),
    )


def run_threshold_sweep(args, split):
    output_root = args.output_root / split
    command = [
        sys.executable,
        str(args.runner),
        "--model",
        args.model,
        "--checkpoint",
        str(args.checkpoint),
        "--dataset-dir",
        str(args.dataset_dir),
        "--output-root",
        str(output_root),
        "--evaluator",
        str(args.evaluator),
        "--split",
        split,
        "--image-size",
        str(args.image_size),
        "--width-mult",
        str(args.width_mult),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--mask-threshold",
        str(args.mask_threshold),
        "--area-thresholds",
        *[str(value) for value in args.area_thresholds],
    ]
    if args.resume:
        command.append("--resume")
    subprocess.run(command, check=True)
    rows = read_rows(output_root / "threshold_sensitivity.csv")
    if not rows:
        raise ValueError(f"No threshold rows found in {output_root}")
    return output_root, rows


def choose_threshold(rows):
    return max(rows, key=score_row)


def write_selection(path, best_val_row, test_row):
    fieldnames = [
        "selected_on_split",
        "selected_area_threshold",
        "validation_macro_f1",
        "validation_accuracy",
        "validation_dice_lesion",
        "validation_dice_normal",
        "test_accuracy",
        "test_macro_f1",
        "test_dice_lesion",
        "test_dice_normal",
        "test_malignant_sensitivity",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "selected_on_split": "val",
                "selected_area_threshold": best_val_row["area_threshold"],
                "validation_macro_f1": best_val_row["macro_f1"],
                "validation_accuracy": best_val_row["accuracy"],
                "validation_dice_lesion": best_val_row["dice_lesion"],
                "validation_dice_normal": best_val_row["dice_normal"],
                "test_accuracy": test_row["accuracy"],
                "test_macro_f1": test_row["macro_f1"],
                "test_dice_lesion": test_row["dice_lesion"],
                "test_dice_normal": test_row["dice_normal"],
                "test_malignant_sensitivity": test_row["malignant_sensitivity"],
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="Choose prediction-refining tau on validation, then report test once."
    )
    parser.add_argument("--model", choices=("mk_mnet", "multitask", "r_cbam_mnet"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/threshold-validation-protocol"),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/experiments/run_threshold_sensitivity.py"),
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("scripts/evaluation/evaluate_medical_metrics.py"),
    )
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

    args.output_root.mkdir(parents=True, exist_ok=True)

    _, val_rows = run_threshold_sweep(args, "val")
    best_val_row = choose_threshold(val_rows)
    selected_threshold = float(best_val_row["area_threshold"])

    test_args = argparse.Namespace(**vars(args))
    test_args.area_thresholds = [selected_threshold]
    _, test_rows = run_threshold_sweep(test_args, "test")
    test_row = test_rows[0]

    selection_path = args.output_root / "selected_threshold_summary.csv"
    write_selection(selection_path, best_val_row, test_row)

    print(f"Selected tau on validation: {selected_threshold:g}")
    print(f"Validation summary: {args.output_root / 'val' / 'threshold_sensitivity.csv'}")
    print(f"Test summary: {args.output_root / 'test' / 'threshold_sensitivity.csv'}")
    print(f"Selection summary: {selection_path}")


if __name__ == "__main__":
    main()
