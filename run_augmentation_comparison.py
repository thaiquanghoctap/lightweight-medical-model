"""Train and evaluate the controlled MK-MNet augmentation comparison."""

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

from augmentation_policies import ALL_POLICIES, COMPARISON_POLICIES


def run_dir(args, policy, seed):
    return (
        args.outputs_root
        / "busi"
        / "mk_mnet"
        / f"img_{args.image_size}"
        / f"width_{args.width_mult:g}"
        / "oversample_1"
        / f"lambda_{args.lambda_weight:g}"
        / f"lr_{args.learning_rate:g}"
        / f"weight_decay_{args.weight_decay:g}"
        / f"patience_{args.patience}"
        / f"augmentation_{policy}"
        / f"seed_{seed}"
    )


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_first(path, predicate=None):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if predicate is None:
        return rows[-1]
    return next(row for row in rows if predicate(row))


def collect_run_metrics(policy, seed, training_dir, evaluation_dir):
    summary = read_first(
        evaluation_dir / "summary.csv", lambda row: row["variant"] == "raw"
    )
    malignant = read_first(
        evaluation_dir / "per_class_metrics.csv",
        lambda row: row["variant"] == "raw" and row["class"] == "malignant",
    )
    training = read_first(training_dir / "result.csv")
    return {
        "policy": policy,
        "seed": seed,
        "accuracy": summary["accuracy"],
        "balanced_accuracy": summary["macro_recall"],
        "macro_f1": summary["macro_f1"],
        "macro_auc_ovr": summary["macro_auc_ovr"],
        "dice_lesion": summary["dice_lesion"],
        "iou_lesion": summary["iou_lesion"],
        "malignant_precision": malignant["precision"],
        "malignant_sensitivity": malignant["sensitivity"],
        "malignant_specificity": malignant["specificity"],
        "malignant_f1": malignant["f1"],
        "malignant_auc": malignant["auc"],
        "training_time_min": training["Training_Time_Min"],
    }


def aggregate(rows):
    metrics = [
        key for key in rows[0] if key not in {"policy", "seed"}
    ]
    output = []
    for policy in dict.fromkeys(row["policy"] for row in rows):
        selected = [row for row in rows if row["policy"] == policy]
        result = {"policy": policy, "runs": len(selected)}
        for name in metrics:
            values = [float(row[name]) for row in selected]
            result[f"{name}_mean"] = statistics.mean(values)
            result[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def tex_escape(value):
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(character, character) for character in str(value))


def mean_std(row, metric_name):
    return (
        f"{float(row[f'{metric_name}_mean']):.4f} "
        f"$\\pm$ {float(row[f'{metric_name}_std']):.4f}"
    )


def generate_latex(path, rows):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Augmentation comparison trên MK-MNet width 0.25 (mean $\pm$ std).}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Policy & Balanced Acc. & Macro-F1 & Mal. Sens. & Mal. Spec. & Lesion Dice \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['policy'])} & {mean_std(row, 'balanced_accuracy')} & "
            f"{mean_std(row, 'macro_f1')} & {mean_std(row, 'malignant_sensitivity')} & "
            f"{mean_std(row, 'malignant_specificity')} & {mean_std(row, 'dice_lesion')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\normalsize",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_command(args, policy, seed):
    return [
        sys.executable,
        str(args.train_script),
        "--dataset-dir", str(args.dataset_dir),
        "--output-dir", str(args.outputs_root),
        "--image-size", str(args.image_size),
        "--width-mult", str(args.width_mult),
        "--lambda", str(args.lambda_weight),
        "--aux-weight", str(args.aux_weight),
        "--deep-supervision", "true",
        "--oversample",
        "--epochs", str(args.epochs),
        "--no-early-stop",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--seed", str(seed),
        "--augmentation-policy", policy,
        "--augmentation-experiment",
    ]


def evaluation_command(args, checkpoint, output_dir):
    return [
        sys.executable,
        str(args.evaluator),
        "--model", "mk_mnet",
        "--checkpoint", str(checkpoint),
        "--dataset-dir", str(args.dataset_dir),
        "--output-dir", str(output_dir),
        "--image-size", str(args.image_size),
        "--width-mult", str(args.width_mult),
        "--batch-size", str(args.evaluation_batch_size),
        "--num-workers", str(args.num_workers),
        "--threshold", "0.5",
        "--area-threshold", "0.005",
    ]


def run(args):
    args.results_root.mkdir(parents=True, exist_ok=True)
    latex_path = args.results_root / "augmentation_comparison_table.tex"
    latex_path.unlink(missing_ok=True)
    statuses = []
    run_metrics = []
    status_fields = ["policy", "seed", "training", "evaluation", "checkpoint"]

    for policy in args.policies:
        for seed in args.seeds:
            training_dir = run_dir(args, policy, seed)
            checkpoint = training_dir / "best_model.pt"
            result_csv = training_dir / "result.csv"
            evaluation_dir = args.results_root / policy / f"seed_{seed}"
            print(f"\n===== {policy} | seed {seed} =====")

            training_status = "completed"
            if args.resume and checkpoint.is_file() and result_csv.is_file():
                print(f"Reuse completed training: {training_dir}")
            else:
                training_result = subprocess.run(train_command(args, policy, seed))
                if training_result.returncode != 0:
                    training_status = f"failed:{training_result.returncode}"

            evaluation_status = "not_run"
            if training_status == "completed" and checkpoint.is_file():
                evaluation_result = subprocess.run(
                    evaluation_command(args, checkpoint, evaluation_dir)
                )
                evaluation_status = (
                    "completed"
                    if evaluation_result.returncode == 0
                    else f"failed:{evaluation_result.returncode}"
                )

            statuses.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "training": training_status,
                    "evaluation": evaluation_status,
                    "checkpoint": str(checkpoint),
                }
            )
            write_csv(args.results_root / "comparison_status.csv", status_fields, statuses)

            if evaluation_status == "completed":
                run_metrics.append(
                    collect_run_metrics(policy, seed, training_dir, evaluation_dir)
                )
                write_csv(
                    args.results_root / "comparison_runs.csv",
                    list(run_metrics[0]),
                    run_metrics,
                )
                summary = aggregate(run_metrics)
                write_csv(
                    args.results_root / "comparison_summary.csv",
                    list(summary[0]),
                    summary,
                )

    failed = [row for row in statuses if row["training"] != "completed" or row["evaluation"] != "completed"]
    if not failed and len(run_metrics) == len(args.policies) * len(args.seeds):
        generate_latex(latex_path, aggregate(run_metrics))
    print(f"\nCompleted: {len(statuses) - len(failed)}/{len(statuses)}")
    print(f"Failed: {len(failed)}")
    print(f"Status: {args.results_root / 'comparison_status.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled MK-MNet augmentation comparison.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--results-root", type=Path, default=Path("augmentation-comparison-results"))
    parser.add_argument("--train-script", type=Path, default=Path("train_mk_mnet.py"))
    parser.add_argument("--evaluator", type=Path, default=Path("evaluate_medical_metrics.py"))
    parser.add_argument("--policies", nargs="+", choices=ALL_POLICIES, default=list(COMPARISON_POLICIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--width-mult", type=float, default=0.25)
    parser.add_argument("--lambda", dest="lambda_weight", type=float, default=0.8)
    parser.add_argument("--aux-weight", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
