"""Discover and evaluate every completed BUSI classification-capable run."""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


TASK_MODELS = {
    "classification": "mednet",
    "classification_cu": "mednet",
    "multi": "multitask",
    "mk_mnet": "mk_mnet",
    "mk_mnet_cu": "mk_mnet",
    "r_cbam_mnet": "r_cbam_mnet",
}


def safe_config_name(path):
    return "__".join(path.parts).replace(".", "_")


def infer_width(path):
    match = re.search(r"(?:^|/)width_([0-9.]+)(?:/|$)", path.as_posix())
    return match.group(1) if match else "1.0"


def discover_runs(outputs_root):
    busi_root = outputs_root / "busi"
    runs = []
    for task_name, model_name in TASK_MODELS.items():
        task_root = busi_root / task_name
        if not task_root.is_dir():
            continue
        candidate_dirs = {
            path.parent for path in task_root.rglob("best_model.pt")
        } | {
            path.parent for path in task_root.rglob("best_joint.pt")
        }
        for run_dir in sorted(candidate_dirs):
            checkpoint = run_dir / "best_model.pt"
            if not checkpoint.is_file():
                checkpoint = run_dir / "best_joint.pt"
            relative = run_dir.relative_to(busi_root)
            runs.append(
                {
                    "config": safe_config_name(relative),
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "width_mult": infer_width(relative),
                    "cbam": "false" if "nocbam" in relative.parts else "true",
                }
            )
    return runs


def combine_csv(output_root, statuses, filename):
    combined = []
    fieldnames = ["config"]
    for status in statuses:
        if status["status"] != "completed":
            continue
        path = Path(status["output_dir"]) / filename
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            combined.append({"config": status["config"], **row})
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not combined:
        return
    with (output_root / f"all_{filename}").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)


def write_status(path, statuses):
    fields = ["config", "model", "checkpoint", "output_dir", "status", "return_code", "error"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(statuses)


def tex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def metric(value):
    if value in (None, ""):
        return "--"
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return tex_escape(value)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def generate_latex(output_root):
    summary_path = output_root / "all_summary.csv"
    per_class_path = output_root / "all_per_class_metrics.csv"
    if not summary_path.is_file() or not per_class_path.is_file():
        return
    summary = read_csv(summary_path)
    per_class = read_csv(per_class_path)
    raw_summary = [row for row in summary if row["variant"] == "raw"]
    refined_summary = [row for row in summary if row["variant"] == "refined"]
    raw_per_class = [row for row in per_class if row["variant"] == "raw"]
    malignant = [row for row in raw_per_class if row["class"] == "malignant"]

    lines = [
        r"\subsubsection*{Tổng quan detailed evaluation (raw)}",
        r"\scriptsize",
        r"\begin{longtable}{p{5.0cm}rrrrr}",
        r"\toprule",
        r"Cấu hình & Accuracy & Macro-F1 & AUC & Dice lesion & IoU lesion \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Cấu hình & Accuracy & Macro-F1 & AUC & Dice lesion & IoU lesion \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in raw_summary:
        lines.append(
            f"{tex_escape(row['config'])} & {metric(row['accuracy'])} & "
            f"{metric(row['macro_f1'])} & {metric(row['macro_auc_ovr'])} & "
            f"{metric(row['dice_lesion'])} & {metric(row['iou_lesion'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])

    lines.extend(
        [
            r"\subsubsection*{Medical metrics theo lớp (raw)}",
            r"\scriptsize",
            r"\begin{longtable}{p{4.2cm}lrrrrrr}",
            r"\toprule",
            r"Cấu hình & Lớp & Precision & Sensitivity & Specificity & F1 & AUC & N \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Cấu hình & Lớp & Precision & Sensitivity & Specificity & F1 & AUC & N \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for row in raw_per_class:
        lines.append(
            f"{tex_escape(row['config'])} & {tex_escape(row['class'])} & "
            f"{metric(row['precision'])} & {metric(row['sensitivity'])} & "
            f"{metric(row['specificity'])} & {metric(row['f1'])} & "
            f"{metric(row['auc'])} & {int(float(row['support']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])

    lines.extend(
        [
            r"\subsubsection*{Đánh giá riêng lớp malignant (raw)}",
            r"\scriptsize",
            r"\begin{longtable}{p{5.2cm}rrrrr}",
            r"\toprule",
            r"Cấu hình & Precision & Sensitivity & Specificity & F1 & AUC \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Cấu hình & Precision & Sensitivity & Specificity & F1 & AUC \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for row in malignant:
        lines.append(
            f"{tex_escape(row['config'])} & {metric(row['precision'])} & "
            f"{metric(row['sensitivity'])} & {metric(row['specificity'])} & "
            f"{metric(row['f1'])} & {metric(row['auc'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])

    if refined_summary:
        lines.extend(
            [
                r"\subsubsection*{Prediction-refining ablation}",
                "Bảng này là kết quả hậu xử lý, không thay thế kết quả raw.",
                r"\scriptsize",
                r"\begin{longtable}{p{5.0cm}rrrrr}",
                r"\toprule",
                r"Cấu hình & Accuracy & Macro-F1 & AUC & Dice lesion & IoU lesion \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"Cấu hình & Accuracy & Macro-F1 & AUC & Dice lesion & IoU lesion \\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for row in refined_summary:
            lines.append(
                f"{tex_escape(row['config'])} & {metric(row['accuracy'])} & "
                f"{metric(row['macro_f1'])} & {metric(row['macro_auc_ovr'])} & "
                f"{metric(row['dice_lesion'])} & {metric(row['iou_lesion'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    (output_root / "medical_metrics_tables.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    figures = []
    for row in raw_summary:
        config = row["config"]
        png = output_root / config / "confusion_matrix_raw.png"
        if not png.is_file():
            continue
        relative = Path("..") / output_root.name / config / png.name
        figures.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                f"\\includegraphics[width=0.62\\textwidth]{{{relative.as_posix()}}}",
                f"\\caption{{Confusion matrix raw: {tex_escape(config)}.}}",
                r"\end{figure}",
            ]
        )
    (output_root / "confusion_matrices_appendix.tex").write_text(
        "\n".join(figures) + "\n", encoding="utf-8"
    )


def run(args):
    runs = discover_runs(args.outputs_root)
    if not runs:
        raise FileNotFoundError(f"No supported checkpoints found under {args.outputs_root / 'busi'}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    statuses = []
    for item in runs:
        output_dir = args.output_root / item["config"]
        command = [
            sys.executable,
            str(args.evaluator),
            "--model", item["model"],
            "--checkpoint", str(item["checkpoint"]),
            "--dataset-dir", str(args.dataset_dir),
            "--output-dir", str(output_dir),
            "--image-size", str(args.image_size),
            "--width-mult", item["width_mult"],
            "--cbam", item["cbam"],
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--threshold", str(args.threshold),
            "--area-threshold", str(args.area_threshold),
        ]
        print(f"\nRUN {item['config']}\n  {item['checkpoint']}")
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        statuses.append(
            {
                "config": item["config"],
                "model": item["model"],
                "checkpoint": str(item["checkpoint"]),
                "output_dir": str(output_dir),
                "status": "completed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "error": "" if result.returncode == 0 else result.stderr.strip()[-2000:],
            }
        )
        write_status(args.output_root / "evaluation_status.csv", statuses)

    combine_csv(args.output_root, statuses, "summary.csv")
    combine_csv(args.output_root, statuses, "per_class_metrics.csv")
    generate_latex(args.output_root)
    completed = [row["config"] for row in statuses if row["status"] == "completed"]
    failed = [row["config"] for row in statuses if row["status"] == "failed"]
    print(f"\nCompleted ({len(completed)}): {completed}")
    print(f"Failed ({len(failed)}): {failed}")
    print(f"Status: {args.output_root / 'evaluation_status.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run detailed evaluation for all BUSI checkpoints.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--output-root", type=Path, default=Path("medical-evaluation-results"))
    parser.add_argument("--evaluator", type=Path, default=Path("evaluate_medical_metrics.py"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--area-threshold", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
