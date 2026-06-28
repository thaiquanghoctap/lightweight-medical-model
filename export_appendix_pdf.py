from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report" / "appendix_evidence.pdf"
IMG = ROOT / "report" / "report_images" / "appendix_evidence"


def add_wrapped_text(ax, text, x, y, width=92, size=10, line_height=0.038, weight=None):
    for line in text.split("\n"):
        if not line:
            y -= line_height
            continue
        for wrapped in wrap(line, width=width):
            ax.text(x, y, wrapped, fontsize=size, va="top", ha="left", fontweight=weight)
            y -= line_height
    return y


def new_page(pdf, title=None):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.9])
    ax.axis("off")
    if title:
        ax.text(0, 1.02, title, fontsize=16, fontweight="bold", va="bottom")
    return fig, ax


def add_table(ax, rows, columns, y, font_size=8):
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.0, y - 0.42, 1.0, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
    return y - 0.46


def add_image(ax, path, box, title=None):
    image = Image.open(path)
    inset = ax.inset_axes(box)
    inset.imshow(image)
    inset.axis("off")
    if title:
        inset.set_title(title, fontsize=9)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        fig, ax = new_page(pdf, "Appendix Evidence for DAMK-Net Experiments")
        y = 0.96
        y = add_wrapped_text(
            ax,
            "This appendix summarizes additional evidence generated from the repository artifacts. "
            "It complements the main DAMK-Net evaluation with detailed medical metrics, "
            "confusion matrices, Grad-CAM visualizations, and an exploratory augmentation comparison.",
            0,
            y,
            size=11,
        )
        y -= 0.04
        y = add_wrapped_text(ax, "Key takeaways", 0, y, size=12, weight="bold")
        y = add_wrapped_text(
            ax,
            "- MK-MNet-S (width 0.5) with deterministic oversampling gives the strongest accuracy-AUC balance.\n"
            "- Prediction refinement substantially improves consistency on normal scans.\n"
            "- Grad-CAM examples show lesion-centered activation in representative benign and malignant cases.\n"
            "- Augmentation results are exploratory because each policy currently has one completed seed.",
            0.02,
            y,
            size=10,
        )
        y -= 0.04
        y = add_wrapped_text(ax, "Detailed medical evaluation", 0, y, size=12, weight="bold")
        rows = [
            ["T, w=0.25", "no", "no", "79.17", "0.899", "76.58", "28.57"],
            ["T, w=0.25", "yes", "no", "77.50", "0.942", "77.75", "61.90"],
            ["S, w=0.5", "no", "no", "79.17", "0.915", "76.77", "33.33"],
            ["S, w=0.5", "yes", "no", "85.00", "0.965", "80.23", "52.38"],
            ["S, w=0.5", "yes", "yes", "85.83", "0.965", "78.48", "85.71"],
            ["Full, w=1.0", "no", "no", "81.67", "0.937", "79.16", "42.86"],
            ["Full, w=1.0", "no", "yes", "81.67", "0.937", "77.52", "90.48"],
            ["Full, w=1.0", "yes", "no", "83.33", "0.955", "80.11", "61.90"],
            ["Full, w=1.0", "yes", "yes", "83.33", "0.955", "78.84", "90.48"],
        ]
        add_table(
            ax,
            rows,
            ["Model", "OS", "PR", "Acc.", "AUC", "Dice lesion", "Dice normal"],
            y,
            font_size=7.5,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_page(pdf, "Confusion Matrix Evidence")
        add_wrapped_text(
            ax,
            "Raw and refined confusion matrices for MK-MNet-S with oversampling. "
            "After refinement, 18 out of 21 normal scans are classified as normal and no malignant scan is reassigned to normal.",
            0,
            0.96,
            size=10,
        )
        add_image(ax, IMG / "confusion_mk_mnet_s_os_raw.png", [0.02, 0.28, 0.46, 0.55], "Raw")
        add_image(ax, IMG / "confusion_mk_mnet_s_os_refined.png", [0.52, 0.28, 0.46, 0.55], "Refined")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_page(pdf, "Grad-CAM Evidence")
        add_wrapped_text(
            ax,
            "Each panel contains the input image, ground-truth mask, predicted mask, Grad-CAM heatmap, and overlay. "
            "The benign and malignant examples show lesion-centered activation; the normal example illustrates a false-positive segmentation response.",
            0,
            0.96,
            size=10,
        )
        add_image(ax, IMG / "gradcam_benign_good.png", [0.00, 0.66, 1.00, 0.22], "Benign")
        add_image(ax, IMG / "gradcam_malignant_good.png", [0.00, 0.39, 1.00, 0.22], "Malignant")
        add_image(ax, IMG / "gradcam_normal_false_positive.png", [0.00, 0.12, 1.00, 0.22], "Normal")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_page(pdf, "Exploratory Augmentation Evidence")
        y = 0.96
        y = add_wrapped_text(
            ax,
            "The augmentation comparison uses MK-MNet-T with deterministic oversampling. "
            "Each completed policy currently has one seed, so the results are exploratory.",
            0,
            y,
            size=10,
        )
        rows = [
            ["None", "66.67", "0.662", "0.853", "63.53", "84.38"],
            ["Rotation", "74.17", "0.718", "0.891", "75.14", "78.13"],
            ["Translate-Y", "70.83", "0.721", "0.923", "76.05", "93.75"],
            ["Scale", "70.00", "0.694", "0.918", "75.47", "93.75"],
            ["Horizontal flip", "74.17", "0.724", "0.897", "66.92", "84.38"],
        ]
        y = add_table(
            ax,
            rows,
            ["Policy", "Acc.", "Macro-F1", "AUC", "Dice lesion", "Malig. sens."],
            y - 0.04,
            font_size=8,
        )
        add_image(ax, IMG / "aug_preview_none.png", [0.00, 0.34, 0.48, 0.15], "None")
        add_image(ax, IMG / "aug_preview_rotate.png", [0.52, 0.34, 0.48, 0.15], "Rotate")
        add_image(ax, IMG / "aug_preview_translate_y.png", [0.00, 0.13, 0.48, 0.15], "Translate-Y")
        add_image(ax, IMG / "aug_preview_scale.png", [0.52, 0.13, 0.48, 0.15], "Scale")
        pdf.savefig(fig)
        plt.close(fig)

    print(OUT)


if __name__ == "__main__":
    main()
