import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


DATASET_LABELS = {
    "bloodmnist": {
        0: "basophil",
        1: "eosinophil",
        2: "erythroblast",
        3: "immature granulocytes",
        4: "lymphocyte",
        5: "monocyte",
        6: "neutrophil",
        7: "platelet",
    },
    "breastmnist": {
        0: "malignant",
        1: "normal benign",
    },
    "dermamnist": {
        0: "actinic keratoses and intraepithelial carcinoma",
        1: "basal cell carcinoma",
        2: "benign keratosis-like lesions",
        3: "dermatofibroma",
        4: "melanoma",
        5: "melanocytic nevi",
        6: "vascular lesions",
    },
    "octmnist": {
        0: "choroidal neovascularization",
        1: "diabetic macular edema",
        2: "drusen",
        3: "normal",
    },
}


def class_folder(label_id, label_name):
    safe_name = label_name.replace(" ", "_").replace(",", "")
    return f"{label_id:02d}_{safe_name}"


def save_split(images, labels, label_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.reshape(-1)

    for index, (image, label) in enumerate(zip(images, labels)):
        label = int(label)
        folder = output_dir / class_folder(label, label_names[label])
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(folder / f"image_{index}.png")

        if index % 1000 == 0:
            print(f"  Saved {index}/{len(images)} images")


def preprocess(dataset_name, data_dir, overwrite=False):
    source_path = data_dir / f"{dataset_name}_224.npz"
    output_dir = data_dir / dataset_name

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    data = np.load(source_path)
    label_names = DATASET_LABELS[dataset_name]

    for split in ("train", "val", "test"):
        images = data[f"{split}_images"]
        labels = data[f"{split}_labels"]
        print(f"{split}: images={images.shape}, labels={labels.shape}")
        save_split(images, labels, label_names, output_dir / split)

    print(f"Processed dataset saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a MedMNIST .npz file to ImageFolder format.")
    parser.add_argument("--dataset", choices=DATASET_LABELS, default="bloodmnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.dataset, args.data_dir, args.overwrite)
