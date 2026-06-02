import argparse
import shutil
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


MEDMNIST_LABELS = {
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


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)


def save_medmnist_split(images, labels, label_names, output_dir):
    labels = labels.reshape(-1)

    for index, (image, label) in enumerate(zip(images, labels)):
        label = int(label)
        image_dir = output_dir / class_folder(label, label_names[label]) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(image_dir / f"image_{index}.png")

        if index % 1000 == 0:
            print(f"  Saved {index}/{len(images)} images")


def preprocess_medmnist(dataset_name, data_dir, overwrite):
    source_path = data_dir / f"{dataset_name}_224.npz"
    output_dir = data_dir / dataset_name

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    prepare_output_dir(output_dir, overwrite)
    label_names = MEDMNIST_LABELS[dataset_name]

    with np.load(source_path) as data:
        for split in ("train", "val", "test"):
            images = data[f"{split}_images"]
            labels = data[f"{split}_labels"]
            print(f"{split}: images={images.shape}, labels={labels.shape}")
            save_medmnist_split(images, labels, label_names, output_dir / split)

    print(f"Processed dataset saved to: {output_dir}")


def save_busi_split(images, masks, labels, files, class_names, output_dir):
    labels = labels.reshape(-1)
    if not (len(images) == len(masks) == len(labels) == len(files)):
        raise ValueError(f"BUSI split has inconsistent array lengths: {output_dir}")

    for class_name in class_names:
        class_dir = output_dir / str(class_name)
        (class_dir / "images").mkdir(parents=True, exist_ok=True)
        (class_dir / "masks").mkdir(parents=True, exist_ok=True)

    for index, (image, mask, label, source_file) in enumerate(
        zip(images, masks, labels, files)
    ):
        class_name = str(class_names[int(label)])
        image_stem = PurePosixPath(str(source_file)).stem
        class_dir = output_dir / class_name

        Image.fromarray(image).save(class_dir / "images" / f"{image_stem}.png")
        Image.fromarray(mask).save(class_dir / "masks" / f"{image_stem}_mask.png")

        if index % 100 == 0:
            print(f"  Saved {index}/{len(images)} BUSI images")


def preprocess_busi(data_dir, overwrite):
    source_path = data_dir / "busi_224.npz"
    output_dir = data_dir / "busi"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {source_path}. Convert the BUSI ZIP to NPZ first."
        )

    prepare_output_dir(output_dir, overwrite)

    with np.load(source_path) as data:
        class_names = data["class_names"]
        for split in ("train", "val", "test"):
            images = data[f"{split}_images"]
            masks = data[f"{split}_masks"]
            labels = data[f"{split}_labels"]
            files = data[f"{split}_files"]
            print(
                f"{split}: images={images.shape}, masks={masks.shape}, "
                f"labels={labels.shape}"
            )
            save_busi_split(
                images, masks, labels, files, class_names, output_dir / split
            )

    print(f"Processed dataset saved to: {output_dir}")


def preprocess(dataset_name, data_dir, overwrite=False):
    if dataset_name == "busi":
        preprocess_busi(data_dir, overwrite)
    else:
        preprocess_medmnist(dataset_name, data_dir, overwrite)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a dataset for classification.")
    parser.add_argument(
        "--dataset", choices=[*MEDMNIST_LABELS, "busi"], default="bloodmnist"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.dataset, args.data_dir, args.overwrite)
