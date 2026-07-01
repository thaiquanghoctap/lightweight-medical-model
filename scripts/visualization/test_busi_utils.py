import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


BUSI_CLASSES = ("benign", "malignant", "normal")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def collect_test_images(dataset_dir, num_samples, seed):
    split_dir = Path(dataset_dir) / "test"
    rng = random.Random(seed)
    images_by_class = {}

    for class_name in BUSI_CLASSES:
        images_dir = split_dir / class_name / "images"
        image_paths = sorted(
            path
            for path in images_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No test images found in {images_dir}")
        rng.shuffle(image_paths)
        images_by_class[class_name] = image_paths

    selected = []
    while len(selected) < num_samples:
        added_image = False
        for class_name in BUSI_CLASSES:
            if images_by_class[class_name] and len(selected) < num_samples:
                selected.append(images_by_class[class_name].pop())
                added_image = True
        if not added_image:
            break

    return selected


def get_class_name(image_path):
    return Path(image_path).parent.parent.name


def get_mask_path(image_path):
    image_path = Path(image_path)
    return image_path.parent.parent / "masks" / f"{image_path.stem}_mask.png"


def load_rgb_image(image_path):
    return Image.open(image_path).convert("RGB")


def prepare_image(image, image_size):
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def prepare_mask(mask_path, image_size):
    mask = Image.open(mask_path).convert("L")
    resized = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def calculate_dice(prediction, target, smooth=1e-6):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    return (2.0 * intersection + smooth) / (
        prediction.sum() + target.sum() + smooth
    )


def mask_to_image(mask):
    mask_array = mask.astype(np.uint8) * 255
    return Image.fromarray(mask_array, mode="L").convert("RGB")


def resize_for_panel(image, image_size):
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR)


def save_panel(output_path, panels, image_size):
    header_height = 58
    canvas = Image.new(
        "RGB",
        (image_size * len(panels), image_size + header_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, (image, caption) in enumerate(panels):
        x = index * image_size
        canvas.paste(image, (x, header_height))
        draw.multiline_text((x + 6, 8), caption, fill="black", spacing=3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def safe_stem(image_path):
    return "".join(
        char if char.isalnum() else "_" for char in Path(image_path).stem
    ).strip("_")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
