"""Save synchronized image/mask previews for augmentation-policy review."""

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image, ImageDraw

from augmentation_policies import ALL_POLICIES, COMPARISON_POLICIES, build_augmentation


def overlay_mask(image, mask):
    image = image.astype(np.float32)
    red = np.zeros_like(image)
    red[..., 0] = 255
    alpha = (mask > 0)[..., None].astype(np.float32) * 0.35
    return (image * (1 - alpha) + red * alpha).astype(np.uint8)


def find_sample(dataset_dir, class_name):
    images_dir = dataset_dir / "train" / class_name / "images"
    image_path = next(iter(sorted(images_dir.glob("*.png"))), None)
    if image_path is None:
        raise FileNotFoundError(f"No PNG image found in {images_dir}")
    mask_path = image_path.parent.parent / "masks" / f"{image_path.stem}_mask.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return image_path, mask_path


def save_policy_preview(policy, image, mask, image_size, samples, output_dir):
    transform = A.Compose(
        [A.Resize(image_size, image_size), *build_augmentation(policy, image_size)]
    )
    header = 42
    canvas = Image.new("RGB", (image_size * samples, image_size + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(samples):
        transformed = transform(image=image, mask=mask)
        panel = Image.fromarray(overlay_mask(transformed["image"], transformed["mask"]))
        canvas.paste(panel, (index * image_size, header))
        draw.text((index * image_size + 6, 10), f"{policy} #{index + 1}", fill="black")
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / f"{policy}.png")


def run(args):
    image_path, mask_path = find_sample(args.dataset_dir, args.class_name)
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    with Image.open(mask_path) as source:
        mask = np.asarray(source.convert("L"))
    for policy in args.policies:
        save_policy_preview(
            policy, image, mask, args.image_size, args.samples, args.output_dir
        )
    print(f"Source: {image_path}")
    print(f"Saved {len(args.policies)} previews to: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preview synchronized BUSI augmentations.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--output-dir", type=Path, default=Path("augmentation-comparison-results/previews"))
    parser.add_argument("--class-name", choices=("benign", "malignant"), default="malignant")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--policies", nargs="+", choices=ALL_POLICIES, default=list(COMPARISON_POLICIES))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
