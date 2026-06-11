import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import RCBAMMNet


BUSI_CLASSES = ("benign", "malignant", "normal")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def collect_test_images(dataset_dir, num_samples, seed):
    rng = random.Random(seed)
    images_by_class = {}

    for class_name in BUSI_CLASSES:
        images_dir = dataset_dir / "test" / class_name / "images"
        image_paths = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No test images found in {images_dir}")
        rng.shuffle(image_paths)
        images_by_class[class_name] = image_paths

    selected = []
    while len(selected) < num_samples:
        added = False
        for class_name in BUSI_CLASSES:
            if images_by_class[class_name] and len(selected) < num_samples:
                selected.append(images_by_class[class_name].pop())
                added = True
        if not added:
            break

    return selected


def prepare_image(image, image_size):
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def prepare_mask(mask_path, image_size):
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


def calculate_dice(prediction, target, smooth=1e-6):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    return (2 * intersection + smooth) / (
        prediction.sum() + target.sum() + smooth
    )


def mask_to_image(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def save_panel(output_path, panels, image_size):
    header_height = 62
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or (
        Path("outputs")
        / "busi"
        / "r_cbam_mnet"
        / f"img_{args.image_size}"
        / f"seg_weight_{args.segmentation_weight:g}"
        / "best_joint.pt"
    )
    output_dir = args.output_dir or (
        Path("outputs")
        / "busi"
        / "r_cbam_mnet"
        / f"img_{args.image_size}"
        / "test_images"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = RCBAMMNet(num_classes=len(BUSI_CLASSES))
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_paths = collect_test_images(
        args.dataset_dir, args.num_samples, args.seed
    )

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            input_tensor = prepare_image(image, args.image_size).unsqueeze(0).to(device)
            classification_logits, segmentation_logits = model(input_tensor)

            probabilities = torch.softmax(classification_logits, dim=1)[0]
            confidence, prediction = probabilities.max(dim=0)
            predicted_mask = (
                torch.sigmoid(segmentation_logits)[0, 0].cpu().numpy()
                >= args.threshold
            )

            class_name = image_path.parent.parent.name
            mask_path = (
                image_path.parent.parent
                / "masks"
                / f"{image_path.stem}_mask.png"
            )
            true_mask = prepare_mask(mask_path, args.image_size)
            dice = calculate_dice(predicted_mask, true_mask)
            predicted_class = BUSI_CLASSES[prediction.item()]

            caption = (
                f"true: {class_name}\n"
                f"predict: {predicted_class}\n"
                f"confidence: {confidence.item():.4f}"
            )
            safe_name = "".join(
                char if char.isalnum() else "_" for char in image_path.stem
            ).strip("_")
            output_path = output_dir / f"{index:03d}_{class_name}_{safe_name}.png"

            save_panel(
                output_path,
                [
                    (
                        image.resize(
                            (args.image_size, args.image_size),
                            Image.Resampling.BILINEAR,
                        ),
                        caption,
                    ),
                    (mask_to_image(true_mask), "ground-truth mask"),
                    (
                        mask_to_image(predicted_mask),
                        f"predicted mask\nDice: {dice:.4f}",
                    ),
                ],
                args.image_size,
            )

    print(f"Checkpoint: {checkpoint}")
    print(f"Saved {len(image_paths)} prediction panels to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize R-CBAM MNet predictions on BUSI."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--segmentation-weight", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
