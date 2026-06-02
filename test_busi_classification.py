import argparse
from pathlib import Path

import torch

from model import MedNet
from test_busi_utils import (
    BUSI_CLASSES,
    collect_test_images,
    get_class_name,
    load_checkpoint,
    load_rgb_image,
    prepare_image,
    resize_for_panel,
    safe_stem,
    save_panel,
)


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbam_name = "cbam" if args.cbam else "nocbam"
    checkpoint = args.checkpoint or (
        Path("outputs")
        / "busi"
        / "classification"
        / f"img_{args.image_size}"
        / cbam_name
        / "run_1"
        / "best_model.pt"
    )
    output_dir = args.output_dir or (
        Path("outputs")
        / "busi"
        / "classification"
        / f"img_{args.image_size}"
        / "test_images"
    )
    model = MedNet(num_classes=len(BUSI_CLASSES), use_cbam=args.cbam)
    model = load_checkpoint(model, checkpoint, device)
    image_paths = collect_test_images(args.dataset_dir, args.num_samples, args.seed)

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            image = load_rgb_image(image_path)
            input_tensor = prepare_image(image, args.image_size).unsqueeze(0).to(device)
            logits, _ = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, prediction = probabilities.max(dim=0)

            true_class = get_class_name(image_path)
            predicted_class = BUSI_CLASSES[prediction.item()]
            caption = (
                f"true: {true_class}\n"
                f"predict: {predicted_class}\n"
                f"confidence: {confidence.item():.4f}"
            )
            output_path = (
                output_dir / f"{index:03d}_{true_class}_{safe_stem(image_path)}.png"
            )
            save_panel(
                output_path,
                [(resize_for_panel(image, args.image_size), caption)],
                args.image_size,
            )

    print(f"Saved {len(image_paths)} classification images to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BUSI classification predictions on the test split."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cbam", type=lambda value: value.lower() == "true", default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
