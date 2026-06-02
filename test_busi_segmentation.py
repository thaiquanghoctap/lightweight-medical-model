import argparse
from pathlib import Path

import torch

from model import MedNetSegmentation
from test_busi_utils import (
    calculate_dice,
    collect_test_images,
    get_class_name,
    get_mask_path,
    load_checkpoint,
    load_rgb_image,
    mask_to_image,
    prepare_image,
    prepare_mask,
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
        / "segmentation"
        / f"img_{args.image_size}"
        / cbam_name
        / "best_model.pt"
    )
    output_dir = args.output_dir or (
        Path("outputs")
        / "busi"
        / "segmentation"
        / f"img_{args.image_size}"
        / "test_images"
    )
    model = MedNetSegmentation(num_classes=1, use_cbam=args.cbam)
    model = load_checkpoint(model, checkpoint, device)
    image_paths = collect_test_images(args.dataset_dir, args.num_samples, args.seed)

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            image = load_rgb_image(image_path)
            input_tensor = prepare_image(image, args.image_size).unsqueeze(0).to(device)
            logits = model(input_tensor)
            predicted_mask = (
                torch.sigmoid(logits)[0, 0].cpu().numpy() >= args.threshold
            )
            true_mask = prepare_mask(get_mask_path(image_path), args.image_size)
            dice = calculate_dice(predicted_mask, true_mask)

            true_class = get_class_name(image_path)
            output_path = (
                output_dir / f"{index:03d}_{true_class}_{safe_stem(image_path)}.png"
            )
            save_panel(
                output_path,
                [
                    (resize_for_panel(image, args.image_size), "original image"),
                    (mask_to_image(true_mask), "true segmentation"),
                    (mask_to_image(predicted_mask), f"predicted segmentation\nDICE: {dice:.4f}"),
                ],
                args.image_size,
            )

    print(f"Saved {len(image_paths)} segmentation panels to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BUSI segmentation predictions on the test split."
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--cbam", type=lambda value: value.lower() == "true", default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
