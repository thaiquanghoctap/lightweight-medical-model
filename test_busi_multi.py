import argparse
from pathlib import Path

import torch

from model import MedNetMultiTask
from test_busi_utils import (
    BUSI_CLASSES,
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
        / "multi"
        / f"img_{args.image_size}"
        / cbam_name
        / f"seg_weight_{args.segmentation_weight:g}"
        / "best_model.pt"
    )
    output_dir = args.output_dir or (
        Path("outputs")
        / "busi"
        / "multi"
        / f"img_{args.image_size}"
        / "test_images"
    )
    model = MedNetMultiTask(
        num_classes=len(BUSI_CLASSES),
        num_segmentation_classes=1,
        use_cbam=args.cbam,
    )
    model = load_checkpoint(model, checkpoint, device)
    image_paths = collect_test_images(args.dataset_dir, args.num_samples, args.seed)

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            image = load_rgb_image(image_path)
            input_tensor = prepare_image(image, args.image_size).unsqueeze(0).to(device)
            classification_logits, segmentation_logits = model(input_tensor)

            probabilities = torch.softmax(classification_logits, dim=1)[0]
            confidence, prediction = probabilities.max(dim=0)
            predicted_mask = (
                torch.sigmoid(segmentation_logits)[0, 0].cpu().numpy() >= args.threshold
            )
            true_mask = prepare_mask(get_mask_path(image_path), args.image_size)
            dice = calculate_dice(predicted_mask, true_mask)

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
                [
                    (resize_for_panel(image, args.image_size), caption),
                    (mask_to_image(true_mask), "true segmentation"),
                    (mask_to_image(predicted_mask), f"predicted segmentation\nDICE: {dice:.4f}"),
                ],
                args.image_size,
            )

    print(f"Saved {len(image_paths)} multi-task panels to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BUSI multi-task predictions on the test split."
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
    parser.add_argument("--segmentation-weight", type=float, default=1.0)
    parser.add_argument(
        "--cbam", type=lambda value: value.lower() == "true", default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
