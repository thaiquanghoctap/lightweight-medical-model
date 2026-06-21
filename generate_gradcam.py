import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model import MKMNet, MedNet, MedNetMultiTask, RCBAMMNet


BUSI_CLASSES = ("benign", "malignant", "normal")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor)
        classification_logits, segmentation_logits = unpack_outputs(output)

        probabilities = torch.softmax(classification_logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        if class_index is None:
            class_index = prediction.item()

        score = classification_logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam = normalize_cam(cam)

        return {
            "cam": cam,
            "classification_logits": classification_logits.detach(),
            "segmentation_logits": None
            if segmentation_logits is None
            else segmentation_logits.detach(),
            "prediction": prediction.item(),
            "confidence": confidence.item(),
            "class_index": class_index,
        }


def unpack_outputs(output):
    if isinstance(output, tuple):
        classification_logits = output[0]
        if len(output) >= 2 and torch.is_tensor(output[1]) and output[1].ndim == 4:
            return classification_logits, output[1]
        return classification_logits, None
    return output, None


def normalize_cam(cam):
    cam = cam - cam.min()
    max_value = cam.max()
    if max_value > 0:
        cam = cam / max_value
    return cam


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
    while num_samples is None or len(selected) < num_samples:
        added = False
        for class_name in BUSI_CLASSES:
            if images_by_class[class_name] and (
                num_samples is None or len(selected) < num_samples
            ):
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
    if not mask_path.exists():
        return np.zeros((image_size, image_size), dtype=bool)
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


def heatmap_to_rgb(cam):
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    heatmap = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((heatmap * 255).astype(np.uint8), mode="RGB")


def overlay_heatmap(image, cam, alpha=0.45):
    heatmap = heatmap_to_rgb(cam)
    return Image.blend(image, heatmap, alpha=alpha)


def save_panel(output_path, panels, image_size):
    header_height = 72
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


def build_model(args):
    if args.model == "mednet":
        return MedNet(num_classes=len(BUSI_CLASSES), use_cbam=True)
    if args.model == "multitask":
        return MedNetMultiTask(num_classes=len(BUSI_CLASSES), use_cbam=True)
    if args.model == "mk_mnet":
        return MKMNet(
            num_classes=len(BUSI_CLASSES),
            deep_supervision=True,
            width_mult=args.width_mult,
        )
    if args.model == "r_cbam_mnet":
        return RCBAMMNet(num_classes=len(BUSI_CLASSES))
    raise ValueError(f"Unsupported model: {args.model}")


def default_output_dir(args):
    parts = [
        args.outputs_root,
        "busi",
        "gradcam",
        args.model,
        f"img_{args.image_size}",
    ]
    if args.model == "mk_mnet":
        parts.append(f"width_{args.width_mult:g}")
    parts.append(f"stage_{args.stage}")
    return Path(*parts)


def load_state_dict(checkpoint, device):
    try:
        return torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location=device)


def generate(args):
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not 1 <= args.stage <= 5:
        raise ValueError("--stage must be in the range 1..5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or default_output_dir(args)

    model = build_model(args)
    model.load_state_dict(load_state_dict(args.checkpoint, device))
    model.to(device)
    model.eval()

    target_layer = model.backbone.stages[args.stage - 1]
    gradcam = GradCAM(model, target_layer)
    num_samples = None if args.all_samples else args.num_samples
    image_paths = collect_test_images(args.dataset_dir, num_samples, args.seed)

    try:
        for index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            resized_image = image.resize(
                (args.image_size, args.image_size),
                Image.Resampling.BILINEAR,
            )
            input_tensor = prepare_image(image, args.image_size).unsqueeze(0).to(device)

            true_class = image_path.parent.parent.name
            true_class_index = BUSI_CLASSES.index(true_class)
            target_class = (
                true_class_index if args.target_class == "true" else None
            )
            result = gradcam(input_tensor, class_index=target_class)

            predicted_class = BUSI_CLASSES[result["prediction"]]
            cam = result["cam"]
            heatmap = heatmap_to_rgb(cam)
            overlay = overlay_heatmap(resized_image, cam, alpha=args.alpha)

            mask_path = (
                image_path.parent.parent
                / "masks"
                / f"{image_path.stem}_mask.png"
            )
            true_mask = prepare_mask(mask_path, args.image_size)

            panels = [
                (
                    resized_image,
                    f"true: {true_class}\n"
                    f"predict: {predicted_class}\n"
                    f"conf: {result['confidence']:.4f}",
                ),
                (mask_to_image(true_mask), "ground-truth\nmask"),
            ]

            segmentation_logits = result["segmentation_logits"]
            if segmentation_logits is not None:
                predicted_mask = (
                    torch.sigmoid(segmentation_logits)[0, 0].cpu().numpy()
                    >= args.threshold
                )
                dice = calculate_dice(predicted_mask, true_mask)
                panels.append(
                    (mask_to_image(predicted_mask), f"predicted mask\nDice: {dice:.4f}")
                )

            target_name = BUSI_CLASSES[result["class_index"]]
            panels.extend(
                [
                    (heatmap, f"Grad-CAM\nstage {args.stage}\nclass: {target_name}"),
                    (overlay, "Grad-CAM\noverlay"),
                ]
            )

            safe_name = "".join(
                char if char.isalnum() else "_" for char in image_path.stem
            ).strip("_")
            output_path = output_dir / f"{index:03d}_{true_class}_{safe_name}.png"
            save_panel(output_path, panels, args.image_size)
    finally:
        gradcam.close()

    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Grad-CAM layer: backbone stage {args.stage}")
    print(f"Saved {len(image_paths)} Grad-CAM panels to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM panels for BUSI classification and multi-task models."
    )
    parser.add_argument(
        "--model",
        choices=("mednet", "multitask", "mk_mnet", "r_cbam_mnet"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=9)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Generate Grad-CAM panels for every image in the BUSI test split.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-class",
        choices=("predicted", "true"),
        default="predicted",
        help="Use the predicted class score or ground-truth class score for Grad-CAM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
