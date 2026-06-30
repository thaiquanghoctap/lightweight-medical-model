"""Profile local model size and CPU latency for BUSI architectures."""

import argparse
import csv
import time
from pathlib import Path

import torch

from model import MKMNet, MedNet, MedNetMultiTask, RCBAMMNet


def build_model(name, width_mult):
    if name == "mednet":
        return MedNet(num_classes=3, use_cbam=True)
    if name == "multitask":
        return MedNetMultiTask(num_classes=3, num_segmentation_classes=1, use_cbam=True)
    if name == "mk_mnet":
        return MKMNet(num_classes=3, deep_supervision=True, width_mult=width_mult)
    if name == "r_cbam_mnet":
        return RCBAMMNet(num_classes=3)
    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model):
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return total, trainable


def estimate_checkpoint_mb(model):
    bytes_total = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    bytes_total += sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return bytes_total / (1024 * 1024)


def measure_latency(model, image_size, warmup, iterations, device):
    model.eval().to(device)
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iterations


def profile_flops(model, image_size, device):
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval().to(device)
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=False,
        ) as profiler:
            with torch.inference_mode():
                model(sample)
        flops = sum(event.flops for event in profiler.key_averages() if event.flops)
        return flops / 1e9 if flops else ""
    except Exception:
        return ""


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Profile local BUSI models.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("mednet", "multitask", "mk_mnet", "r_cbam_mnet"),
        default=("mednet", "mk_mnet", "r_cbam_mnet"),
    )
    parser.add_argument("--width-mults", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("model-profile-results/profile.csv"))
    parser.add_argument("--skip-flops", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows = []
    for name in args.models:
        widths = args.width_mults if name == "mk_mnet" else [1.0]
        for width in widths:
            model = build_model(name, width)
            total, trainable = count_parameters(model)
            latency_ms = measure_latency(model, args.image_size, args.warmup, args.iterations, device)
            gflops = "" if args.skip_flops else profile_flops(model, args.image_size, torch.device("cpu"))
            rows.append(
                {
                    "model": name,
                    "width_mult": width if name == "mk_mnet" else "",
                    "params_m": total / 1e6,
                    "trainable_params_m": trainable / 1e6,
                    "state_size_mb": estimate_checkpoint_mb(model),
                    "cpu_or_device_latency_ms": latency_ms,
                    "profiler_gflops": gflops,
                    "image_size": args.image_size,
                    "device": str(device),
                }
            )
            print(rows[-1])

    write_csv(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
