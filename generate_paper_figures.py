"""Generate paper figures 3 (qualitative), 4 (threshold robustness), 5 (Grad-CAM)
for the headline DAMK-Net-full (width 1.0, oversampled) checkpoint on BUSI test.

One forward pass over the test split is cached, then reused for all figures.
All numbers come from the real checkpoint; no values are hand-edited.

Outputs (PNG + PDF) go to Paper/figures/ and a pgfplots coordinate dump for
Figure 4 is printed so it can be embedded self-contained in main.tex.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from model import MKMNet
from train_mk_mnet import BUSI_CLASSES, build_transforms
from generate_gradcam import GradCAM, prepare_image, IMAGE_EXTENSIONS

NORMAL_INDEX = BUSI_CLASSES.index("normal")
LESION_INDICES = [i for i, n in enumerate(BUSI_CLASSES) if n != "normal"]
SMOOTH = 1e-6


def dice_np(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return (2 * inter + SMOOTH) / (pred.sum() + target.sum() + SMOOTH)


def load_samples(dataset_dir):
    samples = []
    for label, name in enumerate(BUSI_CLASSES):
        images_dir = dataset_dir / "test" / name / "images"
        masks_dir = dataset_dir / "test" / name / "masks"
        for path in sorted(p for p in images_dir.iterdir()
                           if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()):
            mask_path = masks_dir / f"{path.stem}_mask.png"
            samples.append((path, mask_path, label))
    return samples


def main(args):
    device = torch.device("cpu")
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    model = MKMNet(num_classes=len(BUSI_CLASSES), deep_supervision=True, width_mult=1.0)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    _, eval_tf = build_transforms(args.image_size)
    samples = load_samples(args.dataset_dir)

    cache = []  # per-sample dict
    with torch.inference_mode():
        for img_path, mask_path, label in samples:
            disp = np.asarray(Image.open(img_path).convert("RGB")
                              .resize((args.image_size, args.image_size), Image.BILINEAR))
            gt = np.asarray(Image.open(mask_path).convert("L")
                            .resize((args.image_size, args.image_size), Image.NEAREST)) > 127
            tens = eval_tf(image=np.asarray(Image.open(img_path).convert("RGB")).copy(),
                           mask=np.asarray(Image.open(mask_path).convert("L")).copy())["image"]
            x = tens.unsqueeze(0).to(device)
            cls_logits, seg_logits = model(x)[0], model(x)[1]
            prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
            pred_mask = prob >= args.threshold
            cls_pred = int(cls_logits.argmax(1).item())
            conf = float(torch.softmax(cls_logits, 1)[0, cls_pred])
            cache.append(dict(img_path=img_path, label=label, disp=disp, gt=gt,
                              prob=prob, pred_mask=pred_mask, cls_pred=cls_pred,
                              conf=conf, x=x))

    labels = np.array([c["label"] for c in cache])
    cls_pred = np.array([c["cls_pred"] for c in cache])
    areas = np.array([c["pred_mask"].mean() for c in cache])

    # ---------- Figure 4: threshold (tau) robustness ----------
    taus = [0.0, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    acc_list, ndice_list, nfp_list = [], [], []
    for tau in taus:
        refined = cls_pred.copy()
        refined[areas < tau] = NORMAL_INDEX
        acc = float(np.mean(refined == labels))
        # normal Dice after refining (mask emptied when refined-class == normal)
        ndices = []
        for c, rc in zip(cache, refined):
            if c["label"] != NORMAL_INDEX:
                continue
            pm = np.zeros_like(c["pred_mask"]) if rc == NORMAL_INDEX else c["pred_mask"]
            ndices.append(dice_np(pm, c["gt"]))
        ndice = float(np.mean(ndices))
        nmask = labels == NORMAL_INDEX
        nfp = float(np.mean(np.isin(refined[nmask], LESION_INDICES)))
        acc_list.append(acc); ndice_list.append(ndice); nfp_list.append(nfp)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = [max(t, 1e-4) for t in taus]  # 0 plotted at left edge on log axis
    ax.set_xscale("log")
    ax.plot(xs, [a * 100 for a in acc_list], "o-", color="#0072B2", label="Accuracy")
    ax.plot(xs, [d * 100 for d in ndice_list], "s-", color="#E69F00", label="Normal Dice")
    ax.plot(xs, [f * 100 for f in nfp_list], "^-", color="#D55E00", label="Normal FP rate")
    ax.axvspan(5e-4, 5e-3, color="gray", alpha=0.15, label=r"stable band $[5{\times}10^{-4},5{\times}10^{-3}]$")
    ax.set_xlabel(r"Area threshold $\tau$ (log scale)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_threshold.png", dpi=200)
    fig.savefig(fig_dir / "fig4_threshold.pdf")
    plt.close(fig)

    print("\n=== Figure 4 tau-sweep (real data) ===")
    print(f"{'tau':>8}{'acc%':>8}{'nDice%':>8}{'nFP%':>8}")
    for t, a, d, f in zip(taus, acc_list, ndice_list, nfp_list):
        print(f"{t:>8.4f}{a*100:>8.2f}{d*100:>8.2f}{f*100:>8.2f}")
    print("\npgfplots Accuracy coords:")
    print("  " + " ".join(f"({max(t,1e-4):g},{a*100:.2f})" for t, a in zip(taus, acc_list)))
    print("pgfplots NormalDice coords:")
    print("  " + " ".join(f"({max(t,1e-4):g},{d*100:.2f})" for t, d in zip(taus, ndice_list)))
    print("pgfplots NormalFP coords:")
    print("  " + " ".join(f"({max(t,1e-4):g},{f*100:.2f})" for t, f in zip(taus, nfp_list)))

    # ---------- pick representative (correctly classified) per class ----------
    reps = {}
    for cls in range(len(BUSI_CLASSES)):
        cands = [c for c in cache if c["label"] == cls and c["cls_pred"] == cls]
        if cls == NORMAL_INDEX:
            cands.sort(key=lambda c: c["pred_mask"].mean())  # smallest spurious area
        else:
            cands.sort(key=lambda c: -dice_np(c["pred_mask"], c["gt"]))  # best dice
        reps[cls] = cands[0]
        d = dice_np(reps[cls]["pred_mask"], reps[cls]["gt"])
        print(f"rep {BUSI_CLASSES[cls]}: {reps[cls]['img_path'].name} dice={d:.3f} conf={reps[cls]['conf']:.3f}")

    # ---------- Figure 3: qualitative segmentation ----------
    cols = ["Input", "Ground truth", "Prediction"]
    fig, axes = plt.subplots(3, 3, figsize=(6.0, 6.2))
    for r, cls in enumerate(range(len(BUSI_CLASSES))):
        c = reps[cls]
        axes[r, 0].imshow(c["disp"])
        gt_overlay = c["disp"].copy()
        axes[r, 1].imshow(c["disp"])
        axes[r, 1].imshow(np.ma.masked_where(~c["gt"], c["gt"]), cmap="autumn", alpha=0.5)
        axes[r, 2].imshow(c["disp"])
        if c["pred_mask"].any():
            axes[r, 2].imshow(np.ma.masked_where(~c["pred_mask"], c["pred_mask"]), cmap="winter", alpha=0.5)
        for k in range(3):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
        axes[r, 0].set_ylabel(BUSI_CLASSES[cls].capitalize(), fontsize=11)
    for k in range(3):
        axes[0, k].set_title(cols[k], fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_qualitative.png", dpi=200)
    fig.savefig(fig_dir / "fig3_qualitative.pdf")
    plt.close(fig)

    # ---------- Figure 5: Grad-CAM ----------
    target_layer = model.backbone.stages[4]  # stage 5
    gradcam = GradCAM(model, target_layer)
    cols = ["Input", "Grad-CAM", "Overlay"]
    fig, axes = plt.subplots(3, 3, figsize=(6.0, 6.2))
    for r, cls in enumerate(range(len(BUSI_CLASSES))):
        c = reps[cls]
        with Image.open(c["img_path"]) as src:
            cam_input = prepare_image(src.convert("RGB"), args.image_size).unsqueeze(0).to(device)
        res = gradcam(cam_input, class_index=cls)
        cam = res["cam"]
        axes[r, 0].imshow(c["disp"])
        axes[r, 1].imshow(cam, cmap="jet")
        axes[r, 2].imshow(c["disp"])
        axes[r, 2].imshow(cam, cmap="jet", alpha=0.45)
        for k in range(3):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
        axes[r, 0].set_ylabel(BUSI_CLASSES[cls].capitalize(), fontsize=11)
    for k in range(3):
        axes[0, k].set_title(cols[k], fontsize=11)
    gradcam.close()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_gradcam.png", dpi=200)
    fig.savefig(fig_dir / "fig5_gradcam.pdf")
    plt.close(fig)

    print(f"\nSaved figures to {fig_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/busi/mk_mnet/img_224/width_1/oversample_1/lambda_0.8/lr_0.0003/weight_decay_0.0001/patience_10/best_model.pt"))
    p.add_argument("--dataset-dir", type=Path, default=Path("data/busi"))
    p.add_argument("--fig-dir", type=Path, default=Path("../../Paper/figures"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--threshold", type=float, default=0.5)
    main(p.parse_args())
