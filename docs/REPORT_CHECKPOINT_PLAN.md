# Report / Checkpoint Completion Plan

This note tracks the remaining work needed to make the report, checkpoint
references, Grad-CAM outputs, and final artifacts consistent.

Current repo convention:

- runnable entrypoints live under `scripts/`
- exported figures and CSVs live under `artifacts/`
- training checkpoints remain under `outputs/` or external Drive storage

## Checkpoint Status

- MedNet classification has a `best_model.pt` checkpoint:
  `outputs/busi/classification_cu/img_224/cbam/run_1/best_model.pt`
- MedNetSegmentation has a `best_model.pt` checkpoint:
  `outputs/busi/segmentation/img_224/cbam/best_model.pt`
- MedNetMultiTask has a `best_model.pt` checkpoint:
  `outputs/busi/multi/img_224/cbam/seg_weight_1/best_model.pt`
- MK-MNet width `0.25` has `best_model.pt` checkpoints, including:
  `outputs/busi/mk_mnet/img_224/width_0.25/lambda_0.8/lr_0.0003/weight_decay_0.0001/patience_100/best_model.pt`
- MK-MNet width `0.5` has a `best_model.pt` checkpoint:
  `outputs/busi/mk_mnet/img_224/width_0.5/lambda_0.8/lr_0.003/weight_decay_0.001/patience_100/best_model.pt`
- R-CBAM MNet does not save `best_model.pt`. It saves:
  `best_classification.pt`, `best_segmentation.pt`, and `best_joint.pt`.

## Grad-CAM Status

- Grad-CAM implementation exists in `scripts/visualization/generate_gradcam.py`.
- Grad-CAM heatmap images have not yet been generated for the report.
- The script must be run in an environment with PyTorch installed.
- Recommended output location:
  `artifacts/gradcam/...` for archival and `report/report_images/gradcam_*` for
  figures copied into the report bundle.
- Recommended first checkpoints:
  - MK-MNet width `0.25` using `best_model.pt`
  - R-CBAM MNet using `best_joint.pt`

## Remaining Checklist

- [ ] Run Grad-CAM on a representative checkpoint.
- [ ] Save Grad-CAM panels into `report/report_images/gradcam_*`.
- [ ] Update the report `.tex` files to include selected Grad-CAM figures.
- [ ] Add confusion matrix results.
- [ ] Add per-class precision, recall, and F1.
- [ ] Add sensitivity and specificity, especially for `malignant`.
- [ ] Add FLOPs/MACs and inference time if time allows.
- [ ] Recompile the `.tex` files in `report/`.
- [ ] Check that all report image paths still resolve from the `report/` folder.

Detailed-evaluation code and the Colab batch notebook are implemented, but the
three medical-metric items above remain unchecked until real CSV/PNG artifacts
are generated and inserted into the report.

## Augmentation Comparison Status

- [x] Implement paper-aligned single-augmentation and TrivialAugment policies.
- [x] Preserve the original fixed pipeline as the `legacy` policy.
- [x] Add synchronized image-mask previews and resumable Colab batch execution.
- [ ] Run all selected policies for seeds 42, 123, and 2026.
- [ ] Add mean ± std comparison tables to the report.
- [ ] Review geometric transforms visually before using their results.

## Suggested Grad-CAM Commands

Generate Grad-CAM for MK-MNet width `0.25`:

```bash
python -m scripts.visualization.generate_gradcam \
  --model mk_mnet \
  --width-mult 0.25 \
  --checkpoint outputs/busi/mk_mnet/img_224/width_0.25/lambda_0.8/lr_0.0003/weight_decay_0.0001/patience_100/best_model.pt \
  --dataset-dir data/busi \
  --output-dir artifacts/gradcam/mk_mnet_width025/stage_5/target_true \
  --num-samples 9
```

Generate Grad-CAM for R-CBAM MNet using the joint checkpoint:

```bash
python -m scripts.visualization.generate_gradcam \
  --model r_cbam_mnet \
  --checkpoint outputs/busi/r_cbam_mnet/img_224/seg_weight_1.2/lr_0.001/weight_decay_0.001/patience_100/best_joint.pt \
  --dataset-dir data/busi \
  --output-dir artifacts/gradcam/r_cbam_mnet_joint/stage_5/target_true \
  --num-samples 9
```

## Notes

- Do not move checkpoints unless the report paths are updated at the same time.
- Do not treat R-CBAM MNet as having `best_model.pt`.
- Keep archival outputs in `artifacts/`, then copy the final selected figures
  into `report/report_images/` when freezing the report bundle.
