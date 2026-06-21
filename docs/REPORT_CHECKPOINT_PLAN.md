# Report / Checkpoint Completion Plan

This note tracks the remaining work needed to make the report, checkpoint
references, Grad-CAM outputs, and final artifacts consistent.

## Checkpoint Status

- MedNet classification has a `best_model.pt` checkpoint:
  `colab-result/lightweight-medical-model/outputs/busi/classification_cu/img_224/cbam/run_1/best_model.pt`
- MedNetSegmentation has a `best_model.pt` checkpoint:
  `colab-result/lightweight-medical-model/outputs/busi/segmentation/img_224/cbam/best_model.pt`
- MedNetMultiTask has a `best_model.pt` checkpoint:
  `colab-result/lightweight-medical-model/outputs/busi/multi/img_224/cbam/seg_weight_1/best_model.pt`
- MK-MNet width `0.25` has `best_model.pt` checkpoints, including:
  `colab-result/lightweight-medical-model/outputs/busi/mk_mnet/img_224/width_0.25/lambda_0.8/lr_0.0003/weight_decay_0.0001/patience_100/best_model.pt`
- MK-MNet width `0.5` has a `best_model.pt` checkpoint:
  `colab-result/lightweight-medical-model/outputs/busi/mk_mnet/img_224/width_0.5/lambda_0.8/lr_0.003/weight_decay_0.001/patience_100/best_model.pt`
- R-CBAM MNet does not save `best_model.pt`. It saves:
  `best_classification.pt`, `best_segmentation.pt`, and `best_joint.pt`.

## Grad-CAM Status

- Grad-CAM implementation exists in `generate_gradcam.py`.
- Grad-CAM heatmap images have not yet been generated for the report.
- The script must be run in an environment with PyTorch installed.
- Recommended output location:
  `report/report_images/gradcam_*`
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

## Suggested Grad-CAM Commands

Generate Grad-CAM for MK-MNet width `0.25`:

```bash
python generate_gradcam.py \
  --model mk_mnet \
  --width-mult 0.25 \
  --checkpoint colab-result/lightweight-medical-model/outputs/busi/mk_mnet/img_224/width_0.25/lambda_0.8/lr_0.0003/weight_decay_0.0001/patience_100/best_model.pt \
  --dataset-dir colab-result/lightweight-medical-model/data/busi \
  --output-dir report/report_images/gradcam_mk_mnet_width025 \
  --num-samples 9
```

Generate Grad-CAM for R-CBAM MNet using the joint checkpoint:

```bash
python generate_gradcam.py \
  --model r_cbam_mnet \
  --checkpoint colab-result/lightweight-medical-model/outputs/busi/r_cbam_mnet/img_224/seg_weight_1.2/lr_0.001/weight_decay_0.001/patience_100/best_joint.pt \
  --dataset-dir colab-result/lightweight-medical-model/data/busi \
  --output-dir report/report_images/gradcam_r_cbam_mnet_joint \
  --num-samples 9
```

## Notes

- Do not move checkpoints unless the report paths are updated at the same time.
- Do not treat R-CBAM MNet as having `best_model.pt`.
- Keep generated Grad-CAM images inside `report/report_images/` if they are meant
  to be included in the report bundle.
