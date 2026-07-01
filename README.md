# Lightweight Medical Model

Research repo for lightweight breast-ultrasound CAD experiments on BUSI, with
single-task baselines, multi-task models, evaluation utilities, and exported
artifacts used in the course report.

## Repository layout

```text
.
├── artifacts/            # exported experiment results kept with the repo
│   ├── augmentation-comparison/
│   ├── gradcam/
│   ├── medical-evaluation/
│   ├── model-profiles/
│   └── threshold-sensitivity/
├── docs/                 # project notes, proposal, report planning
├── notebooks/colab/      # Colab notebooks used to reproduce larger runs
├── report/               # auxiliary report material inside the repo
├── scripts/
│   ├── analysis/         # profiling and summary utilities
│   ├── evaluation/       # detailed metric evaluators
│   ├── experiments/      # orchestration runners
│   ├── reporting/        # report/export helpers
│   ├── training/         # train entrypoints
│   └── visualization/    # Grad-CAM, previews, qualitative panels
├── tests/                # unit tests
├── third_party/          # external templates and reference material
├── augmentation_policies.py
├── model.py
├── preprocess.py
├── pyproject.toml
└── uv.lock
```

## Quick start

```bash
uv sync
```

### Prepare BUSI data

```bash
uv run preprocess.py --dataset busi --overwrite
```

Expected split layout:

```text
data/busi/
├── train/
├── val/
└── test/
```

Each class contains `images/`; BUSI lesion classes also contain `masks/`.

## Training entrypoints

Run the scripts as Python modules from repo root.

### MedNet classification

```bash
uv run python -m scripts.training.train_busi_classification
```

### Segmentation-only baseline

```bash
uv run python -m scripts.training.train_busi_segmentation
```

### Multi-task MedNet baseline

```bash
uv run python -m scripts.training.train_busi_multi
```

### MK-MNet / DAMK-Net style model

```bash
uv run python -m scripts.training.train_mk_mnet
```

### R-CBAM MNet decoder ablation

```bash
uv run python -m scripts.training.train_r_cbam_mnet
```

Default training checkpoints are written under `outputs/`.

## Evaluation and experiment runners

### Detailed medical evaluation for one checkpoint

```bash
uv run python -m scripts.evaluation.evaluate_medical_metrics \
  --model mk_mnet \
  --checkpoint outputs/busi/mk_mnet/.../best_model.pt \
  --dataset-dir data/busi \
  --output-dir artifacts/medical-evaluation/manual_run
```

### Batch evaluation of discovered BUSI checkpoints

```bash
uv run python -m scripts.experiments.run_all_medical_evaluations
```

### Multi-seed augmentation comparison

```bash
uv run python -m scripts.experiments.run_augmentation_comparison \
  --dataset-dir data/busi \
  --policies none rotate translate_y scale horizontal_flip \
  --seeds 42 123 2026 \
  --resume
```

### Threshold sensitivity on a single split

```bash
uv run python -m scripts.experiments.run_threshold_sensitivity \
  --model mk_mnet \
  --checkpoint outputs/busi/mk_mnet/.../best_model.pt \
  --dataset-dir data/busi \
  --split test \
  --output-root artifacts/threshold-sensitivity
```

### Validation-first threshold selection

```bash
uv run python -m scripts.experiments.run_threshold_validation_protocol \
  --model mk_mnet \
  --checkpoint outputs/busi/mk_mnet/.../best_model.pt \
  --dataset-dir data/busi \
  --resume
```

### Model profiling

```bash
uv run python -m scripts.analysis.profile_models \
  --models mednet mk_mnet r_cbam_mnet \
  --width-mults 0.25 0.5 1.0 \
  --output artifacts/model-profiles/profile.csv
```

### Grad-CAM generation

```bash
uv run python -m scripts.visualization.generate_gradcam \
  --model mk_mnet \
  --checkpoint outputs/busi/mk_mnet/.../best_model.pt \
  --dataset-dir data/busi \
  --target-class true
```

## Artifact folders

- `artifacts/augmentation-comparison/`: policy-by-seed runs, preview images,
  and aggregated mean/std tables.
- `artifacts/gradcam/`: exported Grad-CAM panels for MedNet, MK-MNet, and
  ablations.
- `artifacts/medical-evaluation/`: raw/refined per-class metrics, predictions,
  and confusion matrices.

These folders are versioned because they are used directly in the report.

## House style

- Keep reusable entrypoints under `scripts/`, grouped by purpose.
- Keep exported figures, CSVs, and tables under `artifacts/`.
- Keep heavyweight checkpoints under `outputs/` only; they are not versioned.
- Keep one-off notes and planning material under `docs/`.
- Treat `notebooks/colab/` as optional reproduction helpers, not the main API.

## Notes

- `scripts/` now holds operational entrypoints; the repo root is intentionally
  kept small.
- `notebooks/colab/` contains older Colab notebooks. They may still reference
  legacy paths and should be treated as supporting material, not the primary
  interface.
- `third_party/` contains external template material that is not part of the
  core modeling code.
