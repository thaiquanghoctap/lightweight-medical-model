# Lightweight MedNet

This project contains a cleaned-up version of the original MedNet source code.

## Files

```text
preprocess.py   Convert a MedMNIST .npz file to ImageFolder format
model.py        MedNet architecture and focal loss
train.py        Train MedNet with configurable ablation settings
data/           Original MedMNIST .npz files
outputs/        Training outputs
```

Supported datasets:

```text
bloodmnist
breastmnist
dermamnist
octmnist
```

## Preprocess A Dataset

```bash
uv sync
uv run preprocess.py --dataset bloodmnist
```

The processed images are written to:

```text
data/<dataset>/train/
data/<dataset>/val/
data/<dataset>/test/
```

Use `--overwrite` to replace an existing processed dataset.

## Train A Dataset

```bash
uv run train.py --dataset bloodmnist
```

The default command trains one model:

```text
image size: 224x224
CBAM: enabled
runs: 1
```

To run the original ablation study:

```bash
uv run train.py \
  --dataset bloodmnist \
  --img-sizes 384 224 28 \
  --cbam both \
  --runs 3
```

This starts `3 image sizes x 2 CBAM modes x 3 runs = 18` training sessions.

Outputs are written to:

```text
outputs/<dataset>/
├── results.csv
└── img_<size>/
    ├── cbam/
    │   └── run_<number>/
    │       ├── best_model.pt
    │       ├── epoch_log.csv
    │       └── result.csv
    └── nocbam/
        └── run_<number>/
```
