# Lightweight MedNet

This project contains a cleaned-up version of the original MedNet source code.

## Files

```text
preprocess.py   Prepare MedMNIST or BUSI data for classification
model.py        MedNet architecture and focal loss
train.py        Train MedNet with configurable ablation settings
train_busi.py   Train MedNet on BUSI classification images only
data/           Original MedMNIST .npz files
outputs/        Training outputs
```

Supported datasets:

```text
bloodmnist
breastmnist
dermamnist
octmnist
busi
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

Each class contains an `images/` folder. BUSI classes also contain a `masks/`
folder:

```text
data/busi/train/
├── benign/
│   ├── images/
│   └── masks/
├── malignant/
│   ├── images/
│   └── masks/
└── normal/
    ├── images/
    └── masks/
```

To read `data/busi_224.npz` and write the BUSI image folders:

```bash
uv run preprocess.py --dataset busi --overwrite
```

Use `--overwrite` to replace an existing processed dataset.

## Train A Dataset

```bash
uv run train.py --dataset bloodmnist
```

To train BUSI using classification images while ignoring masks:

```bash
uv run train_busi.py
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
