# Lightweight MedNet

This project contains a cleaned-up version of the original MedNet source code.

## Files

```text
preprocess.py   Prepare MedMNIST or BUSI data for classification
model.py        MedNet architecture and focal loss
train.py        Train MedNet with configurable ablation settings
train_busi_classification.py
                Train MedNet on BUSI classification images only
train_busi_segmentation.py
                Train MedNet segmentation on BUSI image-mask pairs
train_busi_multi.py
                Train shared-backbone classification and segmentation on BUSI
test_busi_classification.py
                Create annotated BUSI classification test images
test_busi_segmentation.py
                Create BUSI segmentation comparison panels
test_busi_multi.py
                Create BUSI multi-task comparison panels
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
uv run train_busi_classification.py
```

To train BUSI segmentation using image-mask pairs:

```bash
uv run train_busi_segmentation.py
```

To train BUSI classification and segmentation together:

```bash
uv run train_busi_multi.py
```

## Visualize BUSI Test Predictions

After training, create annotated images from the BUSI test split:

```bash
uv run test_busi_classification.py
uv run test_busi_segmentation.py
uv run test_busi_multi.py
```

The generated PNG files are written to:

```text
outputs/busi/classification/img_224/test_images/
outputs/busi/segmentation/img_224/test_images/
outputs/busi/multi/img_224/test_images/
```

Use `--num-samples`, `--checkpoint`, or `--output-dir` to change the test
settings.

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

BUSI outputs are grouped by task:

```text
outputs/busi/
├── classification/
│   ├── result.csv
│   └── img_224/
├── segmentation/
│   ├── result.csv
│   └── img_224/
└── multi/
    ├── result.csv
    └── img_224/
```

Each completed BUSI training session appends one row to the `result.csv` file
inside its task folder.
