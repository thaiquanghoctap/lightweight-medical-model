# Lightweight MedNet: Cải tiến và flow chạy

## 1. Phạm vi tài liệu

Tài liệu này mô tả các phần đã cải tiến so với flow classification ban đầu:

- `preprocess.py`: chuyển dữ liệu NPZ thành các folder ảnh.
- `train.py`: train model MedNet cho bài toán classification.

Phần mở rộng BUSI giữ lại bài toán classification và bổ sung thêm segmentation,
multi-task learning, cùng các file test trực quan.

## 2. Các cải tiến so với flow ban đầu

### 2.1. Preprocess dữ liệu BUSI

Flow preprocess ban đầu xử lý các dataset MedMNIST:

```text
data/<dataset>_224.npz
        |
        v
data/<dataset>/<split>/<class>/images/*.png
```

`preprocess.py` hiện tại hỗ trợ thêm BUSI. File đầu vào là:

```text
data/busi_224.npz
```

Sau khi preprocess, dữ liệu BUSI có cấu trúc:

```text
data/busi/
├── train/
├── val/
└── test/
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

Với mỗi ảnh BUSI:

```text
benign (1).png
```

mask tương ứng được lưu với tên:

```text
benign (1)_mask.png
```

File BUSI NPZ cần chứa các array sau cho mỗi split:

```text
<split>_images
<split>_masks
<split>_labels
<split>_files
class_names
```

Lưu ý: repo hiện có `data/busi.zip` và `data/busi_224.npz`, nhưng chưa có file
script chuyển ZIP thành NPZ. Flow trong code hiện tại bắt đầu từ
`data/busi_224.npz`.

### 2.2. Dùng chung backbone

Model classification ban đầu là `MedNet`, sử dụng `ResidualCBAMBackbone`.
Backbone gồm năm stage `ResidualDSCBAMBlock`. Mỗi stage kết hợp:

- depthwise separable convolution;
- residual connection;
- CBAM channel attention và spatial attention nếu bật `use_cbam`.

Các model BUSI mới tái sử dụng cùng backbone này thay vì tạo encoder riêng.

### 2.3. Classification cho BUSI

File `train_busi_classification.py` bổ sung flow classification độc lập cho
BUSI.

So với `train.py`:

- chỉ đọc dữ liệu trong `images/`, bỏ qua `masks/`;
- sử dụng ba class cố định: `benign`, `malignant`, `normal`;
- vẫn hỗ trợ chạy lặp bằng `--runs`;
- sau mỗi run, append một dòng kết quả vào
  `outputs/busi/classification/result.csv`.

Model sử dụng vẫn là `MedNet`.

### 2.4. Segmentation cho BUSI

File `model.py` bổ sung model `MedNetSegmentation`.

Flow model:

```text
ResidualCBAMBackbone
        |
        v
encoder feature maps
        |
        v
decoder blocks và skip connections
        |
        v
binary segmentation mask
```

File `train_busi_segmentation.py` bổ sung:

- augmentation đồng bộ giữa ảnh và mask;
- loss `BCEWithLogitsLoss`;
- metric Dice và IoU cho validation và test;
- lưu checkpoint tốt nhất theo validation Dice;
- append kết quả vào `outputs/busi/segmentation/result.csv`.

### 2.5. Multi-task learning cho BUSI

File `model.py` bổ sung model `MedNetMultiTask`.

Model có một backbone dùng chung và hai nhánh output:

```text
                         ┌─> classification head -> class logits
input -> shared backbone ┤
                         └─> segmentation decoder -> mask logits
```

Hai task cùng điều chỉnh trọng số của `ResidualCBAMBackbone`:

```text
total_loss =
    classification_loss
    + segmentation_weight * segmentation_loss
```

File `train_busi_multi.py` bổ sung:

- classification accuracy và AUC;
- segmentation Dice và IoU;
- tham số `--segmentation-weight`;
- lưu checkpoint tốt nhất theo validation classification accuracy;
- append kết quả vào `outputs/busi/multi/result.csv`.

### 2.6. Xuất ảnh test trực quan

Flow ban đầu chỉ báo cáo metric trên test set. Phần BUSI bổ sung các file tạo
ảnh trực quan từ một số mẫu test.

`test_busi_classification.py` tạo ảnh có nội dung:

```text
ảnh gốc
true: ...
predict: ...
confidence: ...
```

`test_busi_segmentation.py` tạo ảnh gồm ba panel:

```text
ảnh gốc | mask thật | mask dự đoán + DICE
```

`test_busi_multi.py` tạo ảnh gồm ba panel:

```text
ảnh gốc có classification | mask thật | mask dự đoán + DICE
```

`test_busi_utils.py` chứa các helper dùng chung:

- đọc ảnh và mask;
- lấy mẫu test cân bằng giữa ba class;
- load checkpoint;
- tính Dice;
- ghép các panel thành ảnh PNG.

### 2.7. Tổ chức output theo task

Các artifact BUSI được chia theo task:

```text
outputs/busi/
├── classification/
├── segmentation/
└── multi/
```

Mỗi task có một file `result.csv`. Mỗi lần train xong sẽ append thêm một dòng.
Checkpoint và log theo epoch nằm trong folder cấu hình tương ứng.

## 3. Flow chạy code

### 3.1. Cài dependencies

```bash
uv sync
```

### 3.2. Preprocess BUSI

Input:

```text
data/busi_224.npz
```

Chạy:

```bash
uv run preprocess.py --dataset busi --overwrite
```

Output:

```text
data/busi/<split>/<class>/images/
data/busi/<split>/<class>/masks/
```

Chỉ dùng `--overwrite` khi muốn tạo lại folder `data/busi/`.

### 3.3. Train classification

Chạy:

```bash
uv run train_busi_classification.py
```

Cấu hình mặc định:

```text
image size: 224
CBAM: enabled
runs: 1
batch size: 32
```

Để train lặp ba lần:

```bash
uv run train_busi_classification.py --runs 3
```

Flow:

```text
data/busi/<split>/<class>/images/
        |
        v
BUSIImageDataset
        |
        v
augmentation và normalization
        |
        v
MedNet classifier
        |
        v
FocalLoss, AdamW, CosineAnnealingLR, early stopping
        |
        v
best_model.pt, epoch_log.csv, classification/result.csv
```

### 3.4. Train segmentation

Chạy:

```bash
uv run train_busi_segmentation.py
```

Cấu hình mặc định:

```text
image size: 224
CBAM: enabled
batch size: 16
```

Flow:

```text
data/busi/<split>/<class>/{images,masks}/
        |
        v
BUSISegmentationDataset
        |
        v
augmentation đồng bộ ảnh và mask
        |
        v
MedNetSegmentation
        |
        v
BCEWithLogitsLoss, Dice, IoU
        |
        v
best_model.pt, epoch_log.csv, segmentation/result.csv
```

### 3.5. Train multi-task

Chạy:

```bash
uv run train_busi_multi.py
```

Cấu hình mặc định:

```text
image size: 224
CBAM: enabled
segmentation weight: 1.0
batch size: 8
```

Để thay đổi trọng số segmentation loss:

```bash
uv run train_busi_multi.py --segmentation-weight 0.5
```

Flow:

```text
data/busi/<split>/<class>/{images,masks}/
        |
        v
BUSIMultiTaskDataset
        |
        v
shared ResidualCBAMBackbone
        |
        ├─> classification head -> FocalLoss
        |
        └─> segmentation decoder -> BCEWithLogitsLoss
        |
        v
combined loss cập nhật shared backbone
        |
        v
best_model.pt, epoch_log.csv, multi/result.csv
```

### 3.6. Tạo ảnh test trực quan

Sau khi train, chạy:

```bash
uv run test_busi_classification.py
uv run test_busi_segmentation.py
uv run test_busi_multi.py
```

Mỗi file test thực hiện:

1. load checkpoint `best_model.pt`;
2. chọn một số ảnh trong test set từ ba class BUSI;
3. chạy inference;
4. lưu ảnh PNG đã ghi thông tin dự đoán.

Để đổi số lượng ảnh:

```bash
uv run test_busi_multi.py --num-samples 15
```

## 4. Cấu trúc output BUSI

Sau khi chạy ba file train và ba file test với cấu hình mặc định:

```text
outputs/
└── busi/
    ├── classification/
    │   ├── result.csv
    │   └── img_224/
    │       ├── cbam/
    │       │   └── run_1/
    │       │       ├── best_model.pt
    │       │       └── epoch_log.csv
    │       └── test_images/
    │           └── *.png
    ├── segmentation/
    │   ├── result.csv
    │   └── img_224/
    │       ├── cbam/
    │       │   ├── best_model.pt
    │       │   └── epoch_log.csv
    │       └── test_images/
    │           └── *.png
    └── multi/
        ├── result.csv
        └── img_224/
            ├── cbam/
            │   └── seg_weight_1/
            │       ├── best_model.pt
            │       └── epoch_log.csv
            └── test_images/
                └── *.png
```

Nếu chạy classification với `--runs 3`, folder được tạo thêm:

```text
outputs/busi/classification/img_224/cbam/
├── run_1/
├── run_2/
└── run_3/
```

Nếu đổi image size, CBAM mode hoặc segmentation weight, code sẽ tự tạo folder
cấu hình tương ứng.

