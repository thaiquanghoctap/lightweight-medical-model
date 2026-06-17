# Nhận xét đối chiếu hướng nghiên cứu và repo hiện tại

Tài liệu này ghi nhận sự chuyển dịch ý tưởng từ proposal ban đầu sang hướng triển
khai hiện tại, đồng thời xác định các phần repo đã thực hiện và các điểm cần bổ
sung để vừa giữ được tinh thần mở rộng ban đầu, vừa phù hợp với hướng MedNet hiện
tại.

## 1. Từ công trình data augmentation đến proposal ban đầu

Proposal `25C15038_25C15058_Proposal.pdf` được xây dựng trực tiếp từ
*Analyzing Data Augmentation for Medical Images: A Case Study in Ultrasound
Images* với trọng tâm là phân tích data augmentation cho ảnh siêu âm vú.

Công trình gốc đặt vấn đề rằng data augmentation rất quan trọng trong ảnh y khoa vì dữ
liệu thường ít, mất cân bằng và khó thu thập. Tuy nhiên, không thể mặc định lấy
các phép biến đổi từ ảnh tự nhiên áp dụng thẳng sang ảnh y khoa. Với ảnh siêu âm
vú, các phép như crop mạnh, blur mạnh, random erasing hoặc một số biến đổi hình
học/cường độ quá mạnh có thể làm mất hoặc làm sai lệch vùng tổn thương. Vì vậy,
công trình này đề xuất đánh giá augmentation theo hướng có kiểm soát: đánh giá
từng phép riêng lẻ, đánh giá tổ hợp, rồi đánh giá chiến lược sampling/policy.

Proposal ban đầu giữ khá sát tinh thần này:

- Bài toán chính là classification ảnh siêu âm vú trên BUSI.
- Backbone dự kiến là các mô hình phổ biến như ResNet-18 hoặc EfficientNet-B0.
- Trọng tâm không phải xây kiến trúc mới, mà là đánh giá augmentation nào hữu
  ích, augmentation nào trung tính, augmentation nào có thể gây hại.
- Các hướng đánh giá gồm single augmentation, policy comparison, robustness và
  thảo luận mở rộng sang BUS-BRA hoặc X-ray nếu đủ thời gian.
- Metric kỳ vọng gồm Accuracy/AUC và các metric phù hợp y khoa như F1-score,
  sensitivity, specificity.

Nói cách khác, proposal ban đầu là một kế hoạch thực nghiệm xoay quanh câu hỏi:
`augmentation policy nào phù hợp với ảnh siêu âm y khoa?`

## 2. Vấn đề khi thực nghiệm hướng data augmentation và cơ sở chuyển sang MedNet

Khi đưa hướng từ *Analyzing Data Augmentation for Medical Images: A Case Study in
Ultrasound Images* vào thực nghiệm, có một số vấn đề thực tế khiến scope ban đầu
khó triển khai đầy đủ trong điều kiện hiện tại.

Thứ nhất, hướng data augmentation cần nhiều lần huấn luyện để có kết luận đáng
tin. Công trình gốc dùng cách đánh giá có kiểm soát, nhiều cấu hình augmentation và
so sánh có tính thống kê. Nếu tái hiện đúng tinh thần đó trên BUSI, nhóm cần
chạy baseline, từng augmentation riêng lẻ, nhiều tổ hợp augmentation và các
policy như RandAugment/TrivialAugment. Số lượng thí nghiệm tăng rất nhanh, đặc
biệt nếu dùng backbone như ResNet-18/EfficientNet-B0.

Thứ hai, dataset BUSI nhỏ và mất cân bằng. Nếu chỉ chạy một vài cấu hình
augmentation đơn lẻ, kết quả dễ phụ thuộc vào split hoặc seed. Khi đó rất khó
kết luận chắc augmentation nào thật sự có lợi hay gây hại. Điều này làm hướng
proposal ban đầu cần nhiều seed, cross-validation hoặc ít nhất là lặp lại thí
nghiệm, dẫn đến áp lực tài nguyên huấn luyện.

Thứ ba, nhãn classification cấp ảnh không cho biết mô hình đang học vùng tổn
thương nào. Proposal ban đầu cũng đã nhận diện hạn chế này: để biết augmentation
có làm mất vùng bệnh hay không, cần phân tích trực quan. Trong BUSI, dữ liệu có
mask tổn thương, nên chỉ làm classification sẽ bỏ qua một nguồn thông tin quan
trọng có thể giúp kiểm tra model có tập trung vào vùng bệnh hay không.

Từ các vấn đề trên, nhóm chuyển sang dùng *MedNet: a lightweight
attention-augmented CNN for medical image classification* làm cơ sở định hướng
mới. Công trình MedNet đề xuất một mô hình nhẹ cho ảnh y khoa, kết hợp depthwise
separable convolution với CBAM attention trong residual block. Luận điểm chính
của công trình này là ảnh y khoa có độ phân giải thấp, khác biệt liên lớp nhỏ,
biến thiên nội lớp lớn và cần mô hình vừa đủ mạnh vừa hiệu quả tài nguyên. Đây là
cơ sở hợp lý để nhóm chuyển từ hướng `augmentation-heavy trên backbone tổng quát`
sang hướng `lightweight medical model`.

Trong `progress_update.tex`, sự chuyển hướng được thể hiện như sau:

- Giữ bài toán BUSI và mục tiêu classification ban đầu.
- Thay ResNet/EfficientNet bằng MedNet-style lightweight backbone.
- Tận dụng mask BUSI để mở rộng sang segmentation.
- Kết hợp classification và segmentation thành multi-task learning với shared
  backbone.
- Giữ augmentation như hướng cải tiến chọn lọc sau khi baseline MedNet/multi-task
  ổn định, thay vì so sánh toàn bộ augmentation policy như proposal ban đầu.

Vì vậy, thay đổi hiện tại không nên được hiểu là bỏ hoàn toàn proposal, mà là
điều chỉnh trọng tâm: từ `đánh giá augmentation policy` sang `xây dựng baseline
lightweight medical model`, sau đó mới quay lại augmentation ở phạm vi nhỏ hơn.

## 3. Repo hiện tại đang thực hiện điều gì so với progress update

Repo hiện tại đã bám khá sát các nội dung chính được nêu trong
`progress_update.tex` và `docs/DOC.md`.

### 3.1. Preprocess và tổ chức dữ liệu

`preprocess.py` hỗ trợ chuyển dữ liệu BUSI từ `data/busi_224.npz` sang cấu trúc
folder ảnh:

```text
data/busi/<split>/<class>/images/
data/busi/<split>/<class>/masks/
```

Điều này đáp ứng phần đã đề ra trong progress update: preprocess tạo các folder
ảnh, chia train/val/test, để các script train/test load dữ liệu theo task.

Điểm còn thiếu ở đây là repo chưa có script chuyển từ BUSI ZIP/raw dataset sang
`data/busi_224.npz`. Theo `docs/DOC.md`, flow hiện tại bắt đầu từ NPZ đã có sẵn.

### 3.2. Model construction

`model.py` hiện thực hóa hướng từ MedNet:

- `DepthwiseSeparableConv`: giảm chi phí so với convolution chuẩn.
- `CBAM`: attention theo channel và spatial.
- `ResidualDSCBAMBlock`: residual block có depthwise separable convolution và
  CBAM.
- `ResidualCBAMBackbone`: backbone dùng chung.
- `MedNet`: classification model.
- `MedNetSegmentation`: segmentation model với decoder và skip connection.
- `MedNetMultiTask`: shared backbone với hai nhánh classification và
  segmentation.

Như vậy, repo đã thực hiện đúng ý chính trong `progress_update.tex`: dùng một
lightweight medical model theo tinh thần MedNet, rồi mở rộng thêm segmentation
và multi-task cho BUSI.

### 3.3. Train/test theo từng task

Repo đã có ba nhánh train tương ứng:

- `train_busi_classification.py`: train classification ba lớp benign/malignant/normal.
- `train_busi_segmentation.py`: train segmentation với image-mask pairs.
- `train_busi_multi.py`: train multi-task với shared backbone và loss kết hợp.

Repo cũng có ba file test trực quan:

- `test_busi_classification.py`: xuất ảnh kèm true label, predicted label,
  confidence.
- `test_busi_segmentation.py`: xuất panel ảnh gốc, mask thật, mask dự đoán và
  Dice.
- `test_busi_multi.py`: xuất panel classification + segmentation cho cùng ảnh.

Đây là phần đáp ứng tốt yêu cầu trong `progress_update.tex`: không chỉ có metric
định lượng mà còn có output trực quan để kiểm tra kết quả bằng mắt.

### 3.4. Output và artifact

Repo đã tổ chức output theo task:

```text
outputs/busi/
├── classification/
├── segmentation/
└── multi/
```

Mỗi task có checkpoint, `epoch_log.csv`, `result.csv` và ảnh test trực quan. Phần
`report_images/` gom một số ảnh output đại diện để nhúng vào report.

Nhìn chung, phần implementation hiện tại đã hoàn thành mức `prototype
end-to-end`: preprocess, train, test, checkpoint, metric và ảnh minh họa đều có.

## 4. Các vấn đề còn thiếu hoặc cần làm thêm

Để vừa adapt được định hướng mở rộng ban đầu từ proposal, vừa phù hợp với hướng
MedNet hiện tại, repo/report cần bổ sung theo bốn nhóm sau.

### 4.1. Làm rõ thay đổi câu hỏi nghiên cứu

Proposal ban đầu nói rõ đề tài không tập trung xây kiến trúc mới, mà tập trung
đánh giá augmentation. Trong khi đó repo hiện tại lại tập trung vào MedNet,
segmentation và multi-task.

Vì vậy, báo cáo cuối cần nói rõ đây là một sự điều chỉnh câu hỏi nghiên cứu:

- Câu hỏi ban đầu: augmentation nào phù hợp với ảnh siêu âm y khoa?
- Câu hỏi hiện tại: lightweight medical model kiểu MedNet có thể khai thác BUSI
  hiệu quả đến đâu, đặc biệt khi dùng cả nhãn classification và mask
  segmentation?
- Augmentation được giữ lại như phần cải tiến chọn lọc, không còn là toàn bộ
  trục chính như proposal ban đầu.

Nếu không nói rõ, điểm này dễ bị xem là mâu thuẫn với proposal.

### 4.2. Bổ sung nhánh augmentation chọn lọc

Để nối lại với proposal gốc, repo cần có một phần augmentation thực nghiệm nhỏ,
không cần tái hiện toàn bộ *Analyzing Data Augmentation for Medical Images: A
Case Study in Ultrasound Images*.

Một hướng khả thi:

- Giữ MedNet/Multi-task làm baseline.
- Chọn một số augmentation an toàn cho siêu âm:
  - rotation nhỏ;
  - horizontal/vertical flip nếu hợp lý với dữ liệu;
  - brightness/contrast nhẹ;
  - shift/scale vừa phải;
  - speckle noise.
- Tránh hoặc chỉ phân tích thận trọng các augmentation có nguy cơ làm sai lệch
  vùng bệnh:
  - crop mạnh;
  - blur mạnh;
  - random erasing/CutOut.
- Với segmentation/multi-task, mọi augmentation hình học phải áp dụng đồng bộ
  cho ảnh và mask.

Kết quả không cần so sánh toàn bộ policy như RandAugment/TrivialAugment, nhưng
cần có ít nhất một bảng nhỏ cho thấy augmentation chọn lọc ảnh hưởng thế nào tới
classification và segmentation.

### 4.3. Củng cố đánh giá theo metric y khoa

Proposal ban đầu yêu cầu AUC, F1-score, sensitivity và specificity. Repo hiện đã
có Accuracy/AUC cho classification và Dice/IoU cho segmentation, nhưng còn thiếu
các metric y khoa quan trọng:

- confusion matrix;
- per-class precision/recall/F1;
- Macro-F1;
- sensitivity cho malignant;
- specificity;
- one-vs-rest AUC theo từng lớp.

Điểm này đặc biệt quan trọng vì BUSI mất cân bằng lớp. Accuracy cao có thể che
khuất lỗi bỏ sót malignant. Báo cáo hiện tại đã thừa nhận đây là phần còn thiếu,
nhưng repo nên bổ sung script hoặc hàm xuất các metric này vào `result.csv` hoặc
một file `classification_report.csv`.

### 4.4. Chứng minh tính lightweight và so sánh baseline

Hướng hiện tại dựa trên *MedNet: a lightweight attention-augmented CNN for
medical image classification*, nên cần chứng minh rõ tính lightweight:

- số tham số;
- kích thước checkpoint;
- FLOPs/MACs;
- inference time;
- so sánh với ResNet-18/EfficientNet-B0/MobileNetV2 nếu có thể.

Hiện report đã có số tham số và checkpoint size, nhưng chưa có FLOPs/MACs và thời
gian inference. Ngoài ra, nếu tài nguyên không đủ để train ResNet/EfficientNet
đầy đủ, vẫn nên báo cáo ít nhất số tham số/FLOPs lý thuyết để giải thích vì sao
nhóm chuyển hướng.

### 4.5. Hoàn thiện phân tích segmentation và XAI

Segmentation là phần mở rộng ngoài proposal, nhưng lại giúp giải quyết hạn chế
ban đầu: classification label không chỉ rõ vị trí tổn thương.

Cần làm thêm:

- phân tích lỗi segmentation theo lớp benign/malignant/normal;
- tách riêng lỗi false positive trên normal;
- thử Dice Loss hoặc Dice + BCE;
- thử các giá trị segmentation weight trong multi-task;
- triển khai Grad-CAM cho nhánh classification;
- so sánh Grad-CAM với mask thật để xem mô hình có tập trung vào vùng tổn thương
  hay không.

Điểm này giúp kết nối hai hướng: proposal quan tâm tính hợp lý y khoa của
augmentation, còn hướng hiện tại dùng mask/Grad-CAM để kiểm tra mô hình có học
đúng vùng bệnh hay không.

### 4.6. Xác định lại phạm vi BUS-BRA và X-ray

Proposal ban đầu nhắc BUS-BRA để kiểm tra robustness và X-ray như hướng mở rộng.
Repo hiện tại mới làm BUSI.

Trong báo cáo cuối cần ghi rõ:

- BUSI là phạm vi thực nghiệm chính hiện tại.
- BUS-BRA/X-ray chưa triển khai do tài nguyên và do đã chuyển trọng tâm sang
  MedNet/multi-task.
- Nếu còn thời gian, hướng mở rộng hợp lý hơn là trước hết thử robustness trên
  BUS-BRA; X-ray nên để trong phần future work vì khác modality và cần thiết kế
  augmentation riêng.

## 5. Kết luận nhận xét

Luồng phát triển hiện tại có thể được diễn giải hợp lý như sau:

1. *Analyzing Data Augmentation for Medical Images: A Case Study in Ultrasound
   Images* tạo ý tưởng ban đầu: đánh giá augmentation cho ảnh siêu âm vú.
2. `25C15038_25C15058_Proposal.pdf` chuyển ý tưởng đó thành proposal với trọng tâm
   classification + augmentation policy.
3. Khi thực nghiệm, scope augmentation đầy đủ đòi hỏi nhiều tài nguyên và nhiều
   lần train; đồng thời BUSI có mask nhưng proposal ban đầu chưa khai thác.
4. *MedNet: a lightweight attention-augmented CNN for medical image
   classification* cung cấp cơ sở để chuyển sang lightweight medical model:
   depthwise separable convolution, CBAM, residual block và hiệu quả tài nguyên.
5. Repo hiện tại đã triển khai hướng mới thành pipeline BUSI gồm classification,
   segmentation và multi-task.
6. Để hoàn thiện, nhóm cần bổ sung một nhánh augmentation chọn lọc, metric y khoa,
   baseline/ablation, FLOPs/inference time, Grad-CAM và phân tích lỗi có hệ thống.

Như vậy, thay đổi từ proposal sang progress update là có thể bảo vệ được, nhưng
cần trình bày rõ đây là `điều chỉnh trọng tâm nghiên cứu có căn cứ`, không phải
chỉ là thay đổi implementation. Trọng tâm hiện tại là MedNet/multi-task, còn
augmentation được giữ lại ở phạm vi nhỏ hơn để nối lại với proposal ban đầu.

## 6. Note về ý định hoàn thiện hiện tại

Ở thời điểm hiện tại, pipeline chính của hướng MedNet/BUSI đã gần hoàn chỉnh:
repo đã có preprocess, classification, segmentation, multi-task learning,
train/test, checkpoint, metric cơ bản và output trực quan. Vì vậy, để hoàn thiện
câu chuyện nghiên cứu theo hướng hiện tại, hai mảnh còn thiếu quan trọng nhất là:

1. **Data augmentation chọn lọc**: không quay lại full scope của proposal ban đầu
   và không cần so sánh toàn bộ augmentation policy. Chỉ cần bổ sung một nhóm
   augmentation phù hợp với ảnh siêu âm và tài nguyên hiện có, ví dụ rotation nhỏ,
   brightness/contrast nhẹ, shift/scale vừa phải và speckle noise. Phần này giúp
   nối lại nguồn gốc đề tài từ *Analyzing Data Augmentation for Medical Images:
   A Case Study in Ultrasound Images*.

2. **Grad-CAM/XAI**: bổ sung trực quan hóa Grad-CAM cho nhánh classification để
   kiểm tra mô hình có tập trung vào vùng tổn thương hay không. Vì BUSI có mask,
   Grad-CAM có thể được đặt cạnh ảnh gốc, mask thật và mask dự đoán để tăng sức
   thuyết phục về mặt giải thích y khoa.

Các phần như nhiều seed, confusion matrix, per-class metric, sensitivity,
specificity, FLOPs/MACs và inference time vẫn là các cải tiến quan trọng để làm
báo cáo mạnh hơn. Tuy nhiên, xét theo ý định hoàn thiện hướng chính, data
augmentation chọn lọc và Grad-CAM là hai phần ưu tiên nhất để khép lại khoảng
cách giữa proposal ban đầu và implementation hiện tại.

## 7. Note về các công trình nền và hai biến thể multi-task mới

Hướng hiện tại không chỉ dựa trên một công trình duy nhất. Có thể tách vai trò
các công trình nền như sau:

- *MedNet: a lightweight attention-augmented CNN for medical image
  classification*: cơ sở cho nhánh classification và lightweight medical
  backbone. Ý tưởng chính là dùng depthwise separable convolution, residual
  connection và CBAM attention để tạo mô hình gọn hơn cho ảnh y khoa.
- CBAM-RIUNet: cơ sở ý tưởng cho một hướng segmentation dùng residual/inception
  style và CBAM attention. Trong repo hiện tại chưa có bài báo/source public đi
  kèm, nên phần này nên được mô tả là `phỏng theo ý tưởng` thay vì reproduce
  nguyên bản.
- *MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation*: cơ sở
  cho hướng segmentation nhẹ theo MK-UNet, gồm multi-kernel depthwise
  convolution, multi-kernel inverted residual, channel attention, spatial
  attention và grouped attention gate.

Từ các công trình trên, repo hiện có hai biến thể multi-task quan trọng ngoài
`MedNetMultiTask` ban đầu.

### 7.1. R-CBAM MNet

Model tương ứng trong code là `RCBAMMNet`, được train/test bằng:

```text
train_r_cbam_mnet.py
test_r_cbam_mnet.py
```

Ý tưởng của biến thể này là kết hợp MedNet backbone với segmentation decoder lấy
cảm hứng từ hướng CBAM-RIUNet:

- encoder/classification backbone vẫn là `ResidualCBAMBackbone`;
- backbone dùng các `ResidualDSCBAMBlock`, tức residual + depthwise separable
  convolution + CBAM;
- các skip feature được refine bằng `CBAM`;
- decoder dùng `RCBAMDecoderBlock`;
- phần refinement trong decoder dùng `ResidualInceptionDSCBlock`, tức residual
  block với inception-style depthwise separable convolution.

Vì vậy, khi mô tả model này cần nói chính xác rằng decoder không dùng nguyên
`ResidualDSCBAMBlock`; `ResidualDSCBAMBlock` nằm trong MedNet backbone, còn
decoder dùng residual inception depthwise separable block và CBAM trên skip
features.

Về kích thước, model này ở mức khoảng 6.5M parameters theo cấu trúc hiện tại. Đây
là hướng có tính nhất quán kiến trúc với MedNet vì cùng dùng cơ chế CBAM
attention.

### 7.2. MK-MNet

Model tương ứng trong code là `MKMNet`, được train/test bằng:

```text
train_mk_mnet.py
test_mk_mnet.py
```

Ý tưởng của biến thể này là kết hợp MedNet backbone với decoder theo hướng
MK-UNet:

- encoder/classification backbone vẫn là `ResidualCBAMBackbone`;
- decoder dùng các thành phần lấy cảm hứng từ MK-UNet như
  `MKChannelAttention`, `MKSpatialAttention`, `GroupedAttentionGate`,
  `MultiKernelDepthwiseConv` và `MultiKernelInvertedResidual`;
- model có deep supervision qua các auxiliary segmentation heads;
- model có tham số `width_mult` để scale số channel của MedNet backbone.

Cần diễn đạt cẩn thận rằng MK-MNet khác cơ chế attention ở nhánh
decoder/segmentation so với R-CBAM MNet, nhưng backbone vẫn còn CBAM vì vẫn dựa
trên `ResidualCBAMBackbone(use_cbam=True)`.

Điểm quan trọng của MK-MNet là khả năng scale tham số. Do MK-UNet gốc dùng channel
nhỏ hơn MedNet, repo cho phép scale channel của MedNet backbone bằng `width_mult`.
Theo cấu trúc code hiện tại, số parameters xấp xỉ:

```text
width_mult = 0.25 -> khoảng 0.63M params
width_mult = 0.35 -> khoảng 1.15M params
width_mult = 0.50 -> khoảng 2.25M params
width_mult = 0.75 -> khoảng 4.88M params
width_mult = 1.00 -> khoảng 8.51M params
```

Vì vậy, nhận xét `MK-MNet có thể dao động khoảng 600K đến 8M parameters tùy
scale` là hợp lý, nhưng nên ghi là giá trị xấp xỉ và phụ thuộc cấu hình
`width_mult`.

### 7.3. Cách mô tả ngắn gọn trong report

Một cách diễn đạt phù hợp là:

> Ngoài multi-task baseline ban đầu, nhóm phát triển thêm hai biến thể
> multi-task. Biến thể thứ nhất là R-CBAM MNet, kết hợp MedNet backbone với
> decoder lấy cảm hứng từ CBAM-RIUNet, nhấn mạnh residual/depthwise-separable
> design và CBAM attention. Biến thể thứ hai là MK-MNet, kết hợp MedNet backbone
> với decoder theo MK-UNet, dùng multi-kernel inverted residual, channel/spatial
> attention và grouped attention gate. MK-MNet cho phép scale channel bằng
> `width_mult`, nên có thể kiểm soát số tham số từ mức rất nhẹ đến mức đầy đủ.

Như vậy, MedNet giữ vai trò nền cho classification/backbone, còn CBAM-RIUNet và
MK-UNet cung cấp hai hướng ý tưởng khác nhau cho segmentation decoder trong
multi-task learning.
