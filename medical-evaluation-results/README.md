# Detailed medical evaluation results

Thư mục này nhận kết quả từ `run_all_medical_evaluations.py`.

Mỗi cấu hình có một thư mục riêng chứa:

- `summary.csv`: metric tổng quát cho raw/refined;
- `per_class_metrics.csv`: precision, sensitivity/recall, specificity, F1,
  AUC, Dice và IoU theo lớp;
- `predictions.csv`: dự đoán và xác suất của từng ảnh test;
- confusion matrix dạng counts, normalized CSV và PNG.

Ở cấp thư mục gốc, `evaluation_status.csv` ghi cấu hình completed/failed;
`all_summary.csv` và `all_per_class_metrics.csv` tổng hợp mọi run thành công.

Raw là kết quả chính. Refined chỉ là ablation và chỉ có ở model multi-task.
