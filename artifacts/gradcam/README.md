# Grad-CAM experiment results

Thư mục này dùng để lưu và so sánh kết quả Grad-CAM trong quá trình thử nghiệm.
Các ảnh được chọn để đưa vào báo cáo nên được sao chép sang
`report/report_images/`.

Quy ước thư mục:

```text
artifacts/gradcam/
└── <model>/
    └── stage_<stage>/
        └── target_<predicted|true>/
```

Ví dụ:

```text
artifacts/gradcam/mk_mnet_width025/stage_5/target_true/
```

Nên tạo cùng một tập mẫu với `--seed` cố định khi so sánh các stage.

Để sinh kết quả cho toàn bộ test set, dùng `--all-samples`. Khi có cờ này,
`--num-samples` sẽ được bỏ qua.
