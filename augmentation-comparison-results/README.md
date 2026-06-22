# Augmentation comparison results

Kết quả được sinh bởi `run_augmentation_comparison.py` và notebook
`colab-result/run-augmentation-comparison.ipynb`.

Thiết kế mặc định dùng MK-MNet width 0.25, oversampling cố định, 100 epoch và
ba seed (`42`, `123`, `2026`). Raw medical metrics là kết quả chính.

Các file tổng hợp:

- `comparison_status.csv`: trạng thái train/evaluation của từng policy và seed;
- `comparison_runs.csv`: metric của từng run;
- `comparison_summary.csv`: mean và standard deviation theo policy;
- `previews/*.png`: kiểm tra trực quan image-mask synchronization.

Nhóm policy:

- `none`: baseline không augmentation;
- single augmentation theo `docs/ref/data-augmentation.pdf`;
- `legacy`: fixed sequence cũ của repo;
- `trivial_all_1`, `trivial_all_3`: lấy ngẫu nhiên operation từ pool 18 phép;
- `speckle_noise`: mở rộng riêng cho ultrasound, không thuộc 18 phép của paper.

Không kết luận statistical significance từ ba seed trên một split cố định. Kết
quả chỉ được báo cáo dưới dạng exploratory mean ± std.
