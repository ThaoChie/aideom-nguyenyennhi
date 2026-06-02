# AIDEOM-VN (Artificial Intelligence and Digital Economy Optimization Model for Vietnam)

Dự án AIDEOM-VN cung cấp một bộ công cụ tối ưu hóa và phân tích chiến lược đầu tư công vào Trí tuệ Nhân tạo (AI) và Chuyển đổi số. Dự án này bao gồm 12 bài toán ứng dụng các kỹ thuật Tối ưu hóa (Optimization) và Học tăng cường (Reinforcement Learning).

## 1. Cài đặt môi trường

Cài đặt Python 3.10+ và thiết lập môi trường ảo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Cấu trúc thư mục

- `src/`: Mã nguồn chính
  - `modules/`: Các module độc lập (M1-M6) cho Đồ án tổng hợp Bài 12.
  - `data.py`: Xử lý dữ liệu đầu vào.
  - `optimization.py`: Các hàm tối ưu hóa cho Bài 5-10, 12.
  - `rl_env.py`: Môi trường Gymnasium và Q-Learning / DQN cho Bài 11.
- `Dashboard/`: Mã nguồn giao diện Streamlit (app.py).
- `tests/`: Bộ Unit Test sử dụng Pytest.
- `docs/`: Báo cáo và tài liệu.

## 3. Khởi chạy Dashboard

Khởi chạy ứng dụng Streamlit:

```bash
streamlit run Dashboard/app.py
```

Sau khi chạy lệnh trên, trình duyệt sẽ tự động mở trang Dashboard tại địa chỉ `http://localhost:8501`.

## 4. Chạy Unit Test

Đảm bảo các mô-đun cốt lõi hoạt động bình thường bằng Pytest:

```bash
pytest tests/
```

## 5. Báo cáo tổng kết

Tham khảo báo cáo Đồ án tổng hợp trong file `docs/report.md`.
