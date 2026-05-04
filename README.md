# 🐾 Meow-tlier Detection Project

## 1. Giới thiệu
Dự án nhằm phát hiện các hành vi sinh hoạt bất thường (Outliers) của mèo trong tiệm cà phê thông qua dữ liệu về lượng thức ăn và thời gian ngủ.

## 2. Phương pháp tiếp cận
* **Thống kê:** Sử dụng Z-Score ($z = \frac{x - \mu}{\sigma}$) để bắt các điểm nằm ngoài 3 độ lệch chuẩn.
* **Machine Learning:** Sử dụng Elliptic Envelope để xác định ranh giới bao bọc dữ liệu 2 chiều.

## 3. Cấu trúc thư mục dự án
- `/data`: Chứa các file CSV (Clean, Noisy, Extreme).
- `/src`: Mã nguồn chính (Data Loader, Model, Visualization).
- `/notebooks`: File nháp thực nghiệm (Jupyter Notebook).
- `/results`: Hình ảnh đồ thị và bảng kết quả so sánh.

## 4. Quy tắc làm việc cho Team
- **Git:** Mỗi task trên Jira tương ứng với 1 branch `feature/mã-task`.
- **Jira:** Cập nhật trạng thái task hàng ngày (Chưa làm -> Đang làm -> (Đang gặp vấn đề) -> Chờ duyệt -> Hoàn thành).
