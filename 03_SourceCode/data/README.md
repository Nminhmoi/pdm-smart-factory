# Dữ liệu Dự án: AI4I 2020 Predictive Maintenance Dataset

## 1. Nguồn dữ liệu
- Nguồn: [UCI Machine Learning Repository - AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- Kích thước: 10,000 dòng x 14 cột.
- File dữ liệu gốc được đặt tại: `data/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv`

## 2. Từ điển dữ liệu (Data Dictionary)
- `UDI`: Định danh bản ghi (1 đến 10000) - **Loại bỏ để tránh rò rỉ dữ liệu**.
- `Product ID`: Mã sản phẩm (ví dụ: M14860) - **Loại bỏ**.
- `Type`: Cấp chất lượng máy gồm L (Low - 50%), M (Medium - 30%), H (High - 20%).
- `Air temperature [K]`: Nhiệt độ không khí xung quanh máy (Kelvin).
- `Process temperature [K]`: Nhiệt độ quy trình gia công (Kelvin).
- `Rotational speed [rpm]`: Tốc độ quay của trục chính (vòng/phút).
- `Torque [Nm]`: Mô-men xoắn (Newton-mét).
- `Tool wear [min]`: Thời gian mòn dao/dụng cụ cắt (phút).
- `Machine failure`: Nhãn mục tiêu chính (0: Không hỏng, 1: Hỏng) - Tỷ lệ hỏng khoảng 3.39% (Bài toán mất cân bằng dữ liệu).
- Các cờ hỏng hóc thành phần (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`): Các nguyên nhân hỏng hóc - **Tuyệt đối không đưa vào đặc trưng huấn luyện $X$ khi dự đoán `Machine failure`**.
