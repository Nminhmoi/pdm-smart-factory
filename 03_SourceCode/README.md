# Bảo trì dự đoán trong nhà máy thông minh

**Học phần:** Học máy (Machine Learning) – HP 608807  
**Nhóm:** Minh (trưởng nhóm), An, Thành, Nguyên, Vũ  
**Đề tài:** Predictive Maintenance in Industry 4.0  
**Bài toán chính:** Phân loại nhị phân dự đoán `Machine failure` (0: không hỏng, 1: hỏng)

> README này chuyển hóa yêu cầu và rubric trong file đề bài thành kế hoạch thực hiện cho nhóm. Hạn nộp chính thức theo mốc Buổi 7 và lịch giảng viên thông báo; nhóm cần xác nhận đăng ký đề tài được duyệt ở Buổi 2.

## 1. Mục tiêu và phạm vi

Xây dựng quy trình học máy đầu cuối để dự đoán nguy cơ máy bị hỏng từ các thông số vận hành, giải thích lựa chọn mô hình, so sánh ít nhất 3 thuật toán, đóng gói pipeline và mô hình tốt nhất, sau đó trình bày kết quả và demo dự đoán.

Dùng AI4I 2020 Predictive Maintenance Dataset từ [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). Ghi rõ nguồn và ngày tải dữ liệu trong báo cáo. Đề tài/dataset lấy từ đề xuất của nhóm; rubric yêu cầu đề tài tự đề xuất phải được giảng viên phê duyệt.

### Mục tiêu dự đoán

- **Bài toán chính:** dự đoán `Machine failure` nhị phân.
- **Mở rộng tùy thời gian:** phân loại nguyên nhân lỗi cụ thể (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) theo thiết kế nhãn và giới hạn dữ liệu; không thay thế kết quả bài toán chính.
- Nhấn mạnh đây là dữ liệu benchmark/mô phỏng dùng cho học thuật, không đại diện đầy đủ mọi dây chuyền thực tế và không đủ căn cứ để tự động ra quyết định an toàn công nghiệp.

### Đặc trưng và nhãn

Các biến vận hành chính: nhiệt độ không khí (`Air temperature`), nhiệt độ quy trình (`Process temperature`), tốc độ quay (`Rotational speed`), mô-men xoắn (`Torque`), thời gian mòn dụng cụ (`Tool wear`) và cấp chất lượng (`Type`: L/M/H). Bộ dữ liệu có thêm cột định danh và các cờ lỗi thành phần; kiểm tra mô tả dataset để xác định cột nào được dùng.

**Tránh rò rỉ nhãn:** loại `UDI` và `Product ID` khỏi đặc trưng dự đoán. Không đưa `Machine failure` hoặc các cờ nguyên nhân (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) vào X khi y là `Machine failure`, vì các cờ này mô tả trực tiếp sự kiện/nhãn. Báo cáo rõ quyết định chọn biến. Không dùng test set để lựa chọn đặc trưng, ngưỡng hoặc siêu tham số.

## 2. Quy trình thực hiện

### Bước 1 — Chốt đề tài và câu hỏi nghiên cứu

Đăng ký nhóm, thành viên, lớp, đề tài, nguồn dữ liệu và mục tiêu dự đoán. Câu hỏi gợi ý: *Các thông số vận hành và loại máy có giúp phát hiện sớm nguy cơ hỏng tốt hơn baseline không? Mô hình nào cân bằng tốt khả năng phát hiện lỗi và cảnh báo nhầm?*

### Bước 2 — Lấy dữ liệu, khám phá và tiền xử lý

1. Tải dữ liệu gốc, giữ nguyên bản trong `data/raw/`; ghi nguồn, ngày tải, giấy phép/thông tin sử dụng nếu có.
2. Kiểm tra kích thước, kiểu dữ liệu, giá trị thiếu, trùng lặp, phạm vi, phân phối lớp và các cờ lỗi. Mô tả tỷ lệ mẫu lỗi để cho thấy mức mất cân bằng.
3. Thực hiện EDA bằng bảng và biểu đồ: phân phối biến, tương quan, tỷ lệ lỗi theo `Type`, biểu đồ so sánh nhóm lỗi và không lỗi. Chú thích rằng tương quan không chứng minh quan hệ nhân quả.
4. Xử lý dữ liệu dựa trên bằng chứng EDA; không tự ý xóa ngoại lệ hợp lệ. Mã hóa biến phân loại (one-hot phù hợp cho `Type`) và chuẩn hóa mô hình cần scaling.
5. Chia train/test theo phân tầng (ví dụ 80/20, `random_state` cố định). Mọi bước học từ dữ liệu như imputer, encoder, scaler, chọn đặc trưng và resampling chỉ được fit trên train/fold huấn luyện thông qua pipeline. Giữ test độc lập đến đánh giá cuối.

### Bước 3 — Baseline và ít nhất 3 mô hình

Thử ít nhất ba thuật toán khác nhau, chẳng hạn:

1. Logistic Regression (baseline dễ giải thích; có thể cần `class_weight`).
2. Decision Tree (cây quyết định, dễ diễn giải).
3. Random Forest (ensemble cây).

Có thể thử thêm k-NN, Gradient Boosting hoặc HistGradientBoosting. Mô tả ngắn nguyên lý và lý do chọn. Tất cả mô hình phải đi qua quy trình tiền xử lý nhất quán để so sánh công bằng. Nếu xử lý mất cân bằng bằng `class_weight` hoặc phương pháp khác, chỉ áp dụng bên trong tập huấn luyện/fold và nêu rõ.

### Bước 4 — Pipeline, validation và tinh chỉnh

- Dùng `ColumnTransformer` cho cột số/phân loại và `Pipeline` của Scikit-learn để gom tiền xử lý + mô hình.
- Dùng `StratifiedKFold` (ví dụ 5 folds) trên tập train; lưu seed và cấu hình để tái lập.
- Tinh chỉnh mô hình bằng `GridSearchCV` hoặc `RandomizedSearchCV` với metric phù hợp, ưu tiên `recall`/`F1` hoặc `average_precision` tùy mục tiêu đã giải thích. Không tinh chỉnh theo test set.
- Sau khi khóa lựa chọn và cấu hình, đánh giá đúng một lần trên test set. Báo cáo kết quả cross-validation và kết quả test tách biệt.

### Bước 5 — Đánh giá, giải thích và phân tích lỗi

Không chỉ dùng accuracy: báo cáo precision, recall, F1, confusion matrix và ROC-AUC; do dữ liệu có thể mất cân bằng, bổ sung PR-AUC/average precision. Giải thích ý nghĩa đánh đổi: bỏ sót hỏng hóc so với cảnh báo nhầm. Nếu đổi ngưỡng phân loại, chọn ngưỡng trên validation/train CV, ghi rõ cách chọn rồi mới áp dụng cho test.

Phân tích các false negative và false positive tiêu biểu; xem hiệu năng theo nhóm `Type` nếu số mẫu đủ. Trình bày ROC curve, confusion matrix và feature importance/giải thích mô hình với lưu ý phương pháp tính. Kết luận hạn chế về mất cân bằng, tính đại diện và ứng dụng thực tế.

### Bước 6 — Lưu mô hình và demo

Lưu **toàn bộ pipeline đã fit** (tiền xử lý + mô hình) thành `models/best_model.joblib`. Cung cấp `predict.py` để nạp pipeline, nhận một bản ghi mới và trả về nhãn cùng xác suất/điểm rủi ro nếu mô hình hỗ trợ. Kèm ví dụ đầu vào, đầu ra, cách cài dependencies và lệnh chạy. Dùng đúng thứ tự/names/units của đặc trưng huấn luyện; không fit lại tiền xử lý khi dự đoán.

Chuẩn bị demo ngắn có phương án dự phòng bằng ảnh/video nếu môi trường trình diễn lỗi. Không tuyên bố mô hình sẵn sàng vận hành trực tiếp trong nhà máy.

## 3. Phân công chi tiết và tiến độ từng thành viên

Phân công theo đầu việc chính; mọi người cùng review code, hiểu toàn bộ pipeline và tham gia viết báo cáo/thuyết trình. Tỷ lệ dưới đây là **mục tiêu đóng góp ban đầu**, nhóm phải xác nhận và ký bảng phân công trong báo cáo; điều chỉnh theo đóng góp thực tế.

| Thành viên | Vai trò chính | Trọng tâm phụ trách | Tỷ lệ dự kiến |
|---|---|---|---:|
| **Minh (Leader)** | Trưởng nhóm & Tích hợp hệ thống | Điều phối, chống rò rỉ dữ liệu, quản lý Git, tổng hợp báo cáo kỹ thuật | 22% |
| **An** | Kỹ sư Dữ liệu & EDA | Thu thập dữ liệu, phân tích khám phá (EDA), đề xuất tiền xử lý | 20% |
| **Thành** | Kỹ sư Pipeline & Mô hình | Xây dựng Pipeline tiền xử lý, huấn luyện 3 mô hình, Stratified CV, tinh chỉnh siêu tham số | 20% |
| **Nguyên** | Kỹ sư Đánh giá & Diễn giải | Đánh giá độc lập trên test, tối ưu ngưỡng quyết định, phân tích lỗi & feature importance | 19% |
| **Vũ** | Kỹ sư Triển khai & Trình diễn | Đóng gói pipeline (`.joblib`), viết script `predict.py`, chuẩn hóa `requirements.txt`, làm slide & video demo | 19% |

---

### 3.1. Minh (Leader - Điều phối, Tích hợp & Kiểm soát chất lượng)

| Tiến độ | Mã task | Task chi tiết (Hành động cụ thể) | Output bàn giao | Tiêu chí nghiệm thu (DoD) | Mốc hạn |
|:---:|---|---|---|---|:---:|
| [ ] | **M-01** | **Khởi tạo dự án & Quản lý Git repository**: Tạo repo GitHub/GitLab, cấu hình `.gitignore`, chuẩn hóa cây thư mục dự án theo cấu trúc yêu cầu của đề bài, phân quyền và quy định luồng merge branch (PR). | - Repository Git hoạt động<br>- File `.gitignore` chuẩn Python/Jupyter<br>- Khung thư mục nộp bài chuẩn | Repo có branch protected, không commit file rác/cache (`.venv`, `__pycache__`, `.ipynb_checkpoints`). | Milestone 0 |
| [ ] | **M-02** | **Đăng ký đề tài & Phối hợp giảng viên**: Hoàn thiện thông tin nhóm, đề tài "Predictive Maintenance in Industry 4.0", mục tiêu dự đoán nhị phân, nguồn dữ liệu UCI AI4I 2020; nộp và nhận phản hồi chấp thuận từ GV. | - Phiếu/Email đăng ký đề tài được GV duyệt | Đề tài và nguồn dữ liệu được giảng viên phê duyệt chính thức ở Buổi 2. | Milestone 0 |
| [ ] | **M-03** | **Kiểm soát rủi ro Data Leakage & Tính toàn vẹn**: Rà soát độc lập quy trình chia dữ liệu (Train/Test Split); kiểm tra nghiêm ngặt không rò rỉ các cột `UDI`, `Product ID`, các cờ nguyên nhân lỗi (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) vào ma trận đặc trưng $X$; đảm bảo các phép fit (Scaler/Encoder) chỉ nằm trong fold huấn luyện. | - Biên bản checklist rà soát Data Leakage<br>- Code review trên PR của Thành và An | Không có bất kỳ bước tiền xử lý nào fit trên toàn bộ dữ liệu trước khi split hoặc làm rò rỉ dữ liệu test. | Milestone 1 |
| [ ] | **M-04** | **Tổng hợp & Định dạng Báo cáo kỹ thuật**: Tiếp nhận các chương từ An, Thành, Nguyên, Vũ; tổng hợp thành tài liệu báo cáo hoàn chỉnh; định dạng chuẩn yêu cầu (15–25 trang A4, Times New Roman 13, giãn dòng 1.3–1.5, mục lục, bảng biểu, trích dẫn tài liệu tham khảo chuẩn). | - File Word nguồn & File PDF xuất ra: `01_Report/Bao_cao_BTL_[Tên_Nhóm].pdf` | Đủ 5 chương, đúng quy cách căn lề, có trang bìa, bảng phân công nhiệm vụ và chữ ký của cả 5 thành viên. | Milestone 2 |
| [ ] | **M-05** | **Nghiệm thu gói nộp bài cuối cùng (Final Package QC)**: Rà soát kiểm tra toàn bộ cây thư mục gói nộp trước giờ G (Report, Slide, SourceCode, Video demo); đối chiếu rubric chấm điểm, đảm bảo mọi đường dẫn dữ liệu tương đối và notebook chạy thông suốt từ đầu đến cuối. | - Thư mục/File nén zip nộp bài chuẩn cấu trúc LMS<br>- Biên bản tự chấm điểm theo rubric | Đáp ứng 100% danh mục checklist trước khi nộp, nộp trước hạn chót ít nhất 24 giờ. | Milestone 2 |

---

### 3.2. An (Data Engineer & EDA Specialist)

| Tiến độ | Mã task | Task chi tiết (Hành động cụ thể) | Output bàn giao | Tiêu chí nghiệm thu (DoD) | Mốc hạn |
|:---:|---|---|---|---|:---:|
| [ ] | **A-01** | **Thu thập & Lưu trữ dữ liệu gốc**: Tải trọn vẹn bộ dữ liệu AI4I 2020 Predictive Maintenance Dataset từ UCI ML Repository; lưu trữ nguyên bản (read-only) trong `data/raw/`; lập tài liệu nguồn gốc, ngày truy cập và giấy phép sử dụng. | - File `data/raw/ai4i2020.csv`<br>- File `data/README.md` (metadata, từ điển dữ liệu) | Dữ liệu gốc nguyên vẹn không chỉnh sửa bằng tay; `data/README.md` giải thích chi tiết ý nghĩa từng cột và đơn vị đo. | Milestone 0 |
| [ ] | **A-02** | **Kiểm tra chất lượng dữ liệu (Data Quality Audit)**: Lập trình kiểm tra kích thước mẫu (10,000 dòng), kiểu dữ liệu từng cột, phát hiện giá trị thiếu (missing values), dòng trùng lặp (duplicates), phạm vi giá trị bất thường (outliers/min-max vô lý). | - Script/Mục kiểm tra chất lượng trong notebook<br>- Bảng thống kê mô tả (summary statistics) | Xác định chính xác 100% tình trạng missing/duplicate và phân loại các biến số/biến phân loại. | Milestone 1 |
| [ ] | **A-03** | **Phân tích dữ liệu khám phá chuyên sâu (EDA)**: Vẽ biểu đồ trực quan hóa: phân phối tỷ lệ mất cân bằng của nhãn `Machine failure` (~3.39% lỗi); ma trận tương quan giữa các thông số vật lý (nhiệt độ khí, nhiệt độ quy trình, tốc độ quay, mô-men xoắn, độ mòn dụng cụ); phân tích tỷ lệ lỗi theo loại máy (`Type` L/M/H); xuất các hình ảnh đồ họa có độ phân giải cao. | - Thư mục hình ảnh `reports/figures/eda/`<br>- Các biểu đồ: histogram, boxplot so sánh nhóm hỏng vs không hỏng, correlation heatmap | Có chú thích phân tích cho từng hình ảnh; chỉ rõ hiện tượng mất cân bằng lớp trầm trọng cần xử lý ở bước mô hình. | Milestone 1 |
| [ ] | **A-04** | **Xây dựng Notebook EDA & Đề xuất tiền xử lý**: Lập trình hoàn chỉnh notebook `01_EDA_Preprocessing.ipynb`; mã hóa biến phân loại `Type` (One-Hot / Ordinal), đề xuất loại bỏ `UDI`, `Product ID` và giải thích lý do xử lý ngoại lệ dựa trên bằng chứng dữ liệu. | - Notebook `notebooks/01_EDA_Preprocessing.ipynb` (chạy trơn tru từ đầu đến cuối không lỗi) | Code sạch (PEP8), có markdown giải thích ngữ cảnh nghiệp vụ cơ khí công nghiệp trước mỗi đoạn code. | Milestone 1 |
| [ ] | **A-05** | **Soạn thảo nội dung Báo cáo (Chương 1 & Chương 2)**: Viết bản thảo Chương 1 (Giới thiệu bài toán, mục tiêu dự đoán bảo trì thông minh) và Chương 2 (Dữ liệu, từ điển đặc trưng, phát hiện qua EDA, chiến lược tiền xử lý) gửi Minh tích hợp. | - File bản thảo nội dung (Word/Markdown) Chương 1 và 2<br>- Các bảng biểu số liệu và biểu đồ EDA đính kèm | Đúng format quy định, diễn đạt mạch lạc, viện dẫn nguồn UCI và tài liệu tham khảo liên quan. | Milestone 2 |

---

### 3.3. Thành (ML Modeling & Pipeline Engineer)

| Tiến độ | Mã task | Task chi tiết (Hành động cụ thể) | Output bàn giao | Tiêu chí nghiệm thu (DoD) | Mốc hạn |
|:---:|---|---|---|---|:---:|
| [ ] | **T-01** | **Xây dựng Scikit-learn Pipeline chuẩn**: Thiết kế `ColumnTransformer` đóng gói chuẩn hóa biến số (`StandardScaler`) và mã hóa biến hạng mục `Type` (`OneHotEncoder`); gắn kết trực tiếp với Estimator qua `sklearn.pipeline.Pipeline`; cố định `random_state` để tái lập 100%. | - Module Pipeline code tái sử dụng được trong `notebooks/02_Modeling_Evaluation.ipynb` | Toàn bộ tiền xử lý được đóng gói trong Pipeline, không có bước fit ngoài fold. | Milestone 1 |
| [ ] | **T-02** | **Chia dữ liệu phân tầng & Thiết lập Stratified K-Fold CV**: Chia tập Train/Test theo tỷ lệ 80/20 có phân tầng (`stratify=y`) bảo toàn tỷ lệ nhãn lỗi; thiết lập 5-fold Stratified Cross-Validation độc lập trên tập Train để phục vụ huấn luyện và đánh giá khách quan. | - Code phân chia dữ liệu và cấu hình `StratifiedKFold(n_splits=5, shuffle=True)` | Tập Test được cô lập hoàn toàn, không can thiệp vào bất kỳ khâu huấn luyện hay chọn tham số nào. | Milestone 1 |
| [ ] | **T-03** | **Huấn luyện mô hình Baseline & Ít nhất 3 thuật toán**: Xây dựng mô hình cơ sở Dummy/Logistic Regression; huấn luyện ít nhất 3 thuật toán học máy phân loại: Logistic Regression (có `class_weight='balanced'`), Decision Tree, Random Forest (và mở rộng Gradient Boosting / XGBoost nếu có thời gian). | - Code huấn luyện các mô hình trong notebook `02_Modeling_Evaluation.ipynb`<br>- Bảng log kết quả cross-validation sơ bộ | Đủ ít nhất 3 thuật toán khác biệt về bản chất; có xử lý vấn đề mất cân bằng lớp trong quá trình fit. | Milestone 1 |
| [ ] | **T-04** | **Tinh chỉnh siêu tham số (Hyperparameter Tuning)**: Áp dụng `GridSearchCV` hoặc `RandomizedSearchCV` trên không gian tham số của các mô hình; tối ưu hóa theo metric phù hợp với bài toán mất cân bằng (ưu tiên `f1` hoặc `average_precision` / `roc_auc`). | - Bảng so sánh tham số tối ưu (best params)<br>- Biểu đồ/bảng so sánh hiệu năng trước và sau tuning trên 5-fold CV | Có giải thích lý do chọn metric tối ưu thay vì accuracy; lưu lại cấu hình tốt nhất của từng mô hình. | Milestone 2 |
| [ ] | **T-05** | **Soạn thảo nội dung Báo cáo (Chương 3)**: Viết bản thảo Chương 3 (Cơ sở lý thuyết của các thuật toán đã dùng, cấu trúc thiết kế Pipeline, quy trình thẩm định chéo Stratified CV và không gian tinh chỉnh siêu tham số) gửi Minh tích hợp. | - File bản thảo Chương 3 (Word/Markdown)<br>- Sơ đồ khối kiến trúc Pipeline học máy | Diễn giải rõ ràng nguyên lý hoạt động của từng mô hình, ưu nhược điểm và sơ đồ pipeline trực quan. | Milestone 2 |

---

### 3.4. Nguyên (Evaluation & Explainability Specialist)

| Tiến độ | Mã task | Task chi tiết (Hành động cụ thể) | Output bàn giao | Tiêu chí nghiệm thu (DoD) | Mốc hạn |
|:---:|---|---|---|---|:---:|
| [ ] | **N-01** | **Đánh giá khách quan trên tập Test độc lập**: Sau khi Thành khóa lựa chọn siêu tham số tối ưu trên CV, tiến hành đánh giá đúng duy nhất một lần trên tập Test; tính toán bộ metric toàn diện: Accuracy, Precision, Recall, Macro F1, Weighted F1, ROC-AUC, PR-AUC. | - Bảng tổng hợp so sánh hiệu năng các mô hình trên tập Test độc lập | Báo cáo rành mạch, phân biệt rõ điểm số CV trên tập Train và điểm số thực tế trên tập Test. | Milestone 2 |
| [ ] | **N-02** | **Trực quan hóa hiệu năng mô hình (Evaluation Curves)**: Lập trình vẽ các biểu đồ đánh giá chuyên sâu: Ma trận nhầm lẫn (Confusion Matrix cả dạng đếm số lượng và dạng chuẩn hóa tỷ lệ %); đường cong ROC curve kèm trị số AUC; đường cong Precision-Recall curve kèm Average Precision (AP). | - Bộ hình ảnh biểu đồ trong `reports/figures/eval/`<br>- Biểu đồ Confusion Matrix, ROC, PR Curves sắc nét | Biểu đồ có nhãn trục rõ ràng, chú thích đơn vị và ghi chú so sánh trực diện giữa các thuật toán. | Milestone 2 |
| [ ] | **N-03** | **Tối ưu ngưỡng quyết định (Threshold Tuning) & Phân tích đánh đổi**: Phân tích trade-off giữa Precision và Recall trong bối cảnh công nghiệp (chi phí bỏ sót máy hỏng - False Negative so với chi phí kiểm tra nhầm - False Positive); tìm ngưỡng quyết định tối ưu trên tập validation và kiểm chứng trên test. | - Biểu đồ Threshold vs Precision/Recall/F1<br>- Phân tích bảng ma trận chi phí sai sót (Cost Matrix) | Nêu rõ lý do chọn ngưỡng (ví dụ hạ threshold từ 0.5 xuống 0.3 để tăng Recall bảo đảm an toàn vận hành). | Milestone 2 |
| [ ] | **N-04** | **Phân tích lỗi (Error Analysis) & Giải thích mô hình (Explainability)**: Trích xuất và phân tích cụ thể các trường hợp dự đoán sai tiêu biểu (các ca FN nguy hiểm và FP gây lãng phí); tính toán mức độ quan trọng của đặc trưng (Feature Importance / Permutation Importance / SHAP nếu có) để giải thích yếu tố gây hỏng hóc máy. | - Danh sách bản ghi FN/FP phân tích nguyên nhân<br>- Biểu đồ Feature Importance của mô hình tốt nhất | Giải thích được nguyên nhân vật lý (ví dụ: Tool wear cao kết hợp Torque lớn gây hỏng máy); chỉ rõ hạn chế của mô hình. | Milestone 2 |
| [ ] | **N-05** | **Soạn thảo nội dung Báo cáo (Chương 4 & Chương 5)**: Viết bản thảo Chương 4 (Thực nghiệm, bảng kết quả, phân tích đồ thị, phân tích lỗi sai) và Chương 5 (Kết luận, trả lời câu hỏi nghiên cứu, cảnh báo an toàn thực tế, hướng phát triển tương lai) gửi Minh tích hợp. | - File bản thảo Chương 4 và Chương 5 (Word/Markdown)<br>- Các bảng số liệu thực nghiệm và hình vẽ đính kèm | Nhấn mạnh cảnh báo dữ liệu mô phỏng không tự ý dùng ra quyết định an toàn công nghiệp thực tế nếu chưa kiểm chứng. | Milestone 2 |

---

### 3.5. Vũ (Packaging, Deployment & Demo Specialist)

| Tiến độ | Mã task | Task chi tiết (Hành động cụ thể) | Output bàn giao | Tiêu chí nghiệm thu (DoD) | Mốc hạn |
|:---:|---|---|---|---|:---:|
| [ ] | **V-01** | **Đóng gói Pipeline tối ưu thành Artefact**: Tiếp nhận mô hình tốt nhất từ Thành và Nguyên; xuất toàn bộ Pipeline (gồm cả tiền xử lý + mô hình đã fit đầy đủ) ra file tuần tự hóa `models/best_model.joblib`. | - File `models/best_model.joblib`<br>- Script xuất mô hình an toàn, không bị lỗi version | File nạp lại dự đoán bình thường mà không cần gọi hàm `fit` lại từ đầu; kích thước file hợp lý. | Milestone 2 |
| [ ] | **V-02** | **Xây dựng script dự đoán độc lập `predict.py`**: Viết script CLI nhận thông số đầu vào mới của một máy (Type, Air temp, Process temp, Speed, Torque, Tool wear); nạp pipeline từ `best_model.joblib`; trả về kết quả dự đoán nguy cơ hỏng (0/1) cùng xác suất lỗi (Probability %). | - File `predict.py`<br>- Kịch bản test với ít nhất 2 mẫu input (1 bình thường, 1 bất thường) | Chạy được lệnh dự đoán trực tiếp từ Terminal/PowerShell; có bắt lỗi dữ liệu đầu vào không hợp lệ. | Milestone 2 |
| [ ] | **V-03** | **Chuẩn hóa môi trường & Hướng dẫn cài đặt**: Tạo file `requirements.txt` ghim rõ ràng phiên bản các thư viện chính (scikit-learn, pandas, numpy, joblib, matplotlib, seaborn); viết hướng dẫn thiết lập môi trường ảo (`venv`), cài dependencies và lệnh chạy mẫu trong file `README.md`. | - File `requirements.txt`<br>- Phần hướng dẫn cài đặt và chạy trong README | Test thành công quy trình cài đặt từ đầu trên một máy tính hoàn toàn mới (Clean Environment Test). | Milestone 2 |
| [ ] | **V-04** | **Thiết kế Slide thuyết trình chuyên nghiệp**: Biên soạn slide báo cáo bảo vệ đề tài (12–15 slide chuẩn quy định: Bối cảnh -> Bài toán -> Dữ liệu/EDA -> Pipeline -> Kết quả/So sánh -> Demo & Kết luận); chuẩn bị phân chia lượt nói cho cả 5 thành viên. | - File trình chiếu PDF: `02_Slide/Slide_Thuyet_trinh_[Tên_Nhóm].pdf`<br>- Kịch bản phân vai thuyết trình cho 5 người | Slide súc tích, hình ảnh trực quan, cỡ chữ dễ đọc, nêu bật được đóng góp của từng người. | Milestone 2 |
| [ ] | **V-05** | **Xây dựng kịch bản Demo & Video dự phòng**: Chuẩn bị giao diện hoặc kịch bản chạy demo tương tác trực tiếp trước hội đồng; quay trước 01 video demo ngắn (khoảng 3 phút) ghi lại thao tác chạy dự đoán và kết quả để làm phương án dự phòng (fallback) nếu mạng hoặc máy gặp sự cố. | - Video demo: `04_Video_Demo/demo_du_phong.mp4`<br>- Bộ dữ liệu test mẫu nhập tay trong buổi demo | Video nét, âm thanh/thuyết minh rõ ràng; kịch bản demo trực tiếp mượt mà, diễn ra dưới 3 phút. | Milestone 3 |

---

### 3.6. Ma trận phối hợp bắt buộc & Nghiệm thu chéo

Nhằm đảm bảo 100% thành viên đều hiểu sâu dự án và bảo vệ tốt phần vấn đáp cá nhân theo rubric, nhóm thực hiện các phiên làm việc chung sau:

| STT | Hoạt động phối hợp | Thành viên chủ trì | Thành viên tham gia | Kết quả cần đạt (Checklist nghiệm thu) |
|:---:|---|---|---|---|
| 1 | **Code Review & Leakage Check** | Minh | An, Thành, Nguyên | [ ] Duyệt sạch code không có data leakage; xác nhận tập test được cô lập 100%. |
| 2 | **Chốt mô hình & Metric đánh đổi** | Nguyên | Thành, Minh | [ ] Thống nhất bảng metric cuối cùng và ngưỡng quyết định phân loại tối ưu. |
| 3 | **Kiểm thử Artefact `predict.py`** | Vũ | Minh, An | [ ] Chạy thử `predict.py` trên 3 môi trường máy tính khác nhau thành công. |
| 4 | **Tổng duyệt thuyết trình & Vấn đáp thử (Mock Q&A)** | Minh | Cả 5 thành viên | [ ] Từng thành viên trả lời trôi chảy các câu hỏi về: vai trò của mình, lý do chọn thuật toán, cách tiền xử lý, ý nghĩa metric và hạn chế công nghiệp. |
| 5 | **Ký duyệt biên bản đóng góp** | Cả 5 thành viên | Cả 5 thành viên | [ ] Ký xác nhận tỷ lệ đóng góp thực tế trên trang bìa/phụ lục của báo cáo PDF nộp giảng viên. |

## 4. Mốc tiến độ theo đề bài

| Mốc | Thời điểm theo PDF | Công việc và đầu ra |
|---|---|---|
| **Milestone 0** | Cuối Buổi 2 | Đăng ký nhóm, thông tin trưởng nhóm/thành viên, đề tài, nguồn dữ liệu và mục tiêu; xác nhận giảng viên duyệt đề tài tự đề xuất. |
| **Milestone 1** | Cuối Buổi 4 | Báo cáo tiến độ 01: sơ đồ EDA, chiến lược tiền xử lý và notebook có baseline đầu tiên. |
| **Milestone 2** | Trước Buổi 7 (24 giờ) | Nộp đầy đủ báo cáo PDF, slide PDF, source code, dữ liệu/mô tả dữ liệu và file `.joblib` theo yêu cầu LMS. |
| **Milestone 3** | Buổi 7 và Buổi 8 | Thuyết trình trực tiếp; live demo và vấn đáp cá nhân. |

## 5. Sản phẩm phải nộp

PDF hướng dẫn yêu cầu cấu trúc gói nộp như sau (thay `[Tên_Nhóm]`, `[Mã_Lớp]`, `[Tên_Đề_Tài]` bằng thông tin thật):

```text
[Tên_Nhóm]_[Mã_Lớp]_[Tên_Đề_Tài]/
├── 01_Report/
│   └── Bao_cao_BTL_[Tên_Nhóm].pdf
├── 02_Slide/
│   └── Slide_Thuyet_trinh_[Tên_Nhóm].pdf
├── 03_SourceCode/
│   ├── data/
│   │   └── README.md                 # nguồn, cách tải, mô tả dữ liệu
│   ├── notebooks/
│   │   ├── 01_EDA_Preprocessing.ipynb
│   │   └── 02_Modeling_Evaluation.ipynb
│   ├── models/
│   │   └── best_model.joblib
│   ├── predict.py
│   ├── requirements.txt
│   └── README.md
└── 04_Video_Demo/                    # khuyến khích: video demo khoảng 3 phút
```

PDF nêu báo cáo kỹ thuật **15–25 trang A4**, Times New Roman cỡ 13, giãn dòng 1.3–1.5; slide thuyết trình **12–15 trang**. Báo cáo có bìa và bảng phân công (họ tên, mã sinh viên, vai trò, phần trăm đóng góp, chữ ký); 5 chương: (1) giới thiệu bài toán và mục tiêu; (2) EDA và tiền xử lý; (3) cơ sở thuật toán và thiết kế pipeline; (4) kết quả và đánh giá mô hình; (5) kết luận, hạn chế, hướng phát triển; cuối cùng là tài liệu tham khảo.

## 6. Gợi ý cấu trúc báo cáo

1. **Giới thiệu:** bối cảnh bảo trì dự đoán, bài toán, câu hỏi/mục tiêu, mô tả X/y, phạm vi và đóng góp.
2. **Dữ liệu và EDA:** nguồn, từ điển biến, chất lượng dữ liệu, mất cân bằng, biểu đồ và quyết định tiền xử lý.
3. **Thuật toán và pipeline:** lý do chọn ≥3 mô hình, sơ đồ pipeline, chia dữ liệu, cross-validation, không rò rỉ, không gian tìm siêu tham số.
4. **Thực nghiệm:** môi trường/thư viện/seed, bảng CV và test, metric, confusion matrix/ROC/PR, phân tích lỗi, mô hình tốt nhất và giải thích.
5. **Kết luận:** trả lời câu hỏi nghiên cứu, hạn chế, khả năng ứng dụng có điều kiện và hướng phát triển.
6. **Tài liệu tham khảo:** dataset UCI và tài liệu thuật toán/thư viện; trích dẫn nhất quán.

## 7. Đối chiếu rubric

| Tiêu chí | Trọng số | Việc nhóm cần chứng minh |
|---|---:|---|
| Khám phá và tiền xử lý dữ liệu | 20% | EDA có chiều sâu; giải thích missing/outlier/encoding/scaling; tạo đặc trưng khi có cơ sở. |
| Pipeline và thử nghiệm mô hình | 25% | Thử ít nhất 3 mô hình; dùng `Pipeline`/`ColumnTransformer`; quy trình tái lập và so sánh công bằng. |
| Cross-validation và tinh chỉnh | 20% | Stratified K-Fold đúng cách; `GridSearchCV`/`RandomizedSearchCV`; giữ test độc lập. |
| Báo cáo và mã nguồn | 15% | Báo cáo đủ cấu trúc, nguồn tin cậy; code sạch, có `requirements.txt`, `predict.py`, `.joblib`, README hướng dẫn. |
| Thuyết trình, live demo, vấn đáp | 20% | Đúng giờ, slide súc tích, demo trơn tru; từng thành viên trả lời được câu hỏi về code và quyết định nhóm. |

**Các lỗi cần tránh theo PDF:** fit scaler/imputer trên toàn bộ dữ liệu trước khi chia hoặc để lộ test trong cross-validation có thể bị trừ trực tiếp 1–2 điểm. Sao chép project mà không hiểu code có thể khiến cả nhóm nhận 0. Sinh viên không nắm được mã nguồn nhóm nộp có thể bị đánh giá cá nhân riêng. Tỷ lệ đóng góp cần phản ánh công việc thật.

## 8. Chạy dự án

Sau khi hoàn thiện source code, cập nhật đúng phiên bản thư viện trong `requirements.txt`, đảm bảo các đường dẫn dữ liệu tương đối và notebook chạy từ đầu đến cuối. Hướng dẫn dưới đây là mẫu, điều chỉnh theo môi trường của nhóm:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Chạy lần lượt hai notebook theo thứ tự. Sau khi xuất pipeline, thử script dự đoán với một dòng đầu vào mẫu và ghi kết quả kỳ vọng trong README mã nguồn. Không commit file môi trường, cache, dữ liệu cá nhân hoặc file lớn không cần thiết; nếu LMS yêu cầu dữ liệu thì kèm dữ liệu gốc theo đúng quy định lớp.

## 9. Checklist trước khi nộp

- [ ] Đề tài và dataset đã đăng ký/được duyệt; tên nhóm, lớp, thành viên chính xác.
- [ ] EDA và quyết định tiền xử lý được giải thích; có phân tích phân phối nhãn.
- [ ] Có ít nhất 3 mô hình khác nhau và baseline; lý do lựa chọn được nêu.
- [ ] Imputer/encoder/scaler nằm trong pipeline và chỉ fit trong fold huấn luyện.
- [ ] Có Stratified K-Fold và tinh chỉnh; test set độc lập, chỉ dùng đánh giá cuối.
- [ ] Báo cáo nhiều metric phù hợp mất cân bằng, phân tích lỗi và hạn chế.
- [ ] Lưu pipeline hoàn chỉnh vào `.joblib`; `predict.py` chạy với dữ liệu mới.
- [ ] `requirements.txt`, README cài/chạy và hai notebook được kiểm tra trên máy sạch.
- [ ] Báo cáo PDF đúng quy cách, có bảng phân công/chữ ký và tài liệu tham khảo.
- [ ] Slide PDF đủ 12–15 trang; toàn bộ source, artefact và sản phẩm được đặt đúng thư mục.
- [ ] Mỗi thành viên hiểu pipeline, phần việc của mình và đã tập demo/vấn đáp.
