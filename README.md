# Vietnamese Claim Verification - Hard Reasoning Dataset

Dự án nghiên cứu và xây dựng bộ dữ liệu suy luận khó (Hard Reasoning) cho bài toán kiểm chứng thông tin (Claim Verification) bằng tiếng Việt.

## 📌 Tổng quan dự án
Dự án tập trung vào việc bộc lộ những hạn chế của các mô hình học máy truyền thống trước các bẫy logic và thiên kiến trùng lặp từ vựng (Lexical Overlap Bias). 

Nội dung chính:
- Xây dựng bộ dữ liệu **Hard Reasoning** với các bẫy: Paraphrase, Đảo ngược logic, và Suy luận định lượng.
- Đánh giá hiệu năng các mô hình Baseline: **TF-IDF + Logistic Regression/SVM**.
- Thử nghiệm mô hình học sâu: **PhoBERT**.

## 🛠 Công cụ & Thư viện sử dụng
- **Ngôn ngữ:** Python 3.x
- **Thư viện NLP:** `underthesea` (tách từ tiếng Việt).
- **Machine Learning:** `scikit-learn` (TF-IDF, SVM, Logistic Regression).
- **Deep Learning:** `transformers`, `torch` (Fine-tuning PhoBERT).
- **Xử lý dữ liệu:** `pandas`, `openpyxl`.
