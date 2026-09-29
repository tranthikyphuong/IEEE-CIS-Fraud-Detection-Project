# Resampling and Interpretable Machine Learning for Financial Fraud Detection

## Overview | Tổng quan

This project investigates the impact of resampling techniques on financial fraud detection using the **IEEE-CIS Fraud Detection dataset**.

Dự án nghiên cứu tác động của các phương pháp resampling đối với bài toán phát hiện gian lận tài chính trên **IEEE-CIS Fraud Detection dataset**.

The study compares three resampling settings — **No Resampling, Random Over-Sampling (ROS), and SMOTE** — across three model families:

Nghiên cứu so sánh ba thiết lập — **Không Resampling, Random Over-Sampling (ROS), và SMOTE** — trên ba nhóm mô hình:

- **LightGBM**
- **Feedforward Neural Network (FNN)**
- **K-Means**

The study focuses on two main questions:

Nghiên cứu tập trung vào hai câu hỏi chính:

- **RQ1:** Does resampling improve fraud detection performance across different models?
- **RQ1:** Resampling có cải thiện hiệu quả phát hiện gian lận trên các mô hình khác nhau hay không?

- **RQ2:** Does resampling change the features driving model predictions?
- **RQ2:** Resampling có làm thay đổi các đặc trưng mà mô hình dựa vào để đưa ra dự đoán hay không?

---

## Motivation | Động lực nghiên cứu

Fraud detection is a highly imbalanced classification problem, where fraudulent transactions represent only a small proportion of all transactions.

Phát hiện gian lận là một bài toán phân loại mất cân bằng nghiêm trọng, trong đó các giao dịch gian lận chỉ chiếm một tỷ lệ nhỏ trong tổng số giao dịch.

Previous studies have explored SMOTE and neural networks as approaches for handling class imbalance in credit card fraud detection.

Các nghiên cứu trước đây đã sử dụng SMOTE kết hợp với mạng nơ-ron như một hướng tiếp cận để xử lý mất cân bằng trong phát hiện gian lận thẻ tín dụng.

However, resampling does not necessarily improve performance for every model or dataset. This motivates a direct comparison of resampling strategies across different model families on the IEEE-CIS dataset.

Tuy nhiên, resampling không nhất thiết cải thiện hiệu quả đối với mọi mô hình hoặc mọi bộ dữ liệu. Điều này tạo động lực cho việc so sánh trực tiếp các phương pháp resampling trên nhiều nhóm mô hình khác nhau trên bộ dữ liệu IEEE-CIS.

---

## Research Questions | Câu hỏi nghiên cứu

### RQ1 — Resampling Effectiveness | Hiệu quả của Resampling

**Does resampling improve fraud detection performance across different models?**

**Resampling có cải thiện hiệu quả phát hiện gian lận trên các mô hình khác nhau hay không?**

We compare:

So sánh:

`None` vs. `ROS` vs. `SMOTE`

across:

`LightGBM` vs. `FNN` vs. `K-Means`

---

### RQ2 — Feature Reliance | Mức độ phụ thuộc vào đặc trưng

**Does resampling change the features driving model predictions?**

**Resampling có làm thay đổi các đặc trưng mà mô hình dựa vào để đưa ra dự đoán hay không?**

SHAP-based feature importance is used to compare the model's feature reliance across resampling conditions.

Phân tích mức độ quan trọng của đặc trưng bằng SHAP được sử dụng để so sánh sự phụ thuộc của mô hình vào các đặc trưng giữa các phương pháp resampling.

---

## Methodology | Phương pháp
