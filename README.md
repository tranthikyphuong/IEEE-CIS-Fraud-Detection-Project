# Resampling and Interpretable Machine Learning for Financial Fraud Detection

A comparative study of resampling techniques for financial fraud detection, evaluating **three model families** — LightGBM, Feedforward Neural Network (FNN), and K-Means — under three imbalance-handling settings: **No Resampling, Random Over-Sampling (ROS), and SMOTE** on the full IEEE-CIS Fraud Detection dataset (590,540 transactions, 3.50% fraud rate).

The study investigates whether resampling improves fraud detection performance and whether it changes the features that models rely on, beyond headline performance metrics. The evaluation uses a chronological 70/10/20 train/validation/test split, three random seeds, PR-AUC, F1, Recall, Precision, ROC-AUC, and SHAP-based feature importance analysis.

Built as a research project on financial fraud detection. Full methodology, experiments, and findings are documented in the accompanying research report.

---

## Problem

Financial fraud detection is a severely imbalanced classification problem. Only approximately **3.50%** of transactions in the IEEE-CIS dataset are fraudulent, making minority-class detection substantially more difficult than majority-class classification.

Accuracy alone can be misleading in this setting because a model can achieve high overall accuracy while failing to identify a meaningful proportion of fraudulent transactions. Therefore, this study focuses on **PR-AUC, F1-score, Recall, and Precision**, together with ROC-AUC and Accuracy.

Resampling techniques such as ROS and SMOTE are widely used to address class imbalance. However, their effectiveness may depend on the underlying model and feature representation. This study therefore examines not only whether resampling improves predictive performance, but also whether it changes the model's feature reliance.

---

## Research Questions

**RQ1.** Does resampling improve fraud detection performance across different model families?

**RQ2.** Does resampling change the features driving model predictions, beyond headline performance metrics?

---

## Methodology

The study follows a chronological experimental pipeline designed to reduce temporal information leakage and ensure that resampling is applied only to the training data.

### Data Preprocessing

The IEEE-CIS transaction and identity datasets are merged using `TransactionID`.

The preprocessing pipeline includes:

- Merging transaction and identity data
- Removing duplicate records
- Removing highly sparse features
- Engineering temporal features
- Engineering transaction-level features
- Creating aggregation features
- Encoding categorical variables
- Train-only imputation and standardisation

The final dataset contains **590,540 transactions**, with a fraud rate of approximately **3.50%**.

### Data Splitting

A chronological **70/10/20** split is used:

- **Training:** 70%
- **Validation:** 10%
- **Testing:** 20%

The data are ordered by `TransactionDT` before splitting to preserve the temporal structure of the dataset.

Preprocessing statistics and aggregation features are fitted using the training data only.

### Resampling

Three training conditions are compared:

- **None:** Original imbalanced training data
- **ROS:** Random Over-Sampling
- **SMOTE:** Synthetic Minority Over-sampling Technique

ROS and SMOTE use a **10% minority-class sampling target** and are applied **only to the training set**.

The validation and test sets remain untouched.

### Model Families

Three different model families are evaluated:

#### LightGBM

A gradient-boosted decision tree model for tabular fraud detection.

#### Feedforward Neural Network

A multilayer neural network with:

- 256-unit hidden layer
- 128-unit hidden layer
- 64-unit hidden layer
- Batch Normalization
- Dropout
- Adam optimisation
- Early stopping based on validation PR-AUC

#### K-Means

MiniBatch K-Means is used as an unsupervised baseline. Cluster-level fraud rates are estimated from the training data with smoothing and used to generate fraud scores.

### Experimental Design

The main experiment evaluates:

```text
3 Resampling Methods
        ×
3 Model Families
        ×
3 Random Seeds
