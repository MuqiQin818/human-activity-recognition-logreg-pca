# human-activity-recognition-logreg-pca
This project implements a complete machine learning pipeline for human activity recognition using time-series data from the AReM dataset. It includes feature engineering, logistic regression with feature selection, L1-penalized classification, and multiclass classification using both Naïve Bayes and Principal Component Analysis (PCA).
# Human Activity Recognition with Logistic Regression and PCA

This project implements a full machine learning pipeline for **human activity recognition** based on multivariate time-series data. It was completed as part of the EE559 Machine Learning course at the University of Southern California.

---

## 📌 Project Background

The goal of this project is to classify human activities using data collected from wireless sensors. The dataset, **AReM**, contains raw time-series signals across 6 channels for 7 types of human activities, such as sitting, walking, and bending. Each instance contains 480 time steps per channel.

We aim to build and compare several classification models — from **logistic regression with statistical features**, to **L1-penalized logistic regression**, to **Naïve Bayes classifiers and PCA-based models** — for both **binary** and **multiclass** classification tasks.

---

## 🧠 Methods Used

### 🔧 Feature Engineering
- Extracted time-domain features from each time series:
  - Min, Max, Mean, Median, Standard Deviation, Q1, Q3
- Explored statistical properties (variance, confidence intervals, distribution)

### 📊 Binary Classification
- Built logistic regression classifiers to distinguish `bending` vs `non-bending`
- Used both:
  - **Manual feature selection (based on p-values)**
  - **Recursive Feature Elimination (RFE)**
- Applied **Stratified 5-fold cross-validation** to select hyperparameters

### 🧪 L1-Regularized Logistic Regression
- Used L1 penalty (LASSO) for embedded feature selection
- Compared performance with manual p-value pruning

### 🧩 Multiclass Classification
- Implemented L1-penalized multinomial logistic regression and Naïve Bayes
- Evaluated models on **all 7 activity classes**
- Used **Principal Component Analysis (PCA)** for dimensionality reduction

---

## 🔍 Model Comparison & Evaluation

| Task                     | Best Model                        | Test Accuracy | Notes                         |
|--------------------------|------------------------------------|---------------|-------------------------------|
| Binary Classification    | Logistic Regression + RFE         | ~89%          | Balanced folds via stratified CV |
| Binary (with L1 penalty) | L1-Regularized Logistic Regression | ~87%          | Similar performance, simpler pipeline |
| Multiclass Classification| PCA + Naïve Bayes                 | ~75%          | Reduced dimensionality improved generalization |

Evaluation metrics:
- Accuracy, F1-score, Confusion Matrix
- ROC Curve & AUC (binary task)
- Cross-validation vs. holdout performance

---

## 📈 What I Learned

This project helped me deepen my understanding of:
- Building full ML pipelines from raw time series to final classifier
- Comparing feature selection strategies (statistical vs. embedded)
- The impact of regularization and dimensionality reduction on generalization
- Best practices for evaluating classifiers with imbalanced data
- Visualization techniques for feature distributions and classification boundaries

---

## 🗂️ File Structure

