# human-activity-recognition-logreg-pca
This project implements a complete machine learning pipeline for human activity recognition using time-series data from the AReM dataset. It includes feature engineering, logistic regression with feature selection, L1-penalized classification, and multiclass classification using both Naïve Bayes and Principal Component Analysis (PCA).
This is a machine learning project for classifying human activities based on multivariate time-series data from the AReM dataset. It was completed as part of the EE559 Machine Learning course at USC.


---

##  Dataset

- AReM Dataset: https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+(AReM)
- 6-channel time series, 480 time steps per instance, 7 activity classes

---

## Key Steps

- **Feature Engineering**: min, max, mean, median, std, Q1, Q3 from each time series
- **Binary Classification** (`bending` vs. others):
  - Logistic Regression + p-value filtering
  - Recursive Feature Elimination (RFE)
  - L1-Regularized Logistic Regression (LASSO)
- **Multiclass Classification**:
  - Naïve Bayes (Gaussian & Multinomial)
  - PCA + Logistic Regression

---

## Results Summary

| Task                     | Best Model                        | Accuracy (test) |
|--------------------------|------------------------------------|------------------|
| Binary Classification    | Logistic Regression + RFE         | ~89%             |
| L1 Logistic Regression   | LASSO                             | ~87%             |
| Multiclass Classification| PCA + Naïve Bayes                 | ~75%             |

Metrics used: accuracy, F1, ROC-AUC, confusion matrix, CV score

---

## Takeaways

- Feature selection (manual vs. regularized) affects performance and interpretability
- PCA helps when dealing with noisy or high-dimensional time-series data
- Stratified cross-validation is important for imbalanced classes
