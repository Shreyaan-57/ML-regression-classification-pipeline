# ML-regression-classification-pipeline

# End-to-End AI & Machine Learning Pipeline

This repository contains the complete codebase and project deliverables for the **Maincrafts Technology AI/ML Internship**. The project demonstrates an end-to-end data science lifecycle—from exploratory data analysis and baseline modeling to hyperparameter optimization and imbalanced binary classification.

---

## 📌 Project Overview

The project is structured into four core analytical tasks executed across regression and classification workflows:

### Task 1: Exploratory Data Analysis & Baseline Regression
* Conducted initial data analysis, missing value inspection, and correlation analysis on the California Housing dataset.
* Built an unscaled baseline Linear Regression model to establish benchmark performance metrics ($MAE$, $RMSE$, $R^2$).

### Task 2: Feature Scaling & Model Comparison
* Implemented `StandardScaler` to evaluate the impact of feature normalization.
* Compared performance metrics across multiple models: Linear Regression, Ridge Regression, and Decision Trees.

### Task 3: Overfitting Analysis, Cross-Validation & Hyperparameter Tuning
* Analyzed severe train vs. test performance gaps in unconstrained Decision Trees to detect overfitting.
* Applied 5-Fold Cross-Validation (`cross_val_score`) for reliable error estimation.
* Performed systematic hyperparameter optimization using `GridSearchCV` (`max_depth`, `min_samples_split`).

### Task 4: Imbalanced Classification & Model Evaluation
* Evaluated binary classification models on the Breast Cancer dataset using stratified train-test splits.
* Handled class imbalance using `class_weight="balanced"` to optimize sensitivity and recall in medical contexts.
* Analyzed performance via Confusion Matrices, Classification Reports, and ROC-AUC curves ($AUC = 0.9954$).

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Environment:** Jupyter Notebook / Anaconda
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn

---

## 📈 Key Results

| Workflow | Best Model | Primary Optimization / Metric |
| :--- | :--- | :--- |
| **Housing Regression** | Tuned Decision Tree (`max_depth: 10`) | Reduced test error via `GridSearchCV` |
| **Cancer Classification** | Balanced Logistic Regression | **0.9954 ROC-AUC Score** |

---

## 🚀 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/your-username/machine-learning-internship.git](https://github.com/your-username/machine-learning-internship.git)
