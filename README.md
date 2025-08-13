# Diabetes Risk Prediction — Automated Learning & Data Analysis

A production-minded, end-to-end machine learning pipeline to predict diabetes risk from routine clinical features.  
The project emphasizes **class-imbalance handling**, **cross-validated hyperparameter tuning**, and **clinician-friendly explainability**.

## What’s inside
- **Data pipeline:** cleaning (median imputation for implausible zeros), z-score scaling, and **SMOTE** applied safely within CV.
- **Models:** Logistic Regression, Random Forest, and SVM; combined via a **soft-voting ensemble**.
- **Evaluation focus:** minority (diabetic) class with F1/precision/recall, ROC/PR curves, and confusion matrices.
- **Explainability:** **SHAP** global and per-prediction attributions for transparent decision support.

## Results (test set, diabetic class)
- Single models F1 ≈ **0.71–0.77**;  
- **Ensemble:** **F1 ≈ 0.86**, **Precision ≈ 0.98**, **Recall ≈ 0.79**, overall accuracy ≈ **96%**.

## Tech stack
**Python**, **pandas**, **NumPy**, **scikit-learn** (Pipelines, CV, RandomizedSearchCV), **imbalanced-learn** (SMOTE), **SHAP**, **matplotlib**, **Jupyter**.
