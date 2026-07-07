# Playground Series S4E11 — Depression Prediction

**Date**: 2026-02-19

## Competition Info
- URL: https://www.kaggle.com/competitions/playground-series-s4e11
- Problem: Binary classification (predict depression)
- Metric: Accuracy
- Train: 140,700 rows, 20 columns | Test: 93,800 rows
- Target: `Depression` (0/1, 18.2% positive)

## Key EDA Findings
- Working Professional vs Student split is the #1 structural feature: Students (19.8%) have 58.6% depression rate vs Professionals (80.2%) at 8.2%
- Age strongly negatively correlated with target (r=-0.56) — younger (students) more depressed
- Structured missing values: Academic features null for professionals, work features null for students
- Suicidal thoughts highly predictive: Yes→31.8% vs No→4.9% depression
- Academic Pressure strongly correlated with depression among students (r=+0.48)
- Financial Stress correlates with depression in both groups
- Sleep Duration and Dietary Habits have noisy rare categories
- City (98 values) and Degree (115 values) have unseen categories in test
- No duplicates, no leakage detected

## Feature Engineering
- 28 features total
- Cleaned Sleep Duration (4 groups) and Dietary Habits (3 groups + other)
- Structured NaN filled with 0 (absent feature for professional/student)
- Binary encodings: is_student, suicidal_thoughts, family_history, is_male
- Age features: age_decade, is_young
- Interactions: pressure (academic+work), satisfaction, stress_x_pressure, suicidal_x_student, age_x_pressure
- Label-encoded high-cardinality categoricals (City, Profession, Degree)

## Experiment Results

| # | Model | CV Accuracy | Notes |
|---|-------|-------------|-------|
| 1 | Majority Class | 0.81829 | Predict 0 for all |
| 2 | LightGBM | 0.92988 | 28 features, 5-fold, is_unbalance |
| 2b | LightGBM (thresh=0.75) | 0.93738 | Optimized threshold |

## Submissions
- `lgbm_baseline_20260219_150528.csv` → Public LB: **0.94093**, Private LB: **0.93939**

## Model Details
- **Best model**: LightGBM (binary, is_unbalance=True, lr=0.05, num_leaves=63, ~1266 iterations)
- Threshold optimized to 0.75 (from default 0.5)
- Validation: Stratified 5-Fold CV
- Top features: Age (dominant), stress_x_pressure, age_decade, suicidal_thoughts, is_young

## Lessons Learned
- Age is the dominant feature — strongly tied to the Professional/Student split
- Threshold optimization from 0.5→0.75 gave +0.75% accuracy (significant for imbalanced target)
- Public LB (0.941) slightly exceeded CV (0.937) — good generalization
- Structured missing values handled by filling with 0 works well

## Potential Improvements
- Separate models for Professionals vs Students
- XGBoost / CatBoost ensemble
- More aggressive feature engineering for the student subgroup
- Hyperparameter tuning with Optuna
- Feature selection (remove low-importance features)
