"""EDA for playground-series-s3e3 (Employee Attrition, binary classification, ROC-AUC).

Prints findings to stdout. No plots (headless weekend run).
"""
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

COMP = "competitions/playground-series-s3e3"
DATA = f"{COMP}/data"
TARGET, ID = "Attrition", "id"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")


def section(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")


section("DATA OVERVIEW")
print(f"Train shape: {train.shape}  Test shape: {test.shape}")
cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in train.columns if c not in cat_cols + [ID, TARGET]]
print(f"Categorical ({len(cat_cols)}): {cat_cols}")
print(f"Numerical ({len(num_cols)}): {num_cols}")

section("TARGET ANALYSIS")
print(train[TARGET].value_counts())
print(train[TARGET].value_counts(normalize=True))
print(f"Imbalance ratio (neg/pos): {(train[TARGET]==0).sum() / max((train[TARGET]==1).sum(),1):.3f}")

section("MISSING VALUES")
print("Train missing total:", train.isnull().sum().sum())
print("Test missing total:", test.isnull().sum().sum())

section("DUPLICATES")
print("Full-row dup count (train, excl id):", train.drop(columns=[ID]).duplicated().sum())

section("CONSTANT / ZERO-VARIANCE COLUMNS")
for c in num_cols:
    if train[c].nunique() <= 1:
        print(f"  CONSTANT: {c} = {train[c].unique()}")

section("CATEGORICAL CARDINALITY & TRAIN/TEST CONSISTENCY")
for c in cat_cols:
    unseen = set(test[c].unique()) - set(train[c].unique())
    print(f"  {c}: nunique={train[c].nunique()} unseen_in_test={unseen}")
    grp = train.groupby(c)[TARGET].mean().sort_values(ascending=False)
    print(f"    attrition rate by category:\n{grp.to_string()}")

section("NUMERICAL CORRELATION WITH TARGET (point-biserial via pearson)")
corr = train[num_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(corr.to_string())

section("HIGH PAIRWISE CORRELATION AMONG NUMERIC FEATURES (|r|>0.7)")
c = train[num_cols].corr().abs()
pairs = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack()
pairs = pairs[pairs > 0.7].sort_values(ascending=False)
print(pairs.to_string() if len(pairs) else "  none")

section("TRAIN vs TEST NUMERIC DISTRIBUTION SHIFT (mean % diff)")
for c in num_cols:
    tm, em = train[c].mean(), test[c].mean()
    if tm != 0:
        diff_pct = 100 * (em - tm) / abs(tm)
        if abs(diff_pct) > 5:
            print(f"  {c}: train_mean={tm:.2f} test_mean={em:.2f} diff%={diff_pct:.2f}")
print("(only >5% diffs shown; none printed = all stable)")

section("LEAKAGE CHECK: near-perfect single-feature AUC")
from sklearn.metrics import roc_auc_score
for c in num_cols:
    try:
        auc = roc_auc_score(train[TARGET], train[c])
        auc = max(auc, 1 - auc)
        if auc > 0.75:
            print(f"  {c}: single-feature AUC={auc:.4f}")
    except Exception:
        pass
print("(only AUC>0.75 shown)")

section("VALIDATION STRATEGY RECOMMENDATION")
print("StratifiedKFold(5, shuffle=True, seed=42) on Attrition -- imbalanced binary target")
print("(11.9% positive rate), small n=1677 rows -> stratification needed for stable AUC folds.")

section("SUMMARY")
print("""
Key findings:
1. Train 1677 rows / 34 cols (33 features + target), test 1119 rows, no id target col in test.
2. Target imbalanced: ~88.1% class 0 / 11.9% class 1 (attrition). Stratified CV required.
3. No missing values, no duplicate rows.
4. EmployeeCount, StandardHours, Over18 are constant columns (zero variance) -> drop.
5. 8 categorical columns (BusinessTravel, Department, EducationField, Gender, JobRole,
   MaritalStatus, Over18[const], OverTime); no unseen categories in test.
6. OverTime and MaritalStatus show large attrition-rate spread across categories (strong signal).
7. Top numeric correlations with target: StockOptionLevel, Age, JobInvolvement,
   TotalWorkingYears, JobLevel, YearsInCurrentRole, YearsAtCompany, MonthlyIncome,
   YearsWithCurrManager (all negative -- longer tenure / higher level = less attrition).
8. Tenure features (YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager,
   TotalWorkingYears) mutually highly correlated -- candidate for ratio/interaction features
   rather than raw duplicates.
9. No single-feature leakage (>0.75 AUC) detected.
10. Small dataset (252KB) -> fast training, room for feature engineering + blend search.

Feature engineering ideas:
- Frequency/target-encode categoricals (OverTime, JobRole, MaritalStatus, Department).
- Tenure ratios: YearsInCurrentRole/YearsAtCompany, YearsWithCurrManager/YearsAtCompany.
- Income features: MonthlyIncome/JobLevel, MonthlyIncome per YearsAtCompany.
- Satisfaction composite: mean of EnvironmentSatisfaction/JobSatisfaction/
  RelationshipSatisfaction/WorkLifeBalance/JobInvolvement.
- Drop constant columns (EmployeeCount, StandardHours, Over18).
- OverTime x JobLevel interaction (overtime hurts more at lower levels, domain knowledge).

Recommended validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
""")
