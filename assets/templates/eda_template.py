"""
EDA Template for Kaggle Competitions
=====================================
Usage: Modify the CONFIG section below, then run the script.
Prints all findings to stdout for Claude Code to read and interpret.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — Modify these for each competition
# ============================================================
COMPETITION_DIR = "competitions/<name>"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TARGET_COL = "<target>"
ID_COL = "<id>"
# ============================================================

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, TRAIN_FILE))
test = pd.read_csv(os.path.join(data_dir, TEST_FILE))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# --- Data Overview ---
section("DATA OVERVIEW")
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Train columns: {list(train.columns)}")

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)
if ID_COL in num_cols:
    num_cols.remove(ID_COL)
if ID_COL in cat_cols:
    cat_cols.remove(ID_COL)

print(f"\nNumerical features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")


# --- Target Analysis ---
section("TARGET ANALYSIS")
print(f"Target column: {TARGET_COL}")
print(f"Dtype: {train[TARGET_COL].dtype}")
print(f"Unique values: {train[TARGET_COL].nunique()}")
print(f"\nDescribe:\n{train[TARGET_COL].describe()}")

if train[TARGET_COL].nunique() <= 20:
    print(f"\nValue counts:\n{train[TARGET_COL].value_counts()}")
    print(f"\nValue proportions:\n{train[TARGET_COL].value_counts(normalize=True)}")
else:
    print(f"\nSkewness: {train[TARGET_COL].skew():.4f}")
    print(f"Kurtosis: {train[TARGET_COL].kurtosis():.4f}")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(train[TARGET_COL].dropna(), p):.4f}")


# --- Missing Values ---
section("MISSING VALUES")
train_missing = train.isnull().sum()
train_missing_pct = (train_missing / len(train) * 100).round(2)
test_missing = test.isnull().sum()
test_missing_pct = (test_missing / len(test) * 100).round(2)

missing_df = pd.DataFrame({
    "train_count": train_missing,
    "train_pct": train_missing_pct,
    "test_count": test_missing,
    "test_pct": test_missing_pct
})
missing_df = missing_df[(missing_df["train_count"] > 0) | (missing_df["test_count"] > 0)]

if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("No missing values in train or test.")


# --- Numerical Feature Statistics ---
section("NUMERICAL FEATURE STATISTICS")
if num_cols:
    stats = train[num_cols].describe().T
    stats["skew"] = train[num_cols].skew()
    stats["kurtosis"] = train[num_cols].kurtosis()
    print(stats.to_string())
else:
    print("No numerical features.")


# --- Categorical Feature Statistics ---
section("CATEGORICAL FEATURE STATISTICS")
for col in cat_cols:
    n_unique = train[col].nunique()
    top_5 = train[col].value_counts().head(5)
    rare_count = (train[col].value_counts() < len(train) * 0.01).sum()
    test_unseen = set(test[col].dropna().unique()) - set(train[col].dropna().unique())

    print(f"\n--- {col} ---")
    print(f"  Unique values: {n_unique}")
    print(f"  Rare categories (<1%): {rare_count}")
    print(f"  Unseen in test: {len(test_unseen)}")
    print(f"  Top 5:\n{top_5.to_string()}")


# --- Correlation with Target ---
section("CORRELATION WITH TARGET")
if num_cols and train[TARGET_COL].dtype in [np.float64, np.int64, np.float32, np.int32]:
    correlations = train[num_cols].corrwith(train[TARGET_COL]).abs().sort_values(ascending=False)
    print("Top correlations with target:")
    print(correlations.head(20).to_string())


# --- Duplicate Check ---
section("DUPLICATE CHECK")
train_dupes = train.duplicated().sum()
print(f"Duplicate rows in train: {train_dupes}")
if ID_COL in train.columns:
    id_dupes = train[ID_COL].duplicated().sum()
    print(f"Duplicate IDs in train: {id_dupes}")


# --- Train vs Test Distribution ---
section("TRAIN VS TEST DISTRIBUTION COMPARISON")
for col in num_cols[:10]:  # Limit to first 10 for brevity
    train_mean = train[col].mean()
    test_mean = test[col].mean()
    train_std = train[col].std()
    test_std = test[col].std()
    shift = abs(train_mean - test_mean) / (train_std + 1e-8)
    flag = " *** SHIFT ***" if shift > 0.5 else ""
    print(f"{col}: train_mean={train_mean:.4f}, test_mean={test_mean:.4f}, "
          f"normalized_shift={shift:.4f}{flag}")


section("EDA COMPLETE")
print("Review the findings above and proceed to feature engineering.")
