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
if TARGET_COL in cat_cols:
    # 2026-08-03 audit: only num_cols used to be filtered. An object/string target — i.e.
    # every classification competition with text labels — stayed in cat_cols, and the
    # categorical loop below indexes test[col]; test has no target column, so the script
    # died with KeyError before correlation and duplicates were ever printed.
    cat_cols.remove(TARGET_COL)
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
# This section is the ONLY quantitative evidence behind the leak check in
# references/02_eda.md ("features too perfectly correlated with the target"), so it must
# never fail silently. 2026-08-03 audit: the guard used to be an exact dtype whitelist
# [float64, int64, float32, int32], which dropped bool targets, narrow int dtypes and every
# string-labelled target — printing an empty section that reads exactly like "no leak found".
target_numeric = None
if not num_cols:
    print("NOT COMPUTED: no numerical features to correlate against the target.")
elif pd.api.types.is_bool_dtype(train[TARGET_COL]) or pd.api.types.is_numeric_dtype(train[TARGET_COL]):
    target_numeric = train[TARGET_COL].astype(float)
elif train[TARGET_COL].nunique(dropna=True) == 2:
    # Binary non-numeric label ("yes"/"no", "Presence"/"Absence"): the point-biserial
    # correlation on the 0/1 encoding is still the leak signal we need, so encode it.
    codes, levels = pd.factorize(train[TARGET_COL], use_na_sentinel=True)
    target_numeric = pd.Series(codes, index=train.index).where(codes >= 0).astype(float)
    print(f"NOTE: target dtype is {train[TARGET_COL].dtype}; encoded 1='{levels[1]}', "
          f"0='{levels[0]}' — values below are point-biserial correlations.")
else:
    print(f"NOT COMPUTED: target dtype {train[TARGET_COL].dtype} with "
          f"{train[TARGET_COL].nunique()} classes has no linear order, so Pearson r is "
          f"undefined. Run the leak check with per-class target rates or mutual information.")

if target_numeric is not None:
    correlations = train[num_cols].corrwith(target_numeric).abs().sort_values(ascending=False)
    print("Top correlations with target (|r|, descending):")
    print(correlations.head(20).to_string())


# --- Duplicate Check ---
section("DUPLICATE CHECK")
# 2026-08-03 audit: this used to be train.duplicated() over the FULL frame, id column
# included — with a unique id every row is distinct, so the check returned 0 on every
# competition that has one and could never fire. Duplicates are a property of the features.
# The two counts below answer different questions and both matter:
#   feature rows      -> identical features; if their targets differ this is label noise,
#                        it caps achievable accuracy (Bayes floor) and such rows must not
#                        straddle CV folds
#   rows incl. target -> fully identical records, i.e. true duplicates that can be dropped
#                        or down-weighted
feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
dupe_features = int(train.duplicated(subset=feature_cols).sum()) if feature_cols else 0
dupe_with_target = int(train.duplicated(subset=feature_cols + [TARGET_COL]).sum()) if feature_cols else 0
print(f"Duplicate feature rows in train (id/target excluded): {dupe_features}")
print(f"Duplicate rows incl. target: {dupe_with_target}")
if dupe_features > dupe_with_target:
    print("  -> identical features carry DIFFERENT targets: label noise / Bayes-error floor; "
          "keep identical feature rows inside the same fold.")
elif dupe_features > 0:
    print("  -> fully identical records: true duplicates; consider dropping or row weights.")
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
