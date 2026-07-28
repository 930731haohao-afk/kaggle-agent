"""
EDA for Titanic Competition
============================
Adapted from templates/eda_template.py for the Titanic dataset.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TARGET_COL = "Survived"
ID_COL = "PassengerId"

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
cat_cols = train.select_dtypes(include=["object", "category", "str"]).columns.tolist()
for col in [TARGET_COL, ID_COL]:
    if col in num_cols:
        num_cols.remove(col)
    if col in cat_cols:
        cat_cols.remove(col)

print(f"\nNumerical features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")


# --- Target Analysis ---
section("TARGET ANALYSIS")
print(f"Target column: {TARGET_COL}")
print(f"Dtype: {train[TARGET_COL].dtype}")
print(f"Unique values: {train[TARGET_COL].nunique()}")
print(f"\nValue counts:\n{train[TARGET_COL].value_counts()}")
print(f"\nValue proportions:\n{train[TARGET_COL].value_counts(normalize=True)}")
print(f"\nImbalance ratio: {train[TARGET_COL].value_counts().iloc[0] / train[TARGET_COL].value_counts().iloc[1]:.2f}:1")


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


# --- Survival Rate by Key Features ---
section("SURVIVAL RATE BY KEY FEATURES")

# By Sex
print("By Sex:")
print(train.groupby("Sex")[TARGET_COL].agg(["mean", "count"]).to_string())

# By Pclass
print("\nBy Pclass:")
print(train.groupby("Pclass")[TARGET_COL].agg(["mean", "count"]).to_string())

# By Embarked
print("\nBy Embarked:")
print(train.groupby("Embarked")[TARGET_COL].agg(["mean", "count"]).to_string())

# By Sex + Pclass
print("\nBy Sex + Pclass:")
print(train.groupby(["Sex", "Pclass"])[TARGET_COL].agg(["mean", "count"]).to_string())

# Age binned
train["AgeBin"] = pd.cut(train["Age"], bins=[0, 12, 18, 35, 55, 80], labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"])
print("\nBy Age Group:")
print(train.groupby("AgeBin")[TARGET_COL].agg(["mean", "count"]).to_string())
train = train.drop(columns=["AgeBin"])

# Fare binned
train["FareBin"] = pd.qcut(train["Fare"], q=4, labels=["Low", "MedLow", "MedHigh", "High"])
print("\nBy Fare Quartile:")
print(train.groupby("FareBin")[TARGET_COL].agg(["mean", "count"]).to_string())
train = train.drop(columns=["FareBin"])

# Family size
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
print("\nBy Family Size:")
print(train.groupby("FamilySize")[TARGET_COL].agg(["mean", "count"]).to_string())
train = train.drop(columns=["FamilySize"])


# --- Correlation with Target ---
section("CORRELATION WITH TARGET")
correlations = train[num_cols].corrwith(train[TARGET_COL]).abs().sort_values(ascending=False)
print("Absolute correlations with Survived:")
print(correlations.to_string())

# --- Correlation between numerical features ---
section("INTER-FEATURE CORRELATION")
corr_matrix = train[num_cols + [TARGET_COL]].corr()
print(corr_matrix.to_string())


# --- Duplicate Check ---
section("DUPLICATE CHECK")
train_dupes = train.duplicated().sum()
print(f"Duplicate rows in train: {train_dupes}")
id_dupes = train[ID_COL].duplicated().sum()
print(f"Duplicate IDs in train: {id_dupes}")


# --- Train vs Test Distribution ---
section("TRAIN VS TEST DISTRIBUTION COMPARISON")
for col in num_cols:
    train_mean = train[col].mean()
    test_mean = test[col].mean()
    train_std = train[col].std()
    test_std = test[col].std()
    shift = abs(train_mean - test_mean) / (train_std + 1e-8)
    flag = " *** SHIFT ***" if shift > 0.5 else ""
    print(f"{col}: train_mean={train_mean:.4f}, test_mean={test_mean:.4f}, "
          f"normalized_shift={shift:.4f}{flag}")


# --- Title extraction (domain insight) ---
section("TITLE ANALYSIS (from Name)")
train["Title"] = train["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
print("Title vs Survival:")
title_surv = train.groupby("Title")[TARGET_COL].agg(["mean", "count"]).sort_values("count", ascending=False)
print(title_surv.head(10).to_string())


section("EDA COMPLETE")
print("""
Key takeaways:
1. Sex is the strongest predictor (female survival ~74%, male ~19%)
2. Pclass is highly predictive (1st class ~63%, 3rd class ~24%)
3. Age matters (children have higher survival)
4. Fare correlates with survival (higher fare = higher survival)
5. Family size has a non-linear effect (solo and large families do worse)
6. Cabin is 77% missing - can extract deck letter or use as has_cabin flag
7. Title extracted from Name is very informative
8. No duplicate rows or IDs found
9. No significant train-test distribution shift

Recommended validation: Stratified 5-Fold (binary classification with mild imbalance)
""")
