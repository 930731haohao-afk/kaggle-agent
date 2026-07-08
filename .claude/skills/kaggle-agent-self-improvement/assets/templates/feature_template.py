"""
Feature Engineering Template for Kaggle Competitions
=====================================================
Usage: Modify the CONFIG section and the feature engineering functions below.
Saves processed train and test datasets.
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
OUTPUT_TRAIN = "train_processed.csv"
OUTPUT_TEST = "test_processed.csv"
# ============================================================

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, TRAIN_FILE))
test = pd.read_csv(os.path.join(data_dir, TEST_FILE))

# Separate target and ID before processing
y_train = train[TARGET_COL]
train_ids = train[ID_COL]
test_ids = test[ID_COL]

train = train.drop(columns=[TARGET_COL, ID_COL])
test = test.drop(columns=[ID_COL])

print(f"Train shape before: {train.shape}")
print(f"Test shape before:  {test.shape}")


# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# All functions take train and test DataFrames and return modified versions.
# Statistics must be computed on train only, then applied to both.
# ============================================================

def handle_missing_values(train: pd.DataFrame, test: pd.DataFrame):
    """Impute missing values. Fit on train, transform both."""
    num_cols = train.select_dtypes(include=[np.number]).columns
    cat_cols = train.select_dtypes(include=["object", "category"]).columns

    # Numerical: fill with median from train
    for col in num_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

    # Categorical: fill with mode from train
    for col in cat_cols:
        mode_val = train[col].mode()[0] if len(train[col].mode()) > 0 else "MISSING"
        train[col] = train[col].fillna(mode_val)
        test[col] = test[col].fillna(mode_val)

    return train, test


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame):
    """Encode categorical features. Fit on train, transform both."""
    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        # Frequency encoding (safe, no leakage)
        freq_map = train[col].value_counts(normalize=True).to_dict()
        train[col] = train[col].map(freq_map).fillna(0)
        test[col] = test[col].map(freq_map).fillna(0)

    return train, test


def create_features(train: pd.DataFrame, test: pd.DataFrame):
    """Create new features. Add your custom features here."""
    # Example: interaction features
    # train["feat_a_times_b"] = train["feat_a"] * train["feat_b"]
    # test["feat_a_times_b"] = test["feat_a"] * test["feat_b"]

    # Example: aggregation features
    # group_stats = train.groupby("group_col")["value_col"].agg(["mean", "std"]).reset_index()
    # group_stats.columns = ["group_col", "group_mean", "group_std"]
    # train = train.merge(group_stats, on="group_col", how="left")
    # test = test.merge(group_stats, on="group_col", how="left")

    return train, test


def remove_low_value_features(train: pd.DataFrame, test: pd.DataFrame):
    """Remove constant or near-constant features."""
    drop_cols = []
    for col in train.columns:
        if train[col].nunique() <= 1:
            drop_cols.append(col)

    if drop_cols:
        print(f"Dropping {len(drop_cols)} constant features: {drop_cols}")
        train = train.drop(columns=drop_cols)
        test = test.drop(columns=drop_cols)

    return train, test


# ============================================================
# EXECUTE PIPELINE
# ============================================================

train, test = handle_missing_values(train, test)
train, test = encode_categoricals(train, test)
train, test = create_features(train, test)
train, test = remove_low_value_features(train, test)


# ============================================================
# VALIDATION
# ============================================================

print(f"\nTrain shape after: {train.shape}")
print(f"Test shape after:  {test.shape}")

# Check columns match
train_cols = set(train.columns)
test_cols = set(test.columns)
if train_cols != test_cols:
    print(f"\nWARNING: Column mismatch!")
    print(f"  In train but not test: {train_cols - test_cols}")
    print(f"  In test but not train: {test_cols - train_cols}")
else:
    print(f"\nColumn check: PASS ({len(train_cols)} features)")

# Check for NaN
train_nans = train.isnull().sum().sum()
test_nans = test.isnull().sum().sum()
print(f"NaN check: train={train_nans}, test={test_nans}")

# Check dtypes
non_numeric = train.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"WARNING: Non-numeric columns remain: {non_numeric}")
else:
    print("Dtype check: PASS (all numeric)")


# ============================================================
# SAVE
# ============================================================

# Reattach ID and target
train[ID_COL] = train_ids.values
train[TARGET_COL] = y_train.values
test[ID_COL] = test_ids.values

output_dir = os.path.join(COMPETITION_DIR, "data")
train.to_csv(os.path.join(output_dir, OUTPUT_TRAIN), index=False)
test.to_csv(os.path.join(output_dir, OUTPUT_TEST), index=False)
print(f"\nSaved: {OUTPUT_TRAIN}, {OUTPUT_TEST}")
