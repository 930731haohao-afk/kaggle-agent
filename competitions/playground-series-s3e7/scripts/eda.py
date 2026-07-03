"""
EDA — playground-series-s3e7 (Hotel Reservation Cancellation)
================================================================
Target: booking_status (binary, 1 = cancelled). Metric: ROC-AUC (maximize).
All categorical columns are already integer-encoded (meal plan, room type,
market segment). No missing values, no duplicate rows/ids.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/playground-series-s3e7"
TARGET_COL = "booking_status"
ID_COL = "id"

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}\n")


section("DATA OVERVIEW")
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

num_cols = [c for c in train.columns if c not in (TARGET_COL, ID_COL)]
print(f"\nFeature columns ({len(num_cols)}): {num_cols}")

section("TARGET ANALYSIS")
vc = train[TARGET_COL].value_counts()
vc_pct = train[TARGET_COL].value_counts(normalize=True)
print(f"Value counts:\n{vc.to_string()}")
print(f"\nProportions:\n{vc_pct.to_string()}")
print(f"\nImbalance ratio (majority/minority): {vc.max()/vc.min():.3f}")

section("MISSING VALUES")
print("Train missing total:", train.isnull().sum().sum())
print("Test missing total:", test.isnull().sum().sum())

section("DUPLICATE / ID CHECKS")
print("Duplicate rows (train):", train.duplicated().sum())
print("Duplicate ids (train):", train[ID_COL].duplicated().sum())
print("Duplicate ids (test):", test[ID_COL].duplicated().sum())
print("id overlap train/test:", len(set(train[ID_COL]) & set(test[ID_COL])))

section("NUMERICAL / ENCODED FEATURE STATS")
stats = train[num_cols].describe().T
stats["skew"] = train[num_cols].skew()
stats["nunique"] = train[num_cols].nunique()
print(stats.to_string())

section("CARDINALITY OF CATEGORICAL-LIKE (LOW-CARDINALITY INT) COLUMNS")
low_card = [c for c in num_cols if train[c].nunique() <= 10]
for c in low_card:
    print(f"\n--- {c} (train nunique={train[c].nunique()}) ---")
    print(train[c].value_counts().sort_index().to_string())
    unseen = set(test[c].unique()) - set(train[c].unique())
    print(f"  categories in test not in train: {unseen if unseen else 'none'}")

section("CORRELATION WITH TARGET (|pearson|)")
corr = train[num_cols].corrwith(train[TARGET_COL]).abs().sort_values(ascending=False)
print(corr.to_string())

section("TARGET RATE BY KEY CATEGORICALS")
for c in ["market_segment_type", "room_type_reserved", "type_of_meal_plan",
          "required_car_parking_space", "repeated_guest", "no_of_special_requests"]:
    g = train.groupby(c)[TARGET_COL].agg(["mean", "count"])
    print(f"\n--- cancellation rate by {c} ---")
    print(g.to_string())

section("LEAD TIME vs TARGET")
print(train.groupby(pd.cut(train["lead_time"], bins=[-1, 7, 30, 90, 180, 500]))[TARGET_COL]
      .agg(["mean", "count"]).to_string())

section("PRICE vs TARGET")
print(train.groupby(pd.qcut(train["avg_price_per_room"], 5, duplicates="drop"))[TARGET_COL]
      .agg(["mean", "count"]).to_string())

section("TRAIN VS TEST DISTRIBUTION SHIFT")
for col in num_cols:
    tr_m, te_m = train[col].mean(), test[col].mean()
    tr_s = train[col].std() + 1e-8
    shift = abs(tr_m - te_m) / tr_s
    flag = " *** SHIFT ***" if shift > 0.3 else ""
    print(f"{col}: train_mean={tr_m:.4f} test_mean={te_m:.4f} shift={shift:.4f}{flag}")

section("LEAKAGE CHECK — SINGLE-FEATURE AUC")
from sklearn.metrics import roc_auc_score
for c in num_cols:
    try:
        auc = roc_auc_score(train[TARGET_COL], train[c])
        auc = max(auc, 1 - auc)
        flag = " *** SUSPICIOUSLY HIGH ***" if auc > 0.85 else ""
        print(f"{c}: single-feature AUC={auc:.4f}{flag}")
    except Exception as e:
        print(f"{c}: skipped ({e})")

section("EDA COMPLETE")
