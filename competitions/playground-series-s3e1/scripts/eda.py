"""
EDA — playground-series-s3e1 (California Housing, predict MedHouseVal)
Prints all findings to stdout.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/playground-series-s3e1"
TARGET_COL = "MedHouseVal"
ID_COL = "id"

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}\n")


# --- Overview ---
section("DATA OVERVIEW")
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nDtypes:\n{train.dtypes}")

num_cols = [c for c in train.columns if c not in (TARGET_COL, ID_COL)]
print(f"\nFeatures ({len(num_cols)}): {num_cols}")

# --- Missing / duplicates ---
section("MISSING VALUES & DUPLICATES")
print("Train missing:\n", train.isnull().sum())
print("Test missing:\n", test.isnull().sum())
print(f"\nDuplicate rows in train (excluding id): {train.drop(columns=[ID_COL]).duplicated().sum()}")
print(f"Duplicate rows in test (excluding id): {test.drop(columns=[ID_COL]).duplicated().sum()}")
# duplicate feature rows vs original UCI dataset check: any train rows identical to test rows (excl target/id)?
common_cols = [c for c in num_cols]
dup_with_test = train[common_cols].merge(test[common_cols].drop_duplicates(), how="inner")
print(f"Train rows with feature-identical match in test: {len(dup_with_test)}")

# --- Target analysis ---
section("TARGET ANALYSIS: MedHouseVal")
print(train[TARGET_COL].describe())
print(f"Skewness: {train[TARGET_COL].skew():.4f}")
print(f"Kurtosis: {train[TARGET_COL].kurtosis():.4f}")
for p in [1, 5, 25, 50, 75, 95, 99, 99.5]:
    print(f"  {p}th pct: {np.percentile(train[TARGET_COL], p):.4f}")
print(f"\nValue == 5.00001 (top-coded cap) count: {(train[TARGET_COL] >= 5.00001).sum()} "
      f"({(train[TARGET_COL] >= 5.00001).mean()*100:.2f}%)")
print(f"Log1p skew: {np.log1p(train[TARGET_COL]).skew():.4f}")

# --- Feature summary stats ---
section("FEATURE SUMMARY STATS (train)")
print(train[num_cols].describe().T)

section("FEATURE SUMMARY STATS (test)")
print(test[num_cols].describe().T)

# --- Outlier / suspicious value checks (known California Housing quirks) ---
section("KNOWN-QUIRK CHECKS")
print(f"AveRooms > 20: {(train['AveRooms'] > 20).sum()} train rows, max={train['AveRooms'].max():.2f}")
print(f"AveBedrms > 5: {(train['AveBedrms'] > 5).sum()} train rows, max={train['AveBedrms'].max():.2f}")
print(f"AveOccup > 20: {(train['AveOccup'] > 20).sum()} train rows, max={train['AveOccup'].max():.2f}")
print(f"Population == 0: {(train['Population'] == 0).sum()}")
print(f"HouseAge >= 52 (top-coded) count: {(train['HouseAge'] >= 52).sum()} "
      f"({(train['HouseAge'] >= 52).mean()*100:.2f}%)")

# --- Correlations ---
section("CORRELATION WITH TARGET (Pearson)")
corr = train[num_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
print(corr)

section("FEATURE-FEATURE CORRELATION (|r| > 0.7)")
fc = train[num_cols].corr()
pairs = []
for i, a in enumerate(num_cols):
    for b in num_cols[i+1:]:
        r = fc.loc[a, b]
        if abs(r) > 0.7:
            pairs.append((a, b, round(r, 4)))
for p in pairs:
    print(p)

# --- Geo sanity: Lat/Long ---
section("GEO RANGE CHECK")
print(f"Train Latitude range: {train['Latitude'].min()} - {train['Latitude'].max()}")
print(f"Train Longitude range: {train['Longitude'].min()} - {train['Longitude'].max()}")
print(f"Test Latitude range: {test['Latitude'].min()} - {test['Latitude'].max()}")
print(f"Test Longitude range: {test['Longitude'].min()} - {test['Longitude'].max()}")

# --- Train/test distribution shift (simple mean/std comparison) ---
section("TRAIN vs TEST DISTRIBUTION (mean % diff)")
for c in num_cols:
    tm, te = train[c].mean(), test[c].mean()
    pct = abs(tm - te) / (abs(tm) + 1e-9) * 100
    print(f"{c:12s} train_mean={tm:10.4f}  test_mean={te:10.4f}  diff%={pct:6.2f}")

# --- Quick LightGBM importance for validation-strategy insight ---
section("QUICK LGB IMPORTANCE (5-fold sanity check)")
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

X = train[num_cols]
y = train[TARGET_COL]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
importances = np.zeros(len(num_cols))
for tr_idx, va_idx in kf.split(X):
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               random_state=42, verbosity=-1)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof[va_idx] = model.predict(X.iloc[va_idx])
    importances += model.feature_importances_ / kf.n_splits

rmse = mean_squared_error(y, oof) ** 0.5
print(f"Quick 5-fold LGB (raw features, default-ish params) OOF RMSE: {rmse:.5f}")
imp_df = pd.Series(importances, index=num_cols).sort_values(ascending=False)
print("\nFeature importance:\n", imp_df)

print("\nDone.")
