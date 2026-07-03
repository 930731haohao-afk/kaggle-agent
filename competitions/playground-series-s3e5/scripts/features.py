"""Feature engineering — playground-series-s3e5 (wine quality).

Adds a modest set of domain-informed interaction/ratio features on top of the
11 raw physicochemical columns, guided by EDA (scripts/eda.py):
  - alcohol, sulphates, volatile acidity, citric acid are the strongest
    |Spearman r| predictors of quality -> build ratios/interactions around them.
  - total/free sulfur dioxide ratio (bound-SO2 fraction), acidity ratios,
    density*alcohol interaction (both driven by sugar/ethanol content).
All statistics used below (none, currently — pure row-wise arithmetic) would
be fit on train only; there are no fitted encoders here so no leakage risk.

Run: uv run python3 competitions/playground-series-s3e5/scripts/features.py
Writes: data/train_processed.csv, data/test_processed.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
TARGET, IDC = "quality", "Id"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    # bound vs free SO2
    df["free_so2_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + eps)
    df["bound_so2"] = df["total sulfur dioxide"] - df["free sulfur dioxide"]
    # acidity relationships
    df["fixed_to_volatile_acid"] = df["fixed acidity"] / (df["volatile acidity"] + eps)
    df["citric_to_volatile_acid"] = df["citric acid"] / (df["volatile acidity"] + eps)
    df["total_acid"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    # alcohol / sulphates — top correlated features, both positively assoc. with quality
    df["alcohol_x_sulphates"] = df["alcohol"] * df["sulphates"]
    df["alcohol_to_density"] = df["alcohol"] / (df["density"] + eps)
    df["alcohol_x_density"] = df["alcohol"] * df["density"]
    # sugar/alcohol proxy for fermentation completeness
    df["sugar_to_alcohol"] = df["residual sugar"] / (df["alcohol"] + eps)
    # pH vs acid consistency
    df["acid_ph_ratio"] = df["total_acid"] / (df["pH"] + eps)
    return df


train_fe = engineer(train)
test_fe = engineer(test)

feat_cols = [c for c in train_fe.columns if c not in (IDC, TARGET)]
print(f"Original features: {train.shape[1] - 2}  ->  Engineered total: {len(feat_cols)}")
print(f"New features added: {[c for c in feat_cols if c not in train.columns]}")

# Validation checks
assert list(train_fe.drop(columns=[TARGET]).columns) == list(test_fe.columns), "column mismatch train/test"
n_nan_tr = train_fe[feat_cols].isnull().sum().sum()
n_nan_te = test_fe[feat_cols].isnull().sum().sum()
print(f"NaNs introduced -- train: {n_nan_tr}  test: {n_nan_te}")
assert n_nan_tr == 0 and n_nan_te == 0, "unexpected NaNs after feature engineering"
assert all(pd.api.types.is_numeric_dtype(train_fe[c]) for c in feat_cols), "non-numeric feature found"

corr = train_fe[feat_cols + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\nSpearman |r| with quality, top 10 (post feature-engineering):")
print(corr.head(10).round(3))

train_fe.to_csv(os.path.join(DATA, "train_processed.csv"), index=False)
test_fe.to_csv(os.path.join(DATA, "test_processed.csv"), index=False)
print(f"\nWrote train_processed.csv {train_fe.shape}, test_processed.csv {test_fe.shape}")

# Quick feature importance check (LightGBM, default params) to sanity check the new features
import lightgbm as lgb
X = train_fe[feat_cols].to_numpy(np.float32)
y = train_fe[TARGET].to_numpy()
m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
m.fit(X, y)
imp = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\nQuick LightGBM feature importance (top 15):")
print(imp.head(15))
n_new_in_top10 = sum(1 for c in imp.head(10).index if c not in train.columns)
print(f"\nEngineered features in top 10 by importance: {n_new_in_top10}/10")
