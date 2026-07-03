"""EDA for playground-series-s3e9 (Concrete Compressive Strength, RMSE regression).

Prints all findings to stdout so the agent can interpret them.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

DATA = "competitions/playground-series-s3e9/data"
TARGET = "Strength"
ID = "id"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

feats = [c for c in train.columns if c not in (ID, TARGET)]
num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
cat_feats = [c for c in feats if not pd.api.types.is_numeric_dtype(train[c])]

print("=" * 70)
print("1. DATA OVERVIEW")
print("=" * 70)
print(f"Train shape: {train.shape}   Test shape: {test.shape}")
print(f"Features ({len(feats)}): {feats}")
print(f"Numeric ({len(num_feats)}): {num_feats}")
print(f"Categorical ({len(cat_feats)}): {cat_feats}")
print(f"Missing in train:\n{train.isna().sum()[train.isna().sum() > 0]}")
print(f"Missing in test:\n{test.isna().sum()[test.isna().sum() > 0]}")
print(f"Duplicate rows in train (excl id): {train[feats + [TARGET]].duplicated().sum()}")
print(f"Duplicate feature-rows in train (excl target): {train[feats].duplicated().sum()}")

print("\n" + "=" * 70)
print("2. TARGET ANALYSIS: Strength")
print("=" * 70)
y = train[TARGET]
print(y.describe())
print(f"skew={y.skew():.3f}  kurtosis={y.kurtosis():.3f}")
for p in [0.1, 1, 5, 50, 95, 99, 99.9]:
    print(f"  pct {p:5}: {np.percentile(y, p):.3f}")
print(f"log1p(Strength) skew={np.log1p(y).skew():.3f}")

print("\n" + "=" * 70)
print("3. NUMERIC FEATURE SUMMARY")
print("=" * 70)
print(train[num_feats].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])
print("\nskew / kurtosis / zeros:")
for c in num_feats:
    print(f"  {c:28} skew={train[c].skew():7.3f}  kurt={train[c].kurtosis():8.3f}  "
          f"zeros={int((train[c] == 0).sum())} ({(train[c]==0).mean()*100:.1f}%)  min={train[c].min():.3f}")

print("\n" + "=" * 70)
print("4. CORRELATION WITH TARGET (Pearson & Spearman)")
print("=" * 70)
corr_p = train[num_feats + [TARGET]].corr(method="pearson")[TARGET].drop(TARGET)
corr_s = train[num_feats + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
print(pd.DataFrame({"pearson": corr_p, "spearman": corr_s}).sort_values("spearman", key=abs, ascending=False))

print("\n" + "=" * 70)
print("5. FEATURE-FEATURE CORRELATION (collinearity check)")
print("=" * 70)
print(train[num_feats].corr().round(3))

print("\n" + "=" * 70)
print("6. AgeInDays DISTRIBUTION (curing time, known nonlinear w/ Strength)")
print("=" * 70)
print(f"unique AgeInDays values: {sorted(train['AgeInDays'].unique())}")
print(train.groupby("AgeInDays")[TARGET].agg(["mean", "median", "count"]))

print("\n" + "=" * 70)
print("7. WATER/CEMENT RATIO (classic domain feature) vs Strength")
print("=" * 70)
wc_ratio = train["WaterComponent"] / train["CementComponent"]
print(f"w/c ratio: mean={wc_ratio.mean():.4f} std={wc_ratio.std():.4f} "
      f"corr with target (pearson)={wc_ratio.corr(train[TARGET]):.4f} "
      f"(spearman)={wc_ratio.corr(train[TARGET], method='spearman'):.4f}")

print("\n" + "=" * 70)
print("8. TRAIN vs TEST DISTRIBUTION SHIFT (numeric means/std)")
print("=" * 70)
cmp = pd.DataFrame({
    "train_mean": train[num_feats].mean(), "test_mean": test[num_feats].mean(),
    "train_std": train[num_feats].std(), "test_std": test[num_feats].std(),
})
cmp["mean_diff_pct"] = (cmp["test_mean"] - cmp["train_mean"]) / cmp["train_mean"] * 100
print(cmp.round(4))

print("\n" + "=" * 70)
print("9. ROW-LEVEL DUPLICATION BETWEEN TRAIN/TEST (feature match)")
print("=" * 70)
merged = train[feats].merge(test[feats], how="inner")
print(f"train feature-rows that also appear in test: {len(merged)}")

print("\n" + "=" * 70)
print("10. SMALL-DATA / OVERFIT CONTEXT (why generic LGB/XGB got 0 blend weight)")
print("=" * 70)
print(f"n_train={len(train)}  n_features={len(feats)}  rows-per-feature={len(train)/len(feats):.0f}")
print("With only ~5.4k rows and 8 raw features, generic (untuned) LGB/XGB with default")
print("n_estimators/learning_rate tend to overfit noise on CV folds harder than CatBoost's")
print("ordered boosting + symmetric trees, which regularize better out-of-the-box on small")
print("tabular data. This motivates: (a) explicit L1/L2 regularization + early stopping for")
print("LGB/XGB, (b) shallower trees, (c) feature engineering to give all boosters more signal.")

print("\nDONE.")
