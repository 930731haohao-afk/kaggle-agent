"""EDA for playground-series-s3e16 (Crab Age prediction, MAE regression).

Prints all findings to stdout so the agent can interpret them.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

DATA = "competitions/playground-series-s3e16/data"
TARGET = "Age"
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

print("\n" + "=" * 70)
print("2. TARGET ANALYSIS: Age")
print("=" * 70)
y = train[TARGET]
print(y.describe())
print(f"skew={y.skew():.3f}  kurtosis={y.kurtosis():.3f}")
print(f"unique values: {sorted(y.unique())}")
print("value_counts (top 15):")
print(y.value_counts().sort_index().head(30))
for p in [0.1, 1, 5, 50, 95, 99, 99.9]:
    print(f"  pct {p:5}: {np.percentile(y, p):.1f}")

print("\n" + "=" * 70)
print("3. CATEGORICAL: Sex")
print("=" * 70)
for c in cat_feats:
    print(f"\n-- {c} -- cardinality={train[c].nunique()}")
    print("train freq:\n", train[c].value_counts(normalize=True))
    print("test  freq:\n", test[c].value_counts(normalize=True))
    print(f"target mean per {c}:\n", train.groupby(c)[TARGET].agg(["mean", "median", "count"]))
    unseen = set(test[c].unique()) - set(train[c].unique())
    print(f"categories in test not in train: {unseen}")

print("\n" + "=" * 70)
print("4. NUMERIC FEATURE SUMMARY")
print("=" * 70)
print(train[num_feats].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])
print("\nskew / kurtosis:")
for c in num_feats:
    print(f"  {c:16} skew={train[c].skew():7.3f}  kurt={train[c].kurtosis():8.3f}  "
          f"zeros={int((train[c] == 0).sum())}  min={train[c].min():.4f}")

print("\n" + "=" * 70)
print("5. CORRELATION WITH TARGET (Pearson & Spearman)")
print("=" * 70)
corr_p = train[num_feats + [TARGET]].corr(method="pearson")[TARGET].drop(TARGET)
corr_s = train[num_feats + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
print(pd.DataFrame({"pearson": corr_p, "spearman": corr_s}).sort_values("spearman", ascending=False))

print("\n" + "=" * 70)
print("6. FEATURE-FEATURE CORRELATION (collinearity check)")
print("=" * 70)
print(train[num_feats].corr().round(3))

print("\n" + "=" * 70)
print("7. WEIGHT ADDITIVITY CHECK: Weight vs Shucked+Viscera+Shell")
print("=" * 70)
parts = train["Shucked Weight"] + train["Viscera Weight"] + train["Shell Weight"]
resid = train["Weight"] - parts
print(f"Weight - (Shucked+Viscera+Shell): mean={resid.mean():.4f} std={resid.std():.4f} "
      f"min={resid.min():.4f} max={resid.max():.4f}")
print(f"correlation Weight vs sum-of-parts: {train['Weight'].corr(parts):.5f}")
print(f"rows where sum-of-parts > Weight: {(resid < 0).sum()} ({(resid < 0).mean()*100:.2f}%)")

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
print("9. SUSPICIOUS ROWS (Height==0 etc.)")
print("=" * 70)
for c in num_feats:
    z = (train[c] == 0).sum()
    if z:
        print(f"  {c}: {z} rows == 0")
print(f"Height==0 rows target stats:\n{train.loc[train['Height'] == 0, TARGET].describe() if (train['Height']==0).any() else 'none'}")

print("\nDONE.")
