"""EDA for tabular-playground-series-aug-2022 (product failure, AUC)."""
import pandas as pd
import numpy as np

DATA = "competitions/tabular-playground-series-aug-2022/data"
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

print(f"train {train.shape}, test {test.shape}")
print(f"target rate: {train.failure.mean():.4f}")

print("\n== product_code train vs test ==")
print("train:", sorted(train.product_code.unique()), train.product_code.value_counts().to_dict())
print("test :", sorted(test.product_code.unique()), test.product_code.value_counts().to_dict())

print("\n== failure rate per product code ==")
print(train.groupby("product_code").failure.mean())

print("\n== categorical attrs per code (constant within code?) ==")
for c in ["attribute_0", "attribute_1", "attribute_2", "attribute_3"]:
    g = train.groupby("product_code")[c].nunique()
    print(c, "nunique-within-code:", g.to_dict(), "| train values:", train[c].unique()[:6])

print("\n== missing % (train / test) ==")
miss_tr = train.isna().mean()
miss_te = test.isna().mean()
for c in train.columns:
    if miss_tr[c] > 0 or (c in test.columns and miss_te.get(c, 0) > 0):
        print(f"  {c}: {miss_tr[c]*100:.2f}% / {miss_te.get(c, np.nan)*100:.2f}%")

print("\n== missingness vs target (failure rate when missing vs present) ==")
base = train.failure.mean()
for c in train.columns:
    if train[c].isna().sum() > 50:
        r_miss = train.loc[train[c].isna(), "failure"].mean()
        n_miss = train[c].isna().sum()
        print(f"  {c}: miss n={n_miss}, fail_when_miss={r_miss:.4f} (base {base:.4f})")

print("\n== point-biserial corr with target (numeric) ==")
num_cols = [c for c in train.columns if c not in ("id", "product_code", "attribute_0", "attribute_1", "failure")]
corrs = train[num_cols + ["failure"]].corr()["failure"].drop("failure").sort_values(key=abs, ascending=False)
print(corrs.head(12).round(4))

print("\n== measurement_17 corr with other measurements (per product code, for imputation) ==")
m_cols = [f"measurement_{i}" for i in range(3, 17)]
for code, g in train.groupby("product_code"):
    cc = g[m_cols + ["measurement_17"]].corr()["measurement_17"].drop("measurement_17").sort_values(key=abs, ascending=False)
    print(f"  {code}: top4 = {[(i, round(v,3)) for i, v in cc.head(4).items()]}")

print("\n== per-code feature mean shift (loading, m17) train vs test ==")
for c in ["loading", "measurement_17", "measurement_0", "measurement_1", "measurement_2"]:
    print(f"  {c}: train per-code mean {train.groupby('product_code')[c].mean().round(2).to_dict()}")
    print(f"      test per-code mean {test.groupby('product_code')[c].mean().round(2).to_dict()}")

print("\n== attribute_2/3 values train vs test (do they overlap?) ==")
for c in ["attribute_0", "attribute_1", "attribute_2", "attribute_3"]:
    print(f"  {c}: train {sorted(train[c].astype(str).unique())} test {sorted(test[c].astype(str).unique())}")

print("\n== duplicates ==")
feat = [c for c in train.columns if c not in ("id", "failure")]
print("dup feature rows in train:", train.duplicated(subset=feat).sum())
