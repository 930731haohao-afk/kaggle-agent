"""EDA for tabular-playground-series-aug-2022 (product failure, AUC)."""
import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))

print(f"train {train.shape}, test {test.shape}")
print("\n--- dtypes ---")
print(train.dtypes)

print("\n--- target ---")
print(train["failure"].value_counts(normalize=True))

print("\n--- product_code train vs test ---")
print("train:", sorted(train["product_code"].unique()))
print("test :", sorted(test["product_code"].unique()))
print(train.groupby("product_code")["failure"].agg(["mean", "count"]))

cats = ["attribute_0", "attribute_1"]
for c in cats:
    print(f"\n--- {c} ---")
    print("train:", train.groupby([c])["failure"].agg(["mean", "count"]).to_dict())
    print("test unique:", test[c].unique())

print("\n--- attribute constancy within product_code ---")
attrs = ["attribute_0", "attribute_1", "attribute_2", "attribute_3"]
print(train.groupby("product_code")[attrs].nunique())
print(train.groupby("product_code")[attrs].first())
print(test.groupby("product_code")[attrs].first())

print("\n--- missing values (train / test %) ---")
miss = pd.DataFrame({
    "train%": train.isnull().mean() * 100,
    "test%": test.isnull().mean() * 100,
})
print(miss[miss["train%"] + miss["test%"] > 0].round(2))

print("\n--- missingness vs target ---")
for c in train.columns:
    if train[c].isnull().any():
        m = train[c].isnull()
        print(f"{c}: fail-rate missing={train.loc[m,'failure'].mean():.4f} "
              f"present={train.loc[~m,'failure'].mean():.4f} (n_miss={m.sum()})")

num_cols = [c for c in train.columns if c.startswith(("loading", "measurement"))]
print("\n--- point-biserial corr with target ---")
corrs = train[num_cols + ["failure"]].corr()["failure"].drop("failure").sort_values(key=abs, ascending=False)
print(corrs.round(4))

print("\n--- loading stats ---")
print(train["loading"].describe())
print("skew:", train["loading"].skew().round(3))
print("log-loading corr with failure:",
      np.log(train["loading"]).corr(train["failure"]).round(4))

print("\n--- measurement correlations (m3-m17 with m17) per product ---")
mcols = [f"measurement_{i}" for i in range(3, 18)]
for pc in sorted(train["product_code"].unique()):
    sub = train[train["product_code"] == pc]
    c17 = sub[mcols].corr()["measurement_17"].drop("measurement_17")
    top = c17.abs().sort_values(ascending=False).head(4)
    print(pc, {k: round(c17[k], 3) for k in top.index})

print("\n--- train/test numeric shift (mean per product code) ---")
key_cols = ["loading", "measurement_17", "measurement_0", "measurement_1", "measurement_2"]
print("train:\n", train.groupby("product_code")[key_cols].mean().round(2))
print("test:\n", test.groupby("product_code")[key_cols].mean().round(2))

print("\n--- id overlap/order ---")
print("train id range", train["id"].min(), train["id"].max(),
      "test id range", test["id"].min(), test["id"].max())

print("\n--- duplicate rows ---")
print("train dups:", train.drop(columns=["id"]).duplicated().sum())
