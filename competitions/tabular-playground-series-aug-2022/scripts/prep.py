"""Shared data prep for tabular-playground-series-aug-2022.

Builds features validated in the prior run (see knowledge/experience.md, aug-2022 entries):
- per-product-code z-scaling of numeric columns (groups are disjoint train/test, so
  each code is scaled on its own rows; no target used -> no leakage)
- measurement_17 imputed by per-code HuberRegressor on its top-4 correlated
  measurement columns (feature-only model, fit per code)
- remaining measurement NaNs -> per-code median
- missingness indicator flags for every column with NaNs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

DATA = "competitions/tabular-playground-series-aug-2022/data"
M_COLS = [f"measurement_{i}" for i in range(18)]
NUM_COLS = ["loading"] + M_COLS


def _impute_m17(df: pd.DataFrame) -> pd.Series:
    """Per-code Huber regression imputation of measurement_17."""
    out = df["measurement_17"].copy()
    donors = [f"measurement_{i}" for i in range(3, 17)]
    for code, g in df.groupby("product_code"):
        obs = g[g["measurement_17"].notna()]
        corr = obs[donors + ["measurement_17"]].corr()["measurement_17"].drop("measurement_17")
        top = corr.abs().sort_values(ascending=False).head(4).index.tolist()
        X_full = g[top].copy()
        for c in top:
            X_full[c] = X_full[c].fillna(g[c].median())
        fit_mask = g["measurement_17"].notna()
        hr = HuberRegressor(max_iter=500)
        hr.fit(X_full[fit_mask.values], g.loc[fit_mask, "measurement_17"])
        miss_idx = g.index[~fit_mask.values]
        if len(miss_idx):
            out.loc[miss_idx] = hr.predict(X_full.loc[miss_idx])
    return out


def build_features() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (train_feat, test_feat, y, groups_train, test_ids)."""
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    y = train["failure"].values
    groups = train["product_code"].values
    test_ids = test["id"].values

    parts = []
    for df in (train, test):
        f = pd.DataFrame(index=df.index)
        f["product_code"] = df["product_code"]
        # NA flags (before imputation)
        for c in NUM_COLS:
            if df[c].isna().any():
                f[f"{c}_na"] = df[c].isna().astype(int)
        # m17 model-based imputation
        df = df.copy()
        df["measurement_17"] = _impute_m17(df)
        # per-code median fill + per-code z-scale
        for c in NUM_COLS:
            filled = df.groupby("product_code")[c].transform(lambda s: s.fillna(s.median()))
            mu = filled.groupby(df["product_code"]).transform("mean")
            sd = filled.groupby(df["product_code"]).transform("std")
            f[c] = (filled - mu) / sd
            f[f"{c}_raw"] = filled
        parts.append(f)

    tr, te = parts
    # align NA-flag columns (some may exist in one split only)
    na_cols = sorted(set(tr.columns) | set(te.columns))
    for c in na_cols:
        for f in (tr, te):
            if c not in f.columns:
                f[c] = 0
    tr = tr[na_cols]
    te = te[na_cols]
    return tr, te, y, groups, test_ids


if __name__ == "__main__":
    tr, te, y, groups, test_ids = build_features()
    print("train feat:", tr.shape, "test feat:", te.shape)
    print("cols:", list(tr.columns))
    print("NaNs train:", int(tr.drop(columns=['product_code']).isna().sum().sum()),
          "test:", int(te.drop(columns=['product_code']).isna().sum().sum()))
