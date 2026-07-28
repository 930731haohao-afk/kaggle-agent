"""Feature engineering for tabular-playground-series-aug-2022.

Builds features per product_code (products are disjoint between train/test,
so per-product statistics use only that product's own feature columns —
never the target).
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MEAS = [f"measurement_{i}" for i in range(18)]
M_FULL = [f"measurement_{i}" for i in range(3, 17)]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    return train, test


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature build on a frame that may contain several product codes.

    All imputation/scaling stats are computed per product_code from feature
    columns only (no target), so applying to train+test jointly is leakage-free
    and matches the disjoint-product deployment setting.
    """
    df = df.copy()
    df["m3_na"] = df["measurement_3"].isnull().astype(int)
    df["m5_na"] = df["measurement_5"].isnull().astype(int)
    df["na_count"] = df[MEAS + ["loading"]].isnull().sum(axis=1)

    out = []
    for _, sub in df.groupby("product_code", sort=False):
        sub = sub.copy()
        # measurement_17: per-product linear imputation from its 4 most
        # correlated measurement columns (HuberRegressor for robustness)
        corr = sub[M_FULL + ["measurement_17"]].corr()["measurement_17"].drop("measurement_17")
        preds = corr.abs().sort_values(ascending=False).head(4).index.tolist()
        fit_rows = sub[preds + ["measurement_17"]].dropna()
        can_pred = sub["measurement_17"].isnull() & sub[preds].notnull().all(axis=1)
        if len(fit_rows) > 50 and can_pred.any():
            model = HuberRegressor(epsilon=1.35, max_iter=400)
            model.fit(fit_rows[preds], fit_rows["measurement_17"])
            sub.loc[can_pred, "measurement_17"] = model.predict(sub.loc[can_pred, preds])
        # remaining numeric NaN: per-product median
        for c in MEAS + ["loading"]:
            sub[c] = sub[c].fillna(sub[c].median())
        out.append(sub)
    df = pd.concat(out).sort_index()

    df["loading_log"] = np.log(df["loading"])
    df["area"] = df["attribute_2"] * df["attribute_3"]
    df["meas_avg_3_16"] = df[M_FULL].mean(axis=1)
    return df


def standardize_per_product(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """z-score each feature within its product_code (feature-only stats)."""
    df = df.copy()
    g = df.groupby("product_code")[cols]
    df[cols] = (df[cols] - g.transform("mean")) / g.transform("std").replace(0, 1)
    return df
