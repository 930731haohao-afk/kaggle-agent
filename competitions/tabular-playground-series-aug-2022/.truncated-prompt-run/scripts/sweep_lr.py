"""Sweep LR config: C grid, feature subsets, per-product rank postprocess."""
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, load_raw, standardize_per_product

warnings.filterwarnings("ignore")
SEED = 42
COMP_DIR = os.path.join(os.path.dirname(__file__), "..")

SCALE_COLS = ["loading_log", "loading", "measurement_17", "measurement_0",
              "measurement_1", "measurement_2", "na_count", "meas_avg_3_16"]

SUBSETS = {
    "min": ["loading_log", "measurement_17", "m3_na", "m5_na"],
    "min_rawload": ["loading", "measurement_17", "m3_na", "m5_na"],
    "min+m0": ["loading_log", "measurement_17", "m3_na", "m5_na", "measurement_0"],
    "min+m1": ["loading_log", "measurement_17", "m3_na", "m5_na", "measurement_1"],
    "min+m2": ["loading_log", "measurement_17", "m3_na", "m5_na", "measurement_2"],
    "min+m012": ["loading_log", "measurement_17", "m3_na", "m5_na",
                 "measurement_0", "measurement_1", "measurement_2"],
    "min+avg": ["loading_log", "measurement_17", "m3_na", "m5_na", "meas_avg_3_16"],
    "min-nona": ["loading_log", "measurement_17"],
}


def cv_lr(df, cols, C, penalty="l2"):
    y = df["failure"].values
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=5)
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    for tr, va in gkf.split(df, y, df["product_code"].values):
        m = LogisticRegression(C=C, penalty=penalty, solver=solver,
                               max_iter=2000, random_state=SEED)
        m.fit(df.iloc[tr][cols], y[tr])
        oof[va] = m.predict_proba(df.iloc[va][cols])[:, 1]
    return oof


def pooled_and_ranked(df, oof):
    y = df["failure"].values
    raw = roc_auc_score(y, oof)
    # per-product rank normalization
    rn = np.zeros(len(df))
    for pc in df["product_code"].unique():
        m = (df["product_code"] == pc).values
        rn[m] = rankdata(oof[m]) / m.sum()
    ranked = roc_auc_score(y, rn)
    return raw, ranked


def main():
    train, _ = load_raw()
    feat = build_features(train)
    feat = standardize_per_product(feat, SCALE_COLS)

    rows = []
    for name, C in itertools.product(SUBSETS, [1e-4, 1e-3, 1e-2, 1e-1, 1.0]):
        oof = cv_lr(feat, SUBSETS[name], C)
        raw, ranked = pooled_and_ranked(feat, oof)
        rows.append((name, C, raw, ranked))
    res = pd.DataFrame(rows, columns=["subset", "C", "auc", "auc_rankpp"])
    res = res.sort_values("auc", ascending=False)
    print(res.to_string(index=False))
    best = res.iloc[0]
    print(f"\nBEST: {best['subset']} C={best['C']} auc={best['auc']:.5f} "
          f"rankpp={best['auc_rankpp']:.5f}")


if __name__ == "__main__":
    main()
