"""Refine around best LR config; re-blend with LGBM; log final experiments."""
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    os.path.join(COMP_DIR, "..", "..", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py"))
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

SCALE_COLS = ["loading_log", "loading", "measurement_17", "measurement_0",
              "measurement_1", "measurement_2", "na_count", "meas_avg_3_16"]

SUBSETS = {
    "m2": ["loading_log", "measurement_17", "m3_na", "m5_na", "measurement_2"],
    "m2_raw": ["loading", "measurement_17", "m3_na", "m5_na", "measurement_2"],
    "m2_avg": ["loading_log", "measurement_17", "m3_na", "m5_na",
               "measurement_2", "meas_avg_3_16"],
    "m2_nacount": ["loading_log", "measurement_17", "m3_na", "m5_na",
                   "measurement_2", "na_count"],
}


def cv_lr(df, cols, C):
    y = df["failure"].values
    oof = np.zeros(len(df))
    for tr, va in GroupKFold(n_splits=5).split(df, y, df["product_code"].values):
        m = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
        m.fit(df.iloc[tr][cols], y[tr])
        oof[va] = m.predict_proba(df.iloc[va][cols])[:, 1]
    return oof


def rank_pp(df, oof):
    rn = np.zeros(len(df))
    for pc in df["product_code"].unique():
        m = (df["product_code"] == pc).values
        rn[m] = rankdata(oof[m]) / m.sum()
    return rn


def main():
    train, _ = load_raw()
    feat_raw = build_features(train)
    feat = standardize_per_product(feat_raw, SCALE_COLS)
    y = feat["failure"].values

    rows, oofs = [], {}
    for name, C in itertools.product(SUBSETS, [0.02, 0.05, 0.1, 0.2, 0.5]):
        oof = cv_lr(feat, SUBSETS[name], C)
        raw = roc_auc_score(y, oof)
        rp = roc_auc_score(y, rank_pp(feat, oof))
        rows.append((name, C, raw, rp))
        oofs[(name, C)] = oof
    res = pd.DataFrame(rows, columns=["subset", "C", "auc", "auc_rankpp"])
    print(res.sort_values("auc_rankpp", ascending=False).head(10).to_string(index=False))

    bname, bC, bauc, brp = res.sort_values("auc_rankpp", ascending=False).iloc[0]
    best_oof = oofs[(bname, bC)]

    # blend with LGB oof from baseline run
    lgb_oof = np.load(os.path.join(COMP_DIR, "scripts", "oof_lgb_full.npy"))
    r_lr, r_lgb = rankdata(best_oof), rankdata(lgb_oof)
    best_w, best_blend = 0.0, 0.0
    for w in np.arange(0, 0.51, 0.05):
        blend = (1 - w) * r_lr + w * r_lgb
        a = roc_auc_score(y, rank_pp(feat, blend))
        if a > best_blend:
            best_blend, best_w = a, w
    print(f"\nbest LR: {bname} C={bC} auc={bauc:.5f} rankpp={brp:.5f}")
    print(f"blend w_lgb={best_w:.2f} rankpp AUC={best_blend:.5f}")

    np.save(os.path.join(COMP_DIR, "scripts", "oof_lr_best.npy"), best_oof)

    experiment_log.log_experiment_v2(
        COMP_DIR, model=f"LogisticRegression(C={bC})",
        metric="roc_auc", direction="maximize", score=brp,
        cv={"strategy": "GroupKFold(product_code, 5)"},
        features=SUBSETS[bname],
        postprocess=["per-product rank normalization"],
        notes=f"sweep winner {bname}; raw OOF {bauc:.5f}, rank-pp {brp:.5f}",
    )
    experiment_log.log_experiment_v2(
        COMP_DIR, model="rank-blend LR_best+LGB",
        metric="roc_auc", direction="maximize", score=best_blend,
        cv={"strategy": "GroupKFold(product_code, 5)"},
        ensemble={"weights": {"lr_best": round(1 - best_w, 2), "lgb_full": round(best_w, 2)},
                  "space": "rank"},
        postprocess=["per-product rank normalization"],
        notes="final blend candidate",
    )
    with open(os.path.join(COMP_DIR, "scripts", "best_config.txt"), "w") as f:
        f.write(f"{bname}\t{bC}\t{best_w}\t{brp:.6f}\t{best_blend:.6f}\n")
        f.write(",".join(SUBSETS[bname]) + "\n")


if __name__ == "__main__":
    main()
