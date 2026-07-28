"""Baselines: LogisticRegression variants + LightGBM under GroupKFold(product_code)."""
import importlib.util
import os
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
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

LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.03, num_leaves=7,
    min_child_samples=60, reg_alpha=1.0, reg_lambda=2.0, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, n_estimators=800,
    num_threads=10, deterministic=True, force_row_wise=True,
    random_state=SEED, verbosity=-1,
)

M_ALL = [f"measurement_{i}" for i in range(18)]

FEATURE_SETS = {
    "lr_min": ["loading_log", "measurement_17", "m3_na", "m5_na"],
    "lr_plus": ["loading_log", "measurement_17", "m3_na", "m5_na",
                "measurement_0", "measurement_1", "measurement_2", "na_count"],
    "lgb_full": ["loading", "loading_log"] + M_ALL +
                ["m3_na", "m5_na", "na_count", "area", "meas_avg_3_16",
                 "attribute_2", "attribute_3"],
}
SCALE_COLS = ["loading_log", "measurement_17", "measurement_0",
              "measurement_1", "measurement_2", "na_count", "meas_avg_3_16"]


def cv_model(df: pd.DataFrame, cols: list[str], kind: str) -> tuple[np.ndarray, list[float]]:
    """Leave-one-product-out CV. Returns OOF predictions and per-fold AUCs."""
    y = df["failure"].values
    groups = df["product_code"].values
    oof = np.zeros(len(df))
    fold_scores = []
    gkf = GroupKFold(n_splits=df["product_code"].nunique())
    for tr, va in gkf.split(df, y, groups):
        if kind == "lr":
            model = LogisticRegression(C=0.0001, max_iter=1000, random_state=SEED)
            model.fit(df.iloc[tr][cols], y[tr])
            oof[va] = model.predict_proba(df.iloc[va][cols])[:, 1]
        else:
            model = lgb.LGBMClassifier(**LGB_PARAMS)
            model.fit(df.iloc[tr][cols], y[tr],
                      eval_set=[(df.iloc[va][cols], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = model.predict_proba(df.iloc[va][cols])[:, 1]
        fold_scores.append(roc_auc_score(y[va], oof[va]))
    return oof, fold_scores


def main() -> None:
    train, _ = load_raw()
    feat = build_features(train)
    feat_std = standardize_per_product(feat, SCALE_COLS)

    results = {}
    for name, cols in FEATURE_SETS.items():
        kind = "lr" if name.startswith("lr") else "lgb"
        src = feat_std if kind == "lr" else feat
        oof, folds = cv_model(src, cols, kind)
        auc = roc_auc_score(feat["failure"], oof)
        results[name] = (oof, folds, auc, cols, kind)
        print(f"{name:10s} OOF AUC {auc:.5f}  folds "
              f"{[round(f, 5) for f in folds]}")
        np.save(os.path.join(COMP_DIR, "scripts", f"oof_{name}.npy"), oof)

    # blend LR + LGB
    best_lr = max((k for k in results if results[k][4] == "lr"),
                  key=lambda k: results[k][2])
    y = feat["failure"].values
    from scipy.stats import rankdata
    r_lr = rankdata(results[best_lr][0])
    r_lgb = rankdata(results["lgb_full"][0])
    best_w, best_auc = 0.0, 0.0
    for w in np.arange(0, 1.01, 0.05):
        a = roc_auc_score(y, w * r_lgb + (1 - w) * r_lr)
        if a > best_auc:
            best_auc, best_w = a, w
    print(f"blend rank({best_lr})*{1-best_w:.2f} + rank(lgb)*{best_w:.2f} "
          f"-> OOF AUC {best_auc:.5f}")

    for name in results:
        oof, folds, auc, cols, kind = results[name]
        experiment_log.log_experiment_v2(
            COMP_DIR,
            model={"lr": "LogisticRegression(C=1e-4)", "lgb": "LightGBM-shallow"}[kind],
            metric="roc_auc", direction="maximize", score=auc,
            cv={"strategy": "GroupKFold(product_code, 5)", "folds": [round(f, 6) for f in folds]},
            features=cols,
            notes=f"feature set {name}; per-product std-scaling for LR; "
                  "m17 Huber-imputed per product",
        )
    experiment_log.log_experiment_v2(
        COMP_DIR, model="rank-blend LR+LGB",
        metric="roc_auc", direction="maximize", score=best_auc,
        cv={"strategy": "GroupKFold(product_code, 5)"},
        ensemble={"weights": {best_lr: round(1 - best_w, 2), "lgb_full": round(best_w, 2)},
                  "space": "rank"},
        notes="grid 0.05 on rank-averaged OOF",
    )


if __name__ == "__main__":
    main()
