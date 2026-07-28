"""Baselines: LogisticRegression + shallow LightGBM, leave-one-group-out CV.

Post-processing: per-group rank normalization (validated +0.0007 in prior run).
Saves OOF + test predictions to preds/*.npz for blend / tree-search reuse.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

COMP = "competitions/tabular-playground-series-aug-2022"
sys.path.insert(0, f"{COMP}/scripts")
from prep import build_features

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

SEED = 42
os.makedirs(f"{COMP}/preds", exist_ok=True)

tr, te, y, groups, test_ids = build_features()
codes = np.unique(groups)
print("fold groups:", codes)


def rank_pp(preds: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """Per-group rank normalization to [0,1]."""
    out = np.zeros_like(preds, dtype=float)
    for g in np.unique(grp):
        m = grp == g
        out[m] = rankdata(preds[m]) / (m.sum() + 1)
    return out


def logo_cv(model_fn, feats: list[str]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Leave-one-group-out CV. Returns (oof, test_pred_mean, fold_aucs)."""
    oof = np.zeros(len(tr))
    test_pred = np.zeros(len(te))
    fold_aucs = []
    for code in codes:
        va = groups == code
        trn = ~va
        model = model_fn()
        model.fit(tr.loc[trn, feats], y[trn])
        oof[va] = model.predict_proba(tr.loc[va, feats])[:, 1]
        test_pred += model.predict_proba(te[feats])[:, 1] / len(codes)
        fold_aucs.append(roc_auc_score(y[va], oof[va]))
    return oof, test_pred, fold_aucs


def report(name: str, oof: np.ndarray, test_pred: np.ndarray, feats: list[str]):
    raw = roc_auc_score(y, oof)
    pp = roc_auc_score(y, rank_pp(oof, groups))
    print(f"{name}: raw OOF AUC {raw:.5f} | rank-pp {pp:.5f}")
    np.savez(f"{COMP}/preds/{name}.npz", oof=oof, test=test_pred, feats=np.array(feats, dtype=object))
    return raw, pp


LR_FEATS = ["loading", "measurement_17", "measurement_3_na", "measurement_5_na", "measurement_2"]

def make_lr():
    return LogisticRegression(C=0.01, max_iter=2000, random_state=SEED)

lr_oof, lr_test, lr_folds = logo_cv(make_lr, LR_FEATS)
lr_raw, lr_pp = report("lr_base", lr_oof, lr_test, LR_FEATS)

LGB_FEATS = [c for c in tr.columns if c != "product_code" and not c.endswith("_raw")]

def make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=80, reg_alpha=1.0, reg_lambda=2.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        random_state=SEED, deterministic=True, force_row_wise=True,
        num_threads=10, verbosity=-1)

lgb_oof, lgb_test, lgb_folds = logo_cv(make_lgb, LGB_FEATS)
lgb_raw, lgb_pp = report("lgb_base", lgb_oof, lgb_test, LGB_FEATS)

# ---- blend: weight search on rank-pp'd OOF ----
lr_r = rank_pp(lr_oof, groups)
lgb_r = rank_pp(lgb_oof, groups)
best_w, best_auc = 1.0, roc_auc_score(y, lr_r)
for w in np.arange(0, 1.01, 0.05):
    auc = roc_auc_score(y, w * lr_r + (1 - w) * lgb_r)
    if auc > best_auc:
        best_w, best_auc = w, auc
print(f"blend: w_lr={best_w:.2f} rank-pp OOF AUC {best_auc:.5f}")

lr_test_r = rank_pp(lr_test, np.array(pd.read_csv(f"{COMP}/data/test.csv").product_code))
lgb_test_r = rank_pp(lgb_test, np.array(pd.read_csv(f"{COMP}/data/test.csv").product_code))
blend_test = best_w * lr_test_r + (1 - best_w) * lgb_test_r
np.savez(f"{COMP}/preds/blend_base.npz",
         oof=best_w * lr_r + (1 - best_w) * lgb_r, test=blend_test,
         w=np.array([best_w]))

cv_desc = {"strategy": "leave-one-group-out GroupKFold (product_code, 5 folds)", "seed": SEED}
experiment_log.log_experiment_v2(
    COMP, model="LogisticRegression C=0.01", metric="roc_auc", direction="maximize",
    score=lr_pp, cv={**cv_desc, "fold_aucs": [round(a, 5) for a in lr_folds]},
    features=LR_FEATS, postprocess=["per-group rank normalization"],
    notes=f"baseline; raw OOF {lr_raw:.5f}; prior-run reference 0.59131")
experiment_log.log_experiment_v2(
    COMP, model="LGBM shallow (leaves7 depth3)", metric="roc_auc", direction="maximize",
    score=lgb_pp, cv={**cv_desc, "fold_aucs": [round(a, 5) for a in lgb_folds]},
    features=LGB_FEATS, postprocess=["per-group rank normalization"],
    notes=f"baseline; raw OOF {lgb_raw:.5f}")
experiment_log.log_experiment_v2(
    COMP, model="blend LR+LGB (rank space)", metric="roc_auc", direction="maximize",
    score=best_auc, cv=cv_desc,
    ensemble={"weights": {"lr": round(best_w, 2), "lgb": round(1 - best_w, 2)}, "space": "per-group rank"},
    postprocess=["per-group rank normalization"],
    notes="grid weight search on OOF")
print("logged 3 experiments")
