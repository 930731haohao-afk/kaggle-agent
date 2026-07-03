"""Train LGB/XGB/CatBoost with StratifiedKFold OOF, weight-searched blend,
and write a submission. Metric: ROC-AUC (maximize).

CV: 5-fold StratifiedKFold on booking_status (mild 60.8/39.2 imbalance).
Test predictions are the average of the 5 fold models (same convention used
in s3e16/scripts/train.py) rather than a full-data retrain, since it reuses
the already-fitted fold models and best_iteration from early stopping.
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

COMP = "competitions/playground-series-s3e7"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "booking_status", "id"
N_SPLITS, SEED = 5, 42

os.makedirs(SUB, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

Xtr_full = build_features(train)
Xte_full = build_features(test)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.int32)
Xtest = Xte_full[FEATS].to_numpy(np.float32)
print(f"n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))


def run_lgb():
    import lightgbm as lgb
    params = dict(objective="binary", metric="auc", n_estimators=3000,
                  learning_rate=0.03, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
        print(f"  [LGB] fold{f} AUC={roc_auc_score(y[va], oof[va]):.5f} best_iter={m.best_iteration_}")
    return oof, pred, params


def run_xgb():
    import xgboost as xgb
    params = dict(objective="binary:logistic", n_estimators=3000, learning_rate=0.03,
                  max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                  eval_metric="auc", early_stopping_rounds=150)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
        print(f"  [XGB] fold{f} AUC={roc_auc_score(y[va], oof[va]):.5f} best_iter={m.best_iteration}")
    return oof, pred, params


def run_cat():
    from catboost import CatBoostClassifier, Pool
    params = dict(loss_function="Logloss", eval_metric="AUC", iterations=4000,
                  learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                  random_seed=SEED, thread_count=-1, verbose=False)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = CatBoostClassifier(**params)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
        print(f"  [CAT] fold{f} AUC={roc_auc_score(y[va], oof[va]):.5f} best_iter={m.get_best_iteration()}")
    return oof, pred, params


results = {}
for name, fn in [("LGB", run_lgb), ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"\n=== {name} ===")
    oof, pred, params = fn()
    dt = time.time() - t0
    auc = roc_auc_score(y, oof)
    print(f"{name} OOF AUC={auc:.5f}  time={dt:.1f}s")
    results[name] = dict(oof=oof, pred=pred, params=params, auc=auc, time_s=round(dt, 1))

# --- weight-search blend (grid step 0.05, weights sum to 1) ---
names = list(results.keys())
best_w, best_auc = None, -1
grid = [round(w * 0.05, 2) for w in range(21)]
for w in product(grid, repeat=len(names)):
    if abs(sum(w) - 1.0) > 1e-9:
        continue
    blend_oof = sum(wi * results[n]["oof"] for wi, n in zip(w, names))
    a = roc_auc_score(y, blend_oof)
    if a > best_auc:
        best_auc, best_w = a, w

blend_weights = {n: w for n, w in zip(names, best_w)}
blend_oof = sum(blend_weights[n] * results[n]["oof"] for n in names)
blend_pred = sum(blend_weights[n] * results[n]["pred"] for n in names)
blend_auc = roc_auc_score(y, blend_oof)
print(f"\n=== BLEND ===")
print(f"weights={blend_weights}  OOF AUC={blend_auc:.5f}")

BASELINE = 0.89882
print(f"\nBaseline (generic blend): {BASELINE}")
print(f"This blend delta vs baseline: {blend_auc - BASELINE:+.5f}")

# --- quick feature importance (LGB, refit on full data for reporting only) ---
import lightgbm as lgb
imp_model = lgb.LGBMClassifier(**{**results["LGB"]["params"], "n_estimators": 500})
imp_model.set_params(n_estimators=500)
imp_model.fit(X, y)
importances = pd.Series(imp_model.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\nTop 15 feature importances (LGB, full-data refit):")
print(importances.head(15).to_string())

# --- submission ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_name = f"sub_blend_{blend_auc:.5f}_{ts}.csv"
sample_sub = pd.read_csv(f"{DATA}/sample_submission.csv")
submission = pd.DataFrame({ID: test[ID], TARGET: blend_pred})
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == list(sample_sub.columns)
assert submission.isnull().sum().sum() == 0
submission.to_csv(f"{SUB}/{sub_name}", index=False)
print(f"\nSaved submission: {SUB}/{sub_name}")

# --- log experiment (v2 schema, mandatory) ---
base_models = [
    {"model": n, "params": {k: str(v) for k, v in results[n]["params"].items()},
     "oof_auc": round(float(results[n]["auc"]), 6), "time_s": results[n]["time_s"]}
    for n in names
]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB+XGB+CAT weight-searched blend (trimmed features, iter2)",
    metric="roc_auc",
    direction="maximize",
    score=round(float(blend_auc), 6),
    cv={"scheme": "5fold", "n_splits": N_SPLITS, "seed": SEED, "strategy": "StratifiedKFold(booking_status)"},
    features=FEATS,
    base_models=base_models,
    ensemble={"weights": blend_weights, "score": round(float(blend_auc), 6)},
    postprocess=None,
    submission=sub_name,
    notes=(f"Delta vs generic baseline {BASELINE}: {blend_auc - BASELINE:+.5f}. "
           f"Iter2/reflexion: trimmed to 8 engineered features (dropped cyclical "
           f"month/date, price_per_night, special*price interaction) after iter1's "
           f"14-feature set (17 raw + 14 engineered) scored 0.89788, below baseline."),
)
print(f"\nLogged experiment_id={exp_id} to experiments.json")
