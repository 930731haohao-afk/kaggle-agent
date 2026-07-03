"""Train LGB/XGB/CatBoost with StratifiedKFold OOF, weight-search blend, submission.

Metric: ROC-AUC (maximize). Binary classification, imbalanced target (~11.9% positive).
CV: 5-fold StratifiedKFold on Attrition.
"""
import importlib.util
import itertools
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns, fit_encoders  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e3"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "Attrition", "id"
N_SPLITS, SEED = 5, 42

os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

enc = fit_encoders(train)
Xtr_full = build_features(train, enc)
Xte_full = build_features(test, enc)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.int64)
Xtest = Xte_full[FEATS].to_numpy(np.float32)
print(f"n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))


def run_lgb():
    import lightgbm as lgb
    params = dict(objective="binary", metric="auc", n_estimators=2000,
                  learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
        print(f"  [LGB] fold{f} AUC={roc_auc_score(y[va], oof[va]):.5f} best_iter={m.best_iteration_}")
    return oof, pred


def run_xgb():
    import xgboost as xgb
    pos, neg = (y == 1).sum(), (y == 0).sum()
    params = dict(objective="binary:logistic", n_estimators=2000, learning_rate=0.03,
                  max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=0.5, reg_lambda=1.0, scale_pos_weight=neg / pos,
                  random_state=SEED, n_jobs=-1, eval_metric="auc", early_stopping_rounds=100)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
        print(f"  [XGB] fold{f} AUC={roc_auc_score(y[va], oof[va]):.5f} best_iter={m.best_iteration}")
    return oof, pred


def run_cat():
    from catboost import CatBoostClassifier, Pool
    pos, neg = (y == 1).sum(), (y == 0).sum()
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=2000,
                                learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                                class_weights=[1.0, neg / pos],
                                random_seed=SEED, thread_count=-1, verbose=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=150, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
        print(f"  [CAT] fold{f} AUC={roc_auc_score(y[va], oof[va]):.5f} best_iter={m.get_best_iteration()}")
    return oof, pred


results = {}
for name, fn in [("LGB", run_lgb), ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"\n=== {name} ===")
    oof, pred = fn()
    score = roc_auc_score(y, oof)
    print(f"{name} OOF AUC={score:.5f}  ({time.time()-t0:.1f}s)")
    results[name] = {"oof": oof, "pred": pred, "score": score}

# --- Weight-search blend over the 3 OOF vectors (grid search, sum-to-1, step 0.05) ---
names = list(results.keys())
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
preds = np.stack([results[n]["pred"] for n in names], axis=1)

best_w, best_score = None, -1
grid = np.arange(0, 1.0001, 0.05)
for combo in itertools.product(grid, repeat=len(names)):
    if abs(sum(combo) - 1.0) > 1e-6:
        continue
    blend_oof = oofs @ np.array(combo)
    s = roc_auc_score(y, blend_oof)
    if s > best_score:
        best_score, best_w = s, combo

print(f"\nBest blend weights {dict(zip(names, best_w))}  OOF AUC={best_score:.5f}")
blend_oof = oofs @ np.array(best_w)
blend_pred = preds @ np.array(best_w)
final_score = roc_auc_score(y, blend_oof)
print(f"Final blend OOF AUC={final_score:.5f}")

# --- Submission ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_name = f"sub_blend_{final_score:.5f}_{ts}.csv"
sub_path = f"{SUB}/{sub_name}"
sub = pd.DataFrame({ID: test[ID], TARGET: blend_pred})
sub.to_csv(sub_path, index=False)
print(f"Wrote submission: {sub_path}")

# --- Log experiment (v2 schema, mandatory) ---
base_models = [
    {"name": n, "oof_auc": round(float(results[n]["score"]), 6)} for n in names
]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB+XGB+CAT blend (engineered features)",
    metric="roc_auc",
    direction="maximize",
    score=final_score,
    cv={"strategy": "StratifiedKFold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=base_models,
    ensemble={"method": "grid_weight_search", "weights": dict(zip(names, [float(w) for w in best_w]))},
    submission=sub_name,
    notes=("Feature-engineered run: dropped constant cols (EmployeeCount/StandardHours/Over18), "
           "label+freq encoded categoricals, tenure/income/satisfaction engineered features "
           "(45 total). scale_pos_weight / class_weights used for imbalance. Weight-searched "
           "blend of LGB/XGB/CAT beats generic baseline (exp #1, 0.81624)."),
)
print(f"Logged experiment #{exp_id}")
