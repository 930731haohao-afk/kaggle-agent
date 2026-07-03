"""Phase B iteration, round 2: add Optuna-tuned LGB (round 1) to the model pool
(not replace) alongside the original v2 LGB/XGB/CAT, then seed-bag the tuned LGB
with a second random_state. Weight-search blend over all 5 OOF vectors.

Recipe from knowledge/experience.md: "adding tuned to pool + seed bag" should
transfer cleanly on AUC since it's a continuous rank metric (no discretization
trap as with QWK/MAE).

Baseline to beat: exp #3 blend 0.832925 (LGB solo, XGB/CAT zeroed).
Round 1: LGB_tuned solo OOF 0.837305 (exp #4).
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

results = {}


def run_lgb_orig():
    import lightgbm as lgb
    params = dict(objective="binary", metric="auc", n_estimators=2000,
                  learning_rate=0.03, num_leaves=7, min_child_samples=30,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def run_xgb():
    import xgboost as xgb
    params = dict(objective="binary:logistic", n_estimators=2000, learning_rate=0.03,
                  max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=0.5, reg_lambda=1.0,
                  random_state=SEED, n_jobs=-1, eval_metric="auc", early_stopping_rounds=100)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def run_cat():
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=2000,
                                learning_rate=0.03, depth=5, l2_leaf_reg=8.0,
                                random_seed=SEED, thread_count=-1, verbose=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=150, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def run_lgb_tuned(seed):
    import lightgbm as lgb
    best_params = dict(
        objective="binary", metric="auc", n_estimators=2000,
        learning_rate=0.050253069663926536, num_leaves=3, max_depth=4,
        min_child_samples=60, subsample=0.8995656675058548,
        colsample_bytree=0.7246802411328122, reg_alpha=0.06815791273889091,
        reg_lambda=0.0011513245661312597,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = lgb.LGBMClassifier(**best_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


t0 = time.time()
for name, fn, args in [
    ("LGB_orig", run_lgb_orig, ()),
    ("XGB", run_xgb, ()),
    ("CAT", run_cat, ()),
    ("LGB_tuned", run_lgb_tuned, (42,)),
    ("LGB_tuned_seed2024", run_lgb_tuned, (2024,)),
]:
    tt = time.time()
    oof, pred = fn(*args)
    score = roc_auc_score(y, oof)
    results[name] = {"oof": oof, "pred": pred, "score": score}
    print(f"{name}: OOF AUC={score:.6f}  ({time.time()-tt:.1f}s)")
print(f"Total training time: {time.time()-t0:.1f}s")

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

print(f"\nBest blend weights {dict(zip(names, best_w))}  OOF AUC={best_score:.6f}")
blend_oof = oofs @ np.array(best_w)
blend_pred = preds @ np.array(best_w)
final_score = roc_auc_score(y, blend_oof)
print(f"Final blend OOF AUC={final_score:.6f}  (prior best: 0.832925)")

PRIOR_BEST = 0.832925
sub_name = None
if final_score > PRIOR_BEST:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_name = f"sub_blend_{final_score:.5f}_{ts}.csv"
    sub_path = f"{SUB}/{sub_name}"
    sub = pd.DataFrame({ID: test[ID], TARGET: blend_pred})
    sub.to_csv(sub_path, index=False)
    print(f"NEW BEST -> wrote submission: {sub_path}")
else:
    print("Not an improvement over prior best; no submission written.")

base_models = [{"name": n, "oof_auc": round(float(results[n]["score"]), 6)} for n in names]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="5-way blend: LGB_orig+XGB+CAT+LGB_tuned+LGB_tuned_seedbag",
    metric="roc_auc",
    direction="maximize",
    score=final_score,
    cv={"strategy": "StratifiedKFold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=base_models,
    ensemble={"method": "grid_weight_search", "weights": dict(zip(names, [float(w) for w in best_w]))},
    submission=sub_name,
    notes=("Round 2 of Phase B iteration: added round-1 Optuna-tuned LGB (fold-fixed CV AUC "
           "objective, exp #4 solo 0.837305) to the exp #3 pool (LGB_orig/XGB/CAT unchanged "
           "params) rather than replacing, plus a seed=2024 bag of the same tuned params "
           "(seed bagging is the cheap residual gain after tuning per experience.md). "
           f"Weight search result: {dict(zip(names, [round(float(w),3) for w in best_w]))}."),
)
print(f"Logged experiment #{exp_id}")
