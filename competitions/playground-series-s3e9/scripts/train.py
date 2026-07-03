"""Train LGB/XGB/CatBoost with StratifiedKFold OOF, blend, and write submission.

Metric: RMSE (minimize). Competition: playground-series-s3e9 (Concrete Strength).
CV: 5-fold StratifiedKFold on binned Strength (small data, ~5.4k rows -> stable folds).

Baseline to beat (experiment #1, generic untuned blend): RMSE 12.54287, where the
weight search gave CatBoost 100% of the blend (LGB/XGB contributed 0). EDA showed why:
n_train=5407 with only 8 raw features, ~56% duplicate feature-rows with noisy/differing
Strength labels -> untuned, deep, high-capacity LGB/XGB memorize noise and overfit CV
folds harder than CatBoost's ordered-boosting regularization. This script explicitly
regularizes LGB/XGB (shallower leaves, stronger L1/L2, early stopping) and adds
engineered features (log_age, water/binder ratios) to give every booster more signal.
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e9"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "Strength", "id"
N_SPLITS, SEED = 5, 42

os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

Xtr_full = build_features(train)
Xte_full = build_features(test)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.float64)
Xtest = Xte_full[FEATS].to_numpy(np.float32)
print(f"n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")

# stratify on Strength deciles for stable folds on small data
ybin = pd.qcut(y, 10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, ybin))


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def run_lgb():
    import lightgbm as lgb
    params = dict(objective="regression", metric="rmse", n_estimators=3000,
                  learning_rate=0.02, num_leaves=15, max_depth=5, min_child_samples=25,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=2.0, reg_lambda=4.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [LGB] fold{f} RMSE={rmse(y[va], oof[va]):.4f} best_iter={m.best_iteration_}")
    return oof, pred


def run_xgb():
    import xgboost as xgb
    params = dict(objective="reg:squarederror", n_estimators=3000, learning_rate=0.02,
                  max_depth=4, min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=2.0, reg_lambda=4.0, random_state=SEED, n_jobs=-1,
                  eval_metric="rmse", early_stopping_rounds=150)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [XGB] fold{f} RMSE={rmse(y[va], oof[va]):.4f} best_iter={m.best_iteration}")
    return oof, pred


def run_cat():
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=4000,
                              learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                              random_seed=SEED, thread_count=-1, verbose=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [CAT] fold{f} RMSE={rmse(y[va], oof[va]):.4f} best_iter={m.get_best_iteration()}")
    return oof, pred


results = {}
for name, fn in [("LGB", run_lgb), ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"\n=== {name} ===")
    oof, pred = fn()
    dt = time.time() - t0
    score = rmse(y, oof)
    print(f"{name} OOF RMSE = {score:.5f}  [{dt:.0f}s]")
    results[name] = dict(oof=oof, pred=pred, rmse=score, time=dt)

# --- simple average blend ---
blend_oof = np.mean([results[n]["oof"] for n in results], axis=0)
blend_pred = np.mean([results[n]["pred"] for n in results], axis=0)
print(f"\nBLEND(mean) OOF RMSE = {rmse(y, blend_oof):.5f}")

# --- weight search (coarse grid over simplex) ---
best_w, best_s = None, 1e9
names = list(results)
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
for w0 in np.arange(0, 1.01, 0.05):
    for w1 in np.arange(0, 1.01 - w0, 0.05):
        w2 = 1 - w0 - w1
        if w2 < -1e-9:
            continue
        s = rmse(y, oofs @ np.array([w0, w1, w2]))
        if s < best_s:
            best_s, best_w = s, (w0, w1, w2)
print(f"BEST weighted blend {dict(zip(names, np.round(best_w, 2)))} OOF RMSE = {best_s:.5f}")
w = np.array(best_w)
final_oof = oofs @ w
final_pred = np.stack([results[n]["pred"] for n in names], axis=1) @ w
print(f"Final OOF RMSE = {rmse(y, final_oof):.5f}")

sub_pred = np.clip(final_pred, 0, None)

# --- write submission ---
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({ID: test[ID], TARGET: sub_pred})
sub_path = f"{SUB}/sub_blend_{best_s:.5f}_{stamp}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote {sub_path}  shape={sub.shape}")
print(sub.head())
print(sub[TARGET].describe())

# --- log experiment (mandatory v2 schema) ---
base_models = [dict(model=n, rmse=round(results[n]["rmse"], 5),
                     time_s=round(results[n]["time"], 1)) for n in names]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB+XGB+CatBoost weighted blend (regularized, engineered features)",
    metric="rmse",
    direction="minimize",
    score=best_s,
    cv=dict(strategy=f"{N_SPLITS}fold_stratified_strength_decile", seed=SEED),
    features=FEATS,
    base_models=base_models,
    ensemble=dict(weights=dict(zip(names, [round(float(x), 2) for x in best_w])),
                  method="oof_weight_search_grid0.05"),
    submission=os.path.basename(sub_path),
    notes=(f"Regularized LGB(num_leaves=15,depth5,L1=2/L2=4)/XGB(depth4,L1=2/L2=4) + "
           f"CatBoost(depth6,l2=6) with 8 engineered features (log_age, water/binder "
           f"ratios) vs baseline exp#1 (generic untuned blend, RMSE 12.54287, CatBoost "
           f"100% weight)."),
)
print(f"Logged experiment #{exp_id} to {COMP}/experiments.json")
