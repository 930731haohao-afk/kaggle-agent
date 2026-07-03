"""Self-improvement Round 1: duplicate-aware target smoothing (fold-safe).

Hypothesis: EDA/STATUS.md diagnosed that ~56% of train feature-rows (raw 8 columns,
excluding target) are exact duplicates of another train row with a *different*
measured Strength -> irreducible label noise ceiling that punishes high-capacity
models. Directly attacking this noise (rather than just regularizing around it):
for each training fold, group rows by their raw-feature tuple (the same duplicate
definition used in EDA) and replace each row's training target with the *group mean
Strength computed using only rows in that training fold* (fold-safe: never uses
validation-fold targets). Validation targets and the CV metric itself are left
untouched (still scored against the real, un-smoothed Strength) so the CV number
stays comparable to experiment #2/#4 (12.07347) under the same 5-fold
StratifiedKFold(strength decile, seed=42) scheme.

If a group has only 1 member in the training fold, its "smoothed" target equals
itself (no-op) -- smoothing only kicks in for rows that actually collide with a
training-fold duplicate.

Everything else (features, hyperparameters, folds, blend weight search) is
identical to scripts/train.py so this is a single, isolated change per the
iteration protocol.
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
from features import build_features, feature_columns, RAW_COLS  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py")
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

# duplicate-group id from RAW feature columns (same definition EDA used for the
# "56% duplicate feature-rows" diagnosis) -- deterministic from X only, no leakage.
group_id = train[RAW_COLS].astype(np.float64).apply(tuple, axis=1)
group_id = pd.Series(group_id).astype("category").cat.codes.to_numpy()
n_groups = len(np.unique(group_id))
dup_frac = 1 - n_groups / len(group_id)
print(f"n_groups={n_groups}  dup_frac={dup_frac:.4f} (expect ~0.56 per EDA)")

# stratify on Strength deciles for stable folds on small data (IDENTICAL to train.py)
ybin = pd.qcut(y, 10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, ybin))


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def smoothed_y_train(tr_idx):
    """Fold-safe: group-mean computed ONLY from rows in this training fold."""
    g_tr = group_id[tr_idx]
    y_tr = y[tr_idx]
    s = pd.Series(y_tr).groupby(g_tr).transform("mean")
    return s.to_numpy()


def run_lgb():
    import lightgbm as lgb
    params = dict(objective="regression", metric="rmse", n_estimators=3000,
                  learning_rate=0.02, num_leaves=15, max_depth=5, min_child_samples=25,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=2.0, reg_lambda=4.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        y_tr_smooth = smoothed_y_train(tr)
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y_tr_smooth, eval_set=[(X[va], y[va])],
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
        y_tr_smooth = smoothed_y_train(tr)
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y_tr_smooth, eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [XGB] fold{f} RMSE={rmse(y[va], oof[va]):.4f} best_iter={m.best_iteration}")
    return oof, pred


def run_cat():
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        y_tr_smooth = smoothed_y_train(tr)
        m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=4000,
                              learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                              random_seed=SEED, thread_count=-1, verbose=False,
                              allow_writing_files=False)
        m.fit(Pool(X[tr], y_tr_smooth), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [CAT] fold{f} RMSE={rmse(y[va], oof[va]):.4f} best_iter={m.get_best_iteration()}")
    return oof, pred


results = {}
for name, fn in [("LGB", run_lgb), ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"\n=== {name} (dup-smoothed target) ===")
    oof, pred = fn()
    dt = time.time() - t0
    score = rmse(y, oof)
    print(f"{name} OOF RMSE = {score:.5f}  [{dt:.0f}s]")
    results[name] = dict(oof=oof, pred=pred, rmse=score, time=dt)

# --- weight search (coarse grid over simplex), identical method to train.py ---
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
print(f"Final OOF RMSE = {rmse(y, final_oof):.5f}   (baseline to beat: 12.07347)")

sub_pred = np.clip(final_pred, 0, None)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
np.savez(f"{COMP}/scripts/_round1_oof_pred.npz", oof=final_oof, pred=sub_pred, score=best_s)

# --- log experiment (mandatory v2 schema) ---
base_models = [dict(model=n, rmse=round(results[n]["rmse"], 5),
                     time_s=round(results[n]["time"], 1)) for n in names]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB+XGB+CatBoost weighted blend (dup-group fold-safe target smoothing)",
    metric="rmse",
    direction="minimize",
    score=best_s,
    cv=dict(strategy=f"{N_SPLITS}fold_stratified_strength_decile", seed=SEED),
    features=FEATS,
    base_models=base_models,
    ensemble=dict(weights=dict(zip(names, [round(float(x), 2) for x in best_w])),
                  method="oof_weight_search_grid0.05"),
    submission=None,
    notes=(f"Round 1 self-improvement iteration: fold-safe duplicate-group target "
           f"smoothing (group by raw 8-feature tuple = same def. as EDA's 56% "
           f"duplicate-row diagnosis; training target per fold replaced with "
           f"group-mean Strength computed ONLY from that fold's training rows, "
           f"validation targets/CV metric untouched). n_groups={n_groups} "
           f"dup_frac={dup_frac:.4f}. Same features/hyperparams/folds as exp#2/#4 "
           f"(12.07347) -- isolated single change. Result: {best_s:.5f} "
           f"({'IMPROVED' if best_s < 12.073474 else 'no improvement'})."),
)
print(f"Logged experiment #{exp_id} to {COMP}/experiments.json")
