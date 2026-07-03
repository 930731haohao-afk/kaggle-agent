"""Train LGB/XGB/CatBoost with KFold OOF, weight-search blend, and write submission.

Metric: MAE (minimize). All base models use L1/MAE objective directly.
CV: 5-fold KFold (shuffle) -- target is continuous with no groups/time structure and
train/test distributions match closely (see scripts/eda.py), so plain i.i.d. KFold is used
(no stratification needed, unlike s3e16's discrete Age).

Post-processing experiment (per project learning from s3e16): checks whether clipping to the
train yield range, or snapping predictions to the nearest *observed train yield value*, improves
OOF MAE. EDA found yield has no coarse rounding grid (776/15289 unique values, not integer steps),
so this is an empirical check rather than an assumed win.
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py"))
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e14"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "yield", "id"
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

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(X))


def run_lgb():
    import lightgbm as lgb
    params = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                  learning_rate=0.02, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [LGB] fold{f} MAE={mean_absolute_error(y[va], oof[va]):.4f} best_iter={m.best_iteration_}")
    return oof, pred


def run_xgb():
    import xgboost as xgb
    params = dict(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.02,
                  max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1,
                  eval_metric="mae", early_stopping_rounds=150)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [XGB] fold{f} MAE={mean_absolute_error(y[va], oof[va]):.4f} best_iter={m.best_iteration}")
    return oof, pred


def run_cat():
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", iterations=4000,
                              learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                              random_seed=SEED, thread_count=-1, verbose=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [CAT] fold{f} MAE={mean_absolute_error(y[va], oof[va]):.4f} best_iter={m.get_best_iteration()}")
    return oof, pred


def mae(o):
    return mean_absolute_error(y, o)


results = {}
for name, fn in [("LGB", run_lgb), ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"\n=== {name} ===")
    oof, pred = fn()
    dt = time.time() - t0
    print(f"{name} OOF MAE = {mae(oof):.5f}  [{dt:.0f}s]")
    results[name] = dict(oof=oof, pred=pred, mae=mae(oof), time=dt)

# --- weight search (coarse grid over simplex) ---
best_w, best_s = None, 1e9
names = list(results)
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
for w0 in np.arange(0, 1.01, 0.05):
    for w1 in np.arange(0, 1.01 - w0, 0.05):
        w2 = 1 - w0 - w1
        if w2 < -1e-9:
            continue
        s = mae(oofs @ np.array([w0, w1, w2]))
        if s < best_s:
            best_s, best_w = s, (w0, w1, w2)
print(f"\nBEST weighted blend {dict(zip(names, np.round(best_w, 2)))} OOF MAE = {best_s:.5f}")
w = np.array(best_w)
final_oof = oofs @ w
final_pred = np.stack([results[n]["pred"] for n in names], axis=1) @ w

# --- post-processing experiments (per s3e16 learning: check discrete-grid snapping) ---
train_min, train_max = y.min(), y.max()
clip_oof_mae = mae(np.clip(final_oof, train_min, train_max))

train_sorted = np.sort(np.unique(y))


def snap_to_grid(preds, grid):
    idx = np.searchsorted(grid, preds)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    return np.where(np.abs(preds - left) <= np.abs(preds - right), left, right)


snap_oof = snap_to_grid(final_oof, train_sorted)
snap_oof_mae = mae(snap_oof)

print(f"Raw blend OOF MAE      = {mae(final_oof):.5f}")
print(f"Clipped to [min,max]   = {clip_oof_mae:.5f}")
print(f"Snapped to observed grid = {snap_oof_mae:.5f}")

candidates = {"none": (final_oof, mae(final_oof)),
              "clip": (np.clip(final_oof, train_min, train_max), clip_oof_mae),
              "snap_to_grid": (snap_oof, snap_oof_mae)}
best_pp = min(candidates, key=lambda k: candidates[k][1])
best_oof, best_pp_mae = candidates[best_pp]
print(f"Best post-processing: '{best_pp}' -> OOF MAE = {best_pp_mae:.5f}")

if best_pp == "clip":
    sub_pred = np.clip(final_pred, train_min, train_max)
elif best_pp == "snap_to_grid":
    sub_pred = snap_to_grid(final_pred, train_sorted)
else:
    sub_pred = final_pred

# --- write submission ---
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({ID: test[ID], TARGET: sub_pred})
sub_path = f"{SUB}/sub_blend_{best_pp_mae:.5f}_{stamp}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote {sub_path}  shape={sub.shape}")
print(sub.head())
print(sub[TARGET].describe())

# --- log experiment (mandatory v2 schema) ---
base_models = [{"name": n, "score": round(float(results[n]["mae"]), 5)} for n in names]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB+XGB+CAT weighted blend (KFold, MAE objective, feature-engineered v1)",
    metric="mae",
    direction="minimize",
    score=float(best_pp_mae),
    cv={"scheme": "5fold_kfold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=base_models,
    ensemble={"weights": {n: round(float(x), 3) for n, x in zip(names, best_w)},
              "score": round(float(best_s), 5)},
    postprocess=[f"selected='{best_pp}' (candidates: none={mae(final_oof):.5f}, "
                 f"clip={clip_oof_mae:.5f}, snap_to_grid={snap_oof_mae:.5f})"],
    submission=os.path.basename(sub_path),
    notes="v1 feature set: fruit-biology interactions, condensed temp feature, pollinator index, "
          "rain intensity, log clonesize. 16 raw + 11 engineered = 27 features.",
)
print(f"Logged experiment #{exp_id} to experiments.json")
