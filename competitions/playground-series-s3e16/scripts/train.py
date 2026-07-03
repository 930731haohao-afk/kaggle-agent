"""Train LGB/XGB/CatBoost with StratifiedKFold OOF, blend, and write submission.

Metric: MAE (minimize). All base models use L1/MAE objective directly.
CV: 5-fold StratifiedKFold on binned Age (discrete, right-skewed target).
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

COMP = "competitions/playground-series-s3e16"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
EXP = f"{COMP}/experiments.json"
TARGET, ID = "Age", "id"
N_SPLITS, SEED = 5, 42

os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

h_med = train.loc[train["Height"] > 0, "Height"].median()
Xtr_full = build_features(train, height_median=h_med)
Xte_full = build_features(test, height_median=h_med)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.float64)
Xtest = Xte_full[FEATS].to_numpy(np.float32)
print(f"n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")

# stratify on Age, merging rare high ages into one bin for stable folds
ybin = np.where(y >= 20, 20, y).astype(int)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, ybin))


def run_lgb():
    import lightgbm as lgb
    params = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                  learning_rate=0.02, num_leaves=63, min_child_samples=40,
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
    print(f"{name} OOF MAE = {mae(oof):.5f}  (round={mae(np.round(oof)):.5f})  [{dt:.0f}s]")
    results[name] = dict(oof=oof, pred=pred, mae=mae(oof), time=dt)

# --- simple average blend ---
blend_oof = np.mean([results[n]["oof"] for n in results], axis=0)
blend_pred = np.mean([results[n]["pred"] for n in results], axis=0)
print(f"\nBLEND(mean) OOF MAE = {mae(blend_oof):.5f}  (round={mae(np.round(blend_oof)):.5f})")

# --- weight search (coarse grid over simplex) ---
best_w, best_s = None, 1e9
names = list(results)
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
for w0 in np.arange(0, 1.01, 0.1):
    for w1 in np.arange(0, 1.01 - w0, 0.1):
        w2 = 1 - w0 - w1
        if w2 < -1e-9:
            continue
        s = mae(oofs @ np.array([w0, w1, w2]))
        if s < best_s:
            best_s, best_w = s, (w0, w1, w2)
print(f"BEST weighted blend {dict(zip(names, np.round(best_w,2)))} OOF MAE = {best_s:.5f}")
w = np.array(best_w)
final_oof = oofs @ w
final_pred = np.stack([results[n]["pred"] for n in names], axis=1) @ w
print(f"Final OOF MAE = {mae(final_oof):.5f}  round={mae(np.round(final_oof)):.5f}")

# rounding hurts or helps? pick better OOF
use_round = mae(np.round(final_oof)) < mae(final_oof)
sub_pred = np.round(final_pred) if use_round else final_pred
sub_pred = np.clip(sub_pred, 1, None)
print(f"use_round={use_round}")

# --- write submission ---
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({ID: test[ID], TARGET: sub_pred})
sub_path = f"{SUB}/sub_blend_{stamp}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote {sub_path}  shape={sub.shape}")
print(sub.head())
print(sub[TARGET].describe())

# --- log experiments ---
exp_records = []
for n in names:
    exp_records.append(dict(model=n, oof_mae=round(results[n]["mae"], 5),
                            time_s=round(results[n]["time"], 1)))
log = dict(timestamp=datetime.now().isoformat(timespec="seconds"),
           n_features=len(FEATS), features=FEATS, cv=f"{N_SPLITS}fold_stratified_agebin",
           base_models=exp_records,
           blend_weights=dict(zip(names, [round(float(x), 2) for x in best_w])),
           blend_oof_mae=round(best_s, 5), use_round=bool(use_round),
           submission=os.path.basename(sub_path))
history = []
if os.path.exists(EXP):
    history = json.load(open(EXP))
history.append(log)
json.dump(history, open(EXP, "w"), indent=2)
print(f"Logged experiment #{len(history)} to {EXP}")
