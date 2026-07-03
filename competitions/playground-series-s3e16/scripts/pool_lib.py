"""Shared model-pool library for s3e16 iteration rounds.

Each named member is trained once (5-fold OOF + test pred) and cached to
disk (scripts/cache/<name>.npz) so later rounds don't retrain earlier
members. Blend weight search is a generic simplex grid search over any
number of members (step=0.1), which is exhaustive-cheap since MAE eval is
just a numpy dot product.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

COMP = "competitions/playground-series-s3e16"
DATA = f"{COMP}/data"
CACHE = f"{COMP}/scripts/cache"
TARGET, ID = "Age", "id"
N_SPLITS, SEED = 5, 42
os.makedirs(CACHE, exist_ok=True)

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_h_med = _train.loc[_train["Height"] > 0, "Height"].median()
Xtr_full = build_features(_train, height_median=_h_med)
Xte_full = build_features(_test, height_median=_h_med)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = _train[TARGET].to_numpy(np.float64)
Xtest = Xte_full[FEATS].to_numpy(np.float32)
test_ids = _test[ID]

ybin = np.where(y >= 20, 20, y).astype(int)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, ybin))


def mae(o):
    return mean_absolute_error(y, o)


def cache_path(name):
    return f"{CACHE}/{name}.npz"


def load_cached(name):
    p = cache_path(name)
    if os.path.exists(p):
        d = np.load(p)
        return d["oof"], d["pred"], float(d["mae"]), float(d["time_s"])
    return None


def save_cache(name, oof, pred, mae_val, dt):
    np.savez(cache_path(name), oof=oof, pred=pred, mae=mae_val, time_s=dt)


def get_or_train(name, train_fn, force=False):
    if not force:
        cached = load_cached(name)
        if cached is not None:
            oof, pred, m, dt = cached
            print(f"[{name}] cached OOF MAE={m:.5f} (round={mae(np.round(oof)):.5f}) [{dt:.0f}s]")
            return oof, pred, m, dt
    t0 = time.time()
    oof, pred = train_fn()
    dt = time.time() - t0
    m = mae(oof)
    print(f"[{name}] OOF MAE={m:.5f} (round={mae(np.round(oof)):.5f}) [{dt:.0f}s]")
    save_cache(name, oof, pred, m, dt)
    return oof, pred, m, dt


# ---------------- base members (identical params to scripts/train.py exp#1) ----------------

def train_lgb(seed=SEED, params_override=None):
    import lightgbm as lgb
    params = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                  learning_rate=0.02, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=seed, n_jobs=-1, verbose=-1)
    if params_override:
        params.update(params_override)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def train_lgb_tweedie(seed=SEED, tweedie_variance_power=1.3):
    import lightgbm as lgb
    params = dict(objective="tweedie", tweedie_variance_power=tweedie_variance_power,
                  metric="mae", n_estimators=3000, learning_rate=0.02, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=seed, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def train_xgb(seed=SEED):
    import xgboost as xgb
    params = dict(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.02,
                  max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=seed, n_jobs=-1,
                  eval_metric="mae", early_stopping_rounds=150)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def train_cat(seed=SEED):
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", iterations=4000,
                              learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                              random_seed=seed, thread_count=-1, verbose=False,
                              allow_writing_files=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


# ---------------- generic simplex weight search (N members, step=0.1) ----------------

def weight_search(oof_stack, step=0.1):
    """oof_stack: (n_samples, n_models). Returns (best_weights, best_mae)."""
    n = oof_stack.shape[1]
    units = round(1.0 / step)
    best_w, best_s = None, 1e9
    for combo in itertools.product(range(units + 1), repeat=n - 1):
        if sum(combo) > units:
            continue
        last = units - sum(combo)
        w = np.array(list(combo) + [last]) * step
        s = mae(oof_stack @ w)
        if s < best_s:
            best_s, best_w = s, w
    return best_w, best_s


def write_submission(pred, use_round, stamp_prefix="sub_blend"):
    from datetime import datetime
    sub_pred = np.round(pred) if use_round else pred
    sub_pred = np.clip(sub_pred, 1, None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame({ID: test_ids, TARGET: sub_pred})
    path = f"{COMP}/submissions/{stamp_prefix}_{stamp}.csv"
    sub.to_csv(path, index=False)
    return path
