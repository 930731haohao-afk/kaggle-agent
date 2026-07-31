"""tree_search/eval_s3e9.py — per-competition evaluator for playground-series-s3e9
(Concrete Compressive Strength, RMSE, minimize).

config schema (JSON-serializable, matches tree_search.harness Node.config)::

    {
      "model": "lgb" | "xgb" | "cat",
      "params": {...single-model hyperparams, keys match the library's ctor kwargs...},
      "features": {"drop": [<feature name>, ...]},   # subtract from the full 22-feature set
      "postprocess": {"clip_min": 0, "clip_max": null, "clip_oof": false}
    }

Reuses the EXACT same feature set (`competitions/playground-series-s3e9/scripts/features.py`,
unmodified) and the EXACT same CV scheme as `scripts/train.py` (5-fold StratifiedKFold on
Strength deciles, seed=42) so scores here are directly comparable to STATUS.md's numbers
(root LGB solo ~12.11061, CatBoost solo ~12.07459, XGB solo ~12.12086).

Single-model only (no blending) — that's the point of the tree-search prototype: each
node is one cheap, cheap-to-evaluate config, and mutations climb the tree instead of a
human-scripted linear blend pipeline.
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e9")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
from features import build_features, feature_columns  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
TARGET, ID = "Strength", "id"
N_SPLITS, SEED = 5, 42

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_Xtr_full = build_features(_train)
_Xte_full = build_features(_test)
ALL_FEATURES = feature_columns(_Xtr_full)
_y = _train[TARGET].to_numpy(np.float64)

_ybin = pd.qcut(_y, 10, labels=False, duplicates="drop")
_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(np.zeros(len(_y)), _ybin))  # identical to scripts/train.py


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def get_feature_matrix(drop=None):
    drop = set(drop or [])
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {unknown}")
    feats = [c for c in ALL_FEATURES if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    X = _Xtr_full[feats].to_numpy(np.float32)
    Xt = _Xte_full[feats].to_numpy(np.float32)
    return X, Xt, feats


def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    p = dict(objective="regression", metric="rmse", n_jobs=-1, verbose=-1, random_state=SEED)
    p.update(params)
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 150)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = lgb.LGBMRegressor(n_estimators=n_est, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])],
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xtest):
    import xgboost as xgb
    p = dict(objective="reg:squarederror", n_jobs=-1, random_state=SEED, eval_metric="rmse")
    p.update(params)
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 150)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBRegressor(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])], verbose=False)
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest):
    from catboost import CatBoostRegressor, Pool
    p = dict(loss_function="RMSE", eval_metric="RMSE", random_seed=SEED,
             thread_count=-1, verbose=False, allow_writing_files=False)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 4000))
    esr = p.pop("early_stopping_rounds", 200)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostRegressor(iterations=n_est, **p)
        m.fit(Pool(X[tr], _y[tr]), eval_set=Pool(X[va], _y[va]),
              early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


RUNNERS = {"lgb": _run_lgb, "xgb": _run_xgb, "cat": _run_cat}


def evaluate(config: dict, timeout_s: int = 90) -> dict:
    """config -> {status, score, wall_s, n_feats, error}. Never raises: a timeout or
    any exception is captured as status="failed" so the tree-search loop can keep going."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        drop = (config.get("features") or {}).get("drop", [])
        X, Xtest, feats = get_feature_matrix(drop)
        model = config["model"]
        if model not in RUNNERS:
            raise ValueError(f"unknown model type {model!r}")
        params = dict(config.get("params") or {})
        oof, _pred = RUNNERS[model](params, X, Xtest)

        pp = config.get("postprocess") or {}
        clip_min = pp.get("clip_min", 0)
        clip_max = pp.get("clip_max", None)
        score_oof = np.clip(oof, clip_min, clip_max) if pp.get("clip_oof", False) else oof
        score = rmse(_y, score_oof)

        wall = time.time() - t0
        return dict(status="evaluated", score=round(score, 5), wall_s=round(wall, 1),
                     n_feats=len(feats), error=None)
    except EvalTimeout:
        wall = time.time() - t0
        return dict(status="failed", score=None, wall_s=round(wall, 1), n_feats=None,
                     error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - evaluator must never crash the search loop
        wall = time.time() - t0
        return dict(status="failed", score=None, wall_s=round(wall, 1), n_feats=None,
                     error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
