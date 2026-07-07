"""tree_search/eval_s4e1.py — per-competition evaluator for playground-series-s4e1
(Bank Churn, binary classification, ROC-AUC metric, maximize-better).

Tier4 harness_v3 driver support. Same two-node-kind schema as eval_s3e3.py/eval_s3e7.py:

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...}, features:{drop:[...]}}
     Trains one classifier with the SAME canonical 5-fold StratifiedKFold(Exited,
     shuffle=True, seed=42) as scripts/04_train_blend.py / scripts/05_iterate.py (via
     scripts/features.py's make_folds — imported, not re-implemented, so tiers 2-4 are
     guaranteed to share folds byte-for-byte). Cheap on this 165k-row / 28-feature
     dataset (~10-30s per solo node). If `node_id` is given, OOF/test predictions are
     cached to tree_search/cache_s4e1/solo_<node_id>.npz via harness_v2.cache_oof so
     blend nodes never retrain.

  2. blend — {kind:"blend", members:[<solo node id>, ...], weight_search:"dirichlet"}
     Loads cached member OOFs and weight-searches (dispatches to harness_v2.eval_blend)
     for the AUC-maximizing combination. No retraining.

SIGN CONVENTION: harness.py/harness_v2.py/harness_v3.py assume lower-is-better scores;
AUC is maximize-better, so every score handed to add_root/add_node is `-AUC`.
`result["auc"]` carries the real, human-readable, higher-is-better AUC.

Reuses competitions/playground-series-s4e1/scripts/features.py UNMODIFIED (28-feature
Feb-2026 set + the fold-safe surname-TE correction) so scores are directly comparable to
tier2 (scripts/04_train_blend.py) and tier3 (scripts/05_iterate.py) numbers.
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s4e1")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _HERE)
from features import make_folds, build_all, TARGET, ID  # noqa: E402
import harness_v2 as hv2  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s4e1")
os.makedirs(CACHE_DIR, exist_ok=True)
N_SPLITS, SEED = 5, 42

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = _train[TARGET].to_numpy(np.int64)
_FOLDS = make_folds(_y)  # canonical folds -- IDENTICAL to tier2/tier3
_Xtr_full, _Xte_full, ALL_FEATURES = build_all(_train, _test, _y, _FOLDS)


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# feature selection
# ---------------------------------------------------------------------------
def get_feature_frame(drop=None):
    drop = set(drop or [])
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {unknown}")
    feats = [c for c in ALL_FEATURES if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    return _Xtr_full[feats].copy(), _Xte_full[feats].copy(), feats


# ---------------------------------------------------------------------------
# solo-model runners (mirrors scripts/04_train_blend.py / 05_iterate.py exactly)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              max_depth=-1, min_child_samples=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=-1, seed=SEED)
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        dtr = lgb.Dataset(X[tr], label=_y[tr])
        dva = lgb.Dataset(X[va], label=_y[va], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xtest):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.05,
              max_depth=6, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
              random_state=SEED, n_jobs=-1, tree_method="hist")
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 50)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest):
    from catboost import CatBoostClassifier
    p = dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05, depth=7,
              l2_leaf_reg=3.0, random_seed=SEED, verbose=False, allow_writing_files=False,
              thread_count=20)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 2500))
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostClassifier(iterations=n_est, **p)
        m.fit(X[tr], _y[tr], eval_set=(X[va], _y[va]), early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    Xdf, Xtestdf, feats = get_feature_frame(drop)
    X, Xtest = Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32)

    if model == "lgb":
        oof, pred = _run_lgb(params, X, Xtest)
    elif model == "xgb":
        oof, pred = _run_xgb(params, X, Xtest)
    elif model == "cat":
        oof, pred = _run_cat(params, X, Xtest)
    else:
        raise ValueError(f"unknown model type {model!r}")

    score = auc(_y, oof)
    return oof, pred, score, feats


# ---------------------------------------------------------------------------
# ensemble (blend) node
# ---------------------------------------------------------------------------
def _neg_auc(vec):
    return -auc(_y, vec)


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    best_w, best_neg, oofs = hv2.eval_blend(CACHE_DIR, members, _neg_auc, weight_search=method)
    best_score = -best_neg
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, auc=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 180) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or any
    exception is captured as status="failed" so the tree-search loop can keep going.

    score is `-AUC` (harness sign convention: lower-is-better). result["auc"] is the
    real, human-readable, higher-is-better AUC -- use THAT for all reporting/printouts."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, feats = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, auc=score)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "auc": round(score, 6)}
            return dict(status="evaluated", score=round(-score, 6), wall_s=round(wall, 1),
                        result=result, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            wall = time.time() - t0
            return dict(status="evaluated", score=round(-score, 6), wall_s=round(wall, 1),
                        result=result, error=None)
        else:
            raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        wall = time.time() - t0
        return dict(status="failed", score=None, wall_s=round(wall, 1), result=None,
                    error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - evaluator must never crash the search loop
        wall = time.time() - t0
        return dict(status="failed", score=None, wall_s=round(wall, 1), result=None,
                    error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
