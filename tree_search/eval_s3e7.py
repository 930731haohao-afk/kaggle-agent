"""tree_search/eval_s3e7.py — per-competition evaluator for playground-series-s3e7
(Hotel Reservation Cancellation, binary classification, ROC-AUC metric, maximize-better).

Phase D-3 (3rd harness_v2 sweep comp, HARDEST case per the sweep brief: linear iteration
squeezed only +<score> total over its generic baseline, AUC <score>, and is suspected to
be near a genuine ceiling for this feature set / model family). Same two-node-kind schema
as eval_s3e3.py/eval_s3e14.py/eval_s3e5.py:

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...},
             features:{drop:[...]}, want_importance: bool (lgb only)}
     Trains one classifier with 5-fold StratifiedKFold(booking_status, shuffle, seed=42) —
     EXACT same CV as competitions/playground-series-s3e7/scripts/{optuna_lgb,blend_refine}.py.
     Reuses scripts/features.py's build_features/feature_columns UNMODIFIED (variant=
     "trimmed", the iter2/reflexion-winning 25-feature set: 17 raw + 8 engineered — the
     14-engineered "full" variant REGRESSED below baseline per STATUS.md and must never be
     re-added verbatim). ~25-50s per fold-set on this 42k-row dataset. If `node_id` is
     given, its OOF/test prediction vector is cached via harness_v2.cache_oof to
     tree_search/cache_s3e7/solo_<node_id>.npz so blend nodes never retrain.

     `want_importance=True` (lgb only) additionally returns each of the 8 ENGINEERED
     features' average (over folds) LightGBM gain-importance in `result["importance"]` —
     used exactly once, on the root, to pick real weakest-importance engineered features
     for a feature-PRUNING mutation direction (s3e3's winning lever per the Phase D-3
     brief) instead of guessing from EDA correlations alone.

  2. blend — {kind:"blend", members:[<solo node id>, ...],
             weight_search:"dirichlet", space:"prob"|"rank"}
     Loads each member's cached OOF vector via harness_v2.load_oof, weight-searches for
     the blend that maximizes ROC-AUC. "prob" dispatches to harness_v2.eval_blend
     (recommendation #1's generic ensemble machinery); "rank" mirrors STATUS.md exp #6's
     rank-average check (which came out worse than prob by a noise-level -<score> in the
     linear run — re-verify under harness_v2's blend seeds, don't assume it stays worse).

SIGN CONVENTION: harness.py/harness_v2.py assume lower-is-better scores; AUC is
maximize-better, so every score this module hands to add_root/add_node is `-AUC`.
`result["auc"]` (and all prints in run_s3e7.py) carry the real, human-readable AUC.

CatBoost: allow_writing_files=False and thread_count explicit per the s3e11 lesson in
knowledge/experience.md ("反面教訓" — CatBoost's default catboost_info file logging
stalled a long-running sandboxed process for 33 minutes with zero output).
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e7")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _HERE)
import stage2_inputs  # noqa: E402
stage2_inputs.require_module(_SCRIPTS_DIR, "features", comp="playground-series-s3e7",
                             exposes=['build_features', 'feature_columns'])
from features import build_features, feature_columns  # noqa: E402
import harness_v2 as hv2  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s3e7")
TARGET, ID = "booking_status", "id"
N_SPLITS, SEED = 5, 42

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = _train[TARGET].to_numpy(np.int64)

# --- trimmed (iter2/reflexion-winning) feature set, features.py unmodified ---
_Xtr_full = build_features(_train)
_Xte_full = build_features(_test)
ALL_FEATURES = feature_columns(_Xtr_full, variant="trimmed")
RAW_FEATURES = [c for c in ALL_FEATURES if c in
                ["no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
                 "type_of_meal_plan", "required_car_parking_space", "room_type_reserved",
                 "lead_time", "arrival_year", "arrival_month", "arrival_date",
                 "market_segment_type", "repeated_guest", "no_of_previous_cancellations",
                 "no_of_previous_bookings_not_canceled", "avg_price_per_room",
                 "no_of_special_requests"]]
ENGINEERED_FEATURES = [c for c in ALL_FEATURES if c not in RAW_FEATURES]  # the 8 trimmed ones

# MANDATORY: identical fold assignment to scripts/optuna_lgb.py & scripts/blend_refine.py
# (folds depend only on _y / n_samples).
_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(np.zeros(len(_y)), _y))


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
# solo-model runners (mirrors scripts/optuna_lgb.py / scripts/blend_refine.py exactly)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest, feats, want_importance=False):
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", n_jobs=-1, verbose=-1, random_state=SEED)
    p.update(params)
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 150)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    imp_sum = np.zeros(len(feats)) if want_importance else None
    for tr, va in _FOLDS:
        m = lgb.LGBMClassifier(n_estimators=n_est, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])],
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
        if want_importance:
            imp_sum += m.booster_.feature_importance(importance_type="gain")
    importance = dict(zip(feats, (imp_sum / N_SPLITS).tolist())) if want_importance else None
    return oof, pred, importance


def _run_xgb(params, X, Xtest):
    import xgboost as xgb
    p = dict(objective="binary:logistic", n_jobs=-1, random_state=SEED, eval_metric="auc")
    p.update(params)
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 150)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest):
    from catboost import CatBoostClassifier, Pool
    p = dict(loss_function="Logloss", eval_metric="AUC", random_seed=SEED,
             thread_count=4, verbose=False, allow_writing_files=False)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 4000))
    esr = p.pop("early_stopping_rounds", 200)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostClassifier(iterations=n_est, **p)
        m.fit(Pool(X[tr], _y[tr]), eval_set=Pool(X[va], _y[va]),
              early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    want_importance = bool(config.get("want_importance", False)) and model == "lgb"

    Xdf, Xtestdf, feats = get_feature_frame(drop)
    importance = None
    if model == "lgb":
        oof, pred, importance = _run_lgb(params, Xdf.to_numpy(np.float32),
                                          Xtestdf.to_numpy(np.float32), feats,
                                          want_importance=want_importance)
    elif model == "xgb":
        oof, pred = _run_xgb(params, Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32))
    elif model == "cat":
        oof, pred = _run_cat(params, Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32))
    else:
        raise ValueError(f"unknown model type {model!r}")

    score = auc(_y, oof)
    return oof, pred, score, feats, importance


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over harness_v2-CACHED member OOF vectors, no
# retraining.
# ---------------------------------------------------------------------------
def _neg_auc(vec):
    return -auc(_y, vec)


def _rank_weight_search(oofs, seed=42, k=1500):
    n = oofs.shape[1]
    rng = np.random.default_rng(seed)
    candidates = [np.eye(n)[i] for i in range(n)]
    candidates.append(np.full(n, 1.0 / n))
    candidates += list(rng.dirichlet(np.ones(n), size=k))
    best_w, best_s = None, -1e18
    for w in candidates:
        s = auc(_y, oofs @ w)
        if s > best_s:
            best_s, best_w = s, w
    conc = np.clip(best_w, 1e-3, None) * 200.0
    for w in rng.dirichlet(conc, size=max(k // 3, 50)):
        s = auc(_y, oofs @ w)
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s


def evaluate_blend(config):
    from scipy.stats import rankdata
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    space = config.get("space", "prob")

    if space == "prob":
        best_w, best_neg, oofs = hv2.eval_blend(CACHE_DIR, members, _neg_auc, weight_search=method)
        best_score = -best_neg
    elif space == "rank":
        oofs_raw = np.stack([hv2.load_oof(CACHE_DIR, m) for m in members], axis=1)
        n = len(_y)
        oofs = np.stack([rankdata(oofs_raw[:, i]) / n for i in range(oofs_raw.shape[1])], axis=1)
        if method != "dirichlet":
            raise ValueError(f"rank-space blend only supports weight_search='dirichlet', got {method!r}")
        best_w, best_score = _rank_weight_search(oofs)
    else:
        raise ValueError(f"unknown blend space {space!r}")

    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, space=space, auc=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 200) -> dict:
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
            oof, pred, score, feats, importance = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, auc=score)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "auc": round(score, 6)}
            if importance is not None:
                result["importance"] = {k: round(v, 3) for k, v in importance.items()}
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
