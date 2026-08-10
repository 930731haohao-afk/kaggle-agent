"""tree_search/eval_s3e11.py — per-competition evaluator for playground-series-s3e11
(Media Campaign Cost, regression, RMSLE metric, minimize-better).

Phase D-6 (FINAL sweep comp, the largest dataset: 360,336 train / 240,224 test rows).
Two node kinds, same schema convention as eval_s3e1.py/eval_s3e19.py:

  1. solo  — {kind:"solo", model:"lgb"|"cat", params:{...}, features:{drop:[...]}}
     Trains one regressor with 5-fold KFold(shuffle=True, seed=42) on log1p(cost) with
     an RMSE objective — EXACT same CV/target-transform as
     competitions/playground-series-s3e11/scripts/{train,iterate2}.py. RMSLE(y, pred) =
     RMSE(log1p(y), log1p(pred)) so training directly on log1p(cost) optimizes the
     competition metric; predictions are expm1'd + clipped >=0 only INSIDE the scoring
     function (rmsle_from_log), never at the OOF-caching stage — OOF/pred arrays stay in
     LOG1P SPACE throughout, matching scripts/iterate2.py's own weight_search() which
     blends the raw (log-space) model outputs, not expm1'd cost-space outputs. This is
     the one place s3e11 genuinely differs from eval_s3e1.py/eval_s3e19.py's convention
     (those cache cost-scale OOF) — replicating iterate2.py's own blend semantics is what
     makes this evaluator's blend scores byte-comparable to STATUS.md/experiments.json.

     Feature set: scripts/features.py's build_features() (pure function, no import-time
     side effects, safe to import directly — unlike s3e1's script-style features.py)
     gives the 20 non-TE engineered features (base_feature_columns()); `store_te`
     (fold-safe K-fold target encoding of the 111-way store_combo group, computed ONCE
     at module load via the exact fold-safe recipe in scripts/iterate2.py's
     fold_safe_te(), independent of any downstream model hyperparameters) is appended as
     the 21st feature. get_feature_frame(drop=[...]) allows dropping any subset of the
     21 for feature-variant solo nodes.

     ⚠️ Legacy-cache reuse (Phase D-6 budget lever, this run's own addition — no prior
     sweep comp needed this because none had a scale problem): scripts/cache/*.npz
     already holds the linear-iteration's 5-way tuned-CAT-family pool's trained
     OOF/pred/time_s arrays (lgb.npz, cat_orig.npz, cat_tuned.npz,
     cat_tuned_seed2024.npz, cat_tuned_seed7.npz — from iterate2.py's `cached()`
     checkpoint helper). `load_legacy_solo(name, expected_rmsle)` loads one of these
     directly and asserts its RMSLE (computed via THIS module's own rmsle_from_log) is
     digit-for-digit consistent with the historical STATUS.md/experiments.json value —
     this IS the "verify root reproduces digit-for-digit" check the task brief asks for,
     done as a load+recompute (near-instant) rather than a full retrain, so root + all 4
     first-gen pool seeds cost ~0s of training time, freeing the ~10-12 min retrain
     budget for new solo variants and blend-heavy search instead.

  2. blend — {kind:"blend", members:[<solo node id>, ...],
             weight_search:"dirichlet"|"grid_simplex"}
     Loads each member's cached OOF vector (LOG1P SPACE) via harness_v2.load_oof,
     weight-searches for the blend that MINIMIZES rmsle_from_log(y_log, oofs @ w) — same
     semantics as iterate2.py's weight_search(), so the seed_blend() 5-way reproduction
     in run_s3e11.py should land at/near STATUS.md exp #8's <score> (dirichlet search
     vs the linear run's coarser 0.1-step grid, so a small further gain is expected, not
     a regression).

SIGN CONVENTION: harness.py/harness_v2.py assume lower-is-better scores; RMSLE already
IS lower-is-better, so `score` handed to add_root/add_node is the RMSLE itself (no sign
flip needed, same convention as the other RMSE-family comps s3e1/s3e14/s3e19).

CatBoost: allow_writing_files=False and thread_count=4 explicit per the s3e11 lesson in
knowledge/experience.md ("反面教訓" — CatBoost's default catboost_info file logging
stalled iterate.py v1 for 33 minutes with zero output on this exact competition).
LGB: deliberately no n_jobs override (measured oversubscription slowdown on this sandbox
in eval_s3e1.py's note; iterate2.py's own n_jobs=20/thread_count=20 choices are NOT
replicated here for that reason — this only affects wall-clock, not the math, and only
matters for NEWLY trained mutations since root+pool seeds are reused from cache, not
retrained).
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e11")
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402

sys.path.insert(0, os.path.join(_COMP_DIR, "scripts"))
import features as feat_mod  # noqa: E402 -- build_features()/base_feature_columns(), pure functions

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s3e11")
LEGACY_CACHE_DIR = os.path.join(_COMP_DIR, "scripts", "cache")
TARGET, ID = "cost", "id"
N_SPLITS, SEED = 5, 42

# ---------------------------------------------------------------------------
# data + features (loaded once at import time, matches scripts/train.py & iterate2.py)
# ---------------------------------------------------------------------------
_train_raw = pd.read_csv(os.path.join(DATA, "train.csv"))
_test_raw = pd.read_csv(os.path.join(DATA, "test.csv"))
_Xtr_full = feat_mod.build_features(_train_raw)
_Xte_full = feat_mod.build_features(_test_raw)
FEATS_BASE = feat_mod.base_feature_columns()  # 20 feats before store_te
_y_log = np.log1p(_train_raw[TARGET].to_numpy(np.float64))

# MANDATORY: identical fold assignment to scripts/train.py & scripts/iterate2.py --
# KFold(n_splits=5, shuffle=True, random_state=42) depends only on n_samples.
_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_kf.split(np.zeros(len(_y_log))))


def rmsle_from_log(y_true_log, y_pred_log) -> float:
    """Ports scripts/iterate2.py's rmsle_from_log() verbatim: RMSLE computed by
    expm1'ing both sides back to cost-space (clip pred >=0 as a safety net) then
    log1p+RMSE -- but the OOF/blend arrays passed in stay in log1p space until this
    function is called, exactly matching the linear run's own scoring convention."""
    pred_cost = np.clip(np.expm1(y_pred_log), 0, None)
    true_cost = np.expm1(y_true_log)
    diff = np.log1p(true_cost) - np.log1p(pred_cost)
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# fold-safe store_combo target encoding -- computed ONCE (independent of any downstream
# model hyperparameters), verbatim port of scripts/iterate2.py's fold_safe_te().
# ---------------------------------------------------------------------------
def _compute_store_te():
    gm = _y_log.mean()
    gtr, gte = _Xtr_full["store_combo"], _Xte_full["store_combo"]
    te_tr = np.zeros(len(gtr))
    te_te_accum = np.zeros(len(gte))
    for tr, va in _FOLDS:
        m = pd.Series(_y_log[tr], index=gtr.iloc[tr].index).groupby(gtr.iloc[tr]).mean()
        te_tr[va] = gtr.iloc[va].map(m).fillna(gm).to_numpy()
        te_te_accum += gte.map(m).fillna(gm).to_numpy() / N_SPLITS
    return te_tr, te_te_accum


_STORE_TE_TR, _STORE_TE_TE = _compute_store_te()
ALL_FEATURES = FEATS_BASE + ["store_te"]  # 21 total, matches STATUS.md R1-R3/R5's n_features=21


def get_feature_frame(drop=None):
    drop = set(drop or [])
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {unknown}")
    feats = [c for c in ALL_FEATURES if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    base_cols = [c for c in feats if c != "store_te"]
    X = _Xtr_full[base_cols].to_numpy(np.float32)
    Xtest = _Xte_full[base_cols].to_numpy(np.float32)
    if "store_te" in feats:
        X = np.hstack([X, _STORE_TE_TR.reshape(-1, 1).astype(np.float32)])
        Xtest = np.hstack([Xtest, _STORE_TE_TE.reshape(-1, 1).astype(np.float32)])
    return X, Xtest, feats


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# solo-model runners (mirrors scripts/iterate2.py's run_lgb/run_cat, generalized over
# an arbitrary params dict and feature subset)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    # NOTE: deliberately no n_jobs override -- see eval_s3e1.py's identical note, the
    # same n_jobs oversubscription slowdown risk applies here.
    p = dict(objective="regression", metric="rmse", verbosity=-1, random_state=SEED)
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y_log))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = lgb.LGBMRegressor(n_estimators=n_est, **p)
        m.fit(X[tr], _y_log[tr], eval_set=[(X[va], _y_log[va])],
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest):
    from catboost import CatBoostRegressor
    p = dict(depth=8, learning_rate=0.05, l2_leaf_reg=3.0)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 2500))
    esr = p.pop("early_stopping_rounds", 150)
    seed = p.pop("random_seed", SEED)
    oof = np.zeros(len(_y_log))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=n_est,
                               random_seed=seed, thread_count=4, verbose=False,
                               allow_writing_files=False, **p)
        m.fit(X[tr], _y_log[tr], eval_set=(X[va], _y_log[va]), early_stopping_rounds=esr)
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})

    Xdf, Xtestdf, feats = get_feature_frame(drop)
    if model == "lgb":
        oof, pred = _run_lgb(params, Xdf, Xtestdf)
    elif model == "cat":
        oof, pred = _run_cat(params, Xdf, Xtestdf)
    else:
        raise ValueError(f"unknown model type {model!r}")

    score = rmsle_from_log(_y_log, oof)
    extra = dict(n_feats=len(feats))
    return oof, pred, score, feats, extra


# ---------------------------------------------------------------------------
# legacy-cache reuse (Phase D-6 scale lever): load scripts/cache/*.npz's already-trained
# OOF/pred directly, verifying digit-for-digit consistency with the historical
# STATUS.md/experiments.json score -- no retraining, no signal timeout needed.
# ---------------------------------------------------------------------------
def load_legacy_solo(name: str, expected_rmsle: float, tol: float = 2e-4):
    path = os.path.join(LEGACY_CACHE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        raise ValueError(f"legacy cache {path} not found -- cannot reuse, must retrain")
    d = np.load(path)
    oof, pred = d["oof"], d["pred"]
    if len(oof) != len(_y_log):
        raise ValueError(f"legacy cache {name} OOF length {len(oof)} != train length "
                          f"{len(_y_log)} -- stale/incompatible cache")
    score = rmsle_from_log(_y_log, oof)
    if abs(score - expected_rmsle) > tol:
        raise ValueError(f"legacy cache {name} RMSLE {score:.6f} does not reproduce "
                          f"expected {expected_rmsle:.6f} (tol={tol}) -- digit-for-digit "
                          f"reproduction check FAILED, do not reuse")
    return oof, pred, score


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over harness_v2-CACHED member OOF vectors (log1p
# space), no retraining.
# ---------------------------------------------------------------------------
def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")

    def metric_fn(vec_log):
        return rmsle_from_log(_y_log, vec_log)

    best_w, best_score, _oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn, weight_search=method)
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, rmsle=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 300) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or any
    exception is captured as status="failed" so the tree-search loop can keep going.

    RMSLE is lower-is-better already, so score == result["rmsle"] (no sign flip)."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, _feats, extra = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, rmsle=score)
            wall = time.time() - t0
            result = {"n_feats": extra["n_feats"], "rmsle": round(score, 6)}
            return dict(status="evaluated", score=round(score, 6), wall_s=round(wall, 1),
                        result=result, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            wall = time.time() - t0
            return dict(status="evaluated", score=round(score, 6), wall_s=round(wall, 1),
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
