"""tree_search/eval_s3e16_v2.py — per-competition evaluator for playground-series-s3e16
(Crab Age, regression, ROUNDED-INTEGER MAE, and the only comp in the harness_v2 lineup
with a REAL Kaggle LB anchor). Phase E-3 (first harness_v2 tree-search build on a metric
whose acid test is "does a gain survive rounding", not "is the metric discretized" per se
-- s3e16's OOF MAE is continuous, but the DECISION metric (and the one the leaderboard
scores) is `MAE(clip(round(oof), 1, None))`, so this module follows the exact same
non-negotiable rule harness_v2/eval_s3e5_v2.py established for QWK-after-rounder: the
rounding step happens INSIDE metric_fn/evaluate_solo, on every candidate weight vector
during blend search itself, never retroactively applied only to a winner.

--- Why this run is the acid test ---
STATUS.md's Phase B iteration (scripts/round1.py, round2.py) already ran the validated
"Optuna fold-proxy -> add to pool -> seed bag" recipe here and got the recipe's known
FAILURE MODE: raw OOF MAE improved monotonically (<score> -> <score> -> <score>) but
ROUNDED OOF MAE got WORSE both times (<score> -> <score> -> <score>) -- logged verbatim
in knowledge/experience.md's "MAE/整數目標" boundary-condition bullet (search "邊界條件").
The tuned LGB variants only touched depth/lr/reg (same objective/features/folds as the
original LGB), so they cluster tightly in OOF-space with the un-tuned LGB: real but tiny
raw-MAE gains that never cross the discrete rounding boundary. This build's mutation menu
is deliberately shaped to attack exactly that diagnosis -- genuine DIVERSITY (a different
loss surface: Tweedie/Poisson heads for the count-like Age target) and a genuine
UNEXPLORED lever (boundary-push on the Optuna box an edge sits on) rather than another
near-duplicate of the same LGB-with-nudged-hyperparams recipe.

--- Optuna box-edge check (the boundary-push lever, done BEFORE writing this module) ---
scripts/tune_lgb_optuna.py's search box (21/50 trials completed, 300s timeout) and the
winning scripts/lgb_optuna_best.json params:
  The Optuna study's per-parameter outcome (which won value sat where inside its search
  box) is the recorded run's tuned configuration and is NOT reproduced here: handing a
  fresh lane the champion hyper-parameters is a warm start from its own answer
  (2026-08-10 round-9). Only the STRUCTURAL finding survives, because it is the lever
  this module implements and it names no value: exactly one parameter's optimum landed
  on its box's own edge, which is the boundary-push signal LGBBOUND probes.
LGBBOUND is the node kind that acts on that finding: it re-proposes the saturating parameter
past the edge of the box the study searched. Which parameter, and what the box was, are the
recorded run's tuned configuration and stay withheld -- naming them here would undo the
paragraph above (2026-08-10 round-10).

Two node kinds, same schema convention as eval_s3e5_v2.py/eval_s3e9_v2.py:

  1. solo  -- {kind:"solo", model:"lgb", params:{...}, features:{drop:[...]}}
     Trains one LGBMRegressor with the IDENTICAL 5-fold StratifiedKFold-on-binned-Age
     (Age>=20 merged into one bin, shuffle, seed=42) as scripts/pool_lib.py, so folds are
     digit-for-digit reproducible with the Phase-B cached OOFs. Only "lgb" is implemented
     as a fresh-training runner here (Tweedie/Poisson/objective variants and the boundary-
     push are all still LightGBM, just different `objective`/hyperparams) -- XGB/CAT are
     NEVER retrained by this module, only reused read-only from Phase B's own cache (see
     load_legacy_solo below), since this run's new territory is entirely on the LGB axis
     (boundary-push + diverse objective heads) and the feature-pruned axis, not "yet
     another XGB/CAT variant".

  2. blend -- {kind:"blend", members:[<node id>, ...], weight_search:"dirichlet"}
     Loads cached member OOF vectors via harness_v2.load_oof, weight-searches
     (harness_v2.eval_blend, k=800 + a coordinate-ascent refinement stage, per the task
     brief's explicit E-2 lesson: "coarse grids silently tie -- use k=800+coordinate-
     ascent from the start") for the blend minimizing ROUNDED MAE. Rounding happens
     INSIDE metric_fn so it applies to every candidate weight vector, not just the
     winner.

SIGN CONVENTION: MAE is already lower-is-better/minimize, matching harness_v2's
convention throughout -- unlike eval_s3e5_v2.py's QWK (maximize-better, needed a sign
flip), no sign flip is needed anywhere in this module. `score` stored on every node IS
the rounded MAE directly.

Legacy-OOF-reuse (this run's budget lever, ported from eval_s3e5_v2.py's load_v1_solo):
scripts/cache/{LGB,XGB,CAT,LGB_tuned,LGB_tuned_seed2024}.npz (Phase B's own npz cache,
written by scripts/pool_lib.py -- oof/pred/mae/time_s keys, NO stored y since pool_lib
never wrote one) are READ-ONLY inputs here. `load_legacy_solo(name, expected_raw,
expected_rounded)` loads the cached OOF, recomputes BOTH the raw MAE and this module's
own round_clip_mae against this module's OWN reconstructed `_y` (identical to
pool_lib.py's), and asserts BOTH reproduce the historical STATUS.md table digit-for-
digit before anything is re-cached under a v2 node id -- this IS the "verify root
digit-for-digit" check the task brief requires, done as a load + recompute (near-
instant) rather than a full retrain, freeing the ~30 min budget for new territory.
"""
import json
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e16")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
DATA = os.path.join(_COMP_DIR, "data")

# Phase B's own npz cache (READ-ONLY here, never written to): holds oof/pred/mae/time_s
# for LGB, XGB, CAT, LGB_tuned, LGB_tuned_seed2024 -- scripts/pool_lib.py's cache_path().
LEGACY_CACHE_DIR = os.path.join(_SCRIPTS_DIR, "cache")
# This run's OWN new node cache -- a dedicated tree_search/cache_s3e16/ dir (gitignored),
# per the task brief; legacy members re-cached under v2 node-id numbering also live here
# (never inside scripts/cache/, so a v2 node id can never collide with/overwrite a Phase-B
# cache file).
CACHE_DIR = os.path.join(_HERE, "cache_s3e16")

sys.path.insert(0, _HERE)
sys.path.insert(0, _SCRIPTS_DIR)
import harness_v2 as hv2  # noqa: E402
import stage2_inputs  # noqa: E402
stage2_inputs.require_module(_SCRIPTS_DIR, "features", comp="playground-series-s3e16",
                             exposes=['build_features', 'feature_columns'])
from features import build_features, feature_columns  # noqa: E402

TARGET, ID = "Age", "id"
N_SPLITS, SEED = 5, 42

_train = pd.read_csv(os.path.join(DATA, "train.csv"))
_test = pd.read_csv(os.path.join(DATA, "test.csv"))
_h_med = _train.loc[_train["Height"] > 0, "Height"].median()
_Xtr_full = build_features(_train, height_median=_h_med)
_Xte_full = build_features(_test, height_median=_h_med)
ALL_FEATURES = feature_columns(_Xtr_full)
_y = _train[TARGET].to_numpy(np.float64)
AGE_LOW = int(_y.min())  # 1 -- lower clip bound for rounded predictions

_ybin = np.where(_y >= 20, 20, _y).astype(int)  # IDENTICAL binning to scripts/pool_lib.py
_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(np.zeros(len(_y)), _ybin))  # IDENTICAL folds to pool_lib.py/train.py


# ---------------------------------------------------------------------------
# The ONE rounder: MAE(clip(round(oof), AGE_LOW, None)) -- s3e16 has no analog to s3e5's
# fold_avg-vs-full_oof choice, so this is applied uniformly, always INSIDE the scoring
# path (never retroactively).
# ---------------------------------------------------------------------------
def raw_mae(oof_vec) -> float:
    return float(mean_absolute_error(_y, oof_vec))


def round_clip_mae(oof_vec) -> float:
    pred = np.clip(np.round(oof_vec), AGE_LOW, None)
    return float(mean_absolute_error(_y, pred))


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
# solo-model runner -- LightGBM only (XGB/CAT are reused read-only, never retrained here;
# see module docstring). Base params mirror scripts/pool_lib.py's train_lgb() defaults
# exactly so a bare `params={}` override reproduces the Phase-B LGB member digit-for-
# digit if ever re-run (not needed -- that member is reused via load_legacy_solo -- but
# keeps this runner's baseline honest/checkable).
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    p = dict(objective="regression_l1", metric="mae", n_estimators=3000,
              learning_rate=0.02, num_leaves=63, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
              reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = lgb.LGBMRegressor(**p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


RUNNERS = {"lgb": _run_lgb}


def evaluate_solo(config):
    drop = (config.get("features") or {}).get("drop", [])
    Xdf, Xtestdf, feats = get_feature_frame(drop)
    model = config["model"]
    params = dict(config.get("params") or {})
    if model not in RUNNERS:
        raise ValueError(f"unknown/unsupported solo model type {model!r} -- only 'lgb' is "
                          f"a fresh-training runner in this module (XGB/CAT are reused "
                          f"read-only via load_legacy_solo)")
    Xnp, Xtestnp = Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32)
    oof, pred = RUNNERS[model](params, Xnp, Xtestnp)
    raw = raw_mae(oof)
    rounded = round_clip_mae(oof)
    return oof, pred, raw, rounded, feats


# ---------------------------------------------------------------------------
# legacy-OOF-reuse: scripts/cache/<name>.npz, READ-ONLY.
# ---------------------------------------------------------------------------
def load_legacy_solo(name: str, expected_raw: float, expected_rounded: float, tol: float = 2e-4):
    path = os.path.join(LEGACY_CACHE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        raise ValueError(f"legacy cache has no member {name!r} at {path}")
    d = np.load(path)
    oof, pred = d["oof"], d["pred"]
    if len(oof) != len(_y):
        raise ValueError(f"legacy member {name!r} OOF length {len(oof)} != {len(_y)}")
    raw = raw_mae(oof)
    rounded = round_clip_mae(oof)
    if abs(raw - expected_raw) > tol:
        raise ValueError(f"legacy member {name!r} raw MAE {raw:.6f} does not reproduce "
                          f"expected {expected_raw:.6f} (tol={tol}) -- digit-for-digit "
                          f"reproduction FAILED, do not reuse")
    if abs(rounded - expected_rounded) > tol:
        raise ValueError(f"legacy member {name!r} rounded MAE {rounded:.6f} does not "
                          f"reproduce expected {expected_rounded:.6f} (tol={tol}) -- "
                          f"digit-for-digit reproduction FAILED, do not reuse")
    return oof, pred, raw, rounded


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over harness_v2-CACHED member OOF, no retraining.
# k=800 dirichlet + coordinate-ascent refinement, per the task brief's explicit E-2
# lesson ("coarse grids silently tie -- use k=800+coordinate-ascent from the start");
# unlike eval_s3e5_v2.py this was never re-derived via an empirical k-sweep here since
# the brief already mandates the setting directly.
# ---------------------------------------------------------------------------
def _coord_ascent_refine(oofs, metric_fn, w0, s0, rounds=6,
                          deltas=(0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005)):
    best_w, best_s = np.array(w0, dtype=float), s0
    k = len(best_w)
    for _ in range(rounds):
        improved = False
        for i in range(k):
            for d in deltas:
                w_try = best_w.copy()
                w_try[i] = max(0.0, w_try[i] + d)
                if w_try.sum() <= 0:
                    continue
                w_try = w_try / w_try.sum()
                s = metric_fn(oofs @ w_try)
                if s < best_s - 1e-9:
                    best_s, best_w = s, w_try
                    improved = True
        if not improved:
            break
    return best_w, best_s


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")

    def metric_fn(vec):
        return round_clip_mae(vec)  # rounding INSIDE metric_fn, no sign flip needed (MAE
                                     # is already lower-is-better)

    best_w, best_score, oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn,
                                               weight_search=method, k=800)
    best_w, best_score = _coord_ascent_refine(oofs, metric_fn, best_w, best_score)
    raw = raw_mae(oofs @ best_w)
    weights = {str(m): round(float(w), 4) for m, w in zip(members, best_w)}
    return dict(members=members, weights=weights, method=method,
                raw_mae=round(raw, 5)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 180) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises. `score` is the
    ROUNDED MAE always (lower-is-better, no sign flip -- matches harness_v2's convention
    directly)."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, raw, rounded, feats = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, y=_y, mae_raw=raw,
                              mae_rounded=rounded)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "mae_raw": round(raw, 5),
                      "mae_rounded": round(rounded, 5)}
            return dict(status="evaluated", score=round(rounded, 5), wall_s=round(wall, 1),
                        result=result, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            wall = time.time() - t0
            result["mae_rounded"] = round(score, 5)
            return dict(status="evaluated", score=round(score, 5), wall_s=round(wall, 1),
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
