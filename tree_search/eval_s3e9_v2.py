"""tree_search/eval_s3e9_v2.py — per-competition evaluator for playground-series-s3e9
(Concrete Compressive Strength, RMSE, minimize), REVENGE MATCH build on harness_v2
(Phase E-1). v1 (tree_search/eval_s3e9.py, single-model-only nodes) LOST here:
12.07459 vs the linear iteration's 7-way seed-bagged blend 12.07003 (STATUS.md's Phase
C-2a verdict: "to actually beat this competition's current best score, Stage 4 needs
the search space widened beyond single-model nodes"). v2 is exactly that widening —
harness_v2's ensemble-default node space (`kind`: "solo"|"blend") — applied to the same
data/CV/features v1 already used, so any win here is attributable to the harness change,
not a different problem setup.

Two node kinds, same schema convention as eval_s3e11.py/eval_s3e14.py:

  1. solo — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...}, features:{drop:[...]}}
     Trains one regressor with 5-fold StratifiedKFold(shuffle=True, seed=42) on
     Strength deciles (`pd.qcut(y, 10)`) — EXACT same CV scheme as
     competitions/playground-series-s3e9/scripts/train.py and v1's eval_s3e9.py, so
     scores here are directly comparable to STATUS.md/experiments.json AND to v1's
     experiments_tree.json. Feature set: scripts/features.py's build_features()
     (unmodified, 22 features), get_feature_matrix(drop=[...]) allows dropping any
     subset for feature-variant solo nodes (not exercised by this run's own mutation
     queues per the task brief -- feature engineering was already exhaustively probed
     by v1's FEAT lineage and experience.md's "特徵過多反而退步"/interaction-term
     lessons, no new feature-engineering lever is being tested here).

     No postprocess/clip step: score = rmse(y, oof) on the RAW oof vector, matching
     every historical s3e9 script's own weight_search()/scoring convention exactly
     (clipping only ever happened at the FINAL submission-file stage, never during OOF
     scoring) -- this keeps solo/blend scores here byte-comparable to STATUS.md.

     ⚠️ Legacy-OOF-reuse (this run's own budget lever, ported from eval_s3e11.py's
     load_legacy_solo): scripts/_round23_pool.npz already holds 5 of the historical
     pool's solo OOF/pred arrays fully trained (LGB, XGB, CAT, LGB_tuned,
     LGB_tuned_seed2 -- from train_optuna_pool.py's Round 2/3). `load_legacy_solo(name,
     expected_rmse)` loads one of these columns directly and asserts its RMSE (via THIS
     module's own `rmse()`) is digit-for-digit consistent with the historical
     STATUS.md/experiments.json value -- this IS the "verify root reproduces
     digit-for-digit BEFORE searching" check the task brief asks for (done as a
     load+recompute, near-instant, not a full retrain), so root + 4 legacy pool members
     cost ~0s of training time, freeing the budget for NEW solo variants (seed-family
     expansion off CAT and both LGB families, boundary-push probes on CAT depth/l2, one
     deliberately-diverse deep LGB) and blend-heavy weight-search exploration.

  2. blend — {kind:"blend", members:[<solo node id>, ...],
             weight_search:"dirichlet"|"grid_simplex", compose:"raw"|"zscore"}
     compose="raw" (default): loads each member's cached OOF vector via
     harness_v2.load_oof and weight-searches (harness_v2.eval_blend) for the blend that
     minimizes rmse(y, oofs @ w) directly -- same semantics as every historical
     train_*.py weight_search() in this comp's scripts/ dir.
     compose="zscore" (one experimental composition-variant node, per the task brief's
     "composition variants ... rank-average" instruction): each member's OOF is
     standardized (subtract/divide by ITS OWN mean/std) before the dirichlet weight
     search, and the composed z-vector is rescaled via the TARGET's own mean/std before
     RMSE is computed -- a distinct blend parameterization from plain weighted-average-
     of-raw-values, testing whether scale-normalizing members before blending helps.
     Low prior expectation: experience.md's only rank/scale-normalization blend
     experiment (s3e7's "混合層微調...rank-average") found only noise-level differences
     on an AUC (ranking) metric; RMSE here is a continuous, non-rank metric, if
     anything an even weaker case for scale-normalization to matter, so this is an
     honest single-node test, not assumed to help.

SIGN CONVENTION: RMSE is already lower-is-better, so `score` handed to add_root/add_node
is the RMSE itself (no sign flip), same convention as eval_s3e9.py/eval_s3e11.py.

CatBoost: allow_writing_files=False (mandatory per task brief) + explicit
thread_count=4 (s3e11's "反面教訓" lesson: CatBoost's default file-logging + implicit
thread count can stall/oversubscribe a long-running unattended process; v1's
eval_s3e9.py used thread_count=-1 without incident on this comp's much-smaller data,
but there is no reason to reintroduce that risk here).
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
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402

sys.path.insert(0, _SCRIPTS_DIR)
from features import build_features, feature_columns  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s3e9")
LEGACY_POOL_NPZ = os.path.join(_SCRIPTS_DIR, "_round23_pool.npz")
TARGET, ID = "Strength", "id"
N_SPLITS, SEED = 5, 42

# ---------------------------------------------------------------------------
# data + features (loaded once at import time, matches scripts/train.py & v1's
# eval_s3e9.py exactly)
# ---------------------------------------------------------------------------
_train = pd.read_csv(os.path.join(DATA, "train.csv"))
_test = pd.read_csv(os.path.join(DATA, "test.csv"))
_Xtr_full = build_features(_train)
_Xte_full = build_features(_test)
ALL_FEATURES = feature_columns(_Xtr_full)
_y = _train[TARGET].to_numpy(np.float64)

# MANDATORY: identical fold assignment to scripts/train.py & v1's eval_s3e9.py --
# StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on Strength deciles.
_ybin = pd.qcut(_y, 10, labels=False, duplicates="drop")
_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(np.zeros(len(_y)), _ybin))


def rmse(a, b) -> float:
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


# ---------------------------------------------------------------------------
# solo-model runners (verbatim logic from v1 eval_s3e9.py, generalized param dict)
# ---------------------------------------------------------------------------
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
             thread_count=4, verbose=False, allow_writing_files=False)
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


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    if model not in RUNNERS:
        raise ValueError(f"unknown model type {model!r}")

    X, Xtest, feats = get_feature_matrix(drop)
    oof, pred = RUNNERS[model](params, X, Xtest)
    score = rmse(_y, oof)
    extra = dict(n_feats=len(feats))
    return oof, pred, score, feats, extra


# ---------------------------------------------------------------------------
# legacy-OOF-reuse (Phase E-1 budget lever, ported from eval_s3e11.load_legacy_solo):
# scripts/_round23_pool.npz's oof5/pred5/names5 arrays hold 5 already-trained solo
# members -- load + digit-for-digit RMSE recompute, no retraining, no timeout needed.
# ---------------------------------------------------------------------------
def load_legacy_solo(name: str, expected_rmse: float, tol: float = 2e-4):
    if not os.path.exists(LEGACY_POOL_NPZ):
        raise ValueError(f"legacy pool cache {LEGACY_POOL_NPZ} not found -- cannot reuse, "
                          f"must retrain")
    d = np.load(LEGACY_POOL_NPZ, allow_pickle=True)
    names5 = [str(n) for n in d["names5"]]
    if name not in names5:
        raise ValueError(f"legacy pool cache has no member named {name!r} (available: "
                          f"{names5})")
    idx = names5.index(name)
    oof = d["oof5"][:, idx]
    pred = d["pred5"][:, idx]
    if len(oof) != len(_y):
        raise ValueError(f"legacy cache {name} OOF length {len(oof)} != train length "
                          f"{len(_y)} -- stale/incompatible cache")
    score = rmse(_y, oof)
    if abs(score - expected_rmse) > tol:
        raise ValueError(f"legacy cache {name} RMSE {score:.6f} does not reproduce "
                          f"expected {expected_rmse:.6f} (tol={tol}) -- digit-for-digit "
                          f"reproduction check FAILED, do not reuse")
    return oof, pred, score


def _coord_descent_refine(oofs, metric_fn, w0, s0, rounds=60,
                          deltas=(0.05, -0.05, 0.02, -0.02, 0.01, -0.01,
                                  0.005, -0.005, 0.002, -0.002, 0.001, -0.001)):
    """Coordinate-descent fine-refinement AFTER the coarse dirichlet search, ported
    verbatim (same deltas/round cap) from every historical s3e9 train_*.py's own
    weight_search() -- harness_v2.eval_blend's dirichlet search alone (1500 random +
    500 refine draws) is coarser than this and, on its own, plateaued at 12.0713-12.0715
    in this run's own first pass, short of the linear iteration's 12.07003 (which WAS
    found via dirichlet+coordinate-descent). Applying the same two-stage recipe here
    keeps the search comparably thorough to what actually produced the number v2 is
    trying to beat."""
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


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over harness_v2-CACHED member OOF vectors, no
# retraining. compose="raw" (default) or "zscore" (composition-variant probe).
# ---------------------------------------------------------------------------
def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    compose = config.get("compose", "raw")

    if compose == "raw":
        def metric_fn(vec):
            return rmse(_y, vec)
        best_w, best_score, oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn,
                                                   weight_search=method, k=6000)
        best_w, best_score = _coord_descent_refine(oofs, metric_fn, best_w, best_score)
    elif compose == "zscore":
        oofs_raw = np.stack([hv2.load_oof(CACHE_DIR, m) for m in members], axis=1)
        mu = oofs_raw.mean(axis=0)
        sd = oofs_raw.std(axis=0) + 1e-9
        z = (oofs_raw - mu) / sd
        y_mu, y_sd = _y.mean(), _y.std()

        def metric_fn_z(zvec):
            return rmse(_y, zvec * y_sd + y_mu)

        rng = np.random.default_rng(42)
        n = z.shape[1]
        candidates = [np.eye(n)[i] for i in range(n)]
        candidates.append(np.full(n, 1.0 / n))
        candidates += list(rng.dirichlet(np.ones(n), size=1500))
        best_w, best_score = None, None
        for w in candidates:
            s = metric_fn_z(z @ w)
            if best_score is None or s < best_score:
                best_score, best_w = s, w
        conc = np.clip(best_w, 1e-3, None) * 200.0
        for w in rng.dirichlet(conc, size=500):
            s = metric_fn_z(z @ w)
            if s < best_score:
                best_score, best_w = s, w
        best_w, best_score = _coord_descent_refine(z, metric_fn_z, best_w, best_score)
    else:
        raise ValueError(f"unknown compose mode {compose!r}")

    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, compose=compose, rmse=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 90) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or any
    exception is captured as status="failed" so the tree-search loop can keep going."""
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
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, rmse=score)
            wall = time.time() - t0
            result = {"n_feats": extra["n_feats"], "rmse": round(score, 6)}
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
