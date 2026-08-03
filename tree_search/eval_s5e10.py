"""tree_search/eval_s5e10.py -- per-competition evaluator for playground-series-s5e10
(accident_risk, regression, RMSE metric, minimize-better, clip-to-[0,1] post-processing).

Stage-4 harness_v3 driver support. Same two-node-kind schema as eval_s4e11.py /
eval_s4e1.py:

  1. solo  -- {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...}, features:{drop:[...]}}
     Trains one regressor with the SAME canonical 5-fold KFold(shuffle=True, seed=42)
     as scripts/04_train_blend.py / scripts/05_iterate.py (via scripts/features.py's
     make_folds -- imported, not re-implemented; also identical to the Feb-2026 run's
     folds). If `node_id` is given, OOF/test predictions are cached to
     tree_search/cache_s5e10/solo_<node_id>.npz via harness_v2.cache_oof so blend
     nodes never retrain.

  2. blend -- {kind:"blend", members:[<solo node id>, ...], weight_search:"dirichlet"}
     Loads cached member OOFs and weight-searches (dispatches to harness_v2.eval_blend)
     for the RMSE-minimizing combination -- ALWAYS scored through rmse_clip (clip to
     the target's [0,1] range first), matching this comp's post-processing rule
     (features.py docstring / knowledge/experience.md's "decide on the post-processed
     score, never the raw one").

SIGN CONVENTION: harness.py/harness_v2.py/harness_v3.py assume lower-is-better scores;
RMSE already IS lower-is-better, so scores are stored as-is (no negation anywhere).
`result["rmse"]` carries the same human-readable value.

Reuses competitions/playground-series-s5e10/scripts/features.py UNMODIFIED (27-feature
Feb-2026 set, deterministic transforms only) so scores are directly comparable to
stage 2 (scripts/04_train_blend.py) and stage 3 (scripts/05_iterate.py) numbers.
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s5e10")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _HERE)
from features import make_folds, build_all, rmse_clip, TARGET, ID  # noqa: E402
import harness_v2 as hv2  # noqa: E402
import eval_support as esup  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s5e10")
os.makedirs(CACHE_DIR, exist_ok=True)
N_SPLITS, SEED = 5, 42

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = _train[TARGET].to_numpy(np.float64)
_FOLDS = make_folds(_y)  # canonical folds -- IDENTICAL to stage 2/3 and Feb
_Xtr_full, _Xte_full, ALL_FEATURES = build_all(_train, _test)


def rmse(y_true, pred) -> float:
    """Clipped RMSE -- the ONLY decision metric in this comp (see features.py's
    rmse_clip)."""
    return rmse_clip(y_true, pred)


def maybe_postprocess(pred_full, postprocess):
    """Scale, then clip, then round -- all inside the metric, dossier vocabulary
    (`global_scale`/`round_to_int`/`clip_min`/`clip_max`).

    Defaults (no scale, clip to [0,1], no rounding) reproduce the historical
    `rmse_clip` path digit-for-digit -- cache_s5e10 scores and Feb-2026 comparability
    depend on that. This competition's dossier explicitly forbids rounding
    (snap-to-grid was rejected on evidence), so `round_to_int` is honoured but never
    defaulted on. Until 2026-07-31 this evaluator had no postprocess mechanism at all
    -- the one NOT_IMPLEMENTED case in external_data/verify_advisory.py's audit.
    """
    pp = postprocess or {}
    gs = pp.get("global_scale")
    if gs == "auto" or pp.get("auto_scale"):
        raise ValueError("global_scale='auto' is not implemented in eval_s5e10; "
                         "request a numeric scale")
    sc = float(gs) if gs is not None else float(pp.get("scale", 1.0))
    lo = float(pp.get("clip_min", 0.0))
    hi = float(pp.get("clip_max", 1.0))
    v = np.clip(np.asarray(pred_full, dtype=np.float64) * sc, lo, hi)
    if pp.get("round_to_int"):
        v = np.round(v)
    return rmse_clip(_y, v, lo=lo, hi=hi), sc


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
# solo-model runners (mirror scripts/04_train_blend.py / 05_iterate.py exactly)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest, feats):
    import lightgbm as lgb
    p = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63,
             max_depth=-1, min_child_samples=30, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, reg_alpha=0.1, reg_lambda=0.1,
             verbose=-1, n_jobs=-1, seed=SEED)
    p.update(params)
    n_rounds = p.pop("num_boost_round", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        dtr = lgb.Dataset(X[tr], label=_y[tr], feature_name=feats)
        dva = lgb.Dataset(X[va], label=_y[va], feature_name=feats, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(esr, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xtest, feats):
    import xgboost as xgb
    p = dict(objective="reg:squarederror", eval_metric="rmse", learning_rate=0.05,
             max_depth=7, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
             reg_lambda=1.0, random_state=SEED, n_jobs=-1, tree_method="hist")
    p.update(params)
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBRegressor(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])], verbose=False)
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest, feats):
    from catboost import CatBoostRegressor
    p = dict(loss_function="RMSE", eval_metric="RMSE", learning_rate=0.05, depth=8,
             l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
             allow_writing_files=False, thread_count=20)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 3000))
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostRegressor(iterations=n_est, **p)
        m.fit(X[tr], _y[tr], eval_set=(X[va], _y[va]), early_stopping_rounds=esr,
              verbose=False)
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    Xdf, Xtestdf, feats = get_feature_frame(drop)
    X, Xtest = Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32)

    if model == "lgb":
        oof, pred = _run_lgb(params, X, Xtest, feats)
    elif model == "xgb":
        oof, pred = _run_xgb(params, X, Xtest, feats)
    elif model == "cat":
        oof, pred = _run_cat(params, X, Xtest, feats)
    else:
        raise ValueError(f"unknown model type {model!r}")

    postprocess = config.get("postprocess") or {}
    score, _scale_used = maybe_postprocess(oof, postprocess)
    if postprocess:
        esup.write_attestation(_COMP_DIR, "postprocess", {
            "honored_params": sorted(postprocess), "scale_used": round(_scale_used, 3)})
    return oof, pred, score, feats


# ---------------------------------------------------------------------------
# ensemble (blend) node
# ---------------------------------------------------------------------------
def _metric_rmse(vec):
    return rmse(_y, vec)


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    # METRIC SYMMETRY (v5.3, 2026-08-03 audit): solo nodes are scored through
    # maybe_postprocess while blends used the plain clipped RMSE, so a competition-level
    # postprocess request bound one node kind and not the other, and the search compared
    # them with a single min(). The blend metric now reads the same block, inside the
    # weight search, so every candidate is scored on the vector that would be submitted.
    postprocess = config.get("postprocess") or {}
    metric_fn = ((lambda vec: maybe_postprocess(vec, postprocess)[0]) if postprocess
                 else _metric_rmse)
    best_w, best_s, _oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn,
                                          weight_search=method)
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, rmse=round(best_s, 6)), best_s


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 600) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or
    any exception is captured as status="failed" so the tree-search loop keeps going.

    score is clipped-OOF RMSE (lower-is-better, harness-native sign).
    result["rmse"] carries the same value for reporting."""
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
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, rmse=score)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "rmse": round(score, 6)}
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
