"""tree_search/eval_s5e10_july.py — evaluator for playground-series-s5e10
(road accident risk, regression, RMSE, minimize-better).

FRESH module for the July-2026 run. It deliberately does NOT import, read or reuse
tree_search/eval_s5e10.py or tree_search/cache_s5e10 (artefacts of an earlier run of this
same competition) — this run's lane isolation forbids reading its own past answers.
Cache dir is tree_search/cache_s5e10_july_tree.

SIGN CONVENTION: RMSE is minimize-better, which is already the harness convention, so
`score` is the RMSE itself — no sign flip anywhere.

Node kinds:
  solo  {kind:"solo", model:"lgb"|"xgb"|"cat"|"ridge", params:{...}, features:{drop:[...]}}
  blend {kind:"blend", members:[<node id>, ...], weight_search:"dirichlet"}

Comp-local mutation surface (EDA-motivated, see competitions/.../eda_structure.json):
the data-generating process is near-purely ADDITIVE — an additive fit with the right
encoding reaches in-sample RMSE 0.057834, and adding ALL 927 two-way crosses only
reaches 0.056690. So the interesting knobs here are the ones that push a GBDT TOWARD
additivity, which it will not do on its own:
  * params["interaction_constraints"] — LightGBM/XGBoost: "additive" expands to one
    constraint group per feature (a pure additive GBDT); "additive+curv_acc" also allows
    the single materially non-additive pair the 2-way probe found
    (num_reported_accidents x curvature, gain 0.000732; runner-up pair only 0.000136).
  * params["linear_tree"] — LightGBM fits a linear model in each leaf, a natural fit for
    a smooth additive target. Incompatible with LightGBM's categorical handling, so this
    path passes the nominal columns as plain integer codes.
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_COMP = os.path.join(_REPO, "competitions", "playground-series-s5e10")
sys.path.insert(0, os.path.join(_COMP, "scripts"))
sys.path.insert(0, _HERE)
from features import (  # noqa: E402
    ALL_FEATURES, BASE_FEATURES, CATEGORICAL, N_SPLITS, SEED, TARGET, build_all, make_folds,
)
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_s5e10_july_tree2")
os.makedirs(CACHE_DIR, exist_ok=True)
NTHREAD = 10

_train = pd.read_csv(os.path.join(_COMP, "data", "train.csv"))
_test = pd.read_csv(os.path.join(_COMP, "data", "test.csv"))
_y = _train[TARGET].to_numpy(np.float64)
_FOLDS = make_folds(len(_y))
_Xtr, _Xte, _ = build_all(_train, _test)

# the one pair the exhaustive 2-way probe found to carry real interaction signal
CURV_ACC_PAIR = ["num_reported_accidents", "curvature"]


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _clip(p):
    return np.clip(p, 0.0, 1.0)


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def get_feature_frame(drop=None):
    drop = set(drop or [])
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {sorted(unknown)}")
    feats = [c for c in ALL_FEATURES if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    return _Xtr[feats].copy(), _Xte[feats].copy(), feats


def _expand_constraints(spec, feats):
    """'additive' -> one group per feature (pure additive GBDT).
    'additive+curv_acc' -> the same, plus one group holding the curvature x
    num_reported_accidents pair. A literal list is passed straight through."""
    if spec is None:
        return None
    if isinstance(spec, list):
        return spec
    if spec == "additive":
        return [[c] for c in feats]
    if spec == "additive+curv_acc":
        pair = [c for c in CURV_ACC_PAIR if c in feats]
        groups = [[c] for c in feats]
        if len(pair) == 2:
            groups.append(pair)
        return groups
    raise ValueError(f"unknown interaction_constraints spec {spec!r}")


# --------------------------------------------------------------------------- models
def _run_lgb(params, feats):
    import lightgbm as lgb
    X, Xt, _ = get_feature_frame()
    X, Xt = X[feats].copy(), Xt[feats].copy()
    p = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63,
             min_child_samples=40, feature_fraction=0.9, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=NTHREAD, seed=SEED,
             deterministic=True, force_row_wise=True)
    p.update(params)
    n_round = p.pop("num_boost_round", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    # linear_tree is incompatible with LightGBM's categorical handling -> plain int codes
    cats = [] if p.get("linear_tree") else [c for c in CATEGORICAL if c in feats]
    for c in cats:
        X[c] = X[c].astype("category")
        Xt[c] = Xt[c].astype("category")
    ic = _expand_constraints(p.pop("interaction_constraints", None), feats)
    if ic is not None:
        # LightGBM's interaction_constraints takes feature INDICES. Passing names is
        # accepted silently and yields an empty constraint set -> no split is ever legal
        # -> a constant model scoring the mean baseline (0.166379 here). Convert.
        p["interaction_constraints"] = [[feats.index(c) for c in g] for g in ic]
    oof, pred, iters = np.zeros(len(_y)), np.zeros(len(Xt)), []
    for tr, va in _FOLDS:
        dtr = lgb.Dataset(X.iloc[tr], label=_y[tr], categorical_feature=cats)
        dva = lgb.Dataset(X.iloc[va], label=_y[va], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=n_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(esr, verbose=False), lgb.log_evaluation(0)])
        iters.append(int(m.best_iteration))
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(Xt, num_iteration=m.best_iteration) / N_SPLITS
    return _clip(oof), _clip(pred), iters


def _run_xgb(params, feats):
    import xgboost as xgb
    X, Xt, _ = get_feature_frame()
    X, Xt = X[feats], Xt[feats]
    p = dict(objective="reg:squarederror", eval_metric="rmse", learning_rate=0.05,
             max_depth=7, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
             min_child_weight=20, random_state=SEED, n_jobs=NTHREAD, tree_method="hist")
    p.update(params)
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    ic = _expand_constraints(p.pop("interaction_constraints", None), feats)
    if ic is not None:
        p["interaction_constraints"] = [[feats.index(c) for c in g] for g in ic]
    Xn, Xtn = X.to_numpy(np.float32), Xt.to_numpy(np.float32)
    oof, pred, iters = np.zeros(len(_y)), np.zeros(len(Xtn)), []
    for tr, va in _FOLDS:
        m = xgb.XGBRegressor(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(Xn[tr], _y[tr], eval_set=[(Xn[va], _y[va])], verbose=False)
        iters.append(int(m.best_iteration))
        oof[va] = m.predict(Xn[va])
        pred += m.predict(Xtn) / N_SPLITS
    return _clip(oof), _clip(pred), iters


def _run_cat(params, feats):
    from catboost import CatBoostRegressor, Pool
    X, Xt, _ = get_feature_frame()
    X, Xt = X[feats].copy(), Xt[feats].copy()
    cats = [c for c in CATEGORICAL if c in feats]
    for c in cats:
        X[c] = X[c].astype(str)
        Xt[c] = Xt[c].astype(str)
    # cheaper CatBoost defaults than stage 3's: a 3000-iteration / lr 0.06 node cost
    # 163-337s in the first search attempt, which would eat the whole node budget.
    p = dict(loss_function="RMSE", eval_metric="RMSE", learning_rate=0.12, depth=8,
             l2_leaf_reg=3.0, random_seed=SEED, verbose=False, allow_writing_files=False,
             thread_count=NTHREAD)
    p.update(params)
    n_it = p.pop("iterations", 1200)
    esr = p.pop("early_stopping_rounds", 60)
    oof, pred, iters = np.zeros(len(_y)), np.zeros(len(Xt)), []
    test_pool = Pool(Xt, cat_features=cats)
    for tr, va in _FOLDS:
        m = CatBoostRegressor(iterations=n_it, **p)
        m.fit(Pool(X.iloc[tr], _y[tr], cat_features=cats),
              eval_set=Pool(X.iloc[va], _y[va], cat_features=cats),
              early_stopping_rounds=esr, verbose=False)
        iters.append(int(m.get_best_iteration()))
        oof[va] = m.predict(X.iloc[va])
        pred += m.predict(test_pool) / N_SPLITS
    return _clip(oof), _clip(pred), iters


_ADD_CACHE = {}


def _additive_design(n_knots=12):
    key = int(n_knots)
    if key in _ADD_CACHE:
        return _ADD_CACHE[key]
    catlike = ["road_type", "lighting", "weather", "time_of_day", "road_signs_present",
               "public_road", "holiday", "school_season", "num_lanes", "speed_limit",
               "num_reported_accidents"]
    both = pd.concat([_train[catlike].astype(str), _test[catlike].astype(str)], axis=0)
    D = pd.get_dummies(both, drop_first=True).astype(np.float32).to_numpy()
    ntr = len(_train)
    cv_tr = _train["curvature"].to_numpy(np.float64)
    knots = np.quantile(cv_tr, np.linspace(0.05, 0.95, key))

    def spline(v):
        return np.column_stack([v] + [np.maximum(v - k, 0.0) for k in knots]).astype(np.float32)

    res = (np.hstack([D[:ntr], spline(cv_tr)]),
           np.hstack([D[ntr:], spline(_test["curvature"].to_numpy(np.float64))]))
    _ADD_CACHE[key] = res
    return res


def _run_ridge(params, feats):
    from sklearn.linear_model import Ridge
    A, At = _additive_design(params.get("n_knots", 12))
    alpha = params.get("alpha", 1.0)
    oof, pred = np.zeros(len(_y)), np.zeros(len(At))
    for tr, va in _FOLDS:
        m = Ridge(alpha=alpha).fit(A[tr], _y[tr])
        oof[va] = m.predict(A[va])
        pred += m.predict(At) / N_SPLITS
    return _clip(oof), _clip(pred), []


_RUNNERS = {"lgb": _run_lgb, "xgb": _run_xgb, "cat": _run_cat, "ridge": _run_ridge}


def evaluate_solo(config):
    drop = (config.get("features") or {}).get("drop", [])
    _, _, feats = get_feature_frame(drop)
    model = config["model"]
    if model not in _RUNNERS:
        raise ValueError(f"unknown model type {model!r}")
    oof, pred, iters = _RUNNERS[model](dict(config.get("params") or {}), feats)
    return oof, pred, rmse(_y, oof), feats, iters


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    best_w, best_s, _ = hv3.eval_blend(CACHE_DIR, members, lambda v: rmse(_y, _clip(v)),
                                       weight_search=method)
    return dict(members=list(members), weights=[round(float(w), 4) for w in best_w],
                method=method, rmse=round(float(best_s), 6)), float(best_s)


def evaluate(config: dict, node_id: int = None, timeout_s: int = 600) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises; a timeout or any
    exception becomes status="failed" so the search loop keeps going.
    score IS the RMSE (minimize-better == harness convention, no sign flip)."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old = None
    if have_alarm:
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, feats, iters = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, rmse=score)
            return dict(status="evaluated", score=round(score, 6),
                        wall_s=round(time.time() - t0, 1),
                        result={"n_feats": len(feats), "rmse": round(score, 6),
                                "best_iters": iters}, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            if node_id is not None:
                oofs = np.stack([hv2.load_oof(CACHE_DIR, m) for m in result["members"]], axis=1)
                preds = np.stack([np.load(os.path.join(CACHE_DIR, f"solo_{m}.npz"))["pred"]
                                  for m in result["members"]], axis=1)
                w = np.asarray(result["weights"], dtype=float)
                hv2.cache_oof(CACHE_DIR, node_id, _clip(oofs @ w), pred=_clip(preds @ w),
                              rmse=score)
            return dict(status="evaluated", score=round(score, 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
        raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - evaluator must never crash the search loop
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


BASE_DROP = [c for c in ALL_FEATURES if c not in BASE_FEATURES]
