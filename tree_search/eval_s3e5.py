"""tree_search/eval_s3e5.py — per-competition evaluator for playground-series-s3e5
(Wine Quality, ordinal regression target, QWK metric, DISCRETIZED and maximize-better).

Phase C-2c innovation vs eval_s3e9.py (single-model, RMSE) and eval_s3e14.py (solo+blend,
MAE): this is the first tree-search comp with a DISCRETIZED metric. Per
knowledge/experience.md's s3e16/s3e5 lesson, EVERY node's score is the QWK *after*
OptimizedRounder cutpoint-fitting on that node's own OOF vector — the rounder is NOT a
node type or a config flag, it is baked into the evaluator itself so no node can ever be
scored/compared on a pre-rounder (raw regression) number.

IMPORTANT sign convention: harness.py assumes lower-is-better scores throughout (its
docstring: "flip sign before calling in if the comp metric is maximize-better"). QWK is
maximize-better, so every score this module hands to harness.add_node/add_root is
`-QWK`. `result["qwk"]` (and printouts in run_s3e5.py) carry the real, human-readable QWK.

Two node kinds, same schema as eval_s3e14.py:

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...}, features:{drop:[...]}}
     Trains one regressor with 5-fold StratifiedKFold(quality, shuffle, seed=42) — EXACT
     same CV as competitions/playground-series-s3e5/scripts/{train.py,iterate.py}. Cheap
     (~1-5s on this 2056-row dataset). If `node_id` is given, its OOF/test prediction
     vector is cached to tree_search/cache_s3e5/solo_<node_id>.npz so blend nodes never
     retrain.

  2. blend — {kind:"blend", members:[<solo node id>, ...], weight_search:"dirichlet"}
     Loads cached member OOF vectors, Dirichlet-searches blend weights (objective =
     post-rounder QWK of the blended OOF, matching scripts/iterate.py's weight_search
     exactly), returns the QWK. No retraining -> blend nodes cost a fraction of a second.

Reuses competitions/playground-series-s3e5/data/{train,test}_processed.csv (the exact,
already-feature-engineered 21-column output of scripts/features.py, UNMODIFIED) so scores
are directly comparable to STATUS.md's numbers (root tuned-LGB solo ~0.56244, tuned-CAT
solo ~0.56466, linear-iteration best 6-way blend 0.56769).
"""
import json
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e5")
DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s3e5")
TARGET, ID = "quality", "Id"
N_SPLITS, SEED = 5, 42

_train = pd.read_csv(os.path.join(DATA, "train_processed.csv"))
_test = pd.read_csv(os.path.join(DATA, "test_processed.csv"))
ALL_FEATURES = [c for c in _train.columns if c not in (ID, TARGET)]
_y = _train[TARGET].to_numpy(int)
LOW, HIGH = int(_y.min()), int(_y.max())

_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(np.zeros(len(_y)), _y))  # identical to scripts/train.py & iterate.py


# ---------------------------------------------------------------------------
# OptimizedRounder + post-rounder QWK — ported verbatim from scripts/iterate.py.
# This IS the evaluator, applied to every solo AND blend node; never a node config option.
# ---------------------------------------------------------------------------
class OptimizedRounder:
    def __init__(self, low: int, high: int):
        self.low, self.high = low, high
        self.coef_ = np.arange(low + 0.5, high, 1.0)

    def _to_classes(self, x, coef):
        coef = np.sort(coef)
        return np.clip(np.digitize(x, coef) + self.low, self.low, self.high)

    def _loss(self, coef, x, y):
        return -cohen_kappa_score(y, self._to_classes(x, coef), weights="quadratic")

    def fit(self, x, y):
        res = minimize(self._loss, self.coef_, args=(x, y), method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
        self.coef_ = np.sort(res.x)
        return self

    def predict(self, x):
        return self._to_classes(x, self.coef_)


def qwk(y_true, y_pred):
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def post_rounder_qwk(oof_vec):
    """Fit OptimizedRounder on the full 5-fold OOF vector and return (QWK, cutpoints).
    THE ONLY score any node in this comp's tree is ever compared on."""
    r = OptimizedRounder(LOW, HIGH).fit(oof_vec, _y)
    return qwk(_y, r.predict(oof_vec)), r.coef_.copy()


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
    return _train[feats].copy(), _test[feats].copy(), feats


# ---------------------------------------------------------------------------
# solo-model runners (mirrors scripts/iterate.py's run_lgb/run_xgb/run_cat exactly;
# no early stopping / eval_set — original recipe uses fixed n_estimators, data is tiny)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    p = dict(objective="regression", n_estimators=1500, learning_rate=0.03,
             num_leaves=31, max_depth=6, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=15,
             random_state=SEED, n_jobs=-1, verbose=-1)
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = lgb.LGBMRegressor(**p)
        m.fit(X[tr], _y[tr])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xtest):
    import xgboost as xgb
    p = dict(objective="reg:squarederror", n_estimators=1500, learning_rate=0.03,
              max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
              min_child_weight=5, random_state=SEED, n_jobs=-1, tree_method="hist")
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBRegressor(**p)
        m.fit(X[tr], _y[tr])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest):
    from catboost import CatBoostRegressor
    p = dict(loss_function="RMSE", iterations=1500, learning_rate=0.03, depth=6,
             l2_leaf_reg=3.0, random_seed=SEED, verbose=False, allow_writing_files=False,
             thread_count=-1)
    p.update(params)
    if "n_estimators" in p:  # allow the more familiar sklearn alias in mutation configs
        p["iterations"] = p.pop("n_estimators")
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostRegressor(**p)
        m.fit(X[tr], _y[tr])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    drop = (config.get("features") or {}).get("drop", [])
    Xdf, Xtestdf, feats = get_feature_frame(drop)
    model = config["model"]
    params = dict(config.get("params") or {})
    Xnp, Xtestnp = Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32)
    if model == "lgb":
        oof, pred = _run_lgb(params, Xnp, Xtestnp)
    elif model == "xgb":
        oof, pred = _run_xgb(params, Xnp, Xtestnp)
    elif model == "cat":
        oof, pred = _run_cat(params, Xnp, Xtestnp)
    else:
        raise ValueError(f"unknown model type {model!r}")

    score, coef = post_rounder_qwk(oof)
    return oof, pred, score, coef, feats


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over CACHED member OOF vectors, no retraining
# ---------------------------------------------------------------------------
def _cache_path(node_id):
    return os.path.join(CACHE_DIR, f"solo_{node_id}.npz")


def _save_solo_cache(node_id, oof, pred, feats, config, score):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(node_id)
    tmp = path.replace(".npz", ".tmp.npz")
    np.savez(tmp, oof=oof, pred=pred, y=_y, feats=np.array(feats, dtype=object),
              score=score, model=config.get("model"), config_json=json.dumps(config))
    os.replace(tmp, path)


def load_solo_cache(node_id):
    path = _cache_path(node_id)
    if not os.path.exists(path):
        raise ValueError(f"solo node #{node_id} has no cached OOF at {path} -- it must be "
                          f"evaluated (kind='solo') before being referenced as a blend member")
    d = np.load(path, allow_pickle=True)
    if not np.array_equal(d["y"], _y):
        raise ValueError(f"cached y for solo node #{node_id} does not match current target "
                          f"vector -- fold/data drift, cache is stale")
    return d


def weight_search(oof_dict, n_random=800, seed=0):
    """Dirichlet random search + coordinate-ascent refinement over the member simplex,
    scored on post-rounder QWK. Ported verbatim from scripts/iterate.py's weight_search
    (generalizes the exp#3 grid search to arbitrary pool sizes)."""
    names = list(oof_dict)
    oofs = np.stack([oof_dict[n] for n in names], axis=1)
    rng = np.random.default_rng(seed)
    best_w, best_s, best_coef = None, -1e18, None

    def evalw(w):
        blend = oofs @ w
        s, coef = post_rounder_qwk(blend)
        return s, coef

    candidates = [np.eye(len(names))[i] for i in range(len(names))]
    candidates.append(np.full(len(names), 1.0 / len(names)))
    candidates += list(rng.dirichlet(np.ones(len(names)), size=n_random))
    for w in candidates:
        s, coef = evalw(w)
        if s > best_s:
            best_s, best_w, best_coef = s, w, coef
    for _ in range(3):
        improved = False
        for i in range(len(names)):
            for delta in (-0.05, -0.02, 0.02, 0.05):
                w = best_w.copy()
                w[i] = np.clip(w[i] + delta, 0, 1)
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                s, coef = evalw(w)
                if s > best_s:
                    best_s, best_w, best_coef = s, w, coef
                    improved = True
        if not improved:
            break
    return dict(zip(names, best_w.tolist())), best_s, best_coef


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    oof_dict, names = {}, []
    for mid in members:
        d = load_solo_cache(mid)
        name = f"node{mid}"
        oof_dict[name] = d["oof"]
        names.append(name)
    w, score, coef = weight_search(oof_dict)
    return dict(members=members, names=names, weights=dict(zip(members, [w[n] for n in names])),
                method=config.get("weight_search", "dirichlet"),
                cutpoints=[round(float(c), 4) for c in coef]), score, coef


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 120) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or any
    exception is captured as status="failed" so the tree-search loop can keep going.

    score is `-QWK` (harness sign convention: lower-is-better). result["qwk"] is the real,
    human-readable, higher-is-better QWK number -- use THAT for all reporting/printouts."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, coef, feats = evaluate_solo(config)
            if node_id is not None:
                _save_solo_cache(node_id, oof, pred, feats, config, score)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "qwk": round(score, 5),
                      "cutpoints": [round(float(c), 4) for c in coef]}
            return dict(status="evaluated", score=round(-score, 5), wall_s=round(wall, 1),
                        result=result, error=None)
        elif kind == "blend":
            result, score, coef = evaluate_blend(config)
            wall = time.time() - t0
            result["qwk"] = round(score, 5)
            return dict(status="evaluated", score=round(-score, 5), wall_s=round(wall, 1),
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
