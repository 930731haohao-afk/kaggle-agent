"""tree_search/eval_s3e14.py — per-competition evaluator for playground-series-s3e14
(Wild Blueberry Yield, MAE, minimize).

Phase C-2b innovation vs eval_s3e9.py: TWO node kinds instead of one.

  1. solo  — {kind:"solo", model, features:{drop:[...]}, params, postprocess:{snap}}
     Trains one model with 5-fold KFold(seed=42) exactly like scripts/train_v2.py etc.
     Cheap (~5-60s depending on model). If `node_id` is given, its OOF/test prediction
     matrix is cached to tree_search/cache_s3e14/solo_<node_id>.npz so later blend nodes
     never need to retrain it.

  2. blend — {kind:"blend", members:[<solo node id>, ...], weight_search:"dirichlet"|
     "grid_simplex", postprocess:{snap}}
     Loads each member's cached OOF matrix (must already be evaluated -- raises a clear
     error otherwise, caught by evaluate() as a "failed" node so the search loop can keep
     going), weight-searches the blend, applies snap-to-grid, returns the score. No
     retraining at all -- this is what makes ensemble nodes cost seconds instead of
     20-60s, and is the fix for s3e9's key lesson (single-model-only node space can't
     reach where linear iteration won: <score> via a 5-way blend).

Reuses competitions/playground-series-s3e14/scripts/features.py UNMODIFIED and the exact
same CV scheme as scripts/train_v2.py-train_v6.py (5-fold KFold, shuffle, seed=42) so
scores are directly comparable to STATUS.md's numbers (root solo LGB ~<score>, XGB
~<score>, CAT(native-cat) ~<score>, Optuna-tuned LGB ~<score>, linear-iteration best
blend <score>).
"""
import json
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e14")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
import stage2_inputs  # noqa: E402
stage2_inputs.require_module(_SCRIPTS_DIR, "features", comp="playground-series-s3e14",
                             exposes=['build_features', 'feature_columns'])
from features import build_features, feature_columns  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s3e14")
TARGET, ID = "yield", "id"
N_SPLITS, SEED = 5, 42

# Already-proven-dead engineered/raw features (STATUS.md exp #2 -> #3, near-zero LGB gain
# importance). Kept as the default drop set for every node's root/baseline feature set --
# NOT re-tested as a node mutation (experience.md: don't re-test dead ends).
NOISE_FEATS = ["temp_avg", "log_clonesize", "rain_intensity",
               "MinOfLowerTRange", "AverageOfLowerTRange", "AverageOfUpperTRange"]
# Low-cardinality env columns CatBoost can treat as native categoricals (STATUS.md exp #2/#3).
CAT_COLS = ["clonesize", "RainingDays", "AverageRainingDays", "honeybee", "bumbles", "andrena", "osmia"]

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_Xtr_full = build_features(_train)
_Xte_full = build_features(_test)
ALL_FEATURES = feature_columns(_Xtr_full)
_y = _train[TARGET].to_numpy(np.float64)

_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_kf.split(np.zeros(len(_y))))  # identical to scripts/train_v2.py..train_v6.py
_TRAIN_SORTED_Y = np.sort(np.unique(_y))


def mae(a, b):
    return float(mean_absolute_error(a, b))


def snap_to_grid(preds, grid=_TRAIN_SORTED_Y):
    """Snap each prediction to the nearer of its two bracketing observed train yield
    values. Proven winner in every s3e14 iteration so far (STATUS.md); not re-tested as
    a mutation, just applied via the postprocess.snap config flag."""
    idx = np.searchsorted(grid, preds)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    return np.where(np.abs(preds - left) <= np.abs(preds - right), left, right)


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
# solo-model runners (mirrors scripts/train_v2.py / train_v3.py exactly)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    p = dict(objective="regression_l1", metric="mae", n_jobs=-1, verbose=-1, random_state=SEED)
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
    p = dict(objective="reg:absoluteerror", n_jobs=-1, random_state=SEED, eval_metric="mae")
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


def _run_cat(params, Xdf, Xtestdf, cat_names):
    from catboost import CatBoostRegressor, Pool
    p = dict(loss_function="MAE", eval_metric="MAE", random_seed=SEED,
             thread_count=-1, verbose=False, allow_writing_files=False)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 4000))
    esr = p.pop("early_stopping_rounds", 200)
    native_cat = p.pop("native_cat", False)
    if native_cat and cat_names:
        Xc, Xtc = Xdf.copy(), Xtestdf.copy()
        for c in cat_names:
            Xc[c] = Xc[c].astype(str)
            Xtc[c] = Xtc[c].astype(str)
        cats = cat_names
    else:
        Xc, Xtc, cats = Xdf, Xtestdf, []
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtestdf))
    for tr, va in _FOLDS:
        m = CatBoostRegressor(iterations=n_est, **p)
        m.fit(Pool(Xc.iloc[tr], _y[tr], cat_features=cats),
              eval_set=Pool(Xc.iloc[va], _y[va], cat_features=cats),
              early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict(Xc.iloc[va])
        pred += m.predict(Xtc) / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    drop = (config.get("features") or {}).get("drop", [])
    Xdf, Xtestdf, feats = get_feature_frame(drop)
    model = config["model"]
    params = dict(config.get("params") or {})
    if model == "lgb":
        oof, pred = _run_lgb(params, Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32))
    elif model == "xgb":
        oof, pred = _run_xgb(params, Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32))
    elif model == "cat":
        cat_names = [c for c in CAT_COLS if c in feats]
        oof, pred = _run_cat(params, Xdf, Xtestdf, cat_names)
    else:
        raise ValueError(f"unknown model type {model!r}")

    pp = config.get("postprocess") or {}
    # METRIC SYMMETRY (v5.3, 2026-08-03 audit): solo defaulted snap=False while blend
    # defaulted snap=True, so the two node kinds were ranked on different metrics and then
    # compared with a single min(). Both now default to SNAP ON -- snapping to the observed
    # target grid can only help MAE on this competition (run_s3e14's own note: the same
    # blend is <score> raw vs <score> snapped), so it is what a submission would do.
    use_snap = pp.get("snap", True)
    score_oof = snap_to_grid(oof) if use_snap else oof
    score = mae(_y, score_oof)
    return oof, pred, score, feats


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over CACHED member OOF matrices, no retraining
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
    if not np.allclose(d["y"], _y):
        raise ValueError(f"cached y for solo node #{node_id} does not match current target "
                          f"vector -- fold/data drift, cache is stale")
    return d


def _grid_simplex_weights(n_members, step=0.05):
    if n_members > 5:
        raise ValueError(f"grid_simplex only supported up to 5 members ({n_members} given) -- "
                          f"combinatorial blowup, use weight_search='dirichlet' instead")
    import itertools
    grid_vals = np.arange(0, 1.0 + 1e-9, step)
    rows = []
    for combo in itertools.product(grid_vals, repeat=n_members - 1):
        rem = 1.0 - sum(combo)
        if rem < -1e-9:
            continue
        rows.append(combo + (max(rem, 0.0),))
    return np.array(rows)


def weight_search(oofs, y, method="dirichlet", seed=42, k=3000, metric_fn=None):
    """oofs: (n_samples, n_members). Returns (best_weights, best_score).

    `metric_fn(blended_vector) -> score` scores every candidate on the POST-PROCESSED
    vector. Without it the search minimizes raw MAE and any post-processing is applied
    only to the winner, so the reported score is not the one the search optimized
    (2026-08-03 audit).
    """
    n = oofs.shape[1]
    if n == 1:
        w = np.array([1.0])
        return w, (metric_fn(oofs[:, 0]) if metric_fn else mae(y, oofs[:, 0]))

    if metric_fn is not None:
        # Post-processed metrics are discretized, so the fast vectorized paths below do not
        # apply: score candidates one at a time through the real metric.
        rng = np.random.default_rng(seed)
        W = rng.dirichlet(np.ones(n), size=k)
        W = np.vstack([W, np.eye(n)])              # include the unit vectors
        scores = np.array([metric_fn(oofs @ w) for w in W])
        i = int(scores.argmin())
        best_w = W[i]
        conc = np.clip(best_w, 1e-3, None) * 200.0
        W2 = rng.dirichlet(conc, size=k)
        s2 = np.array([metric_fn(oofs @ w) for w in W2])
        if s2.min() < scores[i]:
            best_w = W2[int(s2.argmin())]
        return best_w, float(metric_fn(oofs @ best_w))

    if method == "dirichlet":
        rng = np.random.default_rng(seed)
        oofs32 = oofs.astype(np.float32)
        y32 = y.astype(np.float32)

        def best_of(W):
            s = np.abs(oofs32 @ W.T - y32[:, None]).mean(axis=0)
            i = int(s.argmin())
            return W[i], float(s[i])

        W1 = rng.dirichlet(np.ones(n), size=k)
        best_w, _ = best_of(W1)
        # refine: concentrate a second sampling round around the coarse best
        conc = np.clip(best_w, 1e-3, None) * 200.0
        W2 = rng.dirichlet(conc, size=k)
        cand_w, _ = best_of(np.vstack([W1, W2]))
        best_w = cand_w
        # exact (float64) score for the chosen weights
        best_s = mae(y, oofs @ best_w)
        return best_w, best_s
    elif method == "grid_simplex":
        W = _grid_simplex_weights(n)
        s = np.abs(oofs.astype(np.float32) @ W.T.astype(np.float32) - y.astype(np.float32)[:, None]).mean(axis=0)
        i = int(s.argmin())
        best_w = W[i]
        best_s = mae(y, oofs @ best_w)
        return best_w, best_s
    else:
        raise ValueError(f"unknown weight_search method {method!r}")


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    mats, names = [], []
    for mid in members:
        d = load_solo_cache(mid)
        mats.append(d["oof"])
        names.append(f"node{mid}")
    oofs = np.stack(mats, axis=1)
    method = config.get("weight_search", "dirichlet")
    pp = config.get("postprocess") or {}
    use_snap = pp.get("snap", True)
    # Snap INSIDE the search, not after it: weights chosen on raw MAE are not the weights
    # that minimize snapped MAE, so the old order left free gain on the table AND reported a
    # score the search had not optimized (2026-08-03 audit).
    w, _searched = weight_search(
        oofs, _y, method=method,
        metric_fn=(lambda v: mae(_y, snap_to_grid(v))) if use_snap else None)
    blend_oof = oofs @ w
    raw_score = mae(_y, blend_oof)
    snap_score = mae(_y, snap_to_grid(blend_oof)) if use_snap else raw_score
    final_score = snap_score if use_snap else raw_score
    return dict(members=members, names=names, weights=w.tolist(), method=method,
                raw_score=round(raw_score, 5), snap_score=round(snap_score, 5),
                used_snap=bool(use_snap)), final_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 180) -> dict:
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
            oof, pred, score, feats = evaluate_solo(config)
            if node_id is not None:
                _save_solo_cache(node_id, oof, pred, feats, config, score)
            wall = time.time() - t0
            return dict(status="evaluated", score=round(score, 5), wall_s=round(wall, 1),
                        result={"n_feats": len(feats)}, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            wall = time.time() - t0
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
