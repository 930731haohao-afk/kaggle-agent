"""tree_search/eval_s3e16_run5.py -- per-competition evaluator for playground-series-s3e16
(crab Age, iid tabular regression, integer target, MAE metric, MINIMIZE-better).

Written fresh for this run. It deliberately does not read, import or reuse any artifact of
a previous run of this competition (lane isolation + SKILL.md's "never read the answers
this competition already produced" rule); the cache dir is this run's own.

Two node kinds, the standard v2/v3 schema:

  1. solo  -- {kind:"solo", model:"lgb"|"xgb"|"cat"|"huber", params:{...},
               features:{"set":"raw"|"core"|"eng"|"full", "drop":[...]}}
     Trains one regressor on the canonical folds from competitions/.../scripts/features.py
     (imported, never re-implemented): StratifiedKFold(5, shuffle, seed=42) on the
     tail-capped integer target.

  2. blend -- {kind:"blend", members:[<solo node id>...], k:800}
     Loads the members' cached OOFs; no retraining.

THE METRIC. Age is an integer, so the score is MAE on the POST-PROCESSED prediction
(default mode "round": clip to [1,29] then round to nearest int), never on the raw
continuous OOF -- 07_tree_search.md's rule for discretized decision metrics, whose failure
mode is "raw improves, rounded gets worse". `result` carries raw/clip/round/snap side by
side so the mirror-failure is always visible, but only the post-processed number is the
score.

BLEND HONESTY. Blend weights are a fitted parameter. Fitting them on the same OOF vector
that then scores them is optimistic, so every blend node reports the LEAVE-FOLD-OUT score
(for each fold, weights come from the other four folds' rows) as its score, with
`result["insample_mae"]` and `result["optimism"]` alongside it as the measurement of how
optimistic the in-sample number would have been.

SIGN CONVENTION: MAE is minimize-better and the harness assumes lower-is-better, so the
score is handed over unnegated. `result["mae"]` is the same number, human-readable.
"""

import os
import signal
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e16")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _HERE)
import features as F  # noqa: E402
import harness_v2 as hv2  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_s3e16_run5")
os.makedirs(CACHE_DIR, exist_ok=True)
NUM_THREADS = 10
SEED = F.SEED
PP = "round"

_FRAME_CACHE = {}


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def load_pred(node_id):
    """Load node_id's cached TEST prediction (companion of hv2.load_oof)."""
    path = os.path.join(CACHE_DIR, f"solo_{node_id}.npz")
    if not os.path.exists(path):
        raise ValueError(f"node #{node_id} has no cache at {path}")
    d = np.load(path, allow_pickle=True)
    if "pred" not in d or d["pred"].size == 0:
        raise ValueError(f"node #{node_id} has no cached test prediction")
    return d["pred"]


def _base(feature_set):
    if feature_set not in _FRAME_CACHE:
        _FRAME_CACHE[feature_set] = F.load(feature_set)
    return _FRAME_CACHE[feature_set]


def get_feature_frame(feat_cfg):
    feat_cfg = feat_cfg or {}
    fs = feat_cfg.get("set", "eng")
    X, y, Xte, _ids = _base(fs)
    drop = set(feat_cfg.get("drop") or [])
    unknown = drop - set(X.columns)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {sorted(unknown)}")
    keep = [c for c in X.columns if c not in drop]
    if not keep:
        raise ValueError("drop list removed every feature")
    return X[keep], y, Xte[keep], keep


# ---------------------------------------------------------------------------
# solo runners
# ---------------------------------------------------------------------------
def _run_lgb(params, X, y, Xte, folds):
    import lightgbm as lgb
    p = dict(objective="regression_l1", metric="mae", learning_rate=0.03,
             num_leaves=63, min_child_samples=40, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, reg_alpha=0.1, reg_lambda=1.0,
             verbose=-1, num_threads=NUM_THREADS, deterministic=True,
             force_row_wise=True, seed=SEED)
    p.update(params)
    n_rounds = int(p.pop("num_boost_round", 4000))
    esr = int(p.pop("early_stopping_rounds", 150))
    oof = np.zeros(len(y))
    pred = np.zeros(len(Xte))
    for tr, va in folds:
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(esr, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(Xte, num_iteration=m.best_iteration) / len(folds)
    return oof, pred


def _run_xgb(params, X, y, Xte, folds):
    import xgboost as xgb
    p = dict(objective="reg:absoluteerror", eval_metric="mae", learning_rate=0.03,
             max_depth=7, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
             reg_lambda=1.0, min_child_weight=10, random_state=SEED,
             n_jobs=NUM_THREADS, tree_method="hist")
    p.update(params)
    n_est = int(p.pop("n_estimators", 4000))
    esr = int(p.pop("early_stopping_rounds", 150))
    oof = np.zeros(len(y))
    pred = np.zeros(len(Xte))
    for tr, va in folds:
        m = xgb.XGBRegressor(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = m.predict(X.iloc[va])
        pred += m.predict(Xte) / len(folds)
    return oof, pred


def _run_cat(params, X, y, Xte, folds):
    from catboost import CatBoostRegressor
    p = dict(loss_function="MAE", eval_metric="MAE", learning_rate=0.06, depth=7,
             l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
             allow_writing_files=False, thread_count=NUM_THREADS)
    p.update(params)
    n_est = int(p.pop("iterations", 3000))
    esr = int(p.pop("early_stopping_rounds", 150))
    oof = np.zeros(len(y))
    pred = np.zeros(len(Xte))
    for tr, va in folds:
        m = CatBoostRegressor(iterations=n_est, **p)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]),
              early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict(X.iloc[va])
        pred += m.predict(Xte) / len(folds)
    return oof, pred


def _run_huber(params, X, y, Xte, folds):
    """Bias-dominated linear decorrelator (TASK-BLEND-DECOR)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    pred = np.zeros(len(Xte))
    for tr, va in folds:
        pipe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            HuberRegressor(alpha=float(params.get("alpha", 1.0)),
                           epsilon=float(params.get("epsilon", 1.35)), max_iter=500))
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict(X.iloc[va])
        pred += pipe.predict(Xte) / len(folds)
    return oof, pred


RUNNERS = {"lgb": _run_lgb, "xgb": _run_xgb, "cat": _run_cat, "huber": _run_huber}


def evaluate_solo(config):
    X, y, Xte, feats = get_feature_frame(config.get("features"))
    model = config["model"]
    runner = RUNNERS.get(model)
    if runner is None:
        raise ValueError(f"unknown model type {model!r}")
    folds = F.make_folds(y)
    oof, pred = runner(dict(config.get("params") or {}), X, y, Xte, folds)
    ynp = y.to_numpy(dtype=np.float64)
    scores = {m: F.mae_pp(oof, ynp, m) for m in ("none", "clip", "round", "snap")}
    per_fold = [float(np.abs(F.postprocess(oof[va], PP) - ynp[va]).mean())
                for _, va in folds]
    return oof, pred, scores, per_fold, feats


# ---------------------------------------------------------------------------
# blend node -- weights fitted leave-fold-out
# ---------------------------------------------------------------------------
def _fit_weights(oofs_sub, y_sub, k=800, ascent_rounds=3, seed=SEED):
    rng = np.random.default_rng(seed)
    n = oofs_sub.shape[0]

    def sc(w):
        return F.mae_pp(oofs_sub.T @ w, y_sub, PP)

    cands = [np.full(n, 1.0 / n)]
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        cands.append(e)
    cands.extend(rng.dirichlet(np.ones(n), size=k))
    scores = [sc(w) for w in cands]
    bi = int(np.argmin(scores))
    w, s = np.asarray(cands[bi], float), scores[bi]
    for _ in range(ascent_rounds):
        improved = False
        for i in range(n):
            for d in (0.08, 0.04, 0.02, -0.02, -0.04, -0.08):
                w2 = w.copy()
                w2[i] = max(0.0, w2[i] + d)
                t = w2.sum()
                if t <= 0:
                    continue
                w2 /= t
                s2 = sc(w2)
                if s2 < s - 1e-12:
                    w, s, improved = w2, s2, True
        if not improved:
            break
    return w, s


def evaluate_blend(config):
    members = list(config.get("members") or [])
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    oofs = np.vstack([hv2.load_oof(CACHE_DIR, m) for m in members])
    k = int(config.get("k", 800))
    _X, y, _Xte, _f = _base("raw")
    ynp = y.to_numpy(dtype=np.float64)
    folds = F.make_folds(y)

    w_in, s_in = _fit_weights(oofs, ynp, k=k)

    err = np.zeros(len(ynp))
    for _, va in folds:
        mask = np.ones(len(ynp), dtype=bool)
        mask[va] = False
        w_f, _ = _fit_weights(oofs[:, mask], ynp[mask], k=k)
        err[va] = np.abs(F.postprocess(oofs[:, va].T @ w_f, PP) - ynp[va])
    honest = float(err.mean())
    per_fold = [float(err[va].mean()) for _, va in folds]

    blend_oof = oofs.T @ w_in
    result = dict(members=members, weights=[round(float(x), 4) for x in w_in],
                  mae=round(honest, 6), insample_mae=round(s_in, 6),
                  optimism=round(s_in - honest, 6), k=k, per_fold=per_fold,
                  raw_mae=round(F.mae_pp(blend_oof, ynp, "none"), 6),
                  snap_mae=round(F.mae_pp(blend_oof, ynp, "snap"), 6))
    return result, honest, blend_oof, w_in


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 900) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises.

    score is the honest POST-PROCESSED (round-to-int) MAE; lower is better, which already
    matches the harness sign convention.
    """
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old = None
    if have_alarm:
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_s))
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, scores, per_fold, feats = evaluate_solo(config)
            score = scores[PP]
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, mae=score)
            result = {"n_feats": len(feats), "mae": round(score, 6),
                      "raw_mae": round(scores["none"], 6),
                      "clip_mae": round(scores["clip"], 6),
                      "snap_mae": round(scores["snap"], 6),
                      "per_fold": [round(v, 6) for v in per_fold],
                      "fold_sd": round(float(np.std(per_fold)), 6)}
            return dict(status="evaluated", score=round(score, 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
        if kind == "blend":
            result, honest, blend_oof, _w = evaluate_blend(config)
            if node_id is not None:
                # Cache the in-sample-weighted blend OOF and the matching weighted test
                # prediction, so a blend node can itself be a member of a later blend and
                # so the final submission can be rebuilt from the node id alone. The SCORE
                # stays the honest leave-fold-out number.
                test_pred = np.zeros(0)
                try:
                    test_pred = sum(_w[i] * load_pred(m)
                                    for i, m in enumerate(result["members"]))
                except ValueError:
                    pass  # a member without a cached test pred (e.g. a nested blend)
                hv2.cache_oof(CACHE_DIR, node_id, blend_oof, pred=test_pred, mae=honest)
            return dict(status="evaluated", score=round(honest, 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
        raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - the evaluator must never kill the search loop
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)
