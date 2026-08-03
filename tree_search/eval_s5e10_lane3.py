"""tree_search/eval_s5e10_lane3.py — per-competition evaluator for
playground-series-s5e10 (road accident risk, iid tabular regression, RMSE, minimize).

RMSE is already lower-is-better, so scores are handed to the harness UNNEGATED (contrast
eval_s3e7.py, which must pass -AUC). `result["rmse"]` carries the same number for reporting.

Node kinds (the v2/v3 default solo+blend dual track):

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...},
              features:{variant:"v1"|"v2"|"v3"|"v23", drop:[...]}}
     Trains with KFold(5, shuffle, random_state=42) built from
     competitions/playground-series-s5e10/scripts/common.py, i.e. the EXACT folds every
     linear-iteration experiment in this run used, so tree-search scores are directly
     comparable to the linear best. Predictions are clipped to [0, 1] (the train target's
     own range) before scoring. With node_id given, the OOF and test vectors are cached
     to cache_s5e10_lane3/solo_<node_id>.npz so blend nodes never retrain.

  2. blend — {kind:"blend", members:[<solo node id>, ...]}
     Weight search over cached member OOFs. The reported score is the HONEST
     leave-fold-out blend RMSE: weights are refit on the 4 folds outside each held-out
     fold and applied to that fold, so the number is not inflated by weights that saw the
     rows they are scored on. The optimistic fit-on-all-OOF score is carried alongside in
     `result["fitted_all_oof_rmse"]` for reference only.

     The weight search is a local one (`_fit_w`: exact Gram-matrix objective, warm-started
     from equal weights and every pure member, then harness_v3-style coordinate ascent)
     rather than harness_v3.eval_blend's Dirichlet sampler, because the leave-fold-out
     protocol needs SIX weight fits per blend node instead of one, and a squared-error
     objective over the simplex is convex, so the hill-climb reaches the same optimum
     without the 800-draw sampling cost.

CatBoost is always given allow_writing_files=False and an explicit thread_count per the
s3e11 lesson in knowledge/experience.md (default catboost_info file logging stalled a
sandboxed process for 33 minutes).
"""
import os
import signal
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s5e10")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _HERE)
import common as C  # noqa: E402
import features as F  # noqa: E402
import harness_v2 as hv2  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_s5e10_lane3")
N_THREADS = 10

_train, _test = C.load_raw()
_y = _train[C.TARGET].to_numpy(np.float64)
_folds = C.make_folds(len(_train))

_FEAT_CACHE = {}


def _get_features(spec: dict):
    key = (spec.get("variant", "v1"), tuple(spec.get("drop") or []))
    if key not in _FEAT_CACHE:
        _FEAT_CACHE[key] = F.build(_train, _test, variant=key[0], drop=list(key[1]))
    return _FEAT_CACHE[key]


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ---------------------------------------------------------------------------
# solo
# ---------------------------------------------------------------------------
def _fit_lgb(Xtr, Xte, params):
    import lightgbm as lgb
    p = dict(objective="regression", metric="rmse", learning_rate=0.03, num_leaves=96,
             min_child_samples=40, feature_fraction=0.85, bagging_fraction=0.85,
             bagging_freq=1, lambda_l1=0.0, lambda_l2=1.0, n_estimators=6000,
             num_threads=N_THREADS, deterministic=True, force_row_wise=True,
             seed=42, verbose=-1)
    p.update({k: v for k, v in params.items() if k != "seed_offset"})
    p["seed"] = 42 + int(params.get("seed_offset", 0))
    oof, te = np.zeros(len(_y)), np.zeros(len(Xte))
    for tr_i, va_i in _folds:
        m = lgb.LGBMRegressor(**p)
        m.fit(Xtr.iloc[tr_i], _y[tr_i], eval_set=[(Xtr.iloc[va_i], _y[va_i])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_i] = m.predict(Xtr.iloc[va_i])
        te += m.predict(Xte) / len(_folds)
    return oof, te


def _fit_xgb(Xtr, Xte, params):
    import xgboost as xgb
    p = dict(tree_method="hist", max_depth=8, learning_rate=0.03, subsample=0.85,
             colsample_bytree=0.85, min_child_weight=20, reg_lambda=2.0,
             n_estimators=6000, enable_categorical=True, max_cat_to_onehot=8,
             n_jobs=N_THREADS, random_state=42, early_stopping_rounds=100,
             eval_metric="rmse")
    p.update({k: v for k, v in params.items() if k != "seed_offset"})
    p["random_state"] = 42 + int(params.get("seed_offset", 0))
    oof, te = np.zeros(len(_y)), np.zeros(len(Xte))
    for tr_i, va_i in _folds:
        m = xgb.XGBRegressor(**p)
        m.fit(Xtr.iloc[tr_i], _y[tr_i], eval_set=[(Xtr.iloc[va_i], _y[va_i])], verbose=False)
        oof[va_i] = m.predict(Xtr.iloc[va_i])
        te += m.predict(Xte) / len(_folds)
    return oof, te


def _fit_cat(Xtr, Xte, params):
    import catboost as cb
    # CatBoost wants raw strings for cat_features, not pandas `category` codes
    ctr, cte = Xtr.copy(), Xte.copy()
    cat_idx = []
    for i, c in enumerate(ctr.columns):
        if str(ctr[c].dtype) == "category":
            ctr[c] = ctr[c].astype(str)
            cte[c] = cte[c].astype(str)
            cat_idx.append(i)
    p = dict(loss_function="RMSE", depth=8, learning_rate=0.05, l2_leaf_reg=3.0,
             iterations=6000, random_seed=42, thread_count=N_THREADS,
             allow_writing_files=False, od_type="Iter", od_wait=100, verbose=False)
    p.update({k: v for k, v in params.items() if k != "seed_offset"})
    p["random_seed"] = 42 + int(params.get("seed_offset", 0))
    oof, te = np.zeros(len(_y)), np.zeros(len(cte))
    for tr_i, va_i in _folds:
        m = cb.CatBoostRegressor(**p)
        m.fit(ctr.iloc[tr_i], _y[tr_i], cat_features=cat_idx,
              eval_set=(ctr.iloc[va_i], _y[va_i]), use_best_model=True)
        oof[va_i] = m.predict(ctr.iloc[va_i])
        te += m.predict(cte) / len(_folds)
    return oof, te


_FITTERS = {"lgb": _fit_lgb, "xgb": _fit_xgb, "cat": _fit_cat}


def evaluate_solo(config: dict):
    model = config.get("model", "lgb")
    Xtr, Xte, feats = _get_features(config.get("features", {}) or {})
    oof, te = _FITTERS[model](Xtr, Xte, config.get("params", {}) or {})
    oof, te = np.clip(oof, 0.0, 1.0), np.clip(te, 0.0, 1.0)
    return oof, te, _rmse(_y, oof), feats


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------
def _honest_blend(oofs: np.ndarray) -> float:
    """Leave-fold-out blend score: weights refit on the other 4 folds each time."""
    pred = np.zeros(len(_y))
    for _, va_i in _folds:
        mask = np.ones(len(_y), bool)
        mask[va_i] = False
        w = _fit_w(oofs[mask], _y[mask])
        pred[va_i] = oofs[va_i] @ w
    return _rmse(_y, np.clip(pred, 0.0, 1.0))


def _fit_w(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative, sum-to-one weights minimising MSE, via the Gram matrix + the same
    coordinate-ascent refinement harness_v3 uses (cheap: O(m^2) per candidate)."""
    n, m = P.shape
    A = (P.T @ P) / n
    b = (P.T @ y) / n

    def obj(w):
        return float(w @ A @ w - 2.0 * (b @ w))

    best_w = np.full(m, 1.0 / m)
    best = obj(best_w)
    for i in range(m):  # each pure member as a start point
        w = np.zeros(m)
        w[i] = 1.0
        s = obj(w)
        if s < best:
            best, best_w = s, w
    for _ in range(30):
        improved = False
        for i in range(m):
            for d in (0.08, -0.08, 0.03, -0.03, 0.01, -0.01, 0.004, -0.004):
                w = best_w.copy()
                w[i] = max(0.0, w[i] + d)
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                s = obj(w)
                if s < best - 1e-14:
                    best, best_w, improved = s, w, True
        if not improved:
            break
    return best_w


def evaluate_blend(config: dict):
    members = sorted(config["members"])
    oofs = np.stack([hv2.load_oof(CACHE_DIR, m) for m in members], axis=1)
    w_full = _fit_w(oofs, _y)
    fitted = _rmse(_y, np.clip(oofs @ w_full, 0.0, 1.0))
    honest = _honest_blend(oofs)
    equal = _rmse(_y, np.clip(oofs.mean(axis=1), 0.0, 1.0))
    return dict(members=members, weights=[round(float(w), 4) for w in w_full],
                fitted_all_oof_rmse=round(fitted, 6), equal_weight_rmse=round(equal, 6),
                rmse=round(honest, 6)), honest


def rebuild_blend_test(members: list) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (oof, test_pred, info) for a blend, using weights fitted on the FULL OOF.
    Used only by the submission builder, never by the search loop."""
    oofs = np.stack([hv2.load_oof(CACHE_DIR, m) for m in members], axis=1)
    tests = np.stack([np.load(os.path.join(CACHE_DIR, f"solo_{m}.npz"))["pred"]
                      for m in members], axis=1)
    w = _fit_w(oofs, _y)
    oof = np.clip(oofs @ w, 0.0, 1.0)
    pred = np.clip(tests @ w, 0.0, 1.0)
    return oof, pred, {"members": list(members),
                       "weights": [round(float(x), 4) for x in w],
                       "fitted_all_oof_rmse": round(_rmse(_y, oof), 6),
                       "honest_leave_fold_out_rmse": round(_honest_blend(oofs), 6)}


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 900) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises; a timeout or any
    exception becomes status="failed" so the search loop keeps going.

    score is the RMSE itself (lower-is-better, matching the harness sign convention)."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_s))
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, feats = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, rmse=score)
            return dict(status="evaluated", score=round(score, 6),
                        wall_s=round(time.time() - t0, 1),
                        result={"n_feats": len(feats), "rmse": round(score, 6)}, error=None)
        if kind == "blend":
            result, score = evaluate_blend(config)
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
            signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate(json.loads(sys.argv[1])), indent=2))
