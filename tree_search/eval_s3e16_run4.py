"""tree_search/eval_s3e16_run4.py — per-competition evaluator for playground-series-s3e16
(Crab Age, regression, MAE on an INTEGER target, minimize-better).

Fresh evaluator for this run. The pre-existing tree_search/eval_s3e16_v2.py,
tree_search/run_s3e16_v3.py and tree_search/cache_s3e16/ are artifacts of EARLIER runs of
this same competition and were deliberately not opened (lane isolation) — hence the
distinct module name and the distinct CACHE_DIR, so neither run can read or clobber the
other's cache.

Node kinds (harness_v2/v3 two-kind schema):

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...}, blocks:[...],
              seeds:[int,...]}
     Trains with competitions/playground-series-s3e16/scripts/pool_lib.py's fold split
     (KFold(5, shuffle, seed=42)) — bit-identical to the Stage-3 linear scripts, so a
     tree score and a linear score are directly comparable. `blocks` selects feature
     blocks from scripts/features.py. `seeds` longer than 1 = seed-bagging: the same
     config is trained once per seed and the predictions averaged before scoring.
     OOF and test predictions are cached to cache_s3e16_run4/solo_<node_id>.npz so blend
     nodes never retrain.

  2. blend — {kind:"blend", members:[<solo node id>,...], weight_search:"dirichlet"}
     Weight search over cached member OOFs. Handled by the DRIVER (which calls
     harness_v3.eval_blend_with_cost_guard so features 5+6 actually fire); this module
     implements it too so the evaluator is self-contained when called directly.

SIGN CONVENTION: MAE is minimize-better, matching harness.py's convention, so scores are
passed through unnegated.

METRIC: `mae_pp` — the POST-PROCESSED MAE. The target is an integer in [1,29] and the
scorer is MAE, so the rounding decision is part of the decision surface, not a cosmetic
final step. Per references/07_tree_search.md §2 ("discretized / post-rounding decision
metrics"), the postprocessing lives INSIDE the metric function, so every candidate blend
weight vector is scored after rounding rather than the winner being rounded retroactively.
This is not theoretical here: in Stage 3 the equal-weight blend had a WORSE raw MAE than
the best solo (1.36344 vs 1.35484) and a BETTER rounded MAE (1.33238 vs 1.33864) — picking
on raw would have selected the wrong model. Known cost, stated once: the offset grid is
chosen on the same OOF used to score, so `mae_pp` carries a small selection optimism
(~1e-3 scale); it is applied identically to every node, so between-node comparisons stay
fair, but the absolute number is optimistic.
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
import features as F   # noqa: E402
import pool_lib as P   # noqa: E402
import harness_v2 as hv2  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_s3e16_run4")
_y = None
_CACHE_X = {}


def y():
    global _y
    if _y is None:
        _, _, _y_local, _ = F.build(("base",))
        _y = _y_local
    return _y


def xy(blocks):
    """Feature frames for a block tuple, memoised per process."""
    key = tuple(blocks)
    if key not in _CACHE_X:
        X, Xte, yv, names = F.build(key)
        _CACHE_X[key] = (X, Xte, yv, names)
    return _CACHE_X[key]


def mae_pp(pred) -> float:
    """The metric: MAE after the best postprocessing variant (see module docstring)."""
    return P.postprocess_race(y(), pred)["best_score"]


def best_pp_name(pred) -> str:
    return P.postprocess_race(y(), pred)["best"]


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def _fit_lgb(params, X, Xte, yv, seed):
    import lightgbm as lgb
    p = dict(params)
    p.update(num_threads=P.N_THREADS, deterministic=True, force_row_wise=True,
             verbosity=-1, seed=int(seed), max_bin=p.get("max_bin", 255))
    esr = int(p.pop("early_stopping_rounds", 200))
    oof = np.zeros(len(yv))
    tp = np.zeros(len(Xte))
    for tr, va in P.folds(len(yv)):
        m = lgb.LGBMRegressor(**p)
        m.fit(X.iloc[tr], yv[tr], eval_set=[(X.iloc[va], yv[va])], eval_metric="l1",
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        oof[va] = m.predict(X.iloc[va])
        tp += m.predict(Xte) / P.N_FOLDS
    return oof, tp


def _fit_xgb(params, X, Xte, yv, seed):
    import xgboost as xgb
    p = dict(params)
    p.update(n_jobs=P.N_THREADS, random_state=int(seed), tree_method=p.get("tree_method", "hist"))
    p.setdefault("objective", "reg:absoluteerror")
    p.setdefault("eval_metric", "mae")
    p.setdefault("early_stopping_rounds", 200)
    oof = np.zeros(len(yv))
    tp = np.zeros(len(Xte))
    for tr, va in P.folds(len(yv)):
        m = xgb.XGBRegressor(**p)
        m.fit(X.iloc[tr], yv[tr], eval_set=[(X.iloc[va], yv[va])], verbose=False)
        oof[va] = m.predict(X.iloc[va])
        tp += m.predict(Xte) / P.N_FOLDS
    return oof, tp


def _fit_cat(params, X, Xte, yv, seed):
    from catboost import CatBoostRegressor
    p = dict(params)
    # allow_writing_files=False: CatBoost's default catboost_info logging has stalled
    # long sandboxed runs before; thread_count explicit for the shared-machine cap.
    p.update(random_seed=int(seed), thread_count=P.N_THREADS, allow_writing_files=False)
    p.setdefault("loss_function", "MAE")
    p.setdefault("eval_metric", "MAE")
    p.setdefault("od_type", "Iter")
    p.setdefault("od_wait", 200)
    Xf = X.fillna(-999.0)
    Xtef = Xte.fillna(-999.0)   # CatBoost rejects NaN in float features by default
    oof = np.zeros(len(yv))
    tp = np.zeros(len(Xtef))
    for tr, va in P.folds(len(yv)):
        m = CatBoostRegressor(**p)
        m.fit(Xf.iloc[tr], yv[tr], eval_set=(Xf.iloc[va], yv[va]), verbose=False)
        oof[va] = m.predict(Xf.iloc[va])
        tp += m.predict(Xtef) / P.N_FOLDS
    return oof, tp


_FITTERS = {"lgb": _fit_lgb, "xgb": _fit_xgb, "cat": _fit_cat}


def evaluate_solo(config):
    blocks = tuple(config.get("blocks") or ("base",))
    X, Xte, yv, names = xy(blocks)
    model = config["model"]
    seeds = list(config.get("seeds") or [P.SEED])
    fit = _FITTERS[model]
    oofs, tps = [], []
    for s in seeds:
        o, t = fit(config["params"], X, Xte, yv, s)
        oofs.append(o)
        tps.append(t)
    oof = np.mean(oofs, axis=0)
    tp = np.mean(tps, axis=0)
    return oof, tp, mae_pp(oof), names


def evaluate_blend(config):
    members = list(config["members"])
    best_w, best_s, oofs = hv2.eval_blend(CACHE_DIR, members, mae_pp,
                                          weight_search=config.get("weight_search", "dirichlet"))
    blended = oofs @ best_w
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                mae=round(float(best_s), 6), pp=best_pp_name(blended)), float(best_s)


def evaluate(config, node_id=None, timeout_s=900):
    """Contract shared by every eval_*.py in this repo:
    returns {"score","status","wall_s","result","error"}; NEVER raises."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_s))
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, names = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, mae=score)
            return dict(status="evaluated", score=round(float(score), 6),
                        wall_s=round(time.time() - t0, 1),
                        result={"mae": round(float(score), 6),
                                "raw_mae": round(P.mae(y(), oof), 6),
                                "pp": best_pp_name(oof),
                                "n_feats": len(names),
                                "n_seeds": len(config.get("seeds") or [P.SEED])},
                        error=None)
        if kind == "blend":
            result, score = evaluate_blend(config)
            return dict(status="evaluated", score=round(float(score), 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
        raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 — the evaluator must never crash the search loop
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
