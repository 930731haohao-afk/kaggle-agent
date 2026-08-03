"""tree_search/eval_s3e16_run5b.py -- phase-2 evaluator for playground-series-s3e16.

Extends `eval_s3e16_run5` (which is FROZEN: its scores are already logged as tree nodes)
with the one node kind round 2 showed the phase-1 node space could not express -- an
ordinal member that models P(Age = a | x) with a multiclass head and takes the weighted
median, which is the exact Bayes decision rule for MAE on a discrete target. Everything
else -- folds, metric, cache dir, blend honesty, dispatch contract -- is imported from the
frozen module unchanged, so a phase-1 node config re-evaluated here reproduces its phase-1
score digit for digit.

  solo  -- {kind:"solo", model:"lgbmc", params:{..., "decision":"median"|"mean"},
            features:{"set": ...}}
"""

import os
import signal
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import eval_s3e16_run5 as base  # noqa: E402
import harness_v2 as hv2  # noqa: E402

F = base.F
CACHE_DIR = base.CACHE_DIR
PP = base.PP
NUM_THREADS = base.NUM_THREADS

CLS_LO, CLS_HI = 3, 25
CLASS_VALUES = np.arange(CLS_LO, CLS_HI + 1, dtype=np.float64)


def weighted_median(proba):
    cum = np.cumsum(proba, axis=1)
    return CLASS_VALUES[(cum >= 0.5).argmax(axis=1)]


def _run_lgbmc(params, X, y, Xte, folds):
    import lightgbm as lgb
    params = dict(params)
    decision = params.pop("decision", "median")
    cls = np.clip(y.to_numpy(), CLS_LO, CLS_HI) - CLS_LO
    n_cls = len(CLASS_VALUES)
    p = dict(objective="multiclass", num_class=n_cls, metric="multi_logloss",
             learning_rate=0.05, num_leaves=63, min_child_samples=40,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             reg_alpha=0.1, reg_lambda=1.0, verbose=-1, num_threads=NUM_THREADS,
             deterministic=True, force_row_wise=True, seed=F.SEED)
    p.update(params)
    n_rounds = int(p.pop("num_boost_round", 2000))
    esr = int(p.pop("early_stopping_rounds", 60))
    oof_p = np.zeros((len(y), n_cls))
    te_p = np.zeros((len(Xte), n_cls))
    for tr, va in folds:
        dtr = lgb.Dataset(X.iloc[tr], label=cls[tr])
        dva = lgb.Dataset(X.iloc[va], label=cls[va], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(esr, verbose=False),
                                 lgb.log_evaluation(0)])
        oof_p[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        te_p += m.predict(Xte, num_iteration=m.best_iteration) / len(folds)
    if decision == "mean":
        return oof_p @ CLASS_VALUES, te_p @ CLASS_VALUES
    return weighted_median(oof_p), weighted_median(te_p)


RUNNERS = dict(base.RUNNERS)
RUNNERS["lgbmc"] = _run_lgbmc


def evaluate_solo(config):
    X, y, Xte, feats = base.get_feature_frame(config.get("features"))
    runner = RUNNERS.get(config["model"])
    if runner is None:
        raise ValueError(f"unknown model type {config['model']!r}")
    folds = F.make_folds(y)
    oof, pred = runner(dict(config.get("params") or {}), X, y, Xte, folds)
    ynp = y.to_numpy(dtype=np.float64)
    scores = {m: F.mae_pp(oof, ynp, m) for m in ("none", "clip", "round", "snap")}
    per_fold = [float(np.abs(F.postprocess(oof[va], PP) - ynp[va]).mean())
                for _, va in folds]
    return oof, pred, scores, per_fold, feats


def evaluate(config: dict, node_id: int = None, timeout_s: int = 900) -> dict:
    """Same contract as eval_s3e16_run5.evaluate; blends are delegated to it verbatim."""
    if config.get("kind", "solo") != "solo":
        return base.evaluate(config, node_id=node_id, timeout_s=timeout_s)
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old = None
    if have_alarm:
        old = signal.signal(signal.SIGALRM, base._timeout_handler)
        signal.alarm(int(timeout_s))
    try:
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
    except base.EvalTimeout:
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
