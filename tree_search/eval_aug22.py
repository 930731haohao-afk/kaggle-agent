"""tree_search/eval_aug22.py -- per-competition evaluator for
tabular-playground-series-aug-2022 (product failure, binary, ROC-AUC, MAXIMIZE).

Metric INCLUDES post-processing per 07_tree_search.md section 2: the decision score
is pooled AUC after per-group rank normalization (validated +<score> in the prior
run; all sweep configs agreed in direction). Cached OOF/test preds are RAW
probabilities; rank-pp is applied inside the metric so blends mix in probability
space first, then get rank-normalized (mirrored exactly at submission time).

Node kinds:
  solo -- {kind:"solo", model:"lr"|"lgb", params:{...}, features:{cols:[...]}}
          cols is the explicit, sorted feature list (hashable, dedup-meaningful).
  blend -- handled by the driver via harness_v3.eval_blend_with_cost_guard.

Folds: leave-one-group-out on product_code (5 folds A-E) -- identical to
scripts/baseline.py, so cached-OOF pool reuse is digit-exact.
SIGN CONVENTION: harness score = -AUC_pp (lower better); result carries both.
Threads capped at 10; LGBM deterministic + force_row_wise.
"""
import os
import signal
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_COMP = os.path.join(_REPO, "competitions", "tabular-playground-series-aug-2022")
CACHE_DIR = os.path.join(_HERE, "cache_aug22")
os.makedirs(CACHE_DIR, exist_ok=True)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_COMP, "scripts"))
import harness_v2 as hv2  # noqa: E402
from prep import build_features  # noqa: E402

THREADS = 10
SEED = 42

_tr, _te, _y, _groups, _test_ids = build_features()
_codes = np.unique(_groups)


def rank_pp(preds: np.ndarray, grp: np.ndarray = None) -> np.ndarray:
    grp = _groups if grp is None else grp
    out = np.zeros(len(preds))
    for g in np.unique(grp):
        m = grp == g
        out[m] = rankdata(preds[m]) / (m.sum() + 1)
    return out


def auc_pp(oof: np.ndarray) -> float:
    """Decision metric: pooled AUC after per-group rank normalization."""
    return roc_auc_score(_y, rank_pp(oof))


def auc_raw(oof: np.ndarray) -> float:
    return roc_auc_score(_y, oof)


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# solo runners
# ---------------------------------------------------------------------------
def _logo(model_factory, cols):
    oof = np.zeros(len(_tr))
    pred = np.zeros(len(_te))
    for code in _codes:
        va = _groups == code
        m = model_factory()
        m.fit(_tr.loc[~va, cols], _y[~va])
        oof[va] = m.predict_proba(_tr.loc[va, cols])[:, 1]
        pred += m.predict_proba(_te[cols])[:, 1] / len(_codes)
    return oof, pred


def _run_lr(params, cols):
    p = dict(C=0.01, max_iter=2000, random_state=SEED)
    p.update(params)
    return _logo(lambda: LogisticRegression(**p), cols)


LGB_BASE = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
    min_child_samples=80, reg_alpha=1.0, reg_lambda=2.0,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    random_state=SEED, deterministic=True, force_row_wise=True,
    num_threads=THREADS, verbosity=-1)


def _run_lgb(params, cols):
    import lightgbm as lgb
    p = dict(LGB_BASE)
    p.update(params)
    return _logo(lambda: lgb.LGBMClassifier(**p), cols)


def evaluate_solo(config):
    model = config["model"]
    params = dict(config.get("params") or {})
    cols = list((config.get("features") or {}).get("cols") or [])
    missing = [c for c in cols if c not in _tr.columns]
    if missing:
        raise ValueError(f"unknown feature cols {missing}")
    if model == "lr":
        oof, pred = _run_lr(params, cols)
    elif model == "lgb":
        oof, pred = _run_lgb(params, cols)
    else:
        raise ValueError(f"unknown model {model!r}")
    return oof, pred, auc_pp(oof)


# ---------------------------------------------------------------------------
# dispatch (evaluate contract for eval_solo_subprocess)
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 600) -> dict:
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, auc=score)
            return dict(status="evaluated", score=round(-score, 6),
                        wall_s=round(time.time() - t0, 1),
                        result={"auc_pp": round(score, 6),
                                "auc_raw": round(auc_raw(oof), 6)}, error=None)
        raise ValueError(f"eval_aug22 handles solo only in-subprocess, got {kind!r}")
    except EvalTimeout:
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
