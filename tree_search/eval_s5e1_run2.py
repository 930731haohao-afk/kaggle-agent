"""tree_search/eval_s5e1_run2.py — evaluator for playground-series-s5e1 (run2 benchmark;
daily sticker sales, MAPE, minimize). FRESH cache dir cache_s5e1_run2 — deliberately does
NOT touch cache_s5e1 / cache_ratio_s5e1 / cache_featurejoin_s5e1 (archived arms experiment;
reading them would be self-contamination).

Node kinds (v2/v3 schema convention):
  solo — {kind:"solo", ...model_lib cfg keys...} passed straight to
      competitions/playground-series-s5e1/scripts/model_lib.fit_predict:
      model: "lgb"|"ridge"; target: log|ratio|level_anchor|mape_w; level_rule:
      last|avg2|trend|convNN; ridge design knobs (alpha, doy_k, c104_k, c52_k,
      hol_before, hol_after, hol_per_name, cxp, sxp, cxdow, country_yearc, log_gdp,
      yearc, recency_halflife_years); lgb params under "params".

Validation: expanding-year folds (fit<y, val==y for y in 2014/2015/2016), non-null rows.
"OOF" cached FULL train length (230130) with zeros outside IDX. Score = pooled MAPE x100
with clip>=1 and round-to-int INSIDE the metric (integer target; TASK-MAE-METRIC).
Test preds (fit on all 2010-2016 non-null train) cached via cache_oof(pred=...).
Blend nodes are evaluated in the driver (hv3.eval_blend_with_cost_guard); solos only here.
SIGN: MAPE lower-is-better, no flip.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s5e1")
sys.path.insert(0, os.path.join(_COMP_DIR, "scripts"))
sys.path.insert(0, _HERE)

import harness_v2 as hv2  # noqa: E402
import model_lib as M  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_s5e1_run2")

_df = M.load_df()
_train_mask = (_df.is_train == 1).to_numpy()
_train = _df[_df.is_train == 1].reset_index(drop=True)
N_TRAIN = len(_train)
IDX = ((_train.year.isin(M.VAL_YEARS)) & _train.num_sold.notna()).to_numpy()
Y_TRUE = _train.num_sold.to_numpy(dtype=float)


def metric(oof_full: np.ndarray) -> float:
    """Pooled MAPE x100 over val-year non-null rows; clip>=1 + round inside."""
    return M.mape(Y_TRUE[IDX], oof_full[IDX], clip_min=1.0, round_to_int=True)


def evaluate_solo(config: dict):
    cfg = {k: v for k, v in config.items() if k != "kind"}
    oof = np.zeros(N_TRAIN)
    for _vy, fit, val in M.folds(_df):
        p = M.fit_predict(_df, fit, val, cfg)
        val_pos = np.flatnonzero(val.to_numpy()[_train_mask])
        oof[val_pos] = p
    fit_all = (_df.is_train == 1) & _df.num_sold.notna()
    test_mask = _df.is_train == 0
    pred = M.fit_predict(_df, fit_all, test_mask, cfg)
    return oof, pred, metric(oof)


def evaluate(config: dict, node_id: int = None, timeout_s: int = 300) -> dict:
    t0 = time.time()
    try:
        if config.get("kind", "solo") != "solo":
            raise ValueError("blend nodes are evaluated in the driver, not this module")
        oof, pred, score = evaluate_solo(config)
        if node_id is not None:
            hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, mape=score)
        return dict(status="evaluated", score=round(score, 6),
                    wall_s=round(time.time() - t0, 1),
                    result={"mape": round(score, 6)}, error=None)
    except Exception as e:  # noqa: BLE001 — evaluator must never crash the search loop
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
