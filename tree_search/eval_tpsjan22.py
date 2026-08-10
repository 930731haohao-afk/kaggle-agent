"""tree_search/eval_tpsjan22.py — evaluator for tabular-playground-series-jan-2022
(Kaggle merchandise sales, SMAPE, minimize).

Node kinds (v2/v3 schema convention, cf. eval_s3e19.py):

  solo  — {kind:"solo", model:"ridge"|"lgb", params:{...}}
      ridge params are STRUCTURAL knobs consumed by scripts/common.py's
      build_design/fit_predict_ridge (alpha, fourier_k, fourier_product[_k],
      fourier_country[_k], dow_product, country_dummies, year_c, year_c2,
      holidays, hol_lo, hol_hi, per_product, gdp_offset, recent_weight).
      lgb params: num_leaves, lr, n_estimators, min_child_samples, seed —
      target is log(num_sold/gdp_pc) per the experience-library GDP recipe.

  Validation: expanding-year folds (train<=2016 -> val 2017, train<=2017 -> val
  2018). "OOF" = concatenated val-year predictions, cached FULL-length
  (len(train)=26298) with zeros outside IDX (year in {2017, 2018}) so
  harness_v2.eval_blend's `oofs @ w` works unmodified. Every score is pooled
  SMAPE over IDX rows with round-to-int applied INSIDE the metric (integer
  target; experience.md: round gives a small stable gain).

  Test predictions (2019, fit on full 2015-2018 train) are cached alongside the
  OOF via cache_oof(pred=...) so the final champion blend is a weighted sum of
  cached preds — no refit at submission time.

  Blend nodes are evaluated in the DRIVER via hv3.eval_blend_with_cost_guard
  (k=800 + coordinate ascent, feature 5/6); this module only handles solos.

SIGN CONVENTION: SMAPE is lower-is-better already — score handed to the
harness is the SMAPE itself, no sign flip (same as s3e19).
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "tabular-playground-series-jan-2022")
sys.path.insert(0, os.path.join(_COMP_DIR, "scripts"))
sys.path.insert(0, _HERE)

import stage2_inputs  # noqa: E402
stage2_inputs.require_module(os.path.join(_COMP_DIR, "scripts"), "common", comp="tabular-playground-series-jan-2022",
                             exposes=['FOLDS '])
import common  # noqa: E402
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402


CACHE_DIR = os.path.join(_HERE, "cache_tpsjan22")

_train, _test = common.load_data()
_train_year = _train["date"].dt.year.values
IDX = np.isin(_train_year, [2017, 2018])
Y_TRUE = _train["num_sold"].values.astype(float)


def metric(oof_full: np.ndarray) -> float:
    """Pooled SMAPE over the 2017+2018 validation rows, round-to-int inside."""
    return common.smape(Y_TRUE[IDX], np.round(oof_full[IDX]))


def evaluate_solo(config: dict):
    params = dict(config.get("params", {}))
    model = config.get("model", "ridge")
    cfg = dict(params)
    cfg["model"] = model
    cfg["round_int"] = False  # rounding lives in metric(); cache raw preds for blending
    fit = common.fit_predict_lgb if model == "lgb" else common.fit_predict_ridge
    oof = np.zeros(len(_train))
    for last_tr, val_y in common.FOLDS:
        trm = _train_year <= last_tr
        vam = _train_year == val_y
        oof[vam] = fit(_train[trm], _train[vam], cfg)
    pred = fit(_train, _test, cfg)
    score = metric(oof)
    return oof, pred, score


def evaluate_blend(config):
    """Weighted sum of cached member OOFs -- no refit, so this runs in-process.

    Uses this module's own `metric` (pooled SMAPE with the round-to-int inside it), which is
    lower-is-better already, so the node score needs no sign flip and matches evaluate_solo's
    directly. Owning the metric here rather than in the driver is round 11's fix: this module
    happened to be the ONE of 20 that defined `metric` at module level, which is why the
    template's `ev.metric` looked correct and silently failed everywhere else.
    """
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    best_w, best_s, _oofs, warning = hv3.eval_blend_with_cost_guard(
        CACHE_DIR, members, metric, weight_search=method)
    result = dict(members=members, weights=[round(float(w), 4) for w in best_w],
                  method=method, smape=round(best_s, 6))
    if warning:  # the guard may never coarsen silently (harness_v3 feature 6)
        result["cost_guard_warning"] = warning
    return result, best_s


def evaluate(config: dict, node_id: int = None, timeout_s: int = 300) -> dict:
    t0 = time.time()
    try:
        kind = config.get("kind", "solo")
        if kind == "blend":
            result, score = evaluate_blend(config)
            return dict(status="evaluated", score=round(score, 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
        if kind != "solo":
            raise ValueError(f"unknown node kind {kind!r}")
        oof, pred, score = evaluate_solo(config)
        if node_id is not None:
            hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, smape=score)
        return dict(status="evaluated", score=round(score, 6),
                    wall_s=round(time.time() - t0, 1),
                    result={"smape": round(score, 6)}, error=None)
    except Exception as e:  # noqa: BLE001 — evaluator must never crash the search loop
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
