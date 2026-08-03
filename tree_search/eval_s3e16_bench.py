"""Tree-search node evaluator for playground-series-s3e16 (crab Age, MAE).

Contract: `evaluate(config, node_id=None, timeout_s=None) -> {"score","status","wall_s",
"result","error"}` -- the standard eval_*.py contract every run_*.py driver assumes,
called through `harness_v3.eval_solo_subprocess` for solo nodes.

Node kinds
  kind="solo"   {"model": lgb|xgb|cat|lgbmc, "params": {...}, "features": {"groups":[...],
                 "drop":[...]}, "seed": int}
  kind="blend"  {"members": [node_id, ...], "weight_search": "dirichlet"|"grid_simplex"}

SCORE = MAE AFTER post-processing (clip to the observed Age range 1..29, then round to
the integer grid).  This is deliberate and load-bearing: round 1 of the linear iteration
measured best-raw-MAE = LGB-rawonly 1.356232 while best-rounded-MAE = CatBoost 1.341035,
i.e. the raw and post-processed orderings DISAGREE.  07_tree_search.md §2 row 3 requires
the metric_fn of a discretized decision metric to include the post-processing so every
candidate (including every blend weight vector) is judged on the number that actually
gets submitted.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "competitions", "playground-series-s3e16", "scripts"))

import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402
import model_lib as ML  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_s3e16_bench")
os.makedirs(CACHE_DIR, exist_ok=True)

_Y = ML.get_data(None)[2]
_LO, _HI = float(_Y.min()), float(_Y.max())


def score_pp(vec) -> float:
    """The submitted-form metric: clip to [1, 29] then round to the integer grid."""
    return float(np.mean(np.abs(_Y - np.clip(np.round(np.asarray(vec)), _LO, _HI))))


def core(cfg: dict) -> dict:
    """Canonical hashable stored form of a config (drop output-only fields, sort a
    blend's members) -- so harness dedup hashing is meaningful."""
    out = {k: v for k, v in cfg.items() if k not in ("result", "want_importance", "note")}
    if out.get("kind") == "blend" or "members" in out:
        out["members"] = sorted(out["members"])
    return out


def evaluate_solo(config: dict, node_id):
    r = ML.run_solo({"model": config["model"],
                     "params": config.get("params") or {},
                     "features": config.get("features") or {},
                     "seed": int(config.get("seed", ML.SEED)),
                     "postprocess": "clip_round"})
    if node_id is not None:
        hv2.cache_oof(CACHE_DIR, node_id, r["oof"], pred=r["pred"])
    return score_pp(r["oof"]), {"score_raw": round(r["score_raw"], 6),
                                "pp_scores": {k: round(v, 6) for k, v in r["pp_scores"].items()},
                                "n_features": len(r["feats"]),
                                "best_iters": r["info"].get("best_iters")}


def evaluate_blend(config: dict, node_id):
    members = list(config["members"])
    if len(members) < 2:
        raise ValueError(f"blend needs >=2 members, got {members!r}")
    w, s, oofs = hv3.eval_blend(CACHE_DIR, members, score_pp,
                                weight_search=config.get("weight_search", "dirichlet"))
    if node_id is not None:
        preds = np.stack([np.load(os.path.join(CACHE_DIR, f"solo_{m}.npz"))["pred"]
                          for m in members], axis=1)
        hv2.cache_oof(CACHE_DIR, node_id, oofs @ np.asarray(w), pred=preds @ np.asarray(w))
    return float(s), {"weights": [round(float(x), 6) for x in w],
                      "members": members,
                      "score_raw": round(float(np.mean(np.abs(_Y - oofs @ np.asarray(w)))), 6)}


def evaluate(config: dict, node_id=None, timeout_s=None) -> dict:
    t0 = time.time()
    try:
        kind = config.get("kind", "solo")
        if kind == "blend":
            score, extra = evaluate_blend(config, node_id)
        else:
            score, extra = evaluate_solo(config, node_id)
        return dict(score=round(float(score), 6), status="evaluated",
                    wall_s=round(time.time() - t0, 2), result=extra, error=None)
    except Exception as exc:  # noqa: BLE001 - the harness contract wants a failed dict
        return dict(score=None, status="failed", wall_s=round(time.time() - t0, 2),
                    result=None, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate({"kind": "solo", "model": "cat",
                               "params": {"depth": 6, "learning_rate": 0.06},
                               "features": {"groups": ["cnt"]}}), indent=2))
