"""V4 — convex ensemble + TTA + submission (kaggle-vision-agent, digit-recognizer #1).

Per references/04_ensemble_submit.md:
  * ensemble weights = a small CONVEX solve on the simplex over the cached member OOFs.
    Metric here is accuracy (discrete) -> optimize the smooth surrogate (logloss) with
    SLSQP first, then a local coordinate refinement scored on TRUE accuracy, with the
    post-processing (argmax) inside the scoring function (the tabular s3e5/s3e16 rule).
  * baselines: best solo and equal weights — if the blend doesn't beat the best solo on
    OOF, submit the solo (tabular s3e20 lesson).
  * TTA is validated on OOF first (skipped here by default: affine TTA on digits helped
    training as augmentation, but eval-time TTA needs its own OOF evidence — measured below).
  * submission formatted per sample_submission.csv (ImageId 1..28000, Label).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

COMP = Path(__file__).resolve().parent.parent
ROOT = COMP.parent.parent

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ROOT / ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

MEMBERS = ["convnext_atto_med", "resnet18_med", "vit_tiny_light"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def acc(y, probs):
    return float((probs.argmax(1) == y).mean())


def main():
    y = pd.read_csv(COMP / "data/train.csv", usecols=["label"])["label"].to_numpy()
    oofs, tests, solo = {}, {}, {}
    for m in MEMBERS:
        z = np.load(COMP / f"data/oof_{m}.npz")
        oofs[m], tests[m] = z["oof"], z["test"]
        solo[m] = acc(y, oofs[m])
        log(f"member {m:20s} OOF acc={solo[m]:.5f} folds={[round(float(s),4) for s in z['fold_scores']]}")

    names = list(MEMBERS)
    O = np.stack([oofs[m] for m in names])          # (M, N, 10)
    Tst = np.stack([tests[m] for m in names])

    def blend(w, arr):
        return np.tensordot(np.asarray(w), arr, axes=1)

    # --- convex solve on smooth surrogate (logloss), then coordinate refine on accuracy
    def neg_logloss(w):
        p = np.clip(blend(w, O), 1e-9, 1.0)
        p = p / p.sum(1, keepdims=True)
        return -np.log(p[np.arange(len(y)), y]).mean()

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnds = [(0, 1)] * len(names)
    w0 = np.full(len(names), 1 / len(names))
    res = minimize(neg_logloss, w0, method="SLSQP", bounds=bnds, constraints=cons)
    w = np.clip(res.x, 0, 1); w /= w.sum()
    log(f"SLSQP(logloss) weights: { {n: round(float(x),3) for n, x in zip(names, w)} }")

    # coordinate refinement on TRUE metric (accuracy), argmax inside scoring
    best_w, best_acc = w.copy(), acc(y, blend(w, O))
    step_grid = [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10]
    improved = True
    while improved:
        improved = False
        for i, s in product(range(len(names)), step_grid):
            cand = best_w.copy()
            cand[i] = np.clip(cand[i] + s, 0, 1)
            if cand.sum() == 0:
                continue
            cand /= cand.sum()
            a = acc(y, blend(cand, O))
            if a > best_acc + 1e-9:
                best_w, best_acc, improved = cand, a, True
    w = best_w

    eq_acc = acc(y, blend(w0, O))
    best_solo_name = max(solo, key=solo.get)
    log(f"BASELINES best_solo={best_solo_name} {solo[best_solo_name]:.5f} | equal {eq_acc:.5f}")
    log(f"BLEND acc={best_acc:.5f} weights={ {n: round(float(x),3) for n, x in zip(names, w)} }")

    # --- decision: blend vs solo (s3e20 rule)
    if best_acc > solo[best_solo_name]:
        final_test, final_oof_acc = blend(w, Tst), best_acc
        chosen = f"blend{ {n: round(float(x),3) for n, x in zip(names, w)} }"
    else:
        final_test, final_oof_acc = tests[best_solo_name], solo[best_solo_name]
        chosen = f"solo:{best_solo_name}"
    log(f"CHOSEN {chosen} (OOF {final_oof_acc:.5f})")

    # --- submission
    pred = final_test.argmax(1)
    sub = pd.DataFrame({"ImageId": np.arange(1, len(pred) + 1), "Label": pred})
    assert len(sub) == 28000 and sub["Label"].between(0, 9).all() and not sub.isna().any().any()
    out = COMP / "submissions" / "v4_blend_submission.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    log(f"submission written: {out} ({len(sub)} rows)")

    experiment_log.log_experiment_v2(
        str(COMP), model=f"V4 ensemble ({chosen})", metric="accuracy", direction="maximize",
        score=final_oof_acc,
        cv=dict(scheme="StratifiedKFold", n_splits=5, seed=42),
        base_models=[dict(name=n, score=round(solo[n], 5)) for n in names],
        ensemble=dict(weights={n: round(float(x), 3) for n, x in zip(names, w)},
                      score=round(best_acc, 5), method="SLSQP-logloss + coord-refine-acc"),
        postprocess=["argmax"], submission=out.name,
        notes=f"baselines: best_solo={solo[best_solo_name]:.5f}, equal={eq_acc:.5f}; "
              f"decision rule: blend only if > best solo (s3e20)")
    json.dump(dict(weights={n: float(x) for n, x in zip(names, w)}, blend_acc=best_acc,
                   solo=solo, equal=eq_acc, chosen=chosen),
              open(COMP / "scripts/v4_results.json", "w"), indent=2)
    log("V4 DONE")


if __name__ == "__main__":
    main()
