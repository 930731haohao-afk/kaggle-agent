"""Stage 4 — convex rank blend over real OOFs, with an honest leave-fold-out estimate.

Members are rank-transformed before mixing: AUC is rank-invariant, the members live on very
different score scales (a CNN sigmoid vs a GBDT probability), and rank space makes the convex
weights comparable.

Two scores are reported for every blend:
  * `optimised`     — weights fitted on the full OOF, scored on the full OOF (optimistic:
                      the weights have seen every row they are scored on)
  * `leave_fold_out`— weights refitted on 4 folds, scored on the held-out fold, averaged.
                      This is the number to believe and the one quoted as the run's CV.
"""
from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def rk(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)


def hill_climb(P: np.ndarray, y: np.ndarray, rounds: int = 60) -> np.ndarray:
    """Caruana-style greedy ensemble selection with replacement.

    Robust for a non-differentiable objective like AUC, and it naturally yields sparse
    weights (a member that adds nothing is never picked). The full `rounds` steps are always
    taken — stopping at the first non-improving step would pin the weights to a 1/k grid far
    coarser than the members deserve — and the best-scoring prefix is returned.
    """
    n, m = P.shape
    acc = np.zeros(n)
    counts = np.zeros(m)
    best_score, best_counts = -1.0, None
    for it in range(rounds):
        scores = [C.oof_auc(y, (acc * it + P[:, j]) / (it + 1)) for j in range(m)]
        j = int(np.argmax(scores))
        acc = (acc * it + P[:, j]) / (it + 1)
        counts[j] += 1
        if scores[j] > best_score:
            best_score, best_counts = scores[j], counts.copy()
    return best_counts / best_counts.sum()


def main() -> None:
    tags = sys.argv[1:] if len(sys.argv) > 1 else None
    f = pd.read_csv(C.CACHE / "r2_folds.csv")
    y, folds, gfolds = f.target.values, f.fold.values, f.gfold.values

    if tags is None:
        tags = sorted(p.stem[4:] for p in C.ART.glob("oof_*.npy")
                      if not p.stem.startswith("oof_blend"))
    oofs, preds, solo = {}, {}, {}
    for t in tags:
        o, p = C.load_pred(t)
        oofs[t], preds[t] = rk(o), rk(p)
        solo[t] = C.oof_auc(y, o)
    tags = sorted(tags, key=lambda t: -solo[t])
    log.info("solo AUCs:\n%s", "\n".join(f"  {t:<16} {solo[t]:.5f}" for t in tags))

    P = np.column_stack([oofs[t] for t in tags])
    T = np.column_stack([preds[t] for t in tags])

    equal = C.oof_auc(y, P.mean(1))
    w = hill_climb(P, y)
    opt = C.oof_auc(y, P @ w)

    # honest: weights never see the fold they are scored on
    lfo = []
    for k in range(C.N_FOLDS):
        tr = folds != k
        wk = hill_climb(P[tr], y[tr])
        lfo.append(C.oof_auc(y[~tr], P[~tr] @ wk))
    lfo_mean = float(np.mean(lfo))

    blend_oof, blend_pred = P @ w, T @ w
    gauc = float(np.mean([C.oof_auc(y[gfolds == k], blend_oof[gfolds == k])
                          for k in range(C.N_FOLDS)]))

    log.info("equal-weight      %.5f", equal)
    log.info("optimised (full)  %.5f", opt)
    log.info("leave-fold-out    %.5f  per-fold %s", lfo_mean, [round(a, 5) for a in lfo])
    log.info("patient-grouped   %.5f", gauc)
    log.info("weights: %s", {t: round(float(wi), 4) for t, wi in zip(tags, w) if wi > 0})

    C.save_pred("blend", blend_oof, blend_pred)
    res = dict(members=tags, solo=solo, weights={t: float(wi) for t, wi in zip(tags, w)},
               equal_weight=equal, optimised=opt, leave_fold_out=lfo_mean,
               leave_fold_out_per_fold=lfo, patient_grouped=gauc)
    C.write_json("blend_results.json", res)
    C.log_exp(model="convex rank blend (" + " + ".join(t for t, wi in zip(tags, w) if wi > 0) + ")",
              metric="roc_auc", direction="maximize", score=lfo_mean,
              base_models=[dict(tag=t, solo_auc=solo[t], weight=float(wi))
                           for t, wi in zip(tags, w)],
              ensemble=dict(method="rank-average, Caruana greedy selection with replacement",
                            optimised_full_oof=opt, equal_weight=equal,
                            leave_fold_out=lfo_mean, patient_grouped=gauc),
              cv=dict(scheme="StratifiedKFold(5) by image", per_fold=lfo),
              notes="score reported = honest leave-fold-out (weights refit on 4 folds, scored "
                    "on the 5th); optimised full-OOF number is the optimistic counterpart")


if __name__ == "__main__":
    main()
