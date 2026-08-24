"""V4: convex blend over real OOFs + submission.

ROC-AUC is a pure ranking metric, so members are blended in RANK space (scale-free, immune to
each member's calibration) and weights are optimised directly on OOF AUC -- no differentiable
surrogate needed, because AUC on 29k rows costs ~5 ms.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import common as C


def to_rank(p: np.ndarray) -> np.ndarray:
    return rankdata(p) / len(p)


def opt_weights(R: np.ndarray, y: np.ndarray, seed: int = C.SEED,
                n_dirichlet: int = 4000, rounds: int = 40) -> tuple[np.ndarray, float]:
    """Dirichlet random search + coordinate ascent on the simplex, objective = OOF AUC."""
    m = R.shape[1]
    rng = np.random.default_rng(seed)
    best_w = np.ones(m) / m
    best = C.auc(y, R @ best_w)
    for w in rng.dirichlet(np.ones(m), size=n_dirichlet):
        s = C.auc(y, R @ w)
        if s > best:
            best, best_w = s, w
    step = 0.08
    for _ in range(rounds):
        improved = False
        for i in range(m):
            for d in (step, -step):
                w = best_w.copy()
                w[i] = max(0.0, w[i] + d)
                if w.sum() <= 0:
                    continue
                w /= w.sum()
                s = C.auc(y, R @ w)
                if s > best + 1e-9:
                    best, best_w, improved = s, w, True
        if not improved:
            step /= 2
            if step < 1e-3:
                break
    return best_w, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma-separated artifact tags")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--tag", default="blend")
    a = ap.parse_args()

    tr, te = C.load_meta()
    y = tr.target.to_numpy()
    folds = tr.fold.to_numpy()
    names = [n.strip() for n in a.members.split(",")]

    oofs, preds, solo = [], [], {}
    for n in names:
        o = np.load(C.ART / f"oof_{n}.npy")
        p = np.load(C.ART / f"pred_{n}.npy")
        assert np.isfinite(o).all(), f"{n}: OOF has NaN (incomplete folds)"
        solo[n] = C.auc(y, o)
        oofs.append(o)
        preds.append(p)
    R = np.column_stack([to_rank(o) for o in oofs])
    P = np.column_stack([to_rank(p) for p in preds])

    print("--- solo OOF AUC ---")
    for n in names:
        print(f"  {n:34s} {solo[n]:.5f}")
    print("\n--- OOF rank correlation ---")
    print(pd.DataFrame(np.corrcoef(R.T), index=names, columns=names).round(3).to_string())

    eq = C.auc(y, R.mean(1))
    w, best = opt_weights(R, y)
    print(f"\nequal-weight rank blend : {eq:.5f}")
    print(f"optimised rank blend    : {best:.5f}")
    print("weights: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, w) if v > 1e-4))

    # honesty check: leave-one-fold-out weight fitting (weights are fitted on the full OOF above)
    loo = np.zeros(len(y))
    for k in range(C.N_FOLDS):
        m = folds != k
        wk, _ = opt_weights(R[m], y[m], n_dirichlet=1500, rounds=25)
        loo[~m] = R[~m] @ wk
    loo_auc = C.auc(y, loo)
    print(f"leave-fold-out weight fit: {loo_auc:.5f}  (honest; gap {best - loo_auc:+.5f})")

    blend_pred = P @ w
    per_fold = [C.auc(y[folds == k], (R @ w)[folds == k]) for k in range(C.N_FOLDS)]
    np.save(C.ART / f"oof_{a.tag}.npy", R @ w)
    np.save(C.ART / f"pred_{a.tag}.npy", blend_pred)

    out = {"members": names, "solo": solo, "weights": dict(zip(names, w.round(4).tolist())),
           "equal_weight": eq, "optimised": best, "leave_fold_out": loo_auc, "per_fold": per_fold}
    (C.COMP / f"{a.tag}_results.json").write_text(json.dumps(out, indent=2))

    C.log_experiment(
        model="convex rank blend (" + " + ".join(names) + ")", score=best, fold_scores=per_fold,
        features=[f"member OOF ranks: {n}" for n in names],
        ensemble={"space": "rank", "weights": dict(zip(names, w.round(4).tolist())),
                  "search": "4000-sample Dirichlet + coordinate ascent on OOF AUC",
                  "equal_weight_auc": round(eq, 6),
                  "leave_fold_out_auc": round(loo_auc, 6)},
        submission="submission.csv" if a.submit else None,
        notes=(f"solo: " + "; ".join(f"{n}={solo[n]:.5f}" for n in names)
               + f". Honest leave-fold-out weight fit {loo_auc:.5f} (gap {best - loo_auc:+.5f})."))

    if a.submit:
        ss = pd.read_csv(C.DATA / "sample_submission.csv")
        sub = pd.DataFrame({"image_name": te.image_name.to_numpy(), "target": blend_pred})
        sub = ss[["image_name"]].merge(sub, on="image_name", how="left", validate="1:1")
        assert len(sub) == len(ss) and sub.target.notna().all()
        assert (sub.image_name.values == ss.image_name.values).all()
        sub.to_csv(C.COMP / "submission.csv", index=False)
        print(f"\nwrote submission.csv  rows={len(sub)}  "
              f"range [{sub.target.min():.5f}, {sub.target.max():.5f}]")


if __name__ == "__main__":
    main()
