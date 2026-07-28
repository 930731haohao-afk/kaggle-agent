"""afsis-soil-properties: Ridge variants on spectral preprocessing + blend.

Design (evidence-backed priors from knowledge/experience.md + memory):
- p>>n (1157 x 3578): Ridge generalizes far better than GBDT/SVR/stacking.
- Preprocessing variants: raw, SG-d1, SG-d2, SNV, SNV+SG-d1 (windows 21-31).
- CV: GroupKFold(5) on spatial-signature site (565 Topsoil/Subsoil pairs share
  a site -> plain KFold leaks location).
- Blend: equal-weight vs per-target simplex weight search on OOF.
Deterministic: SVD ridge, fixed fold assignment (hash of sorted group order).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]
SPATIAL = ["BSAN", "BSAS", "BSAV", "CTI", "ELEV", "EVI", "LSTD", "LSTN",
           "REF1", "REF2", "REF3", "REF7", "RELI", "TMAP", "TMFI"]
ALPHAS = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
N_FOLDS = 5
SEED = 42


def load():
    train = pd.read_csv(f"{COMP}/data/training.csv")
    test = pd.read_csv(f"{COMP}/data/sorted_test.csv")
    spec_cols = [c for c in train.columns if c.startswith("m")]
    co2 = [c for c in spec_cols if 2352.0 <= float(c[1:]) <= 2380.0]
    keep_mask = np.array([c not in set(co2) for c in spec_cols])
    Xs_tr = train[spec_cols].values.astype(np.float64)
    Xs_te = test[spec_cols].values.astype(np.float64)
    return train, test, Xs_tr, Xs_te, keep_mask


def make_variants(Xs_tr, Xs_te, keep_mask):
    """Return dict name -> (train_mat, test_mat). SG filters run on full grid,
    CO2 band dropped afterwards."""
    def snv(X):
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        return (X - mu) / sd

    v = {}
    v["raw"] = (Xs_tr[:, keep_mask], Xs_te[:, keep_mask])
    sg1_tr = savgol_filter(Xs_tr, 25, 2, deriv=1, axis=1)
    sg1_te = savgol_filter(Xs_te, 25, 2, deriv=1, axis=1)
    v["sg1"] = (sg1_tr[:, keep_mask], sg1_te[:, keep_mask])
    sg2_tr = savgol_filter(Xs_tr, 31, 2, deriv=2, axis=1)
    sg2_te = savgol_filter(Xs_te, 31, 2, deriv=2, axis=1)
    v["sg2"] = (sg2_tr[:, keep_mask], sg2_te[:, keep_mask])
    snv_tr, snv_te = snv(Xs_tr), snv(Xs_te)
    v["snv"] = (snv_tr[:, keep_mask], snv_te[:, keep_mask])
    v["snv_sg1"] = (savgol_filter(snv_tr, 25, 2, deriv=1, axis=1)[:, keep_mask],
                    savgol_filter(snv_te, 25, 2, deriv=1, axis=1)[:, keep_mask])
    return v


def group_folds(train):
    """Deterministic GroupKFold(5) on spatial signature."""
    key = train[SPATIAL].round(6).apply(tuple, axis=1)
    groups = key.astype("category").cat.codes.values
    uniq = np.unique(groups)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(uniq)
    fold_of_group = {g: i % N_FOLDS for i, g in enumerate(perm)}
    return np.array([fold_of_group[g] for g in groups]), groups


def svd_ridge_cv(X, Y, folds, alphas=ALPHAS):
    """Per-target alpha selection via OOF RMSE. Returns OOF preds (n x T),
    chosen alphas, per-alpha OOF rmse table."""
    n, T = len(X), Y.shape[1]
    oof = np.zeros((len(alphas), n, T))
    for f in range(N_FOLDS):
        tr, va = folds != f, folds == f
        Xtr, Xva = X[tr], X[va]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd
        ym = Y[tr].mean(0)
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        UtY = U.T @ (Y[tr] - ym)
        XvaV = Xva @ Vt.T
        for ai, a in enumerate(alphas):
            d = s / (s ** 2 + a)
            oof[ai][va] = XvaV @ (d[:, None] * UtY) + ym
    rmse = np.sqrt(((oof - Y[None]) ** 2).mean(axis=1))  # (A, T)
    best_ai = rmse.argmin(axis=0)
    oof_best = np.stack([oof[best_ai[t], :, t] for t in range(T)], axis=1)
    return oof_best, alphas[best_ai], rmse


def svd_ridge_full(X, Y, Xte, alphas_per_target):
    """Fit on all train, predict test with per-target alpha."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xtr = (X - mu) / sd
    Xt = (Xte - mu) / sd
    ym = Y.mean(0)
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY = U.T @ (Y - ym)
    XtV = Xt @ Vt.T
    pred = np.zeros((len(Xte), Y.shape[1]))
    for t, a in enumerate(alphas_per_target):
        d = s / (s ** 2 + a)
        pred[:, t] = XtV @ (d * UtY[:, t]) + ym[t]
    return pred


def mcrmse(Y, P):
    return float(np.sqrt(((P - Y) ** 2).mean(axis=0)).mean())


def main():
    train, test, Xs_tr, Xs_te, keep_mask = load()
    Y = train[TARGETS].values.astype(np.float64)
    folds, groups = group_folds(train)
    print("fold sizes:", np.bincount(folds).tolist())

    variants = make_variants(Xs_tr, Xs_te, keep_mask)
    # depth + spatial as extra block for one variant
    depth_tr = (train["Depth"] == "Topsoil").values.astype(np.float64)[:, None]
    depth_te = (test["Depth"] == "Topsoil").values.astype(np.float64)[:, None]
    spat_tr = train[SPATIAL].values.astype(np.float64)
    spat_te = test[SPATIAL].values.astype(np.float64)
    variants["raw_spatial"] = (
        np.hstack([variants["raw"][0], spat_tr, depth_tr]),
        np.hstack([variants["raw"][1], spat_te, depth_te]))
    variants["sg1_spatial"] = (
        np.hstack([variants["sg1"][0], spat_tr, depth_tr]),
        np.hstack([variants["sg1"][1], spat_te, depth_te]))

    # naive baseline
    base = np.zeros_like(Y)
    for f in range(N_FOLDS):
        base[folds == f] = Y[folds != f].mean(0)
    print(f"\nnaive mean baseline MCRMSE: {mcrmse(Y, base):.5f}")

    results = {}
    for name, (Xtr, Xte) in variants.items():
        oof, alphas_t, _ = svd_ridge_cv(Xtr, Y, folds)
        sc = mcrmse(Y, oof)
        per_t = np.sqrt(((oof - Y) ** 2).mean(axis=0))
        results[name] = {"oof": oof, "alphas": alphas_t, "score": sc,
                         "Xtr": Xtr, "Xte": Xte}
        print(f"{name:12s} MCRMSE {sc:.5f}  per-target " +
              " ".join(f"{t}:{v:.4f}" for t, v in zip(TARGETS, per_t)) +
              "  alphas " + " ".join(f"{a:g}" for a in alphas_t))

    np.savez(f"{COMP}/scripts/oof_cache.npz",
             folds=folds, Y=Y,
             **{f"oof_{k}": v["oof"] for k, v in results.items()},
             **{f"alphas_{k}": v["alphas"] for k, v in results.items()})

    # blends over variant OOFs
    names = list(results.keys())
    O = np.stack([results[k]["oof"] for k in names])  # (M, n, T)
    eq = O.mean(0)
    print(f"\nequal-weight blend ({len(names)}) MCRMSE: {mcrmse(Y, eq):.5f}")

    # per-target greedy forward selection with replacement (deterministic)
    sel_weights = np.zeros((len(names), len(TARGETS)))
    for t in range(len(TARGETS)):
        yt = Y[:, t]
        cur = np.zeros(len(yt))
        picks = []
        for step in range(12):
            best_sc, best_m = None, None
            for m in range(len(names)):
                cand = (cur * len(picks) + O[m, :, t]) / (len(picks) + 1)
                scm = np.sqrt(((cand - yt) ** 2).mean())
                if best_sc is None or scm < best_sc - 1e-9:
                    best_sc, best_m = scm, m
            prev = np.sqrt(((cur - yt) ** 2).mean()) if picks else np.inf
            if best_sc >= prev - 1e-6:
                break
            picks.append(best_m)
            cur = (cur * (len(picks) - 1) + O[best_m, :, t]) / len(picks)
        for m in picks:
            sel_weights[m, t] += 1 / len(picks)
    greedy = np.einsum("mnt,mt->nt", O, sel_weights)
    per_t_g = np.sqrt(((greedy - Y) ** 2).mean(axis=0))
    print(f"greedy per-target blend MCRMSE: {mcrmse(Y, greedy):.5f}  " +
          " ".join(f"{t}:{v:.4f}" for t, v in zip(TARGETS, per_t_g)))
    print("greedy weights (variant x target):")
    for m, nm in enumerate(names):
        print(f"  {nm:12s} " + " ".join(f"{sel_weights[m, t]:.2f}" for t in range(len(TARGETS))))

    with open(f"{COMP}/scripts/blend_meta.json", "w") as fh:
        json.dump({"names": names,
                   "greedy_weights": sel_weights.tolist(),
                   "scores": {k: results[k]["score"] for k in names},
                   "equal_score": mcrmse(Y, eq),
                   "greedy_score": mcrmse(Y, greedy),
                   "naive": mcrmse(Y, base)}, fh, indent=1)


if __name__ == "__main__":
    main()
