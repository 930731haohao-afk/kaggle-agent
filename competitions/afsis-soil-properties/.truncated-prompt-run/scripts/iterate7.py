"""Iteration 7: Laplacian-kernel KRR (L1 distance geometry — genuinely different
from the RBF members) with honest inner-CV hyperparameter selection.
Appends to members_honest.npz.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import importlib.util
_b = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties/scripts"
_s = importlib.util.spec_from_file_location("it2", f"{_b}/iterate2.py")
it2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(it2)
_s3 = importlib.util.spec_from_file_location("it3", f"{_b}/iterate3.py")
it3 = importlib.util.module_from_spec(_s3); _s3.loader.exec_module(it3)
_s6 = importlib.util.spec_from_file_location("it6", f"{_b}/iterate6.py")
it6 = importlib.util.module_from_spec(_s6); _s6.loader.exec_module(it6)

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = it2.TARGETS
N_FOLDS = 5
GAMMAS = [0.3, 0.6, 1.0, 2.0]
ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3]


def lap_solve(Xi, Xv, Ytr, gammas, alphas):
    """Return dict (g,a) -> val preds. Distances scaled by mean L1 distance."""
    D = cdist(Xi, Xi, "cityblock")
    Dv = cdist(Xv, Xi, "cityblock")
    scale = D.mean() + 1e-12
    ym = Ytr.mean(0)
    out = {}
    for g in gammas:
        K = np.exp(-g * D / scale)
        Kv = np.exp(-g * Dv / scale)
        w, Q = np.linalg.eigh(K)
        QtY, KvQ = Q.T @ (Ytr - ym), Kv @ Q
        for a in alphas:
            out[(g, a)] = KvQ @ (QtY / (w + a)[:, None]) + ym
    return out


def lap_honest(X, Xte, Y, folds):
    n, T = len(X), Y.shape[1]
    oof = np.zeros((n, T))
    for f in range(N_FOLDS):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        inner_of = folds[tr_idx]
        Xa, Ya = X[tr_idx], Y[tr_idx]
        grid = {(g, a): np.zeros((len(tr_idx), T)) for g in GAMMAS for a in ALPHAS}
        for lab in np.unique(inner_of):
            itr, iva = inner_of != lab, inner_of == lab
            Xi, (Xv,) = it3.standardize(Xa[itr], [Xa[iva]])
            res = lap_solve(Xi, Xv, Ya[itr], GAMMAS, ALPHAS)
            for k, v in res.items():
                grid[k][iva] = v
        best = [min(grid, key=lambda k: np.sqrt(((grid[k][:, t] - Ya[:, t]) ** 2).mean()))
                for t in range(T)]
        Xi, (Xv,) = it3.standardize(Xa, [X[va_idx]])
        res = lap_solve(Xi, Xv, Ya, sorted({k[0] for k in best}),
                        sorted({k[1] for k in best}))
        for t in range(T):
            oof[va_idx, t] = res[best[t]][:, t]
    allidx = np.arange(n)
    grid = {(g, a): np.zeros((n, T)) for g in GAMMAS for a in ALPHAS}
    for lab in np.unique(folds):
        itr, iva = folds != lab, folds == lab
        Xi, (Xv,) = it3.standardize(X[itr], [X[iva]])
        for k, v in lap_solve(Xi, Xv, Y[itr], GAMMAS, ALPHAS).items():
            grid[k][iva] = v
    best = [min(grid, key=lambda k: np.sqrt(((grid[k][:, t] - Y[:, t]) ** 2).mean()))
            for t in range(T)]
    Xf, (Xt,) = it3.standardize(X, [Xte])
    res = lap_solve(Xf, Xt, Y, sorted({k[0] for k in best}), sorted({k[1] for k in best}))
    pred = np.column_stack([res[best[t]][:, t] for t in range(T)])
    return oof, pred, best


def main():
    train = pd.read_csv(f"{COMP}/data/training.csv")
    test = pd.read_csv(f"{COMP}/data/sorted_test.csv")
    V = it6.all_variants(train, test)
    Y = train[TARGETS].values.astype(np.float64)
    folds = it2.group_folds(train)
    old = np.load(f"{COMP}/scripts/members_honest.npz", allow_pickle=True)
    members = {str(k): (old[f"oof__{k}"], old[f"pred__{k}"])
               for k in [str(x) for x in old["names"]]}
    t0 = time.time()

    for v in ["snv_sg1_w41", "sg1", "raw", "snv_sg1"]:
        oof, pred, best = lap_honest(V[v][0], V[v][1], Y, folds)
        members[f"H_lap_{v}"] = (oof, pred)
        print(f"H_lap_{v:14s} {it3.mcrmse(Y, oof):.5f}  hp {best} [{time.time()-t0:.0f}s]")

    np.savez_compressed(
        f"{COMP}/scripts/members_honest.npz", folds=folds, Y=Y,
        names=np.array(list(members.keys())),
        **{f"oof__{k}": v[0] for k, v in members.items()},
        **{f"pred__{k}": v[1] for k, v in members.items()})
    print(f"\ntotal {len(members)} members, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
