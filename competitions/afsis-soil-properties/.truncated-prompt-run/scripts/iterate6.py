"""Iteration 6: rebuild the member pool with HONEST hyperparameter selection.

Every KRR / Ridge member picks (gamma, alpha) by an inner CV on the outer-train
rows only, so its OOF vector carries no selection leakage. Test predictions use
hyperparameters selected by CV over the whole training set, refit on all rows.
Fixed-hyperparameter members (SVR, PLS) are copied over from members.npz —
they never selected anything on OOF, so they were already honest.
Output: members_honest.npz
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import time
import numpy as np
from scipy.signal import savgol_filter

import importlib.util
_b = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties/scripts"
_s = importlib.util.spec_from_file_location("it2", f"{_b}/iterate2.py")
it2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(it2)
_s3 = importlib.util.spec_from_file_location("it3", f"{_b}/iterate3.py")
it3 = importlib.util.module_from_spec(_s3); _s3.loader.exec_module(it3)

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = it2.TARGETS
N_FOLDS = 5
GAMMAS = [0.03, 0.06, 0.1, 0.2, 0.35]
KALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3]
RALPHAS = it2.ALPHAS


def all_variants(train, test):
    _, _, V = it2.build()
    spec = [c for c in train.columns if c.startswith("m")]
    wn = np.array([float(c[1:]) for c in spec])
    keep = ~((wn >= 2352.0) & (wn <= 2380.0))
    A = train[spec].values.astype(np.float64)
    B = test[spec].values.astype(np.float64)
    sa, sb = it2.snv(A), it2.snv(B)
    for w in (11, 41):
        V[f"sg1_w{w}"] = (savgol_filter(A, w, 2, deriv=1, axis=1)[:, keep],
                          savgol_filter(B, w, 2, deriv=1, axis=1)[:, keep])
        V[f"snv_sg1_w{w}"] = (savgol_filter(sa, w, 2, deriv=1, axis=1)[:, keep],
                              savgol_filter(sb, w, 2, deriv=1, axis=1)[:, keep])
    V["snv_sg2"] = (savgol_filter(sa, 31, 2, deriv=2, axis=1)[:, keep],
                    savgol_filter(sb, 31, 2, deriv=2, axis=1)[:, keep])
    return V


def krr_grid_oof(X, Y, tr_idx, va_idx, inner_labels, inner_of):
    """Inner-CV OOF over the grid for rows tr_idx. Returns dict key -> oof."""
    T = Y.shape[1]
    out = {(g, a): np.zeros((len(tr_idx), T)) for g in GAMMAS for a in KALPHAS}
    Xa, Ya = X[tr_idx], Y[tr_idx]
    for lab in inner_labels:
        itr, iva = inner_of != lab, inner_of == lab
        Xi, (Xv,) = it3.standardize(Xa[itr], [Xa[iva]])
        ym = Ya[itr].mean(0)
        D, Dv = it3.sqdist(Xi, Xi), it3.sqdist(Xv, Xi)
        for g in GAMMAS:
            K = np.exp(-g * D / Xi.shape[1])
            Kv = np.exp(-g * Dv / Xi.shape[1])
            w, Q = np.linalg.eigh(K)
            QtY, KvQ = Q.T @ (Ya[itr] - ym), Kv @ Q
            for a in KALPHAS:
                out[(g, a)][iva] = KvQ @ (QtY / (w + a)[:, None]) + ym
    return out


def krr_honest(X, Xte, Y, folds):
    n, T = len(X), Y.shape[1]
    oof = np.zeros((n, T))
    for f in range(N_FOLDS):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        inner_of = folds[tr_idx]
        grid = krr_grid_oof(X, Y, tr_idx, va_idx, np.unique(inner_of), inner_of)
        Ytr = Y[tr_idx]
        best = [min(grid, key=lambda k: np.sqrt(((grid[k][:, t] - Ytr[:, t]) ** 2).mean()))
                for t in range(T)]
        Xi, (Xv,) = it3.standardize(X[tr_idx], [X[va_idx]])
        ym = Ytr.mean(0)
        D, Dv = it3.sqdist(Xi, Xi), it3.sqdist(Xv, Xi)
        for t in range(T):
            g, a = best[t]
            K = np.exp(-g * D / Xi.shape[1])
            Kv = np.exp(-g * Dv / Xi.shape[1])
            w, Q = np.linalg.eigh(K)
            oof[va_idx, t] = Kv @ Q @ ((Q.T @ (Ytr[:, t] - ym[t])) / (w + a)) + ym[t]
    # test: select on full-train CV, refit on all rows
    allidx = np.arange(n)
    grid = krr_grid_oof(X, Y, allidx, allidx, np.unique(folds), folds)
    best = [min(grid, key=lambda k: np.sqrt(((grid[k][:, t] - Y[:, t]) ** 2).mean()))
            for t in range(T)]
    Xf, (Xt,) = it3.standardize(X, [Xte])
    ym = Y.mean(0)
    D, Dt = it3.sqdist(Xf, Xf), it3.sqdist(Xt, Xf)
    pred = np.zeros((len(Xte), T))
    for t in range(T):
        g, a = best[t]
        K = np.exp(-g * D / Xf.shape[1])
        Kt = np.exp(-g * Dt / Xf.shape[1])
        w, Q = np.linalg.eigh(K)
        pred[:, t] = Kt @ Q @ ((Q.T @ (Y[:, t] - ym[t])) / (w + a)) + ym[t]
    return oof, pred, best


def ridge_grid_oof(X, Y, rows, labels, lab_of):
    T = Y.shape[1]
    out = {a: np.zeros((len(rows), T)) for a in RALPHAS}
    Xa, Ya = X[rows], Y[rows]
    for lab in labels:
        itr, iva = lab_of != lab, lab_of == lab
        Xi, (Xv,) = it3.standardize(Xa[itr], [Xa[iva]])
        ym = Ya[itr].mean(0)
        U, s, Vt = np.linalg.svd(Xi, full_matrices=False)
        UtY, XvV = U.T @ (Ya[itr] - ym), Xv @ Vt.T
        for a in RALPHAS:
            d = s / (s ** 2 + a)
            out[a][iva] = XvV @ (d[:, None] * UtY) + ym
    return out


def ridge_honest(X, Xte, Y, folds):
    n, T = len(X), Y.shape[1]
    oof = np.zeros((n, T))
    for f in range(N_FOLDS):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        inner_of = folds[tr_idx]
        grid = ridge_grid_oof(X, Y, tr_idx, np.unique(inner_of), inner_of)
        Ytr = Y[tr_idx]
        best = [min(grid, key=lambda a: np.sqrt(((grid[a][:, t] - Ytr[:, t]) ** 2).mean()))
                for t in range(T)]
        Xi, (Xv,) = it3.standardize(X[tr_idx], [X[va_idx]])
        ym = Ytr.mean(0)
        U, s, Vt = np.linalg.svd(Xi, full_matrices=False)
        UtY, XvV = U.T @ (Ytr - ym), Xv @ Vt.T
        for t in range(T):
            d = s / (s ** 2 + best[t])
            oof[va_idx, t] = XvV @ (d * UtY[:, t]) + ym[t]
    allidx = np.arange(n)
    grid = ridge_grid_oof(X, Y, allidx, np.unique(folds), folds)
    best = [min(grid, key=lambda a: np.sqrt(((grid[a][:, t] - Y[:, t]) ** 2).mean()))
            for t in range(T)]
    Xf, (Xt,) = it3.standardize(X, [Xte])
    ym = Y.mean(0)
    U, s, Vt = np.linalg.svd(Xf, full_matrices=False)
    UtY, XtV = U.T @ (Y - ym), Xt @ Vt.T
    pred = np.zeros((len(Xte), T))
    for t in range(T):
        d = s / (s ** 2 + best[t])
        pred[:, t] = XtV @ (d * UtY[:, t]) + ym[t]
    return oof, pred, best


def main():
    import pandas as pd
    train = pd.read_csv(f"{COMP}/data/training.csv")
    test = pd.read_csv(f"{COMP}/data/sorted_test.csv")
    V = all_variants(train, test)
    Y = train[TARGETS].values.astype(np.float64)
    folds = it2.group_folds(train)
    t0 = time.time()
    members = {}

    krr_vars = ["raw", "sg1", "sg2", "snv", "snv_sg1", "snv_sg2", "sg1_w11",
                "sg1_w41", "snv_sg1_w11", "snv_sg1_w41", "sg1_spatial",
                "raw_spatial", "sg1_fp", "raw_fp"]
    for v in krr_vars:
        oof, pred, best = krr_honest(V[v][0], V[v][1], Y, folds)
        members[f"H_krr_{v}"] = (oof, pred)
        print(f"H_krr_{v:14s} {it3.mcrmse(Y, oof):.5f}  test-hp {best} [{time.time()-t0:.0f}s]")

    for v in ["raw", "sg1", "snv", "snv_sg1", "raw_spatial", "sg1_spatial", "sg1_w41"]:
        oof, pred, best = ridge_honest(V[v][0], V[v][1], Y, folds)
        members[f"H_ridge_{v}"] = (oof, pred)
        print(f"H_ridge_{v:12s} {it3.mcrmse(Y, oof):.5f}  test-alpha {best} [{time.time()-t0:.0f}s]")

    # copy fixed-hyperparameter members (already honest)
    old = np.load(f"{COMP}/scripts/members.npz", allow_pickle=True)
    for k in [str(x) for x in old["names"]]:
        if k.startswith("svr_") or k.startswith("pls"):
            members[k] = (old[f"oof__{k}"], old[f"pred__{k}"])
    print(f"copied {sum(1 for k in members if k.startswith(('svr_', 'pls')))} fixed-hp members")

    np.savez_compressed(
        f"{COMP}/scripts/members_honest.npz", folds=folds, Y=Y,
        names=np.array(list(members.keys())),
        **{f"oof__{k}": v[0] for k, v in members.items()},
        **{f"pred__{k}": v[1] for k, v in members.items()})
    print(f"\ntotal honest members {len(members)}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
