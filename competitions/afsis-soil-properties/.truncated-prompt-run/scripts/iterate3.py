"""Iteration 3: kernel ridge (RBF + polynomial) members over spectral variants,
plus tuned SVR. Kernels are computed once per variant; alpha/gamma grids are
then closed-form cheap. Appends members to members.npz.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.svm import SVR

import importlib.util
_s = importlib.util.spec_from_file_location(
    "it2", "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties/scripts/iterate2.py")
it2 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(it2)

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = it2.TARGETS
N_FOLDS = 5


def standardize(Xtr, Xother):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, [(X - mu) / sd for X in Xother]


def sqdist(A, B):
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T


def krr_variant(X, Xte, Y, folds, kernel, gammas, alphas):
    """Return dict (gamma, alpha) -> (oof (n,T), pred (k,T)) evaluated jointly."""
    n, T = len(X), Y.shape[1]
    out = {}
    oof = {(g, a): np.zeros((n, T)) for g in gammas for a in alphas}
    for f in range(N_FOLDS):
        tr, va = folds != f, folds == f
        Xtr, (Xva,) = standardize(X[tr], [X[va]])
        ym = Y[tr].mean(0)
        Yc = Y[tr] - ym
        if kernel == "rbf":
            D_tr = sqdist(Xtr, Xtr)
            D_va = sqdist(Xva, Xtr)
        else:
            G_tr = Xtr @ Xtr.T / Xtr.shape[1]
            G_va = Xva @ Xtr.T / Xtr.shape[1]
        for g in gammas:
            if kernel == "rbf":
                K = np.exp(-g * D_tr / Xtr.shape[1])
                Kv = np.exp(-g * D_va / Xtr.shape[1])
            else:
                K = (G_tr + 1.0) ** g
                Kv = (G_va + 1.0) ** g
            w, Q = np.linalg.eigh(K)
            QtY = Q.T @ Yc
            KvQ = Kv @ Q
            for a in alphas:
                dual = QtY / (w + a)[:, None]
                oof[(g, a)][va] = KvQ @ dual + ym
    # full fit
    Xf, (Xt,) = standardize(X, [Xte])
    ym = Y.mean(0)
    Yc = Y - ym
    if kernel == "rbf":
        D_f, D_t = sqdist(Xf, Xf), sqdist(Xt, Xf)
    else:
        G_f, G_t = Xf @ Xf.T / Xf.shape[1], Xt @ Xf.T / Xf.shape[1]
    for g in gammas:
        if kernel == "rbf":
            K = np.exp(-g * D_f / Xf.shape[1])
            Kt = np.exp(-g * D_t / Xf.shape[1])
        else:
            K = (G_f + 1.0) ** g
            Kt = (G_t + 1.0) ** g
        w, Q = np.linalg.eigh(K)
        QtY = Q.T @ Yc
        KtQ = Kt @ Q
        for a in alphas:
            dual = QtY / (w + a)[:, None]
            out[(g, a)] = (oof[(g, a)], KtQ @ dual + ym)
    return out


def mcrmse(Y, P):
    return float(np.sqrt(((P - Y) ** 2).mean(0)).mean())


def main():
    train, test, V = it2.build()
    Y = train[TARGETS].values.astype(np.float64)
    folds = it2.group_folds(train)
    old = np.load(f"{COMP}/scripts/members.npz", allow_pickle=True)
    members = {str(k): (old[f"oof__{k}"], old[f"pred__{k}"]) for k in
               [str(x) for x in old["names"]]}
    t0 = time.time()

    gammas_rbf = [0.2, 0.5, 1.0, 2.0]
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
    for vname in ["raw", "sg1", "snv", "snv_sg1", "raw_spatial"]:
        X, Xte = V[vname]
        res = krr_variant(X, Xte, Y, folds, "rbf", gammas_rbf, alphas)
        # keep best (g,a) per target as one combined member + best overall member
        scores = {k: mcrmse(Y, v[0]) for k, v in res.items()}
        bk = min(scores, key=scores.get)
        members[f"krr_{vname}"] = res[bk]
        # per-target best combination member
        oof_pt = np.zeros_like(Y)
        pred_pt = np.zeros((len(Xte), len(TARGETS)))
        chosen = []
        for t in range(len(TARGETS)):
            bt = min(res, key=lambda k: np.sqrt(((res[k][0][:, t] - Y[:, t]) ** 2).mean()))
            oof_pt[:, t] = res[bt][0][:, t]
            pred_pt[:, t] = res[bt][1][:, t]
            chosen.append(bt)
        members[f"krrpt_{vname}"] = (oof_pt, pred_pt)
        print(f"krr_{vname:12s} best{bk} {scores[bk]:.5f} | per-target {mcrmse(Y, oof_pt):.5f} "
              f"{chosen} [{time.time()-t0:.0f}s]")

    for vname in ["raw", "sg1"]:
        X, Xte = V[vname]
        res = krr_variant(X, Xte, Y, folds, "poly", [2, 3], [0.1, 1.0, 10.0])
        scores = {k: mcrmse(Y, v[0]) for k, v in res.items()}
        bk = min(scores, key=scores.get)
        members[f"krrpoly_{vname}"] = res[bk]
        print(f"krrpoly_{vname:8s} best{bk} {scores[bk]:.5f} [{time.time()-t0:.0f}s]")

    # tuned SVR: sweep C/epsilon on sg1 (best solo SVR) and raw
    for vname, C, eps in [("sg1", 100.0, 0.05), ("sg1", 1000.0, 0.1),
                          ("raw", 1000.0, 0.1), ("snv_sg1", 1000.0, 0.1)]:
        X, Xte = V[vname]
        oof, pred = it2.svr_member(X, Xte, Y, folds, C=C, eps=eps)
        key = f"svr_{vname}_C{int(C)}_e{eps}"
        members[key] = (oof, pred)
        print(f"{key:24s} {mcrmse(Y, oof):.5f} [{time.time()-t0:.0f}s]")

    np.savez_compressed(
        f"{COMP}/scripts/members.npz", folds=folds, Y=Y,
        names=np.array(list(members.keys())),
        **{f"oof__{k}": v[0] for k, v in members.items()},
        **{f"pred__{k}": v[1] for k, v in members.items()})
    print(f"\ntotal members {len(members)}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
