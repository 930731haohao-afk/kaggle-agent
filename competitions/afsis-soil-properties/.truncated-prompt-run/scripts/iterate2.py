"""Iteration 2: add diverse members (PLS, SVR-RBF, log-target Ridge) and
per-target shrink calibration. Same GroupKFold folds as train_ridge.py.
Caches every member OOF + test pred to npz so later rounds are cheap.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import json
import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]
SPATIAL = ["BSAN", "BSAS", "BSAV", "CTI", "ELEV", "EVI", "LSTD", "LSTN",
           "REF1", "REF2", "REF3", "REF7", "RELI", "TMAP", "TMFI"]
ALPHAS = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
N_FOLDS, SEED = 5, 42


def snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def build():
    train = pd.read_csv(f"{COMP}/data/training.csv")
    test = pd.read_csv(f"{COMP}/data/sorted_test.csv")
    spec = [c for c in train.columns if c.startswith("m")]
    wn = np.array([float(c[1:]) for c in spec])
    co2 = (wn >= 2352.0) & (wn <= 2380.0)
    keep = ~co2
    A = train[spec].values.astype(np.float64)
    B = test[spec].values.astype(np.float64)
    depth_tr = (train["Depth"] == "Topsoil").values.astype(np.float64)[:, None]
    depth_te = (test["Depth"] == "Topsoil").values.astype(np.float64)[:, None]
    spat_tr = train[SPATIAL].values.astype(np.float64)
    spat_te = test[SPATIAL].values.astype(np.float64)

    v = {}
    v["raw"] = (A[:, keep], B[:, keep])
    sg1 = (savgol_filter(A, 25, 2, deriv=1, axis=1), savgol_filter(B, 25, 2, deriv=1, axis=1))
    v["sg1"] = (sg1[0][:, keep], sg1[1][:, keep])
    sg2 = (savgol_filter(A, 31, 2, deriv=2, axis=1), savgol_filter(B, 31, 2, deriv=2, axis=1))
    v["sg2"] = (sg2[0][:, keep], sg2[1][:, keep])
    sa, sb = snv(A), snv(B)
    v["snv"] = (sa[:, keep], sb[:, keep])
    v["snv_sg1"] = (savgol_filter(sa, 25, 2, deriv=1, axis=1)[:, keep],
                    savgol_filter(sb, 25, 2, deriv=1, axis=1)[:, keep])
    v["raw_spatial"] = (np.hstack([v["raw"][0], spat_tr, depth_tr]),
                        np.hstack([v["raw"][1], spat_te, depth_te]))
    v["sg1_spatial"] = (np.hstack([v["sg1"][0], spat_tr, depth_tr]),
                        np.hstack([v["sg1"][1], spat_te, depth_te]))
    # fingerprint region only (< 2500 cm-1): high-signal MIR region
    fp = keep & (wn < 2500.0)
    v["raw_fp"] = (A[:, fp], B[:, fp])
    v["sg1_fp"] = (sg1[0][:, fp], sg1[1][:, fp])
    return train, test, v


def group_folds(train):
    key = train[SPATIAL].round(6).apply(tuple, axis=1)
    g = key.astype("category").cat.codes.values
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(np.unique(g))
    fmap = {gg: i % N_FOLDS for i, gg in enumerate(perm)}
    return np.array([fmap[gg] for gg in g])


def ridge_member(X, Xte, Y, folds, alphas=ALPHAS):
    n, T = len(X), Y.shape[1]
    oof_all = np.zeros((len(alphas), n, T))
    for f in range(N_FOLDS):
        tr, va = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        ym = Y[tr].mean(0)
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        UtY = U.T @ (Y[tr] - ym)
        XvaV = Xva @ Vt.T
        for ai, a in enumerate(alphas):
            d = s / (s ** 2 + a)
            oof_all[ai][va] = XvaV @ (d[:, None] * UtY) + ym
    rmse = np.sqrt(((oof_all - Y[None]) ** 2).mean(1))
    bi = rmse.argmin(0)
    oof = np.stack([oof_all[bi[t], :, t] for t in range(T)], 1)
    # full-fit test pred
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xf, Xt = (X - mu) / sd, (Xte - mu) / sd
    ym = Y.mean(0)
    U, s, Vt = np.linalg.svd(Xf, full_matrices=False)
    UtY = U.T @ (Y - ym)
    XtV = Xt @ Vt.T
    pred = np.zeros((len(Xte), T))
    for t in range(T):
        d = s / (s ** 2 + ALPHAS[bi[t]])
        pred[:, t] = XtV @ (d * UtY[:, t]) + ym[t]
    return oof, pred, ALPHAS[bi]


def pls_member(X, Xte, Y, folds, ncomp):
    n, T = len(X), Y.shape[1]
    oof = np.zeros((n, T))
    for f in range(N_FOLDS):
        tr, va = folds != f, folds == f
        m = PLSRegression(n_components=ncomp, scale=True)
        m.fit(X[tr], Y[tr])
        oof[va] = m.predict(X[va])
    m = PLSRegression(n_components=ncomp, scale=True)
    m.fit(X, Y)
    return oof, np.asarray(m.predict(Xte))


def svr_member(X, Xte, Y, folds, C=10000.0, gamma=0.0, eps=0.1):
    """SVR-RBF per target on standardized features."""
    n, T = len(X), Y.shape[1]
    oof = np.zeros((n, T))
    pred = np.zeros((len(Xte), T))
    for f in range(N_FOLDS):
        tr, va = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        g = gamma if gamma > 0 else 1.0 / Xtr.shape[1]
        for t in range(T):
            s = SVR(C=C, gamma=g, epsilon=eps, kernel="rbf", cache_size=800)
            s.fit(Xtr, Y[tr, t])
            oof[va, t] = s.predict(Xva)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xf, Xt = (X - mu) / sd, (Xte - mu) / sd
    g = gamma if gamma > 0 else 1.0 / Xf.shape[1]
    for t in range(T):
        s = SVR(C=C, gamma=g, epsilon=eps, kernel="rbf", cache_size=800)
        s.fit(Xf, Y[:, t])
        pred[:, t] = s.predict(Xt)
    return oof, pred


def mcrmse(Y, P):
    return float(np.sqrt(((P - Y) ** 2).mean(0)).mean())


def main():
    train, test, V = build()
    Y = train[TARGETS].values.astype(np.float64)
    folds = group_folds(train)
    members = {}
    t0 = time.time()

    for name in ["raw", "sg1", "sg2", "snv", "snv_sg1", "raw_spatial", "sg1_spatial",
                 "raw_fp", "sg1_fp"]:
        X, Xte = V[name]
        oof, pred, al = ridge_member(X, Xte, Y, folds)
        members[f"ridge_{name}"] = (oof, pred)
        print(f"ridge_{name:12s} {mcrmse(Y, oof):.5f}  alphas {al}  [{time.time()-t0:.0f}s]")

    for name, nc in [("raw", 30), ("sg1", 20), ("snv", 30), ("raw_fp", 30)]:
        X, Xte = V[name]
        oof, pred = pls_member(X, Xte, Y, folds, nc)
        members[f"pls{nc}_{name}"] = (oof, pred)
        print(f"pls{nc}_{name:10s} {mcrmse(Y, oof):.5f}  [{time.time()-t0:.0f}s]")

    for name, C in [("raw", 10000.0), ("sg1", 10000.0), ("snv", 10000.0)]:
        X, Xte = V[name]
        oof, pred = svr_member(X, Xte, Y, folds, C=C)
        members[f"svr_{name}"] = (oof, pred)
        print(f"svr_{name:12s} {mcrmse(Y, oof):.5f}  [{time.time()-t0:.0f}s]")

    np.savez_compressed(
        f"{COMP}/scripts/members.npz", folds=folds, Y=Y,
        names=np.array(list(members.keys())),
        **{f"oof__{k}": v[0] for k, v in members.items()},
        **{f"pred__{k}": v[1] for k, v in members.items()})
    print(f"\nsaved {len(members)} members, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
