"""Iteration 5:
(a) diagnostic — honest inner-CV hyperparameter selection for KRR, to size the
    selection bias in the krr*pt members (which pick (gamma, alpha) on full OOF);
(b) new members — KRR on log-shifted skewed targets (Ca, P, SOC) and on
    winsorized targets (P is skew 7.45; extreme rows destabilize the fit).
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
ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3]


def extra_variants(train, test, V):
    spec = [c for c in train.columns if c.startswith("m")]
    wn = np.array([float(c[1:]) for c in spec])
    keep = ~((wn >= 2352.0) & (wn <= 2380.0))
    A = train[spec].values.astype(np.float64)
    B = test[spec].values.astype(np.float64)
    sa, sb = it2.snv(A), it2.snv(B)
    V["sg1_w41"] = (savgol_filter(A, 41, 2, deriv=1, axis=1)[:, keep],
                    savgol_filter(B, 41, 2, deriv=1, axis=1)[:, keep])
    V["snv_sg1_w41"] = (savgol_filter(sa, 41, 2, deriv=1, axis=1)[:, keep],
                        savgol_filter(sb, 41, 2, deriv=1, axis=1)[:, keep])
    return V


def krr_oof_pred(X, Xte, Yc, folds, g, a):
    """Single (gamma, alpha) KRR, returns oof and full-fit test pred."""
    n, T = len(X), Yc.shape[1]
    oof = np.zeros((n, T))
    for f in range(N_FOLDS):
        tr, va = folds != f, folds == f
        Xtr, (Xva,) = it3.standardize(X[tr], [X[va]])
        ym = Yc[tr].mean(0)
        K = np.exp(-g * it3.sqdist(Xtr, Xtr) / Xtr.shape[1])
        Kv = np.exp(-g * it3.sqdist(Xva, Xtr) / Xtr.shape[1])
        w, Q = np.linalg.eigh(K)
        dual = (Q.T @ (Yc[tr] - ym)) / (w + a)[:, None]
        oof[va] = Kv @ Q @ dual + ym
    Xf, (Xt,) = it3.standardize(X, [Xte])
    ym = Yc.mean(0)
    K = np.exp(-g * it3.sqdist(Xf, Xf) / Xf.shape[1])
    Kt = np.exp(-g * it3.sqdist(Xt, Xf) / Xf.shape[1])
    w, Q = np.linalg.eigh(K)
    dual = (Q.T @ (Yc - ym)) / (w + a)[:, None]
    return oof, Kt @ Q @ dual + ym


def honest_inner_cv(X, Y, folds):
    """Per outer fold: pick (gamma, alpha) per target by inner 4-fold CV on the
    outer-train rows only, then score on the held-out outer fold."""
    n, T = len(X), Y.shape[1]
    oof = np.zeros((n, T))
    for f in range(N_FOLDS):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        inner = folds[tr_idx]
        inner_labels = np.unique(inner)
        Xtr_all, Y_tr = X[tr_idx], Y[tr_idx]
        # inner OOF per (g, a)
        inner_oof = {(g, a): np.zeros((len(tr_idx), T)) for g in GAMMAS for a in ALPHAS}
        for lab in inner_labels:
            itr = inner != lab
            iva = inner == lab
            Xi, (Xv,) = it3.standardize(Xtr_all[itr], [Xtr_all[iva]])
            ym = Y_tr[itr].mean(0)
            D = it3.sqdist(Xi, Xi)
            Dv = it3.sqdist(Xv, Xi)
            for g in GAMMAS:
                K = np.exp(-g * D / Xi.shape[1])
                Kv = np.exp(-g * Dv / Xi.shape[1])
                w, Q = np.linalg.eigh(K)
                QtY = Q.T @ (Y_tr[itr] - ym)
                KvQ = Kv @ Q
                for a in ALPHAS:
                    inner_oof[(g, a)][iva] = KvQ @ (QtY / (w + a)[:, None]) + ym
        best = []
        for t in range(T):
            best.append(min(inner_oof,
                            key=lambda k: np.sqrt(((inner_oof[k][:, t] - Y_tr[:, t]) ** 2).mean())))
        # refit on outer-train, predict outer-val
        Xi, (Xv,) = it3.standardize(Xtr_all, [X[va_idx]])
        ym = Y_tr.mean(0)
        D, Dv = it3.sqdist(Xi, Xi), it3.sqdist(Xv, Xi)
        for t in range(T):
            g, a = best[t]
            K = np.exp(-g * D / Xi.shape[1])
            Kv = np.exp(-g * Dv / Xi.shape[1])
            w, Q = np.linalg.eigh(K)
            dual = (Q.T @ (Y_tr[:, t] - ym[t])) / (w + a)
            oof[va_idx, t] = Kv @ Q @ dual + ym[t]
    return oof


def main():
    train, test, V = it2.build()
    V = extra_variants(train, test, V)
    Y = train[TARGETS].values.astype(np.float64)
    folds = it2.group_folds(train)
    t0 = time.time()

    # (a) honest diagnostic
    for vname in ["snv_sg1_w41", "sg1"]:
        oof = honest_inner_cv(V[vname][0], Y, folds)
        per_t = np.sqrt(((oof - Y) ** 2).mean(0))
        print(f"[honest inner-CV] krr_{vname}: MCRMSE {per_t.mean():.5f}  " +
              " ".join(f"{t}:{v:.4f}" for t, v in zip(TARGETS, per_t)) +
              f"  [{time.time()-t0:.0f}s]")

    old = np.load(f"{COMP}/scripts/members.npz", allow_pickle=True)
    members = {str(k): (old[f"oof__{k}"], old[f"pred__{k}"])
               for k in [str(x) for x in old["names"]]}

    # (b) log-shifted targets for skewed columns
    shift = Y.min(0) - 1e-3
    Ylog = np.log(Y - shift)
    for vname in ["snv_sg1_w41", "sg1", "snv_sg1", "raw"]:
        X, Xte = V[vname]
        best_oof = np.zeros_like(Y); best_pred = np.zeros((len(Xte), len(Y[0])))
        chosen = []
        cache = {}
        for g in [0.06, 0.1, 0.2, 0.35]:
            for a in [0.003, 0.01, 0.03, 0.1]:
                cache[(g, a)] = krr_oof_pred(X, Xte, Ylog, folds, g, a)
        for t in range(len(TARGETS)):
            def sc(k):
                back = np.exp(cache[k][0][:, t]) + shift[t]
                return np.sqrt(((back - Y[:, t]) ** 2).mean())
            bk = min(cache, key=sc)
            chosen.append((bk, round(sc(bk), 4)))
            best_oof[:, t] = np.exp(cache[bk][0][:, t]) + shift[t]
            best_pred[:, t] = np.exp(cache[bk][1][:, t]) + shift[t]
        members[f"krrlog_{vname}"] = (best_oof, best_pred)
        print(f"krrlog_{vname:12s} {it3.mcrmse(Y, best_oof):.5f} {chosen} [{time.time()-t0:.0f}s]")

    # (c) winsorized targets (cap at 99th pct) — tame heavy tails during fit
    Ywin = Y.copy()
    caps = np.percentile(Y, 99, axis=0)
    for t in range(len(TARGETS)):
        Ywin[:, t] = np.minimum(Ywin[:, t], caps[t])
    for vname in ["snv_sg1_w41", "sg1"]:
        X, Xte = V[vname]
        best_oof = np.zeros_like(Y); best_pred = np.zeros((len(Xte), len(TARGETS)))
        cache = {}
        for g in [0.1, 0.2, 0.35]:
            for a in [0.01, 0.03, 0.1]:
                cache[(g, a)] = krr_oof_pred(X, Xte, Ywin, folds, g, a)
        for t in range(len(TARGETS)):
            bk = min(cache, key=lambda k: np.sqrt(((cache[k][0][:, t] - Y[:, t]) ** 2).mean()))
            best_oof[:, t] = cache[bk][0][:, t]
            best_pred[:, t] = cache[bk][1][:, t]
        members[f"krrwin_{vname}"] = (best_oof, best_pred)
        print(f"krrwin_{vname:12s} {it3.mcrmse(Y, best_oof):.5f} [{time.time()-t0:.0f}s]")

    np.savez_compressed(
        f"{COMP}/scripts/members.npz", folds=folds, Y=Y,
        names=np.array(list(members.keys())),
        **{f"oof__{k}": v[0] for k, v in members.items()},
        **{f"pred__{k}": v[1] for k, v in members.items()})
    print(f"\ntotal members {len(members)}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
