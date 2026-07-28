"""Iteration 4: refine kernel-ridge RBF grid (gamma hit lower boundary in it3),
widen variant coverage, add derivative windows. Appends members to members.npz.
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


def main():
    train, test, V = it2.build()
    Y = train[TARGETS].values.astype(np.float64)
    folds = it2.group_folds(train)

    # extra preprocessing variants
    import pandas as pd
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

    old = np.load(f"{COMP}/scripts/members.npz", allow_pickle=True)
    members = {str(k): (old[f"oof__{k}"], old[f"pred__{k}"])
               for k in [str(x) for x in old["names"]]}
    t0 = time.time()

    gammas = [0.03, 0.06, 0.1, 0.2, 0.35]
    alphas = [0.003, 0.01, 0.03, 0.1, 0.3]
    for vname in ["sg1", "snv_sg1", "sg2", "snv_sg2", "sg1_spatial",
                  "sg1_w11", "sg1_w41", "snv_sg1_w11", "snv_sg1_w41", "sg1_fp"]:
        X, Xte = V[vname]
        res = it3.krr_variant(X, Xte, Y, folds, "rbf", gammas, alphas)
        scores = {k: it3.mcrmse(Y, v[0]) for k, v in res.items()}
        bk = min(scores, key=scores.get)
        members[f"krr2_{vname}"] = res[bk]
        oof_pt = np.zeros_like(Y)
        pred_pt = np.zeros((len(Xte), len(TARGETS)))
        chosen = []
        for t in range(len(TARGETS)):
            bt = min(res, key=lambda k: np.sqrt(((res[k][0][:, t] - Y[:, t]) ** 2).mean()))
            oof_pt[:, t] = res[bt][0][:, t]
            pred_pt[:, t] = res[bt][1][:, t]
            chosen.append(bt)
        members[f"krr2pt_{vname}"] = (oof_pt, pred_pt)
        print(f"krr2_{vname:14s} best{bk} {scores[bk]:.5f} | pt {it3.mcrmse(Y, oof_pt):.5f} "
              f"{chosen} [{time.time()-t0:.0f}s]")

    np.savez_compressed(
        f"{COMP}/scripts/members.npz", folds=folds, Y=Y,
        names=np.array(list(members.keys())),
        **{f"oof__{k}": v[0] for k, v in members.items()},
        **{f"pred__{k}": v[1] for k, v in members.items()})
    print(f"\ntotal members {len(members)}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
