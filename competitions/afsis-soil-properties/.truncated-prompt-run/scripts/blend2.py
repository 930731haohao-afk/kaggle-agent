"""Blend v2: compare plain greedy vs bagged greedy (bootstrap-stabilized
forward selection) per target. Method is chosen by NESTED (leave-fold-out)
MCRMSE — the honest estimate — then final weights are fit on the full OOF.
Writes submission.csv.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import json
import numpy as np
import pandas as pd

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]
N_FOLDS, SEED = 5, 42


def greedy(O, y, idx, n_steps=25):
    M = O.shape[0]
    picks, cur, best_prev = [], np.zeros(len(idx)), np.inf
    Oi, yi = O[:, idx], y[idx]
    for _ in range(n_steps):
        cands = (cur[None] * len(picks) + Oi) / (len(picks) + 1)
        sc = np.sqrt(((cands - yi[None]) ** 2).mean(1))
        m = int(sc.argmin())
        if sc[m] >= best_prev - 1e-7:
            break
        picks.append(m)
        cur = (cur * (len(picks) - 1) + Oi[m]) / len(picks)
        best_prev = sc[m]
    w = np.zeros(M)
    for m in picks:
        w[m] += 1.0 / max(len(picks), 1)
    return w


def bagged_greedy(O, y, idx, n_bags=25, frac=0.5, seed=SEED):
    """Caruana-style bagging: each bag sees a random subset of members."""
    M = O.shape[0]
    rng = np.random.RandomState(seed)
    W = np.zeros(M)
    k = max(3, int(M * frac))
    for b in range(n_bags):
        sub = rng.choice(M, k, replace=False)
        w = greedy(O[sub], y, idx)
        W[sub] += w
    return W / n_bags


def evaluate(method, O, Y, folds):
    """Return (full_oof_weights, full_score, nested_score)."""
    M, n, T = O.shape
    allidx = np.arange(n)
    W = np.zeros((M, T))
    for t in range(T):
        W[:, t] = method(O[:, :, t], Y[:, t], allidx)
    blend = np.einsum("mnt,mt->nt", O, W)
    full = float(np.sqrt(((blend - Y) ** 2).mean(0)).mean())
    nested = np.zeros_like(Y)
    for f in range(N_FOLDS):
        fit = np.where(folds != f)[0]
        hold = np.where(folds == f)[0]
        for t in range(T):
            w = method(O[:, :, t], Y[:, t], fit)
            nested[hold, t] = O[:, hold, t].T @ w
    nsc = float(np.sqrt(((nested - Y) ** 2).mean(0)).mean())
    return W, full, nsc, np.sqrt(((blend - Y) ** 2).mean(0))


def main():
    d = np.load(f"{COMP}/scripts/members.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    Y, folds = d["Y"], d["folds"]
    O = np.stack([d[f"oof__{n}"] for n in names])
    P = np.stack([d[f"pred__{n}"] for n in names])
    print(f"{O.shape[0]} members")
    solo = np.array([np.sqrt(((O[m] - Y) ** 2).mean(0)).mean() for m in range(len(names))])
    order = solo.argsort()[:5]
    print("top-5 solo:", {names[i]: round(solo[i], 5) for i in order})

    methods = {
        "greedy": greedy,
        "bagged_f0.5": lambda o, y, i: bagged_greedy(o, y, i, 25, 0.5),
        "bagged_f0.3": lambda o, y, i: bagged_greedy(o, y, i, 30, 0.3),
    }
    results = {}
    for mname, fn in methods.items():
        W, full, nsc, per_t = evaluate(fn, O, Y, folds)
        results[mname] = (W, full, nsc, per_t)
        print(f"{mname:12s} full-OOF {full:.5f}  nested {nsc:.5f}  "
              f"optimism {nsc-full:+.5f}  per-target " +
              " ".join(f"{t}:{v:.4f}" for t, v in zip(TARGETS, per_t)))

    best = min(results, key=lambda k: results[k][2])
    W, full, nsc, per_t = results[best]
    print(f"\nselected by nested: {best} (nested {nsc:.5f}, full-OOF {full:.5f})")
    nz = [(names[m], W[m].round(3).tolist()) for m in range(len(names)) if W[m].sum() > 0.02]
    print(f"non-trivial members ({len(nz)}):")
    for nm, w in sorted(nz, key=lambda x: -sum(x[1]))[:15]:
        print(f"  {nm:22s} " + " ".join(f"{x:.2f}" for x in w))

    test_pred = np.einsum("mkt,mt->kt", P, W)
    ss = pd.read_csv(f"{COMP}/data/sample_submission.csv")
    test = pd.read_csv(f"{COMP}/data/sorted_test.csv", usecols=["PIDN"])
    sub = pd.DataFrame({"PIDN": test["PIDN"]})
    for t, name in enumerate(TARGETS):
        sub[name] = test_pred[:, t]
    sub = sub.set_index("PIDN").loc[ss["PIDN"]].reset_index()
    assert list(sub.columns) == list(ss.columns)
    assert len(sub) == len(ss) and int(sub.isnull().sum().sum()) == 0
    sub.to_csv(f"{COMP}/submission.csv", index=False)
    sub.to_csv(f"{COMP}/submissions/submission_blend_v2.csv", index=False)
    print(f"\nwrote submission.csv {sub.shape}")

    json.dump({"method": best, "full_oof": full, "nested": nsc,
               "per_target": dict(zip(TARGETS, per_t.round(5).tolist())),
               "all_methods": {k: {"full": v[1], "nested": v[2]} for k, v in results.items()},
               "weights": {names[m]: W[m].round(4).tolist()
                           for m in range(len(names)) if W[m].sum() > 0}},
              open(f"{COMP}/scripts/blend2_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
