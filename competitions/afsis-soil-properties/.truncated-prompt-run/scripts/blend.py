"""Blend members: per-target greedy forward selection (with replacement) +
optional shrink calibration. Reports nested (leave-fold-out) estimate of the
greedy selection's optimism. Writes submission.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import json
import numpy as np
import pandas as pd

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]
N_FOLDS = 5
N_STEPS = 20


def greedy(O, y, n_steps=N_STEPS, idx=None):
    """O: (M, n) member OOF for one target. Returns weight vector (M,)."""
    M = O.shape[0]
    if idx is None:
        idx = np.arange(O.shape[1])
    picks = []
    cur = np.zeros(len(idx))
    best_prev = np.inf
    for _ in range(n_steps):
        cands = (cur[None] * len(picks) + O[:, idx]) / (len(picks) + 1)
        sc = np.sqrt(((cands - y[idx][None]) ** 2).mean(1))
        m = int(sc.argmin())
        if sc[m] >= best_prev - 1e-7:
            break
        picks.append(m)
        cur = (cur * (len(picks) - 1) + O[m, idx]) / len(picks)
        best_prev = sc[m]
    w = np.zeros(M)
    for m in picks:
        w[m] += 1.0 / max(len(picks), 1)
    return w


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def main():
    d = np.load(f"{COMP}/scripts/members.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    Y, folds = d["Y"], d["folds"]
    O = np.stack([d[f"oof__{n}"] for n in names])   # (M, n, T)
    P = np.stack([d[f"pred__{n}"] for n in names])  # (M, ntest, T)
    M, n, T = O.shape
    print(f"{M} members, {n} train rows, {P.shape[1]} test rows")

    W = np.zeros((M, T))
    for t in range(T):
        W[:, t] = greedy(O[:, :, t], Y[:, t])
    blend_oof = np.einsum("mnt,mt->nt", O, W)
    per_t = np.sqrt(((blend_oof - Y) ** 2).mean(0))
    score = float(per_t.mean())
    print(f"\ngreedy blend MCRMSE {score:.5f}  " +
          " ".join(f"{t}:{v:.4f}" for t, v in zip(TARGETS, per_t)))
    print("weights:")
    for m in range(M):
        if W[m].sum() > 0:
            print(f"  {names[m]:18s} " + " ".join(f"{W[m, t]:.2f}" for t in range(T)))

    # shrink calibration: p' = mean + b*(p - mean), b chosen per target on OOF
    bs = np.arange(0.5, 1.31, 0.02)
    shrink = np.ones(T)
    for t in range(T):
        mu = Y[:, t].mean()
        scores = [rmse(mu + b * (blend_oof[:, t] - mu), Y[:, t]) for b in bs]
        shrink[t] = bs[int(np.argmin(scores))]
    shrunk = np.column_stack([Y[:, t].mean() + shrink[t] * (blend_oof[:, t] - Y[:, t].mean())
                              for t in range(T)])
    per_ts = np.sqrt(((shrunk - Y) ** 2).mean(0))
    print(f"\nshrink b = {dict(zip(TARGETS, shrink.round(2)))}")
    print(f"shrunk blend MCRMSE {per_ts.mean():.5f}  " +
          " ".join(f"{t}:{v:.4f}" for t, v in zip(TARGETS, per_ts)))
    use_shrink = per_ts.mean() < score - 1e-6

    # nested estimate of greedy-selection optimism
    nested = np.zeros_like(blend_oof)
    for f in range(N_FOLDS):
        fit = np.where(folds != f)[0]
        hold = np.where(folds == f)[0]
        for t in range(T):
            w = greedy(O[:, :, t], Y[:, t], idx=fit)
            nested[hold, t] = O[:, hold, t].T @ w
    per_tn = np.sqrt(((nested - Y) ** 2).mean(0))
    print(f"\nnested (leave-fold-out weight fit) MCRMSE {per_tn.mean():.5f}  "
          f"optimism {per_tn.mean() - score:+.5f}")

    # test predictions
    test_pred = np.einsum("mkt,mt->kt", P, W)
    if use_shrink:
        for t in range(T):
            mu = Y[:, t].mean()
            test_pred[:, t] = mu + shrink[t] * (test_pred[:, t] - mu)
        final = float(per_ts.mean())
    else:
        final = score

    test = pd.read_csv(f"{COMP}/data/sorted_test.csv", usecols=["PIDN"])
    ss = pd.read_csv(f"{COMP}/data/sample_submission.csv")
    sub = pd.DataFrame({"PIDN": test["PIDN"]})
    for t, name in enumerate(TARGETS):
        sub[name] = test_pred[:, t]
    sub = sub.set_index("PIDN").loc[ss["PIDN"]].reset_index()
    assert list(sub.columns) == list(ss.columns), (sub.columns, ss.columns)
    assert len(sub) == len(ss) and sub.isnull().sum().sum() == 0
    sub.to_csv(f"{COMP}/submission.csv", index=False)
    sub.to_csv(f"{COMP}/submissions/submission_greedy_blend.csv", index=False)
    print(f"\nwrote submission.csv {sub.shape}; final CV MCRMSE {final:.5f}")

    json.dump({"names": names, "weights": W.tolist(), "score": score,
               "shrink": shrink.tolist(), "shrunk_score": float(per_ts.mean()),
               "used_shrink": bool(use_shrink), "nested": float(per_tn.mean()),
               "per_target": dict(zip(TARGETS, per_t.round(5).tolist()))},
              open(f"{COMP}/scripts/blend_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
