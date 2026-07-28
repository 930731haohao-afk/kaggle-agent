"""Final blend: honest member pool WITHOUT the Laplacian members (iteration 7
made the nested score worse, 0.44817 -> 0.44903, so it is reverted).

Method choice (plain greedy vs bagged greedy) is made on the NESTED score;
final weights are then fit on the full OOF. Writes submission.csv and logs the
experiment via experiment_log.log_experiment_v2.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import importlib.util
import json
import numpy as np
import pandas as pd

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]
N_FOLDS = 5

_s = importlib.util.spec_from_file_location(
    "blend_honest", f"{COMP}/scripts/blend_honest.py")
bh = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bh) if False else None  # avoid running its main
spec2 = importlib.util.spec_from_file_location("bmod", f"{COMP}/scripts/blend2.py")
bmod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(bmod) if False else None

# re-import the pure functions without executing main
import types
src = open(f"{COMP}/scripts/blend2.py").read().split('def main()')[0]
mod = types.ModuleType("blendfuncs")
exec(compile(src, "blend2_funcs", "exec"), mod.__dict__)
greedy, bagged_greedy, evaluate = mod.greedy, mod.bagged_greedy, mod.evaluate

_e = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_e)
_e.loader.exec_module(experiment_log)


def main():
    d = np.load(f"{COMP}/scripts/members_honest.npz", allow_pickle=True)
    names = [str(x) for x in d["names"] if not str(x).startswith("H_lap_")]
    Y, folds = d["Y"], d["folds"]
    O = np.stack([d[f"oof__{n}"] for n in names])
    P = np.stack([d[f"pred__{n}"] for n in names])
    print(f"{len(names)} members (Laplacian excluded)")

    methods = {"greedy": greedy,
               "bagged_f0.5": lambda o, y, i: bagged_greedy(o, y, i, 25, 0.5)}
    res = {}
    for mn, fn in methods.items():
        W, full, nsc, per_t = evaluate(fn, O, Y, folds)
        res[mn] = (W, full, nsc, per_t)
        print(f"{mn:12s} full-OOF {full:.5f}  nested {nsc:.5f}")
    best = min(res, key=lambda k: res[k][2])
    W, full, nsc, per_t = res[best]
    print(f"selected {best}: nested {nsc:.5f}, full-OOF {full:.5f}")
    print("per-target (full-OOF): " +
          " ".join(f"{t}:{v:.4f}" for t, v in zip(TARGETS, per_t)))

    # per-fold nested scores for the cv record
    nested = np.zeros_like(Y)
    fn = methods[best]
    for f in range(N_FOLDS):
        fit = np.where(folds != f)[0]
        hold = np.where(folds == f)[0]
        for t in range(len(TARGETS)):
            w = fn(O[:, :, t], Y[:, t], fit)
            nested[hold, t] = O[:, hold, t].T @ w
    fold_scores = []
    for f in range(N_FOLDS):
        h = folds == f
        fold_scores.append(float(np.sqrt(((nested[h] - Y[h]) ** 2).mean(0)).mean()))
    print("nested per-fold MCRMSE:", [round(s, 5) for s in fold_scores])

    test_pred = np.einsum("mkt,mt->kt", P, W)
    ss = pd.read_csv(f"{COMP}/data/sample_submission.csv")
    test = pd.read_csv(f"{COMP}/data/sorted_test.csv", usecols=["PIDN"])
    sub = pd.DataFrame({"PIDN": test["PIDN"]})
    for t, name in enumerate(TARGETS):
        sub[name] = test_pred[:, t]
    sub = sub.set_index("PIDN").loc[ss["PIDN"]].reset_index()
    assert list(sub.columns) == list(ss.columns), sub.columns
    assert len(sub) == len(ss) == 727
    assert int(sub.isnull().sum().sum()) == 0
    assert (sub["PIDN"].values == ss["PIDN"].values).all()
    sub.to_csv(f"{COMP}/submission.csv", index=False)
    sub.to_csv(f"{COMP}/submissions/submission_final_honest_blend.csv", index=False)
    print(f"wrote submission.csv {sub.shape}")
    print(sub.describe().T[["mean", "std", "min", "max"]].round(4))

    used = {names[m]: W[m].round(4).tolist() for m in range(len(names)) if W[m].sum() > 0}
    json.dump({"method": best, "full_oof": full, "nested": nsc,
               "per_target_full_oof": dict(zip(TARGETS, per_t.round(5).tolist())),
               "nested_fold_scores": fold_scores, "weights": used},
              open(f"{COMP}/scripts/final_blend_result.json", "w"), indent=1)

    eid = experiment_log.log_experiment_v2(
        COMP,
        model=f"honest-pool {len(names)}-member per-target {best} blend (KRR-RBF / Ridge / SVR / PLS)",
        metric="MCRMSE", direction="minimize", score=nsc,
        cv={"strategy": "GroupKFold-5 on spatial-signature site (seed 42)",
            "folds": N_FOLDS, "fold_scores": fold_scores,
            "note": "score = NESTED (leave-fold-out blend-weight fit); "
                    f"full-OOF weight fit gives {full:.5f} (optimism +{nsc-full:.5f}). "
                    "All member hyperparameters selected by inner CV on outer-train rows only."},
        base_models=[{"name": k, "weights_per_target": v} for k, v in used.items()],
        ensemble={"type": f"per-target {best} forward selection with replacement",
                  "n_members_pool": len(names), "n_members_used": len(used)},
        submission="submission.csv",
        notes="Final. Laplacian-kernel members (iteration 7) reverted: nested 0.44817 -> 0.44903.")
    print(f"logged experiment #{eid}")


if __name__ == "__main__":
    main()
