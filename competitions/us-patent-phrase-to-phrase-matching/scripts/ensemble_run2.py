"""Run-2 ensemble: NNLS weight search over model OOFs + honest leave-fold-out check.

Members: deberta_large_s42, deberta_large_s1337, deberta_base_s42, lgbm.
Weights fitted by NNLS on (z-scored) OOF vs target; honest score = per-fold
NNLS refit on the other 3 GroupKFold-by-anchor folds, evaluated out-of-fold.
Picks the candidate with the best honest score; writes submission.csv.
"""
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold

COMP = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching"
S = f"{COMP}/scripts"

train = pd.read_csv(f"{COMP}/data/train.csv")
test = pd.read_csv(f"{COMP}/data/test.csv")
y = train.score.values

MEMBERS = {
    "large_s42": ("deberta_large_s42_oof.npy", "deberta_large_s42_test.npy"),
    "large_s1337": ("deberta_large_s1337_oof.npy", "deberta_large_s1337_test.npy"),
    "base_s42": ("deberta_base_s42_oof.npy", "deberta_base_s42_test.npy"),
    "lgbm": ("lgbm_oof.npy", "lgbm_test.npy"),
}
oof = {k: np.load(f"{S}/{a}") for k, (a, b) in MEMBERS.items()}
tst = {k: np.load(f"{S}/{b}") for k, (a, b) in MEMBERS.items()}
names = list(MEMBERS)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def nnls_weights(cols, idx):
    X = np.stack([oof[c][idx] for c in cols], axis=1)
    w, _ = nnls(X, y[idx])
    return w / max(w.sum(), 1e-12)


gkf = GroupKFold(n_splits=4)
folds = np.zeros(len(train), dtype=int)
for f, (_, va) in enumerate(gkf.split(train, groups=train.anchor)):
    folds[va] = f


def honest_nnls(cols):
    """Leave-fold-out: fit NNLS on 3 folds, predict held-out fold."""
    pred = np.zeros(len(train))
    for f in range(4):
        tr, va = folds != f, folds == f
        w = nnls_weights(cols, tr)
        pred[va] = np.stack([oof[c][va] for c in cols], axis=1) @ w
    return pearson(y, pred), pred


def fixed_blend(weights):
    p = sum(w * oof[c] for c, w in weights.items())
    t = sum(w * tst[c] for c, w in weights.items())
    return pearson(y, p), t


results = {}

# solo members
for c in names:
    results[f"solo_{c}"] = {"oof": pearson(y, oof[c]), "honest": pearson(y, oof[c]),
                            "test": tst[c], "kind": "solo"}

# fixed candidates (no weight fitting -> oof score already honest)
results["mean_larges"] = dict(zip(["oof", "test"],
                                  fixed_blend({"large_s42": .5, "large_s1337": .5})),
                              kind="fixed")
results["mean_larges"]["honest"] = results["mean_larges"]["oof"]
results["mean_all_deberta"] = dict(zip(["oof", "test"],
                                       fixed_blend({"large_s42": 1/3, "large_s1337": 1/3,
                                                    "base_s42": 1/3})), kind="fixed")
results["mean_all_deberta"]["honest"] = results["mean_all_deberta"]["oof"]

# NNLS candidates
for label, cols in [("nnls_larges", ["large_s42", "large_s1337"]),
                    ("nnls_deberta", ["large_s42", "large_s1337", "base_s42"]),
                    ("nnls_all", names)]:
    w_full = nnls_weights(cols, np.ones(len(train), bool))
    p_full = np.stack([oof[c] for c in cols], axis=1) @ w_full
    t_full = np.stack([tst[c] for c in cols], axis=1) @ w_full
    hon, _ = honest_nnls(cols)
    results[label] = {"oof": pearson(y, p_full), "honest": hon, "test": t_full,
                      "kind": "nnls", "weights": dict(zip(cols, w_full.round(4).tolist()))}

for k, v in results.items():
    extra = f" w={v['weights']}" if "weights" in v else ""
    print(f"{k:20s} oof={v['oof']:.5f} honest={v['honest']:.5f}{extra}")

best = max(results, key=lambda k: results[k]["honest"])
bv = results[best]
print(f"\nCHOSEN: {best} honest={bv['honest']:.5f} oof={bv['oof']:.5f}")

sub = pd.read_csv(f"{COMP}/data/sample_submission.csv")
assert list(sub.id) == list(test.id)
sub["score"] = np.clip(bv["test"], 0, 1)
sub.to_csv(f"{COMP}/submissions/ensemble_run2_{best}.csv", index=False)
sub.to_csv(f"{COMP}/submission.csv", index=False)
print("submission.csv written:", len(sub), "rows")

json.dump({k: {kk: vv for kk, vv in v.items() if kk != "test"}
           for k, v in results.items()} | {"chosen": best},
          open(f"{S}/ensemble_run2_decision.json", "w"), indent=2, default=float)

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)
experiment_log.log_experiment_v2(
    COMP, model=f"ensemble-run2-{best}", metric="pearson_r", direction="maximize",
    score=bv["honest"],
    cv={"scheme": "GroupKFold4_by_anchor", "seed": 42,
        "note": "honest = leave-fold-out NNLS refit; full-OOF score also reported"},
    features=["OOF blend of " + ", ".join(names)],
    submission=f"submissions/ensemble_run2_{best}.csv",
    notes=(f"Candidates: {[ (k, round(v['honest'],5)) for k,v in results.items() ]}. "
           f"Chosen {best}: honest {bv['honest']:.5f}, full-OOF {bv['oof']:.5f}."))
