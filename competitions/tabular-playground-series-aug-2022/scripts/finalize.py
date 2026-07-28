"""Finalize: honest leave-fold-out blend diagnostic + submission from tree champion.

Champion = tree node #52 (kitchen-sink mega-blend). Weights were fit on the full OOF
(greedy/dirichlet+coord-ascent) -> in-sample optimism. Honest check re-fits weights
with the same routine on 4 of 5 product codes and scores the held-out code, pooled.
Submission: weighted raw-prob mix of member test preds, then per-test-group rank-pp
(mirrors the OOF decision metric exactly).
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

COMP = "competitions/tabular-playground-series-aug-2022"
sys.path.insert(0, "tree_search")
CACHE = "tree_search/cache_aug22"

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

rng = np.random.default_rng(42)

train = pd.read_csv(f"{COMP}/data/train.csv")
test = pd.read_csv(f"{COMP}/data/test.csv")
y = train.failure.values
groups = train.product_code.values
te_groups = test.product_code.values

tree = json.load(open(f"{COMP}/experiments_tree_v3.json"))
champ = next(n for n in tree["nodes"] if n["id"] == 52)
res = tree["search_state"]["node_results"]["52"]
members = res["members"]
weights = np.array(res["weights"])
print(f"champion node #52: {len(members)} members, {int((weights > 0).sum())} nonzero weights")

oofs = np.stack([np.load(f"{CACHE}/solo_{m}.npz")["oof"] for m in members])
preds = np.stack([np.load(f"{CACHE}/solo_{m}.npz")["pred"] for m in members])


def rank_pp(p, g):
    out = np.zeros(len(p))
    for c in np.unique(g):
        m = g == c
        out[m] = rankdata(p[m]) / (m.sum() + 1)
    return out


def auc_pp(vec, mask=None):
    if mask is None:
        return roc_auc_score(y, rank_pp(vec, groups))
    return roc_auc_score(y[mask], rank_pp(vec[mask], groups[mask]))


def fit_weights(mask, k=800):
    """Dirichlet draws + coordinate ascent on rank-pp AUC over rows in mask."""
    M = oofs[:, mask]
    yy = y[mask]
    gg = groups[mask]

    def score(w):
        return roc_auc_score(yy, rank_pp(w @ M, gg))

    best_w, best_s = None, -1
    for _ in range(k):
        w = rng.dirichlet(np.ones(len(members)))
        s = score(w)
        if s > best_s:
            best_w, best_s = w, s
    # coordinate ascent
    improved = True
    while improved:
        improved = False
        for i in range(len(members)):
            for delta in (0.05, -0.05, 0.02, -0.02):
                w2 = best_w.copy()
                w2[i] = max(0.0, w2[i] + delta)
                if w2.sum() == 0:
                    continue
                w2 = w2 / w2.sum()
                s2 = score(w2)
                if s2 > best_s + 1e-7:
                    best_w, best_s = w2, s2
                    improved = True
    return best_w, best_s


in_sample = auc_pp(weights @ oofs)
print(f"in-sample blend OOF AUC_pp: {in_sample:.6f} (tree reported {res['auc_pp']})")

# honest leave-fold-out: refit weights without each code, predict that code
holdout = np.zeros(len(y))
for code in np.unique(groups):
    va = groups == code
    w_fit, s_fit = fit_weights(~va)
    holdout[va] = w_fit @ oofs[:, va]
    print(f"  refit w/o {code}: fit-score {s_fit:.6f}, applied to {code}")
honest = roc_auc_score(y, rank_pp(holdout, groups))
print(f"honest leave-fold-out blend AUC_pp: {honest:.6f} (optimism {in_sample - honest:+.6f})")

# root reference for comparison
root_oof = np.load(f"{CACHE}/solo_0.npz")["oof"]
print(f"root LR solo AUC_pp: {auc_pp(root_oof):.6f}")

# ---- submission: in-sample weights (standard), mirror of OOF pipeline ----
test_mix = weights @ preds
test_final = rank_pp(test_mix, te_groups)
sub = pd.DataFrame({"id": test.id.values, "failure": test_final})
sample = pd.read_csv(f"{COMP}/data/sample_submission.csv")
sub = sample[["id"]].merge(sub, on="id", how="left")
assert sub.failure.notna().all() and len(sub) == len(sample)
sub.to_csv(f"{COMP}/submission.csv", index=False)
os.makedirs(f"{COMP}/submissions", exist_ok=True)
sub.to_csv(f"{COMP}/submissions/tree_v3_node52_megablend.csv", index=False)
print(f"submission written: {len(sub)} rows, range [{sub.failure.min():.4f}, {sub.failure.max():.4f}]")

experiment_log.log_experiment_v2(
    COMP, model="tree-search champion: 38-member mega-blend (node #52)",
    metric="roc_auc", direction="maximize", score=in_sample,
    cv={"strategy": "leave-one-group-out GroupKFold (product_code, 5 folds)",
        "honest_leave_fold_out_score": round(honest, 6),
        "optimism": round(in_sample - honest, 6)},
    ensemble={"members": len(members), "nonzero_weights": int((weights > 0).sum()),
              "weight_search": "dirichlet k=800 + coord ascent (harness_v3 eval_blend)"},
    postprocess=["per-group rank normalization"],
    submission="submissions/tree_v3_node52_megablend.csv",
    notes=("harness_v3 tree search, 60 nodes. Root LR 0.591248 -> best solo (LRC+m9) "
           "0.591447 -> pre-burst blend 0.591648 -> mega-blend 0.591955. "
           "OOF-only, not LB-validated."))
print("experiment logged")
