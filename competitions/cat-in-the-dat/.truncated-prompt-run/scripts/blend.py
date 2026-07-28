"""OOF weight search over available members; writes submission + logs experiment.

Usage: uv run python3 blend.py member1 member2 ... (names matching pred_<name>.npz)
"""
import sys
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/cat-in-the-dat"
OUT = f"{COMP}/processed"
members = sys.argv[1:] or ["lr", "lgb", "cat"]

meta = np.load(f"{OUT}/meta.npz")
y, test_id = meta["y"], meta["test_id"]
oofs, tests = [], []
for m in members:
    d = np.load(f"{OUT}/pred_{m}.npz")
    oofs.append(d["oof"])
    tests.append(d["test"])
    print(f"{m}: solo OOF {roc_auc_score(y, d['oof']):.6f}")
oofs = np.array(oofs)
tests = np.array(tests)

def search(mat):
    grid = np.arange(0, 1.0001, 0.05)
    best_w, best_s = None, -1
    for w in product(grid, repeat=len(members) - 1):
        if sum(w) > 1.0001:
            continue
        ww = np.array(list(w) + [1 - sum(w)])
        s = roc_auc_score(y, ww @ mat)
        if s > best_s:
            best_s, best_w = s, ww
    return best_w, best_s

w_prob, s_prob = search(oofs)
rank_oofs = np.array([rankdata(o) / len(o) for o in oofs])
w_rank, s_rank = search(rank_oofs)
print(f"prob-space: {s_prob:.6f} weights {dict(zip(members, w_prob.round(2)))}")
print(f"rank-space: {s_rank:.6f} weights {dict(zip(members, w_rank.round(2)))}")

if s_rank > s_prob:
    space, w, s = "rank", w_rank, s_rank
    test_mat = np.array([rankdata(t) / len(t) for t in tests])
else:
    space, w, s = "prob", w_prob, s_prob
    test_mat = tests
final_test = w @ test_mat

sub = pd.DataFrame({"id": test_id, "target": final_test})
sub.to_csv(f"{COMP}/submission.csv", index=False)
np.savez(f"{OUT}/pred_blend.npz", oof=w @ (rank_oofs if space == "rank" else oofs),
         test=final_test)
print(f"FINAL blend ({space}) OOF {s:.6f} -> submission.csv")

experiment_log.log_experiment_v2(
    COMP, model=f"blend-{space}({'+'.join(members)})", metric="auc",
    direction="maximize", score=s,
    cv={"strategy": "StratifiedKFold(5, shuffle, seed=42)"},
    ensemble={"members": members, "weights": [round(float(x), 3) for x in w],
              "space": space, "search": "simplex grid 0.05"},
    submission="submission.csv",
    notes=f"prob {s_prob:.6f} vs rank {s_rank:.6f}; solo: "
          + ", ".join(f"{m}={roc_auc_score(y, o):.6f}" for m, o in zip(members, oofs)))
