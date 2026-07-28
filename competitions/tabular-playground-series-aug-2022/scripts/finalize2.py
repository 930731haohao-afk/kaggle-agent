"""Final submission: unfitted equal-weight LR-family ensemble.

Decision rationale (honest-reporting rules, 07_tree_search.md section 6):
- Mega-blend #52 in-sample 0.591955 but leave-fold-out honest 0.590944 < root solo
  0.591248 -> fitted-weight blends rejected (pure weight-fitting optimism).
- Instead: RULE-BASED membership (every evaluated LR solo whose feature set contains
  the prior-run 5 features), EQUAL weights (nothing fitted on OOF). Variance
  reduction analog of seed bagging; per-fold check: 4/5 folds above root.
"""
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

COMP = "competitions/tabular-playground-series-aug-2022"
CACHE = "tree_search/cache_aug22"
PRIOR5 = {"loading", "measurement_17", "measurement_2", "measurement_3_na", "measurement_5_na"}

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

train = pd.read_csv(f"{COMP}/data/train.csv")
test = pd.read_csv(f"{COMP}/data/test.csv")
y = train.failure.values
groups = train.product_code.values
te_groups = test.product_code.values

tree = json.load(open(f"{COMP}/experiments_tree_v3.json"))
members = sorted(
    n["id"] for n in tree["nodes"]
    if n["status"] == "evaluated" and n["config"].get("kind") == "solo"
    and n["config"].get("model") == "lr"
    and PRIOR5 <= set(n["config"].get("features", {}).get("cols", [])))
print(f"rule-based LR family: {members}")

oofs = np.stack([np.load(f"{CACHE}/solo_{m}.npz")["oof"] for m in members])
preds = np.stack([np.load(f"{CACHE}/solo_{m}.npz")["pred"] for m in members])


def rank_pp(p, g):
    out = np.zeros(len(p))
    for c in np.unique(g):
        m = g == c
        out[m] = rankdata(p[m]) / (m.sum() + 1)
    return out


mix = oofs.mean(axis=0)
score = roc_auc_score(y, rank_pp(mix, groups))
root_oof = np.load(f"{CACHE}/solo_0.npz")["oof"]
root_score = roc_auc_score(y, rank_pp(root_oof, groups))
folds_mix = [roc_auc_score(y[groups == c], mix[groups == c]) for c in "ABCDE"]
folds_root = [roc_auc_score(y[groups == c], root_oof[groups == c]) for c in "ABCDE"]
print(f"equal-weight family OOF AUC_pp: {score:.6f} (root {root_score:.6f})")
print("per-fold delta vs root:", [round(a - b, 4) for a, b in zip(folds_mix, folds_root)])

test_final = rank_pp(preds.mean(axis=0), te_groups)
sub = pd.DataFrame({"id": test.id.values, "failure": test_final})
sample = pd.read_csv(f"{COMP}/data/sample_submission.csv")
sub = sample[["id"]].merge(sub, on="id", how="left")
assert sub.failure.notna().all() and len(sub) == len(sample)
sub.to_csv(f"{COMP}/submission.csv", index=False)
sub.to_csv(f"{COMP}/submissions/lr_family_equalweight.csv", index=False)
print(f"submission.csv written: {len(sub)} rows")

experiment_log.log_experiment_v2(
    COMP, model=f"FINAL: equal-weight LR family ({len(members)} members, rule-based)",
    metric="roc_auc", direction="maximize", score=score,
    cv={"strategy": "leave-one-group-out GroupKFold (product_code, 5 folds)",
        "per_fold_delta_vs_root": [round(a - b, 4) for a, b in zip(folds_mix, folds_root)]},
    ensemble={"members": [int(m) for m in members], "weights": "equal (unfitted)",
              "rule": "every evaluated LR solo containing the prior-run 5 features"},
    postprocess=["per-group rank normalization"],
    submission="submissions/lr_family_equalweight.csv",
    notes=("Chosen over tree champion #52 (in-sample 0.591955) because its honest "
           "leave-fold-out score 0.590944 fell below the root solo 0.591248 -- fitted "
           "blend weights were pure optimism. Equal weights are unfitted; membership "
           "is rule-based, not score-picked. OOF-only, not LB-validated."))
print("experiment logged")
