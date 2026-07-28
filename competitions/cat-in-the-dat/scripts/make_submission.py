"""Final submission from tree-search champion (node #55 kitchen-sink blend).

1. Rebuild champion blend test prediction from cached member preds + node weights.
2. Verify blend OOF AUC digit-for-digit vs the tree's recorded 0.803272.
3. Honest leave-fold-out weight re-fit (weights re-searched on 4 folds, scored on the
   held-out fold, pooled) to quantify in-sample weight-fitting optimism.
4. Write submission.csv (id,target) in sample_submission order.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "tree_search"))
import harness_v3 as hv3  # noqa: E402

COMP = os.path.join(REPO, "competitions", "cat-in-the-dat")
PROC = os.path.join(COMP, "data_proc")
CACHE = os.path.join(REPO, "tree_search", "cache_citd")
CHAMP_NODE = 55

y = np.load(os.path.join(PROC, "y.npy"))
folds = np.load(os.path.join(PROC, "folds.npy"))
tree = json.load(open(os.path.join(COMP, "experiments_tree_v3.json")))
res = tree["search_state"]["node_results"][str(CHAMP_NODE)]
members, weights = res["members"], np.array(res["weights"])

oofs, preds = [], []
for m in members:
    d = np.load(os.path.join(CACHE, f"solo_{m}.npz"))
    oofs.append(d["oof"])
    preds.append(d["pred"])
oofs = np.stack(oofs, axis=1)
preds = np.stack(preds, axis=1)

blend_oof = oofs @ weights
blend_auc = roc_auc_score(y, blend_oof)
print(f"champion blend OOF AUC: {blend_auc:.6f} (tree recorded {res['auc']})")
assert round(blend_auc, 6) == res["auc"], "champion OOF verification failed"

# honest leave-fold-out weight re-fit
def _search_weights(O, target, k=200, seed=42):
    rng = np.random.default_rng(seed)
    n = O.shape[1]
    cands = [np.eye(n)[i] for i in range(n)] + [np.ones(n) / n]
    cands += list(rng.dirichlet(np.ones(n), size=k))
    best_w, best_s = None, -1
    for w in cands:
        s = roc_auc_score(target, O @ w)
        if s > best_s:
            best_s, best_w = s, w
    # coordinate ascent
    step = 0.05
    for _ in range(3):
        improved = False
        for i in range(n):
            for d in (+step, -step):
                w2 = np.clip(best_w + np.eye(n)[i] * d, 0, None)
                if w2.sum() == 0:
                    continue
                w2 = w2 / w2.sum()
                s = roc_auc_score(target, O @ w2)
                if s > best_s + 1e-7:
                    best_s, best_w, improved = s, w2, True
        if not improved:
            step /= 2
    return best_w

held = np.zeros(len(y))
for f in range(5):
    tr, va = folds != f, folds == f
    w_f = _search_weights(oofs[tr], y[tr])
    held[va] = oofs[va] @ w_f
nested_auc = roc_auc_score(y, held)
print(f"leave-fold-out re-fit pooled AUC: {nested_auc:.6f} "
      f"(in-sample-weights optimism: {blend_auc - nested_auc:+.6f})")

blend_test = preds @ weights
sub = pd.read_csv(os.path.join(COMP, "data", "sample_submission.csv"))
test_ids = pd.read_csv(os.path.join(PROC, "test_ids.csv"))
assert (sub["id"].values == test_ids["id"].values).all()
sub["target"] = np.clip(blend_test, 0, 1)
out = os.path.join(COMP, "submission.csv")
sub.to_csv(out, index=False)
print(f"wrote {out}: {sub.shape}, target range [{sub.target.min():.4f}, {sub.target.max():.4f}], "
      f"mean {sub.target.mean():.4f}")
