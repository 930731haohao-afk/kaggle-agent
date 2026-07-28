"""Greedy forward-selection blend (with replacement) over cached member OOFs. AUC."""
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

COMP = Path("competitions/cat-in-the-dat")
PROC = COMP / "data_proc"
y = np.load(PROC / "y.npy")

names = sys.argv[1:] if len(sys.argv) > 1 else [p.stem for p in sorted((PROC / "members").glob("*.npz"))]
oofs = {n: np.load(PROC / "members" / f"{n}.npz")["oof"] for n in names}
tests = {n: np.load(PROC / "members" / f"{n}.npz")["test"] for n in names}
for n in names:
    print(f"{n}: {roc_auc_score(y, oofs[n]):.6f}")

# greedy with replacement, 60 steps
sel = []
best_auc = 0.0
cur = np.zeros(len(y))
for step in range(60):
    cand_best, cand_name = best_auc, None
    for n in names:
        s = roc_auc_score(y, (cur * len(sel) + oofs[n]) / (len(sel) + 1))
        if s > cand_best + 1e-7:
            cand_best, cand_name = s, n
    if cand_name is None:
        break
    sel.append(cand_name)
    cur = (cur * (len(sel) - 1) + oofs[cand_name]) / len(sel)
    best_auc = cand_best
weights = {n: sel.count(n) / len(sel) for n in set(sel)}
print(f"\nblend OOF AUC: {best_auc:.6f}  weights: {weights}")

blend_test = sum(w * tests[n] for n, w in weights.items())
blend_oof = sum(w * oofs[n] for n, w in weights.items())
np.savez_compressed(PROC / "members" / "_blend_linear.npz", oof=blend_oof, test=blend_test)
print(f"verify blend oof: {roc_auc_score(y, blend_oof):.6f}")
