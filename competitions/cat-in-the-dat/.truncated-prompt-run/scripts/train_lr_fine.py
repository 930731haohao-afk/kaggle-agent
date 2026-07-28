"""Finer C grid around 0.1; overwrite pred_lr.npz if better."""
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/cat-in-the-dat"
OUT = f"{COMP}/processed"
Xtr = sparse.load_npz(f"{OUT}/ohe_train.npz")
Xte = sparse.load_npz(f"{OUT}/ohe_test.npz")
meta = np.load(f"{OUT}/meta.npz")
y, folds = meta["y"], meta["folds"]

best = (0.1, 0.803081)  # from coarse grid
results = {}
for C in [0.08, 0.12, 0.15]:
    oof = np.zeros(len(y))
    test_pred = np.zeros(Xte.shape[0])
    fold_scores = []
    for k in range(5):
        tr, va = folds != k, folds == k
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42)
        clf.fit(Xtr[tr], y[tr])
        oof[va] = clf.predict_proba(Xtr[va])[:, 1]
        test_pred += clf.predict_proba(Xte)[:, 1] / 5
        fold_scores.append(roc_auc_score(y[va], oof[va]))
    auc = roc_auc_score(y, oof)
    results[C] = (auc, fold_scores, oof, test_pred)
    print(f"C={C}: OOF AUC {auc:.6f}", flush=True)

best_fine = max(results, key=lambda c: results[c][0])
if results[best_fine][0] > best[1]:
    auc, fold_scores, oof, test_pred = results[best_fine]
    np.savez(f"{OUT}/pred_lr.npz", oof=oof, test=test_pred)
    experiment_log.log_experiment_v2(
        COMP, model=f"LogisticRegression-OHE(C={best_fine})", metric="auc",
        direction="maximize", score=auc,
        cv={"strategy": "StratifiedKFold(5, shuffle, seed=42)",
            "fold_scores": [round(s, 6) for s in fold_scores]},
        features=["one-hot all 23 cols, 16552 dims"],
        notes=f"fine C grid: " + ", ".join(f"C={c}:{results[c][0]:.6f}" for c in results)
              + "; beats coarse best C=0.1 (0.803081), pred_lr.npz updated")
    print(f"UPDATED pred_lr.npz with C={best_fine} OOF {auc:.6f}")
else:
    print(f"kept C=0.1 (fine best C={best_fine} {results[best_fine][0]:.6f} not better)")
