"""LR on sparse OHE — known-strong family for all-categorical AUC. C grid on same folds."""
import numpy as np
import time
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

results = {}
for C in [0.05, 0.1, 0.2]:
    t0 = time.time()
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
    print(f"C={C}: OOF AUC {auc:.6f} folds {[round(s,5) for s in fold_scores]} "
          f"({time.time()-t0:.0f}s)", flush=True)

best_C = max(results, key=lambda c: results[c][0])
auc, fold_scores, oof, test_pred = results[best_C]
np.savez(f"{OUT}/pred_lr.npz", oof=oof, test=test_pred)
print(f"BEST C={best_C} OOF {auc:.6f}")

experiment_log.log_experiment_v2(
    COMP, model=f"LogisticRegression-OHE(C={best_C})", metric="auc", direction="maximize",
    score=auc,
    cv={"strategy": "StratifiedKFold(5, shuffle, seed=42)",
        "fold_scores": [round(s, 6) for s in fold_scores]},
    features=["one-hot all 23 cols, combined train+test fit, 16552 dims"],
    notes=f"C grid {{0.05,0.1,0.2}} on same folds: "
          + ", ".join(f"C={c}:{results[c][0]:.6f}" for c in results))
