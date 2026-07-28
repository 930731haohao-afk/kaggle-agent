"""LR3: full OHE + standardized ordinal ints + day/month sin-cos appended."""
import numpy as np
import pandas as pd
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

Xtr_ohe = sparse.load_npz(f"{OUT}/ohe_train.npz")
Xte_ohe = sparse.load_npz(f"{OUT}/ohe_test.npz")
meta = np.load(f"{OUT}/meta.npz")
y, folds = meta["y"], meta["folds"]
fr = np.load(f"{OUT}/frames.npz", allow_pickle=True)
df = pd.DataFrame(fr["data"], columns=list(fr["cols"]))
n_tr = len(y)

num = pd.DataFrame(index=df.index)
for c in ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]:
    num[c] = df[c].astype(np.float64)
num["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
num["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
num["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
num["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
num = (num - num.mean()) / num.std()

Xtr = sparse.hstack([Xtr_ohe, num.values[:n_tr]], format="csr")
Xte = sparse.hstack([Xte_ohe, num.values[n_tr:]], format="csr")

results = {}
for C in [0.1, 0.12, 0.15]:
    oof = np.zeros(n_tr)
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

best_C = max(results, key=lambda c: results[c][0])
auc, fold_scores, oof, test_pred = results[best_C]
np.savez(f"{OUT}/pred_lr3.npz", oof=oof, test=test_pred)
print(f"LR3 BEST C={best_C} OOF {auc:.6f} (vs LR 0.803135)")

experiment_log.log_experiment_v2(
    COMP, model=f"LR3-OHE+ordnum(C={best_C})", metric="auc", direction="maximize",
    score=auc,
    cv={"strategy": "StratifiedKFold(5, shuffle, seed=42)",
        "fold_scores": [round(s, 6) for s in fold_scores]},
    features=["OHE 16552", "standardized ord_0..5 ints", "day/month sin-cos"],
    notes="tests whether numeric ordinal trend adds signal beyond OHE under L2 shrinkage")
