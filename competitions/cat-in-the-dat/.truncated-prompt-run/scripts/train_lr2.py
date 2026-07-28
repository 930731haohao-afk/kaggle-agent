"""LR variant for blend diversity: OHE low/mid-card cols + fold-safe smoothed TE
+ log-freq numerics for high-card noms (standardized)."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/cat-in-the-dat"
OUT = f"{COMP}/processed"
DATA = f"{COMP}/data"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
meta = np.load(f"{OUT}/meta.npz")
y, folds = meta["y"], meta["folds"]
n_tr = len(train)

HIGH = ["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]
LOW = [c for c in train.columns if c not in ("id", "target") and c not in HIGH]
GLOBAL_MEAN = y.mean()
SMOOTH = 20.0

combined = pd.concat([train, test], ignore_index=True)
ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
X_low = ohe.fit_transform(combined[LOW].astype(str))
X_low_tr, X_low_te = X_low[:n_tr].tocsr(), X_low[n_tr:].tocsr()

logfreq = {}
for c in HIGH:
    freq = combined[c].value_counts()
    logfreq[c] = np.log1p(combined[c].map(freq).values.astype(np.float64))

results = {}
for C in [0.1, 0.3]:
    oof = np.zeros(n_tr)
    test_pred = np.zeros(len(test))
    fold_scores = []
    for k in range(5):
        tr, va = folds != k, folds == k
        num_tr, num_va, num_te = [], [], []
        for c in HIGH:
            g = pd.DataFrame({"c": train[c][tr].values, "y": y[tr]}).groupby("c")["y"].agg(["mean", "count"])
            te = (g["mean"] * g["count"] + GLOBAL_MEAN * SMOOTH) / (g["count"] + SMOOTH)
            col_tr = train[c][tr].map(te).fillna(GLOBAL_MEAN).values
            col_va = train[c][va].map(te).fillna(GLOBAL_MEAN).values
            col_te = test[c].map(te).fillna(GLOBAL_MEAN).values
            mu, sd = col_tr.mean(), col_tr.std() + 1e-9
            num_tr += [(col_tr - mu) / sd]
            num_va += [(col_va - mu) / sd]
            num_te += [(col_te - mu) / sd]
            lf = logfreq[c]
            lf_tr, lf_va, lf_te = lf[:n_tr][tr], lf[:n_tr][va], lf[n_tr:]
            mu2, sd2 = lf_tr.mean(), lf_tr.std() + 1e-9
            num_tr += [(lf_tr - mu2) / sd2]
            num_va += [(lf_va - mu2) / sd2]
            num_te += [(lf_te - mu2) / sd2]
        Xtr = sparse.hstack([X_low_tr[tr], np.column_stack(num_tr)], format="csr")
        Xva = sparse.hstack([X_low_tr[va], np.column_stack(num_va)], format="csr")
        Xte = sparse.hstack([X_low_te, np.column_stack(num_te)], format="csr")
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42)
        clf.fit(Xtr, y[tr])
        oof[va] = clf.predict_proba(Xva)[:, 1]
        test_pred += clf.predict_proba(Xte)[:, 1] / 5
        fold_scores.append(roc_auc_score(y[va], oof[va]))
    auc = roc_auc_score(y, oof)
    results[C] = (auc, fold_scores, oof, test_pred)
    print(f"C={C}: OOF AUC {auc:.6f}", flush=True)

best_C = max(results, key=lambda c: results[c][0])
auc, fold_scores, oof, test_pred = results[best_C]
np.savez(f"{OUT}/pred_lr2.npz", oof=oof, test=test_pred)
print(f"LR2 BEST C={best_C} OOF {auc:.6f}")

experiment_log.log_experiment_v2(
    COMP, model=f"LR2-TE(C={best_C})", metric="auc", direction="maximize", score=auc,
    cv={"strategy": "StratifiedKFold(5, shuffle, seed=42)",
        "fold_scores": [round(s, 6) for s in fold_scores]},
    features=["OHE 18 low/mid-card cols", "fold-safe smoothed TE + log-freq for nom_5..9 (standardized)"],
    notes="diversity member for blend; TE fit per training fold only")
