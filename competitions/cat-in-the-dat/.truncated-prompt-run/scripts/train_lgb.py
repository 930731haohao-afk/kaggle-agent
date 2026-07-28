"""LGB: ords/bins/day/month as ints, low-card noms native categorical,
high-card noms (nom_5..9) fold-safe smoothed target encoding + frequency."""
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/cat-in-the-dat"
OUT = f"{COMP}/processed"

fr = np.load(f"{OUT}/frames.npz", allow_pickle=True)
cols = list(fr["cols"])
df = pd.DataFrame(fr["data"], columns=cols)
meta = np.load(f"{OUT}/meta.npz")
y, folds = meta["y"], meta["folds"]
n_tr = len(y)

HIGH = ["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]
LOWNOM = ["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"]
GLOBAL_MEAN = y.mean()
SMOOTH = 20.0

# frequency encoding on combined (target-free, safe)
for c in HIGH:
    freq = df[c].value_counts()
    df[f"{c}_freq"] = df[c].map(freq).astype(np.float32)

train_df = df.iloc[:n_tr].reset_index(drop=True)
test_df = df.iloc[n_tr:].reset_index(drop=True)

params = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "num_leaves": 63, "min_child_samples": 50, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "num_threads": 10, "deterministic": True, "force_row_wise": True,
    "seed": 42, "verbosity": -1,
}

feat_cols = [c for c in cols if c not in HIGH] + [f"{c}_freq" for c in HIGH] + [f"{c}_te" for c in HIGH]
cat_feats = LOWNOM

t0 = time.time()
oof = np.zeros(n_tr)
test_pred = np.zeros(len(test_df))
fold_scores, best_iters = [], []
for k in range(5):
    tr, va = folds != k, folds == k
    Xtr = train_df[tr].copy()
    Xva = train_df[va].copy()
    Xte = test_df.copy()
    ytr = y[tr]
    # fold-safe smoothed TE fit on training fold only
    for c in HIGH:
        g = pd.DataFrame({"c": Xtr[c].values, "y": ytr}).groupby("c")["y"].agg(["mean", "count"])
        te = (g["mean"] * g["count"] + GLOBAL_MEAN * SMOOTH) / (g["count"] + SMOOTH)
        for X in (Xtr, Xva, Xte):
            X[f"{c}_te"] = X[c].map(te).fillna(GLOBAL_MEAN).astype(np.float32)
    dtr = lgb.Dataset(Xtr[feat_cols], ytr, categorical_feature=cat_feats)
    dva = lgb.Dataset(Xva[feat_cols], y[va], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
    best_iters.append(model.best_iteration)
    oof[va] = model.predict(Xva[feat_cols], num_iteration=model.best_iteration)
    test_pred += model.predict(Xte[feat_cols], num_iteration=model.best_iteration) / 5
    fold_scores.append(roc_auc_score(y[va], oof[va]))
    print(f"fold {k}: auc {fold_scores[-1]:.6f} iter {model.best_iteration}", flush=True)

auc = roc_auc_score(y, oof)
np.savez(f"{OUT}/pred_lgb.npz", oof=oof, test=test_pred)
print(f"LGB OOF AUC {auc:.6f} ({time.time()-t0:.0f}s) iters {best_iters}")

experiment_log.log_experiment_v2(
    COMP, model="LGB-TE", metric="auc", direction="maximize", score=auc,
    cv={"strategy": "StratifiedKFold(5, shuffle, seed=42)",
        "fold_scores": [round(s, 6) for s in fold_scores]},
    features=feat_cols,
    notes="fold-safe smoothed TE (smooth=20) + freq enc for nom_5..9; ords as ints; "
          f"low-card noms native categorical. best_iters {best_iters}. "
          "No imbalance weighting per AUC prior (s3e3/s4e1/s6e2).")
