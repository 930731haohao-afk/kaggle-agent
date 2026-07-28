"""CatBoost native categoricals — all 23 cols as raw strings (per s3e3 native-cat lesson)."""
import numpy as np
import pandas as pd
import time
from catboost import CatBoostClassifier, Pool
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
feats = [c for c in train.columns if c not in ("id", "target")]
Xtr_all = train[feats].astype(str)
Xte = test[feats].astype(str)
cat_idx = list(range(len(feats)))
te_pool = Pool(Xte, cat_features=cat_idx)

t0 = time.time()
oof = np.zeros(len(y))
test_pred = np.zeros(len(test))
fold_scores, best_iters = [], []
for k in range(5):
    tr, va = folds != k, folds == k
    model = CatBoostClassifier(
        iterations=5000, learning_rate=0.1, depth=6, l2_leaf_reg=3.0,
        eval_metric="AUC", random_seed=42, od_type="Iter", od_wait=200,
        thread_count=10, allow_writing_files=False, verbose=False)
    model.fit(Pool(Xtr_all[tr], y[tr], cat_features=cat_idx),
              eval_set=Pool(Xtr_all[va], y[va], cat_features=cat_idx))
    best_iters.append(model.get_best_iteration())
    oof[va] = model.predict_proba(Xtr_all[va])[:, 1]
    test_pred += model.predict_proba(te_pool)[:, 1] / 5
    fold_scores.append(roc_auc_score(y[va], oof[va]))
    print(f"fold {k}: auc {fold_scores[-1]:.6f} iter {best_iters[-1]} "
          f"({time.time()-t0:.0f}s)", flush=True)

auc = roc_auc_score(y, oof)
np.savez(f"{OUT}/pred_cat.npz", oof=oof, test=test_pred)
print(f"CAT OOF AUC {auc:.6f} ({time.time()-t0:.0f}s)")

experiment_log.log_experiment_v2(
    COMP, model="CatBoost-native", metric="auc", direction="maximize", score=auc,
    cv={"strategy": "StratifiedKFold(5, shuffle, seed=42)",
        "fold_scores": [round(s, 6) for s in fold_scores]},
    features=[f"all 23 raw string cols as cat_features"],
    notes=f"depth 6, lr 0.1, od_wait 200. best_iters {best_iters}. "
          "No class weighting per AUC prior.")
