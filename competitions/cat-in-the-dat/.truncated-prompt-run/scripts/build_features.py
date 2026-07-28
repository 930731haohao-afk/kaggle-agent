"""Build feature sets for cat-in-the-dat.

Outputs (competitions/cat-in-the-dat/processed/):
- ohe.npz            sparse one-hot of all 23 cols, fit on train+test combined
- frames.parquet     label-encoded int frame (train+test stacked) for LGB
- meta.npz           y, train/test ids, fold assignment (StratifiedKFold 5, seed 42)
Raw strings for CatBoost are re-read from csv in its own script.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

DATA = "competitions/cat-in-the-dat/data"
OUT = "competitions/cat-in-the-dat/processed"
import os
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train["target"].values.astype(np.int8)
feats = [c for c in train.columns if c not in ("id", "target")]

combined = pd.concat([train[feats], test[feats]], axis=0, ignore_index=True)
combined = combined.astype(str)

# --- sparse OHE ---
ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
X_all = ohe.fit_transform(combined)
n_tr = len(train)
sparse.save_npz(f"{OUT}/ohe_train.npz", X_all[:n_tr].tocsr())
sparse.save_npz(f"{OUT}/ohe_test.npz", X_all[n_tr:].tocsr())
print(f"OHE dims: {X_all.shape[1]}")

# --- label-encoded ints (factorize on combined) + ordinal maps ---
enc = pd.DataFrame(index=combined.index)
for c in feats:
    enc[c] = pd.factorize(combined[c], sort=True)[0].astype(np.int32)

# proper ordinal orderings override lexicographic where needed
ord_1_map = {"Novice": 0, "Contributor": 1, "Expert": 2, "Master": 3, "Grandmaster": 4}
ord_2_map = {"Freezing": 0, "Cold": 1, "Warm": 2, "Hot": 3, "Boiling Hot": 4, "Lava Hot": 5}
enc["ord_1"] = combined["ord_1"].map(ord_1_map).astype(np.int32)
enc["ord_2"] = combined["ord_2"].map(ord_2_map).astype(np.int32)
# ord_3 (a-o), ord_4 (A-Z), ord_5 (2-char, lexicographic is the standard treatment): sort=True already fine

np.savez_compressed(f"{OUT}/frames.npz", data=enc.values, cols=np.array(feats))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.zeros(n_tr, dtype=np.int8)
for k, (_, va) in enumerate(skf.split(np.zeros(n_tr), y)):
    folds[va] = k
np.savez(f"{OUT}/meta.npz", y=y, folds=folds,
         train_id=train["id"].values, test_id=test["id"].values)
print("fold sizes:", np.bincount(folds), "pos rate per fold:",
      [round(y[folds == k].mean(), 5) for k in range(5)])
