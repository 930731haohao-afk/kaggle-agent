"""Feature prep: sparse OHE (linear models) + encoded frame (GBDTs).

Outputs in competitions/cat-in-the-dat/data_proc/:
  ohe_train.npz / ohe_test.npz  — CSR one-hot over all 23 cols (train+test union cats)
  gbdt_train.parquet / gbdt_test.parquet — int-encoded frame (ordinals mapped by order)
  y.npy, folds.npy — target and fixed 5-fold StratifiedKFold assignment (seed 42)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

COMP = Path("competitions/cat-in-the-dat")
OUT = COMP / "data_proc"
OUT.mkdir(exist_ok=True)

train = pd.read_csv(COMP / "data/train.csv")
test = pd.read_csv(COMP / "data/test.csv")
y = train["target"].values
feats = [c for c in train.columns if c not in ("id", "target")]

# --- OHE over union categories (no target used; standard for this comp) ---
all_df = pd.concat([train[feats], test[feats]], axis=0)
enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
enc.fit(all_df.astype(str))
X_ohe_tr = enc.transform(train[feats].astype(str)).tocsr()
X_ohe_te = enc.transform(test[feats].astype(str)).tocsr()
sparse.save_npz(OUT / "ohe_train.npz", X_ohe_tr)
sparse.save_npz(OUT / "ohe_test.npz", X_ohe_te)
print(f"OHE: train {X_ohe_tr.shape}, test {X_ohe_te.shape}")

# --- GBDT frame ---
ORD_MAPS = {
    "ord_1": ["Novice", "Contributor", "Expert", "Master", "Grandmaster"],
    "ord_2": ["Freezing", "Cold", "Warm", "Hot", "Boiling Hot", "Lava Hot"],
}
def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["bin_0"] = df["bin_0"].astype(np.int8)
    out["bin_1"] = df["bin_1"].astype(np.int8)
    out["bin_2"] = df["bin_2"].astype(np.int8)
    out["bin_3"] = (df["bin_3"] == "T").astype(np.int8)
    out["bin_4"] = (df["bin_4"] == "Y").astype(np.int8)
    out["ord_0"] = df["ord_0"].astype(np.int8)
    for c, order in ORD_MAPS.items():
        out[c] = df[c].map({v: i for i, v in enumerate(order)}).astype(np.int8)
    for c in ["ord_3", "ord_4", "ord_5"]:
        cats = sorted(all_df[c].unique())
        out[c] = df[c].map({v: i for i, v in enumerate(cats)}).astype(np.int16)
    for c in ["nom_0", "nom_1", "nom_2", "nom_3", "nom_4",
              "nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]:
        cats = pd.Index(sorted(all_df[c].unique()))
        out[c] = pd.Categorical(df[c], categories=cats).codes.astype(np.int32)
    out["day"] = df["day"].astype(np.int8)
    out["month"] = df["month"].astype(np.int8)
    out["day_sin"] = np.sin(2 * np.pi * df["day"] / 7).astype(np.float32)
    out["day_cos"] = np.cos(2 * np.pi * df["day"] / 7).astype(np.float32)
    out["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    return out

gb_tr = build(train[feats])
gb_te = build(test[feats])
gb_tr.to_parquet(OUT / "gbdt_train.parquet")
gb_te.to_parquet(OUT / "gbdt_test.parquet")
print(f"GBDT frame: train {gb_tr.shape}, test {gb_te.shape}")

np.save(OUT / "y.npy", y)
folds = np.zeros(len(y), dtype=np.int8)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
    folds[va] = f
np.save(OUT / "folds.npy", folds)
print("fold sizes:", np.bincount(folds), "pos rate per fold:",
      [round(y[folds == f].mean(), 5) for f in range(5)])
test[["id"]].to_csv(OUT / "test_ids.csv", index=False)
