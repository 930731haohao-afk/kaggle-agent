"""Feature engineering for afsis-soil-properties.

Builds spectral preprocessing variants (chemometrics standard):
  raw   : spectra with CO2 band (2352-2380 cm-1) dropped
  snv   : SNV (row-wise standardization)
  sg1   : Savitzky-Golay deriv-1 (window 25, poly 3) then SNV
  sg2   : Savitzky-Golay deriv-2 (window 25, poly 3) then SNV
  sg1w11: Savitzky-Golay deriv-1 (window 11, poly 2) then SNV
  spatial: 15 spatial covariates + Depth (0/1), z-scored on train

Saves everything to scripts/features.npz plus site-group labels for GroupKFold.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DATA = "competitions/afsis-soil-properties/data"
OUT = "competitions/afsis-soil-properties/scripts/features.npz"
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]
SPATIAL = ["BSAN", "BSAS", "BSAV", "CTI", "ELEV", "EVI", "LSTD", "LSTN",
           "REF1", "REF2", "REF3", "REF7", "RELI", "TMAP", "TMFI"]

train = pd.read_csv(f"{DATA}/training.csv")
test = pd.read_csv(f"{DATA}/sorted_test.csv")

spec_cols = [c for c in train.columns if c.startswith("m")]
wn = np.array([float(c[1:]) for c in spec_cols])
keep = ~((wn >= 2352) & (wn <= 2380))  # drop CO2 band
spec_keep = [c for c, k in zip(spec_cols, keep) if k]
print(f"spectral kept {len(spec_keep)}/{len(spec_cols)}")

Xtr_raw = train[spec_keep].to_numpy(dtype=np.float64)
Xte_raw = test[spec_keep].to_numpy(dtype=np.float64)


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.maximum(sd, 1e-12)


def sg(X: np.ndarray, window: int, poly: int, deriv: int) -> np.ndarray:
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=deriv, axis=1)


variants = {
    "raw": (Xtr_raw, Xte_raw),
    "snv": (snv(Xtr_raw), snv(Xte_raw)),
    "sg1": (snv(sg(Xtr_raw, 25, 3, 1)), snv(sg(Xte_raw, 25, 3, 1))),
    "sg2": (snv(sg(Xtr_raw, 25, 3, 2)), snv(sg(Xte_raw, 25, 3, 2))),
    "sg1w11": (snv(sg(Xtr_raw, 11, 2, 1)), snv(sg(Xte_raw, 11, 2, 1))),
}

# spatial + Depth
dep_tr = (train["Depth"] == "Topsoil").astype(float).to_numpy()[:, None]
dep_te = (test["Depth"] == "Topsoil").astype(float).to_numpy()[:, None]
Str = train[SPATIAL].to_numpy(dtype=np.float64)
Ste = test[SPATIAL].to_numpy(dtype=np.float64)
mu, sd = Str.mean(0), Str.std(0)
variants["spatial"] = (np.hstack([(Str - mu) / sd, dep_tr]),
                       np.hstack([(Ste - mu) / sd, dep_te]))

# site groups from spatial signature
sig = train[SPATIAL].round(6).apply(tuple, axis=1)
groups = sig.astype("category").cat.codes.to_numpy()
print(f"groups: {len(np.unique(groups))} unique sites")

save = {"y": train[TARGETS].to_numpy(dtype=np.float64), "groups": groups}
for name, (a, b) in variants.items():
    save[f"tr_{name}"] = a.astype(np.float32)
    save[f"te_{name}"] = b.astype(np.float32)
    print(f"{name}: train {a.shape}, test {b.shape}, "
          f"nan {np.isnan(a).sum() + np.isnan(b).sum()}")

np.savez_compressed(OUT, **save)
train[["PIDN"]].to_csv("competitions/afsis-soil-properties/scripts/train_ids.csv", index=False)
test[["PIDN"]].to_csv("competitions/afsis-soil-properties/scripts/test_ids.csv", index=False)
print("saved", OUT)
