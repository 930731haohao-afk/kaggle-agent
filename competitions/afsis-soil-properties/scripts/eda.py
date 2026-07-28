"""EDA for afsis-soil-properties.

Verifies known structure: 3578 MIR spectral cols, 15 spatial covariates, Depth,
5 targets (Ca, P, pH, SOC, Sand). Confirms hidden site groups via spatial signature.
"""
import numpy as np
import pandas as pd

DATA = "competitions/afsis-soil-properties/data"
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]
SPATIAL = ["BSAN", "BSAS", "BSAV", "CTI", "ELEV", "EVI", "LSTD", "LSTN",
           "REF1", "REF2", "REF3", "REF7", "RELI", "TMAP", "TMFI"]

train = pd.read_csv(f"{DATA}/training.csv")
test = pd.read_csv(f"{DATA}/sorted_test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")

print(f"train {train.shape}, test {test.shape}, sub {sub.shape}")
print("sub cols:", list(sub.columns))

spec_cols = [c for c in train.columns if c.startswith("m")]
print(f"spectral cols: {len(spec_cols)}, first {spec_cols[0]}, last {spec_cols[-1]}")

print("\nDepth:", train["Depth"].value_counts().to_dict(), "| test:", test["Depth"].value_counts().to_dict())
print("nulls train:", int(train.isnull().sum().sum()), "test:", int(test.isnull().sum().sum()))

print("\nTargets:")
print(train[TARGETS].describe().T[["mean", "std", "min", "max"]])
print("skew:", train[TARGETS].skew().round(2).to_dict())

# hidden site groups via spatial signature
sig = train[SPATIAL].round(6).apply(tuple, axis=1)
print(f"\nunique spatial signatures train: {sig.nunique()} / {len(train)}")
counts = sig.value_counts()
print("rows-per-site distribution:", counts.value_counts().sort_index().to_dict())
sig_test = test[SPATIAL].round(6).apply(tuple, axis=1)
print(f"unique spatial signatures test: {sig_test.nunique()} / {len(test)}")
overlap = set(sig) & set(sig_test)
print(f"train/test site overlap: {len(overlap)}")

# CO2 band region (known chemometrics artifact ~2350-2380 cm-1)
wn = np.array([float(c[1:]) for c in spec_cols])
co2 = [(c, w) for c, w in zip(spec_cols, wn) if 2352 <= w <= 2380]
print(f"\nCO2 band cols (2352-2380): {len(co2)}")

# train/test spectral consistency: mean abs diff of column means
mtr = train[spec_cols].mean()
mte = test[spec_cols].mean()
print("max |mean diff| spectral:", float((mtr - mte).abs().max()))
mtr2 = train[SPATIAL].describe().T[["mean", "std"]]
mte2 = test[SPATIAL].describe().T[["mean", "std"]]
print("\nspatial mean train vs test:")
print(pd.concat([mtr2, mte2], axis=1, keys=["train", "test"]).round(3))

# quick target-target corr
print("\ntarget corr:\n", train[TARGETS].corr().round(2))
