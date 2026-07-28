"""EDA for afsis-soil-properties (MCRMSE, 5 targets, p>>n spectral data)."""
import numpy as np
import pandas as pd

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]

train = pd.read_csv(f"{COMP}/data/training.csv")
test = pd.read_csv(f"{COMP}/data/sorted_test.csv")
print(f"train {train.shape}, test {test.shape}")

spec_cols = [c for c in train.columns if c.startswith("m")]
spatial = ["BSAN","BSAS","BSAV","CTI","ELEV","EVI","LSTD","LSTN","REF1","REF2","REF3","REF7","RELI","TMAP","TMFI"]
print(f"spectral cols: {len(spec_cols)}, spatial: {len(spatial)}")

# targets
print("\n== targets ==")
print(train[TARGETS].describe().T[["mean","std","min","25%","50%","75%","max"]])
print("skew:", train[TARGETS].skew().round(2).to_dict())

# Depth
print("\n== Depth ==")
print(train["Depth"].value_counts().to_dict(), "| test:", test["Depth"].value_counts().to_dict())
print(train.groupby("Depth")[TARGETS].mean().round(3))

# nulls / dupes
print("\nnulls train:", int(train.isnull().sum().sum()), "test:", int(test.isnull().sum().sum()))
print("dup spectra rows:", int(train.duplicated(subset=spec_cols).sum()))

# location pairing: same spatial features => same site (Topsoil/Subsoil pair)
site_key = train[spatial].round(6).apply(tuple, axis=1)
n_sites = site_key.nunique()
print(f"\nunique spatial-signature sites in train: {n_sites} (rows {len(train)})")
site_counts = site_key.value_counts()
print("site size distribution:", site_counts.value_counts().to_dict())

# CO2 band m2379.76 - m2352.76 known artifact
co2 = [c for c in spec_cols if 2352.0 <= float(c[1:]) <= 2380.0]
print(f"\nCO2 band cols (2352-2380): {len(co2)}")

# train/test spectra consistency
tr_mean = train[spec_cols].values.mean(axis=0)
te_mean = test[spec_cols].values.mean(axis=0)
print("spectra mean abs diff train vs test:", np.abs(tr_mean - te_mean).mean().round(4))
print("spectra global std train:", train[spec_cols].values.std().round(4), "test:", test[spec_cols].values.std().round(4))

# spatial train/test shift
print("\nspatial mean diff (train-test)/std:")
for c in spatial:
    d = (train[c].mean() - test[c].mean()) / (train[c].std() + 1e-9)
    if abs(d) > 0.15:
        print(f"  {c}: {d:+.3f}")

# quick target-corr with a few PCA-ish summaries: corr of targets with spectra band means
bands = np.array_split(np.arange(len(spec_cols)), 8)
print("\ncorr(band-mean, target):")
bm = np.column_stack([train[spec_cols].values[:, b].mean(axis=1) for b in bands])
for i, t in enumerate(TARGETS):
    cs = [np.corrcoef(bm[:, j], train[t])[0, 1] for j in range(8)]
    print(f"  {t}: " + " ".join(f"{c:+.2f}" for c in cs))
