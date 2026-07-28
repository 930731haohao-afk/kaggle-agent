"""Ridge v3: per-holiday-name offset dummies (replaces pooled holiday flags).

Base design = v2 E_all (fourier6 x product, eoy x country dummies) but holiday
encoding = one dummy per (holiday_name, day_offset in -2..+2), pooled across
countries (flag only fires in countries observing that holiday).
"""
import importlib.util
import json

import numpy as np
import pandas as pd
import holidays as hol
from sklearn.linear_model import Ridge

COMP = "competitions/tabular-playground-series-jan-2022"
VAL_YEARS = [2016, 2017, 2018]
OFFSETS = range(-3, 8)

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

tr = pd.read_parquet(f"{COMP}/train_feat.parquet")

CALS = {
    "Finland": hol.Finland(years=range(2014, 2021)),
    "Norway": hol.Norway(years=range(2014, 2021)),
    "Sweden": hol.Sweden(years=range(2014, 2021)),
}
# map (country, date) -> holiday name; build offset lookup
hol_maps = {}
for c, cal in CALS.items():
    m = {}
    for d, name in cal.items():
        nm = name.split(";")[0].strip()
        for o in OFFSETS:
            m[(pd.Timestamp(d) + pd.Timedelta(days=o), o)] = nm
    hol_maps[c] = m

ALL_NAMES = sorted({nm.split(";")[0].strip() for cal in CALS.values() for nm in cal.values()})
print(f"{len(ALL_NAMES)} distinct holiday names")


def smape(a, f):
    return 200.0 * np.mean(np.abs(f - a) / (np.abs(a) + np.abs(f)))


def ridge_design(df):
    cols = {}
    base = pd.get_dummies(df[["country", "store", "product", "dow"]].astype(str))
    for v in range(1, 19):
        for c in df.country.cat.categories:
            cols[f"eoy_{v}_{c}"] = ((df.eoy_idx == v) & (df.country == c)).astype(int).values
    for k in range(1, 7):
        s = np.sin(2 * np.pi * k * df.doy / 365.0)
        co = np.cos(2 * np.pi * k * df.doy / 365.0)
        for p in df["product"].cat.categories:
            m = (df["product"] == p).astype(int).values
            cols[f"sin{k}_{p}"] = s.values * m
            cols[f"cos{k}_{p}"] = co.values * m
    # holiday name x offset dummies
    for nm in ALL_NAMES:
        for o in OFFSETS:
            cols[f"hol_{nm}_{o}"] = np.zeros(len(df))
    for c, hmap in hol_maps.items():
        mask = (df.country == c).values
        idx = np.where(mask)[0]
        for i, dt in zip(idx, df.date.iloc[idx]):
            for o in OFFSETS:
                nm = hmap.get((dt, o))
                if nm is not None:
                    cols[f"hol_{nm}_{o}"][i] = 1.0
    cols["log_gdp"] = df.log_gdp.values
    cols["year_c"] = (df.year - 2015).values
    return pd.concat([base, pd.DataFrame(cols, index=df.index)], axis=1).astype(float)


oof, actual, scores = {}, {}, {}
for y in VAL_YEARS:
    dtr, dva = tr[tr.year < y], tr[tr.year == y].reset_index(drop=True)
    Xtr, Xva = ridge_design(dtr), ridge_design(dva)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    r = Ridge(alpha=1.0)
    r.fit(Xtr, np.log1p(dtr.num_sold))
    p = np.clip(np.expm1(r.predict(Xva)), 1, None)
    oof[y] = p
    actual[y] = dva.num_sold.values
    scores[y] = smape(actual[y], p)
    print(f"ridge_v3 {y}: {scores[y]:.4f}")
print(f"mean: {np.mean(list(scores.values())):.4f}")

np.savez(f"{COMP}/scripts/oof_ridge_v3b.npz", **{str(y): oof[y] for y in VAL_YEARS})
experiment_log.log_experiment_v2(
    COMP, model="ridge_v3b_off37", metric="SMAPE", direction="minimize",
    score=scores[2018],
    cv={"strategy": "expanding-year (val 2016/2017/2018), primary fold 2018",
        "scores": [scores[y] for y in VAL_YEARS],
        "mean": float(np.mean(list(scores.values())))},
    notes="train_v3b.py: per-holiday-name x offset(-3..7) dummies, fourier6, eoy x country")
print("logged")
