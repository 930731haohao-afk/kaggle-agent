"""Ridge design iteration for tabular-playground-series-jan-2022.

Variants (cumulative candidates evaluated on expanding-year folds):
  A base          — v1 design (Fourier k4 x product, pooled holiday, eoy dummies)
  B +hol_country  — holiday and holiday_win2 interacted with country
  C +fourier6     — Fourier order 6 x product
  D +eoy_country  — eoy dummies x country
  E all           — B + C + D
Plus SMAPE shrink-factor grid on best variant + blend refresh.
"""
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

COMP = "competitions/tabular-playground-series-jan-2022"
VAL_YEARS = [2016, 2017, 2018]

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

tr = pd.read_parquet(f"{COMP}/train_feat.parquet")


def smape(a, f):
    return 200.0 * np.mean(np.abs(f - a) / (np.abs(a) + np.abs(f)))


def ridge_design(df, hol_country=False, fourier_k=4, eoy_country=False):
    X = pd.get_dummies(df[["country", "store", "product", "dow"]].astype(str))
    if hol_country:
        for c in df.country.cat.categories:
            m = (df.country == c).astype(int).values
            X[f"hol_{c}"] = df.is_holiday.values * m
            X[f"holw_{c}"] = df.holiday_win2.values * m
    else:
        X["is_holiday"] = df.is_holiday.values
        X["holiday_win2"] = df.holiday_win2.values
    if eoy_country:
        for v in range(1, 19):
            for c in df.country.cat.categories:
                X[f"eoy_{v}_{c}"] = ((df.eoy_idx == v) & (df.country == c)).astype(int).values
    else:
        for v in range(1, 19):
            X[f"eoy_{v}"] = (df.eoy_idx == v).astype(int).values
    for k in range(1, fourier_k + 1):
        s = np.sin(2 * np.pi * k * df.doy / 365.0)
        c_ = np.cos(2 * np.pi * k * df.doy / 365.0)
        for p in df["product"].cat.categories:
            m = (df["product"] == p).astype(int).values
            X[f"sin{k}_{p}"] = s.values * m
            X[f"cos{k}_{p}"] = c_.values * m
    X["log_gdp"] = df.log_gdp.values
    X["year_c"] = (df.year - 2015).values
    return X.astype(float)


VARIANTS = {
    "A_base": {},
    "B_holc": {"hol_country": True},
    "C_f6": {"fourier_k": 6},
    "D_eoyc": {"eoy_country": True},
    "E_all": {"hol_country": True, "fourier_k": 6, "eoy_country": True},
}

oof, actual, results = {}, {}, {}
for name, kw in VARIANTS.items():
    scores, preds = {}, {}
    for y in VAL_YEARS:
        dtr, dva = tr[tr.year < y], tr[tr.year == y].reset_index(drop=True)
        Xtr, Xva = ridge_design(dtr, **kw), ridge_design(dva, **kw)
        Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
        r = Ridge(alpha=1.0)
        r.fit(Xtr, np.log1p(dtr.num_sold))
        p = np.clip(np.expm1(r.predict(Xva)), 1, None)
        preds[y] = p
        actual[y] = dva.num_sold.values
        scores[y] = smape(actual[y], p)
    oof[name] = preds
    results[name] = scores
    print(f"{name:8s} " + " ".join(f"{y}:{s:.4f}" for y, s in scores.items()),
          f"mean:{np.mean(list(scores.values())):.4f}")

best_name = min(results, key=lambda n: results[n][2018])
print(f"\nbest variant by fold2018: {best_name} ({results[best_name][2018]:.4f})")

# shrink factor grid (SMAPE asymmetry)
print("\nshrink factor on best variant:")
for c in np.arange(0.96, 1.021, 0.01):
    ss = {y: smape(actual[y], oof[best_name][y] * c) for y in VAL_YEARS}
    print(f"  c={c:.2f} " + " ".join(f"{y}:{s:.4f}" for y, s in ss.items()))

with open(f"{COMP}/scripts/train_v2_results.json", "w") as f:
    json.dump({n: results[n] for n in results}, f, indent=2)

for name in results:
    experiment_log.log_experiment_v2(
        COMP, model=f"ridge_{name}", metric="SMAPE", direction="minimize",
        score=results[name][2018],
        cv={"strategy": "expanding-year (val 2016/2017/2018), primary fold 2018",
            "scores": [results[name][y] for y in VAL_YEARS],
            "mean": float(np.mean(list(results[name].values())))},
        notes=f"train_v2.py ridge variant {name}: {VARIANTS[name]}")
print("logged")
