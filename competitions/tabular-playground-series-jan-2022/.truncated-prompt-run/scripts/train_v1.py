"""Baseline + candidate models for tabular-playground-series-jan-2022.

CV: expanding year folds (train ..Y-1, validate year Y) for Y in 2016, 2017, 2018.
Fold 2018 mirrors the real task (full-year-ahead extrapolation) — primary score.

Models:
  naive     — same-day-last-year (dow-aligned, lag 364)
  lgb_raw   — LightGBM on log1p(num_sold)
  lgb_gdp   — LightGBM on log(num_sold / gdp_pc), multiply back
  ridge     — structural: Ridge on log1p, one-hot decomposition + log_gdp + linear year
Then SLSQP-free grid weight search over model OOFs (fold-2018 weights).
"""
import importlib.util
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

COMP = "competitions/tabular-playground-series-jan-2022"
SEED = 42
VAL_YEARS = [2016, 2017, 2018]

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

tr = pd.read_parquet(f"{COMP}/train_feat.parquet")
tr = tr.sort_values(["country", "store", "product", "date"]).reset_index(drop=True)

def smape(a: np.ndarray, f: np.ndarray) -> float:
    return 200.0 * np.mean(np.abs(f - a) / (np.abs(a) + np.abs(f)))

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    lambda_l2=1.0,
    num_threads=10,
    deterministic=True,
    force_row_wise=True,
    seed=SEED,
    verbosity=-1,
)
FEATS_RAW = ["country", "store", "product", "year", "month", "day", "dow", "doy",
             "is_weekend", "sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "sin4", "cos4",
             "is_holiday", "holiday_win2", "eoy_idx", "gdp_pc", "log_gdp"]
FEATS_GDP = [c for c in FEATS_RAW if c not in ("gdp_pc", "log_gdp")]


def ridge_design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[["country", "store", "product", "dow"]].astype(str))
    X["is_holiday"] = df.is_holiday.values
    X["holiday_win2"] = df.holiday_win2.values
    for v in range(1, 19):
        X[f"eoy_{v}"] = (df.eoy_idx == v).astype(int).values
    for k in (1, 2, 3, 4):
        for p in df["product"].cat.categories:
            m = (df["product"] == p).astype(int).values
            X[f"sin{k}_{p}"] = df[f"sin{k}"].values * m
            X[f"cos{k}_{p}"] = df[f"cos{k}"].values * m
    X["log_gdp"] = df.log_gdp.values
    X["year_c"] = (df.year - 2015).values
    return X.astype(float)


def fit_predict(model: str, dtr: pd.DataFrame, dva: pd.DataFrame) -> np.ndarray:
    if model == "naive":
        key = ["country", "store", "product"]
        src = dtr.assign(nd=dtr.date + pd.Timedelta(days=364))
        m = dva.merge(src[key + ["nd", "num_sold"]].rename(
            columns={"nd": "date", "num_sold": "lag364"}), on=key + ["date"], how="left")
        fallback = dtr.groupby(key, observed=True).num_sold.mean()
        m["lag364"] = m.lag364.fillna(m.set_index(key).index.map(fallback).to_series(index=m.index))
        return m.lag364.values
    if model in ("lgb_raw", "lgb_gdp"):
        feats = FEATS_RAW if model == "lgb_raw" else FEATS_GDP
        ytr = (np.log1p(dtr.num_sold) if model == "lgb_raw"
               else np.log(dtr.num_sold / dtr.gdp_pc))
        yva = (np.log1p(dva.num_sold) if model == "lgb_raw"
               else np.log(dva.num_sold / dva.gdp_pc))
        ds_tr = lgb.Dataset(dtr[feats], ytr)
        ds_va = lgb.Dataset(dva[feats], yva, reference=ds_tr)
        bst = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=3000,
                        valid_sets=[ds_va],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = bst.predict(dva[feats], num_iteration=bst.best_iteration)
        return (np.expm1(pred) if model == "lgb_raw" else np.exp(pred) * dva.gdp_pc.values)
    if model == "ridge":
        Xtr, Xva = ridge_design(dtr), ridge_design(dva)
        Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
        r = Ridge(alpha=1.0)
        r.fit(Xtr, np.log1p(dtr.num_sold))
        return np.expm1(r.predict(Xva))
    raise ValueError(model)


MODELS = ["naive", "lgb_raw", "lgb_gdp", "ridge"]
oof = {m: {} for m in MODELS}  # model -> year -> preds
actual = {}
results = {}
for m in MODELS:
    scores = {}
    for y in VAL_YEARS:
        dtr = tr[tr.year < y]
        dva = tr[tr.year == y].reset_index(drop=True)
        p = np.clip(fit_predict(m, dtr, dva), 1, None)
        oof[m][y] = p
        actual[y] = dva.num_sold.values
        scores[y] = smape(dva.num_sold.values, p)
    results[m] = scores
    print(f"{m:8s} " + " ".join(f"{y}:{s:.4f}" for y, s in scores.items()),
          f"mean:{np.mean(list(scores.values())):.4f}")

# --- blend weight search on fold 2018 (mirrors test extrapolation) ---
cand = ["lgb_raw", "lgb_gdp", "ridge"]
best = (None, 1e9)
grid = np.arange(0, 1.01, 0.05)
for w1 in grid:
    for w2 in grid:
        if w1 + w2 > 1:
            continue
        w3 = 1 - w1 - w2
        p = w1 * oof[cand[0]][2018] + w2 * oof[cand[1]][2018] + w3 * oof[cand[2]][2018]
        s = smape(actual[2018], p)
        if s < best[1]:
            best = ((w1, w2, round(w3, 2)), s)
print(f"\nblend {cand} weights={best[0]} fold2018 SMAPE={best[1]:.4f}")
# same weights evaluated on other folds
for y in VAL_YEARS:
    w = best[0]
    p = w[0] * oof[cand[0]][y] + w[1] * oof[cand[1]][y] + w[2] * oof[cand[2]][y]
    print(f"  blend @ {y}: {smape(actual[y], p):.4f}")

with open(f"{COMP}/scripts/train_v1_results.json", "w") as f:
    json.dump({"per_model": results,
               "blend": {"models": cand, "weights": best[0], "fold2018": best[1]}}, f, indent=2)

for m in MODELS:
    experiment_log.log_experiment_v2(
        COMP, model=m, metric="SMAPE", direction="minimize",
        score=results[m][2018],
        cv={"strategy": "expanding-year (val 2016/2017/2018), primary fold 2018",
            "scores": [results[m][y] for y in VAL_YEARS],
            "mean": float(np.mean(list(results[m].values())))},
        features=(FEATS_RAW if m == "lgb_raw" else FEATS_GDP if m == "lgb_gdp" else None),
        notes=f"train_v1.py {m}; naive=lag364 seasonal" if m == "naive" else f"train_v1.py {m}")
experiment_log.log_experiment_v2(
    COMP, model="blend_v1", metric="SMAPE", direction="minimize", score=best[1],
    ensemble={"models": cand, "weights": list(best[0]), "search": "grid 0.05 on fold2018"},
    notes="train_v1.py weight search on fold-2018 OOF")
print("\nlogged to experiments.json")
