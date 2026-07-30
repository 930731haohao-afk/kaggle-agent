"""Held-out-year probe: can a local protocol pick the injection FORM without the leaderboard?

The 2x2 measured on s3e19 and sep-2022 showed that TimeSeriesSplit cross-validation ranks
the operator forms wrongly, in opposite directions on the two competitions, because every
fold validates INSIDE the training range while the real test year lies outside it. Selecting
a form by CV is therefore unsound, and selecting by leaderboard costs a submission per form
(impossible during a live competition).

This probe tests the cheap alternative: hold out the final training year entirely, fit on
everything before it, and score the forms on that year — a split whose geometry matches the
real task (extrapolate one year beyond the training window). If the resulting ranking
reproduces the leaderboard ranking, an agent can choose the form locally.

All three forms share one frame and one covariate column, so the only difference is HOW the
covariate is used:
    baseline    features = calendar/categorical only,        target = log1p(y)
    featurejoin features = ... + covariate + trend + holiday, target = log1p(y)
    ratio       features = ... + trend + holiday,             target = log(y / covariate)

Usage: VIRTUAL_ENV= uv run python3 tree_search/holdout_year_probe.py
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

BASE_FEATS = ["year", "month", "day", "dow", "day_of_year", "weekofyear", "quarter",
              "is_weekend", "is_month_start", "is_month_end", "is_new_year",
              "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
              "country_cat", "store_cat", "product_cat"]
CATS = ["country_cat", "store_cat", "product_cat"]
PARAMS = dict(objective="rmse", num_leaves=31, learning_rate=0.05, n_estimators=400,
              random_state=42, verbosity=-1)

# each arm's ratio workspace carries cov_level + year_c + is_holiday for the same rows
CASES = [
    # comp, workspace holding the augmented CSVs, held-out years to probe, known LB ranking
    ("playground-series-s3e19", "playground-series-s3e19-v5-ratio", [2021],
     ["featurejoin", "ratio", "baseline"]),
    ("tabular-playground-series-sep-2022", "tabular-playground-series-sep-2022-v5-ratio",
     [2019, 2020], ["featurejoin", "ratio", "baseline"]),
]


def smape(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (np.abs(a) + np.abs(b)) / 2
    out = np.zeros_like(d)
    m = d != 0
    out[m] = np.abs(a[m] - b[m]) / d[m]
    return float(out.mean() * 100)


def fit_score(df: pd.DataFrame, feats: list[str], y_fit: np.ndarray, invert,
              tr_mask: np.ndarray, va_mask: np.ndarray) -> float:
    X = df[feats].copy()
    for c in CATS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X[tr_mask], y_fit[tr_mask])
    pred = invert(m.predict(X[va_mask]), va_mask)
    return smape(df.loc[va_mask, "num_sold"].to_numpy(float), pred)


def run_case(comp: str, ws: str, years: list[int], lb_rank: list[str]) -> dict:
    path = os.path.join(_ROOT, "competitions", ws, "data", "train_processed.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["_y"] = df["date"].dt.year
    y = df["num_sold"].to_numpy(float)
    cov = df["cov_level"].to_numpy(float)
    extras = [c for c in ("year_c", "is_holiday") if c in df.columns]

    forms = {
        "baseline":    (BASE_FEATS, np.log1p(y), lambda p, m: np.expm1(p)),
        "featurejoin": (BASE_FEATS + ["cov_level"] + extras, np.log1p(y),
                        lambda p, m: np.expm1(p)),
        "ratio":       (BASE_FEATS + extras, np.log(y / cov),
                        lambda p, m: np.exp(p) * cov[m]),
    }
    out = {"comp": comp, "lb_ranking": lb_rank, "probes": {}}
    for hy in years:
        tr_mask = (df["_y"] < hy).to_numpy()
        va_mask = (df["_y"] == hy).to_numpy()
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue
        scores = {name: round(fit_score(df, f, yf, inv, tr_mask, va_mask), 4)
                  for name, (f, yf, inv) in forms.items()}
        rank = sorted(scores, key=scores.get)
        out["probes"][hy] = {"train_years": f"<{hy}", "n_train": int(tr_mask.sum()),
                             "n_val": int(va_mask.sum()), "smape": scores,
                             "ranking": rank, "matches_lb": rank == lb_rank,
                             "top_matches_lb": rank[0] == lb_rank[0]}
    return out


def main():
    results = []
    for comp, ws, years, lb in CASES:
        r = run_case(comp, ws, years, lb)
        results.append(r)
        print(f"=== {comp}")
        print(f"    leaderboard ranking (private): {' < '.join(lb)}")
        for hy, p in r["probes"].items():
            sc = p["smape"]
            print(f"    hold out {hy} (fit on {p['train_years']}, {p['n_train']} rows -> "
                  f"{p['n_val']} rows)")
            print("        " + "  ".join(f"{k}={v}" for k, v in sc.items()))
            print(f"        ranking: {' < '.join(p['ranking'])}   "
                  f"{'MATCHES LB' if p['matches_lb'] else ('top matches' if p['top_matches_lb'] else 'DISAGREES')}")
        print()
    json.dump(results, open(os.path.join(_HERE, "holdout_year_probe_result.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
