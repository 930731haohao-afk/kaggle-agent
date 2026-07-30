"""Structural-decomposition probe: is the residual on s3e19 a SHAPE error a GBDT cannot express?

Context. Diagnostic probes on the real leaderboard decomposed our best s3e19 arm's 48.50
private SMAPE into roughly 15.5 points of level error (a global x1.6 multiplier reaches
32.93, x2.2 overshoots to 46.78) and roughly 33 points of residual, against a leaderboard
top of 4.67. So neither external GDP data (worth -6.0 measured) nor the level explains the
gap: the dominant term is shape.

The experience library already names the mechanism, with evidence from a different
competition: "panel time series whose shares are constant across years -> multiplicative
decomposition (Ridge on log1p) beats GBDT" (tpsjan22 exp #4/#12/#13, Ridge 4.19 vs GBDT
8.43). The share diagnostic on s3e19 agrees: store shares are constant to CV 0.0012,
day-of-week multipliers to 0.0046, product shares are near-constant but alternate with year
parity, and only the country share drifts (the part GDP is supposed to carry).

This probe fits both models on the years before a held-out year and scores them on it, so the
comparison needs no leaderboard. If the structural model dominates, the missing capability is
a `structure` operator, not more external data.

Usage: VIRTUAL_ENV= uv run python3 tree_search/structural_probe.py
"""
from __future__ import annotations

import os
import sys

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


def smape(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (np.abs(a) + np.abs(b)) / 2
    out = np.zeros_like(d)
    m = d != 0
    out[m] = np.abs(a[m] - b[m]) / d[m]
    return float(out.mean() * 100)


def structural_fit_predict(tr: pd.DataFrame, va: pd.DataFrame, *, level: str) -> np.ndarray:
    """sales = level(year) x country_share x store_share x product_share(parity) x dow x doy.

    `level` selects how the held-out year's total is obtained:
      "oracle"  - the true total of the held-out year (isolates SHAPE error alone)
      "gdp"     - total(last year) scaled by the GDP-per-capita ratio (learnable in advance)
      "carry"   - total(last year) (naive)
    """
    tr = tr.copy(); va = va.copy()
    for d in (tr, va):
        d["parity"] = d["year"] % 2
    total_by_year = tr.groupby("year")["num_sold"].sum()
    grand = tr["num_sold"].sum()

    country_sh = tr.groupby("country_cat")["num_sold"].sum() / grand
    store_sh = tr.groupby("store_cat")["num_sold"].sum() / grand
    # product share depends on year parity (synthetic generator alternates)
    prod_sh = (tr.groupby(["parity", "product_cat"])["num_sold"].sum()
               / tr.groupby("parity")["num_sold"].sum())
    dow_mult = tr.groupby("dow")["num_sold"].mean() / tr["num_sold"].mean()
    doy_mult = (tr.groupby("day_of_year")["num_sold"].mean() / tr["num_sold"].mean())
    doy_mult = doy_mult.rolling(7, center=True, min_periods=1).mean()

    if level == "oracle":
        va_total = va["num_sold"].sum()
    elif level == "gdp":
        last = total_by_year.index.max()
        va_total = float(total_by_year.loc[last]) * float(va["_gdp_ratio"].iloc[0])
    else:
        va_total = float(total_by_year.loc[total_by_year.index.max()])

    n_days = va["day_of_year"].nunique()
    shares = (va["country_cat"].map(country_sh).to_numpy()
              * va["store_cat"].map(store_sh).to_numpy())
    psh = np.array([prod_sh.get((p, c), np.nan) for p, c in zip(va["parity"], va["product_cat"])])
    shares = shares * psh
    mult = (va["dow"].map(dow_mult).fillna(1.0).to_numpy()
            * va["day_of_year"].map(doy_mult).fillna(1.0).to_numpy())
    # shares sum to 1 over series; spread the year total across days, then apply multipliers
    pred = va_total * shares / n_days * mult
    # renormalize so the multipliers do not change the implied total
    pred = pred * (va_total / pred.sum())
    return pred


def gbdt_fit_predict(tr: pd.DataFrame, va: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = pd.concat([tr[feats], va[feats]])
    for c in CATS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    Xtr, Xva = X.iloc[:len(tr)], X.iloc[len(tr):]
    m = lgb.LGBMRegressor(objective="rmse", num_leaves=31, learning_rate=0.05,
                          n_estimators=400, random_state=42, verbosity=-1)
    m.fit(Xtr, np.log1p(tr["num_sold"].to_numpy(float)))
    return np.expm1(m.predict(Xva))


def main():
    ws = os.path.join(_ROOT, "competitions", "playground-series-s3e19-v5-ratio", "data")
    df = pd.read_csv(os.path.join(ws, "train_processed.csv"), parse_dates=["date"])
    raw = pd.read_csv(os.path.join(_ROOT, "competitions", "playground-series-s3e19",
                                   "data", "train.csv"))
    df = df.merge(raw[["id", "country"]], on="id")
    df["year"] = df["date"].dt.year

    for hy in (2020, 2021):
        tr, va = df[df.year < hy].copy(), df[df.year == hy].copy()
        gdp_ratio = (va.groupby("country")["cov_level"].first().mean()
                     / tr[tr.year == hy - 1].groupby("country")["cov_level"].first().mean())
        va["_gdp_ratio"] = gdp_ratio
        y = va["num_sold"].to_numpy(float)
        res = {
            "GBDT (calendar+cat)":        smape(y, gbdt_fit_predict(tr, va, BASE_FEATS)),
            "GBDT + gdp feature":         smape(y, gbdt_fit_predict(tr, va, BASE_FEATS + ["cov_level"])),
            "structural, level=carry":    smape(y, structural_fit_predict(tr, va, level="carry")),
            "structural, level=gdp":      smape(y, structural_fit_predict(tr, va, level="gdp")),
            "structural, level=ORACLE":   smape(y, structural_fit_predict(tr, va, level="oracle")),
        }
        print(f"=== hold out {hy} (fit on <{hy}, {len(tr)} rows -> {len(va)} rows), "
              f"gdp level ratio {gdp_ratio:.3f}")
        for k, v in sorted(res.items(), key=lambda x: x[1]):
            print(f"      {k:28s} SMAPE = {v:7.3f}")
        print()


if __name__ == "__main__":
    main()
