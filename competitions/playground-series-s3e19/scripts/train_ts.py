"""Time-series-aware pipeline for s3e19 Mini-Course Sales (SMAPE).

Fixes the generic pipeline's failure (random CV -> SMAPE 5 local but 50 on LB):
- TIME-BASED validation: hold out 2021 (train on 2017-2020) to mirror the 2022 test.
- Series-level + seasonal features on log1p(num_sold): num_sold is ~separable into
  country x store x product x (day-of-year, day-of-week) seasonal shape.
- Target encodings (series, series x dow, series x month) computed from TRAIN years
  only -> no leakage; for the 2022 submission they use all of 2017-2021.

No raw `year` feature (2022 is unseen -> avoids extrapolation failure); trend is mild
(2022 level ~ 2021) so seasonal+level generalizes.
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

COMP = "competitions/playground-series-s3e19"
DATA, SUB, EXP = f"{COMP}/data", f"{COMP}/submissions", f"{COMP}/experiments.json"
GROUP = ["country", "store", "product"]
os.makedirs(SUB, exist_ok=True)


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y) + np.abs(p)
    return np.mean(np.where(d == 0, 0.0, 2 * np.abs(p - y) / d)) * 100


def add_date_feats(df):
    d = df["date"].dt
    df["month"] = d.month
    df["day"] = d.day
    df["dow"] = d.dayofweek
    df["doy"] = d.dayofyear
    df["woy"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def add_target_enc(train_src, apply_to):
    """Mean of log1p(num_sold) at several granularities, learned from train_src."""
    out = apply_to.copy()
    src = train_src.copy()
    src["logy"] = np.log1p(src["num_sold"])
    gmean = src["logy"].mean()
    specs = {"te_series": GROUP, "te_series_dow": GROUP + ["dow"],
             "te_series_month": GROUP + ["month"], "te_series_woy": GROUP + ["woy"]}
    for name, keys in specs.items():
        lut = src.groupby(keys)["logy"].mean()
        merged = out.merge(lut.rename(name), on=keys, how="left")[name]
        out[name] = merged.fillna(gmean).values
    return out


BASE = ["month", "day", "dow", "doy", "woy", "is_weekend",
        "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
CATF = GROUP
TE = ["te_series", "te_series_dow", "te_series_month", "te_series_woy"]


def encode_cats(tr, *others):
    maps = {}
    for c in CATF:
        cats = pd.Index(tr[c].astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(cats)}
    for frame in (tr, *others):
        for c in CATF:
            frame[c + "_e"] = frame[c].astype(str).map(maps[c]).fillna(-1).astype(int)
    return [c + "_e" for c in CATF]


def fit_predict(Xtr, ytr, Xva):
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    lg = lgb.LGBMRegressor(objective="regression", n_estimators=3000, learning_rate=0.03,
                           num_leaves=127, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
    lg.fit(Xtr, ytr)
    ct = CatBoostRegressor(loss_function="RMSE", iterations=3000, learning_rate=0.03, depth=8,
                           l2_leaf_reg=3.0, random_seed=42, verbose=False)
    ct.fit(Xtr, ytr)
    return 0.5 * lg.predict(Xva) + 0.5 * ct.predict(Xva)


def build_matrix(df, feats):
    return df[feats].to_numpy(np.float32)


def main():
    train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
    test = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date"])
    train = add_date_feats(train)
    test = add_date_feats(test)
    cat_e = encode_cats(train, test)
    FEATS = BASE + cat_e + TE

    # --- time-based validation: hold out 2021 ---
    tr = train[train.date.dt.year <= 2020].copy()
    va = train[train.date.dt.year == 2021].copy()
    tr_e = add_target_enc(tr, tr)
    va_e = add_target_enc(tr, va)          # encodings from <=2020 only (no leak)
    pred_log = fit_predict(build_matrix(tr_e, FEATS), np.log1p(tr["num_sold"].values),
                           build_matrix(va_e, FEATS))
    val_pred = np.clip(np.expm1(pred_log), 0, None)
    val_smape = smape(va["num_sold"].values, val_pred)
    print(f"[s3e19] TIME-BASED val (holdout 2021) SMAPE = {val_smape:.4f}")

    # --- refit on all years, predict 2022 ---
    all_e = add_target_enc(train, train)
    test_e = add_target_enc(train, test)   # encodings from all 2017-2021
    full_log = fit_predict(build_matrix(all_e, FEATS), np.log1p(train["num_sold"].values),
                           build_matrix(test_e, FEATS))
    test_pred = np.clip(np.expm1(full_log), 0, None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame({"id": test["id"], "num_sold": np.round(test_pred).astype(int)})
    path = f"{SUB}/sub_ts_{val_smape:.4f}_{stamp}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}  ({len(sub)} rows)  pred mean={test_pred.mean():.1f}")

    hist = json.load(open(EXP)) if os.path.exists(EXP) else []
    hist.append(dict(experiment_id=len(hist) + 1, timestamp=datetime.now().isoformat(timespec="seconds"),
                     model="LGB+CatBoost time-series (log1p, holdout-2021 CV)", metric="smape",
                     cv="time-based holdout 2021", blend_score=round(float(val_smape), 4),
                     n_features=len(FEATS), notes="TS features + series target encoding; fixes random-CV leak",
                     submission=os.path.basename(path)))
    json.dump(hist, open(EXP, "w"), indent=2)
    print(f"logged experiment #{len(hist)}")


if __name__ == "__main__":
    main()
