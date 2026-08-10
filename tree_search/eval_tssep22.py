"""Tree-search evaluator for tabular-playground-series-sep-2022 (main lane).

Config schema (canonical stored form):
  solo: {"kind": "solo", "model": "ridge_struct"|"lgb_shape"|"lgb_ratio"|"lgb_join"|"naive",
         "params": {...model knobs...},
         "level": {"k_est": "mean"|"last"|"percountry"}}
  blend: handled by the driver via harness eval_blend on cached fold2019 preds.

evaluate(config, node_id=None, timeout_s=None) -> {"score","status","wall_s","result","error"}
  score  = fold2019 SMAPE (postprocessed round+clip; train 2017-18, val 2019)
  result = {"fold_scores": {...}, "n_features": int}
  side effect: caches fold2019 val preds as OOF (+ fold2018 preds as extra) for blend nodes.

Level forecast is honest per fold: k from train years only (2020 k excluded as invalid).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import stage2_inputs  # noqa: E402

COMP = "/home/tjyen/ai_agents/kaggle/competitions/tabular-playground-series-sep-2022"
CACHE_DIR = os.path.join(_HERE, "cache_tssep22_main")

FOLDS = {
    "fold2019": {"train_years": [2017, 2018], "val_year": 2019},
    "fold2018": {"train_years": [2017], "val_year": 2018},
}
PRIMARY = "fold2019"
FOURIER_MAX = 6
PRODUCTS = ["Kaggle Advanced Techniques", "Kaggle Getting Started",
            "Kaggle Recipe Book", "Kaggle for Kids: One Smart Goose"]

_D = {}


def smape(y_true, y_pred):
    yp = np.clip(np.rint(np.asarray(y_pred, dtype=float)), 1, None)
    yt = np.asarray(y_true, dtype=float)
    return 100.0 * np.mean(2.0 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp)))


def _load():
    if _D:
        return _D
    C = "tabular-playground-series-sep-2022"
    tr = pd.read_csv(stage2_inputs.require(
        f"{COMP}/data/train_processed.csv", comp=C, artifact="train_processed.csv",
        columns=["date", "country", "store", "product", "num_sold", "year", "doy365"]),
        parse_dates=["date"])
    lvl = pd.read_csv(stage2_inputs.require(
        f"{COMP}/data/level_table.csv", comp=C, artifact="level_table.csv",
        columns=["country", "year", "daily_level", "gdp_pc", "k_ratio"]))
    hol = pd.read_csv(stage2_inputs.require(
        f"{COMP}/data/holidays.csv", comp=C, artifact="holidays.csv",
        columns=["date", "country", "holiday"],
        produced_by="the external-data pipeline (external_data/apply.py) under this "
                    "competition's own rules verdict"), parse_dates=["date"])
    hol["name"] = hol.holiday.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    m = lvl.set_index(["country", "year"]).daily_level
    tr["daily_level_obs"] = [m[(c, y)] for c, y in zip(tr.country, tr.year)]
    tr["shape"] = tr.num_sold / tr.daily_level_obs
    tr["odd_year"] = (tr.year % 2).astype(int)
    for k in range(5, FOURIER_MAX + 1):
        tr[f"doy_sin{k}"] = np.sin(2 * np.pi * k * tr.doy365 / 365)
        tr[f"doy_cos{k}"] = np.cos(2 * np.pi * k * tr.doy365 / 365)
    _D.update(tr=tr, lvl=lvl, hol=hol)
    return _D


def level_forecast(lvl, train_years, target_year, k_est="mean"):
    n_days = 366 if target_year % 4 == 0 else 365
    gdp_t = lvl[lvl.year == target_year].set_index("country").gdp_pc
    usable = [y for y in train_years if y != 2020]
    sub = lvl[lvl.year.isin(usable)]
    if k_est == "mean":
        k_hat = sub.groupby("year").k_ratio.mean().mean()
        return {c: k_hat * gdp_t[c] / n_days for c in gdp_t.index}
    if k_est == "last":
        last = max(usable)
        k_hat = sub[sub.year == last].k_ratio.mean()
        return {c: k_hat * gdp_t[c] / n_days for c in gdp_t.index}
    if k_est == "percountry":
        kc = sub.groupby("country").k_ratio.mean()
        return {c: kc[c] * gdp_t[c] / n_days for c in gdp_t.index}
    raise ValueError(k_est)


def fourier_cols(k_max):
    return [f"doy_{f}{k}" for k in range(1, k_max + 1) for f in ("sin", "cos")]


def build_design(df, p, hol):
    """Ridge structural design matrix. Knobs in p:
    fourier_k_global, fourier_k_product, kids_parity_k,
    eoy_start (doy365), eoy_end, hol_before, hol_after, per_name (bool),
    dow_store (bool): dow x store interaction."""
    n = len(df)
    dense = []
    dense.append((df.store == "KaggleRama").astype(float).values)
    for prod in PRODUCTS[1:]:
        dense.append((df["product"] == prod).astype(float).values)
    for d in range(1, 7):
        dense.append((df.dow == d).astype(float).values)
    for f in fourier_cols(p["fourier_k_global"]):
        dense.append(df[f].values)
    for prod in PRODUCTS[1:]:
        pm = (df["product"] == prod).astype(float).values
        for f in fourier_cols(p["fourier_k_product"]):
            dense.append(pm * df[f].values)
    if p["kids_parity_k"] > 0:
        kids = (df["product"] == PRODUCTS[3]).astype(float).values
        odd = df["odd_year"].values.astype(float)
        for f in fourier_cols(p["kids_parity_k"]):
            dense.append(kids * odd * df[f].values)
    if p.get("dow_store"):
        rama = (df.store == "KaggleRama").astype(float).values
        for d in range(1, 7):
            dense.append(rama * (df.dow == d).astype(float).values)
    blocks = [sp.csr_matrix(np.column_stack(dense))]
    # EOY day dummies: doy365 in [eoy_start..365] + [1..eoy_end]
    eoy_days = list(range(p["eoy_start"], 366)) + list(range(1, p["eoy_end"] + 1))
    dmap = {d: i for i, d in enumerate(eoy_days)}
    eoy = np.zeros((n, len(eoy_days)))
    for i, dv in enumerate(df.doy365.values):
        j = dmap.get(int(dv))
        if j is not None:
            eoy[i, j] = 1.0
    blocks.append(sp.csr_matrix(eoy))
    # holiday dummies
    offsets = list(range(-p["hol_before"], p["hol_after"] + 1))
    if p["per_name"]:
        names = sorted(hol.name.unique())
        idx = {(nm, o): i * len(offsets) + j for i, nm in enumerate(names)
               for j, o in enumerate(offsets)}
        lookup = {}
        for r in hol.itertuples():
            for off in offsets:
                lookup.setdefault((r.country, r.date + pd.Timedelta(days=off)),
                                  []).append((r.name, off))
        ncol = len(names) * len(offsets)
    else:
        idx = {o: j for j, o in enumerate(offsets)}
        lookup = {}
        for r in hol.itertuples():
            for off in offsets:
                lookup.setdefault((r.country, r.date + pd.Timedelta(days=off)),
                                  []).append(off)
        ncol = len(offsets)
    rows, cols = [], []
    dates = pd.to_datetime(df.date).values.astype("datetime64[D]")
    for i, (c, d) in enumerate(zip(df.country.values, dates)):
        for key in lookup.get((c, pd.Timestamp(d)), []):
            if p["per_name"]:
                rows.append(i); cols.append(idx[key])
            else:
                rows.append(i); cols.append(idx[key])
    blocks.append(sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, ncol)))
    return sp.hstack(blocks).tocsr()


def _lgb_predict(df_tr, y, df_va, feats, p):
    import lightgbm as lgb
    params = dict(objective=p.get("objective", "regression_l1"),
                  n_estimators=p.get("n_estimators", 900),
                  learning_rate=p.get("learning_rate", 0.05),
                  num_leaves=p.get("num_leaves", 64),
                  min_child_samples=p.get("min_child_samples", 20),
                  subsample=p.get("subsample", 0.9), subsample_freq=1,
                  colsample_bytree=p.get("colsample_bytree", 0.8),
                  reg_alpha=p.get("reg_alpha", 0.0), reg_lambda=p.get("reg_lambda", 0.0),
                  deterministic=True, force_row_wise=True, num_threads=10,
                  verbosity=-1, seed=p.get("seed", 42))
    m = lgb.LGBMRegressor(**params)
    m.fit(df_tr[feats], y)
    return m.predict(df_va[feats])


CAT_COLS = ["country", "store", "product"]


def _shape_feats(p):
    base = CAT_COLS + ["doy365", "dow", "is_weekend", "month", "day", "odd_year",
                       "is_holiday", "days_since_hol", "days_until_hol"]
    return base + fourier_cols(p.get("fourier_k", 4))


def evaluate(config, node_id=None, timeout_s=None):
    t0 = time.time()
    try:
        d = _load()
        tr, lvl, hol = d["tr"], d["lvl"], d["hol"]
        model = config["model"]
        p = dict(config.get("params", {}))
        lv_cfg = dict(config.get("level", {"k_est": "mean"}))
        fold_scores = {}
        preds_store = {}
        for fold, cfg in FOLDS.items():
            tr_years, val_year = cfg["train_years"], cfg["val_year"]
            m_tr = tr.year.isin(tr_years).values
            m_va = (tr.year == val_year).values
            y_va = tr.loc[m_va, "num_sold"].values
            lv = level_forecast(lvl, tr_years, val_year, lv_cfg["k_est"])
            lev = tr.loc[m_va, "country"].map(lv).values.astype(float)

            if model == "ridge_struct":
                dfc = tr.copy()
                for c in CAT_COLS:
                    dfc[c] = dfc[c].astype(str)
                X = build_design(dfc, p, hol)
                r = Ridge(alpha=p.get("alpha", 3.0), solver="sparse_cg", random_state=42)
                r.fit(X[m_tr], np.log(tr.loc[m_tr, "shape"].values))
                pred = np.exp(r.predict(X[m_va])) * lev
                nfeat = X.shape[1]
            elif model in ("lgb_shape", "lgb_ratio", "lgb_join"):
                dfc = tr.copy()
                for c in CAT_COLS:
                    dfc[c] = dfc[c].astype("category")
                feats = _shape_feats(p)
                if model == "lgb_shape":
                    yt = np.log(tr.loc[m_tr, "shape"].values)
                    pred = np.exp(_lgb_predict(dfc[m_tr], yt, dfc[m_va], feats, p)) * lev
                elif model == "lgb_ratio":
                    feats = feats + ["year_c"]
                    yt = np.log(tr.loc[m_tr, "num_sold"].values / tr.loc[m_tr, "gdp_pc"].values)
                    pred = np.exp(_lgb_predict(dfc[m_tr], yt, dfc[m_va], feats, p)) * tr.loc[m_va, "gdp_pc"].values
                else:
                    feats = feats + ["gdp_pc", "log_gdp", "year_c"]
                    yt = np.log(tr.loc[m_tr, "num_sold"].values)
                    pred = np.exp(_lgb_predict(dfc[m_tr], yt, dfc[m_va], feats, p))
                nfeat = len(feats)
            elif model == "naive":
                key = ["store", "product", "dow", "month"]
                gm = tr[m_tr].groupby(key, observed=True)["shape"].mean().rename("gm").reset_index()
                shp = tr.loc[m_va, key].merge(gm, on=key, how="left").gm.fillna(1.0).values
                pred = shp * lev
                nfeat = 4
            else:
                raise ValueError(model)
            fold_scores[fold] = round(smape(y_va, pred), 5)
            preds_store[fold] = pred
        score = fold_scores[PRIMARY]
        if node_id is not None:
            os.makedirs(CACHE_DIR, exist_ok=True)
            hv2.cache_oof(CACHE_DIR, node_id, preds_store["fold2019"],
                          pred18=preds_store["fold2018"])
        return {"score": score, "status": "evaluated", "wall_s": round(time.time() - t0, 2),
                "result": {"fold_scores": fold_scores, "n_features": int(nfeat)},
                "error": None}
    except Exception as e:  # noqa: BLE001 - eval contract: never raise
        return {"score": None, "status": "failed", "wall_s": round(time.time() - t0, 2),
                "result": None, "error": f"{type(e).__name__}: {e}"}


def fold_targets():
    d = _load()
    tr = d["tr"]
    return {f: tr.loc[tr.year == c["val_year"], "num_sold"].values
            for f, c in FOLDS.items()}
