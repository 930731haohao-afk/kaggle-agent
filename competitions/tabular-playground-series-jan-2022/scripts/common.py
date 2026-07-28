"""Shared data/feature/eval code for tabular-playground-series-jan-2022.

Structural model family: log(num_sold) ~ log_gdp + year_c + store + product
+ dow + Fourier(doy) [x product/store/country] + per-holiday-name shifted dummies.
Validation: expanding year folds (train<=2016 -> val 2017, train<=2017 -> val 2018).
"""
import numpy as np
import pandas as pd
import holidays as holidays_lib

DIR = "/home/tjyen/ai_agents/kaggle/competitions/tabular-playground-series-jan-2022"

# World Bank NY.GDP.PCAP.CD (current US$), fetched 2026-07-28
GDP = {
    ("Finland", 2015): 42560.35, ("Finland", 2016): 43451.26,
    ("Finland", 2017): 46085.02, ("Finland", 2018): 49654.25,
    ("Finland", 2019): 48358.18,
    ("Norway", 2015): 77220.95, ("Norway", 2016): 73222.40,
    ("Norway", 2017): 78771.22, ("Norway", 2018): 85579.08,
    ("Norway", 2019): 79329.31,
    ("Sweden", 2015): 51188.17, ("Sweden", 2016): 51703.51,
    ("Sweden", 2017): 53210.22, ("Sweden", 2018): 54018.46,
    ("Sweden", 2019): 51648.99,
}

CC = {"Finland": "FI", "Norway": "NO", "Sweden": "SE"}


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(forecast)
    return 200.0 * np.mean(np.abs(forecast - actual) / denom)


def load_data():
    tr = pd.read_csv(f"{DIR}/data/train.csv", parse_dates=["date"])
    te = pd.read_csv(f"{DIR}/data/test.csv", parse_dates=["date"])
    return tr, te


def _holiday_table():
    """(country, date) -> set of English holiday names, split on '; ', drop 'Sunday'."""
    rows = []
    for country, cc in CC.items():
        h = holidays_lib.country_holidays(cc, years=range(2014, 2021), language="en_US")
        for d, raw in h.items():
            for name in str(raw).split("; "):
                if name.strip() and name.strip() != "Sunday":
                    rows.append((country, pd.Timestamp(d), name.strip()))
    return pd.DataFrame(rows, columns=["country", "hdate", "hname"])


_HOL = None


def holiday_frame():
    global _HOL
    if _HOL is None:
        _HOL = _holiday_table()
    return _HOL


def build_design(df: pd.DataFrame, cfg: dict):
    """Return (X DataFrame of floats, column names) for the structural linear model."""
    out = {}
    d = df["date"]
    year = d.dt.year.values
    out["log_gdp"] = np.log([GDP[(c, y)] for c, y in zip(df["country"], year)])
    if cfg.get("year_c", True):
        out["year_c"] = (year - 2017.0)
    if cfg.get("year_c2", False):
        out["year_c2"] = (year - 2017.0) ** 2
    out["store_rama"] = (df["store"] == "KaggleRama").astype(float).values
    for p in ["Kaggle Mug", "Kaggle Sticker"]:
        out[f"prod_{p.split()[-1]}"] = (df["product"] == p).astype(float).values
    if cfg.get("country_dummies", False):
        for c in ["Norway", "Sweden"]:
            out[f"cty_{c}"] = (df["country"] == c).astype(float).values

    dow = d.dt.dayofweek.values
    for k in range(6):
        out[f"dow_{k}"] = (dow == k).astype(float)
    if cfg.get("dow_product", False):
        for p in ["Kaggle Mug", "Kaggle Sticker"]:
            pm = (df["product"] == p).values.astype(float)
            for k in range(6):
                out[f"dow_{k}_x_{p.split()[-1]}"] = (dow == k) * pm

    # Fourier day-of-year
    K = cfg.get("fourier_k", 8)
    t = d.dt.dayofyear.values / 365.25
    fours = {}
    for k in range(1, K + 1):
        fours[f"sin{k}"] = np.sin(2 * np.pi * k * t)
        fours[f"cos{k}"] = np.cos(2 * np.pi * k * t)
    for name, v in fours.items():
        out[f"f_{name}"] = v
    if cfg.get("fourier_product", True):
        KP = cfg.get("fourier_product_k", K)
        for p in ["Kaggle Mug", "Kaggle Sticker"]:
            pm = (df["product"] == p).values.astype(float)
            for k in range(1, KP + 1):
                out[f"f_sin{k}_x_{p.split()[-1]}"] = fours[f"sin{k}"] * pm
                out[f"f_cos{k}_x_{p.split()[-1]}"] = fours[f"cos{k}"] * pm
    if cfg.get("fourier_country", False):
        KC = cfg.get("fourier_country_k", 2)
        for c in ["Norway", "Sweden"]:
            cm = (df["country"] == c).values.astype(float)
            for k in range(1, KC + 1):
                out[f"f_sin{k}_x_{c}"] = fours[f"sin{k}"] * cm
                out[f"f_cos{k}_x_{c}"] = fours[f"cos{k}"] * cm

    # per-holiday-name shifted dummies, pooled across countries by name
    if cfg.get("holidays", True):
        lo, hi = cfg.get("hol_lo", -5), cfg.get("hol_hi", 10)
        hol = holiday_frame()
        key = pd.MultiIndex.from_arrays([df["country"].values, d.values])
        for off in range(lo, hi + 1):
            shifted = hol.copy()
            shifted["kdate"] = shifted["hdate"] + pd.Timedelta(days=off)
            idx = pd.MultiIndex.from_arrays([shifted["country"], shifted["kdate"]])
            for name in hol["hname"].unique():
                sel = idx[shifted["hname"].values == name]
                col = f"hol_{name}_{off:+d}"
                v = key.isin(sel).astype(float)
                if v.sum() > 0:
                    out[col] = v
    X = pd.DataFrame(out, index=df.index)
    return X


FOLDS = [(2016, 2017), (2017, 2018)]  # (last train year, val year)


def _ridge_one(tr_df, va_df, cfg):
    from sklearn.linear_model import Ridge
    Xtr = build_design(tr_df, cfg)
    Xva = build_design(va_df, cfg)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    y = np.log(tr_df["num_sold"].values)
    if cfg.get("gdp_offset", False):
        # fix the GDP elasticity at 1: subtract log_gdp from target, drop the column
        y = y - Xtr["log_gdp"].values
        off_va = Xva["log_gdp"].values
        Xtr = Xtr.drop(columns=["log_gdp"])
        Xva = Xva.drop(columns=["log_gdp"])
    else:
        off_va = 0.0
    sw = None
    rw = cfg.get("recent_weight")
    if rw is not None:
        yrs = tr_df["date"].dt.year.values
        sw = float(rw) ** (yrs.max() - yrs)
    m = Ridge(alpha=cfg.get("alpha", 0.1))
    m.fit(Xtr.values, y, sample_weight=sw)
    return np.exp(m.predict(Xva.values) + off_va)


def fit_predict_ridge(tr_df, va_df, cfg):
    if cfg.get("per_product", False):
        pred = np.empty(len(va_df))
        for p in va_df["product"].unique():
            trm = tr_df["product"] == p
            vam = (va_df["product"] == p).values
            sub = dict(cfg, fourier_product=False, dow_product=False)
            pred[vam] = _ridge_one(tr_df[trm], va_df[vam], sub)
        return pred
    return _ridge_one(tr_df, va_df, cfg)


def fit_predict_lgb(tr_df, va_df, cfg):
    import lightgbm as lgb
    def feats(df):
        d = df["date"]
        X = pd.DataFrame({
            "dow": d.dt.dayofweek, "doy": d.dt.dayofyear, "month": d.dt.month,
            "day": d.dt.day, "year_c": d.dt.year - 2017,
            "country": df["country"].astype("category").cat.codes,
            "store": (df["store"] == "KaggleRama").astype(int),
            "product": df["product"].astype("category").cat.codes,
        }, index=df.index)
        hol = holiday_frame()
        # distance to nearest holiday (any name), per country
        for country in CC:
            pass
        merged = df[["country"]].copy()
        merged["date"] = d
        hh = hol.rename(columns={"hdate": "date"})[["country", "date"]].drop_duplicates()
        hh["is_hol"] = 1
        m = merged.merge(hh, on=["country", "date"], how="left")
        X["is_hol"] = m["is_hol"].fillna(0).values
        return X
    Xtr, Xva = feats(tr_df), feats(va_df)
    gdp_tr = np.array([GDP[(c, y)] for c, y in zip(tr_df["country"], tr_df["date"].dt.year)])
    gdp_va = np.array([GDP[(c, y)] for c, y in zip(va_df["country"], va_df["date"].dt.year)])
    y = np.log(tr_df["num_sold"].values / gdp_tr)
    params = dict(objective="regression", metric="mae", num_leaves=cfg.get("num_leaves", 31),
                  learning_rate=cfg.get("lr", 0.05), n_estimators=cfg.get("n_estimators", 800),
                  min_child_samples=cfg.get("min_child_samples", 20),
                  subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                  num_threads=10, deterministic=True, force_row_wise=True,
                  seed=cfg.get("seed", 42), verbose=-1)
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, y, categorical_feature=["dow", "month", "country", "product"])
    pred = np.exp(m.predict(Xva)) * gdp_va
    return pred


def evaluate(cfg, return_preds=False):
    """Expanding-year folds. Returns dict with per-fold smape + mean, optionally val preds."""
    tr, _ = load_data()
    tr["year"] = tr["date"].dt.year
    fit = fit_predict_lgb if cfg.get("model") == "lgb" else fit_predict_ridge
    scores, preds = {}, {}
    for last_tr, val_y in FOLDS:
        tr_df = tr[tr["year"] <= last_tr]
        va_df = tr[tr["year"] == val_y]
        p = fit(tr_df, va_df, cfg)
        if cfg.get("round_int", True):
            p = np.round(p)
        scores[val_y] = smape(va_df["num_sold"].values, p)
        preds[val_y] = pd.Series(p, index=va_df.index)
    res = {"fold_2017": scores[2017], "fold_2018": scores[2018],
           "mean": (scores[2017] + scores[2018]) / 2}
    if return_preds:
        res["preds"] = preds
    return res


def predict_test(cfg):
    """Fit on full train (2015-2018), predict 2019 test."""
    tr, te = load_data()
    fit = fit_predict_lgb if cfg.get("model") == "lgb" else fit_predict_ridge
    p = fit(tr, te, cfg)
    if cfg.get("round_int", True):
        p = np.round(p)
    return te, p
