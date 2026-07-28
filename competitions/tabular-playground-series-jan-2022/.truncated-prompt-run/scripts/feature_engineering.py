"""Feature engineering for tabular-playground-series-jan-2022.

Outputs train_feat.parquet / test_feat.parquet with GBDT-ready features.
Also prints GDP-proportionality diagnostic (country-year total vs GDP per capita).
"""
import numpy as np
import pandas as pd
import holidays as hol

COMP = "competitions/tabular-playground-series-jan-2022"
DATA = f"{COMP}/data"

train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
test = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date"])
gdp = pd.read_csv(f"{COMP}/data_official/gdp.csv")

# --- GDP proportionality diagnostic ---
t = train.copy()
t["year"] = t.date.dt.year
cy = t.groupby(["country", "year"]).num_sold.sum().reset_index()
cy = cy.merge(gdp, on=["country", "year"])
cy["ratio"] = cy.num_sold / cy.gdp_pc
print("=== country-year total / gdp_pc (constant => proportional) ===")
print(cy.pivot(index="year", columns="country", values="ratio").round(2))

COUNTRY_HOL = {
    "Finland": hol.Finland(years=range(2015, 2020)),
    "Norway": hol.Norway(years=range(2015, 2020)),
    "Sweden": hol.Sweden(years=range(2015, 2020)),
}


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = out.date
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["day"] = d.dt.day
    out["dow"] = d.dt.dayofweek
    out["doy"] = d.dt.dayofyear
    # align doy across leap years (2016 leap): shift post-Feb so Dec 31 aligns
    leap = d.dt.is_leap_year & (out.doy > 59)
    out.loc[leap, "doy"] = out.loc[leap, "doy"] - 1
    out["is_weekend"] = (out.dow >= 5).astype(int)
    for k in (1, 2, 3, 4):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * out.doy / 365.0)
        out[f"cos{k}"] = np.cos(2 * np.pi * k * out.doy / 365.0)
    # holidays per row's own country
    hol_flag = np.zeros(len(out), dtype=int)
    for c, cal in COUNTRY_HOL.items():
        m = (out.country == c).values
        hol_flag[m] = [int(dt in cal) for dt in out.date[m].dt.date]
    out["is_holiday"] = hol_flag
    # holiday window +-2 days (per country)
    win = np.zeros(len(out), dtype=int)
    for c, cal in COUNTRY_HOL.items():
        m = (out.country == c).values
        hd = set(cal.keys())
        win[m] = [
            int(any((dt + pd.Timedelta(days=o)).date() in hd for o in range(-2, 3)))
            for dt in out.date[m]
        ]
    out["holiday_win2"] = win
    # end-of-year ramp: days since Dec 23 within Dec24-Jan10 window, else 0
    mmdd = out.date.dt.strftime("%m-%d")
    eoy_days = {f"12-{dd}": dd - 23 for dd in range(24, 32)}
    eoy_days.update({f"01-0{dd}": dd + 8 for dd in range(1, 10)})
    eoy_days["01-10"] = 18
    out["eoy_idx"] = mmdd.map(eoy_days).fillna(0).astype(int)
    out = out.merge(gdp, on=["country", "year"], how="left")
    out["log_gdp"] = np.log(out.gdp_pc)
    for c in ("country", "store", "product"):
        out[c] = out[c].astype("category")
    return out


tr = build(train)
te = build(test)
assert tr.isnull().sum().sum() == 0 and te.isnull().sum().sum() == 0
feat_cols = [c for c in tr.columns if c not in ("row_id", "date", "num_sold")]
assert [c for c in te.columns if c not in ("row_id", "date")] == feat_cols
tr.to_parquet(f"{COMP}/train_feat.parquet")
te.to_parquet(f"{COMP}/test_feat.parquet")
print(f"\nfeatures ({len(feat_cols)}):", feat_cols)
print("train", tr.shape, "test", te.shape, "nulls OK")
