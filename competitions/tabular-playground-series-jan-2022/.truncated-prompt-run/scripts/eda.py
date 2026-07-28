"""EDA for tabular-playground-series-jan-2022 (SMAPE, 18 daily series 2015-2018 -> 2019)."""
import pandas as pd
import numpy as np

DATA = "competitions/tabular-playground-series-jan-2022/data"

train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
test = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date"])

print("=== SHAPES ===")
print("train", train.shape, "test", test.shape)
print("train dates", train.date.min(), "->", train.date.max())
print("test dates", test.date.min(), "->", test.date.max())
print("nulls train", train.isnull().sum().sum(), "test", test.isnull().sum().sum())

print("\n=== CATEGORIES ===")
for c in ["country", "store", "product"]:
    print(c, sorted(train[c].unique()), "| test same:", set(train[c].unique()) == set(test[c].unique()))
print("n series:", train.groupby(["country", "store", "product"]).ngroups)

print("\n=== TARGET ===")
t = train.num_sold
print(t.describe())
print("skew", round(t.skew(), 3), "log1p skew", round(np.log1p(t).skew(), 3))
print("zeros:", (t == 0).sum(), "integer:", (t == t.astype(int)).all())

print("\n=== TARGET BY CATEGORY (mean) ===")
for c in ["country", "store", "product"]:
    print(train.groupby(c).num_sold.mean().round(1).to_dict())

print("\n=== YEARLY TOTALS (trend) ===")
train["year"] = train.date.dt.year
yearly = train.groupby("year").num_sold.sum()
print(yearly)
print("YoY growth %:", (yearly.pct_change() * 100).round(2).to_dict())

print("\n=== YEARLY BY COUNTRY (share) ===")
yc = train.groupby(["year", "country"]).num_sold.sum().unstack()
print((yc.div(yc.sum(axis=1), axis=0) * 100).round(2))

print("\n=== STORE RATIO PER YEAR (KaggleRama / KaggleMart) ===")
ys = train.groupby(["year", "store"]).num_sold.sum().unstack()
print((ys["KaggleRama"] / ys["KaggleMart"]).round(4))

print("\n=== PRODUCT SHARE PER YEAR ===")
yp = train.groupby(["year", "product"]).num_sold.sum().unstack()
print((yp.div(yp.sum(axis=1), axis=0) * 100).round(2))

print("\n=== WEEKLY PATTERN (mean by dow) ===")
train["dow"] = train.date.dt.dayofweek
print(train.groupby("dow").num_sold.mean().round(1))

print("\n=== MONTHLY PATTERN (mean by month) ===")
train["month"] = train.date.dt.month
print(train.groupby("month").num_sold.mean().round(1))

print("\n=== END-OF-YEAR SPIKE (Dec 20 - Jan 10 daily means, all series) ===")
md = train.assign(mmdd=train.date.dt.strftime("%m-%d"))
eoy = md[md.mmdd.isin([f"12-{d}" for d in range(24, 32)] + [f"01-0{d}" for d in range(1, 10)])]
print(eoy.groupby("mmdd").num_sold.mean().round(1).sort_index())

print("\n=== SERIES-LEVEL CoV (std/mean per series) ===")
g = train.groupby(["country", "store", "product"]).num_sold.agg(["mean", "std"])
g["cov"] = (g["std"] / g["mean"]).round(3)
print(g.sort_values("cov").round(1))

print("\n=== PRODUCT SEASONALITY: product share by month ===")
mp = train.groupby(["month", "product"]).num_sold.sum().unstack()
print((mp.div(mp.sum(axis=1), axis=0) * 100).round(2))
