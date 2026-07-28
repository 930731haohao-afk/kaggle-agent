"""EDA for tabular-playground-series-jan-2022.

Verifies the structural assumptions recorded in knowledge/experience.md:
constant store ratio, constant product shares, shared dow effect, GDP link.
"""
import pandas as pd
import numpy as np

DIR = "/home/tjyen/ai_agents/kaggle/competitions/tabular-playground-series-jan-2022"
tr = pd.read_csv(f"{DIR}/data/train.csv", parse_dates=["date"])
te = pd.read_csv(f"{DIR}/data/test.csv", parse_dates=["date"])

print("train", tr.shape, tr.date.min(), tr.date.max())
print("test ", te.shape, te.date.min(), te.date.max())
print("nulls train", tr.isnull().sum().sum(), "test", te.isnull().sum().sum())
print("countries", sorted(tr.country.unique()))
print("stores", sorted(tr.store.unique()))
print("products", sorted(tr["product"].unique()))
print("target: min %d max %d mean %.1f skew %.2f" % (
    tr.num_sold.min(), tr.num_sold.max(), tr.num_sold.mean(), tr.num_sold.skew()))

tr["year"] = tr.date.dt.year
# yearly totals per country
piv = tr.pivot_table(index="year", columns="country", values="num_sold", aggfunc="sum")
print("\nYearly totals per country:\n", piv)
print("\nYoY growth:\n", piv.pct_change())

# store ratio per year
sr = tr.pivot_table(index="year", columns="store", values="num_sold", aggfunc="sum")
print("\nStore ratio KaggleRama/KaggleMart per year:\n", (sr.iloc[:, 1] / sr.iloc[:, 0]))

# product share per year
ps = tr.pivot_table(index="year", columns="product", values="num_sold", aggfunc="sum")
print("\nProduct shares per year:\n", ps.div(ps.sum(axis=1), axis=0))

# product share per year per country (check cross-country constancy)
psc = tr.pivot_table(index=["country", "year"], columns="product", values="num_sold", aggfunc="sum")
print("\nProduct shares per country-year:\n", psc.div(psc.sum(axis=1), axis=0).round(4))

# dow effect per store (multiplicative, on log scale)
tr["dow"] = tr.date.dt.dayofweek
g = tr.groupby(["store", "dow"]).num_sold.mean().unstack()
print("\nDOW mean by store (normalized to row mean):\n", g.div(g.mean(axis=1), axis=0).round(4))
g2 = tr.groupby(["country", "dow"]).num_sold.mean().unstack()
print("\nDOW mean by country (normalized):\n", g2.div(g2.mean(axis=1), axis=0).round(4))

# monthly seasonality
tr["month"] = tr.date.dt.month
m = tr.groupby(["country", "month"]).num_sold.mean().unstack()
print("\nMonth mean by country (normalized):\n", m.div(m.mean(axis=1), axis=0).round(3))

# GDP per capita check (World Bank NY.GDP.PCAP.CD, current US$) — hardcoded known values
gdp = {
    ("Finland", 2015): 42802, ("Finland", 2016): 43814, ("Finland", 2017): 46412,
    ("Finland", 2018): 50038, ("Finland", 2019): 48712,
    ("Norway", 2015): 74356, ("Norway", 2016): 70461, ("Norway", 2017): 75497,
    ("Norway", 2018): 82268, ("Norway", 2019): 75826,
    ("Sweden", 2015): 51545, ("Sweden", 2016): 52069, ("Sweden", 2017): 53792,
    ("Sweden", 2018): 54589, ("Sweden", 2019): 51687,
}
print("\ntotal(country,year) / gdp_pc:")
for (c, y), v in sorted(gdp.items()):
    if y <= 2018:
        tot = piv.loc[y, c]
        print(f"  {c} {y}: {tot / v:.3f}")
