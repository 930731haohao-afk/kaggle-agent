"""
EDA for playground-series-s3e19 (Forecast Mini-course Sales)
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/playground-series-s3e19"
data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}\n")


section("DATA OVERVIEW")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train date range: {train.date.min()} -> {train.date.max()} ({train.date.nunique()} unique days)")
print(f"Test date range: {test.date.min()} -> {test.date.max()} ({test.date.nunique()} unique days)")
print(f"Countries: {sorted(train.country.unique())}")
print(f"Stores: {sorted(train['store'].unique())}")
print(f"Products: {sorted(train['product'].unique())}")
print(f"Combos train: {train.groupby(['country','store','product']).ngroups}")
print(f"Combos test: {test.groupby(['country','store','product']).ngroups}")
print(f"rows per combo train: {train.groupby(['country','store','product']).size().unique()}")
print(f"rows per combo test: {test.groupby(['country','store','product']).size().unique()}")

section("TARGET DISTRIBUTION")
print(train.num_sold.describe())
print(f"skew: {train.num_sold.skew():.3f}, log1p skew: {np.log1p(train.num_sold).skew():.3f}")

section("MISSING VALUES")
print("train:", train.isnull().sum().sum(), "test:", test.isnull().sum().sum())
print("duplicated rows train:", train.duplicated(subset=['date','country','store','product']).sum())

section("YEARLY TOTALS (growth trend)")
train['year'] = train.date.dt.year
print(train.groupby('year').num_sold.agg(['sum','mean']))

section("COUNTRY / STORE / PRODUCT SHARE STABILITY")
# ratio of country total to grand total, per year -> check if stable over years (multiplicative structure)
tot_by_year = train.groupby('year').num_sold.sum()
country_year = train.groupby(['year','country']).num_sold.sum().unstack()
print("country share of yearly total:")
print((country_year.div(tot_by_year, axis=0)).round(4))

store_year = train.groupby(['year','store']).num_sold.sum().unstack()
print("\nstore share of yearly total:")
print((store_year.div(tot_by_year, axis=0)).round(4))

product_year = train.groupby(['year','product']).num_sold.sum().unstack()
print("\nproduct share of yearly total:")
print((product_year.div(tot_by_year, axis=0)).round(4))

section("DAY OF WEEK / MONTH SEASONALITY")
train['dow'] = train.date.dt.dayofweek
train['month'] = train.date.dt.month
print("mean num_sold by dow:")
print(train.groupby('dow').num_sold.mean())
print("\nmean num_sold by month:")
print(train.groupby('month').num_sold.mean())

section("GDP-LIKE COUNTRY RATIO CHECK (log num_sold ~ additive by country/store/product?)")
# Check if log(num_sold) decomposes additively: log(sold) = f(date) + g(country) + h(store) + k(product)
logs = np.log1p(train.num_sold)
resid = logs - logs.mean()
by_country = resid.groupby(train.country).mean()
print("mean log-residual by country:", dict(by_country.round(3)))
by_store = resid.groupby(train['store']).mean()
print("mean log-residual by store:", dict(by_store.round(3)))
by_product = resid.groupby(train['product']).mean()
print("mean log-residual by product:", dict(by_product.round(3)))

section("HOLIDAY-ISH SPIKES (New Year's Day check)")
nyd = train[(train.date.dt.month==1)&(train.date.dt.day==1)]
non_nyd_same_dow = train[(train.dow==nyd.dow.iloc[0]) & ~((train.date.dt.month==1)&(train.date.dt.day==1))]
print("Jan1 mean:", nyd.num_sold.mean(), " vs overall mean:", train.num_sold.mean())

section("TRAIN/TEST CONSISTENCY (categoricals)")
print("countries match:", set(train.country.unique())==set(test.country.unique()))
print("stores match:", set(train['store'].unique())==set(test['store'].unique()))
print("products match:", set(train['product'].unique())==set(test['product'].unique()))

section("VALIDATION STRATEGY DECISION")
print("Test period (2022) is entirely AFTER train period (2017-2021) with NO overlap.")
print("This is a pure future-year extrapolation task -> time-based validation recommended")
print("over plain random/K-Fold, which would let the model see 2021 data points on both")
print("sides of a fold drawn from within the same seasonal cycle (optimistic bias for a")
print("model that must actually extrapolate an unseen year's trend/growth).")
print("Decision: use sklearn TimeSeriesSplit(n_splits=5) on sorted unique dates so each")
print("fold trains on an expanding history and validates on a strictly later block -- this")
print("mirrors the train(2017-21)->test(2022) gap while still giving 5 folds as required.")
