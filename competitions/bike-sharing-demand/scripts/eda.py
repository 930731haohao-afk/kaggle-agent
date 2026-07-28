"""EDA for Bike Sharing Demand competition."""
import pandas as pd
import numpy as np

data_dir = "competitions/bike-sharing-demand/data"

train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["datetime"])
test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["datetime"])

# ============================================================
# 1. TARGET ANALYSIS
# ============================================================
print("=" * 60)
print("1. TARGET ANALYSIS")
print("=" * 60)

target = train["count"]
print(f"\nDistribution of 'count':")
print(target.describe())
print(f"\nSkewness: {target.skew():.3f}")
print(f"Kurtosis: {target.kurtosis():.3f}")

# Percentiles
for p in [1, 5, 10, 90, 95, 99]:
    print(f"  P{p}: {target.quantile(p/100):.0f}")

# Log-transform check
log_target = np.log1p(target)
print(f"\nLog1p(count) skewness: {log_target.skew():.3f}")
print(f"Log1p(count) kurtosis: {log_target.kurtosis():.3f}")

# casual vs registered breakdown
print(f"\nCasual  mean: {train['casual'].mean():.1f} ({train['casual'].mean()/target.mean()*100:.1f}% of total)")
print(f"Registered mean: {train['registered'].mean():.1f} ({train['registered'].mean()/target.mean()*100:.1f}% of total)")

# ============================================================
# 2. DATETIME ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("2. DATETIME ANALYSIS")
print("=" * 60)

train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek
train["month"] = train["datetime"].dt.month
train["year"] = train["datetime"].dt.year
train["day"] = train["datetime"].dt.day

print(f"\nDate range: {train['datetime'].min()} to {train['datetime'].max()}")
print(f"Years: {sorted(train['year'].unique())}")
print(f"Months: {sorted(train['month'].unique())}")
print(f"Days in train: {sorted(train['day'].unique())}")

test["day"] = test["datetime"].dt.day
print(f"Days in test: {sorted(test['day'].unique())}")

# Hourly pattern
print("\nMean count by hour:")
hourly = train.groupby("hour")["count"].mean()
for h, v in hourly.items():
    bar = "#" * int(v / 10)
    print(f"  Hour {h:2d}: {v:6.1f} {bar}")

# Day of week pattern
print("\nMean count by day of week (0=Mon):")
dow = train.groupby("dayofweek")["count"].mean()
for d, v in dow.items():
    print(f"  Day {d}: {v:.1f}")

# Monthly pattern
print("\nMean count by month:")
monthly = train.groupby("month")["count"].mean()
for m, v in monthly.items():
    print(f"  Month {m:2d}: {v:.1f}")

# Year trend
print("\nMean count by year:")
yearly = train.groupby("year")["count"].mean()
for y, v in yearly.items():
    print(f"  {y}: {v:.1f}")

# ============================================================
# 3. CATEGORICAL FEATURE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("3. CATEGORICAL FEATURES")
print("=" * 60)

for col in ["season", "holiday", "workingday", "weather"]:
    print(f"\n--- {col} ---")
    vc = train[col].value_counts().sort_index()
    means = train.groupby(col)["count"].mean()
    for val in vc.index:
        print(f"  {val}: n={vc[val]:5d} ({vc[val]/len(train)*100:5.1f}%), mean_count={means[val]:.1f}")

# ============================================================
# 4. NUMERICAL FEATURE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("4. NUMERICAL FEATURES")
print("=" * 60)

num_cols = ["temp", "atemp", "humidity", "windspeed"]
print("\nCorrelation with count:")
for col in num_cols:
    corr = train[col].corr(train["count"])
    print(f"  {col}: {corr:.3f}")

print("\nCorrelation between temp and atemp:", train["temp"].corr(train["atemp"]).round(3))

# Windspeed zero check
zero_wind = (train["windspeed"] == 0).sum()
print(f"\nWindspeed == 0: {zero_wind} ({zero_wind/len(train)*100:.1f}%)")

# Humidity extremes
print(f"Humidity == 0: {(train['humidity'] == 0).sum()}")
print(f"Humidity == 100: {(train['humidity'] == 100).sum()}")

# ============================================================
# 5. FEATURE-TARGET RELATIONSHIPS
# ============================================================
print("\n" + "=" * 60)
print("5. FEATURE-TARGET INTERACTIONS")
print("=" * 60)

# Hour x Workingday interaction
print("\nMean count by Hour x Workingday:")
pivot = train.pivot_table(values="count", index="hour", columns="workingday", aggfunc="mean")
print("  Hour | Non-Working | Working")
for h in range(24):
    print(f"  {h:4d} | {pivot.loc[h, 0]:11.1f} | {pivot.loc[h, 1]:7.1f}")

# Season x Hour
print("\nMean count by Season x Hour (top 5 hours):")
season_hour = train.pivot_table(values="count", index="hour", columns="season", aggfunc="mean")
top_hours = season_hour.mean(axis=1).nlargest(5).index
for h in sorted(top_hours):
    vals = " | ".join(f"S{s}:{season_hour.loc[h, s]:5.1f}" for s in [1, 2, 3, 4])
    print(f"  Hour {h:2d}: {vals}")

# ============================================================
# 6. TRAIN-TEST CONSISTENCY
# ============================================================
print("\n" + "=" * 60)
print("6. TRAIN-TEST CONSISTENCY")
print("=" * 60)

test["hour"] = test["datetime"].dt.hour
test["month"] = test["datetime"].dt.month
test["year"] = test["datetime"].dt.year

for col in num_cols:
    tr_mean, te_mean = train[col].mean(), test[col].mean()
    tr_std, te_std = train[col].std(), test[col].std()
    print(f"  {col:12s}: train={tr_mean:.2f}±{tr_std:.2f}, test={te_mean:.2f}±{te_std:.2f}")

# Check weather distribution
print("\nWeather distribution (train vs test):")
for w in [1, 2, 3, 4]:
    tr_pct = (train["weather"] == w).mean() * 100
    te_pct = (test["weather"] == w).mean() * 100
    print(f"  Weather {w}: train={tr_pct:.1f}%, test={te_pct:.1f}%")

# ============================================================
# 7. LEAKAGE CHECK
# ============================================================
print("\n" + "=" * 60)
print("7. LEAKAGE CHECK")
print("=" * 60)

print("casual + registered = count check:")
mismatch = (train["casual"] + train["registered"] != train["count"]).sum()
print(f"  Mismatches: {mismatch}")
print("  -> casual and registered are DIRECT COMPONENTS of count (NOT available in test)")

# Check if datetime order matters
print("\nIs data sorted by datetime?", train["datetime"].is_monotonic_increasing)

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
