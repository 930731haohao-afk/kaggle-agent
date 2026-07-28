"""
EDA Part 1: Target analysis, temporal patterns, client & price data.
For predict-energy-behavior-of-prosumers competition.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

data_dir = "competitions/predict-energy-behavior-of-prosumers/data"
plot_dir = "competitions/predict-energy-behavior-of-prosumers/scripts/plots"
os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
train['datetime'] = pd.to_datetime(train['datetime'])
client = pd.read_csv(os.path.join(data_dir, "client.csv"))
client['date'] = pd.to_datetime(client['date'])
elec = pd.read_csv(os.path.join(data_dir, "electricity_prices.csv"))
elec['forecast_date'] = pd.to_datetime(elec['forecast_date'])
gas = pd.read_csv(os.path.join(data_dir, "gas_prices.csv"))
gas['forecast_date'] = pd.to_datetime(gas['forecast_date'])

print(f"Train: {train.shape}, Client: {client.shape}")
print(f"Electricity prices: {elec.shape}, Gas prices: {gas.shape}")

# ============================================================
# 2. Target Distribution Analysis
# ============================================================
print("\n" + "=" * 70)
print("2. TARGET DISTRIBUTION ANALYSIS")
print("=" * 70)

target = train['target']
print(f"Overall target stats:")
print(f"  Count: {target.count()} / {len(target)} ({target.isnull().sum()} missing)")
print(f"  Mean: {target.mean():.3f}")
print(f"  Std: {target.std():.3f}")
print(f"  Median: {target.median():.3f}")
print(f"  Skewness: {target.skew():.3f}")
print(f"  Kurtosis: {target.kurtosis():.3f}")

# Percentiles
pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
print(f"\nPercentiles:")
for p in pcts:
    val = np.nanpercentile(target, p)
    print(f"  {p:>3}th: {val:>12.3f}")

# By is_consumption
print(f"\nTarget by is_consumption:")
for ic, label in [(0, "Production"), (1, "Consumption")]:
    subset = train[train['is_consumption'] == ic]['target']
    zeros = (subset == 0).sum()
    print(f"  {label} (is_consumption={ic}):")
    print(f"    Mean={subset.mean():.3f}, Median={subset.median():.3f}, "
          f"Std={subset.std():.3f}")
    print(f"    Zeros: {zeros} ({zeros/len(subset)*100:.1f}%)")
    print(f"    Skewness: {subset.skew():.3f}")

# Log transform analysis
print(f"\nLog1p transform analysis:")
log_target = np.log1p(target.dropna())
print(f"  log1p(target) skewness: {log_target.skew():.3f} (vs {target.skew():.3f} raw)")
print(f"  log1p(target) kurtosis: {log_target.kurtosis():.3f} (vs {target.kurtosis():.3f} raw)")

# Target distribution plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(target.dropna().clip(upper=2000), bins=100, alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Target Distribution (clipped at 2000)')
axes[0, 0].set_xlabel('Target (MWh)')

axes[0, 1].hist(log_target, bins=100, alpha=0.7, color='orange', edgecolor='black')
axes[0, 1].set_title('log1p(Target) Distribution')
axes[0, 1].set_xlabel('log1p(Target)')

for ic, color, label in [(0, 'green', 'Production'), (1, 'blue', 'Consumption')]:
    subset = train[train['is_consumption'] == ic]['target'].dropna()
    axes[1, 0].hist(subset.clip(upper=2000), bins=100, alpha=0.5, color=color,
                    label=label, edgecolor='black')
axes[1, 0].set_title('Target by Type (clipped at 2000)')
axes[1, 0].legend()

for ic, color, label in [(0, 'green', 'Production'), (1, 'blue', 'Consumption')]:
    subset = train[train['is_consumption'] == ic]['target'].dropna()
    axes[1, 1].hist(np.log1p(subset), bins=100, alpha=0.5, color=color,
                    label=label, edgecolor='black')
axes[1, 1].set_title('log1p(Target) by Type')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '01_target_distribution.png'), dpi=100)
plt.close()
print("Saved target distribution plot")

# ============================================================
# 3. Temporal Patterns
# ============================================================
print("\n" + "=" * 70)
print("3. TEMPORAL PATTERNS")
print("=" * 70)

train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.dayofweek
train['month'] = train['datetime'].dt.month
train['date'] = train['datetime'].dt.date

# Hourly pattern by consumption type
print("\nHourly pattern (mean target):")
hourly = train.groupby(['hour', 'is_consumption'])['target'].mean().unstack()
hourly.columns = ['Production', 'Consumption']
print(hourly.to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Hourly
for ic, color, label in [(0, 'green', 'Production'), (1, 'blue', 'Consumption')]:
    subset = train[train['is_consumption'] == ic].groupby('hour')['target'].mean()
    axes[0, 0].plot(subset.index, subset.values, color=color, marker='o', label=label)
axes[0, 0].set_title('Mean Target by Hour')
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Mean Target (MWh)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Day of week
for ic, color, label in [(0, 'green', 'Production'), (1, 'blue', 'Consumption')]:
    subset = train[train['is_consumption'] == ic].groupby('dayofweek')['target'].mean()
    axes[0, 1].plot(subset.index, subset.values, color=color, marker='o', label=label)
axes[0, 1].set_title('Mean Target by Day of Week')
axes[0, 1].set_xlabel('Day (0=Mon)')
axes[0, 1].set_ylabel('Mean Target (MWh)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Monthly
for ic, color, label in [(0, 'green', 'Production'), (1, 'blue', 'Consumption')]:
    subset = train[train['is_consumption'] == ic].groupby('month')['target'].mean()
    axes[1, 0].plot(subset.index, subset.values, color=color, marker='o', label=label)
axes[1, 0].set_title('Mean Target by Month')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Mean Target (MWh)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Daily time series (aggregated)
daily = train.groupby(['date', 'is_consumption'])['target'].mean().reset_index()
for ic, color, label in [(0, 'green', 'Production'), (1, 'blue', 'Consumption')]:
    subset = daily[daily['is_consumption'] == ic]
    axes[1, 1].plot(subset['date'], subset['target'], color=color, alpha=0.5, label=label)
axes[1, 1].set_title('Daily Mean Target Over Time')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '02_temporal_patterns.png'), dpi=100)
plt.close()
print("Saved temporal patterns plot")

# Key temporal observations
print("\nKey temporal observations:")
print(f"  Peak production hour: {train[train['is_consumption']==0].groupby('hour')['target'].mean().idxmax()} "
      f"({train[train['is_consumption']==0].groupby('hour')['target'].mean().max():.1f} MWh)")
print(f"  Peak consumption hour: {train[train['is_consumption']==1].groupby('hour')['target'].mean().idxmax()} "
      f"({train[train['is_consumption']==1].groupby('hour')['target'].mean().max():.1f} MWh)")

weekend = train[train['dayofweek'] >= 5]['target'].mean()
weekday = train[train['dayofweek'] < 5]['target'].mean()
print(f"  Weekday vs Weekend: weekday={weekday:.1f}, weekend={weekend:.1f} "
      f"(ratio={weekday/weekend:.3f})")

# ============================================================
# 4. Segment Analysis
# ============================================================
print("\n" + "=" * 70)
print("4. SEGMENT ANALYSIS")
print("=" * 70)

# By county
print("\nTarget by county:")
county_stats = train.groupby('county')['target'].agg(['mean', 'median', 'std', 'count'])
county_stats = county_stats.sort_values('mean', ascending=False)
print(county_stats.head(16).to_string())

# By product_type
print("\nTarget by product_type:")
for pt in sorted(train['product_type'].unique()):
    subset = train[train['product_type'] == pt]['target']
    print(f"  product_type={pt}: mean={subset.mean():.3f}, median={subset.median():.3f}, "
          f"std={subset.std():.3f}, n={len(subset)}")

# By is_business
print("\nTarget by is_business:")
for ib in [0, 1]:
    subset = train[train['is_business'] == ib]['target']
    print(f"  is_business={ib}: mean={subset.mean():.3f}, median={subset.median():.3f}, "
          f"std={subset.std():.3f}, n={len(subset)}")

# By prediction_unit_id
print(f"\nTarget by prediction_unit_id (top 10 by mean):")
unit_stats = train.groupby('prediction_unit_id')['target'].agg(['mean', 'median', 'std', 'count'])
unit_stats = unit_stats.sort_values('mean', ascending=False)
print(unit_stats.head(10).to_string())

# Segment plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# County means
county_means = train.groupby(['county', 'is_consumption'])['target'].mean().unstack()
county_means.columns = ['Production', 'Consumption']
county_means.plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Mean Target by County')
axes[0, 0].set_xlabel('County')
axes[0, 0].tick_params(axis='x', rotation=0)

# Product type means
pt_means = train.groupby(['product_type', 'is_consumption'])['target'].mean().unstack()
pt_means.columns = ['Production', 'Consumption']
pt_means.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Mean Target by Product Type')

# Business vs non-business
biz_means = train.groupby(['is_business', 'is_consumption'])['target'].mean().unstack()
biz_means.columns = ['Production', 'Consumption']
biz_means.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Mean Target: Business vs Non-Business')

# Hourly pattern by business type
for ib, style in [(0, '-'), (1, '--')]:
    for ic, color, label in [(0, 'green', 'Prod'), (1, 'blue', 'Cons')]:
        mask = (train['is_business'] == ib) & (train['is_consumption'] == ic)
        subset = train[mask].groupby('hour')['target'].mean()
        biz_label = "Biz" if ib else "Res"
        axes[1, 1].plot(subset.index, subset.values, color=color, linestyle=style,
                        label=f"{biz_label}-{label}", alpha=0.8)
axes[1, 1].set_title('Hourly Pattern by Business Type')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '03_segment_analysis.png'), dpi=100)
plt.close()
print("Saved segment analysis plot")

# ============================================================
# 5. Client Data Analysis
# ============================================================
print("\n" + "=" * 70)
print("5. CLIENT DATA ANALYSIS")
print("=" * 70)

print(f"Client table: {client.shape}")
print(f"Date range: {client['date'].min()} to {client['date'].max()}")
print(f"Unique data_block_ids: {client['data_block_id'].nunique()}")

# Latest client snapshot
latest_client = client[client['data_block_id'] == client['data_block_id'].max()]
print(f"\nLatest client snapshot ({len(latest_client)} rows):")
print(f"  EIC count range: {latest_client['eic_count'].min()} - {latest_client['eic_count'].max()}")
print(f"  Installed capacity range: {latest_client['installed_capacity'].min():.1f} - "
      f"{latest_client['installed_capacity'].max():.1f} kW")

# How much do client features change over time?
print("\nClient feature variability over time:")
for col in ['eic_count', 'installed_capacity']:
    grouped = client.groupby(['county', 'is_business', 'product_type'])[col]
    cv = (grouped.std() / grouped.mean()).mean()
    print(f"  {col}: mean CV across units = {cv:.4f}")

# Check if installed_capacity correlates with target
# Merge latest client with train (via data_block_id)
train_sample = train[train['data_block_id'] == 300].copy()
client_sample = client[client['data_block_id'] == 300].copy()
merged = train_sample.merge(client_sample,
                            on=['county', 'is_business', 'product_type'],
                            how='left', suffixes=('', '_client'))
if 'installed_capacity' in merged.columns:
    corr_prod = merged[merged['is_consumption'] == 0][['target', 'installed_capacity']].corr().iloc[0, 1]
    corr_cons = merged[merged['is_consumption'] == 1][['target', 'installed_capacity']].corr().iloc[0, 1]
    print(f"\nCorrelation (target vs installed_capacity):")
    print(f"  Production: {corr_prod:.4f}")
    print(f"  Consumption: {corr_cons:.4f}")
    corr_eic = merged[merged['is_consumption'] == 1][['target', 'eic_count']].corr().iloc[0, 1]
    corr_eic_prod = merged[merged['is_consumption'] == 0][['target', 'eic_count']].corr().iloc[0, 1]
    print(f"\nCorrelation (target vs eic_count):")
    print(f"  Production: {corr_eic_prod:.4f}")
    print(f"  Consumption: {corr_eic:.4f}")

# ============================================================
# 6. Energy Price Analysis
# ============================================================
print("\n" + "=" * 70)
print("6. ENERGY PRICE ANALYSIS")
print("=" * 70)

print(f"Electricity prices: {elec.shape}")
print(f"  EUR/MWh range: {elec['euros_per_mwh'].min():.2f} to {elec['euros_per_mwh'].max():.2f}")
print(f"  Mean: {elec['euros_per_mwh'].mean():.2f}, Median: {elec['euros_per_mwh'].median():.2f}")
print(f"  Negative prices: {(elec['euros_per_mwh'] < 0).sum()}")

print(f"\nGas prices: {gas.shape}")
print(f"  Lowest: {gas['lowest_price_per_mwh'].min():.2f} - {gas['lowest_price_per_mwh'].max():.2f}")
print(f"  Highest: {gas['highest_price_per_mwh'].min():.2f} - {gas['highest_price_per_mwh'].max():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(elec['forecast_date'], elec['euros_per_mwh'], alpha=0.5, linewidth=0.5)
axes[0].set_title('Electricity Prices Over Time')
axes[0].set_ylabel('EUR/MWh')
axes[0].tick_params(axis='x', rotation=45)

axes[1].plot(gas['forecast_date'], gas['lowest_price_per_mwh'], label='Lowest', alpha=0.7)
axes[1].plot(gas['forecast_date'], gas['highest_price_per_mwh'], label='Highest', alpha=0.7)
axes[1].set_title('Gas Prices Over Time')
axes[1].set_ylabel('EUR/MWh')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '04_prices.png'), dpi=100)
plt.close()
print("Saved price plots")

# ============================================================
# 7. Missing Values & Data Quality
# ============================================================
print("\n" + "=" * 70)
print("7. MISSING VALUES & DATA QUALITY")
print("=" * 70)

print("Train missing values:")
for col in train.columns:
    n_miss = train[col].isnull().sum()
    if n_miss > 0:
        print(f"  {col}: {n_miss} ({n_miss/len(train)*100:.3f}%)")

# Check for gaps in datetime per prediction_unit
print("\nChecking for temporal gaps per prediction_unit...")
gap_counts = []
for uid in train['prediction_unit_id'].unique()[:5]:  # Sample 5 units
    unit_data = train[train['prediction_unit_id'] == uid].sort_values('datetime')
    for ic in [0, 1]:
        subset = unit_data[unit_data['is_consumption'] == ic]
        diffs = subset['datetime'].diff().dropna()
        expected = pd.Timedelta(hours=1)
        gaps = diffs[diffs != expected]
        gap_counts.append(len(gaps))

avg_gaps = np.mean(gap_counts)
print(f"  Average gaps per unit-consumption pair (sample of 5 units): {avg_gaps:.1f}")

# data_block_id analysis
print(f"\ndata_block_id analysis:")
block_counts = train.groupby('data_block_id').size()
print(f"  Rows per block: min={block_counts.min()}, max={block_counts.max()}, "
      f"mean={block_counts.mean():.0f}")
print(f"  Blocks with fewer rows: {(block_counts < block_counts.max()).sum()}")

# ============================================================
# 8. Lag/Autocorrelation Analysis
# ============================================================
print("\n" + "=" * 70)
print("8. AUTOCORRELATION ANALYSIS")
print("=" * 70)

# Pick one prediction unit to analyze autocorrelation
sample_unit = train[(train['prediction_unit_id'] == 0) &
                    (train['is_consumption'] == 1)].sort_values('datetime')
ts = sample_unit.set_index('datetime')['target'].dropna()

print(f"Autocorrelation for prediction_unit_id=0, consumption:")
for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # 168 = 7 days
    if len(ts) > lag:
        acf = ts.autocorr(lag=lag)
        label = ""
        if lag == 1: label = "(1 hour)"
        elif lag == 24: label = "(1 day)"
        elif lag == 48: label = "(2 days)"
        elif lag == 168: label = "(1 week)"
        print(f"  Lag {lag:>4}: {acf:.4f} {label}")

# Check for a few more units
print("\nAutocorrelation at lag=24 (1 day) across units:")
for uid in [0, 10, 20, 30, 50]:
    for ic, label in [(0, "Prod"), (1, "Cons")]:
        subset = train[(train['prediction_unit_id'] == uid) &
                       (train['is_consumption'] == ic)].sort_values('datetime')
        ts_sub = subset.set_index('datetime')['target'].dropna()
        if len(ts_sub) > 24:
            acf = ts_sub.autocorr(lag=24)
            print(f"  Unit {uid:>2} {label}: {acf:.4f}")

# ============================================================
# 9. Validation Strategy Analysis
# ============================================================
print("\n" + "=" * 70)
print("9. VALIDATION STRATEGY ANALYSIS")
print("=" * 70)

# Check for time-dependent target drift
monthly_stats = train.groupby([train['datetime'].dt.to_period('M'), 'is_consumption'])['target'].mean()
monthly_stats = monthly_stats.unstack()
monthly_stats.columns = ['Production', 'Consumption']
print("Monthly mean target over time:")
print(monthly_stats.to_string())

print("\n=> Time-series split is REQUIRED. Random splits would leak future data.")
print("   Recommendation: Use expanding or sliding window time-series split.")
print("   Example: Train on months 1-18, validate on months 19-21.")

print("\nDone with main EDA.")
