"""
Feature engineering pipeline for predict-energy-behavior-of-prosumers.
Creates all features and saves processed train data for modeling.

Features:
  1. Time features (hour, dayofweek, month, cyclical, is_weekend)
  2. Lag features (target at t-24, t-48, t-168)
  3. Rolling statistics (7d, 14d rolling mean/std per unit)
  4. Client features (installed_capacity, eic_count via data_block_id)
  5. County-aggregated weather (historical, matched by datetime+county)
  6. Energy prices (electricity, gas via data_block_id)
  7. Interaction features
"""
import pandas as pd
import numpy as np
import os
import time
import gc

data_dir = "competitions/predict-energy-behavior-of-prosumers/data"
out_dir = "competitions/predict-energy-behavior-of-prosumers/data"

t0 = time.time()

# ============================================================
# 1. Load raw data
# ============================================================
print("=" * 70)
print("1. LOADING RAW DATA")
print("=" * 70)

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
train['datetime'] = pd.to_datetime(train['datetime'])
print(f"Train: {train.shape}")

client = pd.read_csv(os.path.join(data_dir, "client.csv"))
client['date'] = pd.to_datetime(client['date'])
print(f"Client: {client.shape}")

elec = pd.read_csv(os.path.join(data_dir, "electricity_prices.csv"))
elec['forecast_date'] = pd.to_datetime(elec['forecast_date'])
print(f"Electricity prices: {elec.shape}")

gas = pd.read_csv(os.path.join(data_dir, "gas_prices.csv"))
gas['forecast_date'] = pd.to_datetime(gas['forecast_date'])
print(f"Gas prices: {gas.shape}")

ws_map = pd.read_csv(os.path.join(data_dir, "weather_station_to_county_mapping.csv"))
ws_map = ws_map.dropna(subset=['county'])
ws_map['county'] = ws_map['county'].astype(int)
print(f"Weather station mapping: {ws_map.shape}")

# ============================================================
# 2. Weather: aggregate by county (chunked loading)
# ============================================================
print("\n" + "=" * 70)
print("2. AGGREGATING WEATHER BY COUNTY")
print("=" * 70)

weather_cols = ['temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure',
                'cloudcover_total', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
                'windspeed_10m', 'shortwave_radiation', 'direct_solar_radiation',
                'diffuse_radiation']

print("Loading and aggregating historical weather by county (chunked)...")
county_weather_chunks = []
chunk_size = 200000
for chunk in pd.read_csv(os.path.join(data_dir, "historical_weather.csv"),
                         chunksize=chunk_size):
    chunk['datetime'] = pd.to_datetime(chunk['datetime'])
    # Join with county mapping
    merged = chunk.merge(ws_map[['latitude', 'longitude', 'county']],
                         on=['latitude', 'longitude'], how='inner')
    # Aggregate by datetime + county
    agg = merged.groupby(['datetime', 'county'])[weather_cols].mean().reset_index()
    county_weather_chunks.append(agg)

county_weather = pd.concat(county_weather_chunks, ignore_index=True)
# Re-aggregate in case chunks split same datetime
county_weather = county_weather.groupby(['datetime', 'county'])[weather_cols].mean().reset_index()
print(f"County weather: {county_weather.shape}")
del county_weather_chunks
gc.collect()

# ============================================================
# 3. Sort and prepare train
# ============================================================
print("\n" + "=" * 70)
print("3. PREPARING TRAIN DATA")
print("=" * 70)

train = train.sort_values(['prediction_unit_id', 'is_consumption', 'datetime']).reset_index(drop=True)

# ============================================================
# 4. Time features
# ============================================================
print("Adding time features...")
train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.dayofweek
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['is_weekend'] = (train['dayofweek'] >= 5).astype(int)
train['hour_sin'] = np.sin(2 * np.pi * train['hour'] / 24)
train['hour_cos'] = np.cos(2 * np.pi * train['hour'] / 24)
train['month_sin'] = np.sin(2 * np.pi * train['month'] / 12)
train['month_cos'] = np.cos(2 * np.pi * train['month'] / 12)
train['dayofweek_sin'] = np.sin(2 * np.pi * train['dayofweek'] / 7)
train['dayofweek_cos'] = np.cos(2 * np.pi * train['dayofweek'] / 7)

print(f"  Time features added: hour, dayofweek, month, day, is_weekend, cyclical encodings")

# ============================================================
# 5. Lag features
# ============================================================
print("Adding lag features...")

# Group by prediction unit + consumption type for proper lag computation
group_cols = ['prediction_unit_id', 'is_consumption']

for lag in [1, 2, 3, 24, 48, 168]:
    col_name = f'target_lag_{lag}'
    train[col_name] = train.groupby(group_cols)['target'].shift(lag)
    n_null = train[col_name].isnull().sum()
    print(f"  {col_name}: {n_null} NaN ({n_null/len(train)*100:.1f}%)")

# Diff features
train['target_diff_24'] = train['target_lag_24'] - train['target_lag_48']
train['target_diff_168'] = train['target_lag_24'] - train['target_lag_168']

# Same hour, same day-of-week last week
train['target_lag_same_hour_last_week'] = train.groupby(group_cols)['target'].shift(168)

print(f"  Lag and diff features added")

# ============================================================
# 6. Rolling statistics
# ============================================================
print("Adding rolling statistics...")

# Rolling mean/std of target over past 24h, 168h (7 days)
for window in [24, 168]:
    roll = train.groupby(group_cols)['target'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    train[f'target_roll_mean_{window}'] = roll

    roll_std = train.groupby(group_cols)['target'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    train[f'target_roll_std_{window}'] = roll_std
    print(f"  target_roll_mean_{window}, target_roll_std_{window}")

# Rolling mean for same hour of day (past 7 instances = 7 days)
# This captures "what was the average at this hour over the past week"
train['target_roll_mean_same_hour_7d'] = train.groupby(
    group_cols + ['hour']
)['target'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)
print(f"  target_roll_mean_same_hour_7d")

# ============================================================
# 7. Client features
# ============================================================
print("\nMerging client features...")

# Client data is keyed by (product_type, county, is_business, data_block_id)
# The client data_block_id is 2 ahead of train's (data available 2 days before)
client_merge_cols = ['product_type', 'county', 'is_business', 'data_block_id']
client_features = ['eic_count', 'installed_capacity']

train = train.merge(
    client[client_merge_cols + client_features],
    on=client_merge_cols,
    how='left'
)
n_matched = train['installed_capacity'].notna().sum()
print(f"  Matched: {n_matched}/{len(train)} ({n_matched/len(train)*100:.1f}%)")
print(f"  Missing installed_capacity: {train['installed_capacity'].isnull().sum()}")

# ============================================================
# 8. Weather features (county-aggregated)
# ============================================================
print("\nMerging weather features...")

# Rename weather columns with prefix
weather_rename = {col: f'weather_{col}' for col in weather_cols}
county_weather_renamed = county_weather.rename(columns=weather_rename)

train = train.merge(
    county_weather_renamed,
    on=['datetime', 'county'],
    how='left'
)
n_weather = train['weather_temperature'].notna().sum()
print(f"  Matched: {n_weather}/{len(train)} ({n_weather/len(train)*100:.1f}%)")
del county_weather, county_weather_renamed
gc.collect()

# ============================================================
# 9. Electricity price features
# ============================================================
print("\nMerging electricity price features...")

# Electricity prices: hourly, merge on data_block_id + hour
elec['hour'] = elec['forecast_date'].dt.hour
elec_daily = elec.groupby('data_block_id').agg(
    elec_price_mean=('euros_per_mwh', 'mean'),
    elec_price_max=('euros_per_mwh', 'max'),
    elec_price_min=('euros_per_mwh', 'min'),
    elec_price_std=('euros_per_mwh', 'std'),
).reset_index()

# Also merge hourly price
elec_hourly = elec[['data_block_id', 'hour', 'euros_per_mwh']].rename(
    columns={'euros_per_mwh': 'elec_price_hourly'}
)
train = train.merge(elec_hourly, on=['data_block_id', 'hour'], how='left')
train = train.merge(elec_daily, on='data_block_id', how='left')
print(f"  Hourly elec price matched: {train['elec_price_hourly'].notna().sum()}")
print(f"  Daily elec stats matched: {train['elec_price_mean'].notna().sum()}")

# ============================================================
# 10. Gas price features
# ============================================================
print("\nMerging gas price features...")

gas_features = gas[['data_block_id', 'lowest_price_per_mwh', 'highest_price_per_mwh']].rename(
    columns={'lowest_price_per_mwh': 'gas_price_low',
             'highest_price_per_mwh': 'gas_price_high'}
)
gas_features['gas_price_mid'] = (gas_features['gas_price_low'] + gas_features['gas_price_high']) / 2
train = train.merge(gas_features, on='data_block_id', how='left')
print(f"  Matched: {train['gas_price_low'].notna().sum()}")

# ============================================================
# 11. Interaction features
# ============================================================
print("\nAdding interaction features...")

# Capacity per EIC (solar capacity per metering point)
train['capacity_per_eic'] = train['installed_capacity'] / train['eic_count'].clip(lower=1)

# Solar radiation x installed capacity (for production)
train['solar_x_capacity'] = train['weather_direct_solar_radiation'] * train['installed_capacity']

# Temperature deviation from comfort (heating/cooling demand proxy)
train['temp_deviation'] = np.abs(train['weather_temperature'] - 18)  # 18°C is comfort baseline

print(f"  interaction features added")

# ============================================================
# 12. Summary & Save
# ============================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

feature_cols = [c for c in train.columns if c not in
                ['target', 'row_id', 'datetime', 'date', 'data_block_id']]
print(f"Total features: {len(feature_cols)}")
print(f"Total rows: {len(train)}")
print(f"\nFeature columns:")
for col in sorted(feature_cols):
    n_null = train[col].isnull().sum()
    null_str = f" [{n_null} null]" if n_null > 0 else ""
    print(f"  {col}{null_str}")

# Missing value summary
print(f"\nMissing value summary (features with >0 missing):")
for col in feature_cols:
    n_miss = train[col].isnull().sum()
    if n_miss > 0:
        pct = n_miss / len(train) * 100
        print(f"  {col}: {n_miss} ({pct:.1f}%)")

# Save processed data
out_path = os.path.join(out_dir, "train_processed.parquet")
print(f"\nSaving processed data to {out_path}...")
train.to_parquet(out_path, index=False)
print(f"Saved! File size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")

elapsed = time.time() - t0
print(f"\nTotal feature engineering time: {elapsed:.1f}s")
