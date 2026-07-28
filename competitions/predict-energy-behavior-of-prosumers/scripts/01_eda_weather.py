"""
EDA Part 2: Weather data analysis.
Loads weather data in chunks to manage memory, aggregates by county.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

data_dir = "competitions/predict-energy-behavior-of-prosumers/data"
plot_dir = "competitions/predict-energy-behavior-of-prosumers/scripts/plots"
os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# 1. Weather Station to County Mapping
# ============================================================
print("=" * 70)
print("1. WEATHER STATION TO COUNTY MAPPING")
print("=" * 70)

ws_map = pd.read_csv(os.path.join(data_dir, "weather_station_to_county_mapping.csv"))
print(f"Total grid points: {len(ws_map)}")
print(f"Grid points mapped to counties: {ws_map['county'].notna().sum()}")
print(f"Unmapped grid points: {ws_map['county'].isna().sum()}")
print(f"\nCounty coverage:")
county_stations = ws_map.dropna(subset=['county']).groupby('county_name').size()
print(county_stations.to_string())

# ============================================================
# 2. Historical Weather Analysis (sampled)
# ============================================================
print("\n" + "=" * 70)
print("2. HISTORICAL WEATHER ANALYSIS")
print("=" * 70)

# Load a manageable chunk to understand structure
print("Loading historical weather (first 500K rows)...")
hist_weather = pd.read_csv(os.path.join(data_dir, "historical_weather.csv"), nrows=500000)
hist_weather['datetime'] = pd.to_datetime(hist_weather['datetime'])

print(f"Shape: {hist_weather.shape}")
print(f"Date range (sample): {hist_weather['datetime'].min()} to {hist_weather['datetime'].max()}")
print(f"Grid points: {hist_weather.groupby(['latitude', 'longitude']).ngroups}")

# Missing values
print(f"\nMissing values:")
for col in hist_weather.columns:
    n_miss = hist_weather[col].isnull().sum()
    if n_miss > 0:
        print(f"  {col}: {n_miss} ({n_miss/len(hist_weather)*100:.2f}%)")
if hist_weather.isnull().sum().sum() == 0:
    print("  None!")

# Weather statistics
weather_cols = ['temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure',
                'cloudcover_total', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
                'windspeed_10m', 'winddirection_10m', 'shortwave_radiation',
                'direct_solar_radiation', 'diffuse_radiation']
print(f"\nWeather feature statistics (sample):")
print(hist_weather[weather_cols].describe().to_string())

# Correlation between weather features
print(f"\nWeather feature correlations (|r| > 0.5):")
corr = hist_weather[weather_cols].corr()
for i in range(len(weather_cols)):
    for j in range(i+1, len(weather_cols)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            print(f"  {weather_cols[i]} <-> {weather_cols[j]}: {r:.3f}")

# Aggregate weather by county for the sample
# Merge with county mapping
hist_with_county = hist_weather.merge(
    ws_map[['latitude', 'longitude', 'county']].dropna(),
    on=['latitude', 'longitude'], how='inner'
)
print(f"\nHistorical weather rows matched to counties: {len(hist_with_county)} / {len(hist_weather)}")

# County-level weather aggregation
county_weather = hist_with_county.groupby(['datetime', 'county'])[weather_cols].mean().reset_index()
print(f"County-level weather: {county_weather.shape}")

# ============================================================
# 3. Forecast Weather Analysis (sampled)
# ============================================================
print("\n" + "=" * 70)
print("3. FORECAST WEATHER ANALYSIS")
print("=" * 70)

print("Loading forecast weather (first 500K rows)...")
fc_weather = pd.read_csv(os.path.join(data_dir, "forecast_weather.csv"), nrows=500000)
fc_weather['forecast_datetime'] = pd.to_datetime(fc_weather['forecast_datetime'])
fc_weather['origin_datetime'] = pd.to_datetime(fc_weather['origin_datetime'])

print(f"Shape: {fc_weather.shape}")
print(f"Forecast date range (sample): {fc_weather['forecast_datetime'].min()} to "
      f"{fc_weather['forecast_datetime'].max()}")
print(f"Origin date range (sample): {fc_weather['origin_datetime'].min()} to "
      f"{fc_weather['origin_datetime'].max()}")
print(f"Hours ahead range: {fc_weather['hours_ahead'].min()} to {fc_weather['hours_ahead'].max()}")
print(f"Unique hours_ahead: {sorted(fc_weather['hours_ahead'].unique())[:10]}...")

fc_cols = ['temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low',
           'cloudcover_mid', 'cloudcover_total', '10_metre_u_wind_component',
           '10_metre_v_wind_component', 'direct_solar_radiation',
           'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation']

print(f"\nForecast weather statistics (sample):")
print(fc_weather[fc_cols].describe().to_string())

# Missing values in forecast
print(f"\nForecast weather missing values:")
for col in fc_weather.columns:
    n_miss = fc_weather[col].isnull().sum()
    if n_miss > 0:
        print(f"  {col}: {n_miss} ({n_miss/len(fc_weather)*100:.2f}%)")
if fc_weather.isnull().sum().sum() == 0:
    print("  None!")

# How many forecast origins per day?
fc_origins_per_block = fc_weather.groupby('data_block_id')['origin_datetime'].nunique()
print(f"\nForecast origins per data_block_id:")
print(f"  Min: {fc_origins_per_block.min()}, Max: {fc_origins_per_block.max()}, "
      f"Mean: {fc_origins_per_block.mean():.1f}")

# ============================================================
# 4. Weather-Target Relationship (needs train data)
# ============================================================
print("\n" + "=" * 70)
print("4. WEATHER-TARGET RELATIONSHIP")
print("=" * 70)

# Load train for a specific period that overlaps with our weather sample
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
train['datetime'] = pd.to_datetime(train['datetime'])

# Use the county-aggregated historical weather
# Match on datetime and county
train_with_weather = train.merge(
    county_weather,
    left_on=['datetime', 'county'],
    right_on=['datetime', 'county'],
    how='inner'
)
print(f"Matched train+weather rows: {len(train_with_weather)}")

if len(train_with_weather) > 0:
    # Correlations between weather and target
    print(f"\nCorrelation with target (production, is_consumption=0):")
    prod = train_with_weather[train_with_weather['is_consumption'] == 0]
    for col in weather_cols:
        if col in prod.columns:
            r = prod[['target', col]].corr().iloc[0, 1]
            if abs(r) > 0.05:
                print(f"  {col:<30}: {r:>8.4f}")

    print(f"\nCorrelation with target (consumption, is_consumption=1):")
    cons = train_with_weather[train_with_weather['is_consumption'] == 1]
    for col in weather_cols:
        if col in cons.columns:
            r = cons[['target', col]].corr().iloc[0, 1]
            if abs(r) > 0.05:
                print(f"  {col:<30}: {r:>8.4f}")

    # Key weather plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Solar radiation vs production
    if 'direct_solar_radiation' in prod.columns:
        sample = prod.sample(min(10000, len(prod)), random_state=42)
        axes[0, 0].scatter(sample['direct_solar_radiation'], sample['target'],
                          alpha=0.1, s=1)
        axes[0, 0].set_xlabel('Direct Solar Radiation')
        axes[0, 0].set_ylabel('Production Target')
        axes[0, 0].set_title('Solar Radiation vs Production')

    # Temperature vs consumption
    sample = cons.sample(min(10000, len(cons)), random_state=42)
    axes[0, 1].scatter(sample['temperature'], sample['target'],
                      alpha=0.1, s=1)
    axes[0, 1].set_xlabel('Temperature')
    axes[0, 1].set_ylabel('Consumption Target')
    axes[0, 1].set_title('Temperature vs Consumption')

    # Cloud cover vs production
    if 'cloudcover_total' in prod.columns:
        sample = prod.sample(min(10000, len(prod)), random_state=42)
        axes[1, 0].scatter(sample['cloudcover_total'], sample['target'],
                          alpha=0.1, s=1)
        axes[1, 0].set_xlabel('Cloud Cover Total')
        axes[1, 0].set_ylabel('Production Target')
        axes[1, 0].set_title('Cloud Cover vs Production')

    # Windspeed vs production
    if 'windspeed_10m' in prod.columns:
        sample = prod.sample(min(10000, len(prod)), random_state=42)
        axes[1, 1].scatter(sample['windspeed_10m'], sample['target'],
                          alpha=0.1, s=1)
        axes[1, 1].set_xlabel('Wind Speed 10m')
        axes[1, 1].set_ylabel('Production Target')
        axes[1, 1].set_title('Wind Speed vs Production')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '05_weather_target.png'), dpi=100)
    plt.close()
    print("Saved weather-target plots")

# ============================================================
# 5. Full Weather File Row Counts
# ============================================================
print("\n" + "=" * 70)
print("5. FULL WEATHER FILE SIZE ESTIMATES")
print("=" * 70)

# Count rows in historical weather efficiently
import subprocess
result = subprocess.run(['wc', '-l', os.path.join(data_dir, 'historical_weather.csv')],
                       capture_output=True, text=True)
hw_lines = int(result.stdout.strip().split()[0]) - 1  # subtract header
print(f"Historical weather total rows: {hw_lines:,}")

result = subprocess.run(['wc', '-l', os.path.join(data_dir, 'forecast_weather.csv')],
                       capture_output=True, text=True)
fw_lines = int(result.stdout.strip().split()[0]) - 1
print(f"Forecast weather total rows: {fw_lines:,}")

# Estimate structure
n_stations = 112  # lat/lon grid points
print(f"\nHistorical weather: {hw_lines:,} rows / {n_stations} stations "
      f"≈ {hw_lines // n_stations:,} hours ≈ {hw_lines // n_stations // 24:,} days")
print(f"Forecast weather: {fw_lines:,} rows (much larger due to multiple forecasts per period)")

print("\nDone with weather EDA.")
