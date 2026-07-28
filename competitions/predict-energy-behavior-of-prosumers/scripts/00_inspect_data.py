"""Initial data inspection for predict-energy-behavior-of-prosumers."""
import pandas as pd
import numpy as np
import os
import json

data_dir = "competitions/predict-energy-behavior-of-prosumers/data"

def inspect_csv(filepath, name, nrows=None):
    """Inspect a CSV file and print summary."""
    print(f"\n{'=' * 70}")
    print(f"{name}: {filepath}")
    print(f"{'=' * 70}")

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    df = pd.read_csv(filepath, nrows=nrows)
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        n_null = df[col].isnull().sum()
        null_pct = n_null / len(df) * 100
        n_unique = df[col].nunique()
        dtype = df[col].dtype
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
        null_str = f" [{n_null} null ({null_pct:.1f}%)]" if n_null > 0 else ""
        print(f"  {col:<35} {str(dtype):<10} unique={n_unique:<8}{null_str}  sample: {sample}")

    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    if nrows is None:
        print(f"\nDescribe (numeric):")
        desc = df.describe()
        print(desc.to_string())

    return df

# ============================================================
# Main data files
# ============================================================

# 1. Train
train = inspect_csv(os.path.join(data_dir, "train.csv"), "TRAIN")

# 2. Client
client = inspect_csv(os.path.join(data_dir, "client.csv"), "CLIENT")

# 3. Electricity prices
elec = inspect_csv(os.path.join(data_dir, "electricity_prices.csv"), "ELECTRICITY PRICES")

# 4. Gas prices
gas = inspect_csv(os.path.join(data_dir, "gas_prices.csv"), "GAS PRICES")

# 5. Weather station mapping
ws_map = inspect_csv(os.path.join(data_dir, "weather_station_to_county_mapping.csv"),
                     "WEATHER STATION TO COUNTY MAPPING")

# 6. Historical weather (large — sample)
hist_weather = inspect_csv(os.path.join(data_dir, "historical_weather.csv"),
                           "HISTORICAL WEATHER (first 100K rows)", nrows=100000)

# 7. Forecast weather (very large — sample)
fc_weather = inspect_csv(os.path.join(data_dir, "forecast_weather.csv"),
                         "FORECAST WEATHER (first 100K rows)", nrows=100000)

# ============================================================
# Example test files
# ============================================================
print("\n\n" + "#" * 70)
print("EXAMPLE TEST FILES")
print("#" * 70)

for f in sorted(os.listdir(os.path.join(data_dir, "example_test_files"))):
    filepath = os.path.join(data_dir, "example_test_files", f)
    if f.endswith('.csv'):
        inspect_csv(filepath, f"EXAMPLE: {f}")

# ============================================================
# Key analysis on train
# ============================================================
print("\n\n" + "#" * 70)
print("DETAILED TRAIN ANALYSIS")
print("#" * 70)

print(f"\nDate range: {train['datetime'].min()} to {train['datetime'].max()}")
print(f"\nTarget ('target') stats:")
print(train['target'].describe())
print(f"\nTarget distribution:")
print(f"  Zero values: {(train['target'] == 0).sum()} ({(train['target'] == 0).mean()*100:.1f}%)")
print(f"  Negative values: {(train['target'] < 0).sum()} ({(train['target'] < 0).mean()*100:.1f}%)")
print(f"  Positive values: {(train['target'] > 0).sum()} ({(train['target'] > 0).mean()*100:.1f}%)")

print(f"\nis_consumption distribution:")
print(train['is_consumption'].value_counts())

print(f"\nTarget by is_consumption:")
for ic in train['is_consumption'].unique():
    subset = train[train['is_consumption'] == ic]['target']
    print(f"  is_consumption={ic}: mean={subset.mean():.4f}, median={subset.median():.4f}, "
          f"std={subset.std():.4f}, min={subset.min():.4f}, max={subset.max():.4f}")

print(f"\nCounty distribution:")
print(train['county'].value_counts().sort_index())

print(f"\nProduct type distribution:")
print(train['product_type'].value_counts().sort_index())

print(f"\nUnique prediction_unit_id: {train['prediction_unit_id'].nunique()}")
print(f"Rows per prediction_unit_id (approx): {len(train) / train['prediction_unit_id'].nunique():.0f}")

# Check row_id structure
print(f"\nrow_id range: {train['row_id'].min()} to {train['row_id'].max()}")
print(f"row_id is sequential: {(train['row_id'].diff().dropna() == 1).all()}")

# County-product_type combinations
print(f"\nCounty x product_type x is_consumption combinations:")
combos = train.groupby(['county', 'product_type', 'is_consumption']).size().reset_index(name='count')
print(f"  Total unique: {len(combos)}")
print(f"  Count range: {combos['count'].min()} to {combos['count'].max()}")
