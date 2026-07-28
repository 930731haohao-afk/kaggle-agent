"""
Generate a local submission using the example test files and the MockAPI.
Also validates the approach works end-to-end.
Trains on full training data and generates predictions.
"""
import pandas as pd
import numpy as np
import json
import os
import time
import pickle
import lightgbm as lgb

data_dir = "competitions/predict-energy-behavior-of-prosumers/data"
model_dir = "competitions/predict-energy-behavior-of-prosumers/models"
sub_dir = "competitions/predict-energy-behavior-of-prosumers/submissions"
os.makedirs(sub_dir, exist_ok=True)

# ============================================================
# 1. Load processed data & retrain on FULL training set
# ============================================================
print("=" * 70)
print("1. RETRAINING ON FULL DATA")
print("=" * 70)

train = pd.read_parquet(os.path.join(data_dir, "train_processed.parquet"))
print(f"Full train: {train.shape}")

drop_cols = ['target', 'row_id', 'datetime', 'data_block_id']
feature_cols = [c for c in train.columns if c not in drop_cols]

# Filter to rows with valid target and lag_24
mask = train['target'].notna() & train['target_lag_24'].notna()
X_full = train.loc[mask, feature_cols]
y_full = train.loc[mask, 'target']
print(f"Training set after filtering: {X_full.shape}")

lgb_params = {
    'objective': 'mae',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 255,
    'max_depth': -1,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

t0 = time.time()
dtrain = lgb.Dataset(X_full, label=y_full)

# Use 827 iterations (best from validation)
model_full = lgb.train(lgb_params, dtrain, num_boost_round=827)
train_time = time.time() - t0
print(f"Training time: {train_time:.1f}s")

# Save full model
with open(os.path.join(model_dir, 'lgbm_full.pkl'), 'wb') as f:
    pickle.dump(model_full, f)
print("Full model saved")

# ============================================================
# 2. Generate example test predictions
# ============================================================
print("\n" + "=" * 70)
print("2. LOCAL VALIDATION WITH EXAMPLE TEST FILES")
print("=" * 70)

# Load example test files
test = pd.read_csv(os.path.join(data_dir, "example_test_files/test.csv"))
revealed = pd.read_csv(os.path.join(data_dir, "example_test_files/revealed_targets.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "example_test_files/sample_submission.csv"))
client_test = pd.read_csv(os.path.join(data_dir, "example_test_files/client.csv"))
elec_test = pd.read_csv(os.path.join(data_dir, "example_test_files/electricity_prices.csv"))
gas_test = pd.read_csv(os.path.join(data_dir, "example_test_files/gas_prices.csv"))
hist_weather_test = pd.read_csv(os.path.join(data_dir, "example_test_files/historical_weather.csv"))

print(f"Test rows: {len(test)}")
print(f"Revealed targets: {len(revealed)}")
print(f"Sample submission: {len(sample_sub)}")

test['prediction_datetime'] = pd.to_datetime(test['prediction_datetime'])
revealed['datetime'] = pd.to_datetime(revealed['datetime'])

# ============================================================
# 3. Build features for test using revealed targets + train history
# ============================================================
print("\n" + "=" * 70)
print("3. BUILDING TEST FEATURES")
print("=" * 70)

# Combine train targets with revealed targets for lag computation
train_hist = train[['prediction_unit_id', 'is_consumption', 'datetime', 'target']].copy()
revealed_hist = revealed[['prediction_unit_id', 'is_consumption', 'datetime', 'target']].copy()
full_history = pd.concat([train_hist, revealed_hist], ignore_index=True)
full_history = full_history.sort_values(['prediction_unit_id', 'is_consumption', 'datetime'])
full_history = full_history.drop_duplicates(subset=['prediction_unit_id', 'is_consumption', 'datetime'],
                                            keep='last')

# Weather mapping
ws_map = pd.read_csv(os.path.join(data_dir, "weather_station_to_county_mapping.csv"))
ws_map = ws_map.dropna(subset=['county'])
ws_map['county'] = ws_map['county'].astype(int)

weather_cols = ['temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure',
                'cloudcover_total', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
                'windspeed_10m', 'shortwave_radiation', 'direct_solar_radiation',
                'diffuse_radiation']

# Aggregate test historical weather by county
hist_weather_test['datetime'] = pd.to_datetime(hist_weather_test['datetime'])
hw_with_county = hist_weather_test.merge(ws_map[['latitude', 'longitude', 'county']],
                                          on=['latitude', 'longitude'], how='inner')
county_weather_test = hw_with_county.groupby(['datetime', 'county'])[weather_cols].mean().reset_index()

# Electricity prices
elec_test['forecast_date'] = pd.to_datetime(elec_test['forecast_date'])
elec_test['hour'] = elec_test['forecast_date'].dt.hour
elec_daily_test = elec_test.groupby('data_block_id').agg(
    elec_price_mean=('euros_per_mwh', 'mean'),
    elec_price_max=('euros_per_mwh', 'max'),
    elec_price_min=('euros_per_mwh', 'min'),
    elec_price_std=('euros_per_mwh', 'std'),
).reset_index()
elec_hourly_test = elec_test[['data_block_id', 'hour', 'euros_per_mwh']].rename(
    columns={'euros_per_mwh': 'elec_price_hourly'})

# Gas prices
gas_test['forecast_date'] = pd.to_datetime(gas_test['forecast_date'])
gas_features_test = gas_test[['data_block_id', 'lowest_price_per_mwh', 'highest_price_per_mwh']].rename(
    columns={'lowest_price_per_mwh': 'gas_price_low', 'highest_price_per_mwh': 'gas_price_high'})
gas_features_test['gas_price_mid'] = (gas_features_test['gas_price_low'] + gas_features_test['gas_price_high']) / 2

# Client features
client_test['date'] = pd.to_datetime(client_test['date'])

# Now build features for each test row
test_features = test.copy()
test_features['hour'] = test_features['prediction_datetime'].dt.hour
test_features['dayofweek'] = test_features['prediction_datetime'].dt.dayofweek
test_features['month'] = test_features['prediction_datetime'].dt.month
test_features['day'] = test_features['prediction_datetime'].dt.day
test_features['is_weekend'] = (test_features['dayofweek'] >= 5).astype(int)
test_features['hour_sin'] = np.sin(2 * np.pi * test_features['hour'] / 24)
test_features['hour_cos'] = np.cos(2 * np.pi * test_features['hour'] / 24)
test_features['month_sin'] = np.sin(2 * np.pi * test_features['month'] / 12)
test_features['month_cos'] = np.cos(2 * np.pi * test_features['month'] / 12)
test_features['dayofweek_sin'] = np.sin(2 * np.pi * test_features['dayofweek'] / 7)
test_features['dayofweek_cos'] = np.cos(2 * np.pi * test_features['dayofweek'] / 7)

# Lag features: look up from full_history
print("Computing lag features for test...")
for _, row in test_features.iterrows():
    uid = row['prediction_unit_id']
    ic = row['is_consumption']
    dt = row['prediction_datetime']

    history = full_history[(full_history['prediction_unit_id'] == uid) &
                           (full_history['is_consumption'] == ic) &
                           (full_history['datetime'] < dt)].sort_values('datetime')

    if len(history) == 0:
        continue

    idx = test_features.index[test_features['row_id'] == row['row_id']]

    for lag, hours in [(1, 1), (2, 2), (3, 3), (24, 24), (48, 48), (168, 168)]:
        target_dt = dt - pd.Timedelta(hours=hours)
        match = history[history['datetime'] == target_dt]
        if len(match) > 0:
            test_features.loc[idx, f'target_lag_{lag}'] = match.iloc[-1]['target']

    # Rolling features from history tail
    recent = history.tail(168)
    test_features.loc[idx, 'target_roll_mean_24'] = recent.tail(24)['target'].mean()
    test_features.loc[idx, 'target_roll_std_24'] = recent.tail(24)['target'].std()
    test_features.loc[idx, 'target_roll_mean_168'] = recent['target'].mean()
    test_features.loc[idx, 'target_roll_std_168'] = recent['target'].std()

    # Same hour rolling mean
    same_hour = history[history['datetime'].dt.hour == dt.hour].tail(7)
    test_features.loc[idx, 'target_roll_mean_same_hour_7d'] = same_hour['target'].mean()

    # Diffs
    if f'target_lag_24' in test_features.columns and f'target_lag_48' in test_features.columns:
        lag24 = test_features.loc[idx, 'target_lag_24'].values
        lag48 = test_features.loc[idx, 'target_lag_48'].values
        lag168 = test_features.loc[idx, 'target_lag_168'].values if 'target_lag_168' in test_features.columns else np.nan
        test_features.loc[idx, 'target_diff_24'] = lag24 - lag48
        test_features.loc[idx, 'target_diff_168'] = lag24 - lag168

    test_features.loc[idx, 'target_lag_same_hour_last_week'] = test_features.loc[idx, 'target_lag_168'] if 'target_lag_168' in test_features.columns else np.nan

# Merge client
test_features = test_features.merge(
    client_test[['product_type', 'county', 'is_business', 'data_block_id', 'eic_count', 'installed_capacity']],
    on=['product_type', 'county', 'is_business', 'data_block_id'],
    how='left'
)

# Merge weather
weather_rename = {col: f'weather_{col}' for col in weather_cols}
county_weather_test_renamed = county_weather_test.rename(columns=weather_rename)
test_features = test_features.merge(
    county_weather_test_renamed,
    left_on=['prediction_datetime', 'county'],
    right_on=['datetime', 'county'],
    how='left'
)
if 'datetime' in test_features.columns:
    test_features = test_features.drop(columns=['datetime'])

# Merge prices
test_features = test_features.merge(elec_hourly_test, on=['data_block_id', 'hour'], how='left')
test_features = test_features.merge(elec_daily_test, on='data_block_id', how='left')
test_features = test_features.merge(gas_features_test, on='data_block_id', how='left')

# Interaction features
test_features['capacity_per_eic'] = test_features['installed_capacity'] / test_features['eic_count'].clip(lower=1)
test_features['solar_x_capacity'] = test_features.get('weather_direct_solar_radiation', 0) * test_features.get('installed_capacity', 0)
test_features['temp_deviation'] = np.abs(test_features.get('weather_temperature', 18) - 18)

# ============================================================
# 4. Generate predictions
# ============================================================
print("\n" + "=" * 70)
print("4. GENERATING PREDICTIONS")
print("=" * 70)

# Ensure all feature columns exist (fill missing with NaN)
for col in feature_cols:
    if col not in test_features.columns:
        test_features[col] = np.nan

# Drop non-feature columns
test_X = test_features[feature_cols]

# Handle the currently_scored column (not a feature)
if 'currently_scored' in test_X.columns:
    test_X = test_X.drop(columns=['currently_scored'])

predictions = model_full.predict(test_X)
predictions = np.clip(predictions, 0, None)

# ============================================================
# 5. Create submission
# ============================================================
print("\n" + "=" * 70)
print("5. SUBMISSION")
print("=" * 70)

from datetime import datetime as dt
timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
sub_file = os.path.join(sub_dir, f"lgbm_submission_{timestamp}.csv")

submission = pd.DataFrame({
    'row_id': test_features['row_id'],
    'target': predictions
})

# Validate against sample submission
print(f"Submission shape: {submission.shape} (expected: {sample_sub.shape})")
print(f"Columns: {list(submission.columns)}")
print(f"Target stats: mean={predictions.mean():.2f}, min={predictions.min():.2f}, "
      f"max={predictions.max():.2f}")
print(f"Row_id range: {submission['row_id'].min()} - {submission['row_id'].max()}")
print(f"\nPrediction distribution:")
print(submission['target'].describe())

submission.to_csv(sub_file, index=False)
print(f"\nSubmission saved to {sub_file}")

# Compare with revealed targets (for the overlap period)
test_with_revealed = test_features.merge(
    revealed[['row_id', 'target']].rename(columns={'target': 'actual'}),
    on='row_id', how='left'
)
if test_with_revealed['actual'].notna().sum() > 0:
    matched = test_with_revealed.dropna(subset=['actual'])
    from sklearn.metrics import mean_absolute_error
    local_mae = mean_absolute_error(matched['actual'], predictions[:len(matched)])
    print(f"\nLocal MAE vs revealed targets: {local_mae:.4f}")

print("\nNote: This is a CODE COMPETITION. Final submission requires a Kaggle notebook.")
print("The local submission file validates the pipeline works end-to-end.")
