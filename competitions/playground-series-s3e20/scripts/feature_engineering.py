"""Feature Engineering for playground-series-s3e20"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv("competitions/playground-series-s3e20/data/train.csv")
test = pd.read_csv("competitions/playground-series-s3e20/data/test.csv")

print(f"Original train shape: {train.shape}")
print(f"Original test shape: {test.shape}")

# Store target and ID
target = train['emission'].copy()
train_id = train['ID_LAT_LON_YEAR_WEEK'].copy()
test_id = test['ID_LAT_LON_YEAR_WEEK'].copy()

# Drop ID column
train = train.drop('ID_LAT_LON_YEAR_WEEK', axis=1)
test = test.drop('ID_LAT_LON_YEAR_WEEK', axis=1)

print("\n" + "=" * 80)
print("1. DROP HIGH-MISSING FEATURES")
print("=" * 80)

# Drop features with >90% missing
high_missing = train.columns[train.isnull().sum() / len(train) > 0.9].tolist()
print(f"Dropping {len(high_missing)} features: {high_missing}")

train = train.drop(high_missing, axis=1)
test = test.drop(high_missing, axis=1)

print("\n" + "=" * 80)
print("2. HANDLE MEDIUM-MISSING FEATURES")
print("=" * 80)

# For features with 10-90% missing, create missing indicator and impute
medium_missing = train.columns[(train.isnull().sum() / len(train) > 0.1) &
                                (train.isnull().sum() / len(train) <= 0.9)].tolist()

print(f"Creating missing indicators for {len(medium_missing)} features")
for col in medium_missing:
    if col != 'emission':
        train[f'{col}_missing'] = train[col].isnull().astype(int)
        test[f'{col}_missing'] = test[col].isnull().astype(int)

print("\n" + "=" * 80)
print("3. IMPUTE MISSING VALUES")
print("=" * 80)

# Impute all remaining missing values with median
for col in train.columns:
    if col != 'emission' and train[col].dtype in [np.float64, np.int64]:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        if col in test.columns:
            test[col] = test[col].fillna(median_val)

print("Missing values imputed with median")

print("\n" + "=" * 80)
print("4. CREATE NEW FEATURES")
print("=" * 80)

# Geographic features
print("Creating geographic features...")
train['lat_lon_interaction'] = train['latitude'] * train['longitude']
test['lat_lon_interaction'] = test['latitude'] * test['longitude']

train['lat_squared'] = train['latitude'] ** 2
test['lat_squared'] = test['latitude'] ** 2

train['lon_squared'] = train['longitude'] ** 2
test['lon_squared'] = test['longitude'] ** 2

# Cyclical encoding for week
print("Creating cyclical week features...")
train['week_sin'] = np.sin(2 * np.pi * train['week_no'] / 52)
train['week_cos'] = np.cos(2 * np.pi * train['week_no'] / 52)
test['week_sin'] = np.sin(2 * np.pi * test['week_no'] / 52)
test['week_cos'] = np.cos(2 * np.pi * test['week_no'] / 52)

# Season (approximate for Rwanda which is near equator)
train['season'] = ((train['week_no'] // 13) % 4)  # 0-3 for quarters
test['season'] = ((test['week_no'] // 13) % 4)

# Location-based aggregates (from training data only)
print("Creating location-based features...")
loc_stats = train.groupby(['latitude', 'longitude'])['emission'].agg([
    'mean', 'median', 'std', 'min', 'max', 'count'
]).reset_index()
loc_stats.columns = ['latitude', 'longitude', 'loc_emission_mean', 'loc_emission_median',
                     'loc_emission_std', 'loc_emission_min', 'loc_emission_max', 'loc_count']

train = train.merge(loc_stats, on=['latitude', 'longitude'], how='left')
test = test.merge(loc_stats, on=['latitude', 'longitude'], how='left')

# Year-week interaction
train['year_week'] = train['year'] * 100 + train['week_no']
test['year_week'] = test['year'] * 100 + test['week_no']

# Temporal features
train['weeks_since_start'] = (train['year'] - 2019) * 52 + train['week_no']
test['weeks_since_start'] = (test['year'] - 2019) * 52 + test['week_no']

print(f"\nCreated {train.shape[1] - len(train.columns.intersection(pd.read_csv('competitions/playground-series-s3e20/data/train.csv').columns)) + len(high_missing) + 1} new features")

print("\n" + "=" * 80)
print("5. SAVE PROCESSED DATA")
print("=" * 80)

# Add back ID and target
train['ID_LAT_LON_YEAR_WEEK'] = train_id
train['emission'] = target
test['ID_LAT_LON_YEAR_WEEK'] = test_id

# Reorder columns (ID first, target last for train)
train_cols = ['ID_LAT_LON_YEAR_WEEK'] + [col for col in train.columns if col not in ['ID_LAT_LON_YEAR_WEEK', 'emission']] + ['emission']
test_cols = ['ID_LAT_LON_YEAR_WEEK'] + [col for col in test.columns if col != 'ID_LAT_LON_YEAR_WEEK']

train = train[train_cols]
test = test[test_cols]

# Save
train.to_csv("competitions/playground-series-s3e20/data/train_processed.csv", index=False)
test.to_csv("competitions/playground-series-s3e20/data/test_processed.csv", index=False)

print(f"Final train shape: {train.shape}")
print(f"Final test shape: {test.shape}")
print(f"\nFeature count: {train.shape[1] - 2} (excluding ID and target)")
print("\nProcessed data saved:")
print("  - train_processed.csv")
print("  - test_processed.csv")

# Validation check
print("\n" + "=" * 80)
print("VALIDATION CHECKS")
print("=" * 80)

print(f"Train columns: {train.shape[1]}")
print(f"Test columns: {test.shape[1]}")
print(f"Columns in train but not test: {set(train.columns) - set(test.columns)}")
print(f"Columns in test but not train: {set(test.columns) - set(train.columns)}")

print(f"\nMissing values in train: {train.isnull().sum().sum()}")
print(f"Missing values in test: {test.isnull().sum().sum()}")

if train.isnull().sum().sum() > 0:
    print("\nWARNING: Missing values still present!")
    print(train.isnull().sum()[train.isnull().sum() > 0])
