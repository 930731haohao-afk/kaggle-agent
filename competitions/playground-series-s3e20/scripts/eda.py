"""Exploratory Data Analysis for playground-series-s3e20"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
train = pd.read_csv("competitions/playground-series-s3e20/data/train.csv")
test = pd.read_csv("competitions/playground-series-s3e20/data/test.csv")

print("=" * 80)
print("FEATURE CATEGORIZATION")
print("=" * 80)

# Categorize features
id_col = 'ID_LAT_LON_YEAR_WEEK'
target = 'emission'

# Separate feature types
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(target)

# Feature groups based on prefix
feature_groups = {}
for col in train.columns:
    if col in [id_col, target]:
        continue
    prefix = col.split('_')[0]
    if prefix not in feature_groups:
        feature_groups[prefix] = []
    feature_groups[prefix].append(col)

print(f"\nFeature groups:")
for prefix, cols in sorted(feature_groups.items()):
    print(f"  {prefix:30s}: {len(cols):3d} features")

print("\n" + "=" * 80)
print("MISSING VALUE ANALYSIS")
print("=" * 80)

# Detailed missing value analysis
missing_train = train.isnull().sum()
missing_pct = (missing_train / len(train)) * 100

# Group by severity
high_missing = missing_pct[missing_pct > 90]
medium_missing = missing_pct[(missing_pct > 10) & (missing_pct <= 90)]
low_missing = missing_pct[(missing_pct > 0) & (missing_pct <= 10)]

print(f"\nHigh missing (>90%): {len(high_missing)} features")
if len(high_missing) > 0:
    print(high_missing)

print(f"\nMedium missing (10-90%): {len(medium_missing)} features")
if len(medium_missing) > 0:
    print(medium_missing)

print(f"\nLow missing (<10%): {len(low_missing)} features")
if len(low_missing) > 0:
    print(low_missing.head(10))

# Features to potentially drop
drop_candidates = high_missing.index.tolist()
print(f"\n** Recommendation: Drop {len(drop_candidates)} features with >90% missing **")

print("\n" + "=" * 80)
print("TARGET ANALYSIS")
print("=" * 80)

print(f"\nTarget: {target}")
print(train[target].describe())

# Skewness
skew = train[target].skew()
print(f"\nSkewness: {skew:.4f}")
print(f"Kurtosis: {train[target].kurtosis():.4f}")

# Log transform check
train['log_emission'] = np.log1p(train[target])
print(f"\nLog-transformed skewness: {train['log_emission'].skew():.4f}")
print("** Log transformation significantly reduces skewness **")

# Zeros
zero_count = (train[target] == 0).sum()
print(f"\nZero emissions: {zero_count} ({zero_count/len(train)*100:.2f}%)")

print("\n" + "=" * 80)
print("TEMPORAL PATTERNS")
print("=" * 80)

# Year distribution
print("\nYear distribution:")
print(train['year'].value_counts().sort_index())
print(test['year'].value_counts().sort_index())

# Week patterns
print(f"\nWeek range in train: {train['week_no'].min()} to {train['week_no'].max()}")
print(f"Week range in test: {test['week_no'].min()} to {test['week_no'].max()}")

# Target by year
print("\nEmission statistics by year:")
print(train.groupby('year')[target].agg(['mean', 'median', 'std', 'count']))

# Target by week (seasonal patterns)
week_stats = train.groupby('week_no')[target].agg(['mean', 'median', 'count'])
print(f"\nEmission by week (first 10 weeks):")
print(week_stats.head(10))

print("\n" + "=" * 80)
print("GEOGRAPHIC PATTERNS")
print("=" * 80)

# Location statistics
print(f"\nUnique locations (lat, lon pairs): {train.groupby(['latitude', 'longitude']).ngroups}")
print(f"\nLatitude range: {train['latitude'].min():.3f} to {train['latitude'].max():.3f}")
print(f"Longitude range: {train['longitude'].min():.3f} to {train['longitude'].max():.3f}")

# Target by location
loc_stats = train.groupby(['latitude', 'longitude'])[target].agg(['mean', 'median', 'std', 'count'])
print(f"\nEmission statistics by location (top 5 by mean):")
print(loc_stats.nlargest(5, 'mean'))

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS (Top features)")
print("=" * 80)

# Remove features with >90% missing for correlation
valid_numeric = [col for col in numeric_cols if col not in drop_candidates]

# Sample correlation with target
correlations = train[valid_numeric + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
print(f"\nTop 15 features correlated with target:")
print(correlations.head(15))

print("\n" + "=" * 80)
print("TRAIN-TEST CONSISTENCY")
print("=" * 80)

# Check for new categories in test
common_cols = set(train.columns) & set(test.columns)
print(f"\nCommon columns: {len(common_cols)}")

# Check numeric feature distributions (sample)
print("\nFeature distribution comparison (sample of 5 features):")
for col in ['latitude', 'longitude', 'week_no'] + valid_numeric[:2]:
    train_mean = train[col].mean()
    test_mean = test[col].mean()
    diff_pct = abs(train_mean - test_mean) / train_mean * 100
    print(f"  {col:40s}: train_mean={train_mean:10.4f}, test_mean={test_mean:10.4f}, diff={diff_pct:6.2f}%")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print(f"""
1. **Missing Values:**
   - DROP: {len(drop_candidates)} features with >90% missing
   - IMPUTE: {len(low_missing)} features with <10% missing (use median/mode)

2. **Target Transformation:**
   - USE log1p transformation (reduces skewness from {skew:.2f} to {train['log_emission'].skew():.2f})
   - Handle {zero_count} zero values carefully

3. **Validation Strategy:**
   - TIME-BASED SPLIT: Train on 2019-2020, validate on 2021
   - DO NOT use random K-fold (temporal leakage!)

4. **Feature Engineering Ideas:**
   - Interaction: latitude × longitude (geographic clusters)
   - Interaction: week_no × latitude (seasonal + location)
   - Cyclical encoding: week_no (sin/cos)
   - Aggregations: mean/std emission by location from training
   - Lag features: previous week's emission by location (if ordered)

5. **Top Features to Focus On:**
   {', '.join(correlations.head(10).index.tolist())}
""")
