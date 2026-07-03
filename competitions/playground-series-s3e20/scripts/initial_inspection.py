"""Initial data inspection for playground-series-s3e20"""
import pandas as pd
import numpy as np
import os

data_dir = "competitions/playground-series-s3e20/data"

# List all files
print("=" * 80)
print("DATA FILES")
print("=" * 80)
for f in sorted(os.listdir(data_dir)):
    size = os.path.getsize(os.path.join(data_dir, f))
    print(f"{f:30s} {size / 1024 / 1024:8.2f} MB")

# Load datasets
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("\n" + "=" * 80)
print("TRAIN SET")
print("=" * 80)
print(f"Shape: {train.shape}")
print(f"\nColumn dtypes:")
print(train.dtypes)
print(f"\nFirst 5 rows:")
print(train.head())
print(f"\nBasic statistics:")
print(train.describe())

print("\n" + "=" * 80)
print("TEST SET")
print("=" * 80)
print(f"Shape: {test.shape}")
print(f"\nFirst 5 rows:")
print(test.head())

print("\n" + "=" * 80)
print("MISSING VALUES (Train)")
print("=" * 80)
missing = train.isnull().sum()
missing_pct = (missing / len(train)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percent': missing_pct
})
print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))
if missing.sum() == 0:
    print("No missing values!")

print("\n" + "=" * 80)
print("TARGET ANALYSIS")
print("=" * 80)
# Try to identify target column (usually last column or named target/label)
target_candidates = [col for col in train.columns if col.lower() in ['target', 'label', 'y']]
if not target_candidates:
    # Check if last column is numeric and not in test
    last_col = train.columns[-1]
    if last_col not in test.columns:
        target_candidates = [last_col]

if target_candidates:
    target_col = target_candidates[0]
    print(f"Identified target column: '{target_col}'")
    print(f"\nTarget dtype: {train[target_col].dtype}")
    print(f"Unique values: {train[target_col].nunique()}")
    print(f"\nTarget distribution:")
    print(train[target_col].describe())

    # Check if classification or regression
    if train[target_col].dtype == 'object' or train[target_col].nunique() < 20:
        print(f"\nValue counts:")
        print(train[target_col].value_counts().head(20))
else:
    print("Could not automatically identify target column")
    print("Columns in train but not in test:")
    print(set(train.columns) - set(test.columns))

print("\n" + "=" * 80)
print("SAMPLE SUBMISSION")
print("=" * 80)
print(f"Shape: {sample_sub.shape}")
print(f"Columns: {list(sample_sub.columns)}")
print(f"\nFirst 5 rows:")
print(sample_sub.head())
