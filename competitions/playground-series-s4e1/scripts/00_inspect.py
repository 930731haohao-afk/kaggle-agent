"""Initial inspection for playground-series-s4e1."""
import pandas as pd
import numpy as np
import os

data_dir = "competitions/playground-series-s4e1/data"

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("=" * 70)
print("PLAYGROUND SERIES S4E1 - INITIAL INSPECTION")
print("=" * 70)

print(f"\nTrain: {train.shape}")
print(f"Test: {test.shape}")
print(f"Sample submission: {sample_sub.shape}")

print(f"\n--- TRAIN COLUMNS ---")
for col in train.columns:
    dtype = train[col].dtype
    n_unique = train[col].nunique()
    n_null = train[col].isnull().sum()
    null_str = f" [{n_null} null]" if n_null > 0 else ""
    sample = train[col].dropna().iloc[0] if len(train[col].dropna()) > 0 else "N/A"
    print(f"  {col:<25} {str(dtype):<10} unique={n_unique:<8}{null_str}  sample: {sample}")

print(f"\n--- TEST COLUMNS ---")
for col in test.columns:
    dtype = test[col].dtype
    n_unique = test[col].nunique()
    print(f"  {col:<25} {str(dtype):<10} unique={n_unique}")

print(f"\n--- SAMPLE SUBMISSION ---")
print(sample_sub.head())
print(f"Columns: {list(sample_sub.columns)}")

# Identify target
train_only_cols = set(train.columns) - set(test.columns)
print(f"\nTarget column(s) (in train but not test): {train_only_cols}")

print(f"\n--- TRAIN FIRST 5 ROWS ---")
print(train.head().to_string())

print(f"\n--- TRAIN DESCRIBE ---")
print(train.describe().to_string())

# Target analysis
for col in train_only_cols:
    if col == 'id':
        continue
    target = train[col]
    print(f"\n--- TARGET: {col} ---")
    print(f"  dtype: {target.dtype}")
    print(f"  unique: {target.nunique()}")
    print(f"  nulls: {target.isnull().sum()}")
    if target.dtype in ['int64', 'float64']:
        print(f"  mean: {target.mean():.4f}")
        print(f"  std: {target.std():.4f}")
        print(f"  min: {target.min()}, max: {target.max()}")
        print(f"  skew: {target.skew():.4f}")
    if target.nunique() <= 20:
        print(f"  value_counts:\n{target.value_counts().sort_index()}")

print(f"\n--- MISSING VALUES ---")
print(f"Train: {train.isnull().sum().sum()} total missing")
print(f"Test: {test.isnull().sum().sum()} total missing")
for col in train.columns:
    n = train[col].isnull().sum()
    if n > 0:
        print(f"  Train {col}: {n} ({n/len(train)*100:.1f}%)")
for col in test.columns:
    n = test[col].isnull().sum()
    if n > 0:
        print(f"  Test {col}: {n} ({n/len(test)*100:.1f}%)")
