"""Initial data inspection for ICDAR 2013 Stroke Recovery competition."""
import pandas as pd
import numpy as np
import os

data_dir = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"

# === File inventory ===
print("=" * 60)
print("FILE INVENTORY")
print("=" * 60)
for f in sorted(os.listdir(data_dir)):
    if os.path.isfile(os.path.join(data_dir, f)):
        size = os.path.getsize(os.path.join(data_dir, f))
        print(f"  {f}: {size / 1024:.1f} KB ({size / 1024**2:.2f} MB)")

# === Train data ===
print("\n" + "=" * 60)
print("TRAIN DATA")
print("=" * 60)
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
print(f"Shape: {train.shape}")
print(f"\nColumns: {list(train.columns)}")
print(f"\nDtypes:\n{train.dtypes}")
print(f"\nFirst 10 rows:\n{train.head(10)}")
print(f"\nDescribe:\n{train.describe()}")
print(f"\nMissing values:\n{train.isnull().sum()}")

# === Test data ===
print("\n" + "=" * 60)
print("TEST DATA")
print("=" * 60)
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
print(f"Shape: {test.shape}")
print(f"\nColumns: {list(test.columns)}")
print(f"\nDtypes:\n{test.dtypes}")
print(f"\nFirst 10 rows:\n{test.head(10)}")
print(f"\nDescribe:\n{test.describe()}")
print(f"\nMissing values:\n{test.isnull().sum()}")

# === Example submission ===
print("\n" + "=" * 60)
print("EXAMPLE SUBMISSION")
print("=" * 60)
sub = pd.read_csv(os.path.join(data_dir, "example_submission.csv"))
print(f"Shape: {sub.shape}")
print(f"\nColumns: {list(sub.columns)}")
print(f"\nDtypes:\n{sub.dtypes}")
print(f"\nFirst 10 rows:\n{sub.head(10)}")

# === Submission example (second file) ===
print("\n" + "=" * 60)
print("SUBMISSION EXAMPLE (second file)")
print("=" * 60)
sub2 = pd.read_csv(os.path.join(data_dir, "submission_example.csv"))
print(f"Shape: {sub2.shape}")
print(f"\nColumns: {list(sub2.columns)}")
print(f"\nDtypes:\n{sub2.dtypes}")
print(f"\nFirst 10 rows:\n{sub2.head(10)}")

# === Unnormalized data ===
print("\n" + "=" * 60)
print("UNNORMALIZED DATA")
print("=" * 60)
unnorm = pd.read_csv(os.path.join(data_dir, "unnormalized_data.csv"))
print(f"Shape: {unnorm.shape}")
print(f"\nColumns: {list(unnorm.columns)}")
print(f"\nDtypes:\n{unnorm.dtypes}")
print(f"\nFirst 10 rows:\n{unnorm.head(10)}")
print(f"\nDescribe:\n{unnorm.describe()}")

# === Analyze structure ===
print("\n" + "=" * 60)
print("STRUCTURAL ANALYSIS")
print("=" * 60)

# How many unique samples in train?
if 'sample' in train.columns:
    print(f"\nUnique samples in train: {train['sample'].nunique()}")
    print(f"Samples per image: {train.groupby('sample').size().describe()}")
elif 'id' in train.columns:
    print(f"\nUnique IDs in train: {train['id'].nunique()}")

# Check test structure
if 'sample' in test.columns:
    print(f"\nUnique samples in test: {test['sample'].nunique()}")
elif 'id' in test.columns:
    print(f"\nUnique IDs in test: {test['id'].nunique()}")

# Check if train has image mapping
for col in train.columns:
    if train[col].dtype == 'object':
        print(f"\nString column '{col}': {train[col].nunique()} unique values")
        print(f"  Examples: {train[col].head(5).tolist()}")
