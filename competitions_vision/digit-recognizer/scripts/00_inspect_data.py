"""Initial data inspection for digit-recognizer competition."""
import pandas as pd
import os
import numpy as np

data_dir = "competitions/digit-recognizer/data"

# List all files with sizes
print("=" * 60)
print("FILE INVENTORY")
print("=" * 60)
for f in sorted(os.listdir(data_dir)):
    size = os.path.getsize(os.path.join(data_dir, f))
    print(f"  {f}: {size / (1024*1024):.1f} MB")

# Load data
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

# Train set overview
print("\n" + "=" * 60)
print("TRAIN SET")
print("=" * 60)
print(f"Shape: {train.shape}")
print(f"Columns: label + {train.shape[1] - 1} pixel columns (pixel0..pixel783)")
print(f"\nFirst 5 rows (label + first 10 pixels):")
print(train[['label'] + [f'pixel{i}' for i in range(10)]].head())
print(f"\nDtypes summary: {dict(train.dtypes.value_counts())}")
print(f"Memory usage: {train.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")

# Test set overview
print("\n" + "=" * 60)
print("TEST SET")
print("=" * 60)
print(f"Shape: {test.shape}")
print(f"Columns: {test.shape[1]} pixel columns (pixel0..pixel783)")
print(f"Memory usage: {test.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")

# Target distribution
print("\n" + "=" * 60)
print("TARGET DISTRIBUTION (label)")
print("=" * 60)
label_counts = train['label'].value_counts().sort_index()
for digit, count in label_counts.items():
    pct = count / len(train) * 100
    bar = '#' * int(pct * 2)
    print(f"  {digit}: {count:5d} ({pct:5.2f}%) {bar}")
print(f"\nTotal samples: {len(train)}")
print(f"Min class: {label_counts.min()} ({label_counts.idxmin()})")
print(f"Max class: {label_counts.max()} ({label_counts.idxmax()})")
print(f"Imbalance ratio: {label_counts.max() / label_counts.min():.2f}x")

# Pixel statistics
print("\n" + "=" * 60)
print("PIXEL STATISTICS")
print("=" * 60)
pixel_cols = [c for c in train.columns if c.startswith('pixel')]
pixel_data = train[pixel_cols]
print(f"Value range: [{pixel_data.min().min()}, {pixel_data.max().max()}]")
print(f"Mean pixel value: {pixel_data.values.mean():.2f}")
print(f"Std pixel value: {pixel_data.values.std():.2f}")

# Check for all-zero columns (dead pixels)
zero_cols = (pixel_data.max() == 0).sum()
print(f"All-zero columns (dead pixels): {zero_cols} / {len(pixel_cols)}")

# Check for near-constant columns
low_var = (pixel_data.std() < 1.0).sum()
print(f"Near-constant columns (std < 1): {low_var} / {len(pixel_cols)}")

# Missing values
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
train_missing = train.isnull().sum().sum()
test_missing = test.isnull().sum().sum()
print(f"Train: {train_missing} missing values")
print(f"Test: {test_missing} missing values")

# Sample submission
print("\n" + "=" * 60)
print("SAMPLE SUBMISSION")
print("=" * 60)
print(f"Shape: {sample_sub.shape}")
print(f"Columns: {list(sample_sub.columns)}")
print(f"\nFirst 5 rows:")
print(sample_sub.head())
print(f"\nImageId range: {sample_sub['ImageId'].min()} to {sample_sub['ImageId'].max()}")
