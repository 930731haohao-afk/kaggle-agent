"""Inspect TMDB Box Office Prediction data."""
import pandas as pd
import numpy as np

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/tmdb-box-office-prediction/data"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("=" * 60)
print("SHAPES")
print("=" * 60)
print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")
print(f"Sample submission: {sub.shape}")

print("\n" + "=" * 60)
print("TRAIN COLUMNS & DTYPES")
print("=" * 60)
for col in train.columns:
    null_pct = train[col].isnull().mean() * 100
    dtype = train[col].dtype
    nunique = train[col].nunique()
    sample = str(train[col].dropna().iloc[0])[:80] if train[col].notna().any() else "ALL NULL"
    print(f"  {col:25s} {str(dtype):10s} null={null_pct:5.1f}% unique={nunique:6d}  sample: {sample}")

print("\n" + "=" * 60)
print("TARGET (revenue)")
print("=" * 60)
if 'revenue' in train.columns:
    rev = train['revenue']
    print(f"Min: {rev.min():,.0f}")
    print(f"Max: {rev.max():,.0f}")
    print(f"Mean: {rev.mean():,.0f}")
    print(f"Median: {rev.median():,.0f}")
    print(f"Std: {rev.std():,.0f}")
    print(f"Zeros: {(rev == 0).sum()}")
    print(f"Negative: {(rev < 0).sum()}")
    print(f"Log mean: {np.log1p(rev).mean():.3f}")
    print(f"Log std: {np.log1p(rev).std():.3f}")

print("\n" + "=" * 60)
print("SAMPLE SUBMISSION")
print("=" * 60)
print(sub.head())
print(f"Columns: {sub.columns.tolist()}")

print("\n" + "=" * 60)
print("TEST COLUMNS diff from TRAIN")
print("=" * 60)
train_cols = set(train.columns)
test_cols = set(test.columns)
print(f"In train not test: {train_cols - test_cols}")
print(f"In test not train: {test_cols - train_cols}")

print("\n" + "=" * 60)
print("JSON-LIKE COLUMNS (first few chars)")
print("=" * 60)
for col in train.columns:
    if train[col].dtype == object:
        sample = str(train[col].dropna().iloc[0])[:100]
        if sample.startswith('[') or sample.startswith('{'):
            print(f"  {col}: {sample}")
