"""EDA for playground-series-s6e2 — Predicting Heart Disease."""
import pandas as pd
import numpy as np

DATA_DIR = "C:/Users/user/ai_agents/kaggle/competitions/playground-series-s6e2/data"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Submission shape: {sub.shape}")

print(f"\nColumns: {train.columns.tolist()}")
print(f"\nTarget distribution:")
print(train["Heart Disease"].value_counts())
print(train["Heart Disease"].value_counts(normalize=True))

print(f"\nSubmission target values: {sub['Heart Disease'].unique()}")

print("\n=== DTYPES ===")
print(train.dtypes)

print("\n=== TRAIN DESCRIBE ===")
print(train.describe().to_string())

print("\n=== MISSING VALUES (train) ===")
print(train.isnull().sum().to_string())

print("\n=== MISSING VALUES (test) ===")
print(test.isnull().sum().to_string())

print("\n=== UNIQUE VALUES PER COLUMN ===")
for col in train.columns:
    print(f"  {col}: {train[col].nunique()} unique")

print("\n=== CATEGORICAL COLUMNS VALUES ===")
for col in train.select_dtypes(include=["object", "string"]).columns:
    print(f"  {col}: {train[col].unique()}")
