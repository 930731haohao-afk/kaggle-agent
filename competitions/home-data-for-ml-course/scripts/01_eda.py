"""EDA for home-data-for-ml-course competition."""
import pandas as pd
import numpy as np

DATA_DIR = "C:/Users/user/ai_agents/kaggle/competitions/home-data-for-ml-course/data"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget (SalePrice) stats:")
print(train['SalePrice'].describe())
print(f"\nSkewness: {train['SalePrice'].skew():.3f}")
print(f"Log1p skewness: {np.log1p(train['SalePrice']).skew():.3f}")

# Missing values
print("\n=== MISSING VALUES (train) ===")
missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print(missing_train.to_string())

print("\n=== MISSING VALUES (test) ===")
missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
print(missing_test.to_string())

# Column types
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
print(f"\nNumeric columns: {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")

# Top correlations with target
print("\n=== TOP CORRELATIONS WITH SalePrice ===")
corr = train[num_cols].corr()['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False)
print(corr.head(20).to_string())

# Categorical cardinality
print("\n=== CATEGORICAL CARDINALITY ===")
for col in cat_cols:
    nunique = train[col].nunique()
    print(f"  {col}: {nunique} unique values")
