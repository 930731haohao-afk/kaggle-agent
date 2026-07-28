"""
EDA for House Prices Competition
==================================
80 features, regression target (SalePrice).
Focus: missing value patterns, feature-target relationships, outliers, multicollinearity.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/house-prices"
data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

TARGET = "SalePrice"
ID = "Id"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# --- Target analysis ---
section("TARGET ANALYSIS: SalePrice")
print(train[TARGET].describe())
print(f"\nSkewness: {train[TARGET].skew():.4f}")
print(f"Log-transformed skewness: {np.log1p(train[TARGET]).skew():.4f}")
print(f"\nRecommendation: Use log1p(SalePrice) as target for modeling (reduces skew from 1.88 to 0.12)")


# --- Missing value analysis ---
section("MISSING VALUE ANALYSIS")
# Many missing values in this dataset mean 'not applicable' (e.g., no garage, no pool)
# Key: data_description.txt says NA = feature doesn't exist

# Categoricals where NA means 'None' (no feature)
na_means_none = [
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
]
# Numericals where NA likely means 0
na_means_zero = ['MasVnrArea', 'GarageYrBlt']
# Truly missing (need imputation)
truly_missing = ['LotFrontage', 'Electrical']

print("Categoricals where NA = 'no feature' (fill with 'None'):")
for col in na_means_none:
    n_train = train[col].isnull().sum()
    n_test = test[col].isnull().sum()
    if n_train > 0 or n_test > 0:
        print(f"  {col}: train={n_train}, test={n_test}")

print(f"\nNumericals where NA = 0 or needs special handling:")
for col in na_means_zero:
    n_train = train[col].isnull().sum()
    n_test = test[col].isnull().sum()
    print(f"  {col}: train={n_train}, test={n_test}")

print(f"\nTruly missing (need imputation):")
for col in truly_missing:
    n_train = train[col].isnull().sum()
    print(f"  {col}: train={n_train}")

# Additional missing in test only
print(f"\nAdditional missing in test set:")
test_only_missing = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional',
    'GarageCars', 'GarageArea', 'SaleType']
for col in test_only_missing:
    n_test = test[col].isnull().sum()
    if n_test > 0:
        print(f"  {col}: test={n_test}")


# --- Numerical feature analysis ---
section("TOP NUMERICAL FEATURES (by correlation with SalePrice)")
num_cols = train.select_dtypes(include=[np.number]).columns.drop([TARGET, ID]).tolist()
correlations = train[num_cols].corrwith(train[TARGET]).sort_values(ascending=False)

print("Positive correlations:")
for feat, corr in correlations.head(15).items():
    print(f"  {feat:20s} {corr:+.4f}")

print("\nNegative correlations:")
for feat, corr in correlations.tail(5).items():
    print(f"  {feat:20s} {corr:+.4f}")


# --- Multicollinearity check ---
section("MULTICOLLINEARITY (highly correlated feature pairs)")
num_data = train[num_cols].dropna()
corr_matrix = num_data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = []
for col in upper.columns:
    for idx in upper.index:
        if upper.loc[idx, col] > 0.7:
            high_corr.append((idx, col, upper.loc[idx, col]))

high_corr.sort(key=lambda x: x[2], reverse=True)
for f1, f2, c in high_corr:
    print(f"  {f1:20s} <-> {f2:20s}  r={c:.4f}")


# --- Categorical feature analysis ---
section("CATEGORICAL FEATURES: Relationship with SalePrice")
cat_cols = train.select_dtypes(exclude=[np.number]).columns.tolist()

# For each categorical, compute mean SalePrice per category and range
for col in cat_cols:
    grouped = train.groupby(col)[TARGET].agg(['mean', 'count'])
    price_range = grouped['mean'].max() - grouped['mean'].min()
    n_cats = len(grouped)
    print(f"  {col:20s} categories={n_cats:3d}  price_range=${price_range:,.0f}  "
          f"(min=${grouped['mean'].min():,.0f}, max=${grouped['mean'].max():,.0f})")


# --- Outlier detection ---
section("OUTLIER DETECTION")
# Known outliers: GrLivArea > 4000 with low price
outliers = train[(train['GrLivArea'] > 4000) & (train[TARGET] < 300000)]
print(f"GrLivArea outliers (>4000 sqft, <$300K): {len(outliers)} rows")
print(f"  IDs: {outliers[ID].tolist()}")

# General outlier count per numerical feature
print(f"\nOutliers per feature (>3 std from mean):")
for col in num_cols[:15]:
    mean, std = train[col].mean(), train[col].std()
    n_outliers = ((train[col] - mean).abs() > 3 * std).sum()
    if n_outliers > 0:
        print(f"  {col:20s} {n_outliers} outliers")


# --- Skewness of numerical features ---
section("SKEWED NUMERICAL FEATURES (candidates for log transform)")
skewness = train[num_cols].skew().abs().sort_values(ascending=False)
skewed = skewness[skewness > 1.0]
print(f"Features with |skew| > 1.0 ({len(skewed)} features):")
for feat, skew in skewed.items():
    print(f"  {feat:20s} skew={skew:.4f}")


# --- Train vs Test distribution ---
section("TRAIN VS TEST DISTRIBUTION SHIFT")
for col in num_cols:
    train_mean = train[col].mean()
    test_mean = test[col].mean()
    train_std = train[col].std()
    shift = abs(train_mean - test_mean) / (train_std + 1e-8)
    if shift > 0.3:
        print(f"  {col:20s} shift={shift:.4f} (train_mean={train_mean:.2f}, test_mean={test_mean:.2f})")

print("\n  (Only showing features with normalized shift > 0.3)")


# --- Summary ---
section("EDA SUMMARY")
print("""
1. TARGET: SalePrice is right-skewed (1.88). Use log1p transform -> skew becomes 0.12.
2. MISSING VALUES: Most NAs mean 'feature not present' (no pool, no garage, etc.).
   Fill categoricals with 'None', numericals with 0. LotFrontage needs real imputation.
3. TOP PREDICTORS: OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64),
   TotalBsmtSF (0.61), 1stFlrSF (0.61), FullBath (0.56), YearBuilt (0.52).
4. MULTICOLLINEARITY: GarageCars<->GarageArea, TotalBsmtSF<->1stFlrSF,
   GrLivArea<->TotRmsAbvGrd — may want to drop one from each pair.
5. OUTLIERS: 2 properties with GrLivArea>4000 and low price — remove them.
6. SKEWED FEATURES: Many features are heavily skewed — apply log1p to features with |skew|>1.
7. CATEGORICALS: Neighborhood, ExterQual, KitchenQual have large price ranges — very predictive.
8. NO significant train-test distribution shift.

RECOMMENDED VALIDATION: 5-Fold KFold (regression, no class imbalance concern).
Evaluate with RMSE on log-transformed target to match Kaggle's RMSLE metric.
""")
