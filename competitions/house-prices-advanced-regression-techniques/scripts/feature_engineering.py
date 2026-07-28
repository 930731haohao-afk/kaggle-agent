"""
Feature Engineering for House Prices Competition
==================================================
Based on EDA findings:
- Log transform target (skew 1.88 -> 0.12)
- Most NAs mean 'no feature' -> fill with 'None'/0
- Remove 2 GrLivArea outliers
- Log transform skewed numerical features
- Ordinal encode quality features
- Create total area and age features
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/house-prices"
data_dir = os.path.join(COMPETITION_DIR, "data")
TARGET = "SalePrice"
ID = "Id"

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

# Save target and IDs
y_train = np.log1p(train[TARGET])  # Log transform target
train_ids = train[ID]
test_ids = test[ID]

print(f"Train shape before: {train.shape}")
print(f"Test shape before:  {test.shape}")

# Remove known outliers (2 large houses with low prices)
outlier_idx = train[(train['GrLivArea'] > 4000) & (train[TARGET] < 300000)].index
train = train.drop(outlier_idx).reset_index(drop=True)
y_train = y_train.drop(outlier_idx).reset_index(drop=True)
train_ids = train_ids.drop(outlier_idx).reset_index(drop=True)
print(f"Removed {len(outlier_idx)} outliers. Train shape: {train.shape}")

train = train.drop(columns=[TARGET, ID])
test = test.drop(columns=[ID])

# Combine for consistent processing
n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)


# ============================================================
# 1. MISSING VALUE HANDLING
# ============================================================

# Categoricals where NA means 'None' (no feature)
na_none_cols = [
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
]
for col in na_none_cols:
    combined[col] = combined[col].fillna('None')

# Numericals where NA means 0
na_zero_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
for col in na_zero_cols:
    combined[col] = combined[col].fillna(0)

# GarageYrBlt: fill with YearBuilt when no garage
combined['GarageYrBlt'] = combined['GarageYrBlt'].fillna(combined['YearBuilt'])

# LotFrontage: impute by median per Neighborhood (from train)
train_part = combined.iloc[:n_train]
lot_medians = train_part.groupby('Neighborhood')['LotFrontage'].median()
for idx in combined[combined['LotFrontage'].isnull()].index:
    neighborhood = combined.loc[idx, 'Neighborhood']
    combined.loc[idx, 'LotFrontage'] = lot_medians.get(neighborhood, train_part['LotFrontage'].median())

# Remaining categoricals: fill with mode from train
cat_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()
for col in cat_cols:
    if combined[col].isnull().sum() > 0:
        mode_val = combined.iloc[:n_train][col].mode()[0]
        combined[col] = combined[col].fillna(mode_val)

# Remaining numericals: fill with median from train
num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if combined[col].isnull().sum() > 0:
        median_val = combined.iloc[:n_train][col].median()
        combined[col] = combined[col].fillna(median_val)

print(f"Missing values after imputation: {combined.isnull().sum().sum()}")


# ============================================================
# 2. NEW FEATURES
# ============================================================

# Total area features
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
combined['TotalPorchSF'] = (combined['OpenPorchSF'] + combined['EnclosedPorch'] +
                             combined['3SsnPorch'] + combined['ScreenPorch'] + combined['WoodDeckSF'])
combined['TotalBath'] = (combined['FullBath'] + 0.5 * combined['HalfBath'] +
                          combined['BsmtFullBath'] + 0.5 * combined['BsmtHalfBath'])

# Age features
combined['HouseAge'] = combined['YrSold'] - combined['YearBuilt']
combined['RemodAge'] = combined['YrSold'] - combined['YearRemodAdd']
combined['GarageAge'] = combined['YrSold'] - combined['GarageYrBlt']
combined['IsRemodeled'] = (combined['YearRemodAdd'] != combined['YearBuilt']).astype(int)
combined['IsNewHouse'] = (combined['YrSold'] == combined['YearBuilt']).astype(int)

# Has features (binary)
combined['HasPool'] = (combined['PoolArea'] > 0).astype(int)
combined['HasGarage'] = (combined['GarageArea'] > 0).astype(int)
combined['HasBsmt'] = (combined['TotalBsmtSF'] > 0).astype(int)
combined['Has2ndFlr'] = (combined['2ndFlrSF'] > 0).astype(int)
combined['HasFireplace'] = (combined['Fireplaces'] > 0).astype(int)
combined['HasMasVnr'] = (combined['MasVnrArea'] > 0).astype(int)


# ============================================================
# 3. ORDINAL ENCODING FOR QUALITY FEATURES
# ============================================================

quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for col in quality_cols:
    combined[col] = combined[col].map(quality_map).fillna(0).astype(int)

# Other ordinal mappings
combined['BsmtExposure'] = combined['BsmtExposure'].map(
    {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}).fillna(0).astype(int)

combined['BsmtFinType1'] = combined['BsmtFinType1'].map(
    {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}).fillna(0).astype(int)

combined['BsmtFinType2'] = combined['BsmtFinType2'].map(
    {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}).fillna(0).astype(int)

combined['GarageFinish'] = combined['GarageFinish'].map(
    {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}).fillna(0).astype(int)

combined['Fence'] = combined['Fence'].map(
    {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}).fillna(0).astype(int)

combined['Functional'] = combined['Functional'].map(
    {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}).fillna(8).astype(int)

combined['PavedDrive'] = combined['PavedDrive'].map(
    {'N': 0, 'P': 1, 'Y': 2}).fillna(0).astype(int)

combined['CentralAir'] = combined['CentralAir'].map({'N': 0, 'Y': 1}).fillna(0).astype(int)
combined['Street'] = combined['Street'].map({'Grvl': 0, 'Pave': 1}).fillna(1).astype(int)


# ============================================================
# 4. LABEL ENCODE REMAINING CATEGORICALS
# ============================================================

remaining_cat = combined.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\nLabel encoding {len(remaining_cat)} remaining categorical columns:")

label_encoders = {}
for col in remaining_cat:
    le = LabelEncoder()
    # Fit on combined to handle all categories
    combined[col] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} categories")


# ============================================================
# 5. LOG TRANSFORM SKEWED FEATURES
# ============================================================

num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
skewed = combined[num_cols].skew().abs()
skewed_features = skewed[skewed > 1.0].index.tolist()
print(f"\nLog-transforming {len(skewed_features)} skewed features")
for col in skewed_features:
    combined[col] = np.log1p(combined[col])


# ============================================================
# 6. DROP LOW-VALUE FEATURES
# ============================================================

drop_cols = ['Utilities', 'PoolArea', 'PoolQC', 'MiscVal', 'MiscFeature']
combined = combined.drop(columns=drop_cols, errors='ignore')
print(f"\nDropped {len(drop_cols)} low-value features")


# ============================================================
# SPLIT BACK AND VALIDATE
# ============================================================

train = combined.iloc[:n_train].copy()
test = combined.iloc[n_train:].copy()

print(f"\nTrain shape after: {train.shape}")
print(f"Test shape after:  {test.shape}")

# Validation
train_cols = set(train.columns)
test_cols = set(test.columns)
if train_cols == test_cols:
    print(f"Column check: PASS ({len(train_cols)} features)")
else:
    print(f"Column mismatch! train_only={train_cols - test_cols}, test_only={test_cols - train_cols}")

train_nans = train.isnull().sum().sum()
test_nans = test.isnull().sum().sum()
print(f"NaN check: train={train_nans}, test={test_nans}")

non_numeric = train.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"WARNING: Non-numeric columns: {non_numeric}")
else:
    print("Dtype check: PASS (all numeric)")


# ============================================================
# QUICK FEATURE IMPORTANCE
# ============================================================

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train, y_train)
importances = pd.Series(rf.feature_importances_, index=train.columns).sort_values(ascending=False)
print(f"\nTop 20 Features (RandomForest importance):")
for feat, imp in importances.head(20).items():
    bar = "#" * int(imp * 200)
    print(f"  {feat:20s} {imp:.4f} {bar}")


# ============================================================
# SAVE
# ============================================================

train[ID] = train_ids.values
train[TARGET] = y_train.values  # Already log-transformed
test[ID] = test_ids.values

train.to_csv(os.path.join(data_dir, "train_processed.csv"), index=False)
test.to_csv(os.path.join(data_dir, "test_processed.csv"), index=False)
print(f"\nSaved: train_processed.csv, test_processed.csv")
print(f"NOTE: Target (SalePrice) is log-transformed. Use expm1() to convert predictions back.")
