"""
Feature Engineering for Titanic Competition
=============================================
Based on EDA findings:
- Sex is the strongest predictor
- Title from Name is very informative
- Pclass, Fare, Age are important
- Family size has non-linear effect
- Cabin mostly missing but deck letter may help
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TARGET_COL = "Survived"
ID_COL = "PassengerId"

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, TRAIN_FILE))
test = pd.read_csv(os.path.join(data_dir, TEST_FILE))

# Separate target and ID
y_train = train[TARGET_COL]
train_ids = train[ID_COL]
test_ids = test[ID_COL]

train = train.drop(columns=[TARGET_COL, ID_COL])
test = test.drop(columns=[ID_COL])

# Combine for consistent processing (track split point)
n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)

print(f"Train shape before: {train.shape}")
print(f"Test shape before:  {test.shape}")
print(f"Combined shape:     {combined.shape}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# 1. Title extraction from Name
combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
# Group rare titles
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Capt": "Rare",
    "Countess": "Rare", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare", "Lady": "Rare", "Sir": "Rare"
}
combined["Title"] = combined["Title"].map(title_map).fillna("Rare")
print(f"\nTitle distribution:\n{combined['Title'].value_counts()}")

# 2. Family features
combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
# Non-linear family size: solo, small (2-4), large (5+)
combined["FamilySizeBin"] = combined["FamilySize"].apply(
    lambda x: 0 if x == 1 else (1 if x <= 4 else 2)
)

# 3. Age imputation using median by Title + Pclass (fit on train only)
train_part = combined.iloc[:n_train]
age_medians = train_part.groupby(["Title", "Pclass"])["Age"].median()
for idx in combined[combined["Age"].isnull()].index:
    title = combined.loc[idx, "Title"]
    pclass = combined.loc[idx, "Pclass"]
    try:
        combined.loc[idx, "Age"] = age_medians.loc[(title, pclass)]
    except KeyError:
        combined.loc[idx, "Age"] = train_part["Age"].median()

# Age bins
combined["AgeBin"] = pd.cut(combined["Age"], bins=[0, 12, 18, 35, 55, 80], labels=[0, 1, 2, 3, 4]).astype(int)
combined["IsChild"] = (combined["Age"] <= 12).astype(int)

# 4. Fare imputation (1 missing in test)
train_fare_median = combined.iloc[:n_train]["Fare"].median()
combined["Fare"] = combined["Fare"].fillna(train_fare_median)
# Log transform fare (heavily skewed)
combined["FareLog"] = np.log1p(combined["Fare"])
# Fare bins
combined["FareBin"] = pd.qcut(combined["Fare"], q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int)

# 5. Embarked imputation (2 missing in train)
embarked_mode = combined.iloc[:n_train]["Embarked"].mode()[0]
combined["Embarked"] = combined["Embarked"].fillna(embarked_mode)

# 6. Cabin features
combined["HasCabin"] = combined["Cabin"].notna().astype(int)
combined["Deck"] = combined["Cabin"].str[0].fillna("U")  # U = Unknown

# 7. Sex encoding
combined["Sex"] = combined["Sex"].map({"male": 0, "female": 1})

# 8. Embarked encoding (frequency from train)
embarked_freq = combined.iloc[:n_train]["Embarked"].value_counts(normalize=True).to_dict()
combined["Embarked_freq"] = combined["Embarked"].map(embarked_freq).fillna(0)

# 9. Title encoding (frequency from train)
title_freq = combined.iloc[:n_train]["Title"].value_counts(normalize=True).to_dict()
combined["Title_freq"] = combined["Title"].map(title_freq).fillna(0)

# 10. Deck encoding (frequency from train)
deck_freq = combined.iloc[:n_train]["Deck"].value_counts(normalize=True).to_dict()
combined["Deck_freq"] = combined["Deck"].map(deck_freq).fillna(0)

# 11. Ticket frequency (shared tickets indicate traveling together)
ticket_counts = combined["Ticket"].value_counts()
combined["TicketFreq"] = combined["Ticket"].map(ticket_counts)

# 12. Fare per person (shared ticket)
combined["FarePerPerson"] = combined["Fare"] / combined["TicketFreq"]


# ============================================================
# DROP RAW COLUMNS
# ============================================================

drop_cols = ["Name", "Ticket", "Cabin", "Embarked", "Title", "Deck"]
combined = combined.drop(columns=drop_cols)


# ============================================================
# SPLIT BACK
# ============================================================

train = combined.iloc[:n_train].copy()
test = combined.iloc[n_train:].copy()


# ============================================================
# VALIDATION
# ============================================================

print(f"\nTrain shape after: {train.shape}")
print(f"Test shape after:  {test.shape}")

train_cols = set(train.columns)
test_cols = set(test.columns)
if train_cols != test_cols:
    print(f"\nWARNING: Column mismatch!")
    print(f"  In train but not test: {train_cols - test_cols}")
    print(f"  In test but not train: {test_cols - train_cols}")
else:
    print(f"\nColumn check: PASS ({len(train_cols)} features)")

train_nans = train.isnull().sum().sum()
test_nans = test.isnull().sum().sum()
print(f"NaN check: train={train_nans}, test={test_nans}")
if train_nans > 0:
    print(f"  Train NaN columns: {train.columns[train.isnull().any()].tolist()}")
if test_nans > 0:
    print(f"  Test NaN columns: {test.columns[test.isnull().any()].tolist()}")

non_numeric = train.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"WARNING: Non-numeric columns remain: {non_numeric}")
else:
    print("Dtype check: PASS (all numeric)")

print(f"\nFinal features: {list(train.columns)}")


# ============================================================
# QUICK FEATURE IMPORTANCE
# ============================================================

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train, y_train)
importances = pd.Series(rf.feature_importances_, index=train.columns).sort_values(ascending=False)
print(f"\nFeature Importance (RandomForest):")
for feat, imp in importances.items():
    bar = "#" * int(imp * 100)
    print(f"  {feat:20s} {imp:.4f} {bar}")


# ============================================================
# SAVE
# ============================================================

train[ID_COL] = train_ids.values
train[TARGET_COL] = y_train.values
test[ID_COL] = test_ids.values

output_dir = os.path.join(COMPETITION_DIR, "data")
train.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
test.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
print(f"\nSaved: train_processed.csv, test_processed.csv")
