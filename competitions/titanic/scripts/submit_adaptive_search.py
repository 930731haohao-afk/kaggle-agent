"""
Submission: Ensemble with Survival Rate Features (Adaptive Search)
===================================================================
Uses full training data for survival rate computation (no CV leakage concern
at submission time since test labels are unknown).
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TARGET_COL = "Survived"
ID_COL = "PassengerId"
RANDOM_SEED = 42
DEFAULT_SURVIVAL_RATE = 0.5

# Load raw data
train_raw = pd.read_csv(os.path.join(COMPETITION_DIR, "data/train.csv"))
test_raw = pd.read_csv(os.path.join(COMPETITION_DIR, "data/test.csv"))
sample_sub = pd.read_csv(os.path.join(COMPETITION_DIR, "data/gender_submission.csv"))

y = train_raw[TARGET_COL].astype(int)
n_train = len(train_raw)
test_ids = test_raw[ID_COL]

# Combine
combined = pd.concat([train_raw.drop(columns=[TARGET_COL]), test_raw], axis=0, ignore_index=True)

# --- Standard features ---
combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Capt": "Rare",
    "Countess": "Rare", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare", "Lady": "Rare", "Sir": "Rare"
}
combined["Title"] = combined["Title"].map(title_map).fillna("Rare")

combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)

train_part = combined.iloc[:n_train]
age_medians = train_part.groupby(["Title", "Pclass"])["Age"].median()
for idx in combined[combined["Age"].isnull()].index:
    title = combined.loc[idx, "Title"]
    pclass = combined.loc[idx, "Pclass"]
    try:
        combined.loc[idx, "Age"] = age_medians.loc[(title, pclass)]
    except KeyError:
        combined.loc[idx, "Age"] = train_part["Age"].median()

combined["IsChild"] = (combined["Age"] <= 12).astype(int)
combined["Fare"] = combined["Fare"].fillna(combined.iloc[:n_train]["Fare"].median())
combined["FareLog"] = np.log1p(combined["Fare"])
combined["Embarked"] = combined["Embarked"].fillna(combined.iloc[:n_train]["Embarked"].mode()[0])
combined["HasCabin"] = combined["Cabin"].notna().astype(int)
combined["Deck"] = combined["Cabin"].str[0].fillna("U")
combined["Sex"] = combined["Sex"].map({"male": 0, "female": 1})

for col in ["Embarked", "Title", "Deck"]:
    freq_map = combined.iloc[:n_train][col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq_map).fillna(0)

ticket_counts = combined["Ticket"].value_counts()
combined["TicketFreq"] = combined["Ticket"].map(ticket_counts)
combined["FarePerPerson"] = combined["Fare"] / combined["TicketFreq"]

# --- Adaptive search features (using full train data) ---
combined["Surname"] = combined["Name"].str.split(",").str[0]

train_with_target = combined.iloc[:n_train].copy()
train_with_target[TARGET_COL] = y.values

# Family survival rate
surname_stats = train_with_target.groupby("Surname").agg(
    count=("Surname", "size"), survived=(TARGET_COL, "sum")
)
surname_stats["rate"] = np.where(
    surname_stats["count"] >= 2,
    surname_stats["survived"] / surname_stats["count"],
    DEFAULT_SURVIVAL_RATE
)
combined["Family_Survival_Rate"] = combined["Surname"].map(
    surname_stats["rate"].to_dict()
).fillna(DEFAULT_SURVIVAL_RATE)

# Ticket survival rate
ticket_stats = train_with_target.groupby("Ticket").agg(
    count=("Ticket", "size"), survived=(TARGET_COL, "sum")
)
ticket_stats["rate"] = np.where(
    ticket_stats["count"] >= 2,
    ticket_stats["survived"] / ticket_stats["count"],
    DEFAULT_SURVIVAL_RATE
)
combined["Ticket_Survival_Rate"] = combined["Ticket"].map(
    ticket_stats["rate"].to_dict()
).fillna(DEFAULT_SURVIVAL_RATE)

# Combined
combined["Combined_Survival_Rate"] = (
    combined["Family_Survival_Rate"] + combined["Ticket_Survival_Rate"]
) / 2.0

# Deck grouping
deck_group_map = {"A": "ABC", "B": "ABC", "C": "ABC", "D": "DE", "E": "DE",
                   "F": "FG", "G": "FG", "T": "FG", "U": "U"}
combined["DeckGroup"] = combined["Deck"].map(deck_group_map)
deck_group_freq = combined.iloc[:n_train]["DeckGroup"].value_counts(normalize=True).to_dict()
combined["DeckGroup_freq"] = combined["DeckGroup"].map(deck_group_freq).fillna(0)

# Ticket prefix
def extract_ticket_prefix(ticket):
    parts = ticket.replace(".", "").replace("/", "").split()
    return parts[0].upper() if len(parts) > 1 else "NUMERIC"

combined["TicketPrefix"] = combined["Ticket"].apply(extract_ticket_prefix)
prefix_freq = combined.iloc[:n_train]["TicketPrefix"].value_counts(normalize=True).to_dict()
combined["TicketPrefix_freq"] = combined["TicketPrefix"].map(prefix_freq).fillna(0)

# Drop raw columns
drop_cols = ["Name", "Ticket", "Cabin", "Embarked", "Title", "Deck",
             "Surname", "DeckGroup", "TicketPrefix", ID_COL]
combined = combined.drop(columns=drop_cols)

X_train = combined.iloc[:n_train]
X_test = combined.iloc[n_train:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --- Train ensemble on full data ---
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             min_samples_split=5, random_state=RANDOM_SEED, n_jobs=-1)
xgb_m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=RANDOM_SEED, use_label_encoder=False,
                            verbosity=0, eval_metric="logloss")
lgb_m = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED,
                             n_estimators=300, learning_rate=0.05,
                             num_leaves=31, max_depth=6)

ensemble = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_m), ("lgb", lgb_m)],
    voting="soft"
)

print("Training ensemble on full data with survival rate features...")
ensemble.fit(X_train, y)
predictions = ensemble.predict(X_test)

print(f"Predictions: {len(predictions)}")
print(f"Distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")

# Format submission
submission = pd.DataFrame({
    sample_sub.columns[0]: test_ids,
    sample_sub.columns[1]: predictions.astype(int)
})

print(f"\nValidation:")
print(f"  Shape: {'PASS' if submission.shape == sample_sub.shape else 'FAIL'}")
print(f"  Columns: {'PASS' if list(submission.columns) == list(sample_sub.columns) else 'FAIL'}")
print(f"  NaN: {'PASS' if submission.isnull().sum().sum() == 0 else 'FAIL'}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submission_ensemble_adaptive_{timestamp}.csv"
output_path = os.path.join(COMPETITION_DIR, "submissions", filename)
submission.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
