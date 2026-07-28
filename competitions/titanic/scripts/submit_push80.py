"""
Submission: Push to 0.80 — Two submissions
1. XGBoost with all improvements (highest CV)
2. 6-model ensemble with threshold=0.42
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TARGET_COL = "Survived"
ID_COL = "PassengerId"
RANDOM_SEED = 42
DEFAULT_SURVIVAL_RATE = 0.5

train_raw = pd.read_csv(os.path.join(COMPETITION_DIR, "data/train.csv"))
test_raw = pd.read_csv(os.path.join(COMPETITION_DIR, "data/test.csv"))
sample_sub = pd.read_csv(os.path.join(COMPETITION_DIR, "data/gender_submission.csv"))

y = train_raw[TARGET_COL].astype(int)
n_train = len(train_raw)
test_ids = test_raw[ID_COL]

combined = pd.concat([train_raw.drop(columns=[TARGET_COL]), test_raw], axis=0, ignore_index=True)

# ============================================================
# Full feature engineering (same as push_to_80.py)
# ============================================================
combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Capt": "Rare",
    "Countess": "Rare", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare", "Lady": "Rare", "Sir": "Rare"
}
combined["Title"] = combined["Title"].map(title_map).fillna("Rare")
combined["IsMaster"] = (combined["Title"] == "Master").astype(int)
combined["IsMrs"] = (combined["Title"] == "Mrs").astype(int)
combined["IsMiss"] = (combined["Title"] == "Miss").astype(int)
combined["IsMr"] = (combined["Title"] == "Mr").astype(int)

combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
combined["FamilySizeBin"] = combined["FamilySize"].apply(
    lambda x: 0 if x == 1 else (1 if x == 2 else (2 if x <= 4 else 3))
)

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
combined["IsElderly"] = (combined["Age"] >= 60).astype(int)
combined["Fare"] = combined["Fare"].fillna(combined.iloc[:n_train]["Fare"].median())
combined["FareLog"] = np.log1p(combined["Fare"])
combined["Embarked"] = combined["Embarked"].fillna(combined.iloc[:n_train]["Embarked"].mode()[0])
combined["HasCabin"] = combined["Cabin"].notna().astype(int)
combined["Deck"] = combined["Cabin"].str[0].fillna("U")

deck_group_map = {"A": "ABC", "B": "ABC", "C": "ABC", "D": "DE", "E": "DE",
                   "F": "FG", "G": "FG", "T": "FG", "U": "U"}
combined["DeckGroup"] = combined["Deck"].map(deck_group_map)
combined["Sex"] = combined["Sex"].map({"male": 0, "female": 1})

for col in ["Embarked", "Title", "Deck", "DeckGroup"]:
    freq_map = combined.iloc[:n_train][col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq_map).fillna(0)

ticket_counts = combined["Ticket"].value_counts()
combined["TicketFreq"] = combined["Ticket"].map(ticket_counts)
combined["FarePerPerson"] = combined["Fare"] / combined["TicketFreq"]

def extract_ticket_prefix(ticket):
    parts = ticket.replace(".", "").replace("/", "").split()
    return parts[0].upper() if len(parts) > 1 else "NUMERIC"
combined["TicketPrefix"] = combined["Ticket"].apply(extract_ticket_prefix)
prefix_freq = combined.iloc[:n_train]["TicketPrefix"].value_counts(normalize=True).to_dict()
combined["TicketPrefix_freq"] = combined["TicketPrefix"].map(prefix_freq).fillna(0)

combined["Fare_x_Pclass"] = combined["Fare"] * combined["Pclass"]
combined["FareLog_x_Pclass"] = combined["FareLog"] * combined["Pclass"]
for pclass in [1, 2, 3]:
    mask = combined["Pclass"] == pclass
    train_mask = mask & (combined.index < n_train)
    pclass_median = combined.loc[train_mask, "Fare"].median()
    combined.loc[mask, f"Fare_vs_Pclass{pclass}_median"] = combined.loc[mask, "Fare"] - pclass_median
combined["FarePclassDeviation"] = 0.0
for pclass in [1, 2, 3]:
    col = f"Fare_vs_Pclass{pclass}_median"
    mask = combined["Pclass"] == pclass
    combined.loc[mask, "FarePclassDeviation"] = combined.loc[mask, col]
combined = combined.drop(columns=["Fare_vs_Pclass1_median", "Fare_vs_Pclass2_median",
                                    "Fare_vs_Pclass3_median"])
combined["Sex_x_Pclass"] = combined["Sex"] * combined["Pclass"]

# Survival rate features (full train data — no leakage concern at submission)
combined["Surname"] = combined["Name"].str.split(",").str[0]
train_with_target = combined.iloc[:n_train].copy()
train_with_target[TARGET_COL] = y.values

surname_stats = train_with_target.groupby("Surname").agg(
    count=("Surname", "size"), survived=(TARGET_COL, "sum")
)
surname_stats["rate"] = np.where(surname_stats["count"] >= 2,
    surname_stats["survived"] / surname_stats["count"], DEFAULT_SURVIVAL_RATE)
combined["Family_Survival_Rate"] = combined["Surname"].map(
    surname_stats["rate"].to_dict()).fillna(DEFAULT_SURVIVAL_RATE)

ticket_stats = train_with_target.groupby("Ticket").agg(
    count=("Ticket", "size"), survived=(TARGET_COL, "sum")
)
ticket_stats["rate"] = np.where(ticket_stats["count"] >= 2,
    ticket_stats["survived"] / ticket_stats["count"], DEFAULT_SURVIVAL_RATE)
combined["Ticket_Survival_Rate"] = combined["Ticket"].map(
    ticket_stats["rate"].to_dict()).fillna(DEFAULT_SURVIVAL_RATE)

combined["Combined_Survival_Rate"] = (
    combined["Family_Survival_Rate"] + combined["Ticket_Survival_Rate"]) / 2

drop_cols = ["Name", "Ticket", "Cabin", "Embarked", "Title", "Deck",
             "DeckGroup", "TicketPrefix", "Surname", ID_COL]
combined = combined.drop(columns=drop_cols)

X_train = combined.iloc[:n_train]
X_test = combined.iloc[n_train:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# Submission 1: XGBoost
# ============================================================
print("\nTraining XGBoost on full data...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.03, max_depth=4,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.3, reg_lambda=1.0,
    random_state=RANDOM_SEED, use_label_encoder=False,
    verbosity=0, eval_metric="logloss"
)
xgb_model.fit(X_train, y)
xgb_preds = xgb_model.predict(X_test)

sub1 = pd.DataFrame({sample_sub.columns[0]: test_ids, sample_sub.columns[1]: xgb_preds.astype(int)})
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
f1 = f"submission_xgb_push80_{ts}.csv"
sub1.to_csv(os.path.join(COMPETITION_DIR, "submissions", f1), index=False)
print(f"Saved: {f1} — dist: {dict(zip(*np.unique(xgb_preds, return_counts=True)))}")

# ============================================================
# Submission 2: 6-model ensemble with threshold=0.42
# ============================================================
print("\nTraining 6-model ensemble on full data...")
lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_SEED)
rf = RandomForestClassifier(n_estimators=300, max_depth=7,
                             min_samples_split=5, random_state=RANDOM_SEED, n_jobs=-1)
xgb_m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.7,
                            reg_alpha=0.3, reg_lambda=1.0,
                            random_state=RANDOM_SEED, use_label_encoder=False,
                            verbosity=0, eval_metric="logloss")
lgb_m = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED,
                             n_estimators=300, learning_rate=0.03,
                             num_leaves=20, max_depth=5,
                             min_child_samples=15, reg_alpha=0.5, reg_lambda=1.0)
svm = Pipeline([("scaler", StandardScaler()),
                 ("svm", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                              random_state=RANDOM_SEED))])
knn = Pipeline([("scaler", StandardScaler()),
                 ("knn", KNeighborsClassifier(n_neighbors=11, weights="distance"))])

ensemble = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_m),
                 ("lgb", lgb_m), ("svm", svm), ("knn", knn)],
    voting="soft"
)
ensemble.fit(X_train, y)
ens_probs = ensemble.predict_proba(X_test)[:, 1]
ens_preds = (ens_probs >= 0.42).astype(int)

sub2 = pd.DataFrame({sample_sub.columns[0]: test_ids, sample_sub.columns[1]: ens_preds})
f2 = f"submission_ens6_thresh42_{ts}.csv"
sub2.to_csv(os.path.join(COMPETITION_DIR, "submissions", f2), index=False)
print(f"Saved: {f2} — dist: {dict(zip(*np.unique(ens_preds, return_counts=True)))}")

# Compare prediction differences
diff = (xgb_preds != ens_preds).sum()
print(f"\nPrediction differences between submissions: {diff} / {len(xgb_preds)} ({diff/len(xgb_preds)*100:.1f}%)")
