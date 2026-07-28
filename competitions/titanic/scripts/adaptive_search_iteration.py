"""
Adaptive Search Iteration — Titanic Competition
=================================================
Implements techniques found via Discussion board research:
1. Family Survival Rate (by surname)
2. Ticket Survival Rate (by ticket number)
3. Combined Survival Rate
4. Deck grouping by survival similarity

Source: Top 3% solution (tdody.github.io/Titanic/)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TARGET_COL = "Survived"
ID_COL = "PassengerId"
EVAL_METRIC = "accuracy"
N_FOLDS = 5
RANDOM_SEED = 42
DEFAULT_SURVIVAL_RATE = 0.5

# Load RAW data (need Name, Ticket, Cabin for new features)
train_raw = pd.read_csv(os.path.join(COMPETITION_DIR, "data/train.csv"))
test_raw = pd.read_csv(os.path.join(COMPETITION_DIR, "data/test.csv"))

y = train_raw[TARGET_COL].astype(int)
n_train = len(train_raw)

# Combine for consistent feature engineering
combined = pd.concat([train_raw.drop(columns=[TARGET_COL]), test_raw], axis=0, ignore_index=True)

print(f"Train: {n_train} rows, Test: {len(test_raw)} rows")
print(f"Combined: {len(combined)} rows\n")


# ============================================================
# FEATURE ENGINEERING — Same as before + new adaptive search features
# ============================================================

# --- Standard features (from previous pipeline) ---

# 1. Title extraction
combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Capt": "Rare",
    "Countess": "Rare", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare", "Lady": "Rare", "Sir": "Rare"
}
combined["Title"] = combined["Title"].map(title_map).fillna("Rare")

# 2. Family features
combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)

# 3. Age imputation (from train medians by Title + Pclass)
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

# 4. Fare imputation
train_fare_median = combined.iloc[:n_train]["Fare"].median()
combined["Fare"] = combined["Fare"].fillna(train_fare_median)
combined["FareLog"] = np.log1p(combined["Fare"])

# 5. Embarked imputation
embarked_mode = combined.iloc[:n_train]["Embarked"].mode()[0]
combined["Embarked"] = combined["Embarked"].fillna(embarked_mode)

# 6. Cabin features
combined["HasCabin"] = combined["Cabin"].notna().astype(int)
combined["Deck"] = combined["Cabin"].str[0].fillna("U")

# 7. Sex encoding
combined["Sex"] = combined["Sex"].map({"male": 0, "female": 1})

# 8. Frequency encodings (fit on train only)
for col in ["Embarked", "Title", "Deck"]:
    freq_map = combined.iloc[:n_train][col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq_map).fillna(0)

# 9. Ticket frequency
ticket_counts = combined["Ticket"].value_counts()
combined["TicketFreq"] = combined["Ticket"].map(ticket_counts)
combined["FarePerPerson"] = combined["Fare"] / combined["TicketFreq"]


# ============================================================
# NEW: Adaptive Search Features (from Discussion board research)
# ============================================================

print("=" * 60)
print("ADAPTIVE SEARCH: New features from Discussion board")
print("=" * 60)

# --- NEW FEATURE 1: Family Survival Rate ---
# Extract surname from Name
combined["Surname"] = combined["Name"].str.split(",").str[0]

# Compute family survival rate from TRAIN data only
train_with_target = combined.iloc[:n_train].copy()
train_with_target[TARGET_COL] = y.values

# For families with 2+ members in train, compute mean survival
surname_stats = train_with_target.groupby("Surname").agg(
    family_count=("Surname", "size"),
    family_survived=(TARGET_COL, "sum")
).reset_index()

# Only use families with 2+ members (solo travelers don't have family signal)
surname_stats["Family_Survival_Rate"] = np.where(
    surname_stats["family_count"] >= 2,
    surname_stats["family_survived"] / surname_stats["family_count"],
    DEFAULT_SURVIVAL_RATE
)
surname_survival_map = surname_stats.set_index("Surname")["Family_Survival_Rate"].to_dict()

combined["Family_Survival_Rate"] = combined["Surname"].map(surname_survival_map).fillna(DEFAULT_SURVIVAL_RATE)

print(f"\n1. Family Survival Rate:")
print(f"   Unique surnames: {combined['Surname'].nunique()}")
print(f"   Families with 2+ members in train: {(surname_stats['family_count'] >= 2).sum()}")
print(f"   Rate distribution: mean={combined['Family_Survival_Rate'].mean():.3f}, "
      f"std={combined['Family_Survival_Rate'].std():.3f}")


# --- NEW FEATURE 2: Ticket Survival Rate ---
ticket_stats = train_with_target.groupby("Ticket").agg(
    ticket_count=("Ticket", "size"),
    ticket_survived=(TARGET_COL, "sum")
).reset_index()

# Only use tickets shared by 2+ passengers
ticket_stats["Ticket_Survival_Rate"] = np.where(
    ticket_stats["ticket_count"] >= 2,
    ticket_stats["ticket_survived"] / ticket_stats["ticket_count"],
    DEFAULT_SURVIVAL_RATE
)
ticket_survival_map = ticket_stats.set_index("Ticket")["Ticket_Survival_Rate"].to_dict()

combined["Ticket_Survival_Rate"] = combined["Ticket"].map(ticket_survival_map).fillna(DEFAULT_SURVIVAL_RATE)

print(f"\n2. Ticket Survival Rate:")
print(f"   Unique tickets: {combined['Ticket'].nunique()}")
print(f"   Shared tickets (2+) in train: {(ticket_stats['ticket_count'] >= 2).sum()}")
print(f"   Rate distribution: mean={combined['Ticket_Survival_Rate'].mean():.3f}, "
      f"std={combined['Ticket_Survival_Rate'].std():.3f}")


# --- NEW FEATURE 3: Combined Survival Rate ---
combined["Combined_Survival_Rate"] = (
    combined["Family_Survival_Rate"] + combined["Ticket_Survival_Rate"]
) / 2.0

print(f"\n3. Combined Survival Rate:")
print(f"   mean={combined['Combined_Survival_Rate'].mean():.3f}, "
      f"std={combined['Combined_Survival_Rate'].std():.3f}")


# --- NEW FEATURE 4: Deck Grouping by survival similarity ---
# From top solutions: ABC (upper decks, 1st class), DE (middle), FG (lower), U (unknown)
deck_group_map = {
    "A": "ABC", "B": "ABC", "C": "ABC",
    "D": "DE", "E": "DE",
    "F": "FG", "G": "FG",
    "T": "FG",  # Rare, group with lower
    "U": "U"
}
combined["DeckGroup"] = combined["Deck"].map(deck_group_map)
deck_group_freq = combined.iloc[:n_train]["DeckGroup"].value_counts(normalize=True).to_dict()
combined["DeckGroup_freq"] = combined["DeckGroup"].map(deck_group_freq).fillna(0)

print(f"\n4. Deck Grouping:")
print(f"   Groups: {combined['DeckGroup'].value_counts().to_dict()}")


# --- NEW FEATURE 5: Ticket Prefix ---
def extract_ticket_prefix(ticket):
    parts = ticket.replace(".", "").replace("/", "").split()
    if len(parts) > 1:
        return parts[0].upper()
    return "NUMERIC"

combined["TicketPrefix"] = combined["Ticket"].apply(extract_ticket_prefix)
prefix_freq = combined.iloc[:n_train]["TicketPrefix"].value_counts(normalize=True).to_dict()
combined["TicketPrefix_freq"] = combined["TicketPrefix"].map(prefix_freq).fillna(0)

print(f"\n5. Ticket Prefix:")
print(f"   Unique prefixes: {combined['TicketPrefix'].nunique()}")
print(f"   Top prefixes: {combined['TicketPrefix'].value_counts().head(5).to_dict()}")


# ============================================================
# DROP RAW COLUMNS & SPLIT
# ============================================================

drop_cols = ["Name", "Ticket", "Cabin", "Embarked", "Title", "Deck",
             "Surname", "DeckGroup", "TicketPrefix"]
combined = combined.drop(columns=drop_cols)
combined = combined.drop(columns=[ID_COL])

train_X = combined.iloc[:n_train].copy()
test_X = combined.iloc[n_train:].copy()

print(f"\nFinal feature count: {train_X.shape[1]} (was 19, now {train_X.shape[1]})")
print(f"New features: Family_Survival_Rate, Ticket_Survival_Rate, "
      f"Combined_Survival_Rate, DeckGroup_freq, TicketPrefix_freq")
print(f"Features: {list(train_X.columns)}")


# ============================================================
# IMPORTANT: Handle target leakage in Family/Ticket Survival Rate
# ============================================================
# The survival rate features use the target from train, so we must
# compute them in a leave-one-out fashion during CV to avoid leakage.
# For simplicity in this test, we'll use a "leave-family-out" approach:
# during CV, recompute the survival rate excluding the validation fold.

print("\n" + "=" * 60)
print("CROSS-VALIDATION WITH LEAKAGE-SAFE SURVIVAL RATES")
print("=" * 60)


def log_experiment(model_name, params, scores, notes=""):
    """Log experiment to experiments.json."""
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    with open(exp_file, "r") as f:
        experiments = json.load(f)
    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "features": "adaptive_search_v1",
        "n_features": train_X.shape[1],
        "params": {k: str(v) for k, v in params.items()},
        "cv_strategy": f"{N_FOLDS}-fold-stratified",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 6),
        "cv_std": round(float(np.std(scores)), 6),
        "eval_metric": EVAL_METRIC,
        "notes": notes
    }
    experiments.append(experiment)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)
    return exp_id


def compute_survival_rates_safe(X_tr, y_tr, X_val, raw_train, raw_val):
    """Recompute survival rates using only training fold data."""
    # Surname-based family survival rate
    tr_data = raw_train.copy()
    tr_data["Survived"] = y_tr.values

    surname_stats = tr_data.groupby("Surname").agg(
        count=("Surname", "size"),
        survived=("Survived", "sum")
    )
    surname_stats["rate"] = np.where(
        surname_stats["count"] >= 2,
        surname_stats["survived"] / surname_stats["count"],
        DEFAULT_SURVIVAL_RATE
    )
    surname_map = surname_stats["rate"].to_dict()

    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_tr["Family_Survival_Rate"] = raw_train["Surname"].map(surname_map).fillna(DEFAULT_SURVIVAL_RATE).values
    X_val["Family_Survival_Rate"] = raw_val["Surname"].map(surname_map).fillna(DEFAULT_SURVIVAL_RATE).values

    # Ticket-based survival rate
    ticket_stats = tr_data.groupby("Ticket").agg(
        count=("Ticket", "size"),
        survived=("Survived", "sum")
    )
    ticket_stats["rate"] = np.where(
        ticket_stats["count"] >= 2,
        ticket_stats["survived"] / ticket_stats["count"],
        DEFAULT_SURVIVAL_RATE
    )
    ticket_map = ticket_stats["rate"].to_dict()

    X_tr["Ticket_Survival_Rate"] = raw_train["Ticket"].map(ticket_map).fillna(DEFAULT_SURVIVAL_RATE).values
    X_val["Ticket_Survival_Rate"] = raw_val["Ticket"].map(ticket_map).fillna(DEFAULT_SURVIVAL_RATE).values

    # Combined
    X_tr["Combined_Survival_Rate"] = (X_tr["Family_Survival_Rate"] + X_tr["Ticket_Survival_Rate"]) / 2.0
    X_val["Combined_Survival_Rate"] = (X_val["Family_Survival_Rate"] + X_val["Ticket_Survival_Rate"]) / 2.0

    return X_tr, X_val


# Prepare raw data for leakage-safe CV
raw_surnames = train_raw["Name"].str.split(",").str[0]
raw_tickets = train_raw["Ticket"]
raw_df = pd.DataFrame({"Surname": raw_surnames, "Ticket": raw_tickets})


# ============================================================
# EXPERIMENT 1: XGBoost with new features (leakage-safe CV)
# ============================================================
print("\n--- Experiment: XGBoost + Survival Rate Features ---")

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_X, y)):
    X_tr = train_X.iloc[train_idx]
    X_val = train_X.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    raw_tr = raw_df.iloc[train_idx]
    raw_val = raw_df.iloc[val_idx]

    # Recompute survival rates using only train fold
    X_tr, X_val = compute_survival_rates_safe(X_tr, y_tr, X_val, raw_tr, raw_val)

    model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, use_label_encoder=False,
        verbosity=0, eval_metric="logloss"
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    xgb_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

xgb_mean = np.mean(xgb_scores)
xgb_std = np.std(xgb_scores)
xgb_params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
               "new_features": "Family+Ticket+Combined_Survival_Rate, DeckGroup, TicketPrefix"}
exp_id = log_experiment("XGBoost-AdaptiveSearch", xgb_params, xgb_scores,
                         "XGBoost with survival rate features from Discussion board research")
print(f"  Mean: {xgb_mean:.6f} (+/- {xgb_std:.6f}) [#{exp_id}]")


# ============================================================
# EXPERIMENT 2: Soft Voting Ensemble with new features
# ============================================================
print("\n--- Experiment: Ensemble + Survival Rate Features ---")

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
ens_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_X, y)):
    X_tr = train_X.iloc[train_idx]
    X_val = train_X.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    raw_tr = raw_df.iloc[train_idx]
    raw_val = raw_df.iloc[val_idx]

    X_tr, X_val = compute_survival_rates_safe(X_tr, y_tr, X_val, raw_tr, raw_val)

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
    ensemble.fit(X_tr, y_tr)
    y_pred = ensemble.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    ens_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

ens_mean = np.mean(ens_scores)
ens_std = np.std(ens_scores)
ens_params = {"models": "LR+RF+XGB+LGB", "voting": "soft",
               "new_features": "Family+Ticket+Combined_Survival_Rate, DeckGroup, TicketPrefix"}
exp_id = log_experiment("Ensemble-AdaptiveSearch", ens_params, ens_scores,
                         "Soft voting ensemble with survival rate features from Discussion board")
print(f"  Mean: {ens_mean:.6f} (+/- {ens_std:.6f}) [#{exp_id}]")


# ============================================================
# EXPERIMENT 3: GradientBoosting with new features (conservative)
# ============================================================
print("\n--- Experiment: GradientBoosting + Survival Rate Features ---")

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
gb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_X, y)):
    X_tr = train_X.iloc[train_idx]
    X_val = train_X.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    raw_tr = raw_df.iloc[train_idx]
    raw_val = raw_df.iloc[val_idx]

    X_tr, X_val = compute_survival_rates_safe(X_tr, y_tr, X_val, raw_tr, raw_val)

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_samples_split=10, min_samples_leaf=5,
        subsample=0.8, max_features="sqrt",
        random_state=RANDOM_SEED
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    gb_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

gb_mean = np.mean(gb_scores)
gb_std = np.std(gb_scores)
gb_params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4,
              "new_features": "Family+Ticket+Combined_Survival_Rate, DeckGroup, TicketPrefix"}
exp_id = log_experiment("GBM-AdaptiveSearch", gb_params, gb_scores,
                         "sklearn GBM with survival rate features from Discussion board")
print(f"  Mean: {gb_mean:.6f} (+/- {gb_std:.6f}) [#{exp_id}]")


# ============================================================
# VERIFIABLE REWARDS SUMMARY
# ============================================================
PREVIOUS_BEST_CV = 0.845088  # SoftVotingEnsemble
PREVIOUS_BEST_LB = 0.76794

print("\n" + "=" * 60)
print("ADAPTIVE SEARCH — VERIFIABLE REWARDS SUMMARY")
print("=" * 60)

results = [
    ("Previous best (Ensemble)", PREVIOUS_BEST_CV, 0.019846),
    ("XGBoost + Survival Rates", xgb_mean, xgb_std),
    ("Ensemble + Survival Rates", ens_mean, ens_std),
    ("GBM + Survival Rates", gb_mean, gb_std),
]

print(f"\n{'#':<4} {'Model':<35} {'CV Mean':<10} {'CV Std':<10} {'Delta':<10}")
print("-" * 69)
for i, (name, mean, std) in enumerate(results):
    delta = "—" if i == 0 else f"{mean - PREVIOUS_BEST_CV:+.6f}"
    print(f"{i:<4} {name:<35} {mean:<10.6f} {std:<10.6f} {delta:<10}")

best_new = max(results[1:], key=lambda x: x[1])
print(f"\nBest new: {best_new[0]} (CV {best_new[1]:.6f})")
print(f"Improvement: {best_new[1] - PREVIOUS_BEST_CV:+.6f}")

# Feature importance from XGBoost
print("\n--- Feature Importance (XGBoost with new features) ---")
# Retrain on full data to get importance
X_full = train_X.copy()
X_full["Family_Survival_Rate"] = combined.iloc[:n_train]["Family_Survival_Rate"].values
X_full["Ticket_Survival_Rate"] = combined.iloc[:n_train]["Ticket_Survival_Rate"].values
X_full["Combined_Survival_Rate"] = combined.iloc[:n_train]["Combined_Survival_Rate"].values

model_full = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, use_label_encoder=False,
    verbosity=0, eval_metric="logloss"
)
model_full.fit(X_full, y)
importances = pd.Series(model_full.feature_importances_, index=X_full.columns).sort_values(ascending=False)
print("\nTop 15 features:")
for feat, imp in importances.head(15).items():
    marker = " <<< NEW" if feat in ["Family_Survival_Rate", "Ticket_Survival_Rate",
                                      "Combined_Survival_Rate", "DeckGroup_freq", "TicketPrefix_freq"] else ""
    print(f"  {feat:30s} {imp:.4f}{marker}")
