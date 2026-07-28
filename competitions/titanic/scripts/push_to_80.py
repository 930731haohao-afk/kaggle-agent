"""
Push to 0.80 — Titanic Competition
====================================
5 targeted improvements from reflexion analysis:
1. Leave-one-out family survival rate
2. Better title grouping (Master = boy, explicit handling)
3. Fare-Pclass interaction features
4. Threshold optimization
5. More ensemble diversity (add SVM, KNN)
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
N_FOLDS = 5
RANDOM_SEED = 42
DEFAULT_SURVIVAL_RATE = 0.5

train_raw = pd.read_csv(os.path.join(COMPETITION_DIR, "data/train.csv"))
test_raw = pd.read_csv(os.path.join(COMPETITION_DIR, "data/test.csv"))

y = train_raw[TARGET_COL].astype(int)
n_train = len(train_raw)

combined = pd.concat([train_raw.drop(columns=[TARGET_COL]), test_raw], axis=0, ignore_index=True)

print(f"Train: {n_train}, Test: {len(test_raw)}\n")


# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================

# --- Title with better grouping (Improvement #2) ---
combined["Title"] = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Capt": "Rare",
    "Countess": "Rare", "Don": "Rare", "Dona": "Rare",
    "Jonkheer": "Rare", "Lady": "Rare", "Sir": "Rare"
}
combined["Title"] = combined["Title"].map(title_map).fillna("Rare")

# Explicit title-based survival indicators
combined["IsMaster"] = (combined["Title"] == "Master").astype(int)  # Boys: ~57% survival
combined["IsMrs"] = (combined["Title"] == "Mrs").astype(int)        # Married women: ~79%
combined["IsMiss"] = (combined["Title"] == "Miss").astype(int)      # Unmarried women: ~70%
combined["IsMr"] = (combined["Title"] == "Mr").astype(int)          # Men: ~16%

# --- Family features ---
combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
# Finer family size bins: solo(1), couple(2), small(3-4), large(5+)
combined["FamilySizeBin"] = combined["FamilySize"].apply(
    lambda x: 0 if x == 1 else (1 if x == 2 else (2 if x <= 4 else 3))
)

# --- Age ---
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

# --- Fare ---
combined["Fare"] = combined["Fare"].fillna(combined.iloc[:n_train]["Fare"].median())
combined["FareLog"] = np.log1p(combined["Fare"])

# --- Embarked ---
combined["Embarked"] = combined["Embarked"].fillna(combined.iloc[:n_train]["Embarked"].mode()[0])

# --- Cabin ---
combined["HasCabin"] = combined["Cabin"].notna().astype(int)
combined["Deck"] = combined["Cabin"].str[0].fillna("U")

# Deck grouping by survival similarity
deck_group_map = {"A": "ABC", "B": "ABC", "C": "ABC", "D": "DE", "E": "DE",
                   "F": "FG", "G": "FG", "T": "FG", "U": "U"}
combined["DeckGroup"] = combined["Deck"].map(deck_group_map)

# --- Sex ---
combined["Sex"] = combined["Sex"].map({"male": 0, "female": 1})

# --- Frequency encodings ---
for col in ["Embarked", "Title", "Deck", "DeckGroup"]:
    freq_map = combined.iloc[:n_train][col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq_map).fillna(0)

# --- Ticket features ---
ticket_counts = combined["Ticket"].value_counts()
combined["TicketFreq"] = combined["Ticket"].map(ticket_counts)
combined["FarePerPerson"] = combined["Fare"] / combined["TicketFreq"]

# Ticket prefix
def extract_ticket_prefix(ticket):
    parts = ticket.replace(".", "").replace("/", "").split()
    return parts[0].upper() if len(parts) > 1 else "NUMERIC"

combined["TicketPrefix"] = combined["Ticket"].apply(extract_ticket_prefix)
prefix_freq = combined.iloc[:n_train]["TicketPrefix"].value_counts(normalize=True).to_dict()
combined["TicketPrefix_freq"] = combined["TicketPrefix"].map(prefix_freq).fillna(0)

# --- Improvement #3: Fare-Pclass interactions ---
combined["Fare_x_Pclass"] = combined["Fare"] * combined["Pclass"]
combined["FareLog_x_Pclass"] = combined["FareLog"] * combined["Pclass"]
# Fare percentile within Pclass (identifies misclassified passengers)
for pclass in [1, 2, 3]:
    mask = combined["Pclass"] == pclass
    train_mask = mask & (combined.index < n_train)
    pclass_median = combined.loc[train_mask, "Fare"].median()
    combined.loc[mask, f"Fare_vs_Pclass{pclass}_median"] = combined.loc[mask, "Fare"] - pclass_median

# Combined Pclass fare deviation
combined["FarePclassDeviation"] = 0.0
for pclass in [1, 2, 3]:
    col = f"Fare_vs_Pclass{pclass}_median"
    mask = combined["Pclass"] == pclass
    combined.loc[mask, "FarePclassDeviation"] = combined.loc[mask, col]

# Drop per-pclass columns (keep only the combined deviation)
combined = combined.drop(columns=["Fare_vs_Pclass1_median", "Fare_vs_Pclass2_median",
                                    "Fare_vs_Pclass3_median"])

# --- Sex * Pclass interaction (women in 3rd class have lower survival) ---
combined["Sex_x_Pclass"] = combined["Sex"] * combined["Pclass"]

# --- Surname for family survival ---
combined["Surname"] = combined["Name"].str.split(",").str[0]

# --- Survival rate features (computed later in CV-safe manner) ---
# Placeholder columns
combined["Family_Survival_Rate"] = DEFAULT_SURVIVAL_RATE
combined["Ticket_Survival_Rate"] = DEFAULT_SURVIVAL_RATE
combined["Combined_Survival_Rate"] = DEFAULT_SURVIVAL_RATE

# ============================================================
# DROP RAW COLUMNS
# ============================================================

# Keep Surname and Ticket for CV-safe survival rate computation
raw_surnames = combined["Surname"].copy()
raw_tickets = combined["Ticket"].copy()

drop_cols = ["Name", "Ticket", "Cabin", "Embarked", "Title", "Deck",
             "DeckGroup", "TicketPrefix", "Surname", ID_COL]
combined = combined.drop(columns=drop_cols)

X_all = combined.iloc[:n_train].copy()
X_test_all = combined.iloc[n_train:].copy()

feature_names = [c for c in X_all.columns if c not in
                 ["Family_Survival_Rate", "Ticket_Survival_Rate", "Combined_Survival_Rate"]]

print(f"Feature count: {X_all.shape[1]}")
print(f"Features: {list(X_all.columns)}\n")


# ============================================================
# HELPER: Leakage-safe survival rates
# ============================================================

def compute_survival_rates_loo(X_tr, y_tr, X_val, surnames_tr, surnames_val,
                                tickets_tr, tickets_val):
    """
    Improvement #1: Leave-one-out family survival rate.
    For each passenger in val, compute their family's survival rate
    from training data only.
    """
    X_tr = X_tr.copy()
    X_val = X_val.copy()

    tr_data = pd.DataFrame({
        "Surname": surnames_tr.values,
        "Ticket": tickets_tr.values,
        "Survived": y_tr.values
    })

    # Family survival rate
    surname_stats = tr_data.groupby("Surname").agg(
        count=("Surname", "size"), survived=("Survived", "sum")
    )
    surname_stats["rate"] = np.where(
        surname_stats["count"] >= 2,
        surname_stats["survived"] / surname_stats["count"],
        DEFAULT_SURVIVAL_RATE
    )
    surname_map = surname_stats["rate"].to_dict()

    X_tr["Family_Survival_Rate"] = surnames_tr.map(surname_map).fillna(DEFAULT_SURVIVAL_RATE).values
    X_val["Family_Survival_Rate"] = surnames_val.map(surname_map).fillna(DEFAULT_SURVIVAL_RATE).values

    # Ticket survival rate
    ticket_stats = tr_data.groupby("Ticket").agg(
        count=("Ticket", "size"), survived=("Survived", "sum")
    )
    ticket_stats["rate"] = np.where(
        ticket_stats["count"] >= 2,
        ticket_stats["survived"] / ticket_stats["count"],
        DEFAULT_SURVIVAL_RATE
    )
    ticket_map = ticket_stats["rate"].to_dict()

    X_tr["Ticket_Survival_Rate"] = tickets_tr.map(ticket_map).fillna(DEFAULT_SURVIVAL_RATE).values
    X_val["Ticket_Survival_Rate"] = tickets_val.map(ticket_map).fillna(DEFAULT_SURVIVAL_RATE).values

    # Combined
    X_tr["Combined_Survival_Rate"] = (X_tr["Family_Survival_Rate"] + X_tr["Ticket_Survival_Rate"]) / 2
    X_val["Combined_Survival_Rate"] = (X_val["Family_Survival_Rate"] + X_val["Ticket_Survival_Rate"]) / 2

    return X_tr, X_val


# ============================================================
# Logging
# ============================================================

def log_experiment(model_name, params, scores, notes=""):
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    with open(exp_file, "r") as f:
        experiments = json.load(f)
    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "features": "push_to_80_v1",
        "n_features": X_all.shape[1],
        "params": {k: str(v) for k, v in params.items()},
        "cv_strategy": f"{N_FOLDS}-fold-stratified",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 6),
        "cv_std": round(float(np.std(scores)), 6),
        "eval_metric": "accuracy",
        "notes": notes
    }
    experiments.append(experiment)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)
    return exp_id


raw_surnames_train = raw_surnames.iloc[:n_train]
raw_tickets_train = raw_tickets.iloc[:n_train]


# ============================================================
# EXPERIMENT 1: XGBoost with all improvements
# ============================================================
print("=" * 60)
print("EXP 1: XGBoost + all 5 improvements")
print("=" * 60)

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y)):
    X_tr = X_all.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_tr, X_val = compute_survival_rates_loo(
        X_tr, y_tr, X_val,
        raw_surnames_train.iloc[train_idx], raw_surnames_train.iloc[val_idx],
        raw_tickets_train.iloc[train_idx], raw_tickets_train.iloc[val_idx]
    )

    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=1.0,
        random_state=RANDOM_SEED, use_label_encoder=False,
        verbosity=0, eval_metric="logloss"
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    xgb_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

xgb_mean = np.mean(xgb_scores)
print(f"  Mean: {xgb_mean:.6f} (+/- {np.std(xgb_scores):.6f})")
log_experiment("XGBoost-PushTo80", {"improvements": "all5"}, xgb_scores,
               "XGBoost with LOO survival rates, title indicators, fare-pclass interactions")


# ============================================================
# EXPERIMENT 2: 6-model Diverse Ensemble (Improvement #5)
# ============================================================
print("\n" + "=" * 60)
print("EXP 2: 6-Model Diverse Ensemble (LR+RF+XGB+LGB+SVM+KNN)")
print("=" * 60)

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
ens6_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y)):
    X_tr = X_all.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_tr, X_val = compute_survival_rates_loo(
        X_tr, y_tr, X_val,
        raw_surnames_train.iloc[train_idx], raw_surnames_train.iloc[val_idx],
        raw_tickets_train.iloc[train_idx], raw_tickets_train.iloc[val_idx]
    )

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
                                 min_child_samples=15,
                                 reg_alpha=0.5, reg_lambda=1.0)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                     random_state=RANDOM_SEED))
    ])
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=11, weights="distance"))
    ])

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_m),
                     ("lgb", lgb_m), ("svm", svm), ("knn", knn)],
        voting="soft"
    )
    ensemble.fit(X_tr, y_tr)
    y_pred = ensemble.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    ens6_scores.append(score)
    print(f"  Fold {fold+1}: {score:.6f}")

ens6_mean = np.mean(ens6_scores)
print(f"  Mean: {ens6_mean:.6f} (+/- {np.std(ens6_scores):.6f})")
log_experiment("Ensemble6-PushTo80", {"models": "LR+RF+XGB+LGB+SVM+KNN"}, ens6_scores,
               "6-model diverse ensemble with all improvements")


# ============================================================
# EXPERIMENT 3: Threshold Optimization (Improvement #4)
# ============================================================
print("\n" + "=" * 60)
print("EXP 3: Threshold Optimization on 6-Model Ensemble")
print("=" * 60)

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# Collect OOF probabilities
oof_probs = np.zeros(n_train)
oof_preds_default = np.zeros(n_train)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y)):
    X_tr = X_all.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_tr, X_val = compute_survival_rates_loo(
        X_tr, y_tr, X_val,
        raw_surnames_train.iloc[train_idx], raw_surnames_train.iloc[val_idx],
        raw_tickets_train.iloc[train_idx], raw_tickets_train.iloc[val_idx]
    )

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
                                 min_child_samples=15,
                                 reg_alpha=0.5, reg_lambda=1.0)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                     random_state=RANDOM_SEED))
    ])
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=11, weights="distance"))
    ])

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_m),
                     ("lgb", lgb_m), ("svm", svm), ("knn", knn)],
        voting="soft"
    )
    ensemble.fit(X_tr, y_tr)

    probs = ensemble.predict_proba(X_val)[:, 1]
    oof_probs[val_idx] = probs
    oof_preds_default[val_idx] = (probs >= 0.5).astype(int)

# Search for best threshold
print("\nThreshold search:")
best_thresh = 0.5
best_acc = accuracy_score(y, oof_preds_default)
print(f"  Default (0.50): {best_acc:.6f}")

for thresh in np.arange(0.35, 0.65, 0.01):
    preds = (oof_probs >= thresh).astype(int)
    acc = accuracy_score(y, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"  Best threshold: {best_thresh:.2f} → accuracy: {best_acc:.6f}")

# Re-evaluate with best threshold per fold
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
thresh_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y)):
    preds = (oof_probs[val_idx] >= best_thresh).astype(int)
    score = accuracy_score(y.iloc[val_idx], preds)
    thresh_scores.append(score)
    print(f"  Fold {fold+1} (thresh={best_thresh:.2f}): {score:.6f}")

thresh_mean = np.mean(thresh_scores)
print(f"  Mean: {thresh_mean:.6f} (+/- {np.std(thresh_scores):.6f})")
log_experiment("Ensemble6-ThreshOpt", {"threshold": best_thresh}, thresh_scores,
               f"6-model ensemble with optimized threshold={best_thresh:.2f}")


# ============================================================
# EXPERIMENT 4: GradientBoosting with all improvements
# ============================================================
print("\n" + "=" * 60)
print("EXP 4: sklearn GradientBoosting + all improvements")
print("=" * 60)

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
gb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y)):
    X_tr = X_all.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_tr, X_val = compute_survival_rates_loo(
        X_tr, y_tr, X_val,
        raw_surnames_train.iloc[train_idx], raw_surnames_train.iloc[val_idx],
        raw_tickets_train.iloc[train_idx], raw_tickets_train.iloc[val_idx]
    )

    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=4,
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
print(f"  Mean: {gb_mean:.6f} (+/- {np.std(gb_scores):.6f})")
log_experiment("GBM-PushTo80", {"improvements": "all5"}, gb_scores,
               "sklearn GBM with all 5 improvements")


# ============================================================
# SUMMARY
# ============================================================
PREV_BEST_CV = 0.845088
PREV_BEST_LB = 0.77033

print("\n" + "=" * 60)
print("PUSH TO 0.80 — SUMMARY")
print("=" * 60)

all_results = [
    ("Previous best (Ensemble+SurvRates)", PREV_BEST_LB, "LB", "—"),
    ("XGBoost + all improvements", xgb_mean, "CV", f"{xgb_mean - PREV_BEST_CV:+.6f}"),
    ("6-Model Ensemble", ens6_mean, "CV", f"{ens6_mean - PREV_BEST_CV:+.6f}"),
    (f"6-Model + Threshold={best_thresh:.2f}", thresh_mean, "CV", f"{thresh_mean - PREV_BEST_CV:+.6f}"),
    ("GradientBoosting", gb_mean, "CV", f"{gb_mean - PREV_BEST_CV:+.6f}"),
]

print(f"\n{'Model':<40} {'Score':<10} {'Type':<5} {'vs Prev':<10}")
print("-" * 65)
for name, score, stype, delta in all_results:
    print(f"{name:<40} {score:<10.6f} {stype:<5} {delta:<10}")

# Find best CV
cv_results = [(n, s) for n, s, t, _ in all_results if t == "CV"]
best_cv_name, best_cv_score = max(cv_results, key=lambda x: x[1])
print(f"\nBest new CV: {best_cv_name} ({best_cv_score:.6f})")
print(f"\nNote: Previous LB was 0.770 with CV=0.836 (leakage-safe).")
print(f"A higher CV here may or may not translate to higher LB.")
print(f"Recommend submitting the 6-model ensemble AND the threshold-optimized version.")
