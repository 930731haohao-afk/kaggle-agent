"""
Self-Improvement Iterations for Spaceship Titanic
===================================================
Applies: Experience Library, Best-of-N, Verifiable Rewards, Adaptive Search
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/spaceship-titanic"
data_dir = os.path.join(COMPETITION_DIR, "data")
TARGET = "Transported"
ID = "PassengerId"
N_FOLDS = 5
SEED = 42

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

y = train[TARGET].astype(int)
test_ids = test[ID]
n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================
print("=== Enhanced Feature Engineering ===\n")

# 1. Parse PassengerId → group features
combined['Group'] = combined[ID].apply(lambda x: int(x.split('_')[0]))
combined['PersonInGroup'] = combined[ID].apply(lambda x: int(x.split('_')[1]))
combined['GroupSize'] = combined.groupby('Group')['Group'].transform('count')
combined['IsAlone'] = (combined['GroupSize'] == 1).astype(int)

# 2. Parse Cabin → Deck, Num, Side
combined['Deck'] = combined['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else np.nan)
combined['CabinNum'] = combined['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) else np.nan)
combined['Side'] = combined['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else np.nan)

# 3. Parse Name → Surname for family grouping
combined['Surname'] = combined['Name'].apply(
    lambda x: x.split(' ')[-1] if pd.notna(x) else 'Unknown'
)

# 4. Fill missing values
combined['CryoSleep'] = combined['CryoSleep'].fillna(False).astype(bool)
combined['VIP'] = combined['VIP'].fillna(False).astype(bool)
combined['Age'] = combined['Age'].fillna(combined.iloc[:n_train]['Age'].median())
for col in spending_cols:
    combined[col] = combined[col].fillna(0)
combined['HomePlanet'] = combined['HomePlanet'].fillna(combined.iloc[:n_train]['HomePlanet'].mode()[0])
combined['Destination'] = combined['Destination'].fillna(combined.iloc[:n_train]['Destination'].mode()[0])
combined['Deck'] = combined['Deck'].fillna(combined.iloc[:n_train]['Deck'].mode()[0])
combined['Side'] = combined['Side'].fillna(combined.iloc[:n_train]['Side'].mode()[0])
combined['CabinNum'] = combined['CabinNum'].fillna(combined.iloc[:n_train]['CabinNum'].median())

# 5. Spending features
combined['TotalSpending'] = combined[spending_cols].sum(axis=1)
combined['HasSpent'] = (combined['TotalSpending'] > 0).astype(int)
for col in spending_cols:
    combined[f'{col}_log'] = np.log1p(combined[col])
combined['TotalSpending_log'] = np.log1p(combined['TotalSpending'])

# Luxury vs basic
combined['LuxurySpending'] = combined['Spa'] + combined['VRDeck'] + combined['RoomService']
combined['BasicSpending'] = combined['FoodCourt'] + combined['ShoppingMall']
combined['LuxuryRatio'] = combined['LuxurySpending'] / (combined['TotalSpending'] + 1)

# Spending per person in group
combined['SpendingPerGroupMember'] = combined['TotalSpending'] / combined['GroupSize']

# Number of services used
combined['NumServicesUsed'] = (combined[spending_cols] > 0).sum(axis=1)

# 6. Age features
combined['IsChild'] = (combined['Age'] < 13).astype(int)
combined['IsTeenager'] = ((combined['Age'] >= 13) & (combined['Age'] < 18)).astype(int)
combined['AgeBin'] = pd.cut(combined['Age'], bins=[0, 12, 18, 30, 50, 80],
                            labels=[0, 1, 2, 3, 4]).astype(float)

# 7. Boolean to int
combined['CryoSleep_int'] = combined['CryoSleep'].astype(int)
combined['VIP_int'] = combined['VIP'].astype(int)

# 8. Encode categoricals
le_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
for col in le_cols:
    le = LabelEncoder()
    combined[col + '_enc'] = le.fit_transform(combined[col].astype(str))

# 9. NEW: Interaction features (from Titanic experience)
# CryoSleep × HomePlanet
combined['Cryo_x_Planet'] = combined['CryoSleep_int'] * combined['HomePlanet_enc']
# Age × CryoSleep
combined['Age_x_Cryo'] = combined['Age'] * combined['CryoSleep_int']
# Deck × Side interaction
combined['Deck_x_Side'] = combined['Deck_enc'] * 2 + combined['Side_enc']

# 10. NEW: Group-level features (inspired by Titanic family survival rates)
# Group spending stats
group_spending = combined.groupby('Group')['TotalSpending'].agg(['mean', 'std', 'max'])
group_spending.columns = ['GroupSpend_mean', 'GroupSpend_std', 'GroupSpend_max']
group_spending['GroupSpend_std'] = group_spending['GroupSpend_std'].fillna(0)
combined = combined.merge(group_spending, on='Group', how='left')

# Group age stats
group_age = combined.groupby('Group')['Age'].agg(['mean', 'std'])
group_age.columns = ['GroupAge_mean', 'GroupAge_std']
group_age['GroupAge_std'] = group_age['GroupAge_std'].fillna(0)
combined = combined.merge(group_age, on='Group', how='left')

# 11. NEW: Frequency encoding for Deck (high-cardinality)
deck_freq = combined.iloc[:n_train]['Deck'].value_counts(normalize=True).to_dict()
combined['Deck_freq'] = combined['Deck'].map(deck_freq).fillna(0)

# 12. NEW: Cabin number binned (spatial location on ship)
combined['CabinNumBin'] = pd.qcut(combined['CabinNum'], q=10, labels=False, duplicates='drop')
combined['CabinNumBin'] = combined['CabinNumBin'].fillna(0)

# 13. NEW: Spending anomaly (is spending unusually high for their group?)
combined['SpendingAnomaly'] = combined['TotalSpending'] - combined['GroupSpend_mean']

# Fill any remaining NaN
for col in combined.select_dtypes(include=[np.number]).columns:
    if combined[col].isnull().any():
        combined[col] = combined[col].fillna(combined.iloc[:n_train][col].median())

# Drop raw columns
drop_cols = [TARGET, ID, 'Name', 'Cabin', 'HomePlanet', 'Destination',
             'Deck', 'Side', 'CryoSleep', 'VIP', 'Group', 'Surname']
feature_cols = [c for c in combined.columns if c not in drop_cols]

X = combined.iloc[:n_train][feature_cols].copy()
X_test = combined.iloc[n_train:][feature_cols].copy()

print(f"Features: {len(feature_cols)} (was 32)")
print(f"New features: GroupSpend_mean/std/max, GroupAge_mean/std, Deck_freq, CabinNumBin,")
print(f"  SpendingPerGroupMember, NumServicesUsed, Cryo_x_Planet, Age_x_Cryo, Deck_x_Side, SpendingAnomaly")
print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"NaN: train={X.isnull().sum().sum()}, test={X_test.isnull().sum().sum()}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def log_experiment(model_name, params, scores, n_feat, notes=""):
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    with open(exp_file, "r") as f:
        experiments = json.load(f)
    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "n_features": n_feat,
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


def cv_evaluate(model_fn, model_name, params, notes=""):
    """Run CV and return (mean_score, fold_scores, test_probabilities)"""
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    test_probs = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn(params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        score = accuracy_score(y_val, preds)
        scores.append(score)

        if hasattr(model, 'predict_proba'):
            test_probs += model.predict_proba(X_test)[:, 1] / N_FOLDS
        else:
            test_probs += model.predict(X_test) / N_FOLDS

        print(f"  Fold {fold+1}: {score:.6f}")

    mean_score = np.mean(scores)
    exp_id = log_experiment(model_name, params, scores, X.shape[1], notes)
    print(f"  Mean: {mean_score:.6f} (+/- {np.std(scores):.6f}) [Exp #{exp_id}]")
    return mean_score, scores, test_probs


# ============================================================
# ITERATION 1: Best-of-N with enhanced features
# ============================================================
print("\n" + "="*60)
print("  ITERATION 1: Best-of-N — 5 model candidates")
print("="*60)

PREV_BEST = 0.810076  # XGBoost from previous session
results = {}

# Candidate 1: XGBoost (tuned)
print("\n--- XGBoost (tuned) ---")
xgb_params = {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 5,
              "subsample": 0.8, "colsample_bytree": 0.7,
              "reg_alpha": 0.3, "reg_lambda": 1.0}
xgb_score, _, xgb_probs = cv_evaluate(
    lambda p: xgb.XGBClassifier(**p, random_state=SEED, verbosity=0, eval_metric="logloss"),
    "XGBoost-Enhanced", xgb_params,
    "XGBoost with enhanced features + tuned hyperparams"
)
results['xgb'] = (xgb_score, xgb_probs)

# Candidate 2: LightGBM (tuned)
print("\n--- LightGBM (tuned) ---")
lgb_params = {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 25,
              "max_depth": 5, "min_child_samples": 20,
              "subsample": 0.8, "colsample_bytree": 0.7,
              "reg_alpha": 0.5, "reg_lambda": 1.0}
lgb_score, _, lgb_probs = cv_evaluate(
    lambda p: lgb.LGBMClassifier(verbosity=-1, random_state=SEED, **p),
    "LightGBM-Enhanced", lgb_params,
    "LightGBM with enhanced features + more regularization"
)
results['lgb'] = (lgb_score, lgb_probs)

# Candidate 3: CatBoost
print("\n--- CatBoost ---")
cb_params = {"iterations": 500, "learning_rate": 0.05, "depth": 6,
             "l2_leaf_reg": 3.0, "random_seed": SEED, "verbose": 0}
cb_score, _, cb_probs = cv_evaluate(
    lambda p: CatBoostClassifier(**p),
    "CatBoost-Enhanced", cb_params,
    "CatBoost with enhanced features"
)
results['cb'] = (cb_score, cb_probs)

# Candidate 4: RandomForest (deeper)
print("\n--- RandomForest (deeper) ---")
rf_params = {"n_estimators": 800, "max_depth": 20, "min_samples_split": 3,
             "max_features": "sqrt"}
rf_score, _, rf_probs = cv_evaluate(
    lambda p: RandomForestClassifier(**p, random_state=SEED, n_jobs=-1),
    "RandomForest-Enhanced", rf_params,
    "Deeper RF with enhanced features"
)
results['rf'] = (rf_score, rf_probs)

# Candidate 5: LogisticRegression (scaled)
print("\n--- LogisticRegression (scaled) ---")
lr_params = {"C": 1.0, "max_iter": 1000}
lr_score, _, lr_probs = cv_evaluate(
    lambda p: Pipeline([("scaler", StandardScaler()),
                        ("lr", LogisticRegression(**p, random_state=SEED))]),
    "LogisticRegression-Enhanced", lr_params,
    "Logistic regression with scaled enhanced features"
)
results['lr'] = (lr_score, lr_probs)

# Verifiable Rewards
print("\n--- Verifiable Rewards ---")
print(f"Previous best CV: {PREV_BEST:.6f}")
for name, (score, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    delta = score - PREV_BEST
    signal = "IMPROVEMENT" if delta > 0.001 else ("FLAT" if abs(delta) < 0.001 else "DEGRADATION")
    print(f"  {name:>5}: {score:.6f} (delta={delta:+.4f}) [{signal}]")

best_name = max(results, key=lambda k: results[k][0])
best_score = results[best_name][0]
print(f"\nBest candidate: {best_name} ({best_score:.6f})")


# ============================================================
# ITERATION 2: Soft Voting Ensemble (top models)
# ============================================================
print("\n" + "="*60)
print("  ITERATION 2: Soft Voting Ensemble")
print("="*60)

# Use top 4 diverse models
print("\n--- 4-model Soft Voting (XGB + LGB + CatBoost + RF) ---")
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
ens_scores = []
ens_test_probs = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    m_xgb = xgb.XGBClassifier(**xgb_params, random_state=SEED, verbosity=0, eval_metric="logloss")
    m_lgb = lgb.LGBMClassifier(verbosity=-1, random_state=SEED, **lgb_params)
    m_cb = CatBoostClassifier(**cb_params)
    m_rf = RandomForestClassifier(**rf_params, random_state=SEED, n_jobs=-1)

    ens = VotingClassifier(
        estimators=[("xgb", m_xgb), ("lgb", m_lgb), ("cb", m_cb), ("rf", m_rf)],
        voting="soft"
    )
    ens.fit(X_tr, y_tr)
    preds = ens.predict(X_val)
    score = accuracy_score(y_val, preds)
    ens_scores.append(score)
    ens_test_probs += ens.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  Fold {fold+1}: {score:.6f}")

ens_mean = np.mean(ens_scores)
log_experiment("SoftVoting-4Model", {"models": "XGB+LGB+CatBoost+RF", "voting": "soft"},
               ens_scores, X.shape[1], "4-model soft voting with enhanced features")
print(f"  Mean: {ens_mean:.6f} (+/- {np.std(ens_scores):.6f})")

delta_ens = ens_mean - PREV_BEST
print(f"  vs previous best: {delta_ens:+.4f}")


# ============================================================
# ITERATION 3: 6-model ensemble with SVM + LR
# ============================================================
print("\n" + "="*60)
print("  ITERATION 3: 6-model Ensemble (add SVM + LR)")
print("="*60)

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
ens6_scores = []
ens6_test_probs = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    m_xgb = xgb.XGBClassifier(**xgb_params, random_state=SEED, verbosity=0, eval_metric="logloss")
    m_lgb = lgb.LGBMClassifier(verbosity=-1, random_state=SEED, **lgb_params)
    m_cb = CatBoostClassifier(**cb_params)
    m_rf = RandomForestClassifier(**rf_params, random_state=SEED, n_jobs=-1)
    m_svm = Pipeline([("scaler", StandardScaler()),
                      ("svm", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED))])
    m_lr = Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])

    ens6 = VotingClassifier(
        estimators=[("xgb", m_xgb), ("lgb", m_lgb), ("cb", m_cb),
                    ("rf", m_rf), ("svm", m_svm), ("lr", m_lr)],
        voting="soft"
    )
    ens6.fit(X_tr, y_tr)
    preds = ens6.predict(X_val)
    score = accuracy_score(y_val, preds)
    ens6_scores.append(score)
    ens6_test_probs += ens6.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  Fold {fold+1}: {score:.6f}")

ens6_mean = np.mean(ens6_scores)
log_experiment("SoftVoting-6Model", {"models": "XGB+LGB+CB+RF+SVM+LR", "voting": "soft"},
               ens6_scores, X.shape[1], "6-model soft voting with enhanced features")
print(f"  Mean: {ens6_mean:.6f} (+/- {np.std(ens6_scores):.6f})")

delta_ens6 = ens6_mean - PREV_BEST
print(f"  vs previous best: {delta_ens6:+.4f}")


# ============================================================
# ITERATION 4: Weighted probability blend (manual)
# ============================================================
print("\n" + "="*60)
print("  ITERATION 4: Weighted Probability Blend")
print("="*60)

# Weight by CV score (normalized)
scores_dict = {k: v[0] for k, v in results.items()}
total = sum(scores_dict.values())
weights = {k: v / total for k, v in scores_dict.items()}
print("Blend weights (by CV score):")
for k, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {k}: {w:.4f}")

blend_probs = sum(results[k][1] * weights[k] for k in results)

# Find best threshold
print("\nThreshold search:")
best_thresh = 0.5
best_thresh_acc = 0
# Use CV to evaluate thresholds
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros(n_train)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # Quick XGBoost for OOF probs
    m = xgb.XGBClassifier(**xgb_params, random_state=SEED, verbosity=0, eval_metric="logloss")
    m.fit(X_tr, y_tr)
    oof_probs[val_idx] = m.predict_proba(X_val)[:, 1]

for thresh in np.arange(0.40, 0.60, 0.01):
    acc = accuracy_score(y, (oof_probs >= thresh).astype(int))
    if acc > best_thresh_acc:
        best_thresh_acc = acc
        best_thresh = thresh
    print(f"  threshold={thresh:.2f}: OOF accuracy={acc:.6f}")

print(f"\nBest threshold: {best_thresh:.2f} (OOF acc={best_thresh_acc:.6f})")


# ============================================================
# SUMMARY & REWARD TRACKING
# ============================================================
print("\n" + "="*60)
print("  SUMMARY — Verifiable Rewards")
print("="*60)

all_results = {
    "XGBoost-Enhanced": xgb_score,
    "LightGBM-Enhanced": lgb_score,
    "CatBoost-Enhanced": cb_score,
    "RandomForest-Enhanced": rf_score,
    "LogisticRegression-Enhanced": lr_score,
    "SoftVoting-4Model": ens_mean,
    "SoftVoting-6Model": ens6_mean,
}

print(f"\nPrevious best CV: {PREV_BEST:.6f} (XGBoost, 32 features)")
print(f"Current features: {X.shape[1]}")
print()
for name, score in sorted(all_results.items(), key=lambda x: -x[1]):
    delta = score - PREV_BEST
    signal = "+++" if delta > 0.005 else ("++" if delta > 0.001 else ("=" if abs(delta) < 0.001 else "-"))
    print(f"  {signal} {name:<30} {score:.6f} ({delta:+.4f})")

overall_best_name = max(all_results, key=all_results.get)
overall_best_score = all_results[overall_best_name]
print(f"\nOverall best: {overall_best_name} = {overall_best_score:.6f}")
print(f"Improvement over previous: {overall_best_score - PREV_BEST:+.4f}")

# Choose best test predictions for submission
if overall_best_name == "SoftVoting-4Model":
    final_probs = ens_test_probs
elif overall_best_name == "SoftVoting-6Model":
    final_probs = ens6_test_probs
else:
    final_probs = results[best_name][1]

# Apply best threshold
final_preds = (final_probs >= best_thresh).astype(bool)

print(f"\nFinal predictions (threshold={best_thresh:.2f}):")
print(f"  Transported=True:  {final_preds.sum()} ({final_preds.mean()*100:.1f}%)")
print(f"  Transported=False: {(~final_preds).sum()} ({(~final_preds).mean()*100:.1f}%)")

# Save submission
submission = pd.DataFrame({
    "PassengerId": test_ids.values,
    "Transported": final_preds
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"submission_selfimprove_{overall_best_score:.4f}_{timestamp}.csv"
output_path = os.path.join(COMPETITION_DIR, "submissions", fname)
submission.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

# Also save a 0.50 threshold version using best ensemble
final_preds_50 = (final_probs >= 0.50).astype(bool)
fname2 = f"submission_selfimprove_t50_{overall_best_score:.4f}_{timestamp}.csv"
output_path2 = os.path.join(COMPETITION_DIR, "submissions", fname2)
pd.DataFrame({
    "PassengerId": test_ids.values,
    "Transported": final_preds_50
}).to_csv(output_path2, index=False)
print(f"Saved (threshold=0.50): {output_path2}")
