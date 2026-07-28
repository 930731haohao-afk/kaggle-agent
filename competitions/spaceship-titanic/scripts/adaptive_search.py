"""
Adaptive Search Iteration for Spaceship Titanic
=================================================
Key additions from research:
1. Group transport rate (like Titanic family survival rate)
2. Surname-based family transport rate
3. Smart missing value imputation using group/surname info
4. CryoSleep imputation from spending patterns
5. Spending clustering (K-Means)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
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
DEFAULT_RATE = 0.5

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

y = train[TARGET].astype(int)
test_ids = test[ID]
n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

print("=== Adaptive Search: Enhanced Feature Engineering ===\n")

# ============================================================
# 1. Parse structured columns
# ============================================================
combined['Group'] = combined[ID].apply(lambda x: int(x.split('_')[0]))
combined['PersonInGroup'] = combined[ID].apply(lambda x: int(x.split('_')[1]))
combined['GroupSize'] = combined.groupby('Group')['Group'].transform('count')
combined['IsAlone'] = (combined['GroupSize'] == 1).astype(int)

combined['Deck'] = combined['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else np.nan)
combined['CabinNum'] = combined['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) else np.nan)
combined['Side'] = combined['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else np.nan)

combined['Surname'] = combined['Name'].apply(
    lambda x: x.split(' ')[-1] if pd.notna(x) else 'Unknown'
)

# ============================================================
# 2. Smart imputation: Use group info to fill missing values
# ============================================================
# If CryoSleep is missing but all spending is 0, likely CryoSleep=True
spending_sum = combined[spending_cols].fillna(0).sum(axis=1)
cryo_missing = combined['CryoSleep'].isna()
combined.loc[cryo_missing & (spending_sum == 0), 'CryoSleep'] = True
combined.loc[cryo_missing & (spending_sum > 0), 'CryoSleep'] = False
combined['CryoSleep'] = combined['CryoSleep'].fillna(False).astype(bool)

# If CryoSleep=True, spending should be 0
cryo_mask = combined['CryoSleep'] == True
for col in spending_cols:
    combined.loc[cryo_mask & combined[col].isna(), col] = 0

# Fill remaining missing with group-level info where possible
for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
    # Fill from group members
    group_modes = combined.groupby('Group')[col].apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 and x.notna().any() else np.nan
    )
    missing = combined[col].isna()
    combined.loc[missing, col] = combined.loc[missing, 'Group'].map(group_modes)
    # Fill remaining with overall mode
    combined[col] = combined[col].fillna(combined.iloc[:n_train][col].mode()[0])

combined['VIP'] = combined['VIP'].fillna(False).astype(bool)
combined['Age'] = combined['Age'].fillna(combined.iloc[:n_train]['Age'].median())
for col in spending_cols:
    combined[col] = combined[col].fillna(0)
combined['CabinNum'] = combined['CabinNum'].fillna(combined.iloc[:n_train]['CabinNum'].median())

# ============================================================
# 3. Core spending features
# ============================================================
combined['TotalSpending'] = combined[spending_cols].sum(axis=1)
combined['HasSpent'] = (combined['TotalSpending'] > 0).astype(int)
for col in spending_cols:
    combined[f'{col}_log'] = np.log1p(combined[col])
combined['TotalSpending_log'] = np.log1p(combined['TotalSpending'])

combined['LuxurySpending'] = combined['Spa'] + combined['VRDeck'] + combined['RoomService']
combined['BasicSpending'] = combined['FoodCourt'] + combined['ShoppingMall']
combined['LuxuryRatio'] = combined['LuxurySpending'] / (combined['TotalSpending'] + 1)
combined['SpendingPerGroupMember'] = combined['TotalSpending'] / combined['GroupSize']
combined['NumServicesUsed'] = (combined[spending_cols] > 0).sum(axis=1)

# ============================================================
# 4. Age features
# ============================================================
combined['IsChild'] = (combined['Age'] < 13).astype(int)
combined['IsTeenager'] = ((combined['Age'] >= 13) & (combined['Age'] < 18)).astype(int)
combined['AgeBin'] = pd.cut(combined['Age'], bins=[0, 12, 18, 30, 50, 80],
                            labels=[0, 1, 2, 3, 4]).astype(float)

# ============================================================
# 5. Boolean to int
# ============================================================
combined['CryoSleep_int'] = combined['CryoSleep'].astype(int)
combined['VIP_int'] = combined['VIP'].astype(int)

# ============================================================
# 6. Encode categoricals
# ============================================================
le_cols_list = ['HomePlanet', 'Destination', 'Deck', 'Side']
for col in le_cols_list:
    le = LabelEncoder()
    combined[col + '_enc'] = le.fit_transform(combined[col].astype(str))

# ============================================================
# 7. Interaction features
# ============================================================
combined['Cryo_x_Planet'] = combined['CryoSleep_int'] * combined['HomePlanet_enc']
combined['Age_x_Cryo'] = combined['Age'] * combined['CryoSleep_int']
combined['Deck_x_Side'] = combined['Deck_enc'] * 2 + combined['Side_enc']

# ============================================================
# 8. Group-level features
# ============================================================
group_spending = combined.groupby('Group')['TotalSpending'].agg(['mean', 'std', 'max'])
group_spending.columns = ['GroupSpend_mean', 'GroupSpend_std', 'GroupSpend_max']
group_spending['GroupSpend_std'] = group_spending['GroupSpend_std'].fillna(0)
combined = combined.merge(group_spending, on='Group', how='left')

group_age = combined.groupby('Group')['Age'].agg(['mean', 'std'])
group_age.columns = ['GroupAge_mean', 'GroupAge_std']
group_age['GroupAge_std'] = group_age['GroupAge_std'].fillna(0)
combined = combined.merge(group_age, on='Group', how='left')

combined['SpendingAnomaly'] = combined['TotalSpending'] - combined['GroupSpend_mean']

# Frequency encodings
for col in ['Deck', 'Side', 'HomePlanet', 'Destination']:
    freq = combined.iloc[:n_train][col].value_counts(normalize=True).to_dict()
    combined[f'{col}_freq'] = combined[col].map(freq).fillna(0)

combined['CabinNumBin'] = pd.qcut(combined['CabinNum'], q=10, labels=False, duplicates='drop')
combined['CabinNumBin'] = combined['CabinNumBin'].fillna(0)

# ============================================================
# 9. NEW: Group & Surname Transport Rate (leakage-safe for CV)
# ============================================================
# These are computed ONLY for submission (full train data)
# For CV, we'll recompute inside the fold

train_with_target = combined.iloc[:n_train].copy()
train_with_target[TARGET] = y.values

# Group transport rate (using full train for feature importance check)
group_stats = train_with_target.groupby('Group').agg(
    g_count=('Group', 'size'), g_transported=(TARGET, 'sum')
)
group_stats['Group_Transport_Rate'] = np.where(
    group_stats['g_count'] >= 2,
    group_stats['g_transported'] / group_stats['g_count'],
    DEFAULT_RATE
)
combined['Group_Transport_Rate'] = combined['Group'].map(
    group_stats['Group_Transport_Rate'].to_dict()
).fillna(DEFAULT_RATE)

# Surname transport rate
surname_stats = train_with_target.groupby('Surname').agg(
    s_count=('Surname', 'size'), s_transported=(TARGET, 'sum')
)
surname_stats['Surname_Transport_Rate'] = np.where(
    surname_stats['s_count'] >= 2,
    surname_stats['s_transported'] / surname_stats['s_count'],
    DEFAULT_RATE
)
combined['Surname_Transport_Rate'] = combined['Surname'].map(
    surname_stats['Surname_Transport_Rate'].to_dict()
).fillna(DEFAULT_RATE)

# Combined rate
combined['Combined_Transport_Rate'] = (
    combined['Group_Transport_Rate'] + combined['Surname_Transport_Rate']
) / 2.0

# ============================================================
# 10. NEW: Spending clusters
# ============================================================
spend_data = combined[spending_cols].copy()
spend_log = np.log1p(spend_data)
kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10)
combined['SpendingCluster'] = kmeans.fit_predict(spend_log)

# Fill any remaining NaN
for col in combined.select_dtypes(include=[np.number]).columns:
    if combined[col].isnull().any():
        combined[col] = combined[col].fillna(combined.iloc[:n_train][col].median())

# ============================================================
# Prepare feature matrix
# ============================================================
drop_cols = [TARGET, ID, 'Name', 'Cabin', 'HomePlanet', 'Destination',
             'Deck', 'Side', 'CryoSleep', 'VIP', 'Group', 'Surname']
feature_cols = [c for c in combined.columns if c not in drop_cols]

X = combined.iloc[:n_train][feature_cols].copy()
X_test = combined.iloc[n_train:][feature_cols].copy()

print(f"Features: {len(feature_cols)}")
print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"NaN: train={X.isnull().sum().sum()}, test={X_test.isnull().sum().sum()}")

# New features added
new_feats = ['Group_Transport_Rate', 'Surname_Transport_Rate', 'Combined_Transport_Rate',
             'SpendingCluster', 'HomePlanet_freq', 'Destination_freq', 'Side_freq']
print(f"New features: {new_feats}")


# ============================================================
# HELPER
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


# ============================================================
# Leakage-safe CV with transport rate recomputation
# ============================================================
def compute_transport_rates_safe(X_tr_idx, X_val_idx, combined_df, y_series):
    """Recompute group/surname transport rates using only training fold."""
    train_fold = combined_df.iloc[X_tr_idx].copy()
    train_fold[TARGET] = y_series.iloc[X_tr_idx].values

    # Group rate
    g_stats = train_fold.groupby('Group').agg(
        cnt=('Group', 'size'), transp=(TARGET, 'sum')
    )
    g_stats['rate'] = np.where(g_stats['cnt'] >= 2,
                               g_stats['transp'] / g_stats['cnt'], DEFAULT_RATE)
    g_rate_map = g_stats['rate'].to_dict()

    # Surname rate
    s_stats = train_fold.groupby('Surname').agg(
        cnt=('Surname', 'size'), transp=(TARGET, 'sum')
    )
    s_stats['rate'] = np.where(s_stats['cnt'] >= 2,
                               s_stats['transp'] / s_stats['cnt'], DEFAULT_RATE)
    s_rate_map = s_stats['rate'].to_dict()

    return g_rate_map, s_rate_map


def apply_rates(X_subset, combined_subset, g_map, s_map):
    """Apply transport rate maps to a subset."""
    X_out = X_subset.copy()
    X_out['Group_Transport_Rate'] = combined_subset['Group'].map(g_map).fillna(DEFAULT_RATE).values
    X_out['Surname_Transport_Rate'] = combined_subset['Surname'].map(s_map).fillna(DEFAULT_RATE).values
    X_out['Combined_Transport_Rate'] = (
        X_out['Group_Transport_Rate'] + X_out['Surname_Transport_Rate']
    ) / 2.0
    return X_out


# Store combined info for rate computation
combined_info = combined.iloc[:n_train][['Group', 'Surname']].copy()

PREV_BEST = 0.814218  # From self_improvement.py

# ============================================================
# MODEL EVALUATION
# ============================================================
print("\n" + "="*60)
print("  Adaptive Search: Model Evaluation with Transport Rates")
print("="*60)

def run_cv_safe(model_fn, model_name, params, notes=""):
    """CV with leakage-safe transport rate recomputation."""
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    test_probs = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Recompute rates for this fold
        g_map, s_map = compute_transport_rates_safe(
            train_idx, val_idx, combined_info, y
        )
        X_tr = apply_rates(X.iloc[train_idx], combined_info.iloc[train_idx], g_map, s_map)
        X_val = apply_rates(X.iloc[val_idx], combined_info.iloc[val_idx], g_map, s_map)

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


# Model 1: XGBoost
print("\n--- XGBoost (adaptive) ---")
xgb_params = {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 5,
              "subsample": 0.8, "colsample_bytree": 0.7,
              "reg_alpha": 0.3, "reg_lambda": 1.0}
xgb_score, _, xgb_probs = run_cv_safe(
    lambda p: xgb.XGBClassifier(**p, random_state=SEED, verbosity=0, eval_metric="logloss"),
    "XGBoost-Adaptive", xgb_params,
    "XGBoost with transport rates + smart imputation + spending clusters"
)

# Model 2: LightGBM
print("\n--- LightGBM (adaptive) ---")
lgb_params = {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 25,
              "max_depth": 5, "min_child_samples": 20,
              "subsample": 0.8, "colsample_bytree": 0.7,
              "reg_alpha": 0.5, "reg_lambda": 1.0}
lgb_score, _, lgb_probs = run_cv_safe(
    lambda p: lgb.LGBMClassifier(verbosity=-1, random_state=SEED, **p),
    "LightGBM-Adaptive", lgb_params,
    "LightGBM with transport rates + smart imputation + spending clusters"
)

# Model 3: CatBoost
print("\n--- CatBoost (adaptive) ---")
cb_params = {"iterations": 500, "learning_rate": 0.05, "depth": 6,
             "l2_leaf_reg": 3.0, "random_seed": SEED, "verbose": 0}
cb_score, _, cb_probs = run_cv_safe(
    lambda p: CatBoostClassifier(**p),
    "CatBoost-Adaptive", cb_params,
    "CatBoost with transport rates + smart imputation + spending clusters"
)

# Model 4: 4-model ensemble
print("\n--- 4-model Soft Voting (adaptive) ---")
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
ens_scores = []
ens_probs = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    g_map, s_map = compute_transport_rates_safe(train_idx, val_idx, combined_info, y)
    X_tr = apply_rates(X.iloc[train_idx], combined_info.iloc[train_idx], g_map, s_map)
    X_val = apply_rates(X.iloc[val_idx], combined_info.iloc[val_idx], g_map, s_map)
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    m_xgb = xgb.XGBClassifier(**xgb_params, random_state=SEED, verbosity=0, eval_metric="logloss")
    m_lgb = lgb.LGBMClassifier(verbosity=-1, random_state=SEED, **lgb_params)
    m_cb = CatBoostClassifier(**cb_params)
    m_rf = RandomForestClassifier(n_estimators=800, max_depth=20, min_samples_split=3,
                                   max_features="sqrt", random_state=SEED, n_jobs=-1)

    ens = VotingClassifier(
        estimators=[("xgb", m_xgb), ("lgb", m_lgb), ("cb", m_cb), ("rf", m_rf)],
        voting="soft"
    )
    ens.fit(X_tr, y_tr)
    preds = ens.predict(X_val)
    score = accuracy_score(y_val, preds)
    ens_scores.append(score)
    ens_probs += ens.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  Fold {fold+1}: {score:.6f}")

ens_mean = np.mean(ens_scores)
log_experiment("SoftVoting-4Model-Adaptive",
               {"models": "XGB+LGB+CB+RF", "new_features": "transport_rates+clusters"},
               ens_scores, X.shape[1],
               "4-model ensemble with transport rates + smart imputation")
print(f"  Mean: {ens_mean:.6f} (+/- {np.std(ens_scores):.6f})")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("  SUMMARY — Verifiable Rewards")
print("="*60)

all_results = {
    "XGBoost-Adaptive": (xgb_score, xgb_probs),
    "LightGBM-Adaptive": (lgb_score, lgb_probs),
    "CatBoost-Adaptive": (cb_score, cb_probs),
    "SoftVoting-Adaptive": (ens_mean, ens_probs),
}

print(f"\nPrevious best CV: {PREV_BEST:.6f} (SoftVoting-4Model, 43 features)")
print(f"Current features: {X.shape[1]}")
for name, (score, _) in sorted(all_results.items(), key=lambda x: -x[1][0]):
    delta = score - PREV_BEST
    signal = "+++" if delta > 0.005 else ("++" if delta > 0.001 else ("=" if abs(delta) < 0.001 else "-"))
    print(f"  {signal} {name:<30} {score:.6f} ({delta:+.4f})")

best_name = max(all_results, key=lambda k: all_results[k][0])
best_score = all_results[best_name][0]
best_probs = all_results[best_name][1]

print(f"\nBest: {best_name} = {best_score:.6f}")
print(f"Delta vs previous best: {best_score - PREV_BEST:+.4f}")

# Feature importance (XGBoost)
print("\n--- XGBoost Feature Importance (top 15) ---")
m_final = xgb.XGBClassifier(**xgb_params, random_state=SEED, verbosity=0, eval_metric="logloss")
m_final.fit(X, y)
imp = pd.Series(m_final.feature_importances_, index=X.columns).sort_values(ascending=False)
for feat, val in imp.head(15).items():
    marker = " <<<" if feat in new_feats else ""
    print(f"  {feat:<30} {val:.4f}{marker}")

# Save submissions
final_preds = (best_probs >= 0.5).astype(bool)
print(f"\nFinal predictions (threshold=0.50):")
print(f"  Transported=True:  {final_preds.sum()} ({final_preds.mean()*100:.1f}%)")
print(f"  Transported=False: {(~final_preds).sum()} ({(~final_preds).mean()*100:.1f}%)")

submission = pd.DataFrame({
    "PassengerId": test_ids.values,
    "Transported": final_preds
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"submission_adaptive_{best_score:.4f}_{timestamp}.csv"
output_path = os.path.join(COMPETITION_DIR, "submissions", fname)
submission.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
