"""
Full Pipeline for Spaceship Titanic
=====================================
Binary classification, metric: accuracy.
8,693 train / 4,277 test. Nearly balanced (50.4% transported).
Mixed features: categorical + numerical. Missing values throughout.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
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
train_ids = train[ID]
test_ids = test[ID]

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target: {y.value_counts().to_dict()}\n")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================
# EDA HIGHLIGHTS
# ============================================================
section("EDA HIGHLIGHTS")

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# CryoSleep vs Transported
cryo = train.dropna(subset=['CryoSleep'])
print("CryoSleep vs Transported:")
print(cryo.groupby('CryoSleep')[TARGET].mean().to_string())

# HomePlanet vs Transported
print(f"\nHomePlanet vs Transported:")
print(train.groupby('HomePlanet')[TARGET].mean().to_string())

# Destination vs Transported
print(f"\nDestination vs Transported:")
print(train.groupby('Destination')[TARGET].mean().to_string())

# CryoSleep passengers spend $0
cryo_true = train[train['CryoSleep'] == True]
print(f"\nCryoSleep=True, mean spending: {cryo_true[spending_cols].mean().mean():.2f}")

# Missing values
print(f"\nMissing values:")
for col in train.columns:
    n = train[col].isnull().sum()
    if n > 0:
        print(f"  {col}: {n} ({n/len(train)*100:.1f}%)")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
section("FEATURE ENGINEERING")

n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)

# 1. Parse PassengerId → group and person number
combined['Group'] = combined[ID].apply(lambda x: int(x.split('_')[0]))
combined['PersonInGroup'] = combined[ID].apply(lambda x: int(x.split('_')[1]))
combined['GroupSize'] = combined.groupby('Group')['Group'].transform('count')
combined['IsAlone'] = (combined['GroupSize'] == 1).astype(int)

# 2. Parse Cabin → Deck, Num, Side
combined['Deck'] = combined['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else np.nan)
combined['CabinNum'] = combined['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) else np.nan)
combined['Side'] = combined['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else np.nan)

# 3. Fill missing values
# CryoSleep and VIP: fill with mode (False)
combined['CryoSleep'] = combined['CryoSleep'].fillna(False).astype(bool)
combined['VIP'] = combined['VIP'].fillna(False).astype(bool)

# Age: fill with median
combined['Age'] = combined['Age'].fillna(combined['Age'].median())

# Spending: fill with 0 (most common value, especially for CryoSleep passengers)
for col in spending_cols:
    combined[col] = combined[col].fillna(0)

# HomePlanet, Destination: fill with mode
combined['HomePlanet'] = combined['HomePlanet'].fillna(combined['HomePlanet'].mode()[0])
combined['Destination'] = combined['Destination'].fillna(combined['Destination'].mode()[0])

# Deck, Side: fill with mode
combined['Deck'] = combined['Deck'].fillna(combined['Deck'].mode()[0])
combined['Side'] = combined['Side'].fillna(combined['Side'].mode()[0])
combined['CabinNum'] = combined['CabinNum'].fillna(combined['CabinNum'].median())

# 4. Spending features
combined['TotalSpending'] = combined[spending_cols].sum(axis=1)
combined['HasSpent'] = (combined['TotalSpending'] > 0).astype(int)
combined['SpendingPerFeature'] = combined['TotalSpending'] / 5

# Log-transform spending (highly skewed)
for col in spending_cols:
    combined[f'{col}_log'] = np.log1p(combined[col])
combined['TotalSpending_log'] = np.log1p(combined['TotalSpending'])

# Spending ratios
combined['LuxurySpending'] = combined['Spa'] + combined['VRDeck'] + combined['RoomService']
combined['BasicSpending'] = combined['FoodCourt'] + combined['ShoppingMall']
combined['LuxuryRatio'] = combined['LuxurySpending'] / (combined['TotalSpending'] + 1)

# 5. Age features
combined['IsChild'] = (combined['Age'] < 13).astype(int)
combined['IsTeenager'] = ((combined['Age'] >= 13) & (combined['Age'] < 18)).astype(int)
combined['AgeBin'] = pd.cut(combined['Age'], bins=[0, 12, 18, 30, 50, 80], labels=[0, 1, 2, 3, 4]).astype(float)

# 6. CryoSleep interaction
combined['CryoSleep_int'] = combined['CryoSleep'].astype(int)
combined['VIP_int'] = combined['VIP'].astype(int)

# 7. Encode categoricals
le_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
label_encoders = {}
for col in le_cols:
    le = LabelEncoder()
    combined[col + '_enc'] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le

# 8. Drop original non-numeric columns
drop_cols = [TARGET, ID, 'Name', 'Cabin', 'HomePlanet', 'Destination',
             'Deck', 'Side', 'CryoSleep', 'VIP']
feature_cols = [c for c in combined.columns if c not in drop_cols]

print(f"Features: {len(feature_cols)}")

# Split back
X = combined.iloc[:n_train][feature_cols].copy()
X_test = combined.iloc[n_train:][feature_cols].copy()

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"NaN check: train={X.isnull().sum().sum()}, test={X_test.isnull().sum().sum()}")
print(f"Feature list: {list(X.columns)}")


# ============================================================
# MODELING
# ============================================================
section("MODELING")


def log_experiment(model_name, params, scores, notes=""):
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    with open(exp_file, "r") as f:
        experiments = json.load(f)
    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "n_features": X.shape[1],
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


def cv_model(model_fn, model_name, params, notes=""):
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    test_preds_all = np.zeros((len(X_test), 2))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn(params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        score = accuracy_score(y_val, preds)
        scores.append(score)

        test_preds_all += model.predict_proba(X_test) / N_FOLDS
        print(f"  Fold {fold+1}: accuracy={score:.6f}")

    mean_score = np.mean(scores)
    exp_id = log_experiment(model_name, params, scores, notes)
    print(f"  Mean: {mean_score:.6f} (+/- {np.std(scores):.6f}) [Experiment #{exp_id}]\n")
    return mean_score, test_preds_all


# MODEL 1: Random Forest
print("--- Random Forest ---")
rf_params = {"n_estimators": 500, "max_depth": 15, "min_samples_split": 5, "max_features": "sqrt"}
rf_score, rf_test = cv_model(
    lambda p: RandomForestClassifier(**p, random_state=SEED, n_jobs=-1),
    "RandomForest", rf_params, "Tuned RF for binary classification"
)

# MODEL 2: Extra Trees
print("--- Extra Trees ---")
et_params = {"n_estimators": 500, "max_depth": 15, "min_samples_split": 5, "max_features": "sqrt"}
et_score, et_test = cv_model(
    lambda p: ExtraTreesClassifier(**p, random_state=SEED, n_jobs=-1),
    "ExtraTrees", et_params, "Tuned Extra Trees"
)

# MODEL 3: LightGBM
print("--- LightGBM ---")
lgbm_params = {
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
    "max_depth": -1, "min_child_samples": 20, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1
}
lgbm_score, lgbm_test = cv_model(
    lambda p: lgb.LGBMClassifier(verbosity=-1, random_state=SEED, **p),
    "LightGBM", lgbm_params, "Tuned LightGBM binary"
)

# MODEL 4: XGBoost
print("--- XGBoost ---")
xgb_params = {
    "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0
}
xgb_score, xgb_test = cv_model(
    lambda p: xgb.XGBClassifier(**p, random_state=SEED, verbosity=0, eval_metric="logloss"),
    "XGBoost", xgb_params, "Tuned XGBoost binary"
)


# ============================================================
# ENSEMBLE
# ============================================================
section("ENSEMBLE")

# Weighted average of probabilities
ensemble_test = 0.2 * rf_test + 0.2 * et_test + 0.3 * lgbm_test + 0.3 * xgb_test
ensemble_preds = (ensemble_test[:, 1] >= 0.5).astype(bool)

print(f"Ensemble prediction distribution:")
print(f"  Transported=True:  {ensemble_preds.sum()} ({ensemble_preds.mean()*100:.1f}%)")
print(f"  Transported=False: {(~ensemble_preds).sum()} ({(~ensemble_preds).mean()*100:.1f}%)")


# ============================================================
# LEADERBOARD
# ============================================================
section("EXPERIMENT LEADERBOARD")

with open(os.path.join(COMPETITION_DIR, "experiments.json"), "r") as f:
    experiments = json.load(f)

sorted_exps = sorted(experiments, key=lambda e: e["cv_mean"], reverse=True)
print(f"{'#':<4} {'ID':<5} {'Model':<25} {'CV Acc':<12} {'CV Std':<10}")
print("-" * 58)
for rank, exp in enumerate(sorted_exps, 1):
    print(f"{rank:<4} {exp['experiment_id']:<5} {exp['model'][:24]:<25} "
          f"{exp['cv_mean']:<12.6f} {exp['cv_std']:<10.6f}")

best = sorted_exps[0]
print(f"\nBest single model: {best['model']} with accuracy={best['cv_mean']:.6f}")


# ============================================================
# SUBMISSION
# ============================================================
section("SUBMISSION")

submission = pd.DataFrame({
    "PassengerId": test_ids.values,
    "Transported": ensemble_preds
})

# Validation
shape_ok = submission.shape == sample_sub.shape
cols_ok = list(submission.columns) == list(sample_sub.columns)
nan_count = submission.isnull().sum().sum()
id_match = (submission["PassengerId"].values == sample_sub["PassengerId"].values).all()
values_ok = submission["Transported"].isin([True, False]).all()

print(f"  Shape: {'PASS' if shape_ok else 'FAIL'} ({submission.shape})")
print(f"  Columns: {'PASS' if cols_ok else 'FAIL'}")
print(f"  NaN: {'PASS' if nan_count == 0 else 'FAIL'}")
print(f"  IDs: {'PASS' if id_match else 'FAIL'}")
print(f"  Values [True/False]: {'PASS' if values_ok else 'FAIL'}")

all_pass = shape_ok and cols_ok and nan_count == 0 and id_match and values_ok

if all_pass:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_cv = best['cv_mean']
    filename = f"submission_ensemble_{best_cv:.4f}_{timestamp}.csv"
    output_dir = os.path.join(COMPETITION_DIR, "submissions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")
else:
    print("\nValidation FAILED.")
