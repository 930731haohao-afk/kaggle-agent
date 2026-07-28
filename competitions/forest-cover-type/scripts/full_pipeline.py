"""
Full Pipeline for Forest Cover Type Prediction
================================================
7-class classification, metric: accuracy.
15K train (balanced), 566K test.
Features: 10 continuous + 4 wilderness area + 40 soil type (all numeric).
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/forest-cover-type"
data_dir = os.path.join(COMPETITION_DIR, "data")
TARGET = "Cover_Type"
ID = "Id"
N_FOLDS = 5
SEED = 42

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sampleSubmission.csv"))

y_orig = train[TARGET]
y = train[TARGET] - 1  # Shift to 0-indexed for XGBoost compatibility
train_ids = train[ID]
test_ids = test[ID]
X = train.drop(columns=[TARGET, ID])
X_test = test.drop(columns=[ID])

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Classes: {sorted(y.unique())}, each has {y.value_counts().iloc[0]} samples\n")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================
# EDA HIGHLIGHTS
# ============================================================
section("EDA HIGHLIGHTS")

# Constant columns
const_cols = [c for c in X.columns if X[c].nunique() <= 1]
print(f"Constant columns (drop): {const_cols}")

# Drop constant columns
X = X.drop(columns=const_cols)
X_test = X_test.drop(columns=const_cols)

# Continuous features
cont_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
             'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
             'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
             'Horizontal_Distance_To_Fire_Points']

# Top features by class separation
print(f"\nMean Elevation by Cover_Type:")
for ct in sorted(y.unique()):
    mask = y == ct
    print(f"  Type {ct}: {train.loc[mask, 'Elevation'].mean():.0f}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
section("FEATURE ENGINEERING")

# Combine train and test for consistent processing
n_train = len(X)
combined = pd.concat([X, X_test], axis=0, ignore_index=True)

# 1. Distance features
combined['Dist_Hydro'] = np.sqrt(
    combined['Horizontal_Distance_To_Hydrology']**2 +
    combined['Vertical_Distance_To_Hydrology']**2
)
combined['Dist_Road_Fire_Sum'] = (
    combined['Horizontal_Distance_To_Roadways'] +
    combined['Horizontal_Distance_To_Fire_Points']
)
combined['Dist_Road_Fire_Diff'] = abs(
    combined['Horizontal_Distance_To_Roadways'] -
    combined['Horizontal_Distance_To_Fire_Points']
)

# 2. Hillshade features
combined['Hillshade_Mean'] = (
    combined['Hillshade_9am'] + combined['Hillshade_Noon'] + combined['Hillshade_3pm']
) / 3
combined['Hillshade_9am_Noon_Diff'] = abs(combined['Hillshade_9am'] - combined['Hillshade_Noon'])
combined['Hillshade_Noon_3pm_Diff'] = abs(combined['Hillshade_Noon'] - combined['Hillshade_3pm'])

# 3. Aspect features (cyclical)
combined['Aspect_sin'] = np.sin(np.radians(combined['Aspect']))
combined['Aspect_cos'] = np.cos(np.radians(combined['Aspect']))

# 4. Elevation features
combined['Elev_VDist'] = combined['Elevation'] - combined['Vertical_Distance_To_Hydrology']
combined['Elev_HDist_Hydro'] = combined['Elevation'] - combined['Horizontal_Distance_To_Hydrology']

# 5. Log transforms on skewed distance features
for col in ['Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points']:
    combined[f'{col}_log'] = np.log1p(combined[col])

# 6. Wilderness and Soil type aggregations
wilderness_cols = [c for c in combined.columns if c.startswith('Wilderness_Area')]
soil_cols = [c for c in combined.columns if c.startswith('Soil_Type')]

# Which wilderness area (convert one-hot back to single column)
combined['Wilderness_Type'] = 0
for i, col in enumerate(wilderness_cols):
    combined.loc[combined[col] == 1, 'Wilderness_Type'] = i + 1

# Which soil type (convert one-hot back to single column)
combined['Soil_Type'] = 0
for i, col in enumerate(soil_cols):
    combined.loc[combined[col] == 1, 'Soil_Type'] = i + 1

print(f"Features before: {len(X.columns)}")

# Split back
X = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

print(f"Features after:  {len(X.columns)}")
print(f"New features: {[c for c in X.columns if c not in train.columns]}")
print(f"NaN check: train={X.isnull().sum().sum()}, test={X_test.isnull().sum().sum()}")


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
    test_preds_all = np.zeros((len(X_test), 7))

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
rf_params = {"n_estimators": 500, "max_depth": 25, "min_samples_split": 3, "max_features": "sqrt"}
rf_score, rf_test = cv_model(
    lambda p: RandomForestClassifier(**p, random_state=SEED, n_jobs=-1),
    "RandomForest", rf_params, "Tuned Random Forest"
)

# MODEL 2: Extra Trees
print("--- Extra Trees ---")
et_params = {"n_estimators": 500, "max_depth": 25, "min_samples_split": 3, "max_features": "sqrt"}
et_score, et_test = cv_model(
    lambda p: ExtraTreesClassifier(**p, random_state=SEED, n_jobs=-1),
    "ExtraTrees", et_params, "Tuned Extra Trees"
)

# MODEL 3: LightGBM
print("--- LightGBM ---")
lgbm_params = {
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
    "max_depth": -1, "min_child_samples": 10, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
    "num_class": 7
}
lgbm_score, lgbm_test = cv_model(
    lambda p: lgb.LGBMClassifier(verbosity=-1, random_state=SEED,
                                  objective="multiclass", **{k:v for k,v in p.items() if k != "num_class"}),
    "LightGBM", lgbm_params, "Tuned LightGBM multiclass"
)

# MODEL 4: XGBoost
print("--- XGBoost ---")
xgb_params = {
    "n_estimators": 500, "learning_rate": 0.05, "max_depth": 8,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0
}
xgb_score, xgb_test = cv_model(
    lambda p: xgb.XGBClassifier(**p, random_state=SEED, use_label_encoder=False,
                                 verbosity=0, eval_metric="mlogloss"),
    "XGBoost", xgb_params, "Tuned XGBoost multiclass"
)


# ============================================================
# ENSEMBLE
# ============================================================
section("ENSEMBLE")

# Average probabilities from all 4 models
ensemble_test = 0.2 * rf_test + 0.2 * et_test + 0.3 * lgbm_test + 0.3 * xgb_test
ensemble_preds = np.argmax(ensemble_test, axis=1) + 1  # Classes are 1-7

# Evaluate ensemble on OOF (approximate via majority vote)
# For a proper OOF ensemble, we'd need OOF probabilities — use test predictions for submission
print(f"Ensemble prediction distribution:")
for ct in range(1, 8):
    count = (ensemble_preds == ct).sum()
    print(f"  Type {ct}: {count} ({count/len(ensemble_preds)*100:.1f}%)")


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
    "Id": test_ids.values.astype(int),
    "Cover_Type": ensemble_preds.astype(int)
})

# Validation
shape_ok = submission.shape == sample_sub.shape
cols_ok = list(submission.columns) == list(sample_sub.columns)
nan_count = submission.isnull().sum().sum()
id_match = (submission["Id"].values == sample_sub["Id"].values).all()
values_ok = submission["Cover_Type"].isin(range(1, 8)).all()

print(f"  Shape: {'PASS' if shape_ok else 'FAIL'} ({submission.shape})")
print(f"  Columns: {'PASS' if cols_ok else 'FAIL'}")
print(f"  NaN: {'PASS' if nan_count == 0 else 'FAIL'}")
print(f"  IDs: {'PASS' if id_match else 'FAIL'}")
print(f"  Values [1-7]: {'PASS' if values_ok else 'FAIL'}")

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
