"""
Full Pipeline for Playground Series S6E1
==========================================
Regression, metric: R2 (likely).
Predict student exam scores from study habits and demographics.
630K train / 270K test. No missing values.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/playground-series-s6e1"
data_dir = os.path.join(COMPETITION_DIR, "data")
TARGET = "exam_score"
ID = "id"
N_FOLDS = 5
SEED = 42

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

y = train[TARGET]
train_ids = train[ID]
test_ids = test[ID]

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target: mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.1f}, {y.max():.1f}]\n")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================
# EDA HIGHLIGHTS
# ============================================================
section("EDA HIGHLIGHTS")

print("Correlations with exam_score:")
for col in ['study_hours', 'class_attendance', 'sleep_hours', 'age']:
    corr = train[col].corr(y)
    print(f"  {col}: {corr:.4f}")

print("\nKey categorical effects:")
for col in ['sleep_quality', 'study_method', 'facility_rating']:
    means = train.groupby(col)[TARGET].mean()
    effect_range = means.max() - means.min()
    print(f"  {col}: range={effect_range:.1f} ({means.idxmin()}={means.min():.1f} → {means.idxmax()}={means.max():.1f})")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
section("FEATURE ENGINEERING")

n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)

# 1. Ordinal encoding for ordered categoricals
sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
combined['sleep_quality_ord'] = combined['sleep_quality'].map(sleep_quality_map)
combined['facility_rating_ord'] = combined['facility_rating'].map(facility_rating_map)
combined['exam_difficulty_ord'] = combined['exam_difficulty'].map(exam_difficulty_map)

# 2. Label encode other categoricals
le_gender = LabelEncoder()
combined['gender_enc'] = le_gender.fit_transform(combined['gender'])
le_course = LabelEncoder()
combined['course_enc'] = le_course.fit_transform(combined['course'])
le_internet = LabelEncoder()
combined['internet_enc'] = le_internet.fit_transform(combined['internet_access'])
le_study = LabelEncoder()
combined['study_method_enc'] = le_study.fit_transform(combined['study_method'])

# 3. Interaction features
combined['study_x_attendance'] = combined['study_hours'] * combined['class_attendance']
combined['study_x_sleep_quality'] = combined['study_hours'] * combined['sleep_quality_ord']
combined['study_x_facility'] = combined['study_hours'] * combined['facility_rating_ord']
combined['attendance_x_sleep'] = combined['class_attendance'] * combined['sleep_hours']
combined['study_sq'] = combined['study_hours'] ** 2
combined['attendance_sq'] = combined['class_attendance'] ** 2

# 4. Aggregated study efficiency
combined['total_effort'] = (
    combined['study_hours'] / 8 +          # normalized study hours
    combined['class_attendance'] / 100 +    # normalized attendance
    combined['sleep_quality_ord'] / 2       # normalized sleep quality
) / 3

# 5. Study method effectiveness score
study_method_scores = {'self-study': 0, 'online videos': 1, 'group study': 2, 'mixed': 3, 'coaching': 4}
combined['study_method_score'] = combined['study_method'].map(study_method_scores)
combined['method_x_hours'] = combined['study_method_score'] * combined['study_hours']

# 6. Sleep features
combined['sleep_deficit'] = 8 - combined['sleep_hours']  # deviation from 8 hours
combined['sleep_x_quality'] = combined['sleep_hours'] * combined['sleep_quality_ord']

# Drop original categorical and ID columns
drop_cols = [TARGET, ID, 'gender', 'course', 'internet_access',
             'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
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
        "cv_strategy": f"{N_FOLDS}-fold",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 6),
        "cv_std": round(float(np.std(scores)), 6),
        "eval_metric": "R2",
        "notes": notes
    }
    experiments.append(experiment)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)
    return exp_id


def cv_model(model_fn, model_name, params, notes=""):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    test_preds_all = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn(params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        score = r2_score(y_val, preds)
        scores.append(score)
        oof_preds[val_idx] = preds
        test_preds_all += model.predict(X_test) / N_FOLDS
        print(f"  Fold {fold+1}: R2={score:.6f}")

    mean_score = np.mean(scores)
    exp_id = log_experiment(model_name, params, scores, notes)
    print(f"  Mean: {mean_score:.6f} (+/- {np.std(scores):.6f}) [Experiment #{exp_id}]\n")
    return mean_score, test_preds_all, oof_preds


# MODEL 1: LightGBM
print("--- LightGBM ---")
lgbm_params = {
    "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63,
    "max_depth": -1, "min_child_samples": 30, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1
}
lgbm_score, lgbm_test, lgbm_oof = cv_model(
    lambda p: lgb.LGBMRegressor(verbosity=-1, random_state=SEED, n_jobs=-1, **p),
    "LightGBM", lgbm_params, "Tuned LightGBM"
)

# MODEL 2: XGBoost
print("--- XGBoost ---")
xgb_params = {
    "n_estimators": 800, "learning_rate": 0.05, "max_depth": 7,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0
}
xgb_score, xgb_test, xgb_oof = cv_model(
    lambda p: xgb.XGBRegressor(**p, random_state=SEED, verbosity=0, n_jobs=-1),
    "XGBoost", xgb_params, "Tuned XGBoost"
)

# MODEL 3: LightGBM (more trees, lower LR)
print("--- LightGBM v2 ---")
lgbm2_params = {
    "n_estimators": 1200, "learning_rate": 0.03, "num_leaves": 127,
    "max_depth": -1, "min_child_samples": 20, "subsample": 0.7,
    "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 0.5
}
lgbm2_score, lgbm2_test, lgbm2_oof = cv_model(
    lambda p: lgb.LGBMRegressor(verbosity=-1, random_state=SEED, n_jobs=-1, **p),
    "LightGBM_v2", lgbm2_params, "LightGBM with more trees and regularization"
)


# ============================================================
# ENSEMBLE
# ============================================================
section("ENSEMBLE")

# Find best weights using OOF predictions
best_r2 = -1
best_weights = None
for w1 in np.arange(0.1, 0.8, 0.1):
    for w2 in np.arange(0.1, 1.0 - w1, 0.1):
        w3 = round(1.0 - w1 - w2, 1)
        if w3 < 0.05:
            continue
        oof_blend = w1 * lgbm_oof + w2 * xgb_oof + w3 * lgbm2_oof
        r2 = r2_score(y, oof_blend)
        if r2 > best_r2:
            best_r2 = r2
            best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights
print(f"Best weights: LGBM={w1:.1f}, XGB={w2:.1f}, LGBM_v2={w3:.1f}")
print(f"Ensemble OOF R2: {best_r2:.6f}")

ensemble_test = w1 * lgbm_test + w2 * xgb_test + w3 * lgbm2_test

# Clip to valid range
ensemble_test = np.clip(ensemble_test, 0, 100)

# Log ensemble
exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
with open(exp_file, "r") as f:
    experiments = json.load(f)
exp_id = len(experiments) + 1
experiments.append({
    "experiment_id": exp_id,
    "timestamp": datetime.now().isoformat(),
    "model": "WeightedEnsemble",
    "n_features": X.shape[1],
    "params": {"weights": f"LGBM={w1:.1f}+XGB={w2:.1f}+LGBMv2={w3:.1f}"},
    "cv_strategy": f"{N_FOLDS}-fold",
    "cv_scores": [round(best_r2, 6)],
    "cv_mean": round(best_r2, 6),
    "cv_std": 0.0,
    "eval_metric": "R2",
    "notes": f"Weighted ensemble, OOF R2={best_r2:.6f}"
})
with open(exp_file, "w") as f:
    json.dump(experiments, f, indent=2)

print(f"\nEnsemble prediction stats:")
print(f"  Min: {ensemble_test.min():.2f}, Max: {ensemble_test.max():.2f}")
print(f"  Mean: {ensemble_test.mean():.2f}")


# ============================================================
# LEADERBOARD
# ============================================================
section("EXPERIMENT LEADERBOARD")

with open(os.path.join(COMPETITION_DIR, "experiments.json"), "r") as f:
    experiments = json.load(f)

sorted_exps = sorted(experiments, key=lambda e: e["cv_mean"], reverse=True)
print(f"{'#':<4} {'ID':<5} {'Model':<25} {'CV R2':<12} {'CV Std':<10}")
print("-" * 58)
for rank, exp in enumerate(sorted_exps, 1):
    print(f"{rank:<4} {exp['experiment_id']:<5} {exp['model'][:24]:<25} "
          f"{exp['cv_mean']:<12.6f} {exp['cv_std']:<10.6f}")

best = sorted_exps[0]
print(f"\nBest: {best['model']} with R2={best['cv_mean']:.6f}")


# ============================================================
# SUBMISSION
# ============================================================
section("SUBMISSION")

submission = pd.DataFrame({
    "id": test_ids.values,
    "exam_score": ensemble_test
})

# Validation
shape_ok = submission.shape == sample_sub.shape
cols_ok = list(submission.columns) == list(sample_sub.columns)
nan_count = submission.isnull().sum().sum()
id_match = (submission["id"].values == sample_sub["id"].values).all()

print(f"  Shape: {'PASS' if shape_ok else 'FAIL'} ({submission.shape})")
print(f"  Columns: {'PASS' if cols_ok else 'FAIL'}")
print(f"  NaN: {'PASS' if nan_count == 0 else 'FAIL'}")
print(f"  IDs: {'PASS' if id_match else 'FAIL'}")

all_pass = shape_ok and cols_ok and nan_count == 0 and id_match

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
