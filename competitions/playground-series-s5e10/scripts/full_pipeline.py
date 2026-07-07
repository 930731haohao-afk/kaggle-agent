"""
Full Pipeline for Playground Series S5E10
==========================================
Regression, predict road accident risk (0-1).
517K train / 173K test. No missing values.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/playground-series-s5e10"
data_dir = os.path.join(COMPETITION_DIR, "data")
TARGET = "accident_risk"
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
print(f"Target: mean={y.mean():.4f}, std={y.std():.4f}, range=[{y.min():.2f}, {y.max():.2f}]")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================
# EDA HIGHLIGHTS
# ============================================================
section("EDA HIGHLIGHTS")

print("Numeric correlations with accident_risk:")
for col in ['curvature', 'speed_limit', 'num_reported_accidents', 'num_lanes']:
    corr = train[col].corr(y)
    print(f"  {col}: {corr:.4f}")

print("\nKey categorical effects:")
for col in ['lighting', 'weather', 'road_type']:
    means = train.groupby(col)[TARGET].mean()
    effect_range = means.max() - means.min()
    print(f"  {col}: range={effect_range:.3f} ({means.idxmin()}={means.min():.3f} to {means.idxmax()}={means.max():.3f})")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
section("FEATURE ENGINEERING")

n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)

# 1. Ordinal/label encode categoricals
road_type_map = {'rural': 0, 'urban': 1, 'highway': 2}
lighting_map = {'daylight': 0, 'dim': 1, 'night': 2}
weather_map = {'clear': 0, 'rainy': 1, 'foggy': 2}
time_map = {'morning': 0, 'afternoon': 1, 'evening': 2}

combined['road_type_enc'] = combined['road_type'].map(road_type_map)
combined['lighting_enc'] = combined['lighting'].map(lighting_map)
combined['weather_enc'] = combined['weather'].map(weather_map)
combined['time_of_day_enc'] = combined['time_of_day'].map(time_map)

# 2. Boolean to int
combined['road_signs_int'] = combined['road_signs_present'].astype(int)
combined['public_road_int'] = combined['public_road'].astype(int)
combined['holiday_int'] = combined['holiday'].astype(int)
combined['school_season_int'] = combined['school_season'].astype(int)

# 3. Interactions with top predictors
combined['curvature_x_speed'] = combined['curvature'] * combined['speed_limit']
combined['curvature_x_lighting'] = combined['curvature'] * combined['lighting_enc']
combined['speed_x_lighting'] = combined['speed_limit'] * combined['lighting_enc']
combined['curvature_x_weather'] = combined['curvature'] * combined['weather_enc']
combined['speed_x_weather'] = combined['speed_limit'] * combined['weather_enc']
combined['curvature_x_accidents'] = combined['curvature'] * combined['num_reported_accidents']
combined['speed_x_accidents'] = combined['speed_limit'] * combined['num_reported_accidents']

# 4. Polynomial features for top predictors
combined['curvature_sq'] = combined['curvature'] ** 2
combined['speed_sq'] = combined['speed_limit'] ** 2
combined['curvature_x_speed_sq'] = combined['curvature'] * combined['speed_limit'] ** 2

# 5. Risk factor combinations
combined['night_rain'] = ((combined['lighting_enc'] == 2) & (combined['weather_enc'] == 1)).astype(int)
combined['night_fog'] = ((combined['lighting_enc'] == 2) & (combined['weather_enc'] == 2)).astype(int)
combined['high_speed_curve'] = combined['curvature'] * (combined['speed_limit'] >= 60).astype(int)

# 6. Lanes features
combined['lanes_x_speed'] = combined['num_lanes'] * combined['speed_limit']
combined['lanes_x_curvature'] = combined['num_lanes'] * combined['curvature']

# Drop original categorical and ID columns
drop_cols = [TARGET, ID, 'road_type', 'lighting', 'weather', 'time_of_day',
             'road_signs_present', 'public_road', 'holiday', 'school_season']
feature_cols = [c for c in combined.columns if c not in drop_cols]

print(f"Features: {len(feature_cols)}")

# Split back
X = combined.iloc[:n_train][feature_cols].copy()
X_test = combined.iloc[n_train:][feature_cols].copy()

print(f"Train: {X.shape}, Test: {X_test.shape}")
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
        "cv_strategy": f"{N_FOLDS}-fold",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 6),
        "cv_std": round(float(np.std(scores)), 6),
        "eval_metric": "RMSE",
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
        preds = np.clip(preds, 0, 1)
        score = root_mean_squared_error(y_val, preds)
        scores.append(score)
        oof_preds[val_idx] = preds
        test_preds_all += np.clip(model.predict(X_test), 0, 1) / N_FOLDS
        print(f"  Fold {fold+1}: RMSE={score:.6f}")

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

# MODEL 3: LightGBM v2
print("--- LightGBM v2 ---")
lgbm2_params = {
    "n_estimators": 1200, "learning_rate": 0.03, "num_leaves": 127,
    "max_depth": -1, "min_child_samples": 20, "subsample": 0.7,
    "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 0.5
}
lgbm2_score, lgbm2_test, lgbm2_oof = cv_model(
    lambda p: lgb.LGBMRegressor(verbosity=-1, random_state=SEED, n_jobs=-1, **p),
    "LightGBM_v2", lgbm2_params, "LightGBM more trees + regularization"
)


# ============================================================
# ENSEMBLE
# ============================================================
section("ENSEMBLE")

best_rmse = 999
best_weights = None
for w1 in np.arange(0.1, 0.8, 0.1):
    for w2 in np.arange(0.1, 1.0 - w1, 0.1):
        w3 = round(1.0 - w1 - w2, 1)
        if w3 < 0.05:
            continue
        oof_blend = w1 * lgbm_oof + w2 * xgb_oof + w3 * lgbm2_oof
        rmse = root_mean_squared_error(y, oof_blend)
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights
print(f"Best weights: LGBM={w1:.1f}, XGB={w2:.1f}, LGBM_v2={w3:.1f}")
print(f"Ensemble OOF RMSE: {best_rmse:.6f}")

ensemble_test = w1 * lgbm_test + w2 * xgb_test + w3 * lgbm2_test
ensemble_test = np.clip(ensemble_test, 0, 1)

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
    "cv_scores": [round(best_rmse, 6)],
    "cv_mean": round(best_rmse, 6),
    "cv_std": 0.0,
    "eval_metric": "RMSE",
    "notes": f"Weighted ensemble, OOF RMSE={best_rmse:.6f}"
})
with open(exp_file, "w") as f:
    json.dump(experiments, f, indent=2)

print(f"\nEnsemble prediction stats:")
print(f"  Min: {ensemble_test.min():.4f}, Max: {ensemble_test.max():.4f}, Mean: {ensemble_test.mean():.4f}")


# ============================================================
# LEADERBOARD
# ============================================================
section("EXPERIMENT LEADERBOARD")

with open(os.path.join(COMPETITION_DIR, "experiments.json"), "r") as f:
    experiments = json.load(f)

sorted_exps = sorted(experiments, key=lambda e: e["cv_mean"])
print(f"{'#':<4} {'ID':<5} {'Model':<25} {'CV RMSE':<12} {'CV Std':<10}")
print("-" * 58)
for rank, exp in enumerate(sorted_exps, 1):
    print(f"{rank:<4} {exp['experiment_id']:<5} {exp['model'][:24]:<25} "
          f"{exp['cv_mean']:<12.6f} {exp['cv_std']:<10.6f}")

best = sorted_exps[0]
print(f"\nBest: {best['model']} with RMSE={best['cv_mean']:.6f}")


# ============================================================
# SUBMISSION
# ============================================================
section("SUBMISSION")

submission = pd.DataFrame({
    "id": test_ids.values,
    "accident_risk": ensemble_test
})

shape_ok = submission.shape == sample_sub.shape
cols_ok = list(submission.columns) == list(sample_sub.columns)
nan_count = submission.isnull().sum().sum()
id_match = (submission["id"].values == sample_sub["id"].values).all()
values_ok = (submission["accident_risk"] >= 0).all() and (submission["accident_risk"] <= 1).all()

print(f"  Shape: {'PASS' if shape_ok else 'FAIL'} ({submission.shape})")
print(f"  Columns: {'PASS' if cols_ok else 'FAIL'}")
print(f"  NaN: {'PASS' if nan_count == 0 else 'FAIL'}")
print(f"  IDs: {'PASS' if id_match else 'FAIL'}")
print(f"  Values [0,1]: {'PASS' if values_ok else 'FAIL'}")

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
