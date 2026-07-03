"""Model Training for playground-series-s3e20"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import json
from datetime import datetime

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load processed data
train = pd.read_csv("competitions/playground-series-s3e20/data/train_processed.csv")
test = pd.read_csv("competitions/playground-series-s3e20/data/test_processed.csv")

# Separate features and target
X = train.drop(['ID_LAT_LON_YEAR_WEEK', 'emission'], axis=1)
y = train['emission'].values
X_test = test.drop(['ID_LAT_LON_YEAR_WEEK'], axis=1)

print(f"Training features: {X.shape}")
print(f"Test features: {X_test.shape}")

# Log transform target
y_log = np.log1p(y)
print(f"Target transformed: log1p(emission)")

print("\n" + "=" * 80)
print("TIME-BASED VALIDATION SPLIT")
print("=" * 80)

# Time-based split: 2019-2020 for train, 2021 for validation
train_mask = train['year'].isin([2019, 2020])
val_mask = train['year'] == 2021

X_tr, y_tr = X[train_mask], y_log[train_mask]
X_val, y_val = X[val_mask], y_log[val_mask]
y_val_orig = y[val_mask]  # Original scale for RMSE

print(f"Train: {X_tr.shape[0]} samples (2019-2020)")
print(f"Val:   {X_val.shape[0]} samples (2021)")

# Baseline: predict mean
baseline_pred = np.full(len(y_val_orig), np.expm1(y_tr.mean()))
baseline_rmse = np.sqrt(mean_squared_error(y_val_orig, baseline_pred))
print(f"\nBaseline RMSE (mean prediction): {baseline_rmse:.4f}")

print("\n" + "=" * 80)
print("MODEL 1: LightGBM")
print("=" * 80)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# Predictions
lgb_val_pred_log = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_val_pred = np.expm1(lgb_val_pred_log)
lgb_rmse = np.sqrt(mean_squared_error(y_val_orig, lgb_val_pred))

print(f"\nLightGBM Val RMSE: {lgb_rmse:.4f}")
print(f"Improvement over baseline: {baseline_rmse - lgb_rmse:.4f} ({(baseline_rmse - lgb_rmse) / baseline_rmse * 100:.2f}%)")

print("\n" + "=" * 80)
print("MODEL 2: XGBoost")
print("=" * 80)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'device': 'cuda',
    'tree_method': 'hist'
}

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# Predictions
xgb_val_pred_log = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration + 1))
xgb_val_pred = np.expm1(xgb_val_pred_log)
xgb_rmse = np.sqrt(mean_squared_error(y_val_orig, xgb_val_pred))

print(f"\nXGBoost Val RMSE: {xgb_rmse:.4f}")
print(f"Improvement over baseline: {baseline_rmse - xgb_rmse:.4f} ({(baseline_rmse - xgb_rmse) / baseline_rmse * 100:.2f}%)")

print("\n" + "=" * 80)
print("MODEL 3: Ridge Regression")
print("=" * 80)

ridge_model = Ridge(alpha=10.0, random_state=42)
ridge_model.fit(X_tr, y_tr)

ridge_val_pred_log = ridge_model.predict(X_val)
ridge_val_pred = np.expm1(ridge_val_pred_log)
ridge_rmse = np.sqrt(mean_squared_error(y_val_orig, ridge_val_pred))

print(f"Ridge Val RMSE: {ridge_rmse:.4f}")

print("\n" + "=" * 80)
print("ENSEMBLE")
print("=" * 80)

# Weighted average ensemble
# Optimize weights based on validation RMSE
lgb_weight = 1.0 / (lgb_rmse ** 2)
xgb_weight = 1.0 / (xgb_rmse ** 2)
ridge_weight = 1.0 / (ridge_rmse ** 2)

total_weight = lgb_weight + xgb_weight + ridge_weight
lgb_weight /= total_weight
xgb_weight /= total_weight
ridge_weight /= total_weight

print(f"Ensemble weights: LGB={lgb_weight:.3f}, XGB={xgb_weight:.3f}, Ridge={ridge_weight:.3f}")

ensemble_val_pred = (lgb_val_pred * lgb_weight +
                     xgb_val_pred * xgb_weight +
                     ridge_val_pred * ridge_weight)
ensemble_rmse = np.sqrt(mean_squared_error(y_val_orig, ensemble_val_pred))

print(f"\nEnsemble Val RMSE: {ensemble_rmse:.4f}")
print(f"Improvement over best single model: {min(lgb_rmse, xgb_rmse) - ensemble_rmse:.4f}")

print("\n" + "=" * 80)
print("GENERATE TEST PREDICTIONS")
print("=" * 80)

# Train on full data for final predictions
print("Retraining models on full dataset...")

# LightGBM on full data
lgb_full = lgb.Dataset(X, y_log)
lgb_model_full = lgb.train(
    lgb_params,
    lgb_full,
    num_boost_round=lgb_model.best_iteration,
    valid_sets=[lgb_full],
    valid_names=['train'],
    callbacks=[lgb.log_evaluation(period=200)]
)

lgb_test_pred_log = lgb_model_full.predict(X_test)
lgb_test_pred = np.expm1(lgb_test_pred_log)

# XGBoost on full data
dfull = xgb.DMatrix(X, label=y_log)
dtest = xgb.DMatrix(X_test)

xgb_model_full = xgb.train(
    xgb_params,
    dfull,
    num_boost_round=xgb_model.best_iteration,
    evals=[(dfull, 'train')],
    verbose_eval=200
)

xgb_test_pred_log = xgb_model_full.predict(dtest)
xgb_test_pred = np.expm1(xgb_test_pred_log)

# Ridge on full data
ridge_model_full = Ridge(alpha=10.0, random_state=42)
ridge_model_full.fit(X, y_log)
ridge_test_pred_log = ridge_model_full.predict(X_test)
ridge_test_pred = np.expm1(ridge_test_pred_log)

# Ensemble predictions
ensemble_test_pred = (lgb_test_pred * lgb_weight +
                      xgb_test_pred * xgb_weight +
                      ridge_test_pred * ridge_weight)

print(f"\nTest predictions generated")
print(f"Prediction range: [{ensemble_test_pred.min():.2f}, {ensemble_test_pred.max():.2f}]")
print(f"Prediction mean: {ensemble_test_pred.mean():.2f}")

print("\n" + "=" * 80)
print("SAVE RESULTS")
print("=" * 80)

# Save predictions
submission = pd.DataFrame({
    'ID_LAT_LON_YEAR_WEEK': test['ID_LAT_LON_YEAR_WEEK'],
    'emission': ensemble_test_pred
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f"competitions/playground-series-s3e20/submissions/submission_ensemble_{ensemble_rmse:.4f}_{timestamp}.csv"
submission.to_csv(submission_file, index=False)

print(f"Submission saved: {submission_file}")

# Log experiment
experiment = {
    "experiment_id": 1,
    "timestamp": timestamp,
    "stage": "modeling",
    "model": "LGB+XGB+Ridge ensemble",
    "features": "processed_v1 (102 features)",
    "params": {
        "lgb": lgb_params,
        "xgb": xgb_params,
        "ridge_alpha": 10.0,
        "ensemble_weights": {
            "lgb": float(lgb_weight),
            "xgb": float(xgb_weight),
            "ridge": float(ridge_weight)
        }
    },
    "cv_strategy": "time-based (2019-2020 train, 2021 val)",
    "val_scores": {
        "baseline_rmse": float(baseline_rmse),
        "lgb_rmse": float(lgb_rmse),
        "xgb_rmse": float(xgb_rmse),
        "ridge_rmse": float(ridge_rmse),
        "ensemble_rmse": float(ensemble_rmse)
    },
    "notes": "Log1p target transformation, GPU training, time-based validation",
    "submission_file": submission_file
}

# Save to experiments.json
experiments_file = "competitions/playground-series-s3e20/experiments.json"
try:
    with open(experiments_file, 'r') as f:
        experiments = json.load(f)
except:
    experiments = []

experiments.append(experiment)

with open(experiments_file, 'w') as f:
    json.dump(experiments, f, indent=2)

print(f"Experiment logged to {experiments_file}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Baseline RMSE:      {baseline_rmse:.4f}")
print(f"LightGBM RMSE:      {lgb_rmse:.4f}")
print(f"XGBoost RMSE:       {xgb_rmse:.4f}")
print(f"Ridge RMSE:         {ridge_rmse:.4f}")
print(f"Ensemble RMSE:      {ensemble_rmse:.4f} ⭐")
print(f"\nImprovement: {(baseline_rmse - ensemble_rmse) / baseline_rmse * 100:.2f}%")
