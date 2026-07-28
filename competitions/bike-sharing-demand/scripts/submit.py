"""Generate submission: 3-model ensemble (CatBoost + XGBoost + LightGBM) retrained on full data."""
import json
from datetime import datetime, timezone

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

data_dir = "competitions/bike-sharing-demand/data"
sub_dir = "competitions/bike-sharing-demand/submissions"
exp_file = "competitions/bike-sharing-demand/experiments.json"

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_parquet(f"{data_dir}/train_processed.parquet")
test = pd.read_parquet(f"{data_dir}/test_processed.parquet")
sample_sub = pd.read_csv(f"{data_dir}/sampleSubmission.csv")

drop_cols = ["is_weekend", "season", "windspeed_zero"]
feature_cols = [c for c in train.columns if c not in ["datetime", "count", "log_count"] + drop_cols]

X_train = train[feature_cols]
y_log = train["log_count"]
X_test = test[feature_cols]
test_datetime = test["datetime"]

print(f"Training on full data: {X_train.shape}")
print(f"Predicting on test: {X_test.shape}")
print(f"Features: {feature_cols}")

# ============================================================
# 1. RETRAIN CATBOOST ON FULL DATA
# ============================================================
print("\n--- CatBoost ---")
cat_model = cb.CatBoostRegressor(
    iterations=500, learning_rate=0.05, depth=7,
    l2_leaf_reg=3.0, random_seed=42, verbose=0,
)
cat_model.fit(X_train, y_log)
cat_preds = cat_model.predict(X_test)
print(f"  Predictions range (log): [{cat_preds.min():.3f}, {cat_preds.max():.3f}]")

# ============================================================
# 2. RETRAIN XGBOOST ON FULL DATA
# ============================================================
print("\n--- XGBoost ---")
xgb_params = {
    "objective": "reg:squarederror", "learning_rate": 0.05,
    "max_depth": 7, "min_child_weight": 20, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "seed": 42, "verbosity": 0,
}
dtrain = xgb.DMatrix(X_train, label=y_log)
dtest = xgb.DMatrix(X_test)
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=300)
xgb_preds = xgb_model.predict(dtest)
print(f"  Predictions range (log): [{xgb_preds.min():.3f}, {xgb_preds.max():.3f}]")

# ============================================================
# 3. RETRAIN LIGHTGBM ON FULL DATA
# ============================================================
print("\n--- LightGBM ---")
lgb_params = {
    "objective": "regression", "metric": "rmse", "verbosity": -1,
    "seed": 42,
}
lgb_dtrain = lgb.Dataset(X_train, label=y_log)
lgb_model = lgb.train(lgb_params, lgb_dtrain, num_boost_round=200)
lgb_preds = lgb_model.predict(X_test)
print(f"  Predictions range (log): [{lgb_preds.min():.3f}, {lgb_preds.max():.3f}]")

# ============================================================
# 4. ENSEMBLE: Simple average
# ============================================================
print("\n--- Ensemble (simple average) ---")
ensemble_log = (cat_preds + xgb_preds + lgb_preds) / 3.0
ensemble_raw = np.clip(np.expm1(ensemble_log), 0, None)

# Round to integers (bike counts are discrete)
ensemble_raw = np.round(ensemble_raw).astype(int)

print(f"  Ensemble predictions (raw count):")
print(f"    min={ensemble_raw.min()}, max={ensemble_raw.max()}, mean={ensemble_raw.mean():.1f}")
print(f"    Training target: min={train['count'].min()}, max={train['count'].max()}, mean={train['count'].mean():.1f}")

# ============================================================
# 5. FORMAT SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "datetime": test_datetime,
    "count": ensemble_raw,
})

# ============================================================
# 6. VALIDATE
# ============================================================
print("\n=== VALIDATION ===")
assert submission.shape == sample_sub.shape, f"Shape mismatch: {submission.shape} vs {sample_sub.shape}"
print(f"Shape match: {submission.shape} == {sample_sub.shape}")

assert list(submission.columns) == list(sample_sub.columns), "Column mismatch"
print(f"Columns match: {list(submission.columns)}")

assert submission.isnull().sum().sum() == 0, "NaN in submission"
print("No NaN values: OK")

assert (submission["count"] >= 0).all(), "Negative predictions!"
print(f"All predictions >= 0: OK")

assert len(submission) == len(sample_sub), "Row count mismatch"
print(f"Row count: {len(submission)}")

# ============================================================
# 7. SAVE
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submission_ensemble3_cv0.317_{ts}.csv"
filepath = f"{sub_dir}/{filename}"
submission.to_csv(filepath, index=False)
print(f"\nSaved: {filepath}")
print(f"\nHead:\n{submission.head(10)}")

# Log to experiments
with open(exp_file) as f:
    experiments = json.load(f)

experiments.append({
    "experiment_id": len(experiments) + 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "submission",
    "model": "Ensemble (CatBoost + XGBoost + LightGBM avg)",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "cv_mean": 0.317,
    "retrained_on_full_data": True,
    "post_processing": "round to int, clip >= 0",
    "submission_file": filename,
    "notes": "Simple average of 3 GBMs retrained on full data"
})

with open(exp_file, "w") as f:
    json.dump(experiments, f, indent=2)

print(f"\nExperiment logged to {exp_file}")
