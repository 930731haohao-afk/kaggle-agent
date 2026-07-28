"""Generate submission: Weighted ensemble (LGB=0.3, XGB=0.2, Cat=0.5) retrained on full data."""
import json
from datetime import datetime, timezone

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

data_dir = "competitions/linking-writing-processes-to-writing-quality/data"
sub_dir = "competitions/linking-writing-processes-to-writing-quality/submissions"
exp_file = "competitions/linking-writing-processes-to-writing-quality/experiments.json"

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_parquet(f"{data_dir}/train_processed.parquet")
test = pd.read_parquet(f"{data_dir}/test_processed.parquet")
sample_sub = pd.read_csv(f"{data_dir}/sample_submission.csv")

drop_cols = ["n_paragraphs", "approx_sentences"]
feature_cols = [c for c in train.columns if c not in ["id", "score"] + drop_cols]

X_train = train[feature_cols]
y = train["score"]
X_test = test[[c for c in feature_cols if c in test.columns]]

# Fill NaN in test (iki_std, iki_skew have NaN for 2-event dummy essays)
X_test = X_test.fillna(0)

test_ids = test["id"]

print(f"Training on full data: {X_train.shape}")
print(f"Predicting on test: {X_test.shape}")

# Ensemble weights from OOF optimization
W_LGB, W_XGB, W_CAT = 0.3, 0.2, 0.5

# ============================================================
# 1. RETRAIN LIGHTGBM ON FULL DATA
# ============================================================
print("\n--- LightGBM ---")
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "seed": 42,
}
dtrain = lgb.Dataset(X_train, label=y)
lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=200)
lgb_preds = lgb_model.predict(X_test)
print(f"  Predictions: [{lgb_preds.min():.3f}, {lgb_preds.max():.3f}]")

# ============================================================
# 2. RETRAIN XGBOOST ON FULL DATA
# ============================================================
print("\n--- XGBoost ---")
xgb_params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 30,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "seed": 42,
    "verbosity": 0,
}
dt = xgb.DMatrix(X_train, label=y)
dtest = xgb.DMatrix(X_test)
xgb_model = xgb.train(xgb_params, dt, num_boost_round=250)
xgb_preds = xgb_model.predict(dtest)
print(f"  Predictions: [{xgb_preds.min():.3f}, {xgb_preds.max():.3f}]")

# ============================================================
# 3. RETRAIN CATBOOST ON FULL DATA
# ============================================================
print("\n--- CatBoost ---")
cat_model = cb.CatBoostRegressor(
    iterations=500, learning_rate=0.03, depth=6,
    l2_leaf_reg=5.0, random_seed=42, verbose=0,
)
cat_model.fit(X_train, y)
cat_preds = cat_model.predict(X_test)
print(f"  Predictions: [{cat_preds.min():.3f}, {cat_preds.max():.3f}]")

# ============================================================
# 4. WEIGHTED ENSEMBLE
# ============================================================
print(f"\n--- Ensemble (LGB={W_LGB}, XGB={W_XGB}, Cat={W_CAT}) ---")
ensemble_preds = W_LGB * lgb_preds + W_XGB * xgb_preds + W_CAT * cat_preds

# Clip to valid score range [0.5, 6.0]
ensemble_preds = np.clip(ensemble_preds, 0.5, 6.0)

print(f"  Ensemble predictions:")
print(f"    min={ensemble_preds.min():.3f}, max={ensemble_preds.max():.3f}, mean={ensemble_preds.mean():.3f}")
print(f"    Training target: min={y.min()}, max={y.max()}, mean={y.mean():.3f}")

# ============================================================
# 5. FORMAT SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "id": test_ids,
    "score": ensemble_preds,
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

assert (submission["score"] >= 0.5).all() and (submission["score"] <= 6.0).all(), "Score out of range"
print(f"Score range [0.5, 6.0]: OK")

print(f"Row count: {len(submission)}")

# ============================================================
# 7. SAVE
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submission_ensemble3_cv0.648_{ts}.csv"
filepath = f"{sub_dir}/{filename}"
submission.to_csv(filepath, index=False)
print(f"\nSaved: {filepath}")
print(f"\nHead:\n{submission}")

# Log to experiments
with open(exp_file) as f:
    experiments = json.load(f)

experiments.append({
    "experiment_id": len(experiments) + 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "submission",
    "model": f"Weighted Ensemble (LGB={W_LGB}, XGB={W_XGB}, Cat={W_CAT})",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "cv_mean": 0.648,
    "retrained_on_full_data": True,
    "post_processing": "clip to [0.5, 6.0]",
    "submission_file": filename,
    "notes": "Weighted ensemble retrained on full data"
})

with open(exp_file, "w") as f:
    json.dump(experiments, f, indent=2)

print(f"\nExperiment logged to {exp_file}")
