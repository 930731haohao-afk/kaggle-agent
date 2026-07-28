"""Generate submission for Right Whale Upcall Detection."""
import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

data_dir = "competitions/the-icml-2013-whale-challenge-right-whale-redux/data"
sub_dir = "competitions/the-icml-2013-whale-challenge-right-whale-redux/submissions"
train_dir = f"{data_dir}/train/train2"
test_dir = f"{data_dir}/test/test2"

# Load cached features
print("Loading features...")
train_data = np.load(f"{data_dir}/train_features.npz", allow_pickle=True)
X_train = train_data["X"]
y = train_data["y"]

test_data = np.load(f"{data_dir}/test_features.npz", allow_pickle=True)
X_test = test_data["X"]
test_filenames = test_data["filenames"]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Positive: {y.sum()}, Negative: {(1-y).sum()}")

# Best weights from CV: LGB=0.4, XGB=0.5, Cat=0.1
w_lgb, w_xgb, w_cat = 0.4, 0.5, 0.1

# ============================================================
# RETRAIN ON FULL DATA
# ============================================================
print("\nRetraining on full data...")

# LightGBM
print("  Training LightGBM...")
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "is_unbalance": True,
    "seed": 42,
    "verbose": -1,
}
dtrain = lgb.Dataset(X_train, label=y)
lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=1400)
test_preds_lgb = lgb_model.predict(X_test)

# XGBoost
print("  Training XGBoost...")
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": (1 - y.mean()) / y.mean(),
    "seed": 42,
    "verbosity": 0,
}
dtrain_xgb = xgb.DMatrix(X_train, label=y)
dtest_xgb = xgb.DMatrix(X_test)
xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=1600)
test_preds_xgb = xgb_model.predict(dtest_xgb)

# CatBoost
print("  Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=1850, learning_rate=0.03, depth=6,
    l2_leaf_reg=5.0, random_seed=42, verbose=0,
    auto_class_weights="Balanced",
    eval_metric="AUC",
)
cat_model.fit(X_train, y)
test_preds_cat = cat_model.predict_proba(X_test)[:, 1]

# ============================================================
# WEIGHTED ENSEMBLE
# ============================================================
print("\nBlending predictions...")
ensemble = w_lgb * test_preds_lgb + w_xgb * test_preds_xgb + w_cat * test_preds_cat
ensemble = np.clip(ensemble, 0.001, 0.999)

print(f"Prediction stats:")
print(f"  Range: [{ensemble.min():.4f}, {ensemble.max():.4f}]")
print(f"  Mean: {ensemble.mean():.4f}")
print(f"  >0.5: {(ensemble > 0.5).sum()} ({100*(ensemble > 0.5).mean():.1f}%)")

# ============================================================
# FORMAT SUBMISSION
# ============================================================
# Format: clip,probability
sample_sub = pd.read_csv(f"{data_dir}/sampleSubmission.csv")
print(f"\nSample submission shape: {sample_sub.shape}")
print(f"Columns: {list(sample_sub.columns)}")

# Map filenames to predictions
pred_dict = dict(zip(test_filenames, ensemble))
submission = sample_sub.copy()
submission["probability"] = submission["clip"].map(pred_dict)

# Verify no missing
missing = submission["probability"].isna().sum()
if missing > 0:
    print(f"  WARNING: {missing} missing predictions!")
    submission["probability"] = submission["probability"].fillna(0.5)
else:
    print(f"  All {len(submission)} clips have predictions")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
cv_score = 0.951
sub_file = f"{sub_dir}/submission_ensemble3_cv{cv_score:.3f}_{ts}.csv"
submission.to_csv(sub_file, index=False)

print(f"\nSaved: {sub_file}")
print(f"Shape: {submission.shape}")
print(f"Sample:\n{submission.head()}")
print("\nDone!")
