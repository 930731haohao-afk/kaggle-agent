"""Training script for Bike Sharing Demand with CV and experiment logging."""
import json
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# ============================================================
# SETUP
# ============================================================
data_dir = "competitions/bike-sharing-demand/data"
exp_file = "competitions/bike-sharing-demand/experiments.json"

train = pd.read_parquet(f"{data_dir}/train_processed.parquet")

# Drop zero-importance features from EDA
drop_cols = ["is_weekend", "season", "windspeed_zero"]
feature_cols = [c for c in train.columns if c not in ["datetime", "count", "log_count"] + drop_cols]

X = train[feature_cols]
y_log = train["log_count"]  # Train on log1p(count) — RMSE on this = RMSLE on count
y_raw = train["count"]

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Train shape: {X.shape}")

# CV strategy: GroupKFold by (year, month)
groups = train["datetime"].dt.to_period("M").astype(str)
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

experiments = []


def rmsle_from_log_preds(y_true_raw: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Compute RMSLE given raw targets and log-space predictions."""
    # Clip: expm1 can go negative for very small log preds
    y_pred_raw = np.clip(np.expm1(y_pred_log), 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred_raw) - np.log1p(y_true_raw)) ** 2))


def log_experiment(exp: dict):
    """Append experiment to JSON log."""
    experiments.append(exp)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)


# ============================================================
# 1. BASELINE: Mean predictor
# ============================================================
print("\n" + "=" * 60)
print("1. BASELINE: Mean predictor")
print("=" * 60)

mean_pred = np.full(len(y_raw), y_log.mean())
mean_rmsle = rmsle_from_log_preds(y_raw.values, mean_pred)
print(f"Mean predictor RMSLE: {mean_rmsle:.5f}")

log_experiment({
    "experiment_id": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "mean-predictor",
    "features": f"none (predicts global mean)",
    "cv_scores": [mean_rmsle],
    "cv_mean": round(mean_rmsle, 5),
    "notes": "Naive baseline: predict mean of log1p(count)"
})

# ============================================================
# 2. BASELINE: Ridge Regression
# ============================================================
print("\n" + "=" * 60)
print("2. BASELINE: Ridge Regression")
print("=" * 60)

ridge_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
    model = Ridge(alpha=1.0)
    model.fit(X.iloc[train_idx], y_log.iloc[train_idx])
    preds = model.predict(X.iloc[val_idx])
    score = rmsle_from_log_preds(y_raw.iloc[val_idx].values, preds)
    ridge_scores.append(score)
    print(f"  Fold {fold+1}: RMSLE = {score:.5f}")

print(f"  Mean RMSLE: {np.mean(ridge_scores):.5f} (+/- {np.std(ridge_scores):.5f})")

log_experiment({
    "experiment_id": 2,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "Ridge",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "cv_strategy": f"GroupKFold-{n_splits} by year-month",
    "cv_scores": [round(s, 5) for s in ridge_scores],
    "cv_mean": round(np.mean(ridge_scores), 5),
    "cv_std": round(np.std(ridge_scores), 5),
    "notes": "Ridge regression on log1p(count)"
})

# ============================================================
# 3. LightGBM (default params)
# ============================================================
print("\n" + "=" * 60)
print("3. LightGBM (default params)")
print("=" * 60)

lgb_default_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y_log.iloc[train_idx])
    dval = lgb.Dataset(X.iloc[val_idx], label=y_log.iloc[val_idx], reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": 42,
    }
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict(X.iloc[val_idx])
    score = rmsle_from_log_preds(y_raw.iloc[val_idx].values, preds)
    lgb_default_scores.append(score)
    print(f"  Fold {fold+1}: RMSLE = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean RMSLE: {np.mean(lgb_default_scores):.5f} (+/- {np.std(lgb_default_scores):.5f})")

log_experiment({
    "experiment_id": 3,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "LightGBM-default",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "cv_strategy": f"GroupKFold-{n_splits} by year-month",
    "cv_scores": [round(s, 5) for s in lgb_default_scores],
    "cv_mean": round(np.mean(lgb_default_scores), 5),
    "cv_std": round(np.std(lgb_default_scores), 5),
    "notes": "LightGBM default params, early stopping 50"
})

# ============================================================
# 4. LightGBM (tuned)
# ============================================================
print("\n" + "=" * 60)
print("4. LightGBM (tuned)")
print("=" * 60)

lgb_tuned_scores = []
lgb_tuned_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "seed": 42,
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y_log.iloc[train_idx])
    dval = lgb.Dataset(X.iloc[val_idx], label=y_log.iloc[val_idx], reference=dtrain)

    model = lgb.train(lgb_tuned_params, dtrain, num_boost_round=1000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict(X.iloc[val_idx])
    score = rmsle_from_log_preds(y_raw.iloc[val_idx].values, preds)
    lgb_tuned_scores.append(score)
    print(f"  Fold {fold+1}: RMSLE = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean RMSLE: {np.mean(lgb_tuned_scores):.5f} (+/- {np.std(lgb_tuned_scores):.5f})")

log_experiment({
    "experiment_id": 4,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "LightGBM-tuned",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "params": {k: v for k, v in lgb_tuned_params.items() if k not in ["objective", "metric", "verbosity"]},
    "cv_strategy": f"GroupKFold-{n_splits} by year-month",
    "cv_scores": [round(s, 5) for s in lgb_tuned_scores],
    "cv_mean": round(np.mean(lgb_tuned_scores), 5),
    "cv_std": round(np.std(lgb_tuned_scores), 5),
    "notes": "LightGBM with hand-tuned params"
})

# ============================================================
# 5. XGBoost (tuned)
# ============================================================
print("\n" + "=" * 60)
print("5. XGBoost (tuned)")
print("=" * 60)

xgb_scores = []
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
    "verbosity": 0,
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
    dtrain = xgb.DMatrix(X.iloc[train_idx], label=y_log.iloc[train_idx])
    dval = xgb.DMatrix(X.iloc[val_idx], label=y_log.iloc[val_idx])

    model = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                      evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    preds = model.predict(dval)
    score = rmsle_from_log_preds(y_raw.iloc[val_idx].values, preds)
    xgb_scores.append(score)
    print(f"  Fold {fold+1}: RMSLE = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean RMSLE: {np.mean(xgb_scores):.5f} (+/- {np.std(xgb_scores):.5f})")

log_experiment({
    "experiment_id": 5,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "XGBoost-tuned",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "params": {k: v for k, v in xgb_params.items() if k not in ["objective", "eval_metric", "verbosity"]},
    "cv_strategy": f"GroupKFold-{n_splits} by year-month",
    "cv_scores": [round(s, 5) for s in xgb_scores],
    "cv_mean": round(np.mean(xgb_scores), 5),
    "cv_std": round(np.std(xgb_scores), 5),
    "notes": "XGBoost with tuned params"
})

# ============================================================
# 6. CatBoost
# ============================================================
print("\n" + "=" * 60)
print("6. CatBoost")
print("=" * 60)

from catboost import CatBoostRegressor

cat_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
    model.fit(X.iloc[train_idx], y_log.iloc[train_idx],
              eval_set=(X.iloc[val_idx], y_log.iloc[val_idx]))
    preds = model.predict(X.iloc[val_idx])
    score = rmsle_from_log_preds(y_raw.iloc[val_idx].values, preds)
    cat_scores.append(score)
    print(f"  Fold {fold+1}: RMSLE = {score:.5f} (best_iter={model.best_iteration_})")

print(f"  Mean RMSLE: {np.mean(cat_scores):.5f} (+/- {np.std(cat_scores):.5f})")

log_experiment({
    "experiment_id": 6,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "CatBoost",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "params": {"depth": 7, "l2_leaf_reg": 3.0, "learning_rate": 0.05},
    "cv_strategy": f"GroupKFold-{n_splits} by year-month",
    "cv_scores": [round(s, 5) for s in cat_scores],
    "cv_mean": round(np.mean(cat_scores), 5),
    "cv_std": round(np.std(cat_scores), 5),
    "notes": "CatBoost with tuned params"
})

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT LEADERBOARD")
print("=" * 60)

print(f"\n{'Rank':<5} {'Model':<25} {'CV RMSLE':<12} {'Std':<10}")
print("-" * 55)

results = [
    ("Mean predictor", mean_rmsle, 0),
    ("Ridge", np.mean(ridge_scores), np.std(ridge_scores)),
    ("LightGBM-default", np.mean(lgb_default_scores), np.std(lgb_default_scores)),
    ("LightGBM-tuned", np.mean(lgb_tuned_scores), np.std(lgb_tuned_scores)),
    ("XGBoost-tuned", np.mean(xgb_scores), np.std(xgb_scores)),
    ("CatBoost", np.mean(cat_scores), np.std(cat_scores)),
]
results.sort(key=lambda x: x[1])

for rank, (name, mean, std) in enumerate(results, 1):
    print(f"{rank:<5} {name:<25} {mean:<12.5f} {std:<10.5f}")
