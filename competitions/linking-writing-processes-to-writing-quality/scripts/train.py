"""Training script for Linking Writing Processes to Writing Quality."""
import json
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

data_dir = "competitions/linking-writing-processes-to-writing-quality/data"
exp_file = "competitions/linking-writing-processes-to-writing-quality/experiments.json"

train = pd.read_parquet(f"{data_dir}/train_processed.parquet")

# Drop zero-importance features
drop_cols = ["n_paragraphs", "approx_sentences"]
feature_cols = [c for c in train.columns if c not in ["id", "score"] + drop_cols]

X = train[feature_cols]
y = train["score"]

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Train shape: {X.shape}")
print(f"Target: mean={y.mean():.3f}, std={y.std():.3f}")

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

experiments = []


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def log_experiment(exp: dict):
    experiments.append(exp)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)


# ============================================================
# 1. BASELINE: Mean predictor
# ============================================================
print("\n" + "=" * 60)
print("1. BASELINE: Mean predictor")
print("=" * 60)

mean_pred = np.full(len(y), y.mean())
mean_rmse = rmse(y.values, mean_pred)
print(f"Mean predictor RMSE: {mean_rmse:.5f}")

log_experiment({
    "experiment_id": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "mean-predictor",
    "features": "none",
    "cv_scores": [mean_rmse],
    "cv_mean": round(mean_rmse, 5),
    "notes": "Naive baseline: predict global mean"
})

# ============================================================
# 2. BASELINE: Ridge Regression
# ============================================================
print("\n" + "=" * 60)
print("2. BASELINE: Ridge Regression")
print("=" * 60)

ridge_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    model = Ridge(alpha=1.0)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[val_idx])
    score = rmse(y.iloc[val_idx].values, preds)
    ridge_scores.append(score)
    print(f"  Fold {fold+1}: RMSE = {score:.5f}")

print(f"  Mean RMSE: {np.mean(ridge_scores):.5f} (+/- {np.std(ridge_scores):.5f})")

log_experiment({
    "experiment_id": 2,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "Ridge",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "cv_strategy": f"KFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in ridge_scores],
    "cv_mean": round(np.mean(ridge_scores), 5),
    "cv_std": round(np.std(ridge_scores), 5),
    "notes": "Ridge regression baseline"
})

# ============================================================
# 3. LightGBM (default)
# ============================================================
print("\n" + "=" * 60)
print("3. LightGBM (default)")
print("=" * 60)

lgb_default_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
    dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": 42,
    }
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict(X.iloc[val_idx])
    score = rmse(y.iloc[val_idx].values, preds)
    lgb_default_scores.append(score)
    print(f"  Fold {fold+1}: RMSE = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean RMSE: {np.mean(lgb_default_scores):.5f} (+/- {np.std(lgb_default_scores):.5f})")

log_experiment({
    "experiment_id": 3,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "LightGBM-default",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "cv_strategy": f"KFold-{n_splits}",
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

lgb_tuned_params = {
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

lgb_tuned_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
    dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=dtrain)

    model = lgb.train(lgb_tuned_params, dtrain, num_boost_round=2000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict(X.iloc[val_idx])
    score = rmse(y.iloc[val_idx].values, preds)
    lgb_tuned_scores.append(score)
    print(f"  Fold {fold+1}: RMSE = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean RMSE: {np.mean(lgb_tuned_scores):.5f} (+/- {np.std(lgb_tuned_scores):.5f})")

log_experiment({
    "experiment_id": 4,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "LightGBM-tuned",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "params": {k: v for k, v in lgb_tuned_params.items() if k not in ["objective", "metric", "verbosity"]},
    "cv_strategy": f"KFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in lgb_tuned_scores],
    "cv_mean": round(np.mean(lgb_tuned_scores), 5),
    "cv_std": round(np.std(lgb_tuned_scores), 5),
    "notes": "LightGBM tuned with regularization for small dataset"
})

# ============================================================
# 5. XGBoost (tuned)
# ============================================================
print("\n" + "=" * 60)
print("5. XGBoost (tuned)")
print("=" * 60)

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
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

xgb_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
    dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])

    model = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                      evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    preds = model.predict(dval)
    score = rmse(y.iloc[val_idx].values, preds)
    xgb_scores.append(score)
    print(f"  Fold {fold+1}: RMSE = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean RMSE: {np.mean(xgb_scores):.5f} (+/- {np.std(xgb_scores):.5f})")

log_experiment({
    "experiment_id": 5,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "XGBoost-tuned",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "params": {k: v for k, v in xgb_params.items() if k not in ["objective", "eval_metric", "verbosity"]},
    "cv_strategy": f"KFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in xgb_scores],
    "cv_mean": round(np.mean(xgb_scores), 5),
    "cv_std": round(np.std(xgb_scores), 5),
    "notes": "XGBoost tuned with strong regularization"
})

# ============================================================
# 6. CatBoost
# ============================================================
print("\n" + "=" * 60)
print("6. CatBoost")
print("=" * 60)

cat_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx],
              eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
    preds = model.predict(X.iloc[val_idx])
    score = rmse(y.iloc[val_idx].values, preds)
    cat_scores.append(score)
    print(f"  Fold {fold+1}: RMSE = {score:.5f} (best_iter={model.best_iteration_})")

print(f"  Mean RMSE: {np.mean(cat_scores):.5f} (+/- {np.std(cat_scores):.5f})")

log_experiment({
    "experiment_id": 6,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "CatBoost",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "params": {"depth": 6, "l2_leaf_reg": 5.0, "learning_rate": 0.03},
    "cv_strategy": f"KFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in cat_scores],
    "cv_mean": round(np.mean(cat_scores), 5),
    "cv_std": round(np.std(cat_scores), 5),
    "notes": "CatBoost with strong regularization"
})

# ============================================================
# 7. OOF ENSEMBLE (weighted average optimized on OOF)
# ============================================================
print("\n" + "=" * 60)
print("7. OOF ENSEMBLE")
print("=" * 60)

# Collect OOF predictions
oof_lgb = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # LightGBM
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
    dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=dtrain)
    m = lgb.train(lgb_tuned_params, dtrain, num_boost_round=2000,
                  valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = m.predict(X.iloc[val_idx])

    # XGBoost
    dt = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
    dv = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
    m = xgb.train(xgb_params, dt, num_boost_round=2000,
                  evals=[(dv, "val")], early_stopping_rounds=50, verbose_eval=False)
    oof_xgb[val_idx] = m.predict(dv)

    # CatBoost
    m = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6,
                          l2_leaf_reg=5.0, random_seed=42, verbose=0, early_stopping_rounds=50)
    m.fit(X.iloc[train_idx], y.iloc[train_idx],
          eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
    oof_cat[val_idx] = m.predict(X.iloc[val_idx])

# Try different weights
print("\nWeight search:")
best_rmse = 999
best_weights = None
for w1 in np.arange(0.1, 0.9, 0.1):
    for w2 in np.arange(0.1, 0.9 - w1 + 0.01, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        oof_blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        r = rmse(y.values, oof_blend)
        if r < best_rmse:
            best_rmse = r
            best_weights = (round(w1, 1), round(w2, 1), round(w3, 1))

print(f"  Best weights: LGB={best_weights[0]}, XGB={best_weights[1]}, Cat={best_weights[2]}")
print(f"  Best ensemble RMSE: {best_rmse:.5f}")

# Also simple average
simple_avg = (oof_lgb + oof_xgb + oof_cat) / 3
simple_rmse = rmse(y.values, simple_avg)
print(f"  Simple average RMSE: {simple_rmse:.5f}")

log_experiment({
    "experiment_id": 7,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "ensemble",
    "model": f"Weighted Ensemble (LGB={best_weights[0]}, XGB={best_weights[1]}, Cat={best_weights[2]})",
    "features": f"processed_v1 ({len(feature_cols)} features)",
    "cv_strategy": f"KFold-{n_splits} OOF",
    "cv_mean": round(best_rmse, 5),
    "notes": f"Optimized weights. Simple avg: {simple_rmse:.5f}"
})

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT LEADERBOARD")
print("=" * 60)

results = [
    ("Mean predictor", mean_rmse, 0),
    ("Ridge", np.mean(ridge_scores), np.std(ridge_scores)),
    ("LightGBM-default", np.mean(lgb_default_scores), np.std(lgb_default_scores)),
    ("LightGBM-tuned", np.mean(lgb_tuned_scores), np.std(lgb_tuned_scores)),
    ("XGBoost-tuned", np.mean(xgb_scores), np.std(xgb_scores)),
    ("CatBoost", np.mean(cat_scores), np.std(cat_scores)),
    (f"Ensemble ({best_weights})", best_rmse, 0),
    ("Ensemble (equal)", simple_rmse, 0),
]
results.sort(key=lambda x: x[1])

print(f"\n{'Rank':<5} {'Model':<35} {'CV RMSE':<12} {'Std':<10}")
print("-" * 65)
for rank, (name, mean, std) in enumerate(results, 1):
    print(f"{rank:<5} {name:<35} {mean:<12.5f} {std:<10.5f}")
