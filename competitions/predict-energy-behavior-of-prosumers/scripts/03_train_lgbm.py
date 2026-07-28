"""
LightGBM training for predict-energy-behavior-of-prosumers.
Time-series split validation. Evaluates MAE on raw target.
Trains separate models for production and consumption.
"""
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

data_dir = "competitions/predict-energy-behavior-of-prosumers/data"
exp_file = "competitions/predict-energy-behavior-of-prosumers/experiments.json"

# ============================================================
# 1. Load processed data
# ============================================================
print("=" * 70)
print("1. LOADING PROCESSED DATA")
print("=" * 70)

train = pd.read_parquet(os.path.join(data_dir, "train_processed.parquet"))
print(f"Shape: {train.shape}")

# ============================================================
# 2. Define features and target
# ============================================================
print("\n" + "=" * 70)
print("2. FEATURE SELECTION")
print("=" * 70)

drop_cols = ['target', 'row_id', 'datetime', 'data_block_id']
feature_cols = [c for c in train.columns if c not in drop_cols]
print(f"Features: {len(feature_cols)}")

target_col = 'target'

# ============================================================
# 3. Time-series split
# ============================================================
print("\n" + "=" * 70)
print("3. TIME-SERIES SPLIT")
print("=" * 70)

# Use data_block_id for time-based split
# Train: blocks 0-549 (~86% of data)
# Val: blocks 550-637 (~14% of data, last ~3 months)
TRAIN_CUTOFF = 550

train_mask = train['data_block_id'] < TRAIN_CUTOFF
val_mask = (train['data_block_id'] >= TRAIN_CUTOFF) & train['target'].notna()

# Also need non-null target for training
train_mask = train_mask & train['target'].notna()

# Additionally, require lag_24 to be non-null (skip first day)
train_mask = train_mask & train['target_lag_24'].notna()
val_mask = val_mask & train['target_lag_24'].notna()

X_train = train.loc[train_mask, feature_cols]
y_train = train.loc[train_mask, target_col]
X_val = train.loc[val_mask, feature_cols]
y_val = train.loc[val_mask, target_col]

print(f"Train: {X_train.shape} (blocks 0-{TRAIN_CUTOFF-1})")
print(f"Val:   {X_val.shape} (blocks {TRAIN_CUTOFF}-637)")

# ============================================================
# 4. Baseline: predict using lag_24 (yesterday same hour)
# ============================================================
print("\n" + "=" * 70)
print("4. BASELINE: LAG-24 PREDICTOR")
print("=" * 70)

lag24_pred = X_val['target_lag_24'].fillna(0).values
baseline_mae = mean_absolute_error(y_val, lag24_pred)
print(f"Lag-24 baseline MAE: {baseline_mae:.4f}")

# ============================================================
# 5. LightGBM: Single model (all data)
# ============================================================
print("\n" + "=" * 70)
print("5. LIGHTGBM: SINGLE MODEL (all segments)")
print("=" * 70)

lgb_params = {
    'objective': 'mae',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 255,
    'max_depth': -1,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

t0 = time.time()
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

model_single = lgb.train(
    lgb_params, dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    callbacks=[
        lgb.early_stopping(50, verbose=True),
        lgb.log_evaluation(100)
    ]
)
single_time = time.time() - t0

pred_single = model_single.predict(X_val)
pred_single = np.clip(pred_single, 0, None)  # target is non-negative
mae_single = mean_absolute_error(y_val, pred_single)
print(f"\nSingle model MAE: {mae_single:.4f} (baseline: {baseline_mae:.4f})")
print(f"Improvement over baseline: {(baseline_mae - mae_single)/baseline_mae*100:.1f}%")
print(f"Training time: {single_time:.1f}s, Best iteration: {model_single.best_iteration}")

# ============================================================
# 6. LightGBM: Separate models for production vs consumption
# ============================================================
print("\n" + "=" * 70)
print("6. LIGHTGBM: SEPARATE MODELS (production vs consumption)")
print("=" * 70)

t0 = time.time()
maes_by_type = {}
models_by_type = {}
preds_by_type = np.zeros(len(X_val))

for ic, label in [(0, "Production"), (1, "Consumption")]:
    print(f"\n--- {label} (is_consumption={ic}) ---")
    mask_tr = X_train['is_consumption'] == ic
    mask_val = X_val['is_consumption'] == ic

    X_tr_sub = X_train[mask_tr]
    y_tr_sub = y_train[mask_tr]
    X_val_sub = X_val[mask_val]
    y_val_sub = y_val[mask_val]

    dtrain_sub = lgb.Dataset(X_tr_sub, label=y_tr_sub)
    dval_sub = lgb.Dataset(X_val_sub, label=y_val_sub, reference=dtrain_sub)

    model = lgb.train(
        lgb_params, dtrain_sub,
        num_boost_round=2000,
        valid_sets=[dval_sub],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100)
        ]
    )
    pred = model.predict(X_val_sub)
    pred = np.clip(pred, 0, None)
    mae = mean_absolute_error(y_val_sub, pred)
    print(f"  {label} MAE: {mae:.4f}")

    maes_by_type[label] = mae
    models_by_type[label] = model
    preds_by_type[mask_val.values] = pred

split_time = time.time() - t0
mae_split = mean_absolute_error(y_val, preds_by_type)
print(f"\nSplit models combined MAE: {mae_split:.4f}")
print(f"Improvement over single: {(mae_single - mae_split)/mae_single*100:.1f}%")
print(f"Improvement over baseline: {(baseline_mae - mae_split)/baseline_mae*100:.1f}%")
print(f"Training time: {split_time:.1f}s")

# ============================================================
# 7. Feature Importance
# ============================================================
print("\n" + "=" * 70)
print("7. FEATURE IMPORTANCE (top 25)")
print("=" * 70)

for label, model in models_by_type.items():
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(f"\n{label} model — Top 25 features by gain:")
    for _, row in importance.head(25).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:>12.1f}")

# ============================================================
# 8. Error Analysis by Segment
# ============================================================
print("\n" + "=" * 70)
print("8. ERROR ANALYSIS BY SEGMENT")
print("=" * 70)

val_df = train.loc[val_mask].copy()
val_df['pred'] = preds_by_type
val_df['error'] = np.abs(val_df['target'] - val_df['pred'])

# By consumption type
print("MAE by is_consumption:")
for ic in [0, 1]:
    subset = val_df[val_df['is_consumption'] == ic]
    mae = subset['error'].mean()
    label = "Production" if ic == 0 else "Consumption"
    print(f"  {label}: {mae:.4f}")

# By county
print("\nMAE by county (sorted):")
county_mae = val_df.groupby('county')['error'].mean().sort_values(ascending=False)
for county, mae in county_mae.items():
    print(f"  County {county:>2}: {mae:.4f}")

# By hour
print("\nMAE by hour:")
hour_mae = val_df.groupby('hour')['error'].mean()
print(f"  Peak error hour: {hour_mae.idxmax()} ({hour_mae.max():.4f})")
print(f"  Min error hour: {hour_mae.idxmin()} ({hour_mae.min():.4f})")

# ============================================================
# 9. Results Summary
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Model':<35} {'MAE':>10} {'vs Baseline':>12}")
print("-" * 60)
print(f"{'Lag-24 baseline':<35} {baseline_mae:>10.4f} {'---':>12}")
print(f"{'LightGBM single':<35} {mae_single:>10.4f} {(baseline_mae-mae_single)/baseline_mae*100:>+11.1f}%")
print(f"{'LightGBM split (prod+cons)':<35} {mae_split:>10.4f} {(baseline_mae-mae_split)/baseline_mae*100:>+11.1f}%")
print(f"  Production MAE: {maes_by_type['Production']:.4f}")
print(f"  Consumption MAE: {maes_by_type['Consumption']:.4f}")

# ============================================================
# 10. Log experiments
# ============================================================
with open(exp_file, 'r') as f:
    experiments = json.load(f)

# Baseline
experiments.append({
    "experiment_id": len(experiments) + 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "lag_24_predictor",
    "features": "target_lag_24_only",
    "params": {},
    "cv_strategy": f"time_split_block_{TRAIN_CUTOFF}",
    "cv_scores": [],
    "cv_mean": round(baseline_mae, 4),
    "cv_std": 0,
    "notes": "Predict using yesterday's same-hour value"
})

# Single LightGBM
experiments.append({
    "experiment_id": len(experiments) + 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "modeling",
    "model": "LightGBM-single",
    "features": "engineered_v1_56_features",
    "params": lgb_params,
    "cv_strategy": f"time_split_block_{TRAIN_CUTOFF}",
    "cv_scores": [],
    "cv_mean": round(mae_single, 4),
    "cv_std": 0,
    "training_time_sec": round(single_time, 1),
    "best_iteration": model_single.best_iteration,
    "notes": "Single LightGBM for all segments"
})

# Split LightGBM
experiments.append({
    "experiment_id": len(experiments) + 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "modeling",
    "model": "LightGBM-split-prod-cons",
    "features": "engineered_v1_56_features",
    "params": lgb_params,
    "cv_strategy": f"time_split_block_{TRAIN_CUTOFF}",
    "cv_scores": [round(maes_by_type['Production'], 4),
                  round(maes_by_type['Consumption'], 4)],
    "cv_mean": round(mae_split, 4),
    "cv_std": 0,
    "training_time_sec": round(split_time, 1),
    "notes": "Separate LightGBM for production and consumption"
})

with open(exp_file, 'w') as f:
    json.dump(experiments, f, indent=2)
print(f"\nExperiments saved to {exp_file}")

# Save models for later use
import pickle
model_dir = "competitions/predict-energy-behavior-of-prosumers/models"
os.makedirs(model_dir, exist_ok=True)
for label, model in models_by_type.items():
    path = os.path.join(model_dir, f"lgbm_{label.lower()}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {label} model to {path}")
