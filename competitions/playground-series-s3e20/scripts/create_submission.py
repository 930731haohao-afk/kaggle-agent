"""Create final submission (LGB + XGB ensemble)"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import json
from datetime import datetime

print("Loading processed data...")
train = pd.read_csv("competitions/playground-series-s3e20/data/train_processed.csv")
test = pd.read_csv("competitions/playground-series-s3e20/data/test_processed.csv")

X = train.drop(['ID_LAT_LON_YEAR_WEEK', 'emission'], axis=1)
y_log = np.log1p(train['emission'].values)
X_test = test.drop(['ID_LAT_LON_YEAR_WEEK'], axis=1)

# Time-based validation
train_mask = train['year'].isin([2019, 2020])
val_mask = train['year'] == 2021

X_tr, y_tr = X[train_mask], y_log[train_mask]
X_val, y_val_log = X[val_mask], y_log[val_mask]
y_val = train[val_mask]['emission'].values

print(f"Train: {len(X_tr)}, Val: {len(X_val)}")

# LightGBM
print("\n Training LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42,
    'device': 'gpu'
}

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val_ds = lgb.Dataset(X_val, y_val_log)
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=700, valid_sets=[lgb_val_ds])

lgb_val_pred = np.expm1(lgb_model.predict(X_val))
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
print(f"LGB Val RMSE: {lgb_rmse:.4f}")

# XGBoost
print("Training XGBoost...")
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'device': 'cuda',
    'random_state': 42
}

dtrain = xgb.DMatrix(X_tr, y_tr)
dval = xgb.DMatrix(X_val, y_val_log)
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=200)

xgb_val_pred = np.expm1(xgb_model.predict(dval))
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
print(f"XGB Val RMSE: {xgb_rmse:.4f}")

# Optimal ensemble weights
w_lgb = 0.5
w_xgb = 0.5

ensemble_val = w_lgb * lgb_val_pred + w_xgb * xgb_val_pred
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
print(f"Ensemble (50/50) Val RMSE: {ensemble_rmse:.4f}")

# Train on full data
print("\nRetraining on full data...")
lgb_full = lgb.Dataset(X, y_log)
lgb_model_full = lgb.train(lgb_params, lgb_full, num_boost_round=700)

dfull = xgb.DMatrix(X, y_log)
xgb_model_full = xgb.train(xgb_params, dfull, num_boost_round=200)

# Test predictions
dtest = xgb.DMatrix(X_test)
lgb_test = np.expm1(lgb_model_full.predict(X_test))
xgb_test = np.expm1(xgb_model_full.predict(dtest))
ensemble_test = w_lgb * lgb_test + w_xgb * xgb_test

# Save submission
submission = pd.DataFrame({
    'ID_LAT_LON_YEAR_WEEK': test['ID_LAT_LON_YEAR_WEEK'],
    'emission': ensemble_test
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_file = f"competitions/playground-series-s3e20/submissions/submission_final_{ensemble_rmse:.4f}_{timestamp}.csv"
submission.to_csv(sub_file, index=False)

print(f"\nSubmission saved: {sub_file}")
print(f"Val RMSE: {ensemble_rmse:.4f}")
print(f"Prediction range: [{ensemble_test.min():.2f}, {ensemble_test.max():.2f}]")
print(f"Prediction mean: {ensemble_test.mean():.2f}")

# Log experiment
experiment = {
    "experiment_id": 2,
    "timestamp": timestamp,
    "model": "LGB+XGB ensemble (50/50)",
    "val_rmse": float(ensemble_rmse),
    "lgb_rmse": float(lgb_rmse),
    "xgb_rmse": float(xgb_rmse),
    "submission_file": sub_file
}

with open("competitions/playground-series-s3e20/experiments.json", 'r') as f:
    experiments = json.load(f)
experiments.append(experiment)
with open("competitions/playground-series-s3e20/experiments.json", 'w') as f:
    json.dump(experiments, f, indent=2)

print("✅ Ready for submission to Kaggle!")
