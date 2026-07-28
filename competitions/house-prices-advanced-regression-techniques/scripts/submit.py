"""
Submission for House Prices Competition
=========================================
Best model: Weighted Ensemble (Ridge + Lasso + LightGBM + XGBoost)
CV RMSLE: 0.1141
Target is log-transformed — predictions need expm1() to convert back.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/house-prices"
data_dir = os.path.join(COMPETITION_DIR, "data")
TARGET = "SalePrice"
ID = "Id"
SEED = 42

train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
test = pd.read_csv(os.path.join(data_dir, "test_processed.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

y_train = train[TARGET]  # Already log-transformed
X_train = train.drop(columns=[TARGET, ID])
test_ids = test[ID]
X_test = test.drop(columns=[ID])

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ============================================================
# ERROR ANALYSIS on best single model (XGBoost OOF)
# ============================================================
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

print("\n" + "=" * 50)
print("ERROR ANALYSIS: XGBoost OOF")
print("=" * 50)

xgb_params = {
    "n_estimators": 500, "learning_rate": 0.05, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0
}
xgb_model = xgb.XGBRegressor(**xgb_params, random_state=SEED, verbosity=0)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = cross_val_predict(xgb_model, X_train, y_train, cv=kf)

residuals = y_train - oof
print(f"OOF RMSE: {np.sqrt(mean_squared_error(y_train, oof)):.6f}")
print(f"Residual stats: mean={residuals.mean():.6f}, std={residuals.std():.6f}")
print(f"Worst over-predictions (predicted too high):")
worst_over = residuals.nsmallest(5)
for idx, val in worst_over.items():
    actual = np.expm1(y_train.iloc[idx])
    predicted = np.expm1(oof[idx])
    print(f"  Row {idx}: actual=${actual:,.0f}, predicted=${predicted:,.0f}, error=${actual-predicted:,.0f}")

print(f"Worst under-predictions (predicted too low):")
worst_under = residuals.nlargest(5)
for idx, val in worst_under.items():
    actual = np.expm1(y_train.iloc[idx])
    predicted = np.expm1(oof[idx])
    print(f"  Row {idx}: actual=${actual:,.0f}, predicted=${predicted:,.0f}, error=${actual-predicted:,.0f}")


# ============================================================
# TRAIN FINAL ENSEMBLE ON FULL DATA
# ============================================================
print("\n" + "=" * 50)
print("FINAL MODEL: Weighted Ensemble on full data")
print("=" * 50)

lgbm_params = {
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
    "max_depth": 5, "min_child_samples": 10, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1
}

models = [
    ("Ridge", Ridge(alpha=10.0, random_state=SEED), 0.15),
    ("Lasso", Lasso(alpha=0.0005, random_state=SEED, max_iter=10000), 0.15),
    ("LightGBM", lgb.LGBMRegressor(verbosity=-1, random_state=SEED, **lgbm_params), 0.35),
    ("XGBoost", xgb.XGBRegressor(**xgb_params, random_state=SEED, verbosity=0), 0.35),
]

predictions_log = np.zeros(len(X_test))
for name, model, weight in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions_log += weight * preds
    print(f"  {name}: trained, weight={weight}")

# Convert back from log scale
predictions = np.expm1(predictions_log)

print(f"\nPrediction stats (original scale):")
print(f"  Min:  ${predictions.min():,.0f}")
print(f"  Max:  ${predictions.max():,.0f}")
print(f"  Mean: ${predictions.mean():,.0f}")
print(f"  Median: ${np.median(predictions):,.0f}")
print(f"  (Train mean was ${np.expm1(y_train).mean():,.0f})")


# ============================================================
# FORMAT AND VALIDATE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "Id": test_ids.values.astype(int),
    "SalePrice": predictions
})

print("\n" + "=" * 50)
print("SUBMISSION VALIDATION")
print("=" * 50)

shape_ok = submission.shape == sample_sub.shape
print(f"  Shape: {'PASS' if shape_ok else 'FAIL'} (got {submission.shape}, expected {sample_sub.shape})")

cols_ok = list(submission.columns) == list(sample_sub.columns)
print(f"  Columns: {'PASS' if cols_ok else 'FAIL'}")

nan_count = submission.isnull().sum().sum()
print(f"  NaN: {'PASS' if nan_count == 0 else 'FAIL'} ({nan_count})")

id_match = (submission["Id"].values == sample_sub["Id"].values).all()
print(f"  IDs: {'PASS' if id_match else 'FAIL'}")

positive = (submission["SalePrice"] > 0).all()
print(f"  Positive prices: {'PASS' if positive else 'FAIL'}")

all_pass = shape_ok and cols_ok and nan_count == 0 and id_match and positive


# ============================================================
# SAVE
# ============================================================
if all_pass:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_ensemble_0.1141_{timestamp}.csv"
    output_dir = os.path.join(COMPETITION_DIR, "submissions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")
else:
    print("\nValidation FAILED — submission not saved.")
