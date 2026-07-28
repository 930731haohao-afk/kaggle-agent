"""
Submission Generation for Titanic Competition
===============================================
Retrains best model (XGBoost) on full data, generates and validates submission.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TRAIN_FILE = "data/train_processed.csv"
TEST_FILE = "data/test_processed.csv"
SAMPLE_SUB_FILE = "data/gender_submission.csv"
TARGET_COL = "Survived"
ID_COL = "PassengerId"
RANDOM_SEED = 42

# Load data
train = pd.read_csv(os.path.join(COMPETITION_DIR, TRAIN_FILE))
test = pd.read_csv(os.path.join(COMPETITION_DIR, TEST_FILE))
sample_sub = pd.read_csv(os.path.join(COMPETITION_DIR, SAMPLE_SUB_FILE))

y_train = train[TARGET_COL].astype(int)
X_train = train.drop(columns=[TARGET_COL, ID_COL])
test_ids = test[ID_COL]
X_test = test.drop(columns=[ID_COL])

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Sample submission: {sample_sub.shape}, columns: {list(sample_sub.columns)}")


# ============================================================
# EVALUATION: Error analysis on best model
# ============================================================

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "=" * 50)
print("ERROR ANALYSIS: XGBoost")
print("=" * 50)

best_params = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss"
}

model = xgb.XGBClassifier(**best_params, random_state=RANDOM_SEED, use_label_encoder=False, verbosity=0)

# Out-of-fold predictions for error analysis
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof_preds = cross_val_predict(model, X_train, y_train, cv=kf)

print(f"\nOOF Accuracy: {accuracy_score(y_train, oof_preds):.6f}")
print(f"\nClassification Report:\n{classification_report(y_train, oof_preds)}")

cm = confusion_matrix(y_train, oof_preds)
print(f"Confusion Matrix:")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

# Misclassification analysis
train_with_preds = train.copy()
train_with_preds["Predicted"] = oof_preds
train_with_preds["Correct"] = (train_with_preds[TARGET_COL] == train_with_preds["Predicted"]).astype(int)
misclassified = train_with_preds[train_with_preds["Correct"] == 0]
print(f"\nMisclassified: {len(misclassified)} / {len(train)} ({len(misclassified)/len(train)*100:.1f}%)")

# Patterns in misclassified
print(f"\nMisclassified by Sex:")
print(misclassified.groupby("Sex").size())
print(f"\nMisclassified by Pclass:")
print(misclassified.groupby("Pclass").size())


# ============================================================
# TRAIN FINAL MODEL ON FULL DATA
# ============================================================

print("\n" + "=" * 50)
print("FINAL MODEL: Retrain on full data")
print("=" * 50)

final_model = xgb.XGBClassifier(**best_params, random_state=RANDOM_SEED, use_label_encoder=False, verbosity=0)
final_model.fit(X_train, y_train)
print("Model trained on all 891 rows.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

predictions = final_model.predict(X_test)
print(f"\nPrediction distribution:")
print(f"  Survived=0: {(predictions == 0).sum()}")
print(f"  Survived=1: {(predictions == 1).sum()}")
print(f"  Survival rate: {predictions.mean():.4f} (train was {y_train.mean():.4f})")


# ============================================================
# FORMAT SUBMISSION
# ============================================================

submission = pd.DataFrame({
    sample_sub.columns[0]: test_ids.values,
    sample_sub.columns[1]: predictions.astype(int)
})


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 50)
print("SUBMISSION VALIDATION")
print("=" * 50)

shape_ok = submission.shape == sample_sub.shape
print(f"  Shape: {'PASS' if shape_ok else 'FAIL'} (got {submission.shape}, expected {sample_sub.shape})")

cols_ok = list(submission.columns) == list(sample_sub.columns)
print(f"  Columns: {'PASS' if cols_ok else 'FAIL'} (got {list(submission.columns)})")

nan_count = submission.isnull().sum().sum()
print(f"  NaN: {'PASS' if nan_count == 0 else 'FAIL'} ({nan_count} NaN values)")

id_match = (submission[sample_sub.columns[0]].values == sample_sub[sample_sub.columns[0]].values).all()
print(f"  IDs: {'PASS' if id_match else 'FAIL'}")

values_ok = submission[sample_sub.columns[1]].isin([0, 1]).all()
print(f"  Values: {'PASS' if values_ok else 'FAIL'} (all 0 or 1)")

all_pass = shape_ok and cols_ok and nan_count == 0 and id_match and values_ok


# ============================================================
# SAVE
# ============================================================

if all_pass:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_score = 0.8440
    filename = f"submission_xgboost_{cv_score}_{timestamp}.csv"
    output_dir = os.path.join(COMPETITION_DIR, "submissions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")
    print(f"Rows: {len(submission)}")
else:
    print("\nWARNING: Validation failed! Submission NOT saved.")
