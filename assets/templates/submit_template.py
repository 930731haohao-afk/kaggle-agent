"""
Submission Generation Template for Kaggle Competitions
=======================================================
Usage: Modify the CONFIG section below, then run the script.
Retrains on full data and generates a formatted submission file.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — Modify these for each competition
# ============================================================
COMPETITION_DIR = "competitions/<name>"
TRAIN_FILE = "data/train_processed.csv"
TEST_FILE = "data/test_processed.csv"
SAMPLE_SUB_FILE = "data/sample_submission.csv"
TARGET_COL = "<target>"
ID_COL = "<id>"
PROBLEM_TYPE = "binary_classification"  # binary_classification, multiclass_classification, regression
USE_PROBABILITIES = True  # For classification: submit probabilities instead of hard labels
RANDOM_SEED = 42
# ============================================================

# Load data
train = pd.read_csv(os.path.join(COMPETITION_DIR, TRAIN_FILE))
test = pd.read_csv(os.path.join(COMPETITION_DIR, TEST_FILE))
sample_sub = pd.read_csv(os.path.join(COMPETITION_DIR, SAMPLE_SUB_FILE))

y_train = train[TARGET_COL]
X_train = train.drop(columns=[TARGET_COL, ID_COL])
test_ids = test[ID_COL]
X_test = test.drop(columns=[ID_COL])

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
print(f"Sample submission columns: {list(sample_sub.columns)}")


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================
# Replace this section with the best model and params from experiments.json

import lightgbm as lgb

best_params = {
    # Fill in from experiments.json
}

print("\nTraining final model on full training data...")

if PROBLEM_TYPE == "binary_classification":
    model = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **best_params)
elif PROBLEM_TYPE == "multiclass_classification":
    model = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **best_params)
else:
    model = lgb.LGBMRegressor(verbosity=-1, random_state=RANDOM_SEED, **best_params)

model.fit(X_train, y_train)
print("Model trained.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

if PROBLEM_TYPE in ["binary_classification", "multiclass_classification"]:
    if USE_PROBABILITIES:
        predictions = model.predict_proba(X_test)
        if PROBLEM_TYPE == "binary_classification":
            predictions = predictions[:, 1]
    else:
        predictions = model.predict(X_test)
else:
    predictions = model.predict(X_test)

print(f"\nPrediction stats:")
print(f"  Shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
print(f"  Min: {np.min(predictions):.6f}")
print(f"  Max: {np.max(predictions):.6f}")
print(f"  Mean: {np.mean(predictions):.6f}")


# ============================================================
# POST-PROCESSING (customize as needed)
# ============================================================

# Example: clip predictions to valid range
# predictions = np.clip(predictions, 0, 1)

# Example: threshold optimization for binary classification
# predictions = (predictions > 0.42).astype(int)


# ============================================================
# FORMAT SUBMISSION
# ============================================================

submission = pd.DataFrame({
    sample_sub.columns[0]: test_ids,
    sample_sub.columns[1]: predictions
})


# ============================================================
# VALIDATION
# ============================================================

print("\nValidation checks:")

# Shape check
shape_ok = submission.shape == sample_sub.shape
print(f"  Shape: {'PASS' if shape_ok else 'FAIL'} "
      f"(got {submission.shape}, expected {sample_sub.shape})")

# Column check
cols_ok = list(submission.columns) == list(sample_sub.columns)
print(f"  Columns: {'PASS' if cols_ok else 'FAIL'} "
      f"(got {list(submission.columns)}, expected {list(sample_sub.columns)})")

# NaN check
nan_count = submission.isnull().sum().sum()
print(f"  NaN: {'PASS' if nan_count == 0 else 'FAIL'} ({nan_count} NaN values)")

# ID check
id_match = (submission[sample_sub.columns[0]].values == sample_sub[sample_sub.columns[0]].values).all()
print(f"  IDs: {'PASS' if id_match else 'FAIL'}")

all_pass = shape_ok and cols_ok and nan_count == 0 and id_match


# ============================================================
# SAVE
# ============================================================

if all_pass:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}.csv"
    output_path = os.path.join(COMPETITION_DIR, "submissions", filename)
    os.makedirs(os.path.join(COMPETITION_DIR, "submissions"), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")
    print(f"Rows: {len(submission)}")
else:
    print("\nWARNING: Validation failed! Submission NOT saved.")
    print("Fix the issues above before saving.")
