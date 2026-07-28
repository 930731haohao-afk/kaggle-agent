"""
Submission: Soft Voting Ensemble for Titanic
=============================================
Retrain on full data and generate predictions.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TARGET_COL = "Survived"
ID_COL = "PassengerId"
RANDOM_SEED = 42

# Load data
train = pd.read_csv(os.path.join(COMPETITION_DIR, "data/train_processed.csv"))
test = pd.read_csv(os.path.join(COMPETITION_DIR, "data/test_processed.csv"))
sample_sub = pd.read_csv(os.path.join(COMPETITION_DIR, "data/gender_submission.csv"))

y_train = train[TARGET_COL].astype(int)
X_train = train.drop(columns=[TARGET_COL, ID_COL])
test_ids = test[ID_COL]
X_test = test.drop(columns=[ID_COL])

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Build ensemble
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             min_samples_split=5, random_state=RANDOM_SEED, n_jobs=-1)
xgb_m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=RANDOM_SEED, use_label_encoder=False,
                            verbosity=0, eval_metric="logloss")
lgb_m = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED,
                             n_estimators=300, learning_rate=0.05,
                             num_leaves=31, max_depth=6)

ensemble = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_m), ("lgb", lgb_m)],
    voting="soft"
)

print("Training ensemble on full data...")
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
print(f"Predictions: {len(predictions)} values")
print(f"Distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")

# Format submission
submission = pd.DataFrame({
    sample_sub.columns[0]: test_ids,
    sample_sub.columns[1]: predictions.astype(int)
})

# Validation
print(f"\nValidation:")
print(f"  Shape: {'PASS' if submission.shape == sample_sub.shape else 'FAIL'}")
print(f"  Columns: {'PASS' if list(submission.columns) == list(sample_sub.columns) else 'FAIL'}")
print(f"  NaN: {'PASS' if submission.isnull().sum().sum() == 0 else 'FAIL'}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submission_ensemble_0.845_{timestamp}.csv"
output_path = os.path.join(COMPETITION_DIR, "submissions", filename)
submission.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
