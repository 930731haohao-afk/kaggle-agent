"""
Model Training Template for Kaggle Competitions
=================================================
Usage: Modify the CONFIG section below, then run the script.
Supports baseline models, Auto-ML, and custom training.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — Modify these for each competition
# ============================================================
COMPETITION_DIR = "competitions/<name>"
TRAIN_FILE = "data/train_processed.csv"
TARGET_COL = "<target>"
ID_COL = "<id>"
PROBLEM_TYPE = "binary_classification"  # binary_classification, multiclass_classification, regression
EVAL_METRIC = "auc"  # auc, accuracy, f1, log_loss, rmse, mae, r2
N_FOLDS = 5
RANDOM_SEED = 42
# ============================================================

train = pd.read_csv(os.path.join(COMPETITION_DIR, TRAIN_FILE))
y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, ID_COL])
feature_names = X.columns.tolist()

print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Target distribution:\n{y.describe()}\n")


def evaluate(y_true, y_pred, y_prob=None):
    """Compute evaluation metric."""
    if EVAL_METRIC == "auc":
        return roc_auc_score(y_true, y_prob)
    elif EVAL_METRIC == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif EVAL_METRIC == "f1":
        return f1_score(y_true, y_pred, average="macro")
    elif EVAL_METRIC == "log_loss":
        return log_loss(y_true, y_prob)
    elif EVAL_METRIC == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif EVAL_METRIC == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif EVAL_METRIC == "r2":
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {EVAL_METRIC}")


def cross_validate(model_fn, model_name: str, params: dict):
    """Run K-Fold CV and return scores."""
    if PROBLEM_TYPE in ["binary_classification", "multiclass_classification"]:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn(params)
        model.fit(X_tr, y_tr)

        if PROBLEM_TYPE in ["binary_classification", "multiclass_classification"]:
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)
            if PROBLEM_TYPE == "binary_classification":
                y_prob = y_prob[:, 1]
            score = evaluate(y_val, y_pred, y_prob)
        else:
            y_pred = model.predict(X_val)
            score = evaluate(y_val, y_pred)

        scores.append(score)
        print(f"  Fold {fold+1}: {EVAL_METRIC}={score:.6f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\n  Mean: {mean_score:.6f} (+/- {std_score:.6f})")

    # Log experiment
    log_experiment(model_name, params, scores, mean_score, std_score)
    return mean_score, std_score


def log_experiment(model_name, params, scores, mean_score, std_score):
    """Append experiment to experiments.json."""
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    if os.path.exists(exp_file):
        with open(exp_file, "r") as f:
            experiments = json.load(f)
    else:
        experiments = []

    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "features": TRAIN_FILE,
        "n_features": len(feature_names),
        "params": {k: str(v) for k, v in params.items()},
        "cv_strategy": f"{N_FOLDS}-fold",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(mean_score, 6),
        "cv_std": round(std_score, 6),
        "eval_metric": EVAL_METRIC,
        "notes": ""
    }
    experiments.append(experiment)

    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)

    print(f"\n  Logged as experiment #{exp_id}")


# ============================================================
# BASELINE: LightGBM with defaults
# ============================================================

print("=" * 50)
print("BASELINE: LightGBM (default params)")
print("=" * 50)

import lightgbm as lgb

def lgbm_model(params):
    if PROBLEM_TYPE == "binary_classification":
        return lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **params)
    elif PROBLEM_TYPE == "multiclass_classification":
        return lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **params)
    else:
        return lgb.LGBMRegressor(verbosity=-1, random_state=RANDOM_SEED, **params)

baseline_params = {}
baseline_score, baseline_std = cross_validate(lgbm_model, "LightGBM-baseline", baseline_params)

print(f"\nBaseline {EVAL_METRIC}: {baseline_score:.6f}")
print("Training complete.")
