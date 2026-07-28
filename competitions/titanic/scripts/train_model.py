"""
Model Training for Titanic Competition
========================================
Baseline + multiple models with cross-validation.
Logs all experiments to experiments.json.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/titanic"
TRAIN_FILE = "data/train_processed.csv"
TARGET_COL = "Survived"
ID_COL = "PassengerId"
EVAL_METRIC = "accuracy"
N_FOLDS = 5
RANDOM_SEED = 42

train = pd.read_csv(os.path.join(COMPETITION_DIR, TRAIN_FILE))
y = train[TARGET_COL].astype(int)
X = train.drop(columns=[TARGET_COL, ID_COL])
feature_names = X.columns.tolist()

print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Target distribution: {dict(y.value_counts())}\n")


def log_experiment(model_name, params, scores, notes=""):
    """Log experiment to experiments.json."""
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    with open(exp_file, "r") as f:
        experiments = json.load(f)

    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "features": TRAIN_FILE,
        "n_features": len(feature_names),
        "params": {k: str(v) for k, v in params.items()},
        "cv_strategy": f"{N_FOLDS}-fold-stratified",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 6),
        "cv_std": round(float(np.std(scores)), 6),
        "eval_metric": EVAL_METRIC,
        "notes": notes
    }
    experiments.append(experiment)

    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)

    return exp_id


def cross_validate_model(model, model_name, params, notes=""):
    """Run stratified K-fold CV and log results."""
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)
        print(f"  Fold {fold+1}: {EVAL_METRIC}={score:.6f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    exp_id = log_experiment(model_name, params, scores, notes)
    print(f"  Mean: {mean_score:.6f} (+/- {std_score:.6f}) [Experiment #{exp_id}]\n")
    return mean_score


# ============================================================
# BASELINE 1: Majority class
# ============================================================
print("=" * 50)
print("BASELINE 1: Majority Class Predictor")
print("=" * 50)
majority_class = y.mode()[0]
majority_acc = (y == majority_class).mean()
print(f"  Always predict {majority_class}: accuracy={majority_acc:.6f}\n")
log_experiment("MajorityClass", {"class": int(majority_class)},
               [majority_acc]*N_FOLDS, "Naive baseline - always predict majority class")


# ============================================================
# BASELINE 2: Logistic Regression
# ============================================================
print("=" * 50)
print("BASELINE 2: Logistic Regression")
print("=" * 50)
lr_params = {"C": 1.0, "max_iter": 1000}
lr = LogisticRegression(**lr_params, random_state=RANDOM_SEED)
cross_validate_model(lr, "LogisticRegression", lr_params, "Simple logistic regression baseline")


# ============================================================
# MODEL 1: Random Forest
# ============================================================
print("=" * 50)
print("MODEL 1: Random Forest")
print("=" * 50)
rf_params = {"n_estimators": 200, "max_depth": 8, "min_samples_split": 5}
rf = RandomForestClassifier(**rf_params, random_state=RANDOM_SEED, n_jobs=-1)
cross_validate_model(rf, "RandomForest", rf_params, "Random Forest with tuned depth")


# ============================================================
# MODEL 2: LightGBM (default)
# ============================================================
print("=" * 50)
print("MODEL 2: LightGBM (default)")
print("=" * 50)
lgbm_params = {}
lgbm = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED)
cross_validate_model(lgbm, "LightGBM-default", lgbm_params, "LightGBM with default params")


# ============================================================
# MODEL 3: LightGBM (tuned)
# ============================================================
print("=" * 50)
print("MODEL 3: LightGBM (tuned)")
print("=" * 50)
lgbm_tuned_params = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1
}
lgbm_tuned = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **lgbm_tuned_params)
cross_validate_model(lgbm_tuned, "LightGBM-tuned", lgbm_tuned_params, "LightGBM with manual tuning")


# ============================================================
# MODEL 4: XGBoost
# ============================================================
print("=" * 50)
print("MODEL 4: XGBoost")
print("=" * 50)
xgb_params = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss"
}
xgb_model = xgb.XGBClassifier(**xgb_params, random_state=RANDOM_SEED, use_label_encoder=False, verbosity=0)
cross_validate_model(xgb_model, "XGBoost", xgb_params, "XGBoost with tuned params")


# ============================================================
# LEADERBOARD
# ============================================================
print("=" * 50)
print("EXPERIMENT LEADERBOARD")
print("=" * 50)

with open(os.path.join(COMPETITION_DIR, "experiments.json"), "r") as f:
    experiments = json.load(f)

sorted_exps = sorted(experiments, key=lambda e: e["cv_mean"], reverse=True)
print(f"\n{'#':<4} {'ID':<5} {'Model':<25} {'CV Mean':<10} {'CV Std':<10}")
print("-" * 55)
for rank, exp in enumerate(sorted_exps, 1):
    print(f"{rank:<4} {exp['experiment_id']:<5} {exp['model'][:24]:<25} "
          f"{exp['cv_mean']:<10.6f} {exp['cv_std']:<10.6f}")

print(f"\nBest model: {sorted_exps[0]['model']} with {EVAL_METRIC}={sorted_exps[0]['cv_mean']:.6f}")
