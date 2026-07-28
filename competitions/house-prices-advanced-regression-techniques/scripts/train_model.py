"""
Model Training for House Prices Competition
=============================================
Target is log-transformed SalePrice. Metric: RMSE (which = RMSLE on original scale).
Models: baseline, Ridge, Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, ensemble.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/house-prices"
TRAIN_FILE = "data/train_processed.csv"
TARGET = "SalePrice"
ID = "Id"
N_FOLDS = 5
SEED = 42

train = pd.read_csv(os.path.join(COMPETITION_DIR, TRAIN_FILE))
y = train[TARGET]  # Already log-transformed
X = train.drop(columns=[TARGET, ID])

print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Target (log-transformed): mean={y.mean():.4f}, std={y.std():.4f}\n")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def log_experiment(model_name, params, scores, notes=""):
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    with open(exp_file, "r") as f:
        experiments = json.load(f)
    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "features": TRAIN_FILE,
        "n_features": X.shape[1],
        "params": {k: str(v) for k, v in params.items()},
        "cv_strategy": f"{N_FOLDS}-fold",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 6),
        "cv_std": round(float(np.std(scores)), 6),
        "eval_metric": "rmse_log",
        "notes": notes
    }
    experiments.append(experiment)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)
    return exp_id


def cv_model(model, model_name, params, notes=""):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        score = rmse(y_val, preds)
        scores.append(score)
        print(f"  Fold {fold+1}: RMSE={score:.6f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    exp_id = log_experiment(model_name, params, scores, notes)
    print(f"  Mean: {mean_score:.6f} (+/- {std_score:.6f}) [Experiment #{exp_id}]\n")
    return mean_score, oof_preds


# ============================================================
# BASELINE: Mean predictor
# ============================================================
print("=" * 50)
print("BASELINE: Mean Predictor")
print("=" * 50)
mean_pred = y.mean()
baseline_rmse = rmse(y, np.full(len(y), mean_pred))
print(f"  Always predict mean: RMSE={baseline_rmse:.6f}")
log_experiment("MeanPredictor", {}, [baseline_rmse]*N_FOLDS, "Naive baseline")
print()


# ============================================================
# MODEL 1: Ridge Regression
# ============================================================
print("=" * 50)
print("MODEL 1: Ridge Regression")
print("=" * 50)
ridge_params = {"alpha": 10.0}
ridge = Ridge(**ridge_params, random_state=SEED)
cv_model(ridge, "Ridge", ridge_params, "L2 regularized linear regression")


# ============================================================
# MODEL 2: Lasso Regression
# ============================================================
print("=" * 50)
print("MODEL 2: Lasso Regression")
print("=" * 50)
lasso_params = {"alpha": 0.0005}
lasso = Lasso(**lasso_params, random_state=SEED, max_iter=10000)
cv_model(lasso, "Lasso", lasso_params, "L1 regularized linear regression")


# ============================================================
# MODEL 3: ElasticNet
# ============================================================
print("=" * 50)
print("MODEL 3: ElasticNet")
print("=" * 50)
enet_params = {"alpha": 0.0005, "l1_ratio": 0.5}
enet = ElasticNet(**enet_params, random_state=SEED, max_iter=10000)
cv_model(enet, "ElasticNet", enet_params, "L1+L2 regularized linear regression")


# ============================================================
# MODEL 4: LightGBM
# ============================================================
print("=" * 50)
print("MODEL 4: LightGBM")
print("=" * 50)
lgbm_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1
}
lgbm = lgb.LGBMRegressor(verbosity=-1, random_state=SEED, **lgbm_params)
lgbm_score, lgbm_oof = cv_model(lgbm, "LightGBM", lgbm_params, "Tuned LightGBM")


# ============================================================
# MODEL 5: XGBoost
# ============================================================
print("=" * 50)
print("MODEL 5: XGBoost")
print("=" * 50)
xgb_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0
}
xgb_model = xgb.XGBRegressor(**xgb_params, random_state=SEED, verbosity=0)
xgb_score, xgb_oof = cv_model(xgb_model, "XGBoost", xgb_params, "Tuned XGBoost")


# ============================================================
# MODEL 6: Simple Ensemble (average of Ridge, Lasso, LightGBM, XGBoost)
# ============================================================
print("=" * 50)
print("MODEL 6: Weighted Ensemble")
print("=" * 50)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
ensemble_scores = []

models = [
    ("Ridge", Ridge(alpha=10.0, random_state=SEED)),
    ("Lasso", Lasso(alpha=0.0005, random_state=SEED, max_iter=10000)),
    ("LightGBM", lgb.LGBMRegressor(verbosity=-1, random_state=SEED, **lgbm_params)),
    ("XGBoost", xgb.XGBRegressor(**xgb_params, random_state=SEED, verbosity=0)),
]
weights = [0.15, 0.15, 0.35, 0.35]

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_preds = np.zeros(len(X_val))
    for (name, model), w in zip(models, weights):
        model.fit(X_tr, y_tr)
        fold_preds += w * model.predict(X_val)

    score = rmse(y_val, fold_preds)
    ensemble_scores.append(score)
    print(f"  Fold {fold+1}: RMSE={score:.6f}")

mean_score = np.mean(ensemble_scores)
std_score = np.std(ensemble_scores)
exp_id = log_experiment("WeightedEnsemble",
    {"models": "Ridge+Lasso+LightGBM+XGBoost", "weights": "0.15+0.15+0.35+0.35"},
    ensemble_scores, "Weighted average of 4 models")
print(f"  Mean: {mean_score:.6f} (+/- {std_score:.6f}) [Experiment #{exp_id}]\n")


# ============================================================
# LEADERBOARD
# ============================================================
print("=" * 50)
print("EXPERIMENT LEADERBOARD (lower RMSE is better)")
print("=" * 50)

with open(os.path.join(COMPETITION_DIR, "experiments.json"), "r") as f:
    experiments = json.load(f)

sorted_exps = sorted(experiments, key=lambda e: e["cv_mean"])
print(f"\n{'#':<4} {'ID':<5} {'Model':<25} {'CV RMSE':<12} {'CV Std':<10}")
print("-" * 58)
for rank, exp in enumerate(sorted_exps, 1):
    print(f"{rank:<4} {exp['experiment_id']:<5} {exp['model'][:24]:<25} "
          f"{exp['cv_mean']:<12.6f} {exp['cv_std']:<10.6f}")

best = sorted_exps[0]
print(f"\nBest model: {best['model']} with RMSE={best['cv_mean']:.6f}")
print(f"(On original scale, this corresponds to ~RMSLE of {best['cv_mean']:.6f})")
