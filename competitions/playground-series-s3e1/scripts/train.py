"""
Modeling — playground-series-s3e1 (California Housing)

CV scheme: standard 5-fold KFold (shuffle, seed=42).
Justification (from EDA): target is continuous & roughly unimodal (skew ~0.97,
no discrete classes to stratify on), no group/time structure, and train/test
feature distributions are near-identical (<1.2% mean diff on all columns) ->
plain KFold is appropriate, no need for stratified/group/time splits.

Objective: RMSE (matches config.yaml evaluation_metric = rmse, minimize).
Models: LightGBM, XGBoost, CatBoost (all regression, L2/RMSE objective).
Blend: OOF grid-search over convex combination weights (step 0.05) to
minimize blended OOF RMSE.
"""

import pandas as pd
import numpy as np
import os
import time
import json
import importlib.util
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

COMPETITION_DIR = "competitions/playground-series-s3e1"
TARGET_COL = "MedHouseVal"
ID_COL = "id"
N_SPLITS = 5
SEED = 42

# --- Load experiment_log utility by path (mandatory pattern) ---
_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
test = pd.read_csv(os.path.join(data_dir, "test_processed.csv"))

feature_cols = [c for c in train.columns if c not in (TARGET_COL, ID_COL)]
X = train[feature_cols].values
y = train[TARGET_COL].values
X_test = test[feature_cols].values

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(X))

oof = {"LGB": np.zeros(len(train)), "XGB": np.zeros(len(train)), "CAT": np.zeros(len(train))}
test_preds = {"LGB": np.zeros(len(test)), "XGB": np.zeros(len(test)), "CAT": np.zeros(len(test))}
fold_scores = {"LGB": [], "XGB": [], "CAT": []}
times = {}

lgb_params = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbosity=-1,
    objective="rmse",
)
xgb_params = dict(
    n_estimators=2000, learning_rate=0.03, max_depth=7,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
    objective="reg:squarederror", tree_method="hist", verbosity=0,
)
cat_params = dict(
    iterations=2000, learning_rate=0.03, depth=8,
    l2_leaf_reg=3.0, random_seed=SEED, loss_function="RMSE",
    verbose=False,
)

t0 = time.time()
for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # --- LightGBM ---
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof["LGB"][va_idx] = m.predict(X_va)
    test_preds["LGB"] += m.predict(X_test) / N_SPLITS
    fold_scores["LGB"].append(mean_squared_error(y_va, oof["LGB"][va_idx]) ** 0.5)

    # --- XGBoost ---
    m = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof["XGB"][va_idx] = m.predict(X_va)
    test_preds["XGB"] += m.predict(X_test) / N_SPLITS
    fold_scores["XGB"].append(mean_squared_error(y_va, oof["XGB"][va_idx]) ** 0.5)

    # --- CatBoost ---
    m = CatBoostRegressor(**cat_params)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, use_best_model=True)
    oof["CAT"][va_idx] = m.predict(X_va)
    test_preds["CAT"] += m.predict(X_test) / N_SPLITS
    fold_scores["CAT"].append(mean_squared_error(y_va, oof["CAT"][va_idx]) ** 0.5)

    print(f"Fold {fold_idx}: LGB={fold_scores['LGB'][-1]:.5f} "
          f"XGB={fold_scores['XGB'][-1]:.5f} CAT={fold_scores['CAT'][-1]:.5f}")

train_time = time.time() - t0
print(f"\nTotal training time: {train_time:.1f}s")

oof_scores = {m: mean_squared_error(y, oof[m]) ** 0.5 for m in oof}
print("\nOOF RMSE per model:")
for m, s in oof_scores.items():
    print(f"  {m}: {s:.5f}")

# --- Weight search blend (grid over simplex, step 0.05) ---
best_score = np.inf
best_w = None
step = 0.05
model_names = ["LGB", "XGB", "CAT"]
grid = np.arange(0, 1.0001, step)
for w_lgb in grid:
    for w_xgb in grid:
        w_cat = 1 - w_lgb - w_xgb
        if w_cat < -1e-9 or w_cat > 1 + 1e-9:
            continue
        w_cat = max(0.0, w_cat)
        blend = w_lgb * oof["LGB"] + w_xgb * oof["XGB"] + w_cat * oof["CAT"]
        score = mean_squared_error(y, blend) ** 0.5
        if score < best_score:
            best_score = score
            best_w = {"LGB": round(float(w_lgb), 4), "XGB": round(float(w_xgb), 4), "CAT": round(float(w_cat), 4)}

print(f"\nBest blend weights: {best_w}")
print(f"Best blend OOF RMSE: {best_score:.5f}")

blend_test = (best_w["LGB"] * test_preds["LGB"] + best_w["XGB"] * test_preds["XGB"]
              + best_w["CAT"] * test_preds["CAT"])
# California housing target is strictly positive, cap at observed max (top-coded at 5.00001)
blend_test_clipped = np.clip(blend_test, train[TARGET_COL].min(), train[TARGET_COL].max())

# --- Save submission ---
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
timestamp = time.strftime("%Y%m%d_%H%M%S")
sub_name = f"sub_lgb_xgb_cat_blend_{best_score:.5f}_{timestamp}.csv"
sub_path = os.path.join(COMPETITION_DIR, "submissions", sub_name)
submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: blend_test_clipped})
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == list(sample_sub.columns)
assert submission.isnull().sum().sum() == 0
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

# --- Log experiment (mandatory v2 schema) ---
base_models = [
    {"name": "LGB", "score": round(float(oof_scores["LGB"]), 5), "time_s": None,
     "params": {k: v for k, v in lgb_params.items() if k != "objective"}},
    {"name": "XGB", "score": round(float(oof_scores["XGB"]), 5), "time_s": None,
     "params": {k: v for k, v in xgb_params.items() if k != "objective"}},
    {"name": "CAT", "score": round(float(oof_scores["CAT"]), 5), "time_s": None,
     "params": {k: v for k, v in cat_params.items() if k != "loss_function"}},
]

exp_id = experiment_log.log_experiment_v2(
    COMPETITION_DIR,
    model="LGB+XGB+CAT weight-searched blend (engineered features)",
    metric="rmse",
    direction="minimize",
    score=round(float(best_score), 6),
    cv={"scheme": "KFold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
    features=feature_cols,
    base_models=base_models,
    ensemble={"method": "oof_weight_grid_search", "weights": best_w, "score": round(float(best_score), 6)},
    postprocess=["clip_to_train_target_range"],
    submission=sub_name,
    notes=(f"5-fold KFold on 24 engineered features (rooms_per_person, dist_to_cities, "
           f"geo_cluster, log-transforms, bedroom_ratio). Total train time {train_time:.1f}s. "
           f"Baseline (exp 1, generic 8-feature blend) was 0.56166."),
)
print(f"\nLogged experiment_id={exp_id}")
