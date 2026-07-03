"""
Iteration round 4 — full pipeline retrain on the round-3-adopted feature set
(24 original engineered features + knn_mean_dist_10 + coastal_dist = 26
features), reusing the round-1 Optuna-tuned LGB hyperparameters and the
round-2 seed-bagging recipe. Same 5 members as round 2 (LGB, XGB, CAT,
LGB_TUNED, LGB_TUNED_SEED2024), same KFold(seed=42) scheme, retrained end to
end on the new features, then a fresh 5-way weight search.
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
SEED_BAG = 2024
PREV_BEST = 0.55786  # round 2 result

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train_processed_v2.csv"))
test = pd.read_csv(os.path.join(data_dir, "test_processed_v2.csv"))
feature_cols = [c for c in train.columns if c not in (TARGET_COL, ID_COL)]
X = train[feature_cols].values
y = train[TARGET_COL].values
X_test = test[feature_cols].values

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(X))

with open(os.path.join(data_dir, "round1_best_lgb_params.json")) as f:
    tuned_params = json.load(f)
seed_params = dict(tuned_params)
seed_params["random_state"] = SEED_BAG

lgb_params_orig = dict(
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

model_defs = {
    "LGB": ("lgb", lgb_params_orig),
    "XGB": ("xgb", xgb_params),
    "CAT": ("cat", cat_params),
    "LGB_TUNED": ("lgb", tuned_params),
    "LGB_TUNED_SEED2024": ("lgb", seed_params),
}

oof = {k: np.zeros(len(train)) for k in model_defs}
test_preds = {k: np.zeros(len(test)) for k in model_defs}

t0 = time.time()
for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    for name, (kind, params) in model_defs.items():
        if kind == "lgb":
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        elif kind == "xgb":
            m = xgb.XGBRegressor(**params, early_stopping_rounds=100)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            m = CatBoostRegressor(**params)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, use_best_model=True)
        oof[name][va_idx] = m.predict(X_va)
        test_preds[name] += m.predict(X_test) / N_SPLITS
    print(f"Fold {fold_idx} done.")

train_time = time.time() - t0
oof_scores = {k: round(float(mean_squared_error(y, oof[k]) ** 0.5), 5) for k in oof}
print(f"\nTotal training time: {train_time:.1f}s")
print("OOF scores:", oof_scores)

# --- 5-way weight search ---
model_names = list(model_defs.keys())
step = 0.05
grid = np.round(np.arange(0, 1.0001, step), 4)
best_score = np.inf
best_w = None
for w1 in grid:
    for w2 in grid:
        if w1 + w2 > 1.0001:
            continue
        for w3 in grid:
            if w1 + w2 + w3 > 1.0001:
                continue
            for w4 in grid:
                s = w1 + w2 + w3 + w4
                if s > 1.0001:
                    continue
                w5 = round(1 - s, 4)
                if w5 < -1e-9 or w5 > 1 + 1e-9:
                    continue
                w5 = max(0.0, w5)
                blend = (w1 * oof["LGB"] + w2 * oof["XGB"] + w3 * oof["CAT"]
                         + w4 * oof["LGB_TUNED"] + w5 * oof["LGB_TUNED_SEED2024"])
                score = mean_squared_error(y, blend) ** 0.5
                if score < best_score:
                    best_score = score
                    best_w = {"LGB": round(float(w1), 4), "XGB": round(float(w2), 4),
                               "CAT": round(float(w3), 4), "LGB_TUNED": round(float(w4), 4),
                               "LGB_TUNED_SEED2024": w5}

print(f"\nBest 5-way blend weights: {best_w}")
print(f"Best 5-way blend OOF RMSE: {best_score:.5f} (vs round2 best {PREV_BEST})")

blend_test = sum(best_w[k] * test_preds[k] for k in model_names)
blend_test_clipped = np.clip(blend_test, train[TARGET_COL].min(), train[TARGET_COL].max())

improved = best_score < PREV_BEST
if improved:
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sub_name = f"sub_round4_geofeat_5way_blend_{best_score:.5f}_{timestamp}.csv"
    sub_path = os.path.join(COMPETITION_DIR, "submissions", sub_name)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: blend_test_clipped})
    submission.to_csv(sub_path, index=False)
    print(f"\nNew best -> submission saved: {sub_path}")
else:
    sub_name = None
    print("\nNo improvement over round2 best -- not saving a new submission.")

base_models_log = [{"name": k, "score": oof_scores[k]} for k in model_names]

exp_id = experiment_log.log_experiment_v2(
    COMPETITION_DIR,
    model="5-way blend on 26 features (+knn_mean_dist_10 +coastal_dist)",
    metric="rmse",
    direction="minimize",
    score=round(float(best_score), 6),
    cv={"scheme": "KFold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
    features=feature_cols,
    base_models=base_models_log,
    ensemble={"method": "oof_weight_grid_search_5way", "weights": best_w, "score": round(float(best_score), 6)},
    postprocess=["clip_to_train_target_range"],
    submission=sub_name,
    notes=(f"Iteration round 4: full retrain of the same 5-member pool (LGB, XGB, CAT, "
           f"Optuna-tuned LGB from round1, seed-bagged tuned LGB from round2) on round-3's "
           f"adopted feature set (+knn_mean_dist_10, +coastal_dist; 26 features total). "
           f"Training time {train_time:.1f}s. Round2 best (24 features) was {PREV_BEST}; this "
           f"round {'IMPROVED' if improved else 'did NOT improve'} to {best_score:.6f}."),
)
print(f"\nLogged experiment_id={exp_id}")
