"""
Iteration round 1 — Optuna tuning of the strongest single base model (LGB, OOF
0.56109 in exp #2), using the fold-proxy recipe from knowledge/experience.md
(s3e7 evidence): tune against a single fold (fold 0) instead of full 5-fold CV
to stay within the time budget, then re-validate the winning config with full
5-fold CV. Per the recipe, the tuned model is ADDED to the model pool as a
4th blend member (not swapped in for the original LGB) to preserve ensemble
diversity (s3e14 evidence: replacing lost diversity, adding gained it).

Timeout-guarded: optuna.study.optimize(..., timeout=SECS) hard-caps wall time
regardless of trial count, so a slow trial can't blow the run over budget.
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

COMPETITION_DIR = "competitions/playground-series-s3e1"
TARGET_COL = "MedHouseVal"
ID_COL = "id"
N_SPLITS = 5
SEED = 42
N_TRIALS = 50
TIMEOUT_S = 480  # 8-minute hard cap for the Optuna search itself

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

# --- fold-0 proxy objective ---
tr_idx0, va_idx0 = folds[0]
X_tr0, X_va0 = X[tr_idx0], X[va_idx0]
y_tr0, y_va0 = y[tr_idx0], y[va_idx0]


def objective(trial):
    params = dict(
        n_estimators=2000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 127),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=SEED, verbosity=-1, objective="rmse",
    )
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr0, y_tr0, eval_set=[(X_va0, y_va0)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = m.predict(X_va0)
    return mean_squared_error(y_va0, pred) ** 0.5


t_search0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_S)
search_time = time.time() - t_search0

print(f"Optuna search: {len(study.trials)} trials in {search_time:.1f}s")
print(f"Best fold-0 RMSE: {study.best_value:.5f}")
print(f"Best params: {study.best_params}")

best_params = dict(study.best_params)
best_params.update(dict(n_estimators=2000, random_state=SEED, verbosity=-1, objective="rmse"))

# --- Re-validate winning config with full 5-fold CV ---
oof_tuned = np.zeros(len(train))
test_pred_tuned = np.zeros(len(test))
fold_scores = []
t0 = time.time()
for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    m = lgb.LGBMRegressor(**best_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_tuned[va_idx] = m.predict(X_va)
    test_pred_tuned += m.predict(X_test) / N_SPLITS
    fold_scores.append(mean_squared_error(y_va, oof_tuned[va_idx]) ** 0.5)
    print(f"Fold {fold_idx}: tuned LGB RMSE={fold_scores[-1]:.5f}")

full_cv_time = time.time() - t0
tuned_oof_score = mean_squared_error(y, oof_tuned) ** 0.5
print(f"\nFull 5-fold CV time: {full_cv_time:.1f}s")
print(f"Tuned LGB OOF RMSE: {tuned_oof_score:.5f} (vs exp #2 LGB 0.56109)")

# --- Load exp #2 base OOF/test preds are not persisted; recompute original LGB/XGB/CAT
#     is expensive. Instead: reuse original blend approach by re-training the ORIGINAL
#     3 base models here as well, so we can do the 4-way weight search fairly against
#     the same fold split (guaranteed same folds/seed as exp #2). ---
import xgboost as xgb
from catboost import CatBoostRegressor

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

oof = {"LGB": np.zeros(len(train)), "XGB": np.zeros(len(train)), "CAT": np.zeros(len(train)),
       "LGB_TUNED": oof_tuned}
test_preds = {"LGB": np.zeros(len(test)), "XGB": np.zeros(len(test)), "CAT": np.zeros(len(test)),
              "LGB_TUNED": test_pred_tuned}

t1 = time.time()
for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    m = lgb.LGBMRegressor(**lgb_params_orig)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof["LGB"][va_idx] = m.predict(X_va)
    test_preds["LGB"] += m.predict(X_test) / N_SPLITS

    m = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof["XGB"][va_idx] = m.predict(X_va)
    test_preds["XGB"] += m.predict(X_test) / N_SPLITS

    m = CatBoostRegressor(**cat_params)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, use_best_model=True)
    oof["CAT"][va_idx] = m.predict(X_va)
    test_preds["CAT"] += m.predict(X_test) / N_SPLITS

recompute_time = time.time() - t1
oof_scores = {m: round(float(mean_squared_error(y, oof[m]) ** 0.5), 5) for m in oof}
print(f"\nRecompute base models time: {recompute_time:.1f}s")
print("OOF scores:", oof_scores)

# --- 4-way weight search (grid, step 0.05, simplex) ---
model_names = ["LGB", "XGB", "CAT", "LGB_TUNED"]
step = 0.05
grid = np.round(np.arange(0, 1.0001, step), 4)
best_score = np.inf
best_w = None
for w_lgb in grid:
    for w_xgb in grid:
        if w_lgb + w_xgb > 1.0001:
            continue
        for w_cat in grid:
            if w_lgb + w_xgb + w_cat > 1.0001:
                continue
            w_tuned = 1 - w_lgb - w_xgb - w_cat
            if w_tuned < -1e-9 or w_tuned > 1 + 1e-9:
                continue
            w_tuned = max(0.0, round(float(w_tuned), 4))
            blend = (w_lgb * oof["LGB"] + w_xgb * oof["XGB"] + w_cat * oof["CAT"]
                     + w_tuned * oof["LGB_TUNED"])
            score = mean_squared_error(y, blend) ** 0.5
            if score < best_score:
                best_score = score
                best_w = {"LGB": round(float(w_lgb), 4), "XGB": round(float(w_xgb), 4),
                           "CAT": round(float(w_cat), 4), "LGB_TUNED": w_tuned}

print(f"\nBest 4-way blend weights: {best_w}")
print(f"Best 4-way blend OOF RMSE: {best_score:.5f} (vs exp #2 blend 0.558768)")

blend_test = (best_w["LGB"] * test_preds["LGB"] + best_w["XGB"] * test_preds["XGB"]
              + best_w["CAT"] * test_preds["CAT"] + best_w["LGB_TUNED"] * test_preds["LGB_TUNED"])
blend_test_clipped = np.clip(blend_test, train[TARGET_COL].min(), train[TARGET_COL].max())

# --- Save artifacts needed for later rounds (seed bagging etc.) ---
np.savez(os.path.join(COMPETITION_DIR, "data", "round1_oof_test.npz"),
          oof_LGB=oof["LGB"], oof_XGB=oof["XGB"], oof_CAT=oof["CAT"], oof_LGB_TUNED=oof["LGB_TUNED"],
          test_LGB=test_preds["LGB"], test_XGB=test_preds["XGB"], test_CAT=test_preds["CAT"],
          test_LGB_TUNED=test_preds["LGB_TUNED"], y=y)
with open(os.path.join(COMPETITION_DIR, "data", "round1_best_lgb_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

improved = best_score < 0.558768
if improved:
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sub_name = f"sub_lgb_xgb_cat_lgbtuned_blend_{best_score:.5f}_{timestamp}.csv"
    sub_path = os.path.join(COMPETITION_DIR, "submissions", sub_name)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: blend_test_clipped})
    submission.to_csv(sub_path, index=False)
    print(f"\nNew best -> submission saved: {sub_path}")
else:
    sub_name = None
    print("\nNo improvement over exp #2 best (0.558768) -- not saving a new submission.")

base_models_log = [
    {"name": "LGB", "score": oof_scores["LGB"]},
    {"name": "XGB", "score": oof_scores["XGB"]},
    {"name": "CAT", "score": oof_scores["CAT"]},
    {"name": "LGB_TUNED", "score": oof_scores["LGB_TUNED"], "params": best_params,
     "tuning": {"method": "optuna_tpe_fold0_proxy", "n_trials": len(study.trials),
                "search_time_s": round(search_time, 1), "fold0_proxy_rmse": round(float(study.best_value), 5)}},
]

exp_id = experiment_log.log_experiment_v2(
    COMPETITION_DIR,
    model="LGB+XGB+CAT+Optuna-tuned-LGB 4-way weight-searched blend",
    metric="rmse",
    direction="minimize",
    score=round(float(best_score), 6),
    cv={"scheme": "KFold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
    features=feature_cols,
    base_models=base_models_log,
    ensemble={"method": "oof_weight_grid_search_4way", "weights": best_w, "score": round(float(best_score), 6)},
    postprocess=["clip_to_train_target_range"],
    submission=sub_name,
    notes=(f"Iteration round 1: Optuna TPE fold-0-proxy tuning of LGB ({len(study.trials)} trials, "
           f"{search_time:.1f}s search + {full_cv_time:.1f}s full-CV revalidation). Tuned LGB OOF "
           f"{tuned_oof_score:.5f} (solo) vs exp#2 LGB 0.56109. Per experience.md recipe, ADDED as 4th "
           f"pool member (not replacing original LGB) to preserve ensemble diversity. "
           f"Exp#2 best was 0.558768; this round {'IMPROVED' if improved else 'did NOT improve'} to {best_score:.6f}."),
)
print(f"\nLogged experiment_id={exp_id}")
