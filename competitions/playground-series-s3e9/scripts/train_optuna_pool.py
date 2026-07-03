"""Self-improvement Round 2: Optuna-tuned LGB (full 5-fold CV objective, small data)
added to the model pool (not replacing original LGB), matching the validated
recipe in knowledge/experience.md ("小資料 Optuna 全 CV -> 加入池 -> seed bag").

Data is small (5,407 rows) and each full 5-fold LGB fit takes only ~7-8s (see
experiment #2/#4/#5 timing), so a full 5-fold CV *is* the Optuna objective per
trial (no fold-0 proxy needed) -- same pattern validated in s3e5/s3e3.

Same folds/seed/features as experiment #2 (12.07347, current best). Round 1
(duplicate-group target smoothing, experiment #5) was tried and did NOT improve
(12.08122 > 12.07347) so this round reverts to the ORIGINAL (non-smoothed) targets
and only changes LGB hyperparameters via Optuna, then:
  1. logs the tuned-LGB-added-to-pool blend as one experiment (Round 2)
  2. seed-bags the tuned LGB (2nd random_state) as an extra pool member and logs
     that as a second experiment (Round 3), per the "add to pool, don't replace"
     + "seed bag after tuning" combined recipe.
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e9"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "Strength", "id"
N_SPLITS, SEED = 5, 42

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

Xtr_full = build_features(train)
Xte_full = build_features(test)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.float64)
Xtest = Xte_full[FEATS].to_numpy(np.float32)

ybin = pd.qcut(y, 10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, ybin))


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def fit_lgb_full_cv(params, seed=SEED):
    import lightgbm as lgb
    p = dict(params)
    p.update(objective="regression", metric="rmse", random_state=seed, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def objective(trial):
    params = dict(
        n_estimators=3000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 63),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
    oof, _ = fit_lgb_full_cv(params)
    return rmse(y, oof)


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=60, timeout=400, show_progress_bar=False)
tuning_time = time.time() - t0
print(f"Optuna: {len(study.trials)} trials in {tuning_time:.1f}s, best RMSE={study.best_value:.5f}")
print(f"Best params: {study.best_params}")

best_params = dict(study.best_params)
best_params["n_estimators"] = 3000

# retrain best-params LGB (original seed) to get OOF/pred for pooling
oof_tuned, pred_tuned = fit_lgb_full_cv(best_params, seed=SEED)
score_tuned = rmse(y, oof_tuned)
print(f"LGB_tuned (seed={SEED}) OOF RMSE = {score_tuned:.5f}  (orig LGB was 12.11061)")

# --- reproduce ORIGINAL 3-model pool (identical hyperparams to experiment #2/#4) ---
def run_lgb_orig():
    import lightgbm as lgb
    params = dict(objective="regression", metric="rmse", n_estimators=3000,
                  learning_rate=0.02, num_leaves=15, max_depth=5, min_child_samples=25,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=2.0, reg_lambda=4.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def run_xgb_orig():
    import xgboost as xgb
    params = dict(objective="reg:squarederror", n_estimators=3000, learning_rate=0.02,
                  max_depth=4, min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=2.0, reg_lambda=4.0, random_state=SEED, n_jobs=-1,
                  eval_metric="rmse", early_stopping_rounds=150)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def run_cat_orig():
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=4000,
                              learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                              random_seed=SEED, thread_count=-1, verbose=False,
                              allow_writing_files=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


t0 = time.time(); oof_lgb, pred_lgb = run_lgb_orig(); t_lgb = time.time() - t0
t0 = time.time(); oof_xgb, pred_xgb = run_xgb_orig(); t_xgb = time.time() - t0
t0 = time.time(); oof_cat, pred_cat = run_cat_orig(); t_cat = time.time() - t0
print(f"orig LGB={rmse(y, oof_lgb):.5f}  orig XGB={rmse(y, oof_xgb):.5f}  orig CAT={rmse(y, oof_cat):.5f}")


def weight_search(oof_dict, names):
    oofs = np.stack([oof_dict[n] for n in names], axis=1)
    k = len(names)
    best_w, best_s = None, 1e9
    # coarse simplex grid search generalized to k models via recursive grid (grid=0.05)
    grid = np.arange(0, 1.01, 0.05)
    if k == 3:
        for w0 in grid:
            for w1 in np.arange(0, 1.01 - w0, 0.05):
                w2 = 1 - w0 - w1
                if w2 < -1e-9:
                    continue
                s = rmse(y, oofs @ np.array([w0, w1, w2]))
                if s < best_s:
                    best_s, best_w = s, (w0, w1, w2)
    else:
        # random + coordinate-descent search for k>3 (grid explosion), many restarts
        rng = np.random.RandomState(SEED)
        w = np.ones(k) / k
        best_w, best_s = tuple(w), rmse(y, oofs @ w)
        for _ in range(4000):
            w_try = rng.dirichlet(np.ones(k) * 2.0)
            s = rmse(y, oofs @ w_try)
            if s < best_s:
                best_s, best_w = s, tuple(w_try)
        # local refine via coordinate steps
        for _ in range(30):
            improved = False
            for i in range(k):
                for delta in (0.02, -0.02, 0.05, -0.05):
                    w_try = np.array(best_w, dtype=float)
                    w_try[i] = max(0.0, w_try[i] + delta)
                    if w_try.sum() <= 0:
                        continue
                    w_try = w_try / w_try.sum()
                    s = rmse(y, oofs @ w_try)
                    if s < best_s - 1e-9:
                        best_s, best_w = s, tuple(w_try)
                        improved = True
            if not improved:
                break
    return best_w, best_s


# --- Round 2: 4-way pool (orig LGB, orig XGB, orig CAT, tuned LGB) ---
names4 = ["LGB", "XGB", "CAT", "LGB_tuned"]
oof_dict4 = {"LGB": oof_lgb, "XGB": oof_xgb, "CAT": oof_cat, "LGB_tuned": oof_tuned}
pred_dict4 = {"LGB": pred_lgb, "XGB": pred_xgb, "CAT": pred_cat, "LGB_tuned": pred_tuned}
w4, s4 = weight_search(oof_dict4, names4)
print(f"\nROUND 2 (4-way, +LGB_tuned) weights={dict(zip(names4, np.round(w4, 3)))} OOF RMSE={s4:.5f}")

base_models_r2 = [
    dict(model="LGB", rmse=round(rmse(y, oof_lgb), 5), time_s=round(t_lgb, 1)),
    dict(model="XGB", rmse=round(rmse(y, oof_xgb), 5), time_s=round(t_xgb, 1)),
    dict(model="CAT", rmse=round(rmse(y, oof_cat), 5), time_s=round(t_cat, 1)),
    dict(model="LGB_tuned", rmse=round(score_tuned, 5), time_s=round(tuning_time, 1),
         note="Optuna 60-trial full-5fold-CV objective tuned LGB; params=" + str(best_params)),
]
exp_id2 = experiment_log.log_experiment_v2(
    COMP,
    model="4-way blend: LGB+XGB+CatBoost (orig) + Optuna-tuned LGB added to pool",
    metric="rmse", direction="minimize", score=s4,
    cv=dict(strategy=f"{N_SPLITS}fold_stratified_strength_decile", seed=SEED),
    features=FEATS, base_models=base_models_r2,
    ensemble=dict(weights=dict(zip(names4, [round(float(x), 3) for x in w4])),
                  method="oof_weight_search(simplex_grid0.05_for3+dirichlet_random_for4)"),
    submission=None,
    notes=(f"Round 2 self-improvement: Optuna (TPE, 60 trials, {len(study.trials)} run, "
           f"timeout 400s, {tuning_time:.1f}s actual) tuned LGB on full 5-fold CV RMSE "
           f"objective (same folds/seed as exp#2). Solo tuned LGB OOF {score_tuned:.5f} "
           f"vs orig LGB {rmse(y, oof_lgb):.5f}. Added as 4th pool member (not replacing "
           f"orig LGB) per validated recipe. Reverted Round 1's dup-smoothing (exp#5, "
           f"12.08122, no improvement) back to original targets. Result: {s4:.5f} vs "
           f"best-so-far 12.07347 ({'IMPROVED' if s4 < 12.073474 else 'no improvement'})."),
)
print(f"Logged experiment #{exp_id2}")

# --- Round 3: seed-bag the tuned LGB (2nd seed) as a 5th pool member ---
t0 = time.time()
oof_tuned2, pred_tuned2 = fit_lgb_full_cv(best_params, seed=SEED + 1000)
t_seed = time.time() - t0
score_tuned2 = rmse(y, oof_tuned2)
print(f"\nLGB_tuned seed-bag (seed={SEED+1000}) OOF RMSE = {score_tuned2:.5f}  [{t_seed:.1f}s]")

names5 = names4 + ["LGB_tuned_seed2"]
oof_dict5 = dict(oof_dict4); oof_dict5["LGB_tuned_seed2"] = oof_tuned2
pred_dict5 = dict(pred_dict4); pred_dict5["LGB_tuned_seed2"] = pred_tuned2
w5, s5 = weight_search(oof_dict5, names5)
print(f"ROUND 3 (5-way, +LGB_tuned_seed2) weights={dict(zip(names5, np.round(w5, 3)))} OOF RMSE={s5:.5f}")

base_models_r3 = base_models_r2 + [
    dict(model="LGB_tuned_seed2", rmse=round(score_tuned2, 5), time_s=round(t_seed, 1),
         note=f"seed bag of LGB_tuned with random_state={SEED+1000}, same Optuna params")
]
exp_id3 = experiment_log.log_experiment_v2(
    COMP,
    model="5-way blend: LGB+XGB+CatBoost (orig) + Optuna-tuned LGB + seed-bagged tuned LGB",
    metric="rmse", direction="minimize", score=s5,
    cv=dict(strategy=f"{N_SPLITS}fold_stratified_strength_decile", seed=SEED),
    features=FEATS, base_models=base_models_r3,
    ensemble=dict(weights=dict(zip(names5, [round(float(x), 3) for x in w5])),
                  method="oof_weight_search(dirichlet_random+coord_descent)"),
    submission=None,
    notes=(f"Round 3 self-improvement: seed-bag Round 2's Optuna-tuned LGB with a second "
           f"random_state ({SEED+1000}, same hyperparams) as a cheap 5th pool member "
           f"(validated recipe: seed bagging after tuning is the cheapest residual gain). "
           f"Solo OOF {score_tuned2:.5f}. Result: {s5:.5f} vs Round 2 {s4:.5f} vs "
           f"best-so-far 12.07347 ({'IMPROVED' if s5 < min(12.073474, s4) else 'no improvement'})."),
)
print(f"Logged experiment #{exp_id3}")

# persist best-of-round predictions for potential submission generation
np.savez(f"{COMP}/scripts/_round23_pool.npz",
         oof4=np.stack([oof_dict4[n] for n in names4], axis=1), w4=np.array(w4), s4=s4,
         oof5=np.stack([oof_dict5[n] for n in names5], axis=1), w5=np.array(w5), s5=s5,
         pred4=np.stack([pred_dict4[n] for n in names4], axis=1),
         pred5=np.stack([pred_dict5[n] for n in names5], axis=1),
         names4=np.array(names4), names5=np.array(names5))
print("Saved pool OOF/pred to _round23_pool.npz")
