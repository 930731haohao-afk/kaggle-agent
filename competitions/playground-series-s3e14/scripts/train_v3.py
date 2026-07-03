"""Self-improvement iteration 3 for playground-series-s3e14 (Phase B round 1).

Change (ONE thing vs iteration 2, per experience.md Optuna recipe from s3e7):
  Optuna-tune LGB (the strongest base model: 342.02154 in iter 2, vs XGB 342.21778,
  CAT 343.60196) using the fold-proxy recipe -- 50 trials scored on a single held-out
  fold (fold 0) with early stopping, NOT the full 5-fold CV (s3e7 lesson: full-5-fold
  Optuna objective blew the time budget). The winning config is then re-validated with
  the full 5-fold CV.

  XGB and CAT are kept EXACTLY as in iteration 2 (same params, same code) -- per
  experience.md, re-tuning the second base model with the same recipe hurt blend
  diversity in s3e7 even though its own OOF improved. Also per experience.md, Ridge
  stacking already lost to grid-simplex blend in exp #3 (344.04 vs 340.76) on this same
  competition, so this iteration does NOT re-try Ridge stacking -- goes straight to the
  grid-simplex blend.

Feature set: identical to iteration 2 (21 pruned features, native-cat CatBoost cols).
"""
import importlib.util
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py"))
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e14"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "yield", "id"
N_SPLITS, SEED = 5, 42
NOISE_FEATS = ["temp_avg", "log_clonesize", "rain_intensity",
               "MinOfLowerTRange", "AverageOfLowerTRange", "AverageOfUpperTRange"]
CAT_COLS = ["clonesize", "RainingDays", "AverageRainingDays", "honeybee", "bumbles", "andrena", "osmia"]

os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

Xtr_full = build_features(train)
Xte_full = build_features(test)
FEATS = [c for c in feature_columns(Xtr_full) if c not in NOISE_FEATS]
X = Xtr_full[FEATS].copy()
Xtest = Xte_full[FEATS].copy()
y = train[TARGET].to_numpy(np.float64)
print(f"n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")

Xn = X.to_numpy(np.float32)
Xtn = Xtest.to_numpy(np.float32)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(Xn))


def mae(o):
    return mean_absolute_error(y, o)


# ============================================================
# Step 1: Optuna fold-proxy tuning of LGB (fold 0 only, 50 trials)
# ============================================================
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

tr0, va0 = folds[0]


def objective(trial):
    params = dict(
        objective="regression_l1", metric="mae", n_estimators=3000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 127),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    m = lgb.LGBMRegressor(**params)
    m.fit(Xn[tr0], y[tr0], eval_set=[(Xn[va0], y[va0])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    pred = m.predict(Xn[va0])
    return mean_absolute_error(y[va0], pred)


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, timeout=280)
tune_time = time.time() - t0
print(f"\nOptuna fold-0 proxy: {len(study.trials)} trials in {tune_time:.0f}s, "
      f"best fold-0 MAE={study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_lgb_params = dict(
    objective="regression_l1", metric="mae", n_estimators=3000,
    subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1,
    **study.best_params,
)

# ============================================================
# Step 2: full 5-fold CV with tuned LGB params
# ============================================================


def run_lgb_tuned():
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtn))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**best_lgb_params)
        m.fit(Xn[tr], y[tr], eval_set=[(Xn[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(Xn[va]); pred += m.predict(Xtn) / N_SPLITS
        print(f"  [LGB-tuned] fold{f} MAE={mean_absolute_error(y[va], oof[va]):.4f}")
    return oof, pred


def run_xgb():
    """Unchanged from iteration 2."""
    import xgboost as xgb
    params = dict(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.02,
                  max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1,
                  eval_metric="mae", early_stopping_rounds=150)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtn))
    for f, (tr, va) in enumerate(folds):
        m = xgb.XGBRegressor(**params)
        m.fit(Xn[tr], y[tr], eval_set=[(Xn[va], y[va])], verbose=False)
        oof[va] = m.predict(Xn[va]); pred += m.predict(Xtn) / N_SPLITS
        print(f"  [XGB] fold{f} MAE={mean_absolute_error(y[va], oof[va]):.4f}")
    return oof, pred


def run_cat():
    """Unchanged from iteration 2: native categorical handling for low-cardinality env columns."""
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtn))
    Xc = X.copy()
    Xtc = Xtest.copy()
    for c in CAT_COLS:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype(str)
            Xtc[c] = Xtc[c].astype(str)
    cat_names = [c for c in CAT_COLS if c in Xc.columns]
    for f, (tr, va) in enumerate(folds):
        m = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", iterations=4000,
                              learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                              random_seed=SEED, thread_count=-1, verbose=False)
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=cat_names),
              eval_set=Pool(Xc.iloc[va], y[va], cat_features=cat_names),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict(Xc.iloc[va]); pred += m.predict(Xtc) / N_SPLITS
        print(f"  [CAT] fold{f} MAE={mean_absolute_error(y[va], oof[va]):.4f}")
    return oof, pred


results = {}
for name, fn in [("LGB", run_lgb_tuned), ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"\n=== {name} ===")
    oof, pred = fn()
    dt = time.time() - t0
    print(f"{name} OOF MAE = {mae(oof):.5f}  [{dt:.0f}s]")
    results[name] = dict(oof=oof, pred=pred, mae=mae(oof), time=dt)

names = list(results)
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
preds_stack = np.stack([results[n]["pred"] for n in names], axis=1)

# --- grid-searched simplex blend (Ridge stacking already lost in exp #3, skip re-testing it) ---
best_w, best_s = None, 1e9
for w0 in np.arange(0, 1.01, 0.05):
    for w1 in np.arange(0, 1.01 - w0, 0.05):
        w2 = 1 - w0 - w1
        if w2 < -1e-9:
            continue
        s = mae(oofs @ np.array([w0, w1, w2]))
        if s < best_s:
            best_s, best_w = s, (w0, w1, w2)
grid_oof = oofs @ np.array(best_w)
grid_pred = preds_stack @ np.array(best_w)
print(f"\nGrid simplex blend {dict(zip(names, np.round(best_w, 2)))} OOF MAE = {best_s:.5f}")

final_oof, final_pred = grid_oof, grid_pred
ensemble_info = {"method": "grid_simplex", "weights": {n: round(float(x), 3) for n, x in zip(names, best_w)},
                  "score": round(float(best_s), 5)}

# --- post-processing: snap to observed grid (won on both prior iterations) ---
train_sorted = np.sort(np.unique(y))


def snap_to_grid(preds, grid):
    idx = np.searchsorted(grid, preds)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    return np.where(np.abs(preds - left) <= np.abs(preds - right), left, right)


snap_oof = snap_to_grid(final_oof, train_sorted)
snap_mae = mae(snap_oof)
raw_mae = mae(final_oof)
print(f"Raw blend OOF MAE = {raw_mae:.5f}   snapped = {snap_mae:.5f}")
use_snap = snap_mae < raw_mae
best_pp_mae = snap_mae if use_snap else raw_mae
sub_pred = snap_to_grid(final_pred, train_sorted) if use_snap else final_pred
print(f"Final OOF MAE (iteration 3) = {best_pp_mae:.5f}   (iteration 2 best was 340.71180)")

# --- write submission ---
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({ID: test[ID], TARGET: sub_pred})
sub_path = f"{SUB}/sub_blend_v3_{best_pp_mae:.5f}_{stamp}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote {sub_path}  shape={sub.shape}")
print(sub[TARGET].describe())

# --- log experiment ---
base_models = [{"name": n, "score": round(float(results[n]["mae"]), 5)} for n in names]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB(Optuna fold-proxy tuned)+XGB+CAT(native-cat, unchanged) grid-simplex blend",
    metric="mae",
    direction="minimize",
    score=float(best_pp_mae),
    cv={"scheme": "5fold_kfold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=base_models,
    ensemble=ensemble_info,
    postprocess=[f"snap_to_grid={'used' if use_snap else 'not used'} "
                 f"(raw={raw_mae:.5f}, snapped={snap_mae:.5f})"],
    submission=os.path.basename(sub_path),
    notes=f"Phase B round 1 (change vs iter 2: Optuna-tune LGB only via fold-0 proxy, "
          f"{len(study.trials)} trials in {tune_time:.0f}s; best_params={study.best_params}; "
          f"XGB/CAT kept identical to iteration 2 per experience.md (re-tuning 2nd model hurt "
          f"blend diversity in s3e7); skipped re-testing Ridge stack, already lost in exp #3).",
)
print(f"Logged experiment #{exp_id} to experiments.json")
