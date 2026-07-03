"""Self-improvement iteration 2 for playground-series-s3e14.

Reflexion on iteration 1 (train.py, OOF MAE 340.95856 after snap-to-grid post-processing,
only a modest ~0.45 improvement over the generic baseline 341.40782): a quick LGB feature-importance
probe showed several engineered features (temp_avg, log_clonesize, rain_intensity,
MinOfLowerTRange/AverageOfLowerTRange/AverageOfUpperTRange) carry near-zero importance (noise), while
the low-cardinality env columns (clonesize, RainingDays, AverageRainingDays, honeybee/bumbles/andrena/
osmia -- 6 to 16 discrete levels each) are numeric but might be better exploited by CatBoost's native
categorical handling than as ordinary floats.

This iteration tests two changes vs iteration 1, one at a time is not fully isolated (time budget),
but both changes are logged distinctly so the delta is attributable:
  (a) drop near-zero-importance noise features -> cleaner, faster model.
  (b) CatBoost treats low-cardinality env columns as categorical (native cat_features) instead of float.
  (c) Ridge (unconstrained-ish) stacking meta-model on OOF predictions, compared against the
      grid-searched simplex blend from iteration 1; keep whichever is better.
"""
import importlib.util
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
print(f"n_features={len(FEATS)} (dropped {len(NOISE_FEATS)} near-zero-importance features)  "
      f"train={X.shape}  test={Xtest.shape}")

Xn = X.to_numpy(np.float32)
Xtn = Xtest.to_numpy(np.float32)
cat_idx = [FEATS.index(c) for c in CAT_COLS if c in FEATS]

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(Xn))


def run_lgb():
    import lightgbm as lgb
    params = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                  learning_rate=0.02, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtn))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xn[tr], y[tr], eval_set=[(Xn[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(Xn[va]); pred += m.predict(Xtn) / N_SPLITS
        print(f"  [LGB] fold{f} MAE={mean_absolute_error(y[va], oof[va]):.4f}")
    return oof, pred


def run_xgb():
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
    """CatBoost with native categorical handling for low-cardinality env columns."""
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


def mae(o):
    return mean_absolute_error(y, o)


results = {}
for name, fn in [("LGB", run_lgb), ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"\n=== {name} ===")
    oof, pred = fn()
    dt = time.time() - t0
    print(f"{name} OOF MAE = {mae(oof):.5f}  [{dt:.0f}s]")
    results[name] = dict(oof=oof, pred=pred, mae=mae(oof), time=dt)

names = list(results)
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
preds_stack = np.stack([results[n]["pred"] for n in names], axis=1)

# --- (1) grid-searched simplex blend, same method as iteration 1 ---
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

# --- (2) Ridge stacking meta-model on OOF predictions (5-fold nested to avoid leakage) ---
ridge_oof = np.zeros(len(y))
ridge_pred = np.zeros(len(Xtn))
for f, (tr, va) in enumerate(folds):
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(oofs[tr], y[tr])
    ridge_oof[va] = meta.predict(oofs[va])
meta_full = Ridge(alpha=1.0, positive=True).fit(oofs, y)
ridge_pred = preds_stack @ meta_full.coef_ + meta_full.intercept_
ridge_mae = mae(ridge_oof)
print(f"Ridge stacking (nested, non-negative coefs {np.round(meta_full.coef_, 3)}) OOF MAE = {ridge_mae:.5f}")

if ridge_mae < best_s:
    final_oof, final_pred, blend_method = ridge_oof, ridge_pred, "ridge_stack"
    ensemble_info = {"method": "ridge_stack", "coef": {n: round(float(c), 4) for n, c in zip(names, meta_full.coef_)},
                      "intercept": round(float(meta_full.intercept_), 4), "score": round(float(ridge_mae), 5)}
else:
    final_oof, final_pred, blend_method = grid_oof, grid_pred, "grid_simplex"
    ensemble_info = {"method": "grid_simplex", "weights": {n: round(float(x), 3) for n, x in zip(names, best_w)},
                      "score": round(float(best_s), 5)}
print(f"Selected blend method: {blend_method}")

# --- post-processing: snap to observed grid (won on iteration 1) ---
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
print(f"Final OOF MAE (iteration 2) = {best_pp_mae:.5f}   (iteration 1 was 340.95856)")

# --- write submission ---
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({ID: test[ID], TARGET: sub_pred})
sub_path = f"{SUB}/sub_blend_v2_{best_pp_mae:.5f}_{stamp}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote {sub_path}  shape={sub.shape}")
print(sub[TARGET].describe())

# --- log experiment ---
base_models = [{"name": n, "score": round(float(results[n]["mae"]), 5)} for n in names]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB+XGB+CAT(native-categorical) blend, pruned features, best-of(grid/ridge-stack)",
    metric="mae",
    direction="minimize",
    score=float(best_pp_mae),
    cv={"scheme": "5fold_kfold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=base_models,
    ensemble=ensemble_info,
    postprocess=[f"snap_to_grid={'used' if use_snap else 'not used'} "
                 f"(raw={raw_mae:.5f}, snapped={snap_mae:.5f})",
                 f"grid_simplex={best_s:.5f} vs ridge_stack={ridge_mae:.5f} -> chose {blend_method}"],
    submission=os.path.basename(sub_path),
    notes="Iteration 2 (reflexion on exp #2): dropped 6 near-zero-importance engineered/raw features "
          "(temp_avg, log_clonesize, rain_intensity, MinOfLowerTRange, AverageOfLowerTRange, "
          "AverageOfUpperTRange); CatBoost now treats low-cardinality env cols as native categoricals; "
          "added Ridge (non-negative) stacking meta-model as an alternative to the grid-searched simplex "
          "blend, keeping whichever OOF MAE is lower.",
)
print(f"Logged experiment #{exp_id} to experiments.json")
