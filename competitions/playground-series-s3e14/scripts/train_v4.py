"""Self-improvement iteration 4 for playground-series-s3e14 (Phase B round 2).

Reflexion on round 1 (train_v3.py, exp #4, 340.82694 vs best 340.71180): the Optuna
fold-0-proxy-tuned LGB improved as a SINGLE model (342.02154 -> 341.68775) but REPLACING
the original LGB with it made the blend worse (340.75961 -> 340.95316) -- the tuned config
(depth 5, lr 0.015, near-zero L1/L2) is less diverse w.r.t. XGB, so the ensemble lost more
from diversity than it gained from single-model strength. Same mechanism as s3e7 exp #5
(re-tuning the second model hurt the blend), observed here even for the strongest model.

Change (ONE thing vs iteration 2 / exp #3): keep the ORIGINAL LGB (unchanged from iter 2)
and ADD the tuned LGB as a FOURTH base model -- diversity preserved, strength added.
4-way simplex grid blend (step 0.05), then snap-to-grid. XGB/CAT identical to iteration 2.
Also saves OOF/test predictions per base model to data/oof_v4.npz for cheap future blending.
"""
import importlib.util
import os
import sys
import time
from datetime import datetime
from itertools import product

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

# Winning Optuna fold-0 proxy params from train_v3.py (exp #4)
TUNED_LGB_PARAMS = dict(
    learning_rate=0.015389816656608301, num_leaves=63, max_depth=5,
    min_child_samples=30, subsample=0.7899031923134097,
    colsample_bytree=0.6842166782718478,
    reg_alpha=0.012648426946439915, reg_lambda=0.00346212601571565,
)

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


def run_lgb(params):
    import lightgbm as lgb
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtn))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xn[tr], y[tr], eval_set=[(Xn[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(Xn[va]); pred += m.predict(Xtn) / N_SPLITS
    return oof, pred


def run_lgb_orig():
    """Unchanged from iteration 2."""
    params = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                  learning_rate=0.02, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
    return run_lgb(params)


def run_lgb_tuned():
    """Optuna fold-0 proxy winner from round 1 (exp #4)."""
    params = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                  subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1,
                  **TUNED_LGB_PARAMS)
    return run_lgb(params)


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
    return oof, pred


def run_cat():
    """Unchanged from iteration 2: native categorical handling."""
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
    return oof, pred


results = {}
for name, fn in [("LGB", run_lgb_orig), ("LGB_TUNED", run_lgb_tuned),
                 ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"=== {name} ===", flush=True)
    oof, pred = fn()
    dt = time.time() - t0
    print(f"{name} OOF MAE = {mae(oof):.5f}  [{dt:.0f}s]", flush=True)
    results[name] = dict(oof=oof, pred=pred, mae=mae(oof), time=dt)

names = list(results)
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
preds_stack = np.stack([results[n]["pred"] for n in names], axis=1)

# save OOF/test preds for cheap future blend experiments
np.savez(f"{DATA}/oof_v4.npz", names=np.array(names), oofs=oofs, preds=preds_stack, y=y)

# --- 4-way simplex grid blend, step 0.05 ---
step = 0.05
grid_vals = np.arange(0, 1.0 + 1e-9, step)
best_w, best_s = None, 1e9
for w0, w1, w2 in product(grid_vals, repeat=3):
    w3 = 1.0 - w0 - w1 - w2
    if w3 < -1e-9:
        continue
    w = np.array([w0, w1, w2, max(w3, 0.0)])
    s = mae(oofs @ w)
    if s < best_s:
        best_s, best_w = s, w
grid_oof = oofs @ best_w
grid_pred = preds_stack @ best_w
print(f"\n4-way grid simplex blend {dict(zip(names, np.round(best_w, 2)))} OOF MAE = {best_s:.5f}")

final_oof, final_pred = grid_oof, grid_pred
ensemble_info = {"method": "grid_simplex_4way", "weights": {n: round(float(x), 3) for n, x in zip(names, best_w)},
                  "score": round(float(best_s), 5)}

# --- post-processing: snap to observed grid ---
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
print(f"Final OOF MAE (iteration 4) = {best_pp_mae:.5f}   (best so far 340.71180)")

# --- write submission ---
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({ID: test[ID], TARGET: sub_pred})
sub_path = f"{SUB}/sub_blend_v4_{best_pp_mae:.5f}_{stamp}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote {sub_path}  shape={sub.shape}")

# --- log experiment ---
base_models = [{"name": n, "score": round(float(results[n]["mae"]), 5)} for n in names]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB+LGB(tuned)+XGB+CAT(native-cat) 4-way grid-simplex blend",
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
    notes="Phase B round 2 (reflexion on exp #4: tuned LGB improved solo 342.02->341.69 but "
          "REPLACING original LGB hurt blend 340.76->340.95 via lost diversity). Change: keep "
          "original LGB AND add tuned LGB as a 4th base model; 4-way simplex grid (step 0.05); "
          "XGB/CAT identical to exp #3. OOF/test preds saved to data/oof_v4.npz.",
)
print(f"Logged experiment #{exp_id} to experiments.json")
