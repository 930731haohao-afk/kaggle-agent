"""Self-improvement iteration 6 for playground-series-s3e14 (Phase B round 4, final).

Change (ONE thing vs round 2 / exp #5): add a seed-diversified LGB (identical original
params, random_state 2024 instead of 42; folds unchanged, still KFold seed 42) as a FIFTH
base model. Rationale: round 2 showed the blend gains from model diversity even when the
added model is individually similar (LGB_TUNED); seed bagging is the cheapest diversity
source left. XGB/CAT/LGB/LGB_TUNED predictions loaded unchanged from data/oof_v4.npz.
5-way simplex grid (step 0.05) + snap-to-grid.
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
BAG_SEED = 2024
NOISE_FEATS = ["temp_avg", "log_clonesize", "rain_intensity",
               "MinOfLowerTRange", "AverageOfLowerTRange", "AverageOfUpperTRange"]

os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
Xtr_full = build_features(train)
Xte_full = build_features(test)
FEATS = [c for c in feature_columns(Xtr_full) if c not in NOISE_FEATS]
Xn = Xtr_full[FEATS].to_numpy(np.float32)
Xtn = Xte_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.float64)

npz = np.load(f"{DATA}/oof_v4.npz", allow_pickle=True)
names = [str(n) for n in npz["names"]]
oofs = npz["oofs"]
preds_stack = npz["preds"]
assert np.allclose(npz["y"], y)
print(f"Loaded OOF matrix {oofs.shape}, models={names}")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(Xn))


def mae(o):
    return mean_absolute_error(y, o)


# --- train the 5th base model: original-params LGB with seed 2024 ---
import lightgbm as lgb
params = dict(objective="regression_l1", metric="mae", n_estimators=3000,
              learning_rate=0.02, num_leaves=63, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
              reg_alpha=1.0, reg_lambda=2.0, random_state=BAG_SEED, n_jobs=-1, verbose=-1)
t0 = time.time()
oof5 = np.zeros(len(y)); pred5 = np.zeros(len(Xtn))
for f, (tr, va) in enumerate(folds):
    m = lgb.LGBMRegressor(**params)
    m.fit(Xn[tr], y[tr], eval_set=[(Xn[va], y[va])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    oof5[va] = m.predict(Xn[va]); pred5 += m.predict(Xtn) / N_SPLITS
print(f"LGB_SEED{BAG_SEED} OOF MAE = {mae(oof5):.5f}  [{time.time()-t0:.0f}s]")

names5 = names + [f"LGB_SEED{BAG_SEED}"]
oofs5 = np.column_stack([oofs, oof5])
preds5 = np.column_stack([preds_stack, pred5])

# --- 5-way simplex grid blend, step 0.05 ---
step = 0.05
grid_vals = np.arange(0, 1.0 + 1e-9, step)
best_w, best_s = None, 1e9
for w0, w1, w2, w3 in product(grid_vals, repeat=4):
    w4 = 1.0 - w0 - w1 - w2 - w3
    if w4 < -1e-9:
        continue
    w = np.array([w0, w1, w2, w3, max(w4, 0.0)])
    s = mae(oofs5 @ w)
    if s < best_s:
        best_s, best_w = s, w
blend_oof = oofs5 @ best_w
blend_pred = preds5 @ best_w
print(f"5-way blend {dict(zip(names5, np.round(best_w, 2)))} OOF MAE = {best_s:.5f} "
      f"(round 2 4-way was 340.69924)")

# --- snap to observed grid ---
train_sorted = np.sort(np.unique(y))


def snap_to_grid(preds, grid):
    idx = np.searchsorted(grid, preds)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    return np.where(np.abs(preds - left) <= np.abs(preds - right), left, right)


snap_oof = snap_to_grid(blend_oof, train_sorted)
snap_mae = mae(snap_oof)
print(f"Raw blend OOF MAE = {best_s:.5f}   snapped = {snap_mae:.5f}")
use_snap = snap_mae < best_s
best_pp_mae = snap_mae if use_snap else best_s
sub_pred = snap_to_grid(blend_pred, train_sorted) if use_snap else blend_pred
print(f"Final OOF MAE (iteration 6) = {best_pp_mae:.5f}   (best so far 340.62702)")

improved = best_pp_mae < 340.62702 - 1e-6

sub_path = None
if improved:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame({ID: test[ID], TARGET: sub_pred})
    sub_path = f"{SUB}/sub_blend_v6_{best_pp_mae:.5f}_{stamp}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path}  shape={sub.shape}")
else:
    print("No improvement over round 2 (340.62702) -- no new submission written.")

base_models = ([{"name": n, "score": round(float(mae(oofs5[:, i])), 5)} for i, n in enumerate(names5)])
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB+LGB(tuned)+XGB+CAT+LGB(seed2024) 5-way grid-simplex blend",
    metric="mae",
    direction="minimize",
    score=float(best_pp_mae),
    cv={"scheme": "5fold_kfold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=base_models,
    ensemble={"method": "grid_simplex_5way",
              "weights": {n: round(float(x), 3) for n, x in zip(names5, best_w)},
              "score": round(float(best_s), 5)},
    postprocess=[f"snap_to_grid={'used' if use_snap else 'not used'} "
                 f"(raw={best_s:.5f}, snapped={snap_mae:.5f})"],
    submission=os.path.basename(sub_path) if sub_path else None,
    notes="Phase B round 4 (change vs exp #5: add seed-diversified LGB, identical params but "
          "random_state 2024, as 5th base model; other 4 model predictions loaded unchanged "
          "from oof_v4.npz; folds unchanged KFold seed 42).",
)
print(f"Logged experiment #{exp_id} to experiments.json")
