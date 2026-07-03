"""Self-improvement iteration 5 for playground-series-s3e14 (Phase B round 3).

Change (ONE thing vs round 2 / exp #5): add nested out-of-fold ISOTONIC calibration of
the 4-way blend output before snap-to-grid. Rationale: GBDT blends compress predictions
toward the mean at the extremes; isotonic regression (monotone, nonparametric) can undo
systematic compression. Ridge stacking already lost on this competition (exp #3,
344.04 vs 340.76), so per the iteration protocol we try the isotonic alternative instead.

Base predictions are loaded from data/oof_v4.npz (saved by train_v4.py) -- models and
blend weights search identical to round 2; no retraining. Same folds (KFold seed 42) are
regenerated for the nested calibration to avoid leakage.
"""
import importlib.util
import os
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

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

npz = np.load(f"{DATA}/oof_v4.npz", allow_pickle=True)
names = [str(n) for n in npz["names"]]
oofs = npz["oofs"]          # (n_train, 4)
preds_stack = npz["preds"]  # (n_test, 4)
y = npz["y"]
test = pd.read_csv(f"{DATA}/test.csv")
print(f"Loaded OOF matrix {oofs.shape}, test preds {preds_stack.shape}, models={names}")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(oofs))


def mae(o):
    return mean_absolute_error(y, o)


# --- reproduce round-2 4-way simplex blend (identical search) ---
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
blend_oof = oofs @ best_w
blend_pred = preds_stack @ best_w
print(f"4-way blend {dict(zip(names, np.round(best_w, 2)))} OOF MAE = {best_s:.5f} "
      f"(round 2 got 340.69924 -- must match)")

# --- NEW: nested out-of-fold isotonic calibration of the blend output ---
iso_oof = np.zeros(len(y))
for f, (tr, va) in enumerate(folds):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(blend_oof[tr], y[tr])
    iso_oof[va] = iso.predict(blend_oof[va])
iso_full = IsotonicRegression(out_of_bounds="clip").fit(blend_oof, y)
iso_pred = iso_full.predict(blend_pred)
iso_mae = mae(iso_oof)
print(f"Isotonic-calibrated blend OOF MAE = {iso_mae:.5f}  (raw blend {best_s:.5f})")

use_iso = iso_mae < best_s
cal_oof = iso_oof if use_iso else blend_oof
cal_pred = iso_pred if use_iso else blend_pred
cal_mae = iso_mae if use_iso else best_s
print(f"Calibration {'KEPT' if use_iso else 'REJECTED'}")

# --- post-processing: snap to observed grid ---
train_sorted = np.sort(np.unique(y))


def snap_to_grid(preds, grid):
    idx = np.searchsorted(grid, preds)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    return np.where(np.abs(preds - left) <= np.abs(preds - right), left, right)


snap_oof = snap_to_grid(cal_oof, train_sorted)
snap_mae = mae(snap_oof)
print(f"Calibrated OOF MAE = {cal_mae:.5f}   snapped = {snap_mae:.5f}")
use_snap = snap_mae < cal_mae
best_pp_mae = snap_mae if use_snap else cal_mae
sub_pred = snap_to_grid(cal_pred, train_sorted) if use_snap else cal_pred
print(f"Final OOF MAE (iteration 5) = {best_pp_mae:.5f}   (best so far 340.62702)")

improved = best_pp_mae < 340.62702

# --- write submission only if improved ---
sub_path = None
if improved:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame({ID: test[ID], TARGET: sub_pred})
    sub_path = f"{SUB}/sub_blend_v5_{best_pp_mae:.5f}_{stamp}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path}  shape={sub.shape}")
else:
    print("No improvement over round 2 -- no new submission written.")

# --- log experiment ---
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="4-way blend + nested OOF isotonic calibration (from saved oof_v4.npz)",
    metric="mae",
    direction="minimize",
    score=float(best_pp_mae),
    cv={"scheme": "5fold_kfold", "n_splits": N_SPLITS, "seed": SEED},
    ensemble={"method": "grid_simplex_4way+isotonic",
              "weights": {n: round(float(x), 3) for n, x in zip(names, best_w)},
              "blend_score": round(float(best_s), 5),
              "isotonic_score": round(float(iso_mae), 5),
              "isotonic_kept": bool(use_iso)},
    postprocess=[f"isotonic={'kept' if use_iso else 'rejected'} (raw={best_s:.5f}, iso={iso_mae:.5f})",
                 f"snap_to_grid={'used' if use_snap else 'not used'} "
                 f"(pre={cal_mae:.5f}, snapped={snap_mae:.5f})"],
    submission=os.path.basename(sub_path) if sub_path else None,
    notes="Phase B round 3 (change vs exp #5: nested OOF isotonic calibration of blend output "
          "before snap; base models and weight search identical to exp #5, loaded from oof_v4.npz; "
          "isotonic tried as the stacking alternative after Ridge lost in exp #3).",
)
print(f"Logged experiment #{exp_id} to experiments.json")
