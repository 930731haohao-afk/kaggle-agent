"""
Iteration round 2 — seed bagging on top of the round-1 result.

Per knowledge/experience.md (s3e14 evidence): after tuning, adding a seed-bagged
copy of the SAME winning hyperparameters (different random_state only) is the
cheapest residual gain -- a single extra 5-fold train, no new tuning cost.

Loads round 1's saved OOF/test arrays (LGB, XGB, CAT, LGB_TUNED) + best tuned
LGB params, trains a 5th member (LGB_TUNED with random_state=2024 instead of
42) on the SAME KFold splits (seed=42 fold assignment unchanged -- only the
model's internal random_state differs), then re-runs the weight-search blend
over all 5 members.
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

COMPETITION_DIR = "competitions/playground-series-s3e1"
TARGET_COL = "MedHouseVal"
ID_COL = "id"
N_SPLITS = 5
SEED = 42
SEED_BAG = 2024
PREV_BEST = 0.55798  # round 1 result

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

# --- Load round 1 artifacts ---
r1 = np.load(os.path.join(data_dir, "round1_oof_test.npz"))
oof = {"LGB": r1["oof_LGB"], "XGB": r1["oof_XGB"], "CAT": r1["oof_CAT"], "LGB_TUNED": r1["oof_LGB_TUNED"]}
test_preds = {"LGB": r1["test_LGB"], "XGB": r1["test_XGB"], "CAT": r1["test_CAT"], "LGB_TUNED": r1["test_LGB_TUNED"]}
y_check = r1["y"]
assert np.allclose(y, y_check), "target mismatch vs round1 artifacts"

with open(os.path.join(data_dir, "round1_best_lgb_params.json")) as f:
    tuned_params = json.load(f)

seed_params = dict(tuned_params)
seed_params["random_state"] = SEED_BAG

oof_seed = np.zeros(len(train))
test_pred_seed = np.zeros(len(test))
t0 = time.time()
for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    m = lgb.LGBMRegressor(**seed_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_seed[va_idx] = m.predict(X_va)
    test_pred_seed += m.predict(X_test) / N_SPLITS
train_time = time.time() - t0

oof["LGB_TUNED_SEED2024"] = oof_seed
test_preds["LGB_TUNED_SEED2024"] = test_pred_seed
solo_score = mean_squared_error(y, oof_seed) ** 0.5
print(f"Seed-bagged tuned LGB (seed={SEED_BAG}) OOF RMSE: {solo_score:.5f} in {train_time:.1f}s")

oof_scores = {m: round(float(mean_squared_error(y, oof[m]) ** 0.5), 5) for m in oof}
print("All OOF scores:", oof_scores)

# --- 5-way weight search (grid, step 0.05, simplex, random restarts via coordinate grid) ---
model_names = ["LGB", "XGB", "CAT", "LGB_TUNED", "LGB_TUNED_SEED2024"]
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
print(f"Best 5-way blend OOF RMSE: {best_score:.5f} (vs round1 best {PREV_BEST})")

blend_test = (best_w["LGB"] * test_preds["LGB"] + best_w["XGB"] * test_preds["XGB"]
              + best_w["CAT"] * test_preds["CAT"] + best_w["LGB_TUNED"] * test_preds["LGB_TUNED"]
              + best_w["LGB_TUNED_SEED2024"] * test_preds["LGB_TUNED_SEED2024"])
blend_test_clipped = np.clip(blend_test, train[TARGET_COL].min(), train[TARGET_COL].max())

improved = best_score < PREV_BEST
if improved:
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sub_name = f"sub_5way_seedbag_blend_{best_score:.5f}_{timestamp}.csv"
    sub_path = os.path.join(COMPETITION_DIR, "submissions", sub_name)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: blend_test_clipped})
    submission.to_csv(sub_path, index=False)
    print(f"\nNew best -> submission saved: {sub_path}")
else:
    sub_name = None
    print("\nNo improvement over round1 best -- not saving a new submission.")

# save artifacts for potential round 3
np.savez(os.path.join(data_dir, "round2_oof_test.npz"),
          **{f"oof_{k}": v for k, v in oof.items()}, **{f"test_{k}": v for k, v in test_preds.items()}, y=y)

base_models_log = [{"name": k, "score": oof_scores[k]} for k in model_names]
base_models_log[3]["params"] = tuned_params
base_models_log[4]["params"] = seed_params

exp_id = experiment_log.log_experiment_v2(
    COMPETITION_DIR,
    model="5-way blend: LGB+XGB+CAT+Optuna-tuned-LGB+seed-bagged-tuned-LGB",
    metric="rmse",
    direction="minimize",
    score=round(float(best_score), 6),
    cv={"scheme": "KFold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
    features=feature_cols,
    base_models=base_models_log,
    ensemble={"method": "oof_weight_grid_search_5way", "weights": best_w, "score": round(float(best_score), 6)},
    postprocess=["clip_to_train_target_range"],
    submission=sub_name,
    notes=(f"Iteration round 2: seed bagging (s3e14 recipe) -- added a 5th member: same tuned-LGB "
           f"hyperparams from round 1 but random_state={SEED_BAG} instead of {SEED}. Solo OOF "
           f"{solo_score:.5f}, train time {train_time:.1f}s (cheapest possible residual-gain move, "
           f"no new tuning). Round1 best was {PREV_BEST}; this round "
           f"{'IMPROVED' if improved else 'did NOT improve'} to {best_score:.6f}."),
)
print(f"\nLogged experiment_id={exp_id}")
