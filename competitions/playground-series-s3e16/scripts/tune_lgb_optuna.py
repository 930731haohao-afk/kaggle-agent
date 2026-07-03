"""Optuna fold-0 proxy tuning for LightGBM (L1 objective) — s3e16 Crab Age.

Recipe (validated in knowledge/experience.md across s3e1/s3e7/s3e11/s3e14):
Optuna on a single fold (fold-0) as a fast proxy for full 5-fold CV, then
train the winning config on all 5 folds and ADD it to the model pool
(never replace the original member — diversity matters more than the
single best score).
"""
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

COMP = "competitions/playground-series-s3e16"
DATA = f"{COMP}/data"
TARGET, ID = "Age", "id"
N_SPLITS, SEED = 5, 42

train = pd.read_csv(f"{DATA}/train.csv")
h_med = train.loc[train["Height"] > 0, "Height"].median()
Xtr_full = build_features(train, height_median=h_med)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.float64)

ybin = np.where(y >= 20, 20, y).astype(int)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, ybin))
tr0, va0 = folds[0]  # fold-0 proxy


def objective(trial):
    import lightgbm as lgb
    params = dict(
        objective="regression_l1", metric="mae",
        n_estimators=3000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    m = lgb.LGBMRegressor(**params)
    m.fit(X[tr0], y[tr0], eval_set=[(X[va0], y[va0])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = m.predict(X[va0])
    return mean_absolute_error(y[va0], pred)


if __name__ == "__main__":
    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=50, timeout=300)  # timeout guard: 5 min max
    dt = time.time() - t0
    print(f"n_trials={len(study.trials)}  best_fold0_mae={study.best_value:.5f}  time={dt:.0f}s")
    print("best_params:", study.best_params)

    out = dict(best_params=study.best_params, best_fold0_mae=study.best_value,
               n_trials=len(study.trials), time_s=round(dt, 1))
    with open(f"{COMP}/scripts/lgb_optuna_best.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {COMP}/scripts/lgb_optuna_best.json")
