"""Phase B iteration, round 1: Optuna LGB tuning with full 5-fold CV AUC as the
direct objective (data is tiny — 1677 rows — so full CV per trial is cheap; no
need for a fold-0 proxy per the recipe in knowledge/experience.md).

Baseline to beat: exp #3 LGB OOF AUC 0.832925 (num_leaves=7, reg_alpha=1.0, reg_lambda=2.0).
CV MUST stay identical: StratifiedKFold(5, shuffle=True, random_state=42) on Attrition.
"""
import importlib.util
import itertools
import os
import sys
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns, fit_encoders  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e3"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "Attrition", "id"
N_SPLITS, SEED = 5, 42

os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

enc = fit_encoders(train)
Xtr_full = build_features(train, enc)
Xte_full = build_features(test, enc)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.int64)
Xtest = Xte_full[FEATS].to_numpy(np.float32)
print(f"n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))


def run_lgb_cv(params):
    import lightgbm as lgb
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def objective(trial):
    params = dict(
        objective="binary", metric="auc", n_estimators=2000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 3, 31),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    oof = run_lgb_cv(params)
    return roc_auc_score(y, oof)


t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, timeout=600)
elapsed = time.time() - t0
print(f"\nOptuna done: {len(study.trials)} trials, {elapsed:.1f}s")
print(f"Best value (full-CV AUC): {study.best_value:.6f}")
print(f"Best params: {study.best_params}")

best_params = dict(
    objective="binary", metric="auc", n_estimators=2000,
    random_state=SEED, n_jobs=-1, verbose=-1,
    **study.best_params,
)

# Final full retrain with best params to get OOF + test predictions
import lightgbm as lgb
oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
for f, (tr, va) in enumerate(folds):
    m = lgb.LGBMClassifier(**best_params)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict_proba(X[va])[:, 1]
    pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    print(f"  [LGB_tuned] fold{f} AUC={roc_auc_score(y[va], oof[va]):.5f} best_iter={m.best_iteration_}")

final_score = roc_auc_score(y, oof)
print(f"\nLGB_tuned OOF AUC={final_score:.6f}  (baseline LGB exp#3: 0.832925)")

# Save OOF/pred for reuse in later rounds (pool building, seed bagging)
np.savez(f"{COMP}/scripts/lgb_tuned_oof_pred.npz", oof=oof, pred=pred, params=str(best_params))

# --- Log experiment ---
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB Optuna-tuned (full 5-fold CV AUC objective, solo)",
    metric="roc_auc",
    direction="maximize",
    score=final_score,
    cv={"strategy": "StratifiedKFold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=[{"name": "LGB_tuned", "oof_auc": round(float(final_score), 6)}],
    ensemble={"method": "single_model", "weights": {"LGB_tuned": 1.0}},
    submission=None,
    notes=(f"Optuna TPE, 50 trials (timeout 600s, actual {elapsed:.1f}s), objective = "
           f"direct full 5-fold StratifiedKFold OOF ROC-AUC (data tiny enough that full CV "
           f"per trial is cheap, no fold-0 proxy needed per experience.md recipe). "
           f"Best params: {study.best_params}. Solo LGB OOF {final_score:.6f} vs exp #3 "
           f"baseline LGB 0.832925 ({'improved' if final_score > 0.832925 else 'not improved'})."),
)
print(f"Logged experiment #{exp_id}")
