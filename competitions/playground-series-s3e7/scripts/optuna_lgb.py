"""Round 1 iteration: Optuna-tuned LightGBM (replacing hand-set defaults),
keeping the same trimmed 25-feature set, XGB/CAT unchanged, and re-running
the weight-searched blend. Same CV scheme (5-fold StratifiedKFold, seed=42)
as exp #3 so scores are directly comparable.

Why LGB first: LGB was already the strongest single model (0.898824) and
the current params (num_leaves=63, lr=0.03, reg_alpha=0.5, reg_lambda=1.0)
were hand-picked, never tuned. Experience library note: small samples reward
regularization search over hand-set capacity.
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

COMP = "competitions/playground-series-s3e7"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "booking_status", "id"
N_SPLITS, SEED = 5, 42

os.makedirs(SUB, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

Xtr_full = build_features(train)
Xte_full = build_features(test)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.int32)
Xtest = Xte_full[FEATS].to_numpy(np.float32)
print(f"n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))


def objective(trial):
    # Tune on fold 0 only (8.4k validation rows -- stable enough for AUC
    # ranking) so 50 trials fit the wall-clock budget; the winning config is
    # re-scored on the full 5-fold OOF below. A first attempt that ran full
    # 5-fold per trial blew a 25-min timeout with zero results.
    import lightgbm as lgb
    params = dict(
        objective="binary", metric="auc",
        n_estimators=3000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    tr, va = folds[0]
    m = lgb.LGBMClassifier(**params)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    return roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])


t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, show_progress_bar=False)
tuning_time = time.time() - t0
print(f"\nOptuna done in {tuning_time:.1f}s, best fold0 AUC={study.best_value:.6f}")
print(f"Best params: {study.best_params}")

best_params = dict(
    objective="binary", metric="auc", n_estimators=3000,
    random_state=SEED, n_jobs=-1, verbose=-1,
    subsample_freq=1,
    **study.best_params,
)

# --- refit tuned LGB with test predictions ---
import lightgbm as lgb
oof_lgb = np.zeros(len(y)); pred_lgb = np.zeros(len(Xtest))
t0 = time.time()
for f, (tr, va) in enumerate(folds):
    m = lgb.LGBMClassifier(**best_params)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_lgb[va] = m.predict_proba(X[va])[:, 1]
    pred_lgb += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    print(f"  [LGB-tuned] fold{f} AUC={roc_auc_score(y[va], oof_lgb[va]):.5f}")
lgb_time = time.time() - t0
lgb_auc = roc_auc_score(y, oof_lgb)
print(f"Tuned LGB OOF AUC={lgb_auc:.6f}  time={lgb_time:.1f}s")


def run_xgb():
    import xgboost as xgb
    params = dict(objective="binary:logistic", n_estimators=3000, learning_rate=0.03,
                  max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                  eval_metric="auc", early_stopping_rounds=150)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred, params


def run_cat():
    from catboost import CatBoostClassifier, Pool
    params = dict(loss_function="Logloss", eval_metric="AUC", iterations=4000,
                  learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                  random_seed=SEED, thread_count=-1, verbose=False)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = CatBoostClassifier(**params)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred, params


t0 = time.time()
oof_xgb, pred_xgb, xgb_params = run_xgb()
xgb_time = time.time() - t0
xgb_auc = roc_auc_score(y, oof_xgb)
print(f"XGB OOF AUC={xgb_auc:.6f}  time={xgb_time:.1f}s")

t0 = time.time()
oof_cat, pred_cat, cat_params = run_cat()
cat_time = time.time() - t0
cat_auc = roc_auc_score(y, oof_cat)
print(f"CAT OOF AUC={cat_auc:.6f}  time={cat_time:.1f}s")

results = {
    "LGB": dict(oof=oof_lgb, pred=pred_lgb, params=best_params, auc=lgb_auc, time_s=round(lgb_time, 1)),
    "XGB": dict(oof=oof_xgb, pred=pred_xgb, params=xgb_params, auc=xgb_auc, time_s=round(xgb_time, 1)),
    "CAT": dict(oof=oof_cat, pred=pred_cat, params=cat_params, auc=cat_auc, time_s=round(cat_time, 1)),
}

names = list(results.keys())
best_w, best_auc = None, -1
grid = [round(w * 0.05, 2) for w in range(21)]
for w in product(grid, repeat=len(names)):
    if abs(sum(w) - 1.0) > 1e-9:
        continue
    blend_oof = sum(wi * results[n]["oof"] for wi, n in zip(w, names))
    a = roc_auc_score(y, blend_oof)
    if a > best_auc:
        best_auc, best_w = a, w

blend_weights = {n: w for n, w in zip(names, best_w)}
blend_oof = sum(blend_weights[n] * results[n]["oof"] for n in names)
blend_pred = sum(blend_weights[n] * results[n]["pred"] for n in names)
blend_auc = roc_auc_score(y, blend_oof)
print(f"\n=== BLEND (tuned LGB) ===")
print(f"weights={blend_weights}  OOF AUC={blend_auc:.6f}")

PREV_BEST = 0.899395
print(f"\nPrev best (exp #3): {PREV_BEST}")
print(f"This blend delta vs prev best: {blend_auc - PREV_BEST:+.6f}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_name = f"sub_blend_{blend_auc:.5f}_{ts}.csv"
sample_sub = pd.read_csv(f"{DATA}/sample_submission.csv")
submission = pd.DataFrame({ID: test[ID], TARGET: blend_pred})
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == list(sample_sub.columns)
assert submission.isnull().sum().sum() == 0

if blend_auc > PREV_BEST:
    submission.to_csv(f"{SUB}/{sub_name}", index=False)
    print(f"\nNEW BEST — saved submission: {SUB}/{sub_name}")
else:
    print(f"\nNo improvement — submission NOT saved.")

base_models = [
    {"model": n, "params": {k: str(v) for k, v in results[n]["params"].items()},
     "oof_auc": round(float(results[n]["auc"]), 6), "time_s": results[n]["time_s"]}
    for n in names
]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="LGB(Optuna-tuned,50trials)+XGB+CAT weight-searched blend",
    metric="roc_auc",
    direction="maximize",
    score=round(float(blend_auc), 6),
    cv={"scheme": "5fold", "n_splits": N_SPLITS, "seed": SEED, "strategy": "StratifiedKFold(booking_status)"},
    features=FEATS,
    base_models=base_models,
    ensemble={"weights": blend_weights, "score": round(float(blend_auc), 6)},
    postprocess=None,
    submission=sub_name if blend_auc > PREV_BEST else None,
    notes=(f"Round1: Optuna-tuned LGB (50 trials, TPE, fold-0 proxy objective, "
           f"same folds/features as exp #3) "
           f"replacing hand-set LGB params; XGB/CAT unchanged from exp #3. "
           f"Tuning time {tuning_time:.1f}s. Tuned LGB solo OOF={lgb_auc:.6f} vs "
           f"hand-set exp#3 LGB solo 0.898824. Blend delta vs prev best {PREV_BEST}: "
           f"{blend_auc - PREV_BEST:+.6f}."),
)
print(f"\nLogged experiment_id={exp_id} to experiments.json")
