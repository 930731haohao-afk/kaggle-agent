"""Round 2 iteration: Optuna-tuned XGBoost (fold-0 proxy objective, 50 trials),
keeping Round 1's Optuna-tuned LGB params fixed and CAT unchanged, then re-run
the weight-searched blend. Same CV (5-fold StratifiedKFold seed=42) and same
trimmed 25-feature set as exp #3/#4, so scores are directly comparable.

Rationale: Round 1 (tuned LGB) gave +0.000496; XGB carries 0.4 blend weight
with hand-set params, so it is the highest-expected-value next single change.
Round 1's winner was shallow+regularized (depth 3, reg_alpha 2.1) -- the XGB
search space is centered accordingly but kept wide.
"""
import importlib.util
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

# Round 1 (exp #4) Optuna-tuned LGB params — fixed here.
LGB_TUNED = dict(
    objective="binary", metric="auc", n_estimators=3000,
    learning_rate=0.06811817027632358, num_leaves=178, max_depth=3,
    min_child_samples=47, subsample=0.8888159684496226, subsample_freq=1,
    colsample_bytree=0.5316042232727975, reg_alpha=2.142043500176781,
    reg_lambda=0.011502513321845967, random_state=SEED, n_jobs=-1, verbose=-1,
)


def objective(trial):
    import xgboost as xgb
    params = dict(
        objective="binary:logistic", n_estimators=3000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 100, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=SEED, n_jobs=-1, eval_metric="auc",
        early_stopping_rounds=150,
    )
    tr, va = folds[0]
    m = xgb.XGBClassifier(**params)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    return roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])


t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, show_progress_bar=False)
tuning_time = time.time() - t0
print(f"\nOptuna done in {tuning_time:.1f}s, best fold0 AUC={study.best_value:.6f}")
print(f"Best params: {study.best_params}")

xgb_tuned = dict(
    objective="binary:logistic", n_estimators=3000, random_state=SEED,
    n_jobs=-1, eval_metric="auc", early_stopping_rounds=150,
    **study.best_params,
)

# --- fit all three base models on the fixed folds ---
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

results = {}

t0 = time.time()
oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
for f, (tr, va) in enumerate(folds):
    m = lgb.LGBMClassifier(**LGB_TUNED)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[va] = m.predict_proba(X[va])[:, 1]
    pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
results["LGB"] = dict(oof=oof, pred=pred, params=LGB_TUNED,
                      auc=roc_auc_score(y, oof), time_s=round(time.time() - t0, 1))
print(f"LGB(tuned r1) OOF AUC={results['LGB']['auc']:.6f}")

t0 = time.time()
oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
for f, (tr, va) in enumerate(folds):
    m = xgb.XGBClassifier(**xgb_tuned)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    oof[va] = m.predict_proba(X[va])[:, 1]
    pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
results["XGB"] = dict(oof=oof, pred=pred, params=xgb_tuned,
                      auc=roc_auc_score(y, oof), time_s=round(time.time() - t0, 1))
print(f"XGB(tuned r2) OOF AUC={results['XGB']['auc']:.6f}")

t0 = time.time()
cat_params = dict(loss_function="Logloss", eval_metric="AUC", iterations=4000,
                  learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                  random_seed=SEED, thread_count=-1, verbose=False)
oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
for f, (tr, va) in enumerate(folds):
    m = CatBoostClassifier(**cat_params)
    m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
          early_stopping_rounds=200, verbose=False)
    oof[va] = m.predict_proba(X[va])[:, 1]
    pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
results["CAT"] = dict(oof=oof, pred=pred, params=cat_params,
                      auc=roc_auc_score(y, oof), time_s=round(time.time() - t0, 1))
print(f"CAT OOF AUC={results['CAT']['auc']:.6f}")

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
print(f"\n=== BLEND (tuned LGB + tuned XGB + CAT) ===")
print(f"weights={blend_weights}  OOF AUC={blend_auc:.6f}")

PREV_BEST = 0.899891  # exp #4 (Round 1)
print(f"\nPrev best (exp #4): {PREV_BEST}")
print(f"Delta vs prev best: {blend_auc - PREV_BEST:+.6f}")

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
    model="LGB(tuned r1)+XGB(Optuna-tuned,50trials)+CAT weight-searched blend",
    metric="roc_auc",
    direction="maximize",
    score=round(float(blend_auc), 6),
    cv={"scheme": "5fold", "n_splits": N_SPLITS, "seed": SEED, "strategy": "StratifiedKFold(booking_status)"},
    features=FEATS,
    base_models=base_models,
    ensemble={"weights": blend_weights, "score": round(float(blend_auc), 6)},
    postprocess=None,
    submission=sub_name if blend_auc > PREV_BEST else None,
    notes=(f"Round2: Optuna-tuned XGB (50 trials, TPE, fold-0 proxy objective); "
           f"LGB fixed at Round-1 tuned params, CAT unchanged. Tuning time "
           f"{tuning_time:.1f}s. Tuned XGB solo OOF={results['XGB']['auc']:.6f} vs "
           f"hand-set 0.898765. Blend delta vs Round-1 best {PREV_BEST}: "
           f"{blend_auc - PREV_BEST:+.6f}."),
)
print(f"\nLogged experiment_id={exp_id} to experiments.json")
