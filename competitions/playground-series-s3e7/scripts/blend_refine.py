"""Round 3 iteration: ensemble-stage refinement only. Base models are exactly
Round 1's winners (Optuna-tuned LGB + hand-set XGB + hand-set CAT, exp #4);
the single change is the blending method:
  (a) probability blend, coarse 0.05 grid  (sanity check, = exp #4)
  (b) probability blend, fine 0.01 grid
  (c) rank-average blend, fine 0.01 grid   (AUC is a pure ranking metric;
      rank transform removes calibration differences between models)
Best OOF of (b)/(c) is compared against exp #4's 0.899891.

Rationale for stopping XGB-retuning: Round 2 showed tuned-XGB solo improved
(+0.0001) but the blend REGRESSED (-0.00017) because both tuned models
converged to similar shallow trees, reducing diversity. So we keep the
diverse hand-set XGB and only refine how predictions are combined.
"""
import importlib.util
import os
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

COMP = "competitions/playground-series-s3e7"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "booking_status", "id"
N_SPLITS, SEED = 5, 42

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

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))

LGB_TUNED = dict(
    objective="binary", metric="auc", n_estimators=3000,
    learning_rate=0.06811817027632358, num_leaves=178, max_depth=3,
    min_child_samples=47, subsample=0.8888159684496226, subsample_freq=1,
    colsample_bytree=0.5316042232727975, reg_alpha=2.142043500176781,
    reg_lambda=0.011502513321845967, random_state=SEED, n_jobs=-1, verbose=-1,
)
XGB_HAND = dict(objective="binary:logistic", n_estimators=3000, learning_rate=0.03,
                max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                eval_metric="auc", early_stopping_rounds=150)
CAT_HAND = dict(loss_function="Logloss", eval_metric="AUC", iterations=4000,
                learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                random_seed=SEED, thread_count=-1, verbose=False)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

results = {}
t0 = time.time()
oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
for tr, va in folds:
    m = lgb.LGBMClassifier(**LGB_TUNED)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[va] = m.predict_proba(X[va])[:, 1]
    pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
results["LGB"] = dict(oof=oof, pred=pred, params=LGB_TUNED,
                      auc=roc_auc_score(y, oof), time_s=round(time.time() - t0, 1))
print(f"LGB(tuned) OOF={results['LGB']['auc']:.6f}")

t0 = time.time()
oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
for tr, va in folds:
    m = xgb.XGBClassifier(**XGB_HAND)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    oof[va] = m.predict_proba(X[va])[:, 1]
    pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
results["XGB"] = dict(oof=oof, pred=pred, params=XGB_HAND,
                      auc=roc_auc_score(y, oof), time_s=round(time.time() - t0, 1))
print(f"XGB(hand) OOF={results['XGB']['auc']:.6f}")

t0 = time.time()
oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
for tr, va in folds:
    m = CatBoostClassifier(**CAT_HAND)
    m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
          early_stopping_rounds=200, verbose=False)
    oof[va] = m.predict_proba(X[va])[:, 1]
    pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
results["CAT"] = dict(oof=oof, pred=pred, params=CAT_HAND,
                      auc=roc_auc_score(y, oof), time_s=round(time.time() - t0, 1))
print(f"CAT(hand) OOF={results['CAT']['auc']:.6f}")

names = list(results.keys())


def weight_search(oofs, step):
    grid = [round(w * step, 4) for w in range(int(round(1 / step)) + 1)]
    best_w, best_a = None, -1
    for w in product(grid, repeat=len(oofs)):
        if abs(sum(w) - 1.0) > 1e-9:
            continue
        a = roc_auc_score(y, sum(wi * o for wi, o in zip(w, oofs)))
        if a > best_a:
            best_a, best_w = a, w
    return best_w, best_a


prob_oofs = [results[n]["oof"] for n in names]
rank_oofs = [rankdata(o) / len(o) for o in prob_oofs]

w_a, auc_a = weight_search(prob_oofs, 0.05)
print(f"\n(a) prob blend 0.05 grid : weights={dict(zip(names, w_a))}  AUC={auc_a:.6f}  (sanity vs exp#4 0.899891)")
t0 = time.time()
w_b, auc_b = weight_search(prob_oofs, 0.01)
print(f"(b) prob blend 0.01 grid : weights={dict(zip(names, w_b))}  AUC={auc_b:.6f}  ({time.time()-t0:.0f}s)")
t0 = time.time()
w_c, auc_c = weight_search(rank_oofs, 0.01)
print(f"(c) rank blend 0.01 grid : weights={dict(zip(names, w_c))}  AUC={auc_c:.6f}  ({time.time()-t0:.0f}s)")

PREV_BEST = 0.899891  # exp #4
if auc_c >= auc_b:
    method, best_w, best_auc = "rank_avg_0.01grid", w_c, auc_c
    blend_pred_parts = [rankdata(results[n]["pred"]) / len(results[n]["pred"]) for n in names]
else:
    method, best_w, best_auc = "prob_0.01grid", w_b, auc_b
    blend_pred_parts = [results[n]["pred"] for n in names]

blend_weights = {n: w for n, w in zip(names, best_w)}
blend_pred = sum(w * p for w, p in zip(best_w, blend_pred_parts))
print(f"\nChosen: {method}  weights={blend_weights}  OOF AUC={best_auc:.6f}")
print(f"Delta vs prev best (exp #4 {PREV_BEST}): {best_auc - PREV_BEST:+.6f}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_name = f"sub_blend_{best_auc:.5f}_{ts}.csv"
if best_auc > PREV_BEST:
    sample_sub = pd.read_csv(f"{DATA}/sample_submission.csv")
    submission = pd.DataFrame({ID: test[ID], TARGET: blend_pred})
    assert submission.shape[0] == sample_sub.shape[0]
    assert list(submission.columns) == list(sample_sub.columns)
    assert submission.isnull().sum().sum() == 0
    submission.to_csv(f"{SUB}/{sub_name}", index=False)
    print(f"\nNEW BEST — saved submission: {SUB}/{sub_name}")
else:
    print("\nNo improvement — submission NOT saved.")

base_models = [
    {"model": n, "params": {k: str(v) for k, v in results[n]["params"].items()},
     "oof_auc": round(float(results[n]["auc"]), 6), "time_s": results[n]["time_s"]}
    for n in names
]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model=f"LGB(tuned r1)+XGB+CAT, ensemble refinement ({method})",
    metric="roc_auc",
    direction="maximize",
    score=round(float(best_auc), 6),
    cv={"scheme": "5fold", "n_splits": N_SPLITS, "seed": SEED, "strategy": "StratifiedKFold(booking_status)"},
    features=FEATS,
    base_models=base_models,
    ensemble={"weights": blend_weights, "score": round(float(best_auc), 6), "method": method},
    postprocess=None,
    submission=sub_name if best_auc > PREV_BEST else None,
    notes=(f"Round3: ensemble-stage-only change on exp #4 base models. "
           f"(a) prob 0.05grid={auc_a:.6f} (sanity, exp#4=0.899891), "
           f"(b) prob 0.01grid={auc_b:.6f}, (c) rank 0.01grid={auc_c:.6f}. "
           f"Chosen {method}. Delta vs exp #4: {best_auc - PREV_BEST:+.6f}. "
           f"Motivation: Round2 showed retuning XGB hurt blend via diversity loss, "
           f"so refine combination instead of base models."),
)
print(f"\nLogged experiment_id={exp_id} to experiments.json")
