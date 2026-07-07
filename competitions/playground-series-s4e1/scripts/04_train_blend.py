"""Tier2 (kaggle-agent skill Stage 3): LGB + XGB + CAT 5-fold CV + weight-search blend
for playground-series-s4e1 (Bank Churn, ROC-AUC, maximize).

Reuses scripts/features.py (28-feature Feb-2026 set + corrected fold-safe surname TE)
and the canonical folds (StratifiedKFold(5, shuffle=True, seed=42) on Exited) that
tier3/tier4 must also use. Trains OOF/test predictions, caches them to
scripts/cache/*.npz (solo_LGB / solo_XGB / solo_CAT) so tier3/tier4 can reuse without
retraining, weight-searches a blend, writes a submission, and logs both solo models
and the blend via log_experiment_v2.

One deliberate ablation vs the Feb baseline (STATUS.md exp #2): drops `is_unbalance`
from the LightGBM params. Rationale (knowledge/experience.md, "ROC-AUC(排名指標)"
section): AUC is a ranking metric; imbalance-weighting perturbs the loss surface
rather than improving ranking, and removing it improved AUC on s3e3 (0.81901->0.83292).
That evidence base is a small (1,677-row) dataset -- s4e1 is 165,034 rows, so this run
is itself the first test of whether that prior transfers to a large-sample, cross-season
churn dataset. Both variants get logged (LGB_no_imbalance here; the is_unbalance=True
counterpart is re-tested explicitly in tier3 round 1 as an apples-to-apples ablation).
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import make_folds, build_all, TARGET, ID  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s4e1"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
CACHE = f"{COMP}/scripts/cache"
SEED, N_SPLITS = 42, 5
os.makedirs(SUB, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

train_raw = pd.read_csv(f"{DATA}/train.csv")
test_raw = pd.read_csv(f"{DATA}/test.csv")
y = train_raw[TARGET].to_numpy(np.int64)
FOLDS = make_folds(y)
Xtr_df, Xte_df, FEATS = build_all(train_raw, test_raw, y, FOLDS)
X = Xtr_df[FEATS].to_numpy(np.float32)
Xtest = Xte_df[FEATS].to_numpy(np.float32)
print(f"[s4e1 tier2] n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")


def auc(yt, p):
    return float(roc_auc_score(yt, p))


def cache_path(name):
    return os.path.join(CACHE, f"solo_{name}.npz")


def save_cache(name, oof, pred, score):
    np.savez(cache_path(name), oof=oof, pred=pred, auc=score)


def run_lgb(params=None):
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              max_depth=-1, min_child_samples=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=-1, seed=SEED)
    if params:
        p.update(params)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest)); best_iters = []
    t0 = time.time()
    for tr, va in FOLDS:
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=FEATS)
        dva = lgb.Dataset(X[va], label=y[va], feature_name=FEATS, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
        best_iters.append(m.best_iteration)
    return oof, pred, time.time() - t0, dict(mean_best_iter=int(np.mean(best_iters)))


def run_xgb(params=None):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.05,
              max_depth=6, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
              random_state=SEED, n_jobs=-1, tree_method="hist")
    if params:
        p.update(params)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    t0 = time.time()
    for tr, va in FOLDS:
        m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred, time.time() - t0, {}


def run_cat(params=None):
    from catboost import CatBoostClassifier
    p = dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05, depth=7,
              l2_leaf_reg=3.0, random_seed=SEED, verbose=False, allow_writing_files=False,
              thread_count=20)
    if params:
        p.update(params)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    t0 = time.time()
    for tr, va in FOLDS:
        m = CatBoostClassifier(iterations=2500, early_stopping_rounds=100, **p)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]), verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred, time.time() - t0, {}


RUNNERS = {"LGB": run_lgb, "XGB": run_xgb, "CAT": run_cat}
oof, pred, per_score, extra = {}, {}, {}, {}
for name, fn in RUNNERS.items():
    o, p_, wall, ex = fn()
    s = auc(y, o)
    oof[name], pred[name], per_score[name], extra[name] = o, p_, s, ex
    save_cache(name, o, p_, s)
    print(f"  {name}: OOF AUC={s:.5f}  wall={wall:.1f}s  {ex}")

# weight search (simplex grid, matches run_competition.py convention)
NAMES = list(RUNNERS)
oofs = np.stack([oof[n] for n in NAMES], 1)
best_w, best_s = None, -1e18
for w0 in np.arange(0, 1.001, 0.05):
    for w1 in np.arange(0, 1.001 - w0, 0.05):
        w2 = 1 - w0 - w1
        if w2 < -1e-9:
            continue
        s = auc(y, oofs @ np.array([w0, w1, w2]))
        if s > best_s:
            best_s, best_w = s, (w0, w1, w2)
print(f"BEST blend {dict(zip(NAMES, np.round(best_w, 3)))} -> AUC={best_s:.6f}")

final_test = np.stack([pred[n] for n in NAMES], 1) @ np.array(best_w)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_path = os.path.join(SUB, f"tier2_blend_{best_s:.5f}_{stamp}.csv")
pd.DataFrame({ID: test_raw[ID], TARGET: final_test}).to_csv(sub_path, index=False)
print(f"wrote {sub_path}")
np.savez(os.path.join(CACHE, "blend_tier2.npz"), oof=oofs @ np.array(best_w),
         pred=final_test, weights=np.array(best_w), names=np.array(NAMES))

for name in NAMES:
    experiment_log.log_experiment_v2(
        COMP, model=f"{name} (tier2 solo)", metric="auc", direction="maximize",
        score=round(per_score[name], 6),
        cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
        features=FEATS,
        notes=(f"tier2 skill 6-stage pipeline; 28-feature set (Feb baseline + fold-safe "
               f"surname TE correction); {extra.get(name, {})}"
               + (" LGB drops is_unbalance vs Feb baseline (AUC-ranking-metric prior, "
                  "knowledge/experience.md, see tier3 for ablation)." if name == "LGB" else "")),
    )

experiment_log.log_experiment_v2(
    COMP, model="tier2 LGB+XGB+CAT blend", metric="auc", direction="maximize",
    score=round(best_s, 6),
    cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
    base_models=[dict(name=n, score=round(per_score[n], 6)) for n in NAMES],
    ensemble=dict(weights=dict(zip(NAMES, [round(float(x), 3) for x in best_w])), score=round(best_s, 6)),
    features=FEATS,
    submission=os.path.basename(sub_path),
    notes="tier2 skill 6-stage pipeline blend; simplex grid weight search (step 0.05).",
)
print("RESULT " + json.dumps({"per_model": per_score, "blend": best_s, "weights": dict(zip(NAMES, best_w))}))
