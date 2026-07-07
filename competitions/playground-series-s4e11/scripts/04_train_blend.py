"""Tier2 (kaggle-agent skill Stage 3): LGB + XGB + CAT 5-fold CV + weight-search blend
for playground-series-s4e11 (Depression, Accuracy, maximize).

Reuses scripts/features.py (28-feature Feb-2026 set, label-encoding only -- no target
encoding, so no fold-safety correction is needed here, unlike s4e1's surname TE) and the
canonical folds (StratifiedKFold(5, shuffle=True, seed=42) on Depression) that
tier3/tier4 must also use. Trains OOF/test probability predictions, caches them to
scripts/cache/*.npz (solo_LGB / solo_XGB / solo_CAT) so tier3/tier4 can reuse without
retraining, weight-searches a blend DIRECTLY against threshold-optimized Accuracy (never
against raw prob/logloss -- Accuracy is a hard-label metric, see features.py's
best_threshold_accuracy), writes a submission, and logs both solo models and the blend
via log_experiment_v2.

One deliberate ablation vs the Feb baseline (STATUS.md exp #2, is_unbalance=True): this
tier2 pass drops `is_unbalance` from LightGBM's params. Feb's choice predates any
ablation test; the s4e1 cross-season unit found the AUC-ranking-metric prior
("is_unbalance hurts ranking metrics", knowledge/experience.md) transfers weakly at
large-N. Accuracy is NOT a ranking metric (it is threshold-dependent), so this is a
genuinely new test, re-run explicitly as tier3 round 1 against this tier2 baseline.
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import make_folds, build_all, best_threshold_accuracy, TARGET, ID  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s4e11"
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
print(f"[s4e11 tier2] n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}")


def cache_path(name):
    return os.path.join(CACHE, f"solo_{name}.npz")


def save_cache(name, oof, pred, score, thresh):
    np.savez(cache_path(name), oof=oof, pred=pred, acc=score, thresh=thresh)


def run_lgb(params=None):
    import lightgbm as lgb
    p = dict(objective="binary", metric="binary_logloss", learning_rate=0.05, num_leaves=63,
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
    p = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.05,
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
    p = dict(loss_function="Logloss", eval_metric="Logloss", learning_rate=0.05, depth=7,
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
oof, pred, per_score, per_thresh, extra = {}, {}, {}, {}, {}
for name, fn in RUNNERS.items():
    o, p_, wall, ex = fn()
    acc, t = best_threshold_accuracy(y, o)
    oof[name], pred[name], per_score[name], per_thresh[name], extra[name] = o, p_, acc, t, ex
    save_cache(name, o, p_, acc, t)
    print(f"  {name}: OOF Accuracy(opt-thresh)={acc:.5f} @ t={t:.2f}  wall={wall:.1f}s  {ex}")

# weight search (simplex grid; scored DIRECTLY against threshold-optimized accuracy)
NAMES = list(RUNNERS)
oofs = np.stack([oof[n] for n in NAMES], 1)
best_w, best_s, best_t = None, -1e18, 0.5
for w0 in np.arange(0, 1.001, 0.05):
    for w1 in np.arange(0, 1.001 - w0, 0.05):
        w2 = 1 - w0 - w1
        if w2 < -1e-9:
            continue
        acc, t = best_threshold_accuracy(y, oofs @ np.array([w0, w1, w2]))
        if acc > best_s:
            best_s, best_w, best_t = acc, (w0, w1, w2), t
print(f"BEST blend {dict(zip(NAMES, np.round(best_w, 3)))} -> Accuracy={best_s:.6f} @ t={best_t:.2f}")

final_test = (np.stack([pred[n] for n in NAMES], 1) @ np.array(best_w) >= best_t).astype(int)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_path = os.path.join(SUB, f"tier2_blend_{best_s:.5f}_{stamp}.csv")
pd.DataFrame({ID: test_raw[ID], TARGET: final_test}).to_csv(sub_path, index=False)
print(f"wrote {sub_path}")
np.savez(os.path.join(CACHE, "blend_tier2.npz"), oof=oofs @ np.array(best_w),
         pred=np.stack([pred[n] for n in NAMES], 1) @ np.array(best_w),
         weights=np.array(best_w), names=np.array(NAMES), thresh=best_t)

for name in NAMES:
    experiment_log.log_experiment_v2(
        COMP, model=f"{name} (tier2 solo)", metric="accuracy", direction="maximize",
        score=round(per_score[name], 6),
        cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
        features=FEATS,
        postprocess=[f"threshold={per_thresh[name]:.2f}"],
        notes=(f"tier2 skill 6-stage pipeline; 28-feature set (Feb baseline, label-encoding "
               f"only, no TE so no fold-safety correction needed, see features.py docstring); "
               f"{extra.get(name, {})}"
               + (" LGB drops is_unbalance vs Feb baseline (new ablation for this ACCURACY "
                  "metric -- re-tested explicitly in tier3 r1, see knowledge/experience.md)."
                  if name == "LGB" else "")),
    )

experiment_log.log_experiment_v2(
    COMP, model="tier2 LGB+XGB+CAT blend", metric="accuracy", direction="maximize",
    score=round(best_s, 6),
    cv=dict(scheme=f"{N_SPLITS}fold_stratified", n_splits=N_SPLITS, seed=SEED),
    base_models=[dict(name=n, score=round(per_score[n], 6)) for n in NAMES],
    ensemble=dict(weights=dict(zip(NAMES, [round(float(x), 3) for x in best_w])), score=round(best_s, 6)),
    features=FEATS,
    postprocess=[f"threshold={best_t:.2f}"],
    submission=os.path.basename(sub_path),
    notes="tier2 skill 6-stage pipeline blend; simplex grid weight search (step 0.05) scored directly against threshold-optimized accuracy.",
)
print("RESULT " + json.dumps({"per_model": per_score, "blend": best_s, "thresh": best_t,
                                "weights": dict(zip(NAMES, best_w))}))
