"""Stage 2 (kaggle-agent skill six-stage pipeline) for playground-series-s6e2
(Heart Disease, ROC-AUC, maximize): LGB + XGB + CAT 5-fold CV + weight-search blend.

Skill stages covered: EDA (eda_summary.json + Feb STATUS.md findings) -> CV design
(StratifiedKFold(5, shuffle, seed=42); binary target, no time/group structure,
train/test clean -- and the SAME folds Feb v1 used, so the Feb v1 ensemble stays the
fold-identical stage-1 anchor) -> feature engineering (features.py, Feb's focused
33-feature v3 set carried forward; its 20 engineered columns get a controlled prior
check in stage 3 round 1 before further trust) -> modeling (three GBDTs below) ->
metric-aware post-processing (NONE -- AUC is a ranking metric; the submission carries
raw positive-class probabilities) -> submission file.

Trains OOF/test predictions, caches them to scripts/cache/solo_*.npz (solo_LGB /
solo_XGB / solo_CAT) so stage 3/stage 4 can reuse without retraining, weight-searches a
blend DIRECTLY against OOF AUC, writes a submission (raw probabilities), and logs solos
+ blend via log_experiment_v2 (metric AUC, direction maximize).

vs Feb-2026 v1 (the fold-identical stage-1 record, OOF AUC 0.95498): same folds; this
pass adds CatBoost as a third distinct architecture (Feb v1's ensemble was LGB+XGB
only), moves to Feb's concluded-best focused 33-feature set, switches fixed tree counts
to early-stopped boosting, and adds an OOF weight search over all three. All training
is bit-reproducible cross-process (LGB deterministic+force_row_wise, all models
num_threads=16) so the stage-4 OOF-reproduction gate holds digit-for-digit -- the exact
same base params + num_boost_round + early_stopping this file, scripts/05_iterate.py and
tree_search/eval_s6e2.py all share.
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
from features import make_folds, build_all, encode_target, auc, TARGET, ID  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s6e2"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
CACHE = f"{COMP}/scripts/cache"
SEED, N_SPLITS = 42, 5
os.makedirs(SUB, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

train_raw = pd.read_csv(f"{DATA}/train.csv")
test_raw = pd.read_csv(f"{DATA}/test.csv")
y = encode_target(train_raw)
FOLDS = make_folds(y)
Xtr_df, Xte_df, FEATS = build_all(train_raw, test_raw)
X = Xtr_df[FEATS].to_numpy(np.float32)
Xtest = Xte_df[FEATS].to_numpy(np.float32)
print(f"[s6e2 stage2] n_features={len(FEATS)}  train={X.shape}  test={Xtest.shape}  "
      f"pos_rate={y.mean():.5f}")


def cache_path(name):
    return os.path.join(CACHE, f"solo_{name}.npz")


def save_cache(name, oof, pred, score):
    np.savez(cache_path(name), oof=oof, pred=pred, auc=score)


def run_lgb(params=None):
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
             max_depth=-1, min_child_samples=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, reg_alpha=0.1, reg_lambda=1.0,
             verbose=-1, num_threads=16, deterministic=True, force_row_wise=True,
             seed=SEED)
    if params:
        p.update(params)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest)); best_iters = []
    t0 = time.time()
    for tr, va in FOLDS:
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
        best_iters.append(m.best_iteration)
    return oof, pred, time.time() - t0, dict(mean_best_iter=int(np.mean(best_iters)))


def run_xgb(params=None):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
             max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
             reg_lambda=1.0, random_state=SEED, n_jobs=16, tree_method="hist")
    if params:
        p.update(params)
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    t0 = time.time()
    for tr, va in FOLDS:
        m = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred, time.time() - t0, {}


def run_cat(params=None):
    from catboost import CatBoostClassifier
    p = dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.03, depth=8,
             l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
             allow_writing_files=False, thread_count=16)
    if params:
        p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 3000))
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    t0 = time.time()
    for tr, va in FOLDS:
        m = CatBoostClassifier(iterations=n_est, **p)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]), early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred, time.time() - t0, {}


if __name__ == "__main__":
    RUNNERS = {"LGB": run_lgb, "XGB": run_xgb, "CAT": run_cat}
    oof, pred, per_score, extra = {}, {}, {}, {}
    for name, fn in RUNNERS.items():
        o, p_, wall, ex = fn()
        s = auc(y, o)
        oof[name], pred[name], per_score[name], extra[name] = o, p_, s, ex
        save_cache(name, o, p_, s)
        print(f"  {name}: OOF AUC={s:.6f}  wall={wall:.1f}s  {ex}")

    # weight search (simplex grid; scored DIRECTLY against OOF AUC, maximize)
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
    sub_path = os.path.join(SUB, f"stage2_blend_{best_s:.5f}_{stamp}.csv")
    pd.DataFrame({ID: test_raw[ID], TARGET: final_test}).to_csv(sub_path, index=False)
    print(f"wrote {sub_path}")
    np.savez(os.path.join(CACHE, "blend_stage2.npz"), oof=oofs @ np.array(best_w),
             pred=final_test, weights=np.array(best_w), names=np.array(NAMES))

    for name in NAMES:
        experiment_log.log_experiment_v2(
            COMP, model=f"{name} (stage2 solo)", metric="AUC", direction="maximize",
            score=round(per_score[name], 6),
            cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED),
            features=FEATS,
            postprocess=[],
            notes=(f"stage2 skill six-stage pipeline; Feb focused 33-feature set on "
                   f"Feb-v1-identical StratifiedKFold(5,shuffle,seed42) folds; early-stopped "
                   f"boosting; bit-reproducible training (LGB deterministic+force_row_wise, all "
                   f"models num_threads=16) so the stage-4 OOF-reproduction gate holds "
                   f"cross-process. {extra.get(name, {})}"
                   + (" CAT is new vs Feb v1 (whose ensemble was LGB+XGB only)."
                      if name == "CAT" else "")),
        )

    experiment_log.log_experiment_v2(
        COMP, model="stage2 LGB+XGB+CAT blend", metric="AUC", direction="maximize",
        score=round(best_s, 6),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED),
        base_models=[dict(name=n, score=round(per_score[n], 6)) for n in NAMES],
        ensemble=dict(weights=dict(zip(NAMES, [round(float(x), 3) for x in best_w])),
                      score=round(best_s, 6)),
        features=FEATS,
        postprocess=[],
        submission=os.path.basename(sub_path),
        notes=("stage2 skill six-stage pipeline blend; simplex grid weight search "
               "(step 0.05) scored directly against OOF AUC; folds identical to the "
               "Feb v1 stage-1 record so scores are directly comparable. No "
               "post-processing (AUC is a ranking metric; submit raw probabilities)."),
    )
    print("RESULT " + json.dumps({"per_model": per_score, "blend": best_s,
                                  "weights": dict(zip(NAMES, best_w))}))
