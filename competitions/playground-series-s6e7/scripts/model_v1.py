"""s6e7 modeling v1 — 3-class balanced-accuracy classification.

Metric = balanced accuracy => class balancing is MANDATORY (rewards minority-class recall).
LGBM + XGB + CatBoost, class-balanced, native categoricals, 5-fold StratifiedKFold OOF.
Blend on OOF balanced accuracy + per-class prior-adjustment tuning. Deterministic settings
for bit-reproducible OOF (experience.md lgbm-determinism gate).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

COMP = Path(__file__).resolve().parent.parent
ROOT = COMP.parent.parent
_spec = importlib.util.spec_from_file_location(
    "experiment_log", ROOT / ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(experiment_log)

SEED, NF = 42, 5
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]
NTHREAD = 8


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    tr = pd.read_csv(COMP / "data/train.csv"); te = pd.read_csv(COMP / "data/test.csv")
    classes = sorted(tr["health_condition"].unique())      # ['at-risk','fit','unhealthy']
    c2i = {c: i for i, c in enumerate(classes)}
    y = tr["health_condition"].map(c2i).to_numpy()
    for c in CAT:
        tr[c] = tr[c].fillna("missing").astype(str)   # CatBoost: NaN must be a string
        te[c] = te[c].fillna("missing").astype(str)
        cats = sorted(set(tr[c].unique()) | set(te[c].unique()))
        tr[c] = pd.Categorical(tr[c], categories=cats)
        te[c] = pd.Categorical(te[c], categories=cats)
    return tr, te, y, classes


def bal_acc(y, proba, adjust=None):
    p = proba if adjust is None else proba * adjust
    return balanced_accuracy_score(y, p.argmax(1))


def run_lgbm(tr, te, y, folds, nclass):
    import lightgbm as lgb
    oof = np.zeros((len(tr), nclass)); test = np.zeros((len(te), nclass))
    cw = {i: v for i, v in enumerate(len(y) / (nclass * np.bincount(y)))}
    params = dict(objective="multiclass", num_class=nclass, learning_rate=0.03, num_leaves=63,
                  min_child_samples=80, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  reg_lambda=2.0, class_weight=cw, n_estimators=1200, n_jobs=NTHREAD, seed=SEED,
                  deterministic=True, force_row_wise=True, verbose=-1)
    X = tr[NUM + CAT]; Xt = te[NUM + CAT]
    for k, (tri, vai) in enumerate(folds):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tri], y[tri], categorical_feature=CAT,
              eval_set=[(X.iloc[vai], y[vai])], callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[vai] = m.predict_proba(X.iloc[vai]); test += m.predict_proba(Xt) / NF
        log(f"  LGB fold{k}: balacc={balanced_accuracy_score(y[vai], oof[vai].argmax(1)):.4f}")
    return oof, test


def run_cat(tr, te, y, folds, nclass):
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros((len(tr), nclass)); test = np.zeros((len(te), nclass))
    X = tr[NUM + CAT]; Xt = te[NUM + CAT]
    for k, (tri, vai) in enumerate(folds):
        m = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=7, l2_leaf_reg=5,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=SEED, thread_count=NTHREAD, allow_writing_files=False, verbose=0)
        m.fit(Pool(X.iloc[tri], y[tri], cat_features=CAT),
              eval_set=Pool(X.iloc[vai], y[vai], cat_features=CAT), early_stopping_rounds=80)
        oof[vai] = m.predict_proba(X.iloc[vai]); test += m.predict_proba(Xt) / NF
        log(f"  CAT fold{k}: balacc={balanced_accuracy_score(y[vai], oof[vai].argmax(1)):.4f}")
    return oof, test


def run_xgb(tr, te, y, folds, nclass):
    import xgboost as xgb
    oof = np.zeros((len(tr), nclass)); test = np.zeros((len(te), nclass))
    Xc = tr[NUM + CAT].copy(); Xtc = te[NUM + CAT].copy()
    w_per_class = len(y) / (nclass * np.bincount(y)); w = w_per_class[y]
    X = Xc; Xt = Xtc
    for k, (tri, vai) in enumerate(folds):
        m = xgb.XGBClassifier(n_estimators=1200, learning_rate=0.03, max_depth=7, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=2.0, tree_method="hist",
                              enable_categorical=True, eval_metric="mlogloss", n_jobs=NTHREAD,
                              random_state=SEED, early_stopping_rounds=80)
        m.fit(X.iloc[tri], y[tri], sample_weight=w[tri], eval_set=[(X.iloc[vai], y[vai])], verbose=False)
        oof[vai] = m.predict_proba(X.iloc[vai]); test += m.predict_proba(Xt) / NF
        log(f"  XGB fold{k}: balacc={balanced_accuracy_score(y[vai], oof[vai].argmax(1)):.4f}")
    return oof, test


def tune_adjust(y, oof, nclass):
    """Grid-search a per-class multiplier on probabilities to maximize balanced accuracy."""
    base = bal_acc(y, oof); best_adj, best = np.ones(nclass), base
    grid = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]
    for a in grid:
        for b in grid:
            adj = np.array([1.0, a, b])  # keep class0 (majority) fixed, scale minorities
            s = bal_acc(y, oof, adj)
            if s > best: best, best_adj = s, adj
    return best_adj, base, best


def main():
    t0 = time.time()
    tr, te, y, classes = load()
    nclass = len(classes)
    log(f"s6e7 train={tr.shape} classes={classes} priors={np.bincount(y)/len(y)}")
    folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(tr, y))

    models = {}
    for name, fn in [("lgbm", run_lgbm), ("cat", run_cat), ("xgb", run_xgb)]:
        oofp = COMP / f"data/oof_{name}.npz"
        if oofp.exists():
            z = np.load(oofp); oof, test = z["oof"], z["test"]
            log(f"=== {name} (resumed from cache) ===")
        else:
            log(f"=== {name} ===")
            oof, test = fn(tr, te, y, folds, nclass)
            np.savez_compressed(oofp, oof=oof, test=test)
        raw = bal_acc(y, oof); adj, _, adjbest = tune_adjust(y, oof, nclass)
        log(f"{name}: OOF balacc raw={raw:.5f} adjusted={adjbest:.5f} adj={adj}")
        experiment_log.log_experiment_v2(
            str(COMP), model=f"{name} balanced 5fold", metric="balanced_accuracy", direction="maximize",
            score=round(adjbest, 6), cv=dict(scheme="StratifiedKFold", n_splits=NF, seed=SEED),
            features=NUM + CAT, notes=f"class-balanced; raw={raw:.5f}; prior-adjust={adj.tolist()}; determinism on")
        models[name] = (oof, test)

    # simplex blend maximizing balanced accuracy (with prior adjust)
    from itertools import product
    names = list(models); O = np.stack([models[n][0] for n in names]); T = np.stack([models[n][1] for n in names])
    best = (None, None, -1)
    for w in product(np.linspace(0, 1, 6), repeat=len(names)):
        if abs(sum(w) - 1) > 1e-6: continue
        blend = np.tensordot(w, O, axes=1)
        adj, _, s = tune_adjust(y, blend, nclass)
        if s > best[2]: best = (np.array(w), adj, s)
    w, adj, s = best
    log(f"BLEND balacc={s:.5f} weights={dict(zip(names, w.round(3)))} adj={adj}")
    test_blend = np.tensordot(w, T, axes=1) * adj
    pred = test_blend.argmax(1)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [classes[i] for i in pred]})
    out = COMP / "submissions" / "s6e7_blend_v1.csv"; out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    experiment_log.log_experiment_v2(
        str(COMP), model=f"blend {dict(zip(names, w.round(3)))}", metric="balanced_accuracy",
        direction="maximize", score=round(s, 6), cv=dict(scheme="StratifiedKFold", n_splits=NF, seed=SEED),
        base_models=[dict(name=n, score=round(bal_acc(y, models[n][0]), 5)) for n in names],
        ensemble=dict(weights=dict(zip(names, [round(float(x), 3) for x in w])), score=round(s, 5)),
        submission=out.name, notes=f"prior-adjust={adj.tolist()}; balanced-accuracy blend")
    json.dump(dict(blend_balacc=s, weights=dict(zip(names, w.tolist())), adjust=adj.tolist(),
                   solo={n: bal_acc(y, models[n][0]) for n in names}),
              open(COMP / "scripts/v1_results.json", "w"), indent=2)
    log(f"submission: {out} ({len(sub)} rows) pred dist={np.bincount(pred)/len(pred)}")
    log(f"DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
