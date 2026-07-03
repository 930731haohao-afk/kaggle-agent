"""Phase B self-improvement iteration for playground-series-s3e5 (wine quality, QWK).

Current best: OOF QWK 0.52687 (LGB/XGB/CAT regression blend 0.2/0.2/0.6 +
OptimizedRounder Nelder-Mead cutpoints). CV MUST stay identical to train.py:
5-fold StratifiedKFold(shuffle=True, random_state=42) on raw `quality`.

Per knowledge/experience.md s3e16 lesson: QWK-after-rounder is a DISCRETIZED
metric. All keep/reject decisions in this script are made on the *final*
post-rounder blend QWK, never on a member's raw OOF score.

Usage: uv run python3 competitions/playground-series-s3e5/scripts/iterate.py <step>
  base        -- rebuild the 3 original members (LGB/XGB/CAT), sanity-check
                 against experiments.json exp#3 (0.52687) using a Dirichlet
                 random-search weight-search (replaces the original grid so it
                 scales to >3 members later).
  tune_lgb    -- Optuna, full 5-fold CV, direct objective = post-rounder QWK of
                 the single LGB member (dodges the s3e16 raw-vs-rounded trap).
  r1_pool     -- add tuned LGB as a 4th pool member, re-run weight search,
                 log experiment, compare final QWK to 0.52687.
  seed_bag    -- add a second-seed variant of whichever member is carrying the
                 most blend weight, re-run weight search, log.
  nested_cut  -- diagnostic: nested (leave-fold-out) cutpoint fitting vs the
                 current full-OOF-fit cutpoints, to size the overfit risk
                 flagged in STATUS.md.
  submit      -- if the running best beats 0.52687, retrain full data + write
                 a timestamped submission.

Every member's (oof, pred) is cached to scripts/cache/*.npz so steps are
resumable. CatBoost: allow_writing_files=False, explicit thread_count.
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SUB_DIR = os.path.join(ROOT, "submissions")
CACHE = os.path.join(ROOT, "scripts", "cache")
TARGET, IDC = "quality", "Id"
SEED, N_SPLITS = 42, 5
N_THREADS = 8
os.makedirs(CACHE, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "experiment_log", os.path.join(ROOT, "..", "..", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py"))
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

train = pd.read_csv(os.path.join(DATA, "train_processed.csv"))
test = pd.read_csv(os.path.join(DATA, "test_processed.csv"))
FEAT_COLS = [c for c in train.columns if c not in (IDC, TARGET)]
X = train[FEAT_COLS].to_numpy(np.float32)
y = train[TARGET].to_numpy(int)
Xtest = test[FEAT_COLS].to_numpy(np.float32)
LOW, HIGH = int(y.min()), int(y.max())
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))

STEP = sys.argv[1] if len(sys.argv) > 1 else "base"


class OptimizedRounder:
    def __init__(self, low: int, high: int):
        self.low, self.high = low, high
        self.coef_ = np.arange(low + 0.5, high, 1.0)

    def _to_classes(self, x, coef):
        coef = np.sort(coef)
        return np.clip(np.digitize(x, coef) + self.low, self.low, self.high)

    def _loss(self, coef, x, y):
        return -cohen_kappa_score(y, self._to_classes(x, coef), weights="quadratic")

    def fit(self, x, y):
        res = minimize(self._loss, self.coef_, args=(x, y), method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
        self.coef_ = np.sort(res.x)
        return self

    def predict(self, x):
        return self._to_classes(x, self.coef_)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def post_rounder_qwk(oof_vec):
    """Fit OptimizedRounder on the full OOF vector (matches train.py's approach)
    and return the resulting QWK -- this is the ONLY score used for decisions."""
    r = OptimizedRounder(LOW, HIGH).fit(oof_vec, y)
    return qwk(y, r.predict(oof_vec)), r.coef_.copy()


def cached(name, fn):
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        oof = d["oof"]
        s, _ = post_rounder_qwk(oof)
        print(f"[cache hit] {name}  post-rounder QWK={s:.5f}", flush=True)
        return d["oof"], d["pred"], float(d["time_s"])
    t0 = time.time()
    oof, pred = fn()
    dt = time.time() - t0
    np.savez_compressed(path, oof=oof, pred=pred, time_s=dt)
    s, _ = post_rounder_qwk(oof)
    print(f"[trained] {name}  post-rounder QWK={s:.5f}  [{dt:.1f}s]", flush=True)
    return oof, pred, dt


def run_lgb(params=None, seed=SEED):
    import lightgbm as lgb
    p = dict(objective="regression", n_estimators=1500, learning_rate=0.03,
              num_leaves=31, max_depth=6, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=15,
              random_state=seed, n_jobs=N_THREADS, verbose=-1)
    if params:
        p.update(params)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**p)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def run_xgb(seed=SEED):
    import xgboost as xgb
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1500, learning_rate=0.03,
                              max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                              min_child_weight=5, random_state=seed, n_jobs=N_THREADS, tree_method="hist")
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def run_cat(seed=SEED):
    from catboost import CatBoostRegressor
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = CatBoostRegressor(loss_function="RMSE", iterations=1500, learning_rate=0.03,
                               depth=6, l2_leaf_reg=3.0, random_seed=seed, verbose=False,
                               allow_writing_files=False, thread_count=N_THREADS)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def weight_search(oof_dict, n_random=800, seed=0):
    """Dirichlet random search + local refinement over the simplex, scored on
    post-rounder QWK. Generalizes the original 3-member grid to any pool size."""
    names = list(oof_dict)
    oofs = np.stack([oof_dict[n] for n in names], axis=1)
    rng = np.random.default_rng(seed)
    best_w, best_s, best_coef = None, -1e18, None

    def evalw(w):
        blend = oofs @ w
        s, coef = post_rounder_qwk(blend)
        return s, coef

    # include the corners and uniform as candidates too
    candidates = [np.eye(len(names))[i] for i in range(len(names))]
    candidates.append(np.full(len(names), 1.0 / len(names)))
    candidates += list(rng.dirichlet(np.ones(len(names)), size=n_random))
    for w in candidates:
        s, coef = evalw(w)
        if s > best_s:
            best_s, best_w, best_coef = s, w, coef
    # local coordinate refinement around the best random draw
    for _ in range(3):
        improved = False
        for i in range(len(names)):
            for delta in (-0.05, -0.02, 0.02, 0.05):
                w = best_w.copy()
                w[i] = np.clip(w[i] + delta, 0, 1)
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                s, coef = evalw(w)
                if s > best_s:
                    best_s, best_w, best_coef = s, w, coef
                    improved = True
        if not improved:
            break
    return dict(zip(names, best_w.tolist())), best_s, best_coef


if STEP == "base":
    oof_lgb, pred_lgb, _ = cached("lgb_orig", run_lgb)
    oof_xgb, pred_xgb, _ = cached("xgb_orig", run_xgb)
    oof_cat, pred_cat, _ = cached("cat_orig", run_cat)
    pool = {"LGB": oof_lgb, "XGB": oof_xgb, "CAT": oof_cat}
    w, s, coef = weight_search(pool)
    print(f"\nBASE sanity check: weights={w}  post-rounder QWK={s:.5f}  (exp#3 logged 0.52687)")
    print(f"cutpoints={np.round(coef,3)}")
    if s > 0.52687:
        base_scores = {n: post_rounder_qwk(pool[n])[0] for n in pool}
        exp_id = experiment_log.log_experiment_v2(
            ROOT, model="LGB+XGB+CAT regression blend (finer weight search)",
            metric="quadratic_weighted_kappa", direction="maximize", score=round(float(s), 5),
            cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED, strategy="StratifiedKFold on quality"),
            base_models=[dict(name=n, score=round(float(base_scores[n]), 5)) for n in pool],
            ensemble=dict(weights={k: round(float(v), 3) for k, v in w.items()}, score=round(float(s), 5)),
            postprocess=[f"OptimizedRounder(cutpoints={[round(float(c), 3) for c in coef]})", f"clip[{LOW},{HIGH}]"],
            features=list(FEAT_COLS),
            notes=(f"Round 0 (methodology-only, no new model/feature): exp#3 used a coarse 0.1-step "
                   f"grid over the 3-model simplex; replaced with Dirichlet random search (800 draws) "
                   f"+ coordinate-ascent refinement, scored on post-rounder QWK. Same 3 members, same "
                   f"CV/folds/seed. Final QWK {s:.5f} vs exp#3 grid-search 0.52687 (delta {s-0.52687:+.5f})."),
        )
        print(f"Logged experiment_id={exp_id} (finer weight-search win)")
        with open(os.path.join(CACHE, "running_best.json"), "w") as f:
            json.dump(dict(stage="base_finer_search", score=s, weights=w, coef=coef.tolist(),
                            members=list(pool.keys())), f, indent=2)

elif STEP == "tune_lgb":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 3, 40),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
        )
        oof, _ = run_lgb(params)
        s, _ = post_rounder_qwk(oof)  # direct full-CV post-rounder objective (dodges s3e16 trap)
        return s

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40, timeout=600, show_progress_bar=False)
    dt = time.time() - t0
    print(f"Optuna done in {dt:.1f}s, {len(study.trials)} trials")
    print(f"Best trial post-rounder QWK (single LGB): {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    with open(os.path.join(CACHE, "lgb_tuned_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    oof, pred, _ = cached("lgb_tuned", lambda: run_lgb(study.best_params))
    s, _ = post_rounder_qwk(oof)
    print(f"Tuned LGB single-model post-rounder QWK = {s:.5f} (orig LGB naive-round baseline 0.45220)")

elif STEP == "r1_pool":
    oof_lgb, pred_lgb, _ = cached("lgb_orig", run_lgb)
    oof_xgb, pred_xgb, _ = cached("xgb_orig", run_xgb)
    oof_cat, pred_cat, _ = cached("cat_orig", run_cat)
    with open(os.path.join(CACHE, "lgb_tuned_params.json")) as f:
        tuned_params = json.load(f)
    oof_lgbt, pred_lgbt, _ = cached("lgb_tuned", lambda: run_lgb(tuned_params))

    pool = {"LGB": oof_lgb, "XGB": oof_xgb, "CAT": oof_cat, "LGB_tuned": oof_lgbt}
    w, s, coef = weight_search(pool)
    print(f"\nR1 pool (4-way) weights={w}  post-rounder QWK={s:.5f}  vs baseline 0.52687  delta={s-0.52687:+.5f}")

    base_scores = {n: post_rounder_qwk(pool[n])[0] for n in pool}
    exp_id = experiment_log.log_experiment_v2(
        ROOT, model="LGB+XGB+CAT+LGB_tuned(Optuna full-CV direct-QWK) blend",
        metric="quadratic_weighted_kappa", direction="maximize", score=round(float(s), 5),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED, strategy="StratifiedKFold on quality"),
        base_models=[dict(name=n, score=round(float(base_scores[n]), 5)) for n in pool],
        ensemble=dict(weights={k: round(float(v), 3) for k, v in w.items()}, score=round(float(s), 5)),
        postprocess=[f"OptimizedRounder(cutpoints={[round(float(c), 3) for c in coef]})", f"clip[{LOW},{HIGH}]"],
        features=list(FEAT_COLS),
        notes=(f"Round 1: Optuna (TPE, 40 trials/600s timeout, full 5-fold CV, objective = "
               f"post-rounder QWK directly on OOF -- dodges s3e16 raw-vs-rounded trap) tuned LGB "
               f"added as 4th pool member (kept, not replaced, per s3e14 lesson). "
               f"Weight search: Dirichlet random (800) + coordinate refinement, scored on "
               f"post-rounder QWK. Final QWK {s:.5f} vs exp#3 baseline 0.52687 "
               f"(delta {s-0.52687:+.5f}). Decision made on POST-ROUNDER score only."),
    )
    print(f"Logged experiment_id={exp_id}")
    with open(os.path.join(CACHE, "running_best.json"), "w") as f:
        json.dump(dict(stage="r1_pool", score=s, weights=w, coef=coef.tolist(),
                        members=list(pool.keys())), f, indent=2)

elif STEP == "tune_cat":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def run_cat_params(params, seed=SEED):
        from catboost import CatBoostRegressor
        oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
        for tr, va in folds:
            m = CatBoostRegressor(loss_function="RMSE", random_seed=seed, verbose=False,
                                   allow_writing_files=False, thread_count=N_THREADS, **params)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        return oof, pred

    def objective(trial):
        params = dict(
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
            iterations=trial.suggest_int("iterations", 200, 1500),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 40),
        )
        oof, _ = run_cat_params(params)
        s, _ = post_rounder_qwk(oof)
        return s

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40, timeout=600, show_progress_bar=False)
    dt = time.time() - t0
    print(f"Optuna done in {dt:.1f}s, {len(study.trials)} trials")
    print(f"Best trial post-rounder QWK (single CAT): {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")
    with open(os.path.join(CACHE, "cat_tuned_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    oof, pred, _ = cached("cat_tuned", lambda: run_cat_params(study.best_params))
    s, _ = post_rounder_qwk(oof)
    print(f"Tuned CAT single-model post-rounder QWK = {s:.5f} (orig CAT post-rounder 0.52517)")

elif STEP == "r2_pool":
    with open(os.path.join(CACHE, "running_best.json")) as f:
        rb = json.load(f)
    name_map = {"LGB": "lgb_orig", "XGB": "xgb_orig", "CAT": "cat_orig", "LGB_tuned": "lgb_tuned",
                "LGB_tuned_seed2024": "lgb_tuned_seed2024"}
    pool = {}
    for n in rb["members"]:
        d = np.load(os.path.join(CACHE, f"{name_map[n]}.npz"))
        pool[n] = d["oof"]
    with open(os.path.join(CACHE, "cat_tuned_params.json")) as f:
        cat_params = json.load(f)

    def run_cat_params(params, seed=SEED):
        from catboost import CatBoostRegressor
        oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
        for tr, va in folds:
            m = CatBoostRegressor(loss_function="RMSE", random_seed=seed, verbose=False,
                                   allow_writing_files=False, thread_count=N_THREADS, **params)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        return oof, pred

    oof_catt, pred_catt, _ = cached("cat_tuned", lambda: run_cat_params(cat_params))
    pool["CAT_tuned"] = oof_catt
    w, s, coef = weight_search(pool)
    print(f"\nR2 pool ({len(pool)}-way) weights={w}  post-rounder QWK={s:.5f}  "
          f"vs running best {rb['score']:.5f}  vs original baseline 0.52687")

    base_scores = {n: post_rounder_qwk(pool[n])[0] for n in pool}
    exp_id = experiment_log.log_experiment_v2(
        ROOT, model=f"{len(pool)}-way blend + Optuna-tuned CatBoost (full-CV direct-QWK)",
        metric="quadratic_weighted_kappa", direction="maximize", score=round(float(s), 5),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED, strategy="StratifiedKFold on quality"),
        base_models=[dict(name=n, score=round(float(base_scores[n]), 5)) for n in pool],
        ensemble=dict(weights={k: round(float(v), 3) for k, v in w.items()}, score=round(float(s), 5)),
        postprocess=[f"OptimizedRounder(cutpoints={[round(float(c), 3) for c in coef]})", f"clip[{LOW},{HIGH}]"],
        features=list(FEAT_COLS),
        notes=(f"Round 3: Optuna (TPE, 40 trials/600s timeout, full 5-fold CV, direct post-rounder-QWK "
               f"objective) tuned CatBoost added as an extra pool member (kept, not replaced). "
               f"Final QWK {s:.5f} vs running best {rb['score']:.5f} vs original baseline 0.52687."),
    )
    print(f"Logged experiment_id={exp_id}")
    if s > rb["score"]:
        with open(os.path.join(CACHE, "running_best.json"), "w") as f:
            json.dump(dict(stage="r2_pool", score=s, weights=w, coef=coef.tolist(),
                            members=list(pool.keys())), f, indent=2)
        print("New running best -> saved.")
    else:
        print("Did not improve running best -> not saved.")

elif STEP == "seed_bag":
    with open(os.path.join(CACHE, "running_best.json")) as f:
        rb = json.load(f)
    name_map = {"LGB": "lgb_orig", "XGB": "xgb_orig", "CAT": "cat_orig", "LGB_tuned": "lgb_tuned",
                "LGB_tuned_seed2024": "lgb_tuned_seed2024", "CAT_seed2024": "cat_seed2024",
                "XGB_seed2024": "xgb_seed2024", "LGB_seed2024": "lgb_seed2024", "CAT_tuned": "cat_tuned"}
    pool = {}
    for n in rb["members"]:
        d = np.load(os.path.join(CACHE, f"{name_map[n]}.npz"))
        pool[n] = d["oof"]

    # bag a second seed of whichever member carries the most weight in the running best pool
    top_member = max(rb["weights"], key=rb["weights"].get)
    print(f"Seed-bagging top member: {top_member} (weight {rb['weights'][top_member]:.3f})")
    seed2 = 2024
    if top_member == "LGB_tuned":
        with open(os.path.join(CACHE, "lgb_tuned_params.json")) as f:
            params = json.load(f)
        oof_extra, pred_extra, _ = cached("lgb_tuned_seed2024", lambda: run_lgb(params, seed=seed2))
    elif top_member == "CAT_tuned":
        with open(os.path.join(CACHE, "cat_tuned_params.json")) as f:
            params = json.load(f)
        from catboost import CatBoostRegressor

        def run_cat_tuned_seed():
            oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
            for tr, va in folds:
                m = CatBoostRegressor(loss_function="RMSE", random_seed=seed2, verbose=False,
                                       allow_writing_files=False, thread_count=N_THREADS, **params)
                m.fit(X[tr], y[tr])
                oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
            return oof, pred
        oof_extra, pred_extra, _ = cached("cat_tuned_seed2024", run_cat_tuned_seed)
    elif top_member == "CAT":
        oof_extra, pred_extra, _ = cached("cat_seed2024", lambda: run_cat(seed=seed2))
    elif top_member == "XGB":
        oof_extra, pred_extra, _ = cached("xgb_seed2024", lambda: run_xgb(seed=seed2))
    else:
        oof_extra, pred_extra, _ = cached("lgb_seed2024", lambda: run_lgb(seed=seed2))
    new_member = f"{top_member}_seed2024b" if f"{top_member}_seed2024" in pool else f"{top_member}_seed2024"
    pool[new_member] = oof_extra

    w, s, coef = weight_search(pool)
    print(f"\nSEED_BAG pool ({len(pool)}-way) weights={w}  post-rounder QWK={s:.5f}  "
          f"vs running best {rb['score']:.5f}  vs baseline 0.52687")

    base_scores = {n: post_rounder_qwk(pool[n])[0] for n in pool}
    exp_id = experiment_log.log_experiment_v2(
        ROOT, model=f"{len(pool)}-way blend + seed-bagged {top_member}",
        metric="quadratic_weighted_kappa", direction="maximize", score=round(float(s), 5),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED, strategy="StratifiedKFold on quality"),
        base_models=[dict(name=n, score=round(float(base_scores[n]), 5)) for n in pool],
        ensemble=dict(weights={k: round(float(v), 3) for k, v in w.items()}, score=round(float(s), 5)),
        postprocess=[f"OptimizedRounder(cutpoints={[round(float(c), 3) for c in coef]})", f"clip[{LOW},{HIGH}]"],
        features=list(FEAT_COLS),
        notes=(f"Seed-bag round: added a second random_state={seed2} copy of the top-weighted "
               f"member ({top_member}) from the running best pool, added (not replaced). "
               f"Final post-rounder QWK {s:.5f} vs running best {rb['score']:.5f} vs baseline 0.52687."),
    )
    print(f"Logged experiment_id={exp_id}")
    if s > rb["score"]:
        with open(os.path.join(CACHE, "running_best.json"), "w") as f:
            json.dump(dict(stage="seed_bag", score=s, weights=w, coef=coef.tolist(),
                            members=list(pool.keys())), f, indent=2)
        print("New running best -> saved.")
    else:
        print("Did not improve running best -> not saved.")

elif STEP == "r3_multiclass":
    # Candidate: LGB multiclass classifier (expected-value decode) as a diverse
    # ordinal-head member, per protocol's "ordinal/classification head" strategy.
    import lightgbm as lgb
    with open(os.path.join(CACHE, "running_best.json")) as f:
        rb = json.load(f)
    name_map = {"LGB": "lgb_orig", "XGB": "xgb_orig", "CAT": "cat_orig", "LGB_tuned": "lgb_tuned",
                "LGB_tuned_seed2024": "lgb_tuned_seed2024", "CAT_seed2024": "cat_seed2024",
                "XGB_seed2024": "xgb_seed2024", "LGB_seed2024": "lgb_seed2024", "CAT_tuned": "cat_tuned"}
    pool = {}
    for n in rb["members"]:
        d = np.load(os.path.join(CACHE, f"{name_map[n]}.npz"))
        pool[n] = d["oof"]
    n_classes = HIGH - LOW + 1
    class_values = np.arange(LOW, HIGH + 1)

    def run_lgb_multiclass_ev():
        oof = np.zeros(len(y)); pred = np.zeros((len(Xtest), n_classes))
        y_idx = y - LOW
        for tr, va in folds:
            m = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, n_estimators=600,
                                    learning_rate=0.05, num_leaves=31, max_depth=6, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                                    min_child_samples=15, class_weight="balanced",
                                    random_state=SEED, n_jobs=N_THREADS, verbose=-1)
            m.fit(X[tr], y_idx[tr])
            proba_va = m.predict_proba(X[va])
            oof[va] = proba_va @ class_values
            pred += m.predict_proba(Xtest) / N_SPLITS
        pred_ev = pred @ class_values
        return oof, pred_ev

    oof_mc, pred_mc, _ = cached("lgb_multiclass_ev", run_lgb_multiclass_ev)
    pool["LGB_multiclass_EV"] = oof_mc
    w, s, coef = weight_search(pool)
    print(f"\nR3 multiclass-EV pool ({len(pool)}-way) weights={w}  post-rounder QWK={s:.5f}  "
          f"vs running best {rb['score']:.5f}  vs baseline 0.52687")

    base_scores = {n: post_rounder_qwk(pool[n])[0] for n in pool}
    exp_id = experiment_log.log_experiment_v2(
        ROOT, model=f"{len(pool)}-way blend + LGB multiclass (class_weight=balanced) expected-value decode",
        metric="quadratic_weighted_kappa", direction="maximize", score=round(float(s), 5),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED, strategy="StratifiedKFold on quality"),
        base_models=[dict(name=n, score=round(float(base_scores[n]), 5)) for n in pool],
        ensemble=dict(weights={k: round(float(v), 3) for k, v in w.items()}, score=round(float(s), 5)),
        postprocess=[f"OptimizedRounder(cutpoints={[round(float(c), 3) for c in coef]})", f"clip[{LOW},{HIGH}]"],
        features=list(FEAT_COLS),
        notes=(f"Round: diverse ordinal head -- LGB multiclass classifier (class_weight=balanced, "
               f"targeting the extreme-class data-starvation flagged in STATUS.md) with expected-value "
               f"decode (sum class_i * P(class_i)) added as an extra pool member (kept, not replaced). "
               f"Final QWK {s:.5f} vs running best {rb['score']:.5f} vs baseline 0.52687."),
    )
    print(f"Logged experiment_id={exp_id}")
    if s > rb["score"]:
        with open(os.path.join(CACHE, "running_best.json"), "w") as f:
            json.dump(dict(stage="r3_multiclass", score=s, weights=w, coef=coef.tolist(),
                            members=list(pool.keys())), f, indent=2)
        print("New running best -> saved.")
    else:
        print("Did not improve running best -> not saved.")

elif STEP == "nested_cut":
    # Diagnostic: compare full-OOF-fit cutpoints (current practice) against
    # nested (leave-fold-out) cutpoint fitting, using the running-best blend.
    with open(os.path.join(CACHE, "running_best.json")) as f:
        rb = json.load(f)
    members = rb["members"]
    oof_map = {}
    for n in members:
        base = n.replace("_seed2024", "_seed2024") if False else n
        name_map = {"LGB": "lgb_orig", "XGB": "xgb_orig", "CAT": "cat_orig", "LGB_tuned": "lgb_tuned",
                    "LGB_tuned_seed2024": "lgb_tuned_seed2024", "CAT_seed2024": "cat_seed2024",
                    "XGB_seed2024": "xgb_seed2024", "LGB_seed2024": "lgb_seed2024", "CAT_tuned": "cat_tuned"}
        d = np.load(os.path.join(CACHE, f"{name_map[n]}.npz"))
        oof_map[n] = d["oof"]
    w = np.array([rb["weights"][n] for n in members])
    oofs = np.stack([oof_map[n] for n in members], axis=1)
    blend_oof = oofs @ w

    full_s, full_coef = post_rounder_qwk(blend_oof)

    nested_pred = np.zeros(len(y))
    fold_coefs = []
    for tr, va in folds:
        r = OptimizedRounder(LOW, HIGH).fit(blend_oof[tr], y[tr])
        nested_pred[va] = r.predict(blend_oof[va])
        fold_coefs.append(r.coef_.copy())
    nested_s = qwk(y, nested_pred)
    avg_coef = np.mean(np.stack(fold_coefs), axis=0)
    avg_pred = OptimizedRounder(LOW, HIGH)
    avg_pred.coef_ = avg_coef
    avg_s = qwk(y, avg_pred.predict(blend_oof))

    print(f"Full-OOF-fit cutpoints:    QWK={full_s:.5f}  coef={np.round(full_coef,3)}")
    print(f"Nested (leave-fold-out):  QWK={nested_s:.5f}  fold coefs=\n{np.round(np.stack(fold_coefs),3)}")
    print(f"Averaged-fold cutpoints applied globally: QWK={avg_s:.5f}  coef={np.round(avg_coef,3)}")
    print(f"Gap (full-OOF minus nested): {full_s - nested_s:+.5f}  "
          f"-> {'modest, current practice OK' if full_s - nested_s < 0.02 else 'meaningful overfit risk'}")

    exp_id = experiment_log.log_experiment_v2(
        ROOT, model=f"nested-cutpoint diagnostic on running-best ({len(members)}-way) blend",
        metric="quadratic_weighted_kappa", direction="maximize", score=round(float(nested_s), 5),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED, strategy="StratifiedKFold on quality"),
        ensemble=dict(weights=rb["weights"], score=round(float(nested_s), 5)),
        postprocess=["OptimizedRounder(nested: fit on 4 folds, applied to held-out fold)", f"clip[{LOW},{HIGH}]"],
        features=list(FEAT_COLS),
        notes=(f"Diagnostic (not a model change): full-OOF-fit cutpoints QWK={full_s:.5f} vs "
               f"honest nested (leave-fold-out) cutpoint QWK={nested_s:.5f}, gap={full_s-nested_s:+.5f}. "
               f"Averaged-fold cutpoints applied globally = {avg_s:.5f}. "
               f"Addresses STATUS.md overfit-risk flag on full-OOF cutpoint fitting."),
    )
    print(f"Logged experiment_id={exp_id}")

elif STEP == "submit":
    with open(os.path.join(CACHE, "running_best.json")) as f:
        rb = json.load(f)
    if rb["score"] <= 0.52687:
        print(f"Running best {rb['score']:.5f} does not beat 0.52687 -- no submission written.")
        sys.exit(0)
    members = rb["members"]
    name_map = {"LGB": "lgb_orig", "XGB": "xgb_orig", "CAT": "cat_orig", "LGB_tuned": "lgb_tuned",
                "LGB_tuned_seed2024": "lgb_tuned_seed2024", "CAT_seed2024": "cat_seed2024",
                "XGB_seed2024": "xgb_seed2024", "LGB_seed2024": "lgb_seed2024", "CAT_tuned": "cat_tuned"}
    pred_map = {}
    for n in members:
        d = np.load(os.path.join(CACHE, f"{name_map[n]}.npz"))
        pred_map[n] = d["pred"]
    w = np.array([rb["weights"][n] for n in members])
    preds_stack = np.stack([pred_map[n] for n in members], axis=1)
    final_test_raw = preds_stack @ w
    coef = np.array(rb["coef"])
    rounder = OptimizedRounder(LOW, HIGH)
    rounder.coef_ = coef
    final_test_class = rounder.predict(final_test_raw)

    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = pd.DataFrame({ss.columns[0]: test[IDC].astype(int), ss.columns[1]: final_test_class.astype(int)})
    assert sub.shape == ss.shape
    assert sub.isnull().sum().sum() == 0
    assert (sub[ss.columns[0]] == ss[ss.columns[0]]).all()
    assert sub[ss.columns[1]].between(LOW, HIGH).all()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SUB_DIR, f"sub_iterate_{rb['score']:.5f}_{stamp}.csv")
    sub.to_csv(path, index=False)
    print(f"Wrote submission: {os.path.basename(path)}  ({len(sub)} rows)")
    print(f"Test pred distribution:\n{sub[ss.columns[1]].value_counts().sort_index()}")

else:
    print(f"Unknown step: {STEP}")
    sys.exit(1)
