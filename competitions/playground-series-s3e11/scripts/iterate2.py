"""Phase B self-improvement iteration for playground-series-s3e11 (checkpointed runner).

Usage: uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py <step>
  r1      -- drop XGB: LGB+CAT pool on engineered 21 feats (XGB had zero blend weight twice)
  tune    -- Optuna fold-0 proxy tuning of CatBoost (timeout guard), saves best params JSON
  r2      -- add tuned CAT to pool (LGB + CAT_orig + CAT_tuned), full 5-fold
  r3      -- seed bagging: + CAT_tuned with random_seed=2024 (4-way pool)
  r4      -- deeper store-profile aggregates (fold-safe per-combo means), retrain 4-way pool

Every trained member's (oof, pred) is checkpointed to scripts/cache/*.npz so steps are
resumable and later rounds reuse earlier members without retraining.
CatBoost: allow_writing_files=False (default file logging suspected of hanging in the
sandboxed background run), explicit thread_count.
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, base_feature_columns  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e11"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
CACHE = f"{COMP}/scripts/cache"
TARGET, ID = "cost", "id"
N_SPLITS, SEED = 5, 42
N_THREADS = 20
os.makedirs(SUB, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

STEP = sys.argv[1]

train_raw = pd.read_csv(f"{DATA}/train.csv")
test_raw = pd.read_csv(f"{DATA}/test.csv")
Xtr_full = build_features(train_raw)
Xte_full = build_features(test_raw)
FEATS_BASE = base_feature_columns()
y = np.log1p(train_raw[TARGET].to_numpy(np.float64))
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(Xtr_full))


def rmsle_from_log(y_true_log, y_pred_log):
    pred_cost = np.clip(np.expm1(y_pred_log), 0, None)
    return np.sqrt(mean_squared_error(np.log1p(np.expm1(y_true_log)), np.log1p(pred_cost)))


def fold_safe_te(gtr, gte):
    gm = y.mean()
    te_tr = np.zeros(len(gtr)); te_te = np.zeros(len(gte))
    for tr, va in folds:
        m = pd.Series(y[tr], index=gtr.iloc[tr].index).groupby(gtr.iloc[tr]).mean()
        te_tr[va] = gtr.iloc[va].map(m).fillna(gm).to_numpy()
        te_te += gte.map(m).fillna(gm).to_numpy() / N_SPLITS
    return te_tr, te_te


def fold_safe_group_mean(gtr, gte, series_tr):
    gs = series_tr.mean()
    out_tr = np.zeros(len(gtr)); out_te = np.zeros(len(gte))
    for tr, va in folds:
        m = series_tr.iloc[tr].groupby(gtr.iloc[tr]).mean()
        out_tr[va] = gtr.iloc[va].map(m).fillna(gs).to_numpy()
        out_te += gte.map(m).fillna(gs).to_numpy() / N_SPLITS
    return out_tr, out_te


def build_xy(deep_store=False):
    te_tr, te_te = fold_safe_te(Xtr_full["store_combo"], Xte_full["store_combo"])
    extra_tr = {"store_te": te_tr}
    extra_te = {"store_te": te_te}
    if deep_store:
        for col in ["sales_ratio", "weight_per_case", "gross_weight"]:
            a, b = fold_safe_group_mean(Xtr_full["store_combo"], Xte_full["store_combo"], Xtr_full[col])
            extra_tr[f"combo_mean_{col}"] = a
            extra_te[f"combo_mean_{col}"] = b
    X = Xtr_full[FEATS_BASE].to_numpy(np.float32)
    Xt = Xte_full[FEATS_BASE].to_numpy(np.float32)
    names = list(FEATS_BASE)
    for k in extra_tr:
        X = np.hstack([X, extra_tr[k].reshape(-1, 1).astype(np.float32)])
        Xt = np.hstack([Xt, extra_te[k].reshape(-1, 1).astype(np.float32)])
        names.append(k)
    return X, Xt, names


def run_lgb(X, Xt, seed=SEED):
    import lightgbm as lgb
    params = dict(objective="regression", metric="rmse", n_estimators=2000,
                  learning_rate=0.04, num_leaves=63, min_child_samples=60,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=0.5, reg_lambda=1.0, random_state=seed, n_jobs=N_THREADS, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xt))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xt) / N_SPLITS
        print(f"  [LGB] fold{f} rmsle={rmsle_from_log(y[va], oof[va]):.5f} it={m.best_iteration_}", flush=True)
    return oof, pred


def run_cat(X, Xt, seed=SEED, fold_subset=None, iterations=2500, esr=150, **params):
    from catboost import CatBoostRegressor, Pool
    defaults = dict(depth=8, learning_rate=0.05, l2_leaf_reg=3.0)
    defaults.update(params)
    use_folds = fold_subset if fold_subset is not None else folds
    oof = np.zeros(len(y)); pred = np.zeros(len(Xt))
    for f, (tr, va) in enumerate(use_folds):
        m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=iterations,
                               random_seed=seed, thread_count=N_THREADS, verbose=False,
                               allow_writing_files=False, **defaults)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xt) / len(use_folds)
        print(f"  [CAT] fold{f} rmsle={rmsle_from_log(y[va], oof[va]):.5f} it={m.get_best_iteration()}", flush=True)
    return oof, pred


def cached(name, fn):
    path = f"{CACHE}/{name}.npz"
    if os.path.exists(path):
        d = np.load(path)
        print(f"[cache hit] {name} oof_rmsle={rmsle_from_log(y, d['oof']):.6f}", flush=True)
        return d["oof"], d["pred"], float(d["time_s"])
    t0 = time.time()
    oof, pred = fn()
    dt = time.time() - t0
    np.savez_compressed(path, oof=oof, pred=pred, time_s=dt)
    print(f"[trained] {name} oof_rmsle={rmsle_from_log(y, oof):.6f} [{dt:.0f}s]", flush=True)
    return oof, pred, dt


def weight_search(oof_dict):
    names = list(oof_dict)
    oofs = np.stack([oof_dict[nm] for nm in names], axis=1)
    grid = np.round(np.arange(0, 1.0001, 0.1), 2)
    best_w, best_s = None, 1e9
    for combo in product(grid, repeat=len(names) - 1):
        s = sum(combo)
        if s > 1 + 1e-9:
            continue
        w = np.array(list(combo) + [round(1 - s, 4)])
        sc = rmsle_from_log(y, oofs @ w)
        if sc < best_s:
            best_s, best_w = sc, w
    return dict(zip(names, [round(float(x), 3) for x in best_w])), best_s


def finish_round(tag, model_name, members, weights, score, feats, notes, pred_dict):
    names = list(pred_dict)
    wv = np.array([weights[nm] for nm in names])
    final_log = np.stack([pred_dict[nm] for nm in names], axis=1) @ wv
    final_cost = np.clip(np.expm1(final_log), 0, None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = f"{SUB}/sub_{tag}_{score:.5f}_{stamp}.csv"
    pd.DataFrame({ID: test_raw[ID], TARGET: final_cost}).to_csv(sub_path, index=False)
    experiment_log.log_experiment_v2(
        COMP, model=model_name, metric="rmsle", direction="minimize", score=float(score),
        cv=dict(strategy="5fold_kfold_shuffle", seed=SEED), features=feats,
        base_models=members, ensemble=dict(weights=weights),
        submission=os.path.basename(sub_path), notes=notes)
    print(f"ROUND {tag}: blend={score:.6f} weights={weights}", flush=True)
    print(f"wrote {sub_path} and logged experiment", flush=True)


# ----- shared members -----
def member_lgb(X, Xt, suffix=""):
    return cached(f"lgb{suffix}", lambda: run_lgb(X, Xt))


def member_cat_orig(X, Xt, suffix=""):
    return cached(f"cat_orig{suffix}", lambda: run_cat(X, Xt))


def load_best_params():
    with open(f"{CACHE}/cat_best_params.json") as f:
        return json.load(f)


if STEP == "r1":
    X, Xt, names = build_xy()
    oof_l, pred_l, t_l = member_lgb(X, Xt)
    oof_c, pred_c, t_c = member_cat_orig(X, Xt)
    w, s = weight_search({"LGB": oof_l, "CAT_orig": oof_c})
    finish_round("r1_noxgb", "R1: LGB+CAT (XGB dropped, engineered 21 feats)",
                 [dict(model="LGB", oof_rmsle=round(rmsle_from_log(y, oof_l), 5), time_s=round(t_l, 1)),
                  dict(model="CAT_orig", oof_rmsle=round(rmsle_from_log(y, oof_c), 5), time_s=round(t_c, 1))],
                 w, s, names,
                 "Dropped XGB from pool (zero blend weight in exp#2 and exp#3); same engineered "
                 "21-feature set + fold-safe store_te as exp#3. Frees budget for CatBoost tuning.",
                 {"LGB": pred_l, "CAT_orig": pred_c})

elif STEP == "tune":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, Xt, names = build_xy()
    tr0, va0 = folds[0]

    def objective(trial):
        params = dict(
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
            random_strength=trial.suggest_float("random_strength", 0.0, 10.0),
        )
        oof_f, _ = run_cat(X, Xt, fold_subset=[folds[0]], iterations=2000, esr=100, **params)
        return rmsle_from_log(y[va0], oof_f[va0])

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40, timeout=480)
    print(f"Optuna: {len(study.trials)} trials in {time.time()-t0:.0f}s "
          f"best fold0 rmsle={study.best_value:.6f}", flush=True)
    print(f"best params: {study.best_params}", flush=True)
    with open(f"{CACHE}/cat_best_params.json", "w") as f:
        json.dump(dict(study.best_params,
                       _fold0_rmsle=study.best_value, _n_trials=len(study.trials),
                       _time_s=round(time.time() - t0, 1)), f, indent=2)

elif STEP == "r2":
    X, Xt, names = build_xy()
    bp = {k: v for k, v in load_best_params().items() if not k.startswith("_")}
    meta = load_best_params()
    oof_l, pred_l, t_l = member_lgb(X, Xt)
    oof_c, pred_c, t_c = member_cat_orig(X, Xt)
    oof_t, pred_t, t_t = cached("cat_tuned", lambda: run_cat(X, Xt, **bp))
    w, s = weight_search({"LGB": oof_l, "CAT_orig": oof_c, "CAT_tuned": oof_t})
    finish_round("r2_cattuned",
                 "R2: LGB+CAT_orig+CAT_tuned (Optuna fold-0 proxy CatBoost, added to pool)",
                 [dict(model="LGB", oof_rmsle=round(rmsle_from_log(y, oof_l), 5), time_s=round(t_l, 1)),
                  dict(model="CAT_orig", oof_rmsle=round(rmsle_from_log(y, oof_c), 5), time_s=round(t_c, 1)),
                  dict(model="CAT_tuned", oof_rmsle=round(rmsle_from_log(y, oof_t), 5), time_s=round(t_t, 1))],
                 w, s, names,
                 f"Optuna TPE {meta['_n_trials']} trials ({meta['_time_s']}s, timeout guard 480s) "
                 f"fold-0 proxy tuning CatBoost depth/lr/l2/min_data_in_leaf/random_strength; "
                 f"best params {bp}; tuned CAT ADDED as pool member (original kept, per "
                 f"experience.md add-don't-replace recipe).",
                 {"LGB": pred_l, "CAT_orig": pred_c, "CAT_tuned": pred_t})

elif STEP == "r3":
    X, Xt, names = build_xy()
    bp = {k: v for k, v in load_best_params().items() if not k.startswith("_")}
    oof_l, pred_l, t_l = member_lgb(X, Xt)
    oof_c, pred_c, t_c = member_cat_orig(X, Xt)
    oof_t, pred_t, t_t = cached("cat_tuned", lambda: run_cat(X, Xt, **bp))
    oof_s, pred_s, t_s = cached("cat_tuned_seed2024", lambda: run_cat(X, Xt, seed=2024, **bp))
    w, s = weight_search({"LGB": oof_l, "CAT_orig": oof_c, "CAT_tuned": oof_t,
                          "CAT_tuned_s2024": oof_s})
    finish_round("r3_seedbag",
                 "R3: 4-way LGB+CAT_orig+CAT_tuned+CAT_tuned_seed2024 (seed bagging)",
                 [dict(model="LGB", oof_rmsle=round(rmsle_from_log(y, oof_l), 5), time_s=round(t_l, 1)),
                  dict(model="CAT_orig", oof_rmsle=round(rmsle_from_log(y, oof_c), 5), time_s=round(t_c, 1)),
                  dict(model="CAT_tuned", oof_rmsle=round(rmsle_from_log(y, oof_t), 5), time_s=round(t_t, 1)),
                  dict(model="CAT_tuned_seed2024", oof_rmsle=round(rmsle_from_log(y, oof_s), 5), time_s=round(t_s, 1))],
                 w, s, names,
                 "Seed bagging: tuned CatBoost hyperparams retrained with random_seed=2024, added "
                 "as 4th pool member (cheapest residual gain per experience.md recipe).",
                 {"LGB": pred_l, "CAT_orig": pred_c, "CAT_tuned": pred_t, "CAT_tuned_s2024": pred_s})

elif STEP == "r4":
    X, Xt, names = build_xy(deep_store=True)
    bp = {k: v for k, v in load_best_params().items() if not k.startswith("_")}
    oof_l, pred_l, t_l = cached("lgb_deep", lambda: run_lgb(X, Xt))
    oof_c, pred_c, t_c = cached("cat_orig_deep", lambda: run_cat(X, Xt))
    oof_t, pred_t, t_t = cached("cat_tuned_deep", lambda: run_cat(X, Xt, **bp))
    oof_s, pred_s, t_s = cached("cat_tuned_s2024_deep", lambda: run_cat(X, Xt, seed=2024, **bp))
    w, s = weight_search({"LGB": oof_l, "CAT_orig": oof_c, "CAT_tuned": oof_t,
                          "CAT_tuned_s2024": oof_s})
    finish_round("r4_deepstore",
                 "R4: 4-way pool + deeper store-profile aggregates (fold-safe combo means)",
                 [dict(model="LGB", oof_rmsle=round(rmsle_from_log(y, oof_l), 5), time_s=round(t_l, 1)),
                  dict(model="CAT_orig", oof_rmsle=round(rmsle_from_log(y, oof_c), 5), time_s=round(t_c, 1)),
                  dict(model="CAT_tuned", oof_rmsle=round(rmsle_from_log(y, oof_t), 5), time_s=round(t_t, 1)),
                  dict(model="CAT_tuned_seed2024", oof_rmsle=round(rmsle_from_log(y, oof_s), 5), time_s=round(t_s, 1))],
                 w, s, names,
                 "Added fold-safe per-store_combo mean of sales_ratio/weight_per_case/gross_weight "
                 "on top of store_te (24 feats); same 4-way pool as R3, all retrained.",
                 {"LGB": pred_l, "CAT_orig": pred_c, "CAT_tuned": pred_t, "CAT_tuned_s2024": pred_s})

elif STEP == "r5":
    X, Xt, names = build_xy()
    bp = {k: v for k, v in load_best_params().items() if not k.startswith("_")}
    oof_l, pred_l, t_l = member_lgb(X, Xt)
    oof_c, pred_c, t_c = member_cat_orig(X, Xt)
    oof_t, pred_t, t_t = cached("cat_tuned", lambda: run_cat(X, Xt, **bp))
    oof_s, pred_s, t_s = cached("cat_tuned_seed2024", lambda: run_cat(X, Xt, seed=2024, **bp))
    oof_7, pred_7, t_7 = cached("cat_tuned_seed7", lambda: run_cat(X, Xt, seed=7, **bp))
    w, s = weight_search({"LGB": oof_l, "CAT_orig": oof_c, "CAT_tuned": oof_t,
                          "CAT_tuned_s2024": oof_s, "CAT_tuned_s7": oof_7})
    finish_round("r5_seedbag3",
                 "R5: 5-way pool, third seed of tuned CatBoost (seed bagging extension)",
                 [dict(model="LGB", oof_rmsle=round(rmsle_from_log(y, oof_l), 5), time_s=round(t_l, 1)),
                  dict(model="CAT_orig", oof_rmsle=round(rmsle_from_log(y, oof_c), 5), time_s=round(t_c, 1)),
                  dict(model="CAT_tuned", oof_rmsle=round(rmsle_from_log(y, oof_t), 5), time_s=round(t_t, 1)),
                  dict(model="CAT_tuned_seed2024", oof_rmsle=round(rmsle_from_log(y, oof_s), 5), time_s=round(t_s, 1)),
                  dict(model="CAT_tuned_seed7", oof_rmsle=round(rmsle_from_log(y, oof_7), 5), time_s=round(t_7, 1))],
                 w, s, names,
                 "Third seed (7) of tuned CatBoost added as 5th pool member; back on the 21-feat "
                 "set (R4 deep store aggregates rejected after regression).",
                 {"LGB": pred_l, "CAT_orig": pred_c, "CAT_tuned": pred_t,
                  "CAT_tuned_s2024": pred_s, "CAT_tuned_s7": pred_7})

else:
    raise SystemExit(f"unknown step {STEP}")
