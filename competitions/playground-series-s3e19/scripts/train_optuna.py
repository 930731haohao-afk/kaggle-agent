"""
Phase B round 4 for playground-series-s3e19: Optuna fold-proxy tuning of LGB
(candidate #4 in the iteration protocol), after round 3's extended seed
bagging (exp #6) improved 10.16605 -> 10.15721.

Fold-proxy choice: unlike the KFold-based competitions in experience.md where
fold-0 is a representative random slice, here CV is TimeSeriesSplit and the
folds are NOT interchangeable -- fold 5 (the last, largest-training-window
fold) is the one whose validation block is closest in spirit to the real
train(2017-21)->test(2022) extrapolation gap. Optuna therefore optimizes SMAPE
on fold 5 only (single fit per trial, fast), then the winning config is
validated on the full TimeSeriesSplit 5-fold OOF for an honest comparison,
and -- if it helps -- added as a NEW pool member (not a replacement, per the
validated "keep diverse members, let weight search decide" pattern).
"""
import importlib.util
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

COMPETITION_DIR = "competitions/playground-series-s3e19"
data_dir = os.path.join(COMPETITION_DIR, "data")
sub_dir = os.path.join(COMPETITION_DIR, "submissions")

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

FEATURE_COLS = [
    "year", "month", "day", "dow", "day_of_year", "weekofyear", "quarter",
    "is_weekend", "is_month_start", "is_month_end", "is_new_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "country_cat", "store_cat", "product_cat",
]
CAT_FEATURES = ["country_cat", "store_cat", "product_cat"]


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    ratio = np.where(denom == 0, 0.0, diff / denom)
    return 100.0 * np.mean(ratio)


def get_time_folds(dates, n_splits=5):
    unique_dates = np.sort(dates.unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for tr_idx, va_idx in tscv.split(unique_dates):
        tr_dates = set(unique_dates[tr_idx])
        va_dates = set(unique_dates[va_idx])
        folds.append((tr_dates, va_dates))
    return folds


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(data_dir, "test_processed.csv"), parse_dates=["date"])

    for c in CAT_FEATURES:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")

    y_log = train["log_num_sold"].values
    y_true = train["num_sold"].values
    X = train[FEATURE_COLS]
    X_test = test[FEATURE_COLS]

    folds = get_time_folds(train["date"], n_splits=5)
    tr_dates5, va_dates5 = folds[4]
    tr_mask5 = train["date"].isin(tr_dates5).values
    va_mask5 = train["date"].isin(va_dates5).values
    X_tr5, X_va5 = X[tr_mask5], X[va_mask5]
    y_tr5, y_va5 = y_log[tr_mask5], y_log[va_mask5]
    y_va5_true = y_true[va_mask5]

    def objective(trial):
        params = dict(
            n_estimators=1500,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            random_state=42, verbosity=-1,
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr5, y_tr5, eval_set=[(X_va5, y_va5)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = np.expm1(m.predict(X_va5))
        return smape(y_va5_true, pred)

    t_optuna0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=40, timeout=400)
    t_optuna = time.time() - t_optuna0
    print(f"Optuna: {len(study.trials)} trials, {t_optuna:.1f}s, best fold-5 SMAPE = {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")

    best_params = dict(study.best_params)
    best_params["n_estimators"] = 2000
    best_params["random_state"] = 42
    best_params["verbosity"] = -1

    # Validate tuned config on full 5-fold OOF, alongside existing pool members
    # (LGB seeds 42/2024/7, CAT seeds 42/2024) reloaded via fresh fits since
    # OOF arrays from prior rounds were not persisted to disk.
    oof_members = {
        "LGB_s42": np.zeros(len(train)), "LGB_s2024": np.zeros(len(train)),
        "LGB_s7": np.zeros(len(train)), "CAT_s42": np.zeros(len(train)),
        "CAT_s2024": np.zeros(len(train)), "LGB_tuned": np.zeros(len(train)),
    }
    fold_mask_any = np.zeros(len(train), dtype=bool)
    fold_scores = {k: [] for k in oof_members}

    lgb_seed_list = [("LGB_s42", 42), ("LGB_s2024", 2024), ("LGB_s7", 7)]
    cat_seed_list = [("CAT_s42", 42), ("CAT_s2024", 2024)]

    for i, (tr_d, va_d) in enumerate(folds, 1):
        tr_mask = train["date"].isin(tr_d).values
        va_mask = train["date"].isin(va_d).values
        fold_mask_any |= va_mask
        X_tr, X_va = X[tr_mask], X[va_mask]
        y_tr, y_va = y_log[tr_mask], y_log[va_mask]
        y_va_true = y_true[va_mask]

        for name, sd in lgb_seed_list:
            m = lgb.LGBMRegressor(
                n_estimators=2000, learning_rate=0.03, num_leaves=63,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=sd, verbosity=-1,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            pred = np.expm1(m.predict(X_va))
            oof_members[name][va_mask] = pred
            fold_scores[name].append(round(smape(y_va_true, pred), 5))

        for name, sd in cat_seed_list:
            m = CatBoostRegressor(
                iterations=2000, learning_rate=0.05, depth=8,
                l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
                random_seed=sd, verbose=False, early_stopping_rounds=100,
                allow_writing_files=False, thread_count=-1,
            )
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            pred = np.expm1(m.predict(X_va))
            oof_members[name][va_mask] = pred
            fold_scores[name].append(round(smape(y_va_true, pred), 5))

        m_tuned = lgb.LGBMRegressor(**best_params)
        m_tuned.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
        pred_tuned = np.expm1(m_tuned.predict(X_va))
        oof_members["LGB_tuned"][va_mask] = pred_tuned
        fold_scores["LGB_tuned"].append(round(smape(y_va_true, pred_tuned), 5))

        print(f"fold {i} done: " + " ".join(f"{k}={fold_scores[k][-1]:.4f}" for k in oof_members))

    idx = fold_mask_any
    y_oof_true = y_true[idx]
    solo_scores = {k: smape(y_oof_true, v[idx]) for k, v in oof_members.items()}
    print("\nSolo OOF SMAPE:", {k: round(v, 5) for k, v in solo_scores.items()})

    # 6-way weight search via coordinate-ascent-ish coarse random search (grid
    # search over 6 simplex dims at step 0.1 is ~C(16,5)=4368 combos; direct
    # nested loop with pruning stays cheap).
    names = list(oof_members.keys())
    arrs = [oof_members[n][idx] for n in names]
    best_w = None
    best_score = np.inf
    step = 0.1
    grid = np.round(np.arange(0, 1.0 + step / 2, step), 2)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0 + 1e-9:
                continue
            for w3 in grid:
                if w1 + w2 + w3 > 1.0 + 1e-9:
                    continue
                for w4 in grid:
                    if w1 + w2 + w3 + w4 > 1.0 + 1e-9:
                        continue
                    for w5 in grid:
                        s5 = w1 + w2 + w3 + w4 + w5
                        if s5 > 1.0 + 1e-9:
                            continue
                        w6 = round(1 - s5, 2)
                        if w6 < 0 or w6 > 1:
                            continue
                        ws = (w1, w2, w3, w4, w5, w6)
                        blend = sum(w * a for w, a in zip(ws, arrs))
                        s = smape(y_oof_true, blend)
                        if s < best_score:
                            best_score = s
                            best_w = ws

    print(f"Best blend weights {names} = {best_w} -> SMAPE {best_score:.5f}")
    elapsed = time.time() - t0
    print(f"Total elapsed: {elapsed:.1f}s")

    cv_dict = {
        "strategy": "TimeSeriesSplit-5fold-on-unique-dates",
        "n_splits": 5,
        "per_model_fold_scores": fold_scores,
        "optuna_fold_proxy": "fold_5 (last, largest training window -- closest to real 2022 extrapolation)",
    }
    base_models = [{"model": k, "oof_smape": round(float(v), 5)} for k, v in solo_scores.items()]
    ensemble = {
        "type": "weighted-average-oof-search",
        "weights": dict(zip(names, best_w)),
        "grid_step": step,
    }

    exp_id = experiment_log.log_experiment_v2(
        COMPETITION_DIR,
        model="LGB(3 seeds)+CAT(2 seeds)+LGB_tuned(Optuna fold-5-proxy) blend",
        metric="smape",
        direction="minimize",
        score=float(best_score),
        cv=cv_dict,
        features=FEATURE_COLS,
        base_models=base_models,
        ensemble=ensemble,
        notes=(
            f"Phase B round 4: Optuna (TPE, 40 trials, {t_optuna:.1f}s) tuned "
            "LGB hyperparameters using fold 5 (the last TimeSeriesSplit fold, "
            "closest in spirit to the real 2022 extrapolation gap) as a fast "
            f"single-fold proxy objective; best fold-5 SMAPE {study.best_value:.5f} "
            f"with params {best_params}. Tuned config re-validated on the full "
            "5-fold OOF and added as a NEW pool member (not a replacement) "
            "alongside the round-3 seed-bagged LGB/CAT pool; weight search "
            "over all 6 members decides the final blend."
        ),
    )
    print(f"Logged experiment_id={exp_id}")

    prev_best = 10.15721
    improved = best_score < prev_best
    print(f"\nPrevious best (exp #6) = {prev_best}")
    print(f"This round blend = {best_score:.5f} -> {'IMPROVED' if improved else 'NOT improved'}")

    if improved:
        print("\nRetraining final models on full training data for submission...")
        preds_test = {}
        for name, sd in lgb_seed_list:
            m = lgb.LGBMRegressor(
                n_estimators=2000, learning_rate=0.03, num_leaves=63,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=sd, verbosity=-1,
            )
            m.fit(X, y_log)
            preds_test[name] = np.expm1(m.predict(X_test))
        for name, sd in cat_seed_list:
            m = CatBoostRegressor(
                iterations=2000, learning_rate=0.05, depth=8,
                l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
                random_seed=sd, verbose=False, allow_writing_files=False, thread_count=-1,
            )
            m.fit(X, y_log)
            preds_test[name] = np.expm1(m.predict(X_test))
        m_tuned_final = lgb.LGBMRegressor(**best_params)
        m_tuned_final.fit(X, y_log)
        preds_test["LGB_tuned"] = np.expm1(m_tuned_final.predict(X_test))

        final_pred = sum(w * preds_test[n] for w, n in zip(best_w, names))
        final_pred = np.clip(final_pred, a_min=0, a_max=None)

        sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
        submission = pd.DataFrame({
            sample_sub.columns[0]: test["id"].values,
            sample_sub.columns[1]: final_pred,
        })
        assert submission.shape == sample_sub.shape
        assert submission.isnull().sum().sum() == 0
        assert (submission[sample_sub.columns[0]].values == sample_sub[sample_sub.columns[0]].values).all()

        os.makedirs(sub_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_str = f"{best_score:.5f}"
        sub_name = f"sub_6way_optuna_blend_{score_str}_{ts}.csv"
        sub_path = os.path.join(sub_dir, sub_name)
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved: {sub_path}")
        print(submission.describe())

    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
