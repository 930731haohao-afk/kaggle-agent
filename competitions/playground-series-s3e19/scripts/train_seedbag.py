"""
Phase B round 2 for playground-series-s3e19: seed bagging on the dominant
LGB member (validated cheap residual-gain pattern across s3e14/s3e7/s3e1/s3e11,
see knowledge/experience.md), after round 1 (ratio decomposition, exp #4)
was rejected -- RD weight-searched to 0 (solo OOF SMAPE 14.75 vs LGB's 10.18).

Members: LGB seed=42 (original), LGB seed=2024 (new), CAT seed=42.
XGB dropped (weight-searched to 0 twice already, per instruction).
RD dropped (round 1 rejected, weight 0).

Same TimeSeriesSplit(5) CV as exp #2/#4 for direct comparability.
"""
import importlib.util
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
from catboost import CatBoostRegressor

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

    oof_lgb1 = np.zeros(len(train))
    oof_lgb2 = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    fold_mask_any = np.zeros(len(train), dtype=bool)
    fold_scores = {"LGB_s42": [], "LGB_s2024": [], "CAT": []}

    for i, (tr_dates, va_dates) in enumerate(folds, 1):
        tr_mask = train["date"].isin(tr_dates).values
        va_mask = train["date"].isin(va_dates).values
        fold_mask_any |= va_mask

        X_tr, X_va = X[tr_mask], X[va_mask]
        y_tr, y_va = y_log[tr_mask], y_log[va_mask]
        y_va_true = y_true[va_mask]

        lgb1 = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        lgb1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                 callbacks=[lgb.early_stopping(100, verbose=False)])
        pred1 = np.expm1(lgb1.predict(X_va))
        oof_lgb1[va_mask] = pred1
        fold_scores["LGB_s42"].append(round(smape(y_va_true, pred1), 5))

        lgb2 = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=2024, verbosity=-1,
        )
        lgb2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                 callbacks=[lgb.early_stopping(100, verbose=False)])
        pred2 = np.expm1(lgb2.predict(X_va))
        oof_lgb2[va_mask] = pred2
        fold_scores["LGB_s2024"].append(round(smape(y_va_true, pred2), 5))

        cat_model = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
            random_seed=42, verbose=False, early_stopping_rounds=100,
            allow_writing_files=False, thread_count=-1,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        pred_cat = np.expm1(cat_model.predict(X_va))
        oof_cat[va_mask] = pred_cat
        fold_scores["CAT"].append(round(smape(y_va_true, pred_cat), 5))

        print(f"fold {i}: LGB_s42 {fold_scores['LGB_s42'][-1]:.5f} "
              f"LGB_s2024 {fold_scores['LGB_s2024'][-1]:.5f} "
              f"CAT {fold_scores['CAT'][-1]:.5f}")

    idx = fold_mask_any
    y_oof_true = y_true[idx]
    lgb1_smape = smape(y_oof_true, oof_lgb1[idx])
    lgb2_smape = smape(y_oof_true, oof_lgb2[idx])
    cat_smape = smape(y_oof_true, oof_cat[idx])
    print(f"\nOverall OOF SMAPE: LGB_s42={lgb1_smape:.5f} LGB_s2024={lgb2_smape:.5f} CAT={cat_smape:.5f}")

    best_w = None
    best_score = np.inf
    step = 0.05
    grid = np.round(np.arange(0, 1.0 + step / 2, step), 2)
    for w1 in grid:
        for w2 in grid:
            w3 = round(1 - w1 - w2, 2)
            if w3 < 0 or w3 > 1:
                continue
            blend = w1 * oof_lgb1[idx] + w2 * oof_lgb2[idx] + w3 * oof_cat[idx]
            s = smape(y_oof_true, blend)
            if s < best_score:
                best_score = s
                best_w = (w1, w2, w3)

    print(f"Best blend weights (LGB_s42, LGB_s2024, CAT) = {best_w} -> SMAPE {best_score:.5f}")
    elapsed = time.time() - t0
    print(f"CV + weight search elapsed: {elapsed:.1f}s")

    cv_dict = {
        "strategy": "TimeSeriesSplit-5fold-on-unique-dates",
        "n_splits": 5,
        "per_model_fold_scores": fold_scores,
    }
    base_models = [
        {"model": "LightGBM_seed42", "oof_smape": round(float(lgb1_smape), 5)},
        {"model": "LightGBM_seed2024", "oof_smape": round(float(lgb2_smape), 5)},
        {"model": "CatBoost_seed42", "oof_smape": round(float(cat_smape), 5)},
    ]
    ensemble = {
        "type": "weighted-average-oof-search",
        "weights": {"LGB_s42": best_w[0], "LGB_s2024": best_w[1], "CAT": best_w[2]},
        "grid_step": step,
    }

    exp_id = experiment_log.log_experiment_v2(
        COMPETITION_DIR,
        model="LGB(seed42)+LGB(seed2024)+CAT seed-bagged blend (RD rejected in exp #4; XGB dropped)",
        metric="smape",
        direction="minimize",
        score=float(best_score),
        cv=cv_dict,
        features=FEATURE_COLS,
        base_models=base_models,
        ensemble=ensemble,
        notes=(
            "Phase B round 2: after ratio-decomposition (exp #4) was rejected "
            "(RD weight-searched to 0, solo OOF 14.75 vs LGB 10.18 -- the "
            "structural signal does NOT beat GBDT here, unlike s3e20; oracle "
            "diagnostics showed the bottleneck is grand-total trend "
            "forecasting under a volatile 5-year/COVID-affected series, not "
            "the store/product/country share decomposition itself, which "
            "roughly matches GBDT when given the true total), pivoted to a "
            "cheap validated pattern from experience.md: seed bagging the "
            "dominant LGB member. Added LGB(seed=2024) alongside the "
            "original LGB(seed=42) and CAT(seed=42); XGB stays dropped "
            "(weight-searched to 0 in exp #2/#3)."
        ),
    )
    print(f"Logged experiment_id={exp_id}")

    prev_best = 10.17489
    improved = best_score < prev_best
    print(f"\nPrevious best (exp #4) = {prev_best}")
    print(f"This round blend = {best_score:.5f} -> {'IMPROVED' if improved else 'NOT improved'}")

    if improved:
        print("\nRetraining final models on full training data for submission...")
        lgb1_final = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        lgb1_final.fit(X, y_log)
        pred1_test = np.expm1(lgb1_final.predict(X_test))

        lgb2_final = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=2024, verbosity=-1,
        )
        lgb2_final.fit(X, y_log)
        pred2_test = np.expm1(lgb2_final.predict(X_test))

        cat_final = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
            random_seed=42, verbose=False, allow_writing_files=False, thread_count=-1,
        )
        cat_final.fit(X, y_log)
        pred_cat_test = np.expm1(cat_final.predict(X_test))

        final_pred = (best_w[0] * pred1_test + best_w[1] * pred2_test + best_w[2] * pred_cat_test)
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
        sub_name = f"sub_lgb_seedbag_cat_blend_{score_str}_{ts}.csv"
        sub_path = os.path.join(sub_dir, sub_name)
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved: {sub_path}")
        print(submission.describe())

    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
