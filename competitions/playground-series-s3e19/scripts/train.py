"""
Modeling for playground-series-s3e19 (Forecast Mini-course Sales)

CV strategy: TimeSeriesSplit(n_splits=5) on the 1826 sorted unique train dates.
Rationale (see scripts/eda.py output): test (2022) is entirely AFTER train
(2017-2021) with zero overlap -> a plain random/KFold split would validate on
days interleaved with training days from the same seasonal cycles, giving an
optimistic estimate of a model that in reality must extrapolate into an unseen
future year. TimeSeriesSplit gives each fold an expanding training window and a
strictly-later validation block, which mirrors the real train->test gap while
still producing 5 folds as required by the pipeline.

Models: LightGBM, XGBoost, CatBoost, all trained on log1p(num_sold) with early
stopping on the fold's own validation block, predictions inverted with expm1
before scoring (SMAPE is defined on the original scale). OOF predictions are
blended via a coarse weight search (same approach as the generic baseline,
experiment #1, which found XGB weight 0 -> LGB dominant).
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
import xgboost as xgb
from catboost import CatBoostRegressor

COMPETITION_DIR = "competitions/playground-series-s3e19"
data_dir = os.path.join(COMPETITION_DIR, "data")
sub_dir = os.path.join(COMPETITION_DIR, "submissions")

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
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

    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    fold_mask_any = np.zeros(len(train), dtype=bool)

    fold_scores = {"LGB": [], "XGB": [], "CAT": []}

    for i, (tr_dates, va_dates) in enumerate(folds, 1):
        tr_mask = train["date"].isin(tr_dates).values
        va_mask = train["date"].isin(va_dates).values
        fold_mask_any |= va_mask

        X_tr, X_va = X[tr_mask], X[va_mask]
        y_tr, y_va = y_log[tr_mask], y_log[va_mask]
        y_va_true = y_true[va_mask]

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        lgb_model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred_lgb = np.expm1(lgb_model.predict(X_va))
        oof_lgb[va_mask] = pred_lgb
        s = smape(y_va_true, pred_lgb)
        fold_scores["LGB"].append(round(s, 5))

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=7,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            enable_categorical=True, tree_method="hist",
            early_stopping_rounds=100, random_state=42, verbosity=0,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred_xgb = np.expm1(xgb_model.predict(X_va))
        oof_xgb[va_mask] = pred_xgb
        s = smape(y_va_true, pred_xgb)
        fold_scores["XGB"].append(round(s, 5))

        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
            random_seed=42, verbose=False, early_stopping_rounds=100,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        pred_cat = np.expm1(cat_model.predict(X_va))
        oof_cat[va_mask] = pred_cat
        s = smape(y_va_true, pred_cat)
        fold_scores["CAT"].append(round(s, 5))

        print(f"fold {i}: train {tr_mask.sum()} rows, val {va_mask.sum()} rows | "
              f"LGB {fold_scores['LGB'][-1]:.5f} XGB {fold_scores['XGB'][-1]:.5f} "
              f"CAT {fold_scores['CAT'][-1]:.5f}")

    # Only rows covered by a validation fold have OOF predictions
    idx = fold_mask_any
    y_oof_true = y_true[idx]
    lgb_smape = smape(y_oof_true, oof_lgb[idx])
    xgb_smape = smape(y_oof_true, oof_xgb[idx])
    cat_smape = smape(y_oof_true, oof_cat[idx])
    print(f"\nOverall OOF SMAPE (rows covered: {idx.sum()}): "
          f"LGB={lgb_smape:.5f} XGB={xgb_smape:.5f} CAT={cat_smape:.5f}")

    # Weight search (coarse grid, weights sum to 1, step 0.1)
    best_w = None
    best_score = np.inf
    step = 0.1
    grid = np.round(np.arange(0, 1.0 + step / 2, step), 2)
    for w_lgb in grid:
        for w_xgb in grid:
            w_cat = round(1 - w_lgb - w_xgb, 2)
            if w_cat < 0 or w_cat > 1:
                continue
            blend = w_lgb * oof_lgb[idx] + w_xgb * oof_xgb[idx] + w_cat * oof_cat[idx]
            s = smape(y_oof_true, blend)
            if s < best_score:
                best_score = s
                best_w = (w_lgb, w_xgb, w_cat)

    print(f"Best blend weights (LGB, XGB, CAT) = {best_w} -> SMAPE {best_score:.5f}")

    elapsed = time.time() - t0
    print(f"\nCV + weight search elapsed: {elapsed:.1f}s")

    # ---- Log experiment ----
    cv_dict = {
        "strategy": "TimeSeriesSplit-5fold-on-unique-dates",
        "n_splits": 5,
        "per_model_fold_scores": fold_scores,
    }
    base_models = [
        {"model": "LightGBM", "params": {"n_estimators": 2000, "learning_rate": 0.03,
                                          "num_leaves": 63, "min_child_samples": 20,
                                          "subsample": 0.9, "colsample_bytree": 0.8,
                                          "reg_lambda": 1.0}, "oof_smape": round(float(lgb_smape), 5)},
        {"model": "XGBoost", "params": {"n_estimators": 2000, "learning_rate": 0.03,
                                         "max_depth": 7, "subsample": 0.9,
                                         "colsample_bytree": 0.8, "reg_lambda": 1.0},
         "oof_smape": round(float(xgb_smape), 5)},
        {"model": "CatBoost", "params": {"iterations": 2000, "learning_rate": 0.05,
                                          "depth": 8, "l2_leaf_reg": 3.0},
         "oof_smape": round(float(cat_smape), 5)},
    ]
    ensemble = {
        "type": "weighted-average-oof-search",
        "weights": {"LGB": best_w[0], "XGB": best_w[1], "CAT": best_w[2]},
        "grid_step": step,
    }

    # ---- Retrain on full data, generate submission ----
    print("\nRetraining final models on full training data...")

    lgb_final = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbosity=-1,
    )
    lgb_final.fit(X, y_log)
    pred_lgb_test = np.expm1(lgb_final.predict(X_test))

    xgb_final = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=7,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        enable_categorical=True, tree_method="hist",
        random_state=42, verbosity=0,
    )
    xgb_final.fit(X, y_log)
    pred_xgb_test = np.expm1(xgb_final.predict(X_test))

    cat_final = CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=8,
        l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
        random_seed=42, verbose=False,
    )
    cat_final.fit(X, y_log)
    pred_cat_test = np.expm1(cat_final.predict(X_test))

    final_pred = (best_w[0] * pred_lgb_test + best_w[1] * pred_xgb_test
                  + best_w[2] * pred_cat_test)
    final_pred = np.clip(final_pred, a_min=0, a_max=None)

    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    submission = pd.DataFrame({
        sample_sub.columns[0]: test["id"].values,
        sample_sub.columns[1]: final_pred,
    })
    assert submission.shape == sample_sub.shape, "shape mismatch vs sample_submission"
    assert submission.isnull().sum().sum() == 0, "NaN in submission"
    assert (submission[sample_sub.columns[0]].values == sample_sub[sample_sub.columns[0]].values).all(), "id mismatch"

    os.makedirs(sub_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    score_str = f"{best_score:.5f}"
    sub_name = f"sub_lgb_xgb_cat_blend_{score_str}_{ts}.csv"
    sub_path = os.path.join(sub_dir, sub_name)
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}")
    print(submission.describe())

    exp_id = experiment_log.log_experiment_v2(
        COMPETITION_DIR,
        model="LGB+XGB+CAT blend (calendar + cyclical + categorical features, log1p target)",
        metric="smape",
        direction="minimize",
        score=float(best_score),
        cv=cv_dict,
        features=FEATURE_COLS,
        base_models=base_models,
        ensemble=ensemble,
        submission=sub_name,
        notes=(
            "Time-based CV (TimeSeriesSplit on unique dates) chosen because test=2022 is "
            "strictly after train=2017-2021 with zero date overlap (see scripts/eda.py). "
            "Features: calendar (year/month/day/dow/doy/weekofyear/quarter), cyclical "
            "sin/cos for month/dow/doy, is_weekend/is_month_start/is_month_end/is_new_year "
            "flags, and label-encoded country/store/product used natively as categoricals "
            "in all three models. Target trained as log1p(num_sold), inverted with expm1 "
            "before SMAPE scoring. Final models retrained on 100% of training data before "
            "generating the submission."
        ),
    )
    print(f"Logged experiment_id={exp_id}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
