"""
Reflexion diagnostic: experiment #2 (TimeSeriesSplit CV) scored SMAPE 10.17540,
nearly 2x worse than the generic baseline's 5.31891 (experiment #1, plain 5-fold).
This script isolates whether the gap is driven by (a) the CV *scheme* (time-based
vs random row-level KFold) or (b) the engineered feature set, by re-running the
SAME engineered features under plain shuffled 5-fold KFold -- the scheme the
baseline almost certainly used (its log has no time-awareness fields).

This run is diagnostic only; it is not used to pick the submission model.
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

COMPETITION_DIR = "competitions/playground-series-s3e19"
data_dir = os.path.join(COMPETITION_DIR, "data")

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


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
    for c in CAT_FEATURES:
        train[c] = train[c].astype("category")

    y_log = train["log_num_sold"].values
    y_true = train["num_sold"].values
    X = train[FEATURE_COLS]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    fold_scores = {"LGB": [], "XGB": [], "CAT": []}

    for i, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]
        y_va_true = y_true[va_idx]

        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pred_lgb = np.expm1(lgb_model.predict(X_va))
        oof_lgb[va_idx] = pred_lgb
        fold_scores["LGB"].append(round(smape(y_va_true, pred_lgb), 5))

        xgb_model = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=7,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            enable_categorical=True, tree_method="hist",
            early_stopping_rounds=100, random_state=42, verbosity=0,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred_xgb = np.expm1(xgb_model.predict(X_va))
        oof_xgb[va_idx] = pred_xgb
        fold_scores["XGB"].append(round(smape(y_va_true, pred_xgb), 5))

        cat_model = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
            random_seed=42, verbose=False, early_stopping_rounds=100,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        pred_cat = np.expm1(cat_model.predict(X_va))
        oof_cat[va_idx] = pred_cat
        fold_scores["CAT"].append(round(smape(y_va_true, pred_cat), 5))

        print(f"fold {i}: LGB {fold_scores['LGB'][-1]:.5f} "
              f"XGB {fold_scores['XGB'][-1]:.5f} CAT {fold_scores['CAT'][-1]:.5f}")

    lgb_smape = smape(y_true, oof_lgb)
    xgb_smape = smape(y_true, oof_xgb)
    cat_smape = smape(y_true, oof_cat)
    print(f"\nOverall OOF SMAPE: LGB={lgb_smape:.5f} XGB={xgb_smape:.5f} CAT={cat_smape:.5f}")

    best_w, best_score = None, np.inf
    step = 0.1
    grid = np.round(np.arange(0, 1.0 + step / 2, step), 2)
    for w_lgb in grid:
        for w_xgb in grid:
            w_cat = round(1 - w_lgb - w_xgb, 2)
            if w_cat < 0 or w_cat > 1:
                continue
            blend = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
            s = smape(y_true, blend)
            if s < best_score:
                best_score, best_w = s, (w_lgb, w_xgb, w_cat)

    print(f"Best blend weights (LGB, XGB, CAT) = {best_w} -> SMAPE {best_score:.5f}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    exp_id = experiment_log.log_experiment_v2(
        COMPETITION_DIR,
        model="DIAGNOSTIC: LGB+XGB+CAT blend, same 20 engineered features, plain shuffled 5-fold KFold",
        metric="smape",
        direction="minimize",
        score=float(best_score),
        cv={"strategy": "KFold-5fold-shuffled-seed42", "n_splits": 5,
            "per_model_fold_scores": fold_scores},
        features=FEATURE_COLS,
        base_models=[
            {"model": "LightGBM", "oof_smape": round(float(lgb_smape), 5)},
            {"model": "XGBoost", "oof_smape": round(float(xgb_smape), 5)},
            {"model": "CatBoost", "oof_smape": round(float(cat_smape), 5)},
        ],
        ensemble={"type": "weighted-average-oof-search",
                  "weights": {"LGB": best_w[0], "XGB": best_w[1], "CAT": best_w[2]},
                  "grid_step": step},
        notes=(
            "Reflexion diagnostic, not used for submission. Purpose: isolate whether "
            "experiment #2's much worse SMAPE (10.17540 vs baseline 5.31891) comes from "
            "the CV *scheme* (time-based extrapolation is a harder, more honest test) or "
            "from the engineered feature set itself. Same 20 features and same 3 models "
            "as experiment #2, but validated with plain shuffled row-level 5-fold KFold "
            "(matching what the baseline experiment #1 almost certainly used, since its "
            "log has no time-awareness field). If this diagnostic scores close to or "
            "better than 5.31891, the feature engineering is sound and the CV-scheme "
            "choice (documented in scripts/eda.py) fully explains the gap."
        ),
    )
    print(f"Logged experiment_id={exp_id}")


if __name__ == "__main__":
    main()
