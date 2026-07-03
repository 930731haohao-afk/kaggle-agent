"""
Phase B iteration for playground-series-s3e19: ratio-decomposition model as a
new ensemble member, blended with LGB + CatBoost (XGB dropped: weight-searched
to 0 in both prior experiments #2 and #3 -- see STATUS.md / experience.md).

CV strategy: IDENTICAL to train.py -- TimeSeriesSplit(n_splits=5) on the 1826
sorted unique train dates -- so scores are directly comparable to experiment #2
(10.17540) and remain the honest extrapolation yardstick.

Ratio decomposition model (RD), fold-safe (fit only on each fold's training
rows, predicted onto that fold's validation rows):
  1. Daily grand-total series total(date) = sum of num_sold over all 75 combos
     on that date (computed from the fold's TRAIN rows only). A harmonic linear
     regression (year + calendar cyclical terms + is_weekend + is_new_year) on
     log1p(total) captures trend + seasonality -- deliberately NOT a GBDT, to
     keep this member structurally different from LGB/CAT (per STATUS.md
     "ratio decomposition" idea and the s3e20 precedent where a structural
     signal beat GBDT for extrapolation).
  2. Country share of the grand total drifts across years (EDA finding) ->
     fit a per-country linear trend of yearly share vs year, extrapolate to
     the validation year, then renormalize across countries so shares sum to 1
     for that year (raw independent per-country trend fits need not sum to 1).
  3. Store x Product share WITHIN a country is close to constant across years
     (EDA finding) -> pooled ratio combo_total / country_total from the fold's
     training data (sums to 1 by construction within each country, no fit
     needed).
  4. Prediction = total_pred(date) * country_share_pred(country, year) *
     combo_share_within_country(country, store, product).

A quick pre-check (log1p(num_sold) ~ country + store + product, additive OLS)
showed this additive/multiplicative structure explains R^2=0.974 of the target
variance using ONLY the categoricals (no date at all) -- strong support for
this decomposition.
"""
import importlib.util
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

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
TREND_FEATURES = ["year", "month_sin", "month_cos", "dow_sin", "dow_cos",
                   "doy_sin", "doy_cos", "is_weekend", "is_new_year"]


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


def fit_ratio_decomp(df_tr):
    """df_tr: rows with date, year, month, dow, day_of_year, is_weekend,
    is_new_year, country, store, product, num_sold (raw string categoricals)."""
    daily = df_tr.groupby("date").agg(total=("num_sold", "sum")).reset_index()
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["dow"] = daily["date"].dt.dayofweek
    daily["doy"] = daily["date"].dt.dayofyear
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["is_new_year"] = ((daily["date"].dt.month == 1) & (daily["date"].dt.day == 1)).astype(int)
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily["dow_sin"] = np.sin(2 * np.pi * daily["dow"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["dow"] / 7)
    daily["doy_sin"] = np.sin(2 * np.pi * daily["doy"] / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * daily["doy"] / 365.25)
    daily["log_total"] = np.log1p(daily["total"])

    lr = LinearRegression()
    lr.fit(daily[TREND_FEATURES], daily["log_total"])

    yearly_country = df_tr.groupby(["year", "country"])["num_sold"].sum().unstack(fill_value=0)
    yearly_total = df_tr.groupby("year")["num_sold"].sum()
    country_share_by_year = yearly_country.div(yearly_total, axis=0)
    years_avail = country_share_by_year.index.values.astype(float)

    country_trend_models = {}
    for c in country_share_by_year.columns:
        vals = country_share_by_year[c].values
        if len(years_avail) >= 2:
            m = LinearRegression().fit(years_avail.reshape(-1, 1), vals)
            country_trend_models[c] = ("trend", m)
        else:
            country_trend_models[c] = ("flat", float(vals[-1]))

    combo_total = df_tr.groupby(["country", "store", "product"])["num_sold"].sum()
    country_total_pooled = df_tr.groupby("country")["num_sold"].sum()
    combo_share_within_country = combo_total / country_total_pooled  # index: (country,store,product)

    return dict(
        lr=lr,
        country_trend_models=country_trend_models,
        countries=list(country_share_by_year.columns),
        combo_share=combo_share_within_country,
    )


def predict_ratio_decomp(components, df):
    d = df.copy()
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)
    d["doy_sin"] = np.sin(2 * np.pi * d["day_of_year"] / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["day_of_year"] / 365.25)

    log_total_pred = components["lr"].predict(d[TREND_FEATURES])
    total_pred = np.expm1(log_total_pred)

    # country share prediction per unique year present in df, then renormalize
    # across countries so shares sum to 1 for that year.
    years_in_df = sorted(d["year"].unique())
    countries = components["countries"]
    year_country_share = {}  # year -> {country: share}
    for yr in years_in_df:
        raw = {}
        for c in countries:
            kind, obj = components["country_trend_models"][c]
            if kind == "trend":
                raw[c] = float(obj.predict([[float(yr)]])[0])
            else:
                raw[c] = obj
        raw = {c: max(v, 1e-6) for c, v in raw.items()}
        s = sum(raw.values())
        year_country_share[yr] = {c: v / s for c, v in raw.items()}

    country_share_pred = np.array([
        year_country_share[row_year][row_country]
        for row_year, row_country in zip(d["year"].values, d["country"].values)
    ])

    combo_idx = list(zip(d["country"].values, d["store"].values, d["product"].values))
    combo_share_pred = components["combo_share"].reindex(combo_idx).values
    # fallback for unseen combos (shouldn't happen: fixed 75-combo panel)
    if np.isnan(combo_share_pred).any():
        fallback = components["combo_share"].mean()
        combo_share_pred = np.where(np.isnan(combo_share_pred), fallback, combo_share_pred)

    pred = total_pred * country_share_pred * combo_share_pred
    return np.clip(pred, a_min=0, a_max=None)


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(data_dir, "test_processed.csv"), parse_dates=["date"])
    raw_train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    raw_test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])

    for c in CAT_FEATURES:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")

    # attach raw string categoricals (needed for ratio-decomp share lookups)
    train["country"] = raw_train["country"].values
    train["store"] = raw_train["store"].values
    train["product"] = raw_train["product"].values
    test["country"] = raw_test["country"].values
    test["store"] = raw_test["store"].values
    test["product"] = raw_test["product"].values

    y_log = train["log_num_sold"].values
    y_true = train["num_sold"].values
    X = train[FEATURE_COLS]
    X_test = test[FEATURE_COLS]

    folds = get_time_folds(train["date"], n_splits=5)

    oof_lgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    oof_rd = np.zeros(len(train))
    fold_mask_any = np.zeros(len(train), dtype=bool)
    fold_scores = {"LGB": [], "CAT": [], "RD": []}

    for i, (tr_dates, va_dates) in enumerate(folds, 1):
        tr_mask = train["date"].isin(tr_dates).values
        va_mask = train["date"].isin(va_dates).values
        fold_mask_any |= va_mask

        X_tr, X_va = X[tr_mask], X[va_mask]
        y_tr, y_va = y_log[tr_mask], y_log[va_mask]
        y_va_true = y_true[va_mask]

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

        cat_model = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
            random_seed=42, verbose=False, early_stopping_rounds=100,
            allow_writing_files=False, thread_count=-1,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        pred_cat = np.expm1(cat_model.predict(X_va))
        oof_cat[va_mask] = pred_cat
        s = smape(y_va_true, pred_cat)
        fold_scores["CAT"].append(round(s, 5))

        df_tr = train.loc[tr_mask, ["date", "year", "month", "dow", "day_of_year",
                                     "is_weekend", "is_new_year", "country", "store",
                                     "product", "num_sold"]]
        df_va = train.loc[va_mask, ["date", "year", "month", "dow", "day_of_year",
                                     "is_weekend", "is_new_year", "country", "store",
                                     "product"]]
        components = fit_ratio_decomp(df_tr)
        pred_rd = predict_ratio_decomp(components, df_va)
        oof_rd[va_mask] = pred_rd
        s = smape(y_va_true, pred_rd)
        fold_scores["RD"].append(round(s, 5))

        print(f"fold {i}: train {tr_mask.sum()} rows, val {va_mask.sum()} rows | "
              f"LGB {fold_scores['LGB'][-1]:.5f} CAT {fold_scores['CAT'][-1]:.5f} "
              f"RD {fold_scores['RD'][-1]:.5f}")

    idx = fold_mask_any
    y_oof_true = y_true[idx]
    lgb_smape = smape(y_oof_true, oof_lgb[idx])
    cat_smape = smape(y_oof_true, oof_cat[idx])
    rd_smape = smape(y_oof_true, oof_rd[idx])
    print(f"\nOverall OOF SMAPE (rows covered: {idx.sum()}): "
          f"LGB={lgb_smape:.5f} CAT={cat_smape:.5f} RD={rd_smape:.5f}")

    # 3-way weight search (LGB, CAT, RD) -- XGB dropped (weight-searched to 0
    # in exp #2 and #3, see experience.md).
    best_w = None
    best_score = np.inf
    step = 0.05
    grid = np.round(np.arange(0, 1.0 + step / 2, step), 2)
    for w_lgb in grid:
        for w_cat in grid:
            w_rd = round(1 - w_lgb - w_cat, 2)
            if w_rd < 0 or w_rd > 1:
                continue
            blend = w_lgb * oof_lgb[idx] + w_cat * oof_cat[idx] + w_rd * oof_rd[idx]
            s = smape(y_oof_true, blend)
            if s < best_score:
                best_score = s
                best_w = (w_lgb, w_cat, w_rd)

    print(f"Best blend weights (LGB, CAT, RD) = {best_w} -> SMAPE {best_score:.5f}")
    elapsed = time.time() - t0
    print(f"\nCV + weight search elapsed: {elapsed:.1f}s")

    cv_dict = {
        "strategy": "TimeSeriesSplit-5fold-on-unique-dates",
        "n_splits": 5,
        "per_model_fold_scores": fold_scores,
    }
    base_models = [
        {"model": "LightGBM", "oof_smape": round(float(lgb_smape), 5)},
        {"model": "CatBoost", "oof_smape": round(float(cat_smape), 5)},
        {"model": "RatioDecomp", "oof_smape": round(float(rd_smape), 5)},
    ]
    ensemble = {
        "type": "weighted-average-oof-search",
        "weights": {"LGB": best_w[0], "CAT": best_w[1], "RD": best_w[2]},
        "grid_step": step,
    }

    exp_id = experiment_log.log_experiment_v2(
        COMPETITION_DIR,
        model="LGB+CAT+RatioDecomp blend (XGB dropped: weight-searched to 0 in exp #2/#3)",
        metric="smape",
        direction="minimize",
        score=float(best_score),
        cv=cv_dict,
        features=FEATURE_COLS,
        base_models=base_models,
        ensemble=ensemble,
        notes=(
            "Phase B round 1: added ratio-decomposition model (RD) as a new "
            "ensemble member. RD = harmonic linear regression (not GBDT) for "
            "the daily grand-total trend/seasonality x per-country linear-"
            "trend-extrapolated share (renormalized to sum 1 per year) x "
            "pooled store/product share within country (near-constant per "
            "EDA). Pre-check: additive OLS of log1p(num_sold) on country+"
            "store+product alone (no date) explains R^2=0.974 of target "
            "variance, motivating this decomposition. Same TimeSeriesSplit "
            "5-fold CV as exp #2 for direct comparability. XGB dropped per "
            "instruction (weight-searched to 0 twice already)."
        ),
    )
    print(f"Logged experiment_id={exp_id}")

    baseline = 10.175397
    improved = best_score < baseline
    print(f"\nBaseline (exp #2, TimeSeriesSplit) = {baseline}")
    print(f"This round blend = {best_score:.5f} -> {'IMPROVED' if improved else 'NOT improved'}")

    if improved:
        print("\nRetraining final models on full training data for submission...")
        lgb_final = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        lgb_final.fit(X, y_log)
        pred_lgb_test = np.expm1(lgb_final.predict(X_test))

        cat_final = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, cat_features=CAT_FEATURES,
            random_seed=42, verbose=False, allow_writing_files=False, thread_count=-1,
        )
        cat_final.fit(X, y_log)
        pred_cat_test = np.expm1(cat_final.predict(X_test))

        df_full = train[["date", "year", "month", "dow", "day_of_year",
                          "is_weekend", "is_new_year", "country", "store",
                          "product", "num_sold"]]
        df_test_rd = test[["date", "year", "month", "dow", "day_of_year",
                            "is_weekend", "is_new_year", "country", "store", "product"]]
        components_full = fit_ratio_decomp(df_full)
        pred_rd_test = predict_ratio_decomp(components_full, df_test_rd)

        final_pred = (best_w[0] * pred_lgb_test + best_w[1] * pred_cat_test
                      + best_w[2] * pred_rd_test)
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
        sub_name = f"sub_lgb_cat_rd_blend_{score_str}_{ts}.csv"
        sub_path = os.path.join(sub_dir, sub_name)
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved: {sub_path}")
        print(submission.describe())

    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
