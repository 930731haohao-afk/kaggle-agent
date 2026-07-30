"""
Feature engineering for playground-series-s3e19 (Forecast Mini-course Sales)

EDA findings driving this design:
- store/product shares of yearly totals are essentially constant across 2017-2021
  (e.g. Kagglazon always ~0.69, "Improve Your Coding" always ~0.27) -> tree models can
  learn combo-level baselines directly from the categorical columns; no need for a
  hand-built ratio feature.
- country shares drift slowly year over year (Argentina 0.0999->0.0663, Canada
  0.3052->0.3244) -> keep country as its own categorical + let year/date features
  capture drift, rather than assuming a fixed ratio.
- strong day-of-week (weekend/Sunday peak) and month (Dec/Jan peak, spring/summer
  trough) seasonality -> calendar + cyclical encodings.
- New Year's Day (Jan 1) spikes hard (216.7 vs 165.5 overall mean) -> explicit flag.
- target is right-skewed (skew 1.75) but log1p(target) is nearly symmetric
  (skew -0.22) -> train on log1p(num_sold), invert with expm1 for SMAPE/submission.
- yearly totals are noisy/non-monotonic (2020 dip) -> no linear "year trend" feature
  is added beyond the raw year value; extrapolation is left to seasonality + categoricals.
"""
import pandas as pd
import numpy as np
import os

COMPETITION_DIR = "competitions/tabular-playground-series-sep-2022"
data_dir = os.path.join(COMPETITION_DIR, "data")

CAT_COLS = ["country", "store", "product"]


def add_calendar_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_new_year"] = ((df["month"] == 1) & (df["day"] == 1)).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def encode_categoricals(train, test):
    for c in CAT_COLS:
        cats = pd.Categorical(pd.concat([train[c], test[c]], axis=0)).categories
        train[c + "_cat"] = pd.Categorical(train[c], categories=cats).codes
        test[c + "_cat"] = pd.Categorical(test[c], categories=cats).codes
    return train, test


def build():
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train = add_calendar_features(train)
    test = add_calendar_features(test)
    train, test = encode_categoricals(train, test)

    train["log_num_sold"] = np.log1p(train["num_sold"])

    feature_cols = [
        "year", "month", "day", "dow", "day_of_year", "weekofyear", "quarter",
        "is_weekend", "is_month_start", "is_month_end", "is_new_year",
        "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "country_cat", "store_cat", "product_cat",
    ]

    train_out = train[["row_id", "date"] + feature_cols + ["num_sold", "log_num_sold"]]
    test_out = test[["row_id", "date"] + feature_cols]

    train_out.to_csv(os.path.join(data_dir, "train_processed.csv"), index=False)
    test_out.to_csv(os.path.join(data_dir, "test_processed.csv"), index=False)

    print(f"n_features = {len(feature_cols)}")
    print("feature_cols =", feature_cols)
    print("train_processed shape:", train_out.shape)
    print("test_processed shape:", test_out.shape)
    print("NaN check train:", train_out.isnull().sum().sum())
    print("NaN check test:", test_out.isnull().sum().sum())
    return feature_cols


if __name__ == "__main__":
    build()
