"""Feature engineering for playground-series-s5e1 (Forecasting Sticker Sales).

Deliberately mirrors competitions/playground-series-s3e19/scripts/features.py: the same
calendar block and the same categorical encoding, because s5e1 is the same synthetic
country x store x product panel generator and the v5 arms on the two competitions are meant
to be read side by side. Where this file differs from s3e19's, the difference is forced by
the data and is called out in a comment.

TARGET TRUNCATION -- the one substantive difference, and the decision that keeps the arm
comparison controlled.

8871 of 230130 training targets (3.85%) are null. They are not missing at random: the
dataset records no value below 5.0 anywhere, and the nulls are exactly the (series, day)
cells whose value would have fallen under that floor. Evidence in
competitions/playground-series-s5e1/dossier.json; briefly, the nulls sit only in Canada and
Kenya, concentrate in one product, and in a partially-null cell the null rate tracks the
cell's own growth (75.9% in 2010 -> 0% in 2016 as its mean rises 5.3 -> 6.7), with the
lowest-demand weekdays truncated 32-35% against Sunday's 7.1%.

Treatment: null-target rows are DROPPED from training and excluded from CV scoring. They
are never imputed.

  - Not imputed at 5, because the true value is "at most 5", possibly 1. Under MAPE, which
    divides by the actual, writing 5 where the truth is 1 manufactures a 400% error signal
    out of an assumption. Dropping says "unknown"; imputing says something we did not
    observe.
  - Excluding them from CV scoring is forced rather than chosen: MAPE is undefined when the
    actual is null.
  - Every test row still receives a prediction, including the two cells that are 100% null
    across all seven training years (Canada and Kenya x Holographic Goose x Discount
    Stickers, 2.22% of test rows). Those are predicted from the country/store/product
    structure like any other row; they simply contribute no training target.

Why this cannot confound the operator comparison: no operator changes WHICH rows are null.
ratio_target fits on log(y / covariate) and null/anything is still null, so the identical
8871 rows drop out of the baseline, the join_feature arm and the ratio_target arm alike.
Verified empirically before this file was written. The treatment does bias the learned level
upward by removing the lowest-demand days -- a real cost if the goal were to win the
competition, and irrelevant here because it applies equally to every arm.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(_HERE), "data")

CAT_COLS = ["country", "store", "product"]
TARGET = "num_sold"

# the floor the generator truncates at; asserted against the data in build()
TRUNCATION_FLOOR = 5.0


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


FEATURE_COLS = [
    "year", "month", "day", "dow", "day_of_year", "weekofyear", "quarter",
    "is_weekend", "is_month_start", "is_month_end", "is_new_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "country_cat", "store_cat", "product_cat",
]


def drop_truncated(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove null-target rows and report exactly what was removed.

    Returns (train_without_nulls, report). The report is written into the workspace so the
    treatment is auditable rather than implicit -- an arm that silently dropped a different
    number of rows would break the controlled comparison, and this is how that gets caught.
    """
    null_mask = train[TARGET].isna()
    kept = train.loc[~null_mask].reset_index(drop=True)
    observed_min = float(train[TARGET].min())
    report = {
        "rows_before": int(len(train)),
        "rows_dropped": int(null_mask.sum()),
        "rows_after": int(len(kept)),
        "dropped_pct": round(float(null_mask.mean()) * 100, 3),
        "observed_min_non_null": observed_min,
        "truncation_floor_asserted": TRUNCATION_FLOOR,
        "treatment": "dropped, never imputed -- see this module's docstring",
    }
    assert observed_min >= TRUNCATION_FLOOR, (
        f"expected no observed value below the {TRUNCATION_FLOOR} floor, found {observed_min}; "
        f"the truncation premise this treatment rests on does not hold, stop and re-check")
    return kept, report


def build():
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train = add_calendar_features(train)
    test = add_calendar_features(test)
    train, test = encode_categoricals(train, test)

    train, trunc_report = drop_truncated(train)
    train["log_num_sold"] = np.log1p(train[TARGET])

    return train, test, FEATURE_COLS, trunc_report


if __name__ == "__main__":
    import json

    tr, te, feats, rep = build()
    tr.to_csv(os.path.join(data_dir, "train_processed.csv"), index=False)
    te.to_csv(os.path.join(data_dir, "test_processed.csv"), index=False)
    with open(os.path.join(data_dir, "truncation_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"train_processed {tr.shape}, test_processed {te.shape}, {len(feats)} features")
    print(json.dumps(rep, indent=2))
