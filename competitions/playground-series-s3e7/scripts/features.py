"""
Feature engineering — playground-series-s3e7 (Hotel Reservation Cancellation)
================================================================================
All raw columns are already numeric/label-encoded (no missing, no leakage).
EDA findings driving these features:
  - lead_time is the single strongest predictor (single-feature AUC 0.73),
    monotonic with cancellation rate → keep raw + log1p (helps linear/GBM alike).
  - no_of_special_requests, repeated_guest, required_car_parking_space are
    strongly protective (fewer cancellations) → keep raw, add prior-booking
    reliability ratio.
  - avg_price_per_room has a non-monotonic relationship with target → keep raw
    + a per-person normalization (larger parties naturally pay more per room).
  - arrival_month/arrival_date are weakly linear (corr < 0.01) but likely
    encode seasonality → add cyclical sin/cos encoding for wrap-around.
No target/test leakage: every feature below is a deterministic row-wise
function of already-present columns, fit-free (no train-only statistics).
"""
import numpy as np
import pandas as pd

RAW_COLS = [
    "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
    "type_of_meal_plan", "required_car_parking_space", "room_type_reserved",
    "lead_time", "arrival_year", "arrival_month", "arrival_date",
    "market_segment_type", "repeated_guest", "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled", "avg_price_per_room",
    "no_of_special_requests",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["total_nights"] = out["no_of_weekend_nights"] + out["no_of_week_nights"]
    out["total_guests"] = out["no_of_adults"] + out["no_of_children"]
    out["has_children"] = (out["no_of_children"] > 0).astype(np.int8)
    out["weekend_ratio"] = out["no_of_weekend_nights"] / (out["total_nights"] + 1)

    guests_safe = out["total_guests"].replace(0, 1)
    out["price_per_person"] = out["avg_price_per_room"] / guests_safe
    out["price_per_night"] = out["avg_price_per_room"] / (out["total_nights"] + 1)

    out["lead_time_log"] = np.log1p(out["lead_time"])

    prior_total = out["no_of_previous_cancellations"] + out["no_of_previous_bookings_not_canceled"]
    out["prior_cancel_rate"] = out["no_of_previous_cancellations"] / (prior_total + 1)
    out["has_prior_history"] = (prior_total > 0).astype(np.int8)

    out["month_sin"] = np.sin(2 * np.pi * out["arrival_month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["arrival_month"] / 12)
    out["date_sin"] = np.sin(2 * np.pi * out["arrival_date"] / 31)
    out["date_cos"] = np.cos(2 * np.pi * out["arrival_date"] / 31)

    out["no_special_and_price"] = out["no_of_special_requests"] * out["avg_price_per_room"]

    return out


def feature_columns(df: pd.DataFrame, variant: str = "trimmed") -> list:
    full = [
        "total_nights", "total_guests", "has_children", "weekend_ratio",
        "price_per_person", "price_per_night", "lead_time_log",
        "prior_cancel_rate", "has_prior_history",
        "month_sin", "month_cos", "date_sin", "date_cos",
        "no_special_and_price",
    ]
    # trimmed: iteration-2 (reflexion) — dropped cyclical month/date encodings,
    # price_per_night, and the special_requests*price interaction. These four
    # come from raw columns with near-zero target correlation (arrival_date
    # 0.003, arrival_month 0.008) or are redundant with GBM-learnable splits;
    # iteration-1 (full set) scored OOF AUC 0.89788, below the 0.89882
    # generic-baseline — suspected noise/dilution from these low-signal derived cols.
    trimmed = [
        "total_nights", "total_guests", "has_children", "weekend_ratio",
        "price_per_person", "lead_time_log", "prior_cancel_rate", "has_prior_history",
    ]
    engineered = trimmed if variant == "trimmed" else full
    return [c for c in RAW_COLS if c in df.columns] + engineered
