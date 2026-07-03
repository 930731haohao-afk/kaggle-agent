"""Feature engineering for playground-series-s3e11.

Only leakage-free, target-independent transforms live here (safe to apply once
to train+test). The one genuinely useful signal found in EDA -- store-profile
group means -- is a *target encoding* and must be fit per-CV-fold, so it is
implemented separately in train.py (fit on train fold only, applied to val/test).
"""
import numpy as np
import pandas as pd

STORE_COLS = ["store_sqft", "coffee_bar", "video_store", "salad_bar", "prepared_food", "florist"]
AMEN_COLS = ["coffee_bar", "video_store", "salad_bar", "prepared_food", "florist"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # store profile key (raw string combo) -- used later for out-of-fold target encoding
    out["store_combo"] = out[STORE_COLS].astype(str).agg("_".join, axis=1)

    # amenity aggregate
    out["amenity_count"] = out[AMEN_COLS].sum(axis=1)

    # simple ratios / derived numerics
    out["weight_per_case"] = out["gross_weight"] / out["units_per_case"].replace(0, np.nan)
    out["sales_ratio"] = out["store_sales(in millions)"] / out["unit_sales(in millions)"].replace(0, np.nan)
    out["children_away"] = out["total_children"] - out["num_children_at_home"]
    out["cars_per_child"] = out["avg_cars_at home(approx).1"] / (out["total_children"] + 1)

    out["weight_per_case"] = out["weight_per_case"].fillna(out["weight_per_case"].median())
    out["sales_ratio"] = out["sales_ratio"].fillna(out["sales_ratio"].median())

    return out


def base_feature_columns():
    """Numeric feature columns available before target-encoding is added (train.py appends
    'store_te' on top of this list after fold-safe encoding)."""
    return [
        "store_sales(in millions)", "unit_sales(in millions)", "total_children",
        "num_children_at_home", "avg_cars_at home(approx).1", "gross_weight",
        "recyclable_package", "low_fat", "units_per_case", "store_sqft",
        "coffee_bar", "video_store", "salad_bar", "prepared_food", "florist",
        "amenity_count", "weight_per_case", "sales_ratio", "children_away", "cars_per_child",
    ]
