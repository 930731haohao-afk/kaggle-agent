"""Feature engineering for Wild Blueberry Yield (playground-series-s3e14).

Based on EDA findings (scripts/eda.py):
- fruitset/fruitmass/seeds are the dominant predictive block (r=0.83-0.89 with yield) ->
  add interaction/product terms among them.
- The 6 Upper/Lower TRange columns are near-perfectly collinear (r>=0.999 pairwise) ->
  condense into a spread + average feature instead of keeping all 6 raw columns duplicated.
- Pollinator columns (honeybee, bumbles, andrena, osmia) are individually weak -> add a
  combined total_pollinators index and honeybee share.
- RainingDays * AverageRainingDays as a combined rain-intensity feature.
- clonesize is low-cardinality and negatively correlated -> keep raw plus log1p transform.
No missing values, no categoricals to encode, no train/test distribution shift.
"""
import numpy as np
import pandas as pd

TEMP_COLS = ["MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange",
             "MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange"]
POLL_COLS = ["honeybee", "bumbles", "andrena", "osmia"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    eps = 1e-6

    # --- fruit-biology interactions (dominant predictive block) ---
    X["fruitset_x_seeds"] = X["fruitset"] * X["seeds"]
    X["fruitset_x_fruitmass"] = X["fruitset"] * X["fruitmass"]
    X["fruitmass_x_seeds"] = X["fruitmass"] * X["seeds"]
    X["fruit_triple"] = X["fruitset"] * X["fruitmass"] * X["seeds"]
    X["seeds_per_fruitset"] = X["seeds"] / (X["fruitset"] + eps)

    # --- condensed temperature feature (6 raw cols are near-perfectly collinear) ---
    X["temp_spread"] = X["MaxOfUpperTRange"] - X["MinOfLowerTRange"]
    X["temp_avg"] = (X["AverageOfUpperTRange"] + X["AverageOfLowerTRange"]) / 2

    # --- pollinator index ---
    total_poll = X[POLL_COLS].sum(axis=1)
    X["total_pollinators"] = total_poll
    X["honeybee_share"] = X["honeybee"] / (total_poll + eps)

    # --- rain intensity ---
    X["rain_intensity"] = X["RainingDays"] * X["AverageRainingDays"]

    # --- clonesize transform (low-cardinality, monotonic negative relationship) ---
    X["log_clonesize"] = np.log1p(X["clonesize"])

    return X


def feature_columns(X: pd.DataFrame, id_col: str = "id", target: str = "yield") -> list[str]:
    return [c for c in X.columns if c not in (id_col, target)]
