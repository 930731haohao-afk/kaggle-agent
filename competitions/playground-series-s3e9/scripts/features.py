"""Feature engineering for Concrete Compressive Strength (playground-series-s3e9).

Based on EDA findings:
- AgeInDays is heavily right-skewed (skew 2.75) but log1p(AgeInDays) correlates far
  more strongly with Strength (pearson 0.56 vs 0.33 raw) -> concrete strength develops
  roughly log-linearly with curing time.
- Classic concrete-engineering ratios (water/binder, aggregate/binder, superplasticizer/binder)
  using *total cementitious binder* (Cement + BlastFurnaceSlag + FlyAsh) correlate more
  strongly with Strength than the raw components or the naive water/cement ratio
  (water/binder pearson -0.23 vs water/cement -0.15), since slag/fly-ash are supplementary
  cementitious materials that also react with water.
- BlastFurnaceSlag and FlyAshComponent are ~58-73% zero (used only in some mixes) -> keep
  raw values (tree models handle zero-inflation natively) but add is_used flags.
- ~56% of train feature-rows are exact duplicates of another train row (measurement /
  synthetic-resampling noise: same recipe, different measured Strength). This caps
  achievable CV score (irreducible label noise) and rewards regularized, not memorizing,
  models -> no leakage-prone row-level features are added.
"""
import numpy as np
import pandas as pd

RAW_COLS = [
    "CementComponent", "BlastFurnaceSlag", "FlyAshComponent", "WaterComponent",
    "SuperplasticizerComponent", "CoarseAggregateComponent", "FineAggregateComponent",
    "AgeInDays",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    eps = 1e-6

    # --- age: dominant nonlinear driver of strength (curing time) ---
    X["log_age"] = np.log1p(X["AgeInDays"])
    X["sqrt_age"] = np.sqrt(X["AgeInDays"])

    # --- zero-inflation flags for supplementary cementitious materials ---
    X["has_slag"] = (X["BlastFurnaceSlag"] > 0).astype(np.int8)
    X["has_flyash"] = (X["FlyAshComponent"] > 0).astype(np.int8)
    X["has_superplasticizer"] = (X["SuperplasticizerComponent"] > 0).astype(np.int8)

    # --- total cementitious binder (cement + slag + fly ash all react w/ water) ---
    binder = X["CementComponent"] + X["BlastFurnaceSlag"] + X["FlyAshComponent"] + eps
    X["binder"] = binder
    X["water_binder_ratio"] = X["WaterComponent"] / binder
    X["water_cement_ratio"] = X["WaterComponent"] / (X["CementComponent"] + eps)
    X["sp_binder_ratio"] = X["SuperplasticizerComponent"] / binder
    X["agg_binder_ratio"] = (X["CoarseAggregateComponent"] + X["FineAggregateComponent"]) / binder
    X["fine_coarse_ratio"] = X["FineAggregateComponent"] / (X["CoarseAggregateComponent"] + eps)
    X["slag_cement_ratio"] = X["BlastFurnaceSlag"] / (X["CementComponent"] + eps)
    X["flyash_cement_ratio"] = X["FlyAshComponent"] / (X["CementComponent"] + eps)

    # --- total mix volume proxy (mass sum, no density conversion available) ---
    X["total_mass"] = (X["CementComponent"] + X["BlastFurnaceSlag"] + X["FlyAshComponent"]
                        + X["WaterComponent"] + X["SuperplasticizerComponent"]
                        + X["CoarseAggregateComponent"] + X["FineAggregateComponent"])

    # NOTE: tried explicit binder*log_age / cement*log_age / water_binder*log_age
    # interaction features (self-improvement iteration, experiment #3) - OOF RMSE
    # got *worse* (12.09483 vs 12.07347) because tree models already capture these
    # multiplicative interactions via splits, and the extra collinear columns just
    # add noise/dimensionality on this small (5.4k row) dataset. Reverted; kept as
    # a documented negative result (see experiments.json #3 and STATUS.md).

    return X


def feature_columns(X: pd.DataFrame, id_col="id", target="Strength") -> list[str]:
    return [c for c in X.columns if c not in (id_col, target)]
