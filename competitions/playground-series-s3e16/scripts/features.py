"""Feature engineering for Crab Age (playground-series-s3e16).

Based on EDA findings:
- Sex 'I' (infant) is a strong young-age signal -> is_infant flag + one-hot.
- Heavy collinearity among size/weight -> derive ratios / proportions / density.
- Weight ~ Shucked+Viscera+Shell but not exact -> residual feature.
- Height==0 is invalid (physically impossible) -> treat as missing, impute median.
"""
import numpy as np
import pandas as pd

CAT_COL = "Sex"
WEIGHT_PARTS = ["Shucked Weight", "Viscera Weight", "Shell Weight"]


def build_features(df: pd.DataFrame, height_median: float | None = None) -> pd.DataFrame:
    X = df.copy()

    # --- fix invalid Height==0 (data error) ---
    X["Height_was_zero"] = (X["Height"] == 0).astype(np.int8)
    X.loc[X["Height"] == 0, "Height"] = np.nan
    if height_median is None:
        height_median = X["Height"].median()
    X["Height"] = X["Height"].fillna(height_median)

    # --- Sex encoding ---
    X["is_infant"] = (X[CAT_COL] == "I").astype(np.int8)
    X["is_male"] = (X[CAT_COL] == "M").astype(np.int8)
    X["is_female"] = (X[CAT_COL] == "F").astype(np.int8)

    eps = 1e-6
    w = X["Weight"] + eps

    # --- weight part proportions (shape/composition, scale-invariant) ---
    for p in WEIGHT_PARTS:
        X[f"{p}_ratio"] = X[p] / w
    parts_sum = X[WEIGHT_PARTS].sum(axis=1)
    X["weight_resid"] = X["Weight"] - parts_sum          # whole minus parts
    X["parts_sum"] = parts_sum

    # --- size ratios ---
    X["diam_len"] = X["Diameter"] / (X["Length"] + eps)
    X["height_len"] = X["Height"] / (X["Length"] + eps)
    X["height_diam"] = X["Height"] / (X["Diameter"] + eps)

    # --- volume & density ---
    vol = X["Length"] * X["Diameter"] * X["Height"] + eps
    X["volume"] = vol
    X["density"] = X["Weight"] / vol                     # weight per unit volume
    X["shell_density"] = X["Shell Weight"] / vol

    # --- meat vs shell balance ---
    X["meat_to_shell"] = (X["Shucked Weight"] + X["Viscera Weight"]) / (X["Shell Weight"] + eps)
    X["shucked_to_shell"] = X["Shucked Weight"] / (X["Shell Weight"] + eps)

    return X.drop(columns=[CAT_COL])


def feature_columns(X: pd.DataFrame, id_col="id", target="Age") -> list[str]:
    return [c for c in X.columns if c not in (id_col, target)]
