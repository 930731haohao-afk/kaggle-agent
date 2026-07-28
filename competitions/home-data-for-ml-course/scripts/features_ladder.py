"""Bespoke feature engineering for the Ames housing comps (regression).
Shared by house-prices-advanced-regression-techniques and home-data-for-ml-course
(same dataset). Contract: build(tr, te, target, idc) -> (Xtr_df, Xte_df, y_raw).

Domain FE that lifts it above the generic builder:
  * Ordinal-encode the quality/condition ladders (Ex>Gd>TA>Fa>Po>None) instead of label noise.
  * Absence-aware NaN fill (categorical 'None' = feature absent; numeric 0 for areas).
  * Engineered: TotalSF, TotalBath, TotalPorch, HouseAge/RemodAge/GarageAge, Has* flags,
    OverallQual*Cond, OverallQual*TotalSF, LotFrontage by-neighborhood median.
The driver log1p-transforms the target, so we model in log space.
"""
import numpy as np
import pandas as pd

QMAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
QUAL_COLS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
             "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
FIN = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1}


def build(tr, te, target, idc):
    y_raw = tr[target].to_numpy()
    ntr = len(tr)
    df = pd.concat([tr.drop(columns=[target]), te], ignore_index=True)

    def col(c):
        return df[c].fillna(0) if c in df.columns else pd.Series(0, index=df.index)

    # ordinal quality ladders
    for c in QUAL_COLS:
        if c in df.columns:
            df[c] = df[c].map(QMAP).fillna(0)
    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map({"Gd": 4, "Av": 3, "Mn": 2, "No": 1}).fillna(0)
    for c in ["BsmtFinType1", "BsmtFinType2"]:
        if c in df.columns:
            df[c] = df[c].map(FIN).fillna(0)

    # engineered aggregates
    df["TotalSF"] = col("TotalBsmtSF") + col("1stFlrSF") + col("2ndFlrSF")
    df["TotalBath"] = col("FullBath") + 0.5 * col("HalfBath") + col("BsmtFullBath") + 0.5 * col("BsmtHalfBath")
    df["TotalPorch"] = (col("OpenPorchSF") + col("EnclosedPorch") + col("3SsnPorch")
                        + col("ScreenPorch") + col("WoodDeckSF"))
    if "YrSold" in df.columns:
        df["HouseAge"] = df["YrSold"] - col("YearBuilt")
        df["RemodAge"] = df["YrSold"] - col("YearRemodAdd")
        df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).fillna(0) if "GarageYrBlt" in df.columns else 0
        df["IsRemodeled"] = (col("YearBuilt") != col("YearRemodAdd")).astype(int)
        df["IsNew"] = (col("YearBuilt") == df["YrSold"]).astype(int)
    df["HasPool"] = (col("PoolArea") > 0).astype(int)
    df["HasGarage"] = (col("GarageArea") > 0).astype(int)
    df["HasBsmt"] = (col("TotalBsmtSF") > 0).astype(int)
    df["Has2ndFloor"] = (col("2ndFlrSF") > 0).astype(int)
    df["HasFireplace"] = (col("Fireplaces") > 0).astype(int)
    df["HasMasVnr"] = (col("MasVnrArea") > 0).astype(int)
    df["OverallScore"] = col("OverallQual") * col("OverallCond")
    df["QualSF"] = col("OverallQual") * df["TotalSF"]

    # LotFrontage by neighborhood median
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda s: s.fillna(s.median()))
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # remaining nominal categoricals: 'None' = absent, label-encode on union
    obj = [c for c in df.columns if c != idc and not pd.api.types.is_numeric_dtype(df[c])]
    for c in obj:
        u = df[c].fillna("None").astype(str)
        df[c] = u.map({v: i for i, v in enumerate(sorted(u.unique()))}).astype(int)

    # numeric fill (median)
    for c in df.columns:
        if c != idc and pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    if idc in df.columns:
        df = df.drop(columns=[idc])
    Xtr = df.iloc[:ntr].reset_index(drop=True)
    Xte = df.iloc[ntr:].reset_index(drop=True)
    assert Xtr.select_dtypes(include="object").empty, "object cols remain"
    return Xtr, Xte, y_raw
