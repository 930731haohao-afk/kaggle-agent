"""Bespoke feature engineering for spaceship-titanic (binary, accuracy).
Contract: build(tr, te, target, idc) -> (Xtr_df, Xte_df, y_raw). Numeric-only output.

Domain structure exploited (this is what lifts it above the generic builder):
  * PassengerId "GGGG_PP" -> travel Group, GroupSize, Alone, PersonNo.
  * Cabin "deck/num/side" -> Deck, CabinNum, Side, CabinRegion.
  * 5 spending cols -> TotalSpend, NoSpend flag, #nonzero, log1p each (CryoSleep => $0).
  * Name -> Surname -> FamilySize.
  * CryoSleep/VIP tri-state (with missing indicator); Age -> IsChild.
Group/Surname sizes are structural (computed over train+test), not target leakage.
"""
import numpy as np
import pandas as pd

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_LOW = ["HomePlanet", "Destination", "Deck", "Side"]


def build(tr, te, target, idc):
    y_raw = tr[target].to_numpy()
    ntr = len(tr)
    df = pd.concat([tr.drop(columns=[target]), te], ignore_index=True)

    # --- PassengerId GGGG_PP ---
    parts = df[idc].str.split("_", expand=True)
    df["Group"] = parts[0]
    df["PersonNo"] = pd.to_numeric(parts[1], errors="coerce").fillna(1).astype(int)
    gsize = df.groupby("Group")[idc].transform("count")
    df["GroupSize"] = gsize.astype(int)
    df["Alone"] = (gsize == 1).astype(int)

    # --- Cabin deck/num/side ---
    cab = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cab[0]
    df["Side"] = cab[2]
    df["CabinNum"] = pd.to_numeric(cab[1], errors="coerce")
    df["CabinRegion"] = (df["CabinNum"] // 300).fillna(-1).astype(int)
    df["CabinNum"] = df["CabinNum"].fillna(df["CabinNum"].median())

    # --- spending (CryoSleep passengers spend $0) ---
    for c in SPEND:
        df[c] = df[c].fillna(0.0)
    df["TotalSpend"] = df[SPEND].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["NNZSpend"] = (df[SPEND] > 0).sum(axis=1)
    for c in SPEND:
        df[f"log_{c}"] = np.log1p(df[c])
    df["log_TotalSpend"] = np.log1p(df["TotalSpend"])
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"] + df["RoomService"]
    df["BasicSpend"] = df["FoodCourt"] + df["ShoppingMall"]

    # --- CryoSleep / VIP tri-state (0/1/-1) + missing indicator ---
    for c in ["CryoSleep", "VIP"]:
        df[f"{c}_isna"] = df[c].isna().astype(int)
        df[c] = df[c].map({True: 1, False: 0}).fillna(-1).astype(int)

    # --- Age ---
    df["Age_isna"] = df["Age"].isna().astype(int)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["IsChild"] = (df["Age"] < 13).astype(int)

    # --- Name -> Surname -> FamilySize ---
    df["Surname"] = df["Name"].str.split().str[-1]
    df["FamilySize"] = df.groupby("Surname")[idc].transform("count").fillna(1).astype(int)

    # --- low-card categoricals: missing indicator + label encode on union ---
    for c in ["HomePlanet", "Destination"]:
        df[f"{c}_isna"] = df[c].isna().astype(int)
    for c in CAT_LOW:
        u = df[c].fillna("__nan__").astype(str)
        df[c] = u.map({v: i for i, v in enumerate(sorted(u.unique()))}).astype(int)

    # --- drop raw / high-cardinality string columns (kept only as sizes) ---
    df = df.drop(columns=[c for c in (idc, "Cabin", "Name", "Group", "Surname") if c in df.columns])

    Xtr = df.iloc[:ntr].reset_index(drop=True)
    Xte = df.iloc[ntr:].reset_index(drop=True)
    assert Xtr.select_dtypes(include="object").empty, "object cols remain"
    return Xtr, Xte, y_raw
