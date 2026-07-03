"""
Feature engineering — playground-series-s3e1 (California Housing)

EDA findings driving these choices:
- MedInc dominates target correlation (0.70); Latitude/Longitude are strongly
  anti-correlated (-0.94) with each other but individually only weakly linear
  with target -> geo signal is likely non-linear (location clusters / distance
  to major cities), so add clustering + distance features instead of relying
  on raw lat/long linearity.
- AveRooms, AveBedrms, Population, AveOccup have extreme outliers (e.g. AveOccup
  max ~503, AveRooms max ~57 in test) -> add ratio features that normalize by
  household count, and log1p transforms to tame skew/outliers for tree models
  (trees are outlier-robust, but ratios still add signal trees can't derive
  from raw division splits easily at high cardinality).
- Population/AveOccup recovers implied household count; AveRooms/AveOccup gives
  rooms-per-person; AveBedrms/AveRooms gives bedroom ratio.
- No missing values, no train/test distribution shift -> no imputation/encoding
  needed, fit-on-train-only rules still followed for cluster centers.
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans

COMPETITION_DIR = "competitions/playground-series-s3e1"
TARGET_COL = "MedHouseVal"
ID_COL = "id"

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

orig_features = [c for c in train.columns if c not in (TARGET_COL, ID_COL)]

# Major CA population centers (fixed domain knowledge, no data leakage)
CITIES = {
    "LA": (34.05, -118.24),
    "SF": (37.77, -122.41),
    "SanDiego": (32.72, -117.16),
    "Sacramento": (38.58, -121.49),
    "SanJose": (37.34, -121.89),
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # household / ratio features
    df["households"] = df["Population"] / df["AveOccup"].replace(0, np.nan)
    df["households"] = df["households"].fillna(df["households"].median())
    df["bedroom_ratio"] = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
    df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"].replace(0, np.nan)
    df["bedroom_ratio"] = df["bedroom_ratio"].fillna(df["bedroom_ratio"].median())
    df["rooms_per_person"] = df["rooms_per_person"].fillna(df["rooms_per_person"].median())

    # log1p transforms to tame skew/outliers
    for c in ["AveRooms", "AveBedrms", "Population", "AveOccup", "households"]:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))

    # distance to major cities (Euclidean on lat/long, good proxy in this dataset)
    for name, (lat, lon) in CITIES.items():
        df[f"dist_{name}"] = np.sqrt((df["Latitude"] - lat) ** 2 + (df["Longitude"] - lon) ** 2)
    df["dist_nearest_city"] = df[[f"dist_{n}" for n in CITIES]].min(axis=1)

    # simple lat/long interaction
    df["lat_long"] = df["Latitude"] * df["Longitude"]
    return df


train_fe = add_features(train)
test_fe = add_features(test)

# Geo clustering: fit KMeans on train coordinates only, transform both
kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
train_fe["geo_cluster"] = kmeans.fit_predict(train[["Latitude", "Longitude"]])
test_fe["geo_cluster"] = kmeans.predict(test[["Latitude", "Longitude"]])

new_features = [
    "households", "bedroom_ratio", "rooms_per_person",
    "log_AveRooms", "log_AveBedrms", "log_Population", "log_AveOccup", "log_households",
    "dist_LA", "dist_SF", "dist_SanDiego", "dist_Sacramento", "dist_SanJose", "dist_nearest_city",
    "lat_long", "geo_cluster",
]
all_features = orig_features + new_features

# --- Validation checks ---
assert set(all_features) - set(train_fe.columns) == set()
assert set(all_features) - set(test_fe.columns) == set()
assert train_fe[all_features].isnull().sum().sum() == 0, "NaNs in train features"
assert test_fe[all_features].isnull().sum().sum() == 0, "NaNs in test features"
for c in all_features:
    assert np.issubdtype(train_fe[c].dtype, np.number), f"{c} not numeric"

out_train = train_fe[[ID_COL] + all_features + [TARGET_COL]]
out_test = test_fe[[ID_COL] + all_features]

out_train.to_csv(os.path.join(data_dir, "train_processed.csv"), index=False)
out_test.to_csv(os.path.join(data_dir, "test_processed.csv"), index=False)

print(f"Original features: {len(orig_features)}")
print(f"New features: {len(new_features)}")
print(f"Total features: {len(all_features)}")
print(f"Train processed shape: {out_train.shape}")
print(f"Test processed shape: {out_test.shape}")

# Leakage sanity check: correlation of new features with target should not be suspiciously perfect
corr = out_train[new_features + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
print("\nNew feature correlation with target:\n", corr)
