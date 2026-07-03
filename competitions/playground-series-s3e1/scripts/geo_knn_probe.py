"""
Iteration round 3 (candidate) — finer geo features: KNN-distance aggregates +
coastal distance, as raw numeric features (NOT target-encoded, learning from
round-1-era exp #3: trees already exploit raw geo features via splits, but an
explicit *target*-encoding of geo groupings added nothing; this round instead
tests whether additional *non-target* geometric features -- local block
density (mean distance to k nearest neighbors in lat/long space) and distance
to the CA coastline -- carry information the existing city-distance features
don't).

No leakage: KNN neighbor search and coastal-anchor distances use ONLY
Latitude/Longitude coordinates (never the target), computed over the combined
train+test coordinate set, so it's safe to compute before the CV split.

Only LightGBM (fastest) is used to cheaply probe the hypothesis before
committing to a full 3-model retrain + reweighting, mirroring exp #3's
methodology.
"""

import pandas as pd
import numpy as np
import os
import importlib.util
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

COMPETITION_DIR = "competitions/playground-series-s3e1"
TARGET_COL = "MedHouseVal"
ID_COL = "id"
N_SPLITS = 5
SEED = 42
K_NEIGHBORS = 10

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
test = pd.read_csv(os.path.join(data_dir, "test_processed.csv"))

feature_cols = [c for c in train.columns if c not in (TARGET_COL, ID_COL)]
y = train[TARGET_COL].values

# --- Approximate CA coastline anchor points (fixed domain knowledge, no
#     leakage -- these are public geographic facts, not derived from data) ---
COASTLINE = [
    (32.55, -117.13),  # near San Diego / border
    (33.17, -117.35),  # Oceanside
    (33.76, -118.20),  # Long Beach
    (34.02, -118.49),  # Santa Monica
    (34.42, -119.70),  # Santa Barbara
    (35.37, -120.85),  # San Luis Obispo coast
    (36.60, -121.90),  # Monterey
    (37.62, -122.49),  # SF / Pacifica
    (38.35, -123.05),  # Bodega Bay
    (39.44, -123.80),  # Fort Bragg
    (40.80, -124.16),  # Eureka
    (41.76, -124.20),  # Crescent City
]


def coastal_distance(lat, lon):
    dists = np.stack([np.sqrt((lat - a) ** 2 + (lon - b) ** 2) for a, b in COASTLINE], axis=1)
    return dists.min(axis=1)


all_coords = np.vstack([
    train[["Latitude", "Longitude"]].values,
    test[["Latitude", "Longitude"]].values,
])

nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)  # +1 to exclude self
nn.fit(all_coords)
dist_all, _ = nn.kneighbors(all_coords)
knn_mean_dist = dist_all[:, 1:].mean(axis=1)  # exclude self (col 0, dist=0)

n_train = len(train)
train["knn_mean_dist_10"] = knn_mean_dist[:n_train]
test["knn_mean_dist_10"] = knn_mean_dist[n_train:]

train["coastal_dist"] = coastal_distance(train["Latitude"].values, train["Longitude"].values)
test["coastal_dist"] = coastal_distance(test["Latitude"].values, test["Longitude"].values)

new_geo_features = ["knn_mean_dist_10", "coastal_dist"]
feature_cols_new = feature_cols + new_geo_features

corr = train[new_geo_features + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
print("New feature correlation with target:\n", corr)

X = train[feature_cols_new].values
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(X))

lgb_params = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbosity=-1,
    objective="rmse",
)

oof = np.zeros(len(train))
fold_scores = []
for tr_idx, va_idx in folds:
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = m.predict(X[va_idx])
    fold_scores.append(mean_squared_error(y[va_idx], oof[va_idx]) ** 0.5)

score = mean_squared_error(y, oof) ** 0.5
print(f"Fold scores: {[round(s,5) for s in fold_scores]}")
print(f"LGB + knn_mean_dist_10 + coastal_dist OOF RMSE: {score:.5f}")
print(f"Baseline LGB (exp 2, no new geo feats): 0.56109")
print(f"Delta: {score - 0.56109:+.5f}")

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

adopted = score < 0.56109 - 0.0005  # require a real, non-noise-level gain to adopt

exp_id = experiment_log.log_experiment_v2(
    COMPETITION_DIR,
    model="LGB single-model probe: +knn_mean_dist_10 +coastal_dist",
    metric="rmse",
    direction="minimize",
    score=round(float(score), 6),
    cv={"scheme": "KFold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
    features=feature_cols_new,
    base_models=[{"name": "LGB", "score": round(float(score), 5), "params": lgb_params}],
    notes=(f"Iteration round 3 candidate: KNN(k={K_NEIGHBORS}) mean-distance local-density "
           f"feature + coastal-distance feature (raw numeric, not target-encoded), computed "
           f"over combined train+test coords (no target leakage). Baseline LGB (exp 2) was "
           f"0.56109. Result: {score:.5f} ({'ADOPTED' if adopted else 'NOT adopted -- noise-level or worse'})."),
)
print(f"\nLogged experiment_id={exp_id} ({'adopted' if adopted else 'probe only, not adopted'})")

if adopted:
    train.to_csv(os.path.join(data_dir, "train_processed_v2.csv"), index=False)
    test.to_csv(os.path.join(data_dir, "test_processed_v2.csv"), index=False)
    print("Saved train_processed_v2.csv / test_processed_v2.csv for full blend retrain.")
