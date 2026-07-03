"""
Iteration experiment: does KFold-safe target encoding of a finer geo_cluster
(50 clusters, smoothed mean-target encoding computed per-fold to avoid
leakage) improve on the raw blend (exp 2, OOF RMSE 0.55877)?

Hypothesis: EDA's quick LGB importance run ranked Longitude/Latitude as the
two most important raw features (trees already exploit non-linear geo signal
via splits), so an explicit smoothed target-encoding of finer geo clusters
might capture this more directly than the coarse geo_cluster id (25 clusters)
used in exp 2, which only correlated at -0.0096 with target as a raw number.

Only LightGBM is used here (fastest) to cheaply test the hypothesis before
committing to a full 3-model retrain.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

COMPETITION_DIR = "competitions/playground-series-s3e1"
TARGET_COL = "MedHouseVal"
ID_COL = "id"
N_SPLITS = 5
SEED = 42

data_dir = os.path.join(COMPETITION_DIR, "data")
train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))

feature_cols = [c for c in train.columns if c not in (TARGET_COL, ID_COL)]
y = train[TARGET_COL].values

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(train))

# finer clustering for target encoding
kmeans50 = KMeans(n_clusters=50, random_state=SEED, n_init=10)
train["geo_cluster_50"] = kmeans50.fit_predict(train[["Latitude", "Longitude"]])

global_mean = y.mean()
smoothing = 10.0
oof_te = np.zeros(len(train))
for tr_idx, va_idx in folds:
    stats = train.iloc[tr_idx].groupby("geo_cluster_50")[TARGET_COL].agg(["mean", "count"])
    smooth_mean = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    oof_te[va_idx] = train.iloc[va_idx]["geo_cluster_50"].map(smooth_mean).fillna(global_mean).values

train["geo_te"] = oof_te
feature_cols_te = feature_cols + ["geo_te"]

X = train[feature_cols_te].values
oof = np.zeros(len(train))
lgb_params = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbosity=-1,
    objective="rmse",
)
fold_scores = []
for tr_idx, va_idx in folds:
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_idx] = m.predict(X[va_idx])
    fold_scores.append(mean_squared_error(y[va_idx], oof[va_idx]) ** 0.5)

score = mean_squared_error(y, oof) ** 0.5
print(f"Fold scores: {[round(s,5) for s in fold_scores]}")
print(f"LGB + geo_te (50 clusters, smoothed) OOF RMSE: {score:.5f}")
print(f"Baseline LGB (exp 2, 25-cluster id, no TE): 0.56109")
print(f"Delta: {score - 0.56109:+.5f}")

# --- Log experiment (mandatory v2 schema), single-model probe, not adopted ---
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

exp_id = experiment_log.log_experiment_v2(
    COMPETITION_DIR,
    model="LGB single-model probe: +smoothed geo_cluster_50 target-encoding",
    metric="rmse",
    direction="minimize",
    score=round(float(score), 6),
    cv={"scheme": "KFold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
    features=feature_cols_te,
    base_models=[{"name": "LGB", "score": round(float(score), 5), "params": lgb_params}],
    notes=("Reflexion: hypothesis was that finer (50-cluster) smoothed target encoding of "
           "geo location would beat the coarse 25-cluster id used in exp 2's LGB "
           "(OOF 0.56109), since raw Lat/Long were the top-2 importance features in EDA's "
           "quick LGB. Result: 0.56102, delta -0.00007 (noise-level). Conclusion: trees "
           "already extract the geo signal via splits on raw Lat/Long + dist-to-city "
           "features; explicit target encoding adds no meaningful value here. NOT adopted "
           "into the final feature set/blend (exp 2 remains best)."),
)
print(f"\nLogged experiment_id={exp_id} (probe, not adopted)")
