"""Generate EMVIC submission: retrain best models on full data, predict test."""
import json
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_dir = "competitions/emvic/data"
sub_dir = "competitions/emvic/submissions"

train = pd.read_csv(f"{data_dir}/train.csv")
test = pd.read_csv(f"{data_dir}/test.csv")

y = train["class"].values
X_raw = train.drop(columns=["class"]).values
X_test_raw = test.drop(columns=["class"]).values

n_classes = 37


# ============================================================
# FEATURE ENGINEERING (same as train.py)
# ============================================================
def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """Extract statistical features from 4-channel eye movement time series."""
    lx = X[:, 0:2048]
    ly = X[:, 2048:4096]
    rx = X[:, 4096:6144]
    ry = X[:, 6144:8192]

    features = []
    for ch_name, ch in [("lx", lx), ("ly", ly), ("rx", rx), ("ry", ry)]:
        features.append(ch.mean(axis=1, keepdims=True))
        features.append(ch.std(axis=1, keepdims=True))
        features.append(np.median(ch, axis=1, keepdims=True))
        features.append(ch.min(axis=1, keepdims=True))
        features.append(ch.max(axis=1, keepdims=True))
        features.append((ch.max(axis=1) - ch.min(axis=1)).reshape(-1, 1))

        for p in [10, 25, 75, 90]:
            features.append(np.percentile(ch, p, axis=1).reshape(-1, 1))

        vel = np.diff(ch, axis=1)
        features.append(vel.mean(axis=1, keepdims=True))
        features.append(vel.std(axis=1, keepdims=True))
        features.append(np.abs(vel).mean(axis=1, keepdims=True))
        features.append(np.abs(vel).max(axis=1, keepdims=True))

        acc = np.diff(vel, axis=1)
        features.append(acc.std(axis=1, keepdims=True))
        features.append(np.abs(acc).mean(axis=1, keepdims=True))

        zcr = np.sum(np.diff(np.sign(vel), axis=1) != 0, axis=1).reshape(-1, 1)
        features.append(zcr)

        quarter = 2048 // 4
        for q in range(4):
            seg = ch[:, q * quarter:(q + 1) * quarter]
            features.append(seg.mean(axis=1, keepdims=True))
            features.append(seg.std(axis=1, keepdims=True))

    features.append((lx.mean(axis=1) - rx.mean(axis=1)).reshape(-1, 1))
    features.append((ly.mean(axis=1) - ry.mean(axis=1)).reshape(-1, 1))
    features.append((lx.std(axis=1) - rx.std(axis=1)).reshape(-1, 1))
    features.append((ly.std(axis=1) - ry.std(axis=1)).reshape(-1, 1))

    eye_dist = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2).mean(axis=1).reshape(-1, 1)
    features.append(eye_dist)

    lx_vel = np.diff(lx, axis=1)
    ly_vel = np.diff(ly, axis=1)
    gaze_speed = np.sqrt(lx_vel ** 2 + ly_vel ** 2)
    features.append(gaze_speed.mean(axis=1, keepdims=True))
    features.append(gaze_speed.std(axis=1, keepdims=True))
    features.append(gaze_speed.max(axis=1, keepdims=True))

    rx_vel = np.diff(rx, axis=1)
    ry_vel = np.diff(ry, axis=1)
    gaze_speed_r = np.sqrt(rx_vel ** 2 + ry_vel ** 2)
    features.append(gaze_speed_r.mean(axis=1, keepdims=True))
    features.append(gaze_speed_r.std(axis=1, keepdims=True))
    features.append(gaze_speed_r.max(axis=1, keepdims=True))

    fix_threshold = 2.0
    n_fixations_l = (gaze_speed < fix_threshold).sum(axis=1).reshape(-1, 1)
    n_fixations_r = (gaze_speed_r < fix_threshold).sum(axis=1).reshape(-1, 1)
    features.append(n_fixations_l)
    features.append(n_fixations_r)

    sac_threshold = 20.0
    n_saccades_l = (gaze_speed > sac_threshold).sum(axis=1).reshape(-1, 1)
    n_saccades_r = (gaze_speed_r > sac_threshold).sum(axis=1).reshape(-1, 1)
    features.append(n_saccades_l)
    features.append(n_saccades_r)

    return np.hstack(features)


print("Extracting features...")
X_stat_train = extract_stat_features(X_raw)
X_stat_test = extract_stat_features(X_test_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(n_components=100, random_state=42)
X_pca_train = pca.fit_transform(X_scaled)
X_pca_test = pca.transform(X_test_scaled)

X_train_all = np.hstack([X_stat_train, X_pca_train])
X_test_all = np.hstack([X_stat_test, X_pca_test])
print(f"Features: {X_train_all.shape[1]} (stat={X_stat_train.shape[1]} + PCA=100)")

# ============================================================
# RETRAIN ON FULL DATA
# ============================================================
# Best weights from CV: LGB=0.7, XGB=0.1, Cat=0.2 (RF=0.0)
w_lgb, w_xgb, w_cat = 0.7, 0.1, 0.2

print("\nRetraining on full data...")

# LightGBM (0-indexed)
print("  Training LightGBM...")
y_lgb = y - 1
lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "is_unbalance": True,
    "seed": 42,
}
dtrain = lgb.Dataset(X_train_all, label=y_lgb)
# Use ~900 iterations (average of best_iter from CV folds)
lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=900)
test_probs_lgb = lgb_model.predict(X_test_all)

# XGBoost (0-indexed)
print("  Training XGBoost...")
xgb_params = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "seed": 42,
    "verbosity": 0,
}
dtrain_xgb = xgb.DMatrix(X_train_all, label=y_lgb)
dtest_xgb = xgb.DMatrix(X_test_all)
# Use ~775 iterations (average of best_iter from CV)
xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=775)
test_probs_xgb = xgb_model.predict(dtest_xgb)

# CatBoost (original labels)
print("  Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=998, learning_rate=0.05, depth=6,
    l2_leaf_reg=5.0, random_seed=42, verbose=0,
    auto_class_weights="Balanced", loss_function="MultiClass",
)
cat_model.fit(X_train_all, y)
test_probs_cat = cat_model.predict_proba(X_test_all)

# ============================================================
# WEIGHTED ENSEMBLE
# ============================================================
print("\nBlending predictions...")
# LGB and XGB output 37 cols (0-indexed), Cat outputs 37 cols (class order from model)
# All are (326, 37) — combine them
ensemble = w_lgb * test_probs_lgb + w_xgb * test_probs_xgb + w_cat * test_probs_cat
ensemble = np.clip(ensemble, 0.001, 1.0)
ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)

# ============================================================
# FORMAT SUBMISSION
# ============================================================
# Format: 326 rows x 37 columns of probabilities
# NO header, NO ID column
# Columns = classes 1-37 (probability for each class)
# Minimum probability ~0.001

print(f"\nSubmission shape: {ensemble.shape}")
print(f"Row sums: min={ensemble.sum(axis=1).min():.6f}, max={ensemble.sum(axis=1).max():.6f}")
print(f"Prob range: [{ensemble.min():.6f}, {ensemble.max():.6f}]")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
cv_score = 0.902  # Best CV from training
sub_file = f"{sub_dir}/submission_ensemble3_cv{cv_score:.3f}_{ts}.csv"

# Save without header or index
np.savetxt(sub_file, ensemble, delimiter=",", fmt="%.6f")

print(f"\nSaved: {sub_file}")

# Validate against benchmark
benchmark = pd.read_csv(f"{data_dir}/uniform_benchmark.csv", header=None)
print(f"Benchmark shape: {benchmark.shape}")
print(f"Submission matches benchmark format: {ensemble.shape == benchmark.shape}")

print("\nDone!")
