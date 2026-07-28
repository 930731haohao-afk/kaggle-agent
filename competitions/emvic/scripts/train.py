"""Full pipeline: Feature Engineering + Training for EMVIC.

Eye movement time series (4 channels x 2048 timepoints) -> 37-class classification.
Combines statistical features + PCA + multiple models.
"""
import json
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

data_dir = "competitions/emvic/data"
exp_file = "competitions/emvic/experiments.json"

train = pd.read_csv(f"{data_dir}/train.csv")
test = pd.read_csv(f"{data_dir}/test.csv")

y = train["class"].values
X_raw = train.drop(columns=["class"]).values
X_test_raw = test.drop(columns=["class"]).values

n_classes = 37
class_labels = list(range(1, n_classes + 1))

print(f"Train: {X_raw.shape}, Test: {X_test_raw.shape}, Classes: {n_classes}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """Extract statistical features from 4-channel eye movement time series."""
    n_samples = X.shape[0]
    # Split into 4 channels (2048 each)
    lx = X[:, 0:2048]
    ly = X[:, 2048:4096]
    rx = X[:, 4096:6144]
    ry = X[:, 6144:8192]

    features = []
    for ch_name, ch in [("lx", lx), ("ly", ly), ("rx", rx), ("ry", ry)]:
        # Basic stats
        features.append(ch.mean(axis=1, keepdims=True))
        features.append(ch.std(axis=1, keepdims=True))
        features.append(np.median(ch, axis=1, keepdims=True))
        features.append(ch.min(axis=1, keepdims=True))
        features.append(ch.max(axis=1, keepdims=True))
        features.append((ch.max(axis=1) - ch.min(axis=1)).reshape(-1, 1))  # range

        # Percentiles
        for p in [10, 25, 75, 90]:
            features.append(np.percentile(ch, p, axis=1).reshape(-1, 1))

        # Velocity (first difference)
        vel = np.diff(ch, axis=1)
        features.append(vel.mean(axis=1, keepdims=True))
        features.append(vel.std(axis=1, keepdims=True))
        features.append(np.abs(vel).mean(axis=1, keepdims=True))  # mean absolute velocity
        features.append(np.abs(vel).max(axis=1, keepdims=True))  # max absolute velocity

        # Acceleration (second difference)
        acc = np.diff(vel, axis=1)
        features.append(acc.std(axis=1, keepdims=True))
        features.append(np.abs(acc).mean(axis=1, keepdims=True))

        # Zero-crossing rate of velocity
        zcr = np.sum(np.diff(np.sign(vel), axis=1) != 0, axis=1).reshape(-1, 1)
        features.append(zcr)

        # Segment stats (split time series into 4 quarters)
        quarter = 2048 // 4
        for q in range(4):
            seg = ch[:, q * quarter:(q + 1) * quarter]
            features.append(seg.mean(axis=1, keepdims=True))
            features.append(seg.std(axis=1, keepdims=True))

    # Cross-channel features
    # Left-right eye differences
    features.append((lx.mean(axis=1) - rx.mean(axis=1)).reshape(-1, 1))
    features.append((ly.mean(axis=1) - ry.mean(axis=1)).reshape(-1, 1))
    features.append((lx.std(axis=1) - rx.std(axis=1)).reshape(-1, 1))
    features.append((ly.std(axis=1) - ry.std(axis=1)).reshape(-1, 1))

    # Euclidean distance between eyes (mean)
    eye_dist = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2).mean(axis=1).reshape(-1, 1)
    features.append(eye_dist)

    # Gaze velocity magnitude (left eye)
    lx_vel = np.diff(lx, axis=1)
    ly_vel = np.diff(ly, axis=1)
    gaze_speed = np.sqrt(lx_vel ** 2 + ly_vel ** 2)
    features.append(gaze_speed.mean(axis=1, keepdims=True))
    features.append(gaze_speed.std(axis=1, keepdims=True))
    features.append(gaze_speed.max(axis=1, keepdims=True))

    # Gaze velocity magnitude (right eye)
    rx_vel = np.diff(rx, axis=1)
    ry_vel = np.diff(ry, axis=1)
    gaze_speed_r = np.sqrt(rx_vel ** 2 + ry_vel ** 2)
    features.append(gaze_speed_r.mean(axis=1, keepdims=True))
    features.append(gaze_speed_r.std(axis=1, keepdims=True))
    features.append(gaze_speed_r.max(axis=1, keepdims=True))

    # Fixation detection: count of low-velocity segments
    fix_threshold = 2.0
    n_fixations_l = (gaze_speed < fix_threshold).sum(axis=1).reshape(-1, 1)
    n_fixations_r = (gaze_speed_r < fix_threshold).sum(axis=1).reshape(-1, 1)
    features.append(n_fixations_l)
    features.append(n_fixations_r)

    # Saccade detection: count of high-velocity segments
    sac_threshold = 20.0
    n_saccades_l = (gaze_speed > sac_threshold).sum(axis=1).reshape(-1, 1)
    n_saccades_r = (gaze_speed_r > sac_threshold).sum(axis=1).reshape(-1, 1)
    features.append(n_saccades_l)
    features.append(n_saccades_r)

    return np.hstack(features)


print("Extracting statistical features...")
X_stat_train = extract_stat_features(X_raw)
X_stat_test = extract_stat_features(X_test_raw)
print(f"  Statistical features: {X_stat_train.shape[1]}")

# PCA on raw features
print("Computing PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_test_scaled = scaler.transform(X_test_raw)

n_pca = 100
pca = PCA(n_components=n_pca, random_state=42)
X_pca_train = pca.fit_transform(X_scaled)
X_pca_test = pca.transform(X_test_scaled)
print(f"  PCA features: {n_pca} ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

# Combine stat + PCA features
X_train_all = np.hstack([X_stat_train, X_pca_train])
X_test_all = np.hstack([X_stat_test, X_pca_test])
print(f"\nCombined features: {X_train_all.shape[1]} (stat={X_stat_train.shape[1]} + PCA={n_pca})")

# Scale combined features for models that need it
scaler2 = StandardScaler()
X_train_scaled = scaler2.fit_transform(X_train_all)
X_test_scaled_all = scaler2.transform(X_test_all)

# Save processed data
np.savez(f"{data_dir}/processed.npz",
         X_train=X_train_all, X_test=X_test_all,
         X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled_all,
         y=y)
print("Saved processed.npz")

# ============================================================
# TRAINING
# ============================================================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
experiments = []


def log_experiment(exp: dict):
    experiments.append(exp)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)


# ============================================================
# 1. Random Forest (handles high-dim well)
# ============================================================
print("\n" + "=" * 60)
print("1. Random Forest")
print("=" * 60)

rf_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train_all[train_idx], y[train_idx])
    probs = model.predict_proba(X_train_all[val_idx])
    # Ensure all 37 classes are represented in probs
    full_probs = np.full((len(val_idx), n_classes), 1e-6)
    for i, c in enumerate(model.classes_):
        full_probs[:, c - 1] = probs[:, i]
    full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)
    score = log_loss(y[val_idx], full_probs, labels=class_labels)
    rf_scores.append(score)
    print(f"  Fold {fold+1}: log_loss = {score:.5f}")

print(f"  Mean: {np.mean(rf_scores):.5f} (+/- {np.std(rf_scores):.5f})")

log_experiment({
    "experiment_id": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "RandomForest",
    "features": f"stat+PCA ({X_train_all.shape[1]})",
    "cv_strategy": f"StratifiedKFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in rf_scores],
    "cv_mean": round(np.mean(rf_scores), 5),
    "cv_std": round(np.std(rf_scores), 5),
    "notes": "RF 500 trees, balanced class weight"
})

# ============================================================
# 2. LightGBM
# ============================================================
print("\n" + "=" * 60)
print("2. LightGBM")
print("=" * 60)

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 5,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "seed": 42,
}

lgb_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    # LightGBM needs 0-indexed classes
    y_lgb = y - 1
    dtrain = lgb.Dataset(X_train_all[train_idx], label=y_lgb[train_idx])
    dval = lgb.Dataset(X_train_all[val_idx], label=y_lgb[val_idx], reference=dtrain)

    model = lgb.train(lgb_params, dtrain, num_boost_round=1000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
    probs = model.predict(X_train_all[val_idx])
    # Clip and normalize
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    score = log_loss(y_lgb[val_idx], probs, labels=list(range(n_classes)))
    lgb_scores.append(score)
    print(f"  Fold {fold+1}: log_loss = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean: {np.mean(lgb_scores):.5f} (+/- {np.std(lgb_scores):.5f})")

log_experiment({
    "experiment_id": 2,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "LightGBM",
    "features": f"stat+PCA ({X_train_all.shape[1]})",
    "cv_strategy": f"StratifiedKFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in lgb_scores],
    "cv_mean": round(np.mean(lgb_scores), 5),
    "cv_std": round(np.std(lgb_scores), 5),
    "notes": "LightGBM multiclass, balanced, 0-indexed labels"
})

# ============================================================
# 3. XGBoost
# ============================================================
print("\n" + "=" * 60)
print("3. XGBoost")
print("=" * 60)

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

xgb_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    y_xgb = y - 1  # 0-indexed
    dtrain = xgb.DMatrix(X_train_all[train_idx], label=y_xgb[train_idx])
    dval = xgb.DMatrix(X_train_all[val_idx], label=y_xgb[val_idx])

    model = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                      evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    probs = model.predict(dval)
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    score = log_loss(y_xgb[val_idx], probs, labels=list(range(n_classes)))
    xgb_scores.append(score)
    print(f"  Fold {fold+1}: log_loss = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean: {np.mean(xgb_scores):.5f} (+/- {np.std(xgb_scores):.5f})")

log_experiment({
    "experiment_id": 3,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "XGBoost",
    "features": f"stat+PCA ({X_train_all.shape[1]})",
    "cv_strategy": f"StratifiedKFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in xgb_scores],
    "cv_mean": round(np.mean(xgb_scores), 5),
    "cv_std": round(np.std(xgb_scores), 5),
    "notes": "XGBoost multiclass softprob, 0-indexed labels"
})

# ============================================================
# 4. CatBoost
# ============================================================
print("\n" + "=" * 60)
print("4. CatBoost")
print("=" * 60)

cat_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        l2_leaf_reg=5.0, random_seed=42, verbose=0,
        early_stopping_rounds=50, auto_class_weights="Balanced",
        loss_function="MultiClass",
    )
    model.fit(X_train_all[train_idx], y[train_idx],
              eval_set=(X_train_all[val_idx], y[val_idx]))
    probs = model.predict_proba(X_train_all[val_idx])
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    score = log_loss(y[val_idx], probs, labels=class_labels)
    cat_scores.append(score)
    print(f"  Fold {fold+1}: log_loss = {score:.5f} (best_iter={model.best_iteration_})")

print(f"  Mean: {np.mean(cat_scores):.5f} (+/- {np.std(cat_scores):.5f})")

log_experiment({
    "experiment_id": 4,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "tuned",
    "model": "CatBoost",
    "features": f"stat+PCA ({X_train_all.shape[1]})",
    "cv_strategy": f"StratifiedKFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in cat_scores],
    "cv_mean": round(np.mean(cat_scores), 5),
    "cv_std": round(np.std(cat_scores), 5),
    "notes": "CatBoost MultiClass, balanced weights"
})

# ============================================================
# 5. OOF ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print("5. OOF ENSEMBLE")
print("=" * 60)

oof_rf = np.zeros((len(y), n_classes))
oof_lgb = np.zeros((len(y), n_classes))
oof_xgb = np.zeros((len(y), n_classes))
oof_cat = np.zeros((len(y), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    y_0idx = y - 1

    # RF
    m = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    m.fit(X_train_all[train_idx], y[train_idx])
    p = m.predict_proba(X_train_all[val_idx])
    full_p = np.full((len(val_idx), n_classes), 1e-6)
    for i, c in enumerate(m.classes_):
        full_p[:, c - 1] = p[:, i]
    oof_rf[val_idx] = full_p / full_p.sum(axis=1, keepdims=True)

    # LGB
    dt = lgb.Dataset(X_train_all[train_idx], label=y_0idx[train_idx])
    dv = lgb.Dataset(X_train_all[val_idx], label=y_0idx[val_idx], reference=dt)
    m = lgb.train(lgb_params, dt, num_boost_round=1000,
                  valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = m.predict(X_train_all[val_idx])

    # XGB
    dt = xgb.DMatrix(X_train_all[train_idx], label=y_0idx[train_idx])
    dv = xgb.DMatrix(X_train_all[val_idx], label=y_0idx[val_idx])
    m = xgb.train(xgb_params, dt, num_boost_round=1000,
                  evals=[(dv, "val")], early_stopping_rounds=50, verbose_eval=False)
    oof_xgb[val_idx] = m.predict(dv)

    # CatBoost
    m = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6,
                           l2_leaf_reg=5.0, random_seed=42, verbose=0,
                           early_stopping_rounds=50, auto_class_weights="Balanced",
                           loss_function="MultiClass")
    m.fit(X_train_all[train_idx], y[train_idx],
          eval_set=(X_train_all[val_idx], y[val_idx]))
    oof_cat[val_idx] = m.predict_proba(X_train_all[val_idx])

# Weight search
print("Weight search (RF, LGB, XGB, Cat):")
best_ll = 999
best_w = None
y_0idx = y - 1
for w_rf in [0.0, 0.1, 0.2, 0.3]:
    for w_lgb in np.arange(0.1, 0.8, 0.1):
        for w_xgb in np.arange(0.1, 0.8, 0.1):
            w_cat = 1.0 - w_rf - w_lgb - w_xgb
            if w_cat < 0.05 or w_cat > 0.8:
                continue
            blend = w_rf * oof_rf + w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
            blend = np.clip(blend, 1e-6, 1.0)
            blend = blend / blend.sum(axis=1, keepdims=True)
            ll = log_loss(y_0idx, blend, labels=list(range(n_classes)))
            if ll < best_ll:
                best_ll = ll
                best_w = (round(w_rf, 1), round(w_lgb, 1), round(w_xgb, 1), round(w_cat, 1))

print(f"  Best weights: RF={best_w[0]}, LGB={best_w[1]}, XGB={best_w[2]}, Cat={best_w[3]}")
print(f"  Best ensemble log_loss: {best_ll:.5f}")

# Simple average
simple = (oof_rf + oof_lgb + oof_xgb + oof_cat) / 4
simple = np.clip(simple, 1e-6, 1.0)
simple = simple / simple.sum(axis=1, keepdims=True)
simple_ll = log_loss(y_0idx, simple, labels=list(range(n_classes)))
print(f"  Simple average log_loss: {simple_ll:.5f}")

log_experiment({
    "experiment_id": 5,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "ensemble",
    "model": f"Weighted Ensemble (RF={best_w[0]}, LGB={best_w[1]}, XGB={best_w[2]}, Cat={best_w[3]})",
    "features": f"stat+PCA ({X_train_all.shape[1]})",
    "cv_strategy": f"StratifiedKFold-{n_splits} OOF",
    "cv_mean": round(best_ll, 5),
    "notes": f"Best weights. Simple avg: {simple_ll:.5f}"
})

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT LEADERBOARD")
print("=" * 60)

results = [
    ("RandomForest", np.mean(rf_scores), np.std(rf_scores)),
    ("LightGBM", np.mean(lgb_scores), np.std(lgb_scores)),
    ("XGBoost", np.mean(xgb_scores), np.std(xgb_scores)),
    ("CatBoost", np.mean(cat_scores), np.std(cat_scores)),
    (f"Ensemble ({best_w})", best_ll, 0),
    ("Ensemble (equal)", simple_ll, 0),
]
results.sort(key=lambda x: x[1])

print(f"\n{'Rank':<5} {'Model':<45} {'CV LogLoss':<12} {'Std':<10}")
print("-" * 75)
for rank, (name, mean, std) in enumerate(results, 1):
    print(f"{rank:<5} {name:<45} {mean:<12.5f} {std:<10.5f}")

# Save best weights for submission
np.save(f"{data_dir}/best_weights.npy", np.array(best_w))
print(f"\nBest weights saved for submission.")
