"""
Track A: Tabular ML baseline for digit-recognizer.
Feature engineering + Logistic Regression + LightGBM baselines.
Stratified 5-Fold CV with accuracy metric.
"""
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timezone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ============================================================
# Load Data
# ============================================================
data_dir = "competitions/digit-recognizer/data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

pixel_cols = [c for c in train.columns if c.startswith('pixel')]
X_raw = train[pixel_cols].values.astype(np.float32)
y = train['label'].values
X_test_raw = test[pixel_cols].values.astype(np.float32)

print(f"Raw data: train={X_raw.shape}, test={X_test_raw.shape}")

# ============================================================
# Feature Engineering
# ============================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

def engineer_features(X: np.ndarray) -> np.ndarray:
    """Create engineered features from raw pixel data."""
    n = X.shape[0]
    features = []

    # 1. Normalized pixels (0-1)
    X_norm = X / 255.0
    features.append(X_norm)

    # 2. Remove dead pixels (variance == 0 on train)
    # Handled implicitly — we keep all for simplicity, LightGBM handles zero-variance

    # 3. Image-level statistics
    imgs = X.reshape(n, 28, 28)

    # Total ink (sum of pixel values)
    total_ink = X.sum(axis=1, keepdims=True) / 255.0
    features.append(total_ink)

    # Non-zero pixel count
    nonzero_count = (X > 0).sum(axis=1, keepdims=True).astype(np.float32)
    features.append(nonzero_count)

    # Mean intensity of active pixels
    mean_active = np.where(nonzero_count > 0, total_ink * 255.0 / nonzero_count, 0)
    features.append(mean_active)

    # 4. Row and column projections (28 each)
    row_sums = imgs.sum(axis=2) / 255.0  # (n, 28)
    col_sums = imgs.sum(axis=1) / 255.0  # (n, 28)
    features.append(row_sums)
    features.append(col_sums)

    # 5. Quadrant features (2x2 grid)
    q1 = imgs[:, :14, :14].reshape(n, -1).sum(axis=1, keepdims=True) / 255.0
    q2 = imgs[:, :14, 14:].reshape(n, -1).sum(axis=1, keepdims=True) / 255.0
    q3 = imgs[:, 14:, :14].reshape(n, -1).sum(axis=1, keepdims=True) / 255.0
    q4 = imgs[:, 14:, 14:].reshape(n, -1).sum(axis=1, keepdims=True) / 255.0
    features.append(np.hstack([q1, q2, q3, q4]))

    # 6. Horizontal and vertical symmetry
    h_sym = np.abs(imgs - imgs[:, :, ::-1]).sum(axis=(1, 2), keepdims=False) / 255.0
    v_sym = np.abs(imgs - imgs[:, ::-1, :]).sum(axis=(1, 2), keepdims=False) / 255.0
    features.append(h_sym.reshape(-1, 1))
    features.append(v_sym.reshape(-1, 1))

    # 7. Center of mass
    rows = np.arange(28).reshape(1, 28, 1)
    cols = np.arange(28).reshape(1, 1, 28)
    total = imgs.sum(axis=(1, 2), keepdims=True) + 1e-8
    center_row = (imgs * rows).sum(axis=(1, 2), keepdims=True) / total
    center_col = (imgs * cols).sum(axis=(1, 2), keepdims=True) / total
    features.append(center_row.reshape(-1, 1))
    features.append(center_col.reshape(-1, 1))

    # 8. Bounding box features
    has_ink = (imgs > 0)
    row_any = has_ink.any(axis=2)  # (n, 28)
    col_any = has_ink.any(axis=1)  # (n, 28)

    top = np.argmax(row_any, axis=1).reshape(-1, 1).astype(np.float32)
    bottom = (27 - np.argmax(row_any[:, ::-1], axis=1)).reshape(-1, 1).astype(np.float32)
    left = np.argmax(col_any, axis=1).reshape(-1, 1).astype(np.float32)
    right = (27 - np.argmax(col_any[:, ::-1], axis=1)).reshape(-1, 1).astype(np.float32)
    height = bottom - top + 1
    width = right - left + 1
    aspect_ratio = np.where(height > 0, width / height, 0)
    features.append(np.hstack([top, bottom, left, right, height, width, aspect_ratio]))

    return np.hstack(features)

print("Engineering features for train...")
X_eng = engineer_features(X_raw)
print(f"Train engineered: {X_eng.shape}")

print("Engineering features for test...")
X_test_eng = engineer_features(X_test_raw)
print(f"Test engineered: {X_test_eng.shape}")

# ============================================================
# CV Setup
# ============================================================
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
experiments = []

def log_experiment(exp: dict):
    experiments.append(exp)
    print(f"\n  CV Mean: {exp['cv_mean']:.5f} +/- {exp['cv_std']:.5f}")
    print(f"  Per-fold: {[f'{s:.5f}' for s in exp['cv_scores']]}")

# ============================================================
# Baseline 1: Majority Class
# ============================================================
print("\n" + "=" * 70)
print("BASELINE 1: Majority Class Predictor")
print("=" * 70)
from collections import Counter
majority_class = Counter(y).most_common(1)[0][0]
majority_acc = (y == majority_class).mean()
print(f"Majority class: {majority_class}, accuracy: {majority_acc:.5f}")
experiments.append({
    "experiment_id": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "majority_class",
    "features": "none",
    "params": {"majority_class": int(majority_class)},
    "cv_strategy": f"{N_FOLDS}-fold-stratified",
    "cv_scores": [round(majority_acc, 5)] * N_FOLDS,
    "cv_mean": round(majority_acc, 5),
    "cv_std": 0.0,
    "notes": "Trivial baseline — always predict most common digit"
})

# ============================================================
# Baseline 2: Logistic Regression on normalized pixels
# ============================================================
print("\n" + "=" * 70)
print("BASELINE 2: Logistic Regression (normalized pixels)")
print("=" * 70)
t0 = time.time()
lr_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_eng, y)):
    X_tr, X_val = X_eng[train_idx], X_eng[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(max_iter=300, solver='lbfgs',
                               C=1.0, random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_val_s)
    acc = accuracy_score(y_val, pred)
    lr_scores.append(acc)
    print(f"  Fold {fold+1}: {acc:.5f}")

lr_time = time.time() - t0
log_experiment({
    "experiment_id": 2,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "LogisticRegression",
    "features": "engineered_v1",
    "params": {"C": 1.0, "max_iter": 300, "solver": "lbfgs"},
    "cv_strategy": f"{N_FOLDS}-fold-stratified",
    "cv_scores": [round(s, 5) for s in lr_scores],
    "cv_mean": round(np.mean(lr_scores), 5),
    "cv_std": round(np.std(lr_scores), 5),
    "training_time_sec": round(lr_time, 1),
    "notes": "Logistic Regression on engineered features (scaled)"
})

# ============================================================
# Baseline 3: LightGBM with default params
# ============================================================
print("\n" + "=" * 70)
print("BASELINE 3: LightGBM (default params, engineered features)")
print("=" * 70)
t0 = time.time()
lgb_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_eng, y)):
    X_tr, X_val = X_eng[train_idx], X_eng[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        'objective': 'multiclass',
        'num_class': 10,
        'metric': 'multi_logloss',
        'learning_rate': 0.1,
        'num_leaves': 127,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, pred)
    lgb_scores.append(acc)
    print(f"  Fold {fold+1}: {acc:.5f} (best iter: {model.best_iteration})")

lgb_time = time.time() - t0
log_experiment({
    "experiment_id": 3,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "LightGBM-default",
    "features": "engineered_v1",
    "params": {"learning_rate": 0.1, "num_leaves": 127, "num_boost_round": 500,
               "early_stopping": 50},
    "cv_strategy": f"{N_FOLDS}-fold-stratified",
    "cv_scores": [round(s, 5) for s in lgb_scores],
    "cv_mean": round(np.mean(lgb_scores), 5),
    "cv_std": round(np.std(lgb_scores), 5),
    "training_time_sec": round(lgb_time, 1),
    "notes": "LightGBM default on engineered features"
})

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("BASELINE SUMMARY")
print("=" * 70)
print(f"{'Model':<30} {'CV Accuracy':>12} {'Std':>8} {'Time':>8}")
print("-" * 60)
for exp in experiments:
    t = exp.get('training_time_sec', 0)
    print(f"{exp['model']:<30} {exp['cv_mean']:>12.5f} {exp['cv_std']:>8.5f} {t:>7.1f}s")

# Save experiments
exp_file = "competitions/digit-recognizer/experiments.json"
with open(exp_file, 'w') as f:
    json.dump(experiments, f, indent=2)
print(f"\nExperiments saved to {exp_file}")
