"""Full pipeline: Audio Feature Extraction + Training for Whale Detection.

2-second underwater audio clips -> binary classification (whale upcall or not).
Extracts mel spectrogram, MFCC, spectral, and temporal features.
"""
import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import librosa
import numpy as np
import soundfile as sf
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

data_dir = "competitions/the-icml-2013-whale-challenge-right-whale-redux/data"
exp_file = "competitions/the-icml-2013-whale-challenge-right-whale-redux/experiments.json"
train_dir = f"{data_dir}/train/train2"
test_dir = f"{data_dir}/test/test2"

N_FFT = 256
HOP_LENGTH = 64
N_MELS = 64
N_MFCC = 20
SR = 2000


def log_experiment(entry):
    with open(exp_file, "r") as f:
        experiments = json.load(f)
    experiments.append(entry)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


def extract_features(filepath: str) -> np.ndarray:
    """Extract audio features from a single clip."""
    data, sr = sf.read(filepath)

    # Pad short clips to 4000 samples (2 seconds at 2000 Hz)
    if len(data) < 4000:
        data = np.pad(data, (0, 4000 - len(data)), mode="constant")

    features = []

    # 1. Basic waveform stats (5)
    features.append(np.sqrt(np.mean(data**2)))  # RMS
    features.append(np.abs(data).max())  # max absolute amplitude
    features.append(np.mean(data))  # DC offset
    features.append(np.std(data))  # standard deviation
    features.append(np.sum(np.diff(np.sign(data)) != 0) / len(data))  # zero-crossing rate

    # 2. Frequency domain (10)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    power = np.abs(fft) ** 2

    # Band energies (whale upcalls: 50-200 Hz)
    bands = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 1000)]
    for flo, fhi in bands:
        mask = (freqs >= flo) & (freqs < fhi)
        features.append(np.log1p(power[mask].sum()))

    # Spectral centroid, bandwidth, rolloff
    total_power = power.sum()
    if total_power > 0:
        spectral_centroid = np.sum(freqs * power) / total_power
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / total_power)
    else:
        spectral_centroid = 0
        spectral_bandwidth = 0
    features.append(spectral_centroid)
    features.append(spectral_bandwidth)
    # Spectral rolloff (85% energy)
    cumpower = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumpower, 0.85 * cumpower[-1])
    features.append(freqs[min(rolloff_idx, len(freqs) - 1)])

    # 3. Mel spectrogram stats (64*4 = 256 features: mean, std, min, max per mel band)
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(mel_db.mean(axis=1))  # mean per band
    features.extend(mel_db.std(axis=1))   # std per band
    features.extend(mel_db.min(axis=1))   # min per band
    features.extend(mel_db.max(axis=1))   # max per band

    # 4. MFCC stats (20*4 = 80 features: mean, std, min, max per coefficient)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend(mfcc.mean(axis=1))
    features.extend(mfcc.std(axis=1))
    features.extend(mfcc.min(axis=1))
    features.extend(mfcc.max(axis=1))

    # 5. Delta MFCC (first derivative, 20*2 = 40)
    try:
        delta_mfcc = librosa.feature.delta(mfcc, width=min(9, mfcc.shape[1] - (mfcc.shape[1] % 2 == 0)))
        features.extend(delta_mfcc.mean(axis=1))
        features.extend(delta_mfcc.std(axis=1))
    except Exception:
        features.extend([0.0] * N_MFCC * 2)

    # 6. Temporal envelope features (8)
    # Split into 4 quarters and compute energy per quarter
    quarter = len(data) // 4
    for q in range(4):
        seg = data[q * quarter:(q + 1) * quarter]
        features.append(np.sqrt(np.mean(seg**2)))
        features.append(np.abs(seg).max())

    # 7. Energy ratio features (3)
    # Whale upcall band (50-200 Hz) vs rest
    whale_mask = (freqs >= 50) & (freqs <= 200)
    other_mask = ~whale_mask & (freqs > 0)
    whale_energy = power[whale_mask].sum()
    other_energy = power[other_mask].sum()
    features.append(np.log1p(whale_energy))
    features.append(np.log1p(other_energy))
    features.append(whale_energy / (other_energy + 1e-10))  # whale/other ratio

    return np.array(features, dtype=np.float32)


# Load or extract train features
train_files = sorted(os.listdir(train_dir))
n_train = len(train_files)
labels = np.array([int(f.split("_")[-1].replace(".aif", "")) for f in train_files])

train_cache = f"{data_dir}/train_features.npz"
if os.path.exists(train_cache):
    print("Loading cached train features...")
    cached = np.load(train_cache, allow_pickle=True)
    X_train = cached["X"]
    n_features = X_train.shape[1]
    print(f"  Loaded: {X_train.shape}, {labels.sum()} positive, {(1-labels).sum()} negative")
else:
    print("Extracting train features...")
    print(f"  Train files: {n_train}, Positive: {labels.sum()}, Negative: {(1-labels).sum()}")
    sample_feat = extract_features(f"{train_dir}/{train_files[0]}")
    n_features = len(sample_feat)
    print(f"  Features per clip: {n_features}")
    X_train = np.zeros((n_train, n_features), dtype=np.float32)
    for i, fname in enumerate(train_files):
        X_train[i] = extract_features(f"{train_dir}/{fname}")
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n_train} done...")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(train_cache, X=X_train, y=labels, filenames=np.array(train_files))
    print(f"  Saved train_features.npz")

print(f"  Train features shape: {X_train.shape}")

# Load or extract test features
test_files = sorted(os.listdir(test_dir))
n_test = len(test_files)

test_cache = f"{data_dir}/test_features.npz"
if os.path.exists(test_cache):
    print("\nLoading cached test features...")
    cached = np.load(test_cache, allow_pickle=True)
    X_test = cached["X"]
    print(f"  Loaded: {X_test.shape}")
else:
    print("\nExtracting test features...")
    X_test = np.zeros((n_test, n_features), dtype=np.float32)
    for i, fname in enumerate(test_files):
        X_test[i] = extract_features(f"{test_dir}/{fname}")
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n_test} done...")
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(test_cache, X=X_test, filenames=np.array(test_files))
    print(f"  Saved test_features.npz")

print(f"  Test features shape: {X_test.shape}")

y = labels
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ============================================================
# 1. LightGBM
# ============================================================
print("\n" + "=" * 60)
print("1. LightGBM")
print("=" * 60)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "is_unbalance": True,
    "seed": 42,
    "verbose": -1,
}

lgb_scores = []
oof_lgb = np.zeros(n_train)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    dtrain = lgb.Dataset(X_train[train_idx], label=y[train_idx])
    dval = lgb.Dataset(X_train[val_idx], label=y[val_idx], reference=dtrain)
    model = lgb.train(lgb_params, dtrain, num_boost_round=2000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds = model.predict(X_train[val_idx])
    oof_lgb[val_idx] = preds
    score = roc_auc_score(y[val_idx], preds)
    lgb_scores.append(score)
    print(f"  Fold {fold+1}: AUC = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean: {np.mean(lgb_scores):.5f} (+/- {np.std(lgb_scores):.5f})")
print(f"  OOF AUC: {roc_auc_score(y, oof_lgb):.5f}")

# Feature importance
importance = model.feature_importance(importance_type="gain")
top_idx = np.argsort(importance)[::-1][:15]
print("\n  Top 15 features by gain:")
for idx in top_idx:
    print(f"    Feature {idx:3d}: {importance[idx]:.1f}")

log_experiment({
    "experiment_id": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "LightGBM",
    "features": f"audio features ({n_features})",
    "cv_strategy": f"StratifiedKFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in lgb_scores],
    "cv_mean": round(np.mean(lgb_scores), 5),
    "cv_std": round(np.std(lgb_scores), 5),
    "oof_auc": round(roc_auc_score(y, oof_lgb), 5),
    "notes": "LightGBM binary, unbalanced, audio features (mel+MFCC+spectral)"
})

# ============================================================
# 2. XGBoost
# ============================================================
print("\n" + "=" * 60)
print("2. XGBoost")
print("=" * 60)

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": (1 - y.mean()) / y.mean(),
    "seed": 42,
    "verbosity": 0,
}

xgb_scores = []
oof_xgb = np.zeros(n_train)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    dtrain = xgb.DMatrix(X_train[train_idx], label=y[train_idx])
    dval = xgb.DMatrix(X_train[val_idx], label=y[val_idx])
    model = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                      evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False)
    preds = model.predict(dval)
    oof_xgb[val_idx] = preds
    score = roc_auc_score(y[val_idx], preds)
    xgb_scores.append(score)
    print(f"  Fold {fold+1}: AUC = {score:.5f} (best_iter={model.best_iteration})")

print(f"  Mean: {np.mean(xgb_scores):.5f} (+/- {np.std(xgb_scores):.5f})")
print(f"  OOF AUC: {roc_auc_score(y, oof_xgb):.5f}")

log_experiment({
    "experiment_id": 2,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "XGBoost",
    "features": f"audio features ({n_features})",
    "cv_strategy": f"StratifiedKFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in xgb_scores],
    "cv_mean": round(np.mean(xgb_scores), 5),
    "cv_std": round(np.std(xgb_scores), 5),
    "oof_auc": round(roc_auc_score(y, oof_xgb), 5),
    "notes": "XGBoost binary, scale_pos_weight, audio features"
})

# ============================================================
# 3. CatBoost
# ============================================================
print("\n" + "=" * 60)
print("3. CatBoost")
print("=" * 60)

cat_scores = []
oof_cat = np.zeros(n_train)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=6,
        l2_leaf_reg=5.0, random_seed=42, verbose=0,
        early_stopping_rounds=100, auto_class_weights="Balanced",
        eval_metric="AUC",
    )
    model.fit(X_train[train_idx], y[train_idx],
              eval_set=(X_train[val_idx], y[val_idx]))
    preds = model.predict_proba(X_train[val_idx])[:, 1]
    oof_cat[val_idx] = preds
    score = roc_auc_score(y[val_idx], preds)
    cat_scores.append(score)
    print(f"  Fold {fold+1}: AUC = {score:.5f} (best_iter={model.best_iteration_})")

print(f"  Mean: {np.mean(cat_scores):.5f} (+/- {np.std(cat_scores):.5f})")
print(f"  OOF AUC: {roc_auc_score(y, oof_cat):.5f}")

log_experiment({
    "experiment_id": 3,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "baseline",
    "model": "CatBoost",
    "features": f"audio features ({n_features})",
    "cv_strategy": f"StratifiedKFold-{n_splits}",
    "cv_scores": [round(s, 5) for s in cat_scores],
    "cv_mean": round(np.mean(cat_scores), 5),
    "cv_std": round(np.std(cat_scores), 5),
    "oof_auc": round(roc_auc_score(y, oof_cat), 5),
    "notes": "CatBoost binary, balanced weights, audio features"
})

# ============================================================
# 4. OOF ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print("4. OOF ENSEMBLE")
print("=" * 60)

print("Weight search (LGB, XGB, Cat):")
best_auc = 0
best_w = None
for w_lgb_w in np.arange(0.1, 0.9, 0.1):
    for w_xgb_w in np.arange(0.1, 0.9, 0.1):
        w_cat_w = 1.0 - w_lgb_w - w_xgb_w
        if w_cat_w < 0.05 or w_cat_w > 0.8:
            continue
        blend = w_lgb_w * oof_lgb + w_xgb_w * oof_xgb + w_cat_w * oof_cat
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc = auc
            best_w = (round(w_lgb_w, 1), round(w_xgb_w, 1), round(w_cat_w, 1))

print(f"  Best weights: LGB={best_w[0]}, XGB={best_w[1]}, Cat={best_w[2]}")
print(f"  Best ensemble AUC: {best_auc:.5f}")

simple_blend = (oof_lgb + oof_xgb + oof_cat) / 3
simple_auc = roc_auc_score(y, simple_blend)
print(f"  Simple average AUC: {simple_auc:.5f}")

log_experiment({
    "experiment_id": 4,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "ensemble",
    "model": f"Weighted Ensemble (LGB={best_w[0]}, XGB={best_w[1]}, Cat={best_w[2]})",
    "features": f"audio features ({n_features})",
    "cv_strategy": f"StratifiedKFold-{n_splits} OOF",
    "cv_mean": round(best_auc, 5),
    "notes": f"Best weights. Simple avg: {simple_auc:.5f}"
})

# Save best weights
np.save(f"{data_dir}/best_weights.npy", np.array(best_w))

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT LEADERBOARD")
print("=" * 60)

results = [
    ("LightGBM", np.mean(lgb_scores), roc_auc_score(y, oof_lgb)),
    ("XGBoost", np.mean(xgb_scores), roc_auc_score(y, oof_xgb)),
    ("CatBoost", np.mean(cat_scores), roc_auc_score(y, oof_cat)),
    (f"Ensemble ({best_w})", best_auc, best_auc),
    ("Ensemble (equal)", simple_auc, simple_auc),
]
results.sort(key=lambda x: -x[2])

print(f"\n{'Rank':<5} {'Model':<45} {'Mean CV AUC':<14} {'OOF AUC':<10}")
print("-" * 75)
for rank, (name, mean, oof) in enumerate(results, 1):
    print(f"{rank:<5} {name:<45} {mean:<14.5f} {oof:<10.5f}")

print(f"\nBest weights saved.")
