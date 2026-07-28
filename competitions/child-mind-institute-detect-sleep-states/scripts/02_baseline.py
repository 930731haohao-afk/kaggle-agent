"""
Baseline for child-mind-institute-detect-sleep-states.

Approach:
1. Downsample to 12-step (1-minute) windows with aggregated features
2. Compute rolling statistics (30min, 60min) and their changes
3. Label each minute as containing an onset/wakeup event (or not)
4. Train LightGBM to predict event probability
5. Peak detection to extract candidate events
6. Validate using GroupKFold by series_id
"""
import pandas as pd
import numpy as np
import os
import json
import time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from collections import defaultdict

data_dir = "competitions/child-mind-institute-detect-sleep-states/data"
exp_file = "competitions/child-mind-institute-detect-sleep-states/experiments.json"

print("=" * 70)
print("DETECT SLEEP STATES - BASELINE")
print("=" * 70)

# ============================================================
# 1. LOAD AND DOWNSAMPLE
# ============================================================
print("\n1. Loading and downsampling train series...")
t0 = time.time()

train_series = pd.read_parquet(os.path.join(data_dir, "train_series.parquet"))
events = pd.read_csv(os.path.join(data_dir, "train_events.csv"))
events = events.dropna(subset=['step'])
events['step'] = events['step'].astype(int)

print(f"  Loaded {len(train_series):,} rows, {train_series['series_id'].nunique()} series in {time.time()-t0:.1f}s")

# Downsample to 12-step (1-minute) windows
DOWNSAMPLE = 12  # 12 steps = 1 minute
train_series['minute_bin'] = train_series['step'] // DOWNSAMPLE

print("  Aggregating to 1-minute windows...")
t1 = time.time()
minute_data = train_series.groupby(['series_id', 'minute_bin']).agg(
    step_start=('step', 'first'),
    step_end=('step', 'last'),
    enmo_mean=('enmo', 'mean'),
    enmo_std=('enmo', 'std'),
    enmo_max=('enmo', 'max'),
    enmo_min=('enmo', 'min'),
    anglez_mean=('anglez', 'mean'),
    anglez_std=('anglez', 'std'),
    anglez_max=('anglez', 'max'),
    anglez_min=('anglez', 'min'),
).reset_index()

minute_data['anglez_range'] = minute_data['anglez_max'] - minute_data['anglez_min']
minute_data['enmo_range'] = minute_data['enmo_max'] - minute_data['enmo_min']
minute_data['enmo_std'] = minute_data['enmo_std'].fillna(0)
minute_data['anglez_std'] = minute_data['anglez_std'].fillna(0)

del train_series  # Free memory
print(f"  Downsampled to {len(minute_data):,} minute windows in {time.time()-t1:.1f}s")

# ============================================================
# 2. ROLLING FEATURES
# ============================================================
print("\n2. Computing rolling features...")
t2 = time.time()

# Sort for correct rolling computation
minute_data = minute_data.sort_values(['series_id', 'minute_bin']).reset_index(drop=True)

# Rolling windows: 30 minutes, 60 minutes
for win in [15, 30, 60]:
    for col in ['enmo_mean', 'anglez_mean', 'anglez_std']:
        minute_data[f'{col}_roll{win}'] = (
            minute_data.groupby('series_id')[col]
            .transform(lambda x: x.rolling(win, center=True, min_periods=1).mean())
        )

# Compute changes (derivatives) - key for detecting transitions
for col in ['enmo_mean', 'anglez_std']:
    # Diff from 30min ago
    minute_data[f'{col}_diff30'] = (
        minute_data.groupby('series_id')[f'{col}_roll30']
        .transform(lambda x: x.diff(30))
    )
    minute_data[f'{col}_diff60'] = (
        minute_data.groupby('series_id')[f'{col}_roll60']
        .transform(lambda x: x.diff(60))
    )

# Absolute anglez (sleep often at extreme angles)
minute_data['anglez_abs_mean'] = minute_data['anglez_mean'].abs()

# Time-of-day features (using minute_bin)
steps_per_day = 17280  # 5s intervals per day
minutes_per_day = steps_per_day // DOWNSAMPLE  # 1440 minutes
minute_data['minute_of_day'] = minute_data['minute_bin'] % minutes_per_day
minute_data['hour'] = minute_data['minute_of_day'] // 60
minute_data['hour_sin'] = np.sin(2 * np.pi * minute_data['hour'] / 24)
minute_data['hour_cos'] = np.cos(2 * np.pi * minute_data['hour'] / 24)

print(f"  Rolling features computed in {time.time()-t2:.1f}s")

# ============================================================
# 3. LABEL EVENTS
# ============================================================
print("\n3. Labeling events...")

# Binary labels: mark minute bins that are within TOLERANCE of a ground truth event
TOLERANCE = 3  # minutes (±3 = ±36 seconds at step level)

minute_data['is_onset'] = 0
minute_data['is_wakeup'] = 0

# Build lookup: series_id -> set of event minute bins
onset_bins = defaultdict(set)
wakeup_bins = defaultdict(set)

for _, ev in events.iterrows():
    sid = ev['series_id']
    ev_minute = int(ev['step']) // DOWNSAMPLE
    if ev['event'] == 'onset':
        for offset in range(-TOLERANCE, TOLERANCE + 1):
            onset_bins[sid].add(ev_minute + offset)
    else:
        for offset in range(-TOLERANCE, TOLERANCE + 1):
            wakeup_bins[sid].add(ev_minute + offset)

# Vectorized labeling per series
for sid in minute_data['series_id'].unique():
    mask = minute_data['series_id'] == sid
    bins = minute_data.loc[mask, 'minute_bin'].values
    if sid in onset_bins:
        onset_set = onset_bins[sid]
        minute_data.loc[mask, 'is_onset'] = np.array([1 if b in onset_set else 0 for b in bins])
    if sid in wakeup_bins:
        wakeup_set = wakeup_bins[sid]
        minute_data.loc[mask, 'is_wakeup'] = np.array([1 if b in wakeup_set else 0 for b in bins])

n_onset = minute_data['is_onset'].sum()
n_wakeup = minute_data['is_wakeup'].sum()
print(f"  Onset windows: {n_onset} ({n_onset/len(minute_data)*100:.3f}%)")
print(f"  Wakeup windows: {n_wakeup} ({n_wakeup/len(minute_data)*100:.3f}%)")

# ============================================================
# 4. PREPARE FEATURES
# ============================================================
print("\n4. Preparing features...")

feature_cols = [
    'enmo_mean', 'enmo_std', 'enmo_max', 'enmo_min', 'enmo_range',
    'anglez_mean', 'anglez_std', 'anglez_max', 'anglez_min', 'anglez_range',
    'anglez_abs_mean',
    'enmo_mean_roll15', 'enmo_mean_roll30', 'enmo_mean_roll60',
    'anglez_mean_roll15', 'anglez_mean_roll30', 'anglez_mean_roll60',
    'anglez_std_roll15', 'anglez_std_roll30', 'anglez_std_roll60',
    'enmo_mean_diff30', 'enmo_mean_diff60',
    'anglez_std_diff30', 'anglez_std_diff60',
    'hour', 'hour_sin', 'hour_cos',
]

# Fill NaN from rolling/diff operations
minute_data[feature_cols] = minute_data[feature_cols].fillna(0)

X = minute_data[feature_cols].values
y_onset = minute_data['is_onset'].values
y_wakeup = minute_data['is_wakeup'].values
groups = minute_data['series_id'].values

print(f"  Features: {len(feature_cols)}")
print(f"  X shape: {X.shape}")

# ============================================================
# 5. TRAIN LIGHTGBM (GroupKFold)
# ============================================================
print("\n5. Training LightGBM...")

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'scale_pos_weight': 100,  # Heavy class imbalance
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

n_folds = 5
gkf = GroupKFold(n_splits=n_folds)
unique_groups = minute_data['series_id'].unique()

# Train separate models for onset and wakeup
for event_name, y_target in [('onset', y_onset), ('wakeup', y_wakeup)]:
    print(f"\n  --- {event_name.upper()} ---")
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_target, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_target[train_idx], y_target[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

        model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred

        val_ap = average_precision_score(y_val, val_pred)
        print(f"    Fold {fold+1}: AP={val_ap:.5f}, best_iter={model.best_iteration}")

    overall_ap = average_precision_score(y_target, oof_preds)
    print(f"  OOF AP ({event_name}): {overall_ap:.5f}")

    minute_data[f'pred_{event_name}'] = oof_preds

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"  Top 10 features:")
    for name, imp in feat_imp[:10]:
        print(f"    {name:<30} {imp:.1f}")

# ============================================================
# 6. EVENT EXTRACTION (peak detection)
# ============================================================
print("\n6. Extracting events via peak detection...")

def extract_events(df, pred_col, event_type, min_distance=30, threshold=0.1):
    """Extract events from prediction scores using peak detection."""
    all_events = []
    for sid in df['series_id'].unique():
        s = df[df['series_id'] == sid].sort_values('minute_bin')
        preds = s[pred_col].values
        steps = s['step_start'].values
        bins = s['minute_bin'].values

        # Simple peak detection: find local maxima above threshold
        candidates = []
        for i in range(1, len(preds) - 1):
            if preds[i] > threshold and preds[i] > preds[i-1] and preds[i] > preds[i+1]:
                candidates.append((i, preds[i]))

        # NMS: keep only the best within min_distance
        candidates.sort(key=lambda x: x[1], reverse=True)
        kept = []
        used = set()
        for idx, score in candidates:
            if any(abs(idx - k) < min_distance for k in used):
                continue
            kept.append((sid, int(steps[idx]), event_type, float(score)))
            used.add(idx)

        all_events.extend(kept)
    return all_events

onset_events = extract_events(minute_data, 'pred_onset', 'onset', min_distance=60, threshold=0.05)
wakeup_events = extract_events(minute_data, 'pred_wakeup', 'wakeup', min_distance=60, threshold=0.05)

all_predicted = onset_events + wakeup_events
print(f"  Predicted onset events: {len(onset_events)}")
print(f"  Predicted wakeup events: {len(wakeup_events)}")
print(f"  Total predicted events: {len(all_predicted)}")

# ============================================================
# 7. LOCAL EVALUATION (approximate EDAP)
# ============================================================
print("\n7. Local evaluation (approximate EDAP)...")

def compute_edap_approx(true_events, pred_events, tolerance_steps=720):
    """
    Approximate EDAP: for each event type, match predictions to ground truth
    within tolerance and compute AP.
    tolerance_steps=720 = 1 hour (720 * 5s = 3600s)
    """
    results = {}
    for event_type in ['onset', 'wakeup']:
        true_et = [(sid, step) for sid, _, ev, step in
                   [(r['series_id'], r['step'], r['event'], r['step'])
                    for _, r in true_events[true_events['event'] == event_type].iterrows()]]
        # Reformat
        true_et = [(r['series_id'], int(r['step'])) for _, r in
                   true_events[true_events['event'] == event_type].iterrows()]

        pred_et = [(sid, step, score) for sid, step, ev, score in pred_events if ev == event_type]
        pred_et.sort(key=lambda x: x[2], reverse=True)

        if not true_et or not pred_et:
            results[event_type] = 0.0
            continue

        # Group by series
        true_by_series = defaultdict(list)
        for sid, step in true_et:
            true_by_series[sid].append(step)

        tp = 0
        fp = 0
        matched = set()
        precisions = []
        recalls = []
        n_true = len(true_et)

        for sid, step, score in pred_et:
            if sid not in true_by_series:
                fp += 1
            else:
                # Find closest unmatched ground truth
                best_match = None
                best_dist = float('inf')
                for i, gt_step in enumerate(true_by_series[sid]):
                    key = (sid, i)
                    if key in matched:
                        continue
                    dist = abs(step - gt_step)
                    if dist <= tolerance_steps and dist < best_dist:
                        best_dist = dist
                        best_match = key
                if best_match is not None:
                    tp += 1
                    matched.add(best_match)
                else:
                    fp += 1

            precision = tp / (tp + fp)
            recall = tp / n_true
            precisions.append(precision)
            recalls.append(recall)

        # Compute AP (area under precision-recall curve)
        if not recalls:
            results[event_type] = 0.0
            continue

        # Interpolate
        recalls = [0] + recalls
        precisions = [1] + precisions
        ap = 0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
        results[event_type] = ap

    return results

valid_events = events.dropna(subset=['step'])

# Evaluate at multiple tolerances (in steps: 720=1hr, 2160=3hr, 4320=6hr)
for tol_name, tol_steps in [('30min', 360), ('1hr', 720), ('3hr', 2160), ('6hr', 4320)]:
    scores = compute_edap_approx(valid_events, all_predicted, tolerance_steps=tol_steps)
    mean_ap = np.mean(list(scores.values()))
    print(f"  Tolerance={tol_name}: onset_AP={scores['onset']:.4f}, "
          f"wakeup_AP={scores['wakeup']:.4f}, mean_AP={mean_ap:.4f}")

# ============================================================
# 8. GENERATE SUBMISSION
# ============================================================
print("\n8. Generating submission...")

sub_dir = "competitions/child-mind-institute-detect-sleep-states/submissions"
os.makedirs(sub_dir, exist_ok=True)

# Format for submission
rows = []
for i, (sid, step, event, score) in enumerate(sorted(all_predicted, key=lambda x: (x[0], x[1]))):
    rows.append({'row_id': i, 'series_id': sid, 'step': step, 'event': event, 'score': round(score, 4)})

submission = pd.DataFrame(rows)

from datetime import datetime as dt
timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
sub_file = os.path.join(sub_dir, f"lgbm_baseline_{timestamp}.csv")
submission.to_csv(sub_file, index=False)

print(f"  Submission saved: {sub_file}")
print(f"  Shape: {submission.shape}")
print(f"  Events by type: {submission['event'].value_counts().to_dict()}")
print(f"  Score stats: mean={submission['score'].mean():.4f}, "
      f"min={submission['score'].min():.4f}, max={submission['score'].max():.4f}")

# ============================================================
# 9. LOG EXPERIMENT
# ============================================================
print("\n9. Logging experiment...")

with open(exp_file, 'r') as f:
    exp_data = json.load(f)

exp_data['experiments'].append({
    'id': 1,
    'name': 'LightGBM Baseline (1-min downsample)',
    'model': 'lightgbm',
    'params': lgb_params,
    'n_features': len(feature_cols),
    'downsample': DOWNSAMPLE,
    'notes': ('1-minute downsampled, rolling 15/30/60min features, '
              'peak detection, GroupKFold by series_id')
})

with open(exp_file, 'w') as f:
    json.dump(exp_data, f, indent=2)

print(f"\n{'='*70}")
print("BASELINE COMPLETE")
print(f"{'='*70}")
