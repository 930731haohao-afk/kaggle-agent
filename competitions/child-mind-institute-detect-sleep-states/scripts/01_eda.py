"""EDA for child-mind-institute-detect-sleep-states.

Efficient analysis of 128M-row accelerometer data + event labels.
Focuses on understanding sleep patterns, signal characteristics, and label quality.
"""
import pandas as pd
import numpy as np
import os
from collections import Counter

data_dir = "competitions/child-mind-institute-detect-sleep-states/data"

print("=" * 70)
print("DETECT SLEEP STATES - EDA")
print("=" * 70)

# ============================================================
# 1. EVENT LABEL ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("1. EVENT LABEL ANALYSIS")
print("=" * 70)

events = pd.read_csv(os.path.join(data_dir, "train_events.csv"))
events['timestamp'] = pd.to_datetime(events['timestamp'], utc=True)

print(f"Total events: {len(events)}")
print(f"Event types: {events['event'].value_counts().to_dict()}")
print(f"Unique series: {events['series_id'].nunique()}")

# Null analysis
null_mask = events['step'].isnull()
print(f"\nNull step events: {null_mask.sum()} ({null_mask.sum()/len(events)*100:.1f}%)")
null_series = events[null_mask]['series_id'].nunique()
print(f"Series with null events: {null_series}/{events['series_id'].nunique()}")

# Nights per series
valid_events = events.dropna(subset=['step'])
print(f"Valid events (non-null step): {len(valid_events)}")

# Onset-wakeup pairing
nights_by_series = valid_events.groupby('series_id')['night'].nunique()
print(f"\nValid nights per series: min={nights_by_series.min()}, max={nights_by_series.max()}, "
      f"mean={nights_by_series.mean():.1f}")

# Sleep duration analysis
print("\n--- SLEEP DURATION ---")
onset_events = valid_events[valid_events['event'] == 'onset'].copy()
wakeup_events = valid_events[valid_events['event'] == 'wakeup'].copy()

# Match onset-wakeup pairs by series_id and night
pairs = onset_events.merge(wakeup_events, on=['series_id', 'night'], suffixes=('_onset', '_wakeup'))
pairs['sleep_steps'] = pairs['step_wakeup'] - pairs['step_onset']
pairs['sleep_hours'] = pairs['sleep_steps'] * 5 / 3600  # 5 seconds per step

print(f"Matched onset-wakeup pairs: {len(pairs)}")
print(f"Sleep duration (hours): mean={pairs['sleep_hours'].mean():.2f}, "
      f"std={pairs['sleep_hours'].std():.2f}")
print(f"  min={pairs['sleep_hours'].min():.2f}, 25%={pairs['sleep_hours'].quantile(0.25):.2f}, "
      f"50%={pairs['sleep_hours'].median():.2f}, 75%={pairs['sleep_hours'].quantile(0.75):.2f}, "
      f"max={pairs['sleep_hours'].max():.2f}")

# Negative sleep durations (onset after wakeup)?
neg = (pairs['sleep_hours'] < 0).sum()
very_short = (pairs['sleep_hours'] < 2).sum()
very_long = (pairs['sleep_hours'] > 14).sum()
print(f"\nAnomolies: negative_duration={neg}, <2h={very_short}, >14h={very_long}")

# Onset time of day
onset_events['hour'] = onset_events['timestamp'].dt.hour
wakeup_events['hour'] = wakeup_events['timestamp'].dt.hour

print("\n--- ONSET TIME DISTRIBUTION ---")
onset_hour_counts = onset_events['hour'].value_counts().sort_index()
for h, cnt in onset_hour_counts.items():
    bar = '#' * (cnt // 20)
    print(f"  {h:02d}:00  {cnt:>4}  {bar}")

print("\n--- WAKEUP TIME DISTRIBUTION ---")
wakeup_hour_counts = wakeup_events['hour'].value_counts().sort_index()
for h, cnt in wakeup_hour_counts.items():
    bar = '#' * (cnt // 20)
    print(f"  {h:02d}:00  {cnt:>4}  {bar}")

# ============================================================
# 2. SIGNAL ANALYSIS (sampled)
# ============================================================
print("\n" + "=" * 70)
print("2. SIGNAL ANALYSIS")
print("=" * 70)

# Load full data but process per-series
train = pd.read_parquet(os.path.join(data_dir, "train_series.parquet"))
print(f"Full train loaded: {train.shape}")

# Overall stats
for col in ['enmo', 'anglez']:
    vals = train[col]
    print(f"\n{col}:")
    print(f"  mean={vals.mean():.4f}, std={vals.std():.4f}")
    print(f"  min={vals.min():.4f}, q1={vals.quantile(0.25):.4f}, "
          f"median={vals.median():.4f}, q3={vals.quantile(0.75):.4f}, max={vals.max():.4f}")
    print(f"  skew={vals.skew():.3f}")
    # Zero/near-zero counts
    near_zero = (vals.abs() < 0.001).sum()
    print(f"  near-zero (<0.001): {near_zero} ({near_zero/len(vals)*100:.1f}%)")

# ============================================================
# 3. SIGNAL AROUND EVENTS
# ============================================================
print("\n" + "=" * 70)
print("3. SIGNAL AROUND EVENTS")
print("=" * 70)

# Analyze signal behavior around sleep onset and wakeup
# Sample a few series for efficiency
sample_series = valid_events['series_id'].unique()[:20]
window = 360  # 360 steps = 30 minutes

onset_stats = {'enmo_before': [], 'enmo_after': [], 'anglez_before': [], 'anglez_after': [],
               'anglez_std_before': [], 'anglez_std_after': [], 'enmo_std_before': [], 'enmo_std_after': []}
wakeup_stats = {'enmo_before': [], 'enmo_after': [], 'anglez_before': [], 'anglez_after': [],
                'anglez_std_before': [], 'anglez_std_after': [], 'enmo_std_before': [], 'enmo_std_after': []}

for sid in sample_series:
    series_data = train[train['series_id'] == sid]
    series_events = valid_events[valid_events['series_id'] == sid]

    for _, ev in series_events.iterrows():
        step = int(ev['step'])
        before = series_data[(series_data['step'] >= step - window) & (series_data['step'] < step)]
        after = series_data[(series_data['step'] >= step) & (series_data['step'] < step + window)]

        if len(before) < window // 2 or len(after) < window // 2:
            continue

        stats = onset_stats if ev['event'] == 'onset' else wakeup_stats
        stats['enmo_before'].append(before['enmo'].mean())
        stats['enmo_after'].append(after['enmo'].mean())
        stats['anglez_before'].append(before['anglez'].mean())
        stats['anglez_after'].append(after['anglez'].mean())
        stats['anglez_std_before'].append(before['anglez'].std())
        stats['anglez_std_after'].append(after['anglez'].std())
        stats['enmo_std_before'].append(before['enmo'].std())
        stats['enmo_std_after'].append(after['enmo'].std())

print(f"\nOnset events analyzed: {len(onset_stats['enmo_before'])}")
print("Signal around ONSET (sleep start):")
print(f"  enmo:   before={np.mean(onset_stats['enmo_before']):.4f} → after={np.mean(onset_stats['enmo_after']):.4f}")
print(f"  anglez: before_mean={np.mean(onset_stats['anglez_before']):.2f} → after_mean={np.mean(onset_stats['anglez_after']):.2f}")
print(f"  anglez: before_std={np.mean(onset_stats['anglez_std_before']):.2f} → after_std={np.mean(onset_stats['anglez_std_after']):.2f}")
print(f"  enmo:   before_std={np.mean(onset_stats['enmo_std_before']):.4f} → after_std={np.mean(onset_stats['enmo_std_after']):.4f}")

print(f"\nWakeup events analyzed: {len(wakeup_stats['enmo_before'])}")
print("Signal around WAKEUP:")
print(f"  enmo:   before={np.mean(wakeup_stats['enmo_before']):.4f} → after={np.mean(wakeup_stats['enmo_after']):.4f}")
print(f"  anglez: before_mean={np.mean(wakeup_stats['anglez_before']):.2f} → after_mean={np.mean(wakeup_stats['anglez_after']):.2f}")
print(f"  anglez: before_std={np.mean(wakeup_stats['anglez_std_before']):.2f} → after_std={np.mean(wakeup_stats['anglez_std_after']):.2f}")
print(f"  enmo:   before_std={np.mean(wakeup_stats['enmo_std_before']):.4f} → after_std={np.mean(wakeup_stats['enmo_std_after']):.4f}")

# ============================================================
# 4. DOWNSAMPLED ANALYSIS (5-min windows)
# ============================================================
print("\n" + "=" * 70)
print("4. DOWNSAMPLED SIGNAL ANALYSIS (12-step = 1-minute windows)")
print("=" * 70)

# Compute 12-step (1-minute) aggregated features for a sample series
sample_sid = sample_series[0]
sample_data = train[train['series_id'] == sample_sid].copy()
sample_data['minute_bin'] = sample_data['step'] // 12

minute_agg = sample_data.groupby('minute_bin').agg(
    enmo_mean=('enmo', 'mean'),
    enmo_std=('enmo', 'std'),
    enmo_max=('enmo', 'max'),
    anglez_mean=('anglez', 'mean'),
    anglez_std=('anglez', 'std'),
    anglez_range=('anglez', lambda x: x.max() - x.min()),
).reset_index()

print(f"Sample series {sample_sid}: {len(sample_data)} steps → {len(minute_agg)} minute bins")
print(f"\n1-minute aggregated stats:")
for col in minute_agg.columns:
    if col == 'minute_bin':
        continue
    print(f"  {col}: mean={minute_agg[col].mean():.4f}, std={minute_agg[col].std():.4f}")

# ============================================================
# 5. SERIES-LEVEL STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("5. SERIES-LEVEL STATISTICS")
print("=" * 70)

series_stats = train.groupby('series_id').agg(
    n_steps=('step', 'count'),
    enmo_mean=('enmo', 'mean'),
    enmo_std=('enmo', 'std'),
    anglez_mean=('anglez', 'mean'),
    anglez_std=('anglez', 'std'),
).reset_index()

print(f"Series-level statistics ({len(series_stats)} series):")
for col in ['n_steps', 'enmo_mean', 'enmo_std', 'anglez_mean', 'anglez_std']:
    vals = series_stats[col]
    print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
          f"min={vals.min():.4f}, max={vals.max():.4f}")

# Variation across series
print(f"\nInter-series coefficient of variation:")
for col in ['enmo_mean', 'anglez_mean']:
    cv = series_stats[col].std() / series_stats[col].mean() if series_stats[col].mean() != 0 else float('inf')
    print(f"  {col}: CV={abs(cv):.3f}")

# ============================================================
# 6. DEVICE-OFF DETECTION
# ============================================================
print("\n" + "=" * 70)
print("6. DEVICE-OFF / CONSTANT SIGNAL DETECTION")
print("=" * 70)

# Detect periods where the device might be off (constant anglez, zero enmo)
n_constant = 0
n_total_steps = 0
constant_series = []

for sid in train['series_id'].unique():
    s = train[train['series_id'] == sid]
    n_total_steps += len(s)

    # Check for long runs of zero enmo
    zero_enmo = (s['enmo'] == 0).values
    # Find runs
    if zero_enmo.any():
        diffs = np.diff(zero_enmo.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if zero_enmo[0]:
            starts = np.concatenate([[0], starts])
        if zero_enmo[-1]:
            ends = np.concatenate([ends, [len(zero_enmo) - 1]])
        if len(starts) > 0 and len(ends) > 0:
            min_len = min(len(starts), len(ends))
            run_lengths = ends[:min_len] - starts[:min_len]
            long_runs = (run_lengths > 720).sum()  # >1 hour of zero enmo
            if long_runs > 0:
                constant_series.append(sid)
                n_constant += run_lengths[run_lengths > 720].sum()

print(f"Series with long zero-enmo runs (>1h): {len(constant_series)}/{len(train['series_id'].unique())}")
print(f"Total steps in long zero-enmo runs: {n_constant:,} ({n_constant/n_total_steps*100:.2f}%)")

del train  # Free memory

# ============================================================
# 7. LABEL QUALITY CHECK
# ============================================================
print("\n" + "=" * 70)
print("7. LABEL QUALITY CHECK")
print("=" * 70)

# Check for unpaired events
for sid in events['series_id'].unique():
    s_events = events[events['series_id'] == sid]
    for night in s_events['night'].unique():
        night_events = s_events[s_events['night'] == night]
        has_onset = (night_events['event'] == 'onset').any()
        has_wakeup = (night_events['event'] == 'wakeup').any()
        if has_onset != has_wakeup:
            print(f"  WARNING: Unpaired event in {sid} night {night}")

# Check onset always comes before wakeup
bad_order = pairs[pairs['step_onset'] >= pairs['step_wakeup']]
print(f"\nOnset after wakeup (bad order): {len(bad_order)}")

# Events per series
events_per_series = valid_events.groupby('series_id').size()
print(f"\nValid events per series: mean={events_per_series.mean():.1f}, "
      f"min={events_per_series.min()}, max={events_per_series.max()}")

# Consecutive onset-wakeup alternation check
print("\nChecking event alternation (onset-wakeup-onset-wakeup)...")
alternation_errors = 0
for sid in valid_events['series_id'].unique()[:50]:  # check first 50
    s = valid_events[valid_events['series_id'] == sid].sort_values('step')
    prev_event = None
    for _, row in s.iterrows():
        if prev_event == row['event']:
            alternation_errors += 1
        prev_event = row['event']
if alternation_errors > 0:
    print(f"  Alternation errors in first 50 series: {alternation_errors}")
else:
    print(f"  All events properly alternate in first 50 series")

print(f"\n{'='*70}")
print("EDA COMPLETE")
print(f"{'='*70}")
