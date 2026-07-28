"""Initial inspection for child-mind-institute-detect-sleep-states."""
import pandas as pd
import numpy as np
import os

data_dir = "competitions/child-mind-institute-detect-sleep-states/data"

# ============================================================
# 1. Sample submission
# ============================================================
print("=" * 70)
print("CHILD MIND INSTITUTE - DETECT SLEEP STATES - INITIAL INSPECTION")
print("=" * 70)

sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
print(f"\n--- SAMPLE SUBMISSION ---")
print(f"Shape: {sample_sub.shape}")
print(f"Columns: {list(sample_sub.columns)}")
print(sample_sub.to_string())

# ============================================================
# 2. Train events (labels)
# ============================================================
print(f"\n--- TRAIN EVENTS ---")
events = pd.read_csv(os.path.join(data_dir, "train_events.csv"))
print(f"Shape: {events.shape}")
print(f"Columns: {list(events.columns)}")
for col in events.columns:
    dtype = events[col].dtype
    n_unique = events[col].nunique()
    n_null = events[col].isnull().sum()
    null_str = f" [{n_null} null]" if n_null > 0 else ""
    sample = events[col].dropna().iloc[0] if events[col].notna().any() else "N/A"
    print(f"  {col:<20} {str(dtype):<10} unique={n_unique:<8}{null_str}  sample: {sample}")

print(f"\nFirst 10 rows:")
print(events.head(10).to_string())

print(f"\nEvent types: {events['event'].value_counts().to_dict()}")
print(f"Unique series_ids: {events['series_id'].nunique()}")

# Night counts per series
night_counts = events.groupby('series_id')['night'].max()
print(f"\nNights per series: min={night_counts.min()}, max={night_counts.max()}, "
      f"mean={night_counts.mean():.1f}, median={night_counts.median()}")

# Check for missing events
null_events = events[events['event'].isnull()]
print(f"\nRows with null event: {len(null_events)}")
if len(null_events) > 0:
    null_series = null_events['series_id'].nunique()
    print(f"  Affecting {null_series} series")

# ============================================================
# 3. Test series (small)
# ============================================================
print(f"\n--- TEST SERIES ---")
test = pd.read_parquet(os.path.join(data_dir, "test_series.parquet"))
print(f"Shape: {test.shape}")
print(f"Columns: {list(test.columns)}")
for col in test.columns:
    dtype = test[col].dtype
    n_unique = test[col].nunique()
    n_null = test[col].isnull().sum()
    null_str = f" [{n_null} null]" if n_null > 0 else ""
    sample = test[col].dropna().iloc[0] if test[col].notna().any() else "N/A"
    print(f"  {col:<20} {str(dtype):<10} unique={n_unique:<8}{null_str}  sample: {sample}")
print(f"\nAll rows:")
print(test.to_string())

# ============================================================
# 4. Train series (large - inspect metadata only)
# ============================================================
print(f"\n--- TRAIN SERIES (metadata) ---")
# Read just the metadata columns first
train_meta = pd.read_parquet(os.path.join(data_dir, "train_series.parquet"),
                              columns=['series_id', 'step', 'timestamp'])
print(f"Total rows: {len(train_meta):,}")
print(f"Unique series_ids: {train_meta['series_id'].nunique()}")
print(f"Step range: {train_meta['step'].min()} to {train_meta['step'].max()}")

# Steps per series
steps_per_series = train_meta.groupby('series_id')['step'].count()
print(f"\nSteps per series: min={steps_per_series.min():,}, max={steps_per_series.max():,}, "
      f"mean={steps_per_series.mean():,.0f}, median={steps_per_series.median():,.0f}")

# Timestamp analysis
train_meta['timestamp'] = pd.to_datetime(train_meta['timestamp'], utc=True)
print(f"\nTimestamp range: {train_meta['timestamp'].min()} to {train_meta['timestamp'].max()}")

# Duration per series
duration = train_meta.groupby('series_id')['timestamp'].agg(['min', 'max'])
duration['days'] = (duration['max'] - duration['min']).dt.total_seconds() / 86400
print(f"Recording duration (days): min={duration['days'].min():.1f}, "
      f"max={duration['days'].max():.1f}, mean={duration['days'].mean():.1f}")

del train_meta

# Now read a small sample of the full data
print(f"\n--- TRAIN SERIES (sample) ---")
train_sample = pd.read_parquet(os.path.join(data_dir, "train_series.parquet"),
                                columns=['series_id', 'step', 'timestamp', 'enmo', 'anglez'])
# Get first 1000 rows
first_rows = train_sample.head(20)
print(f"Columns: {list(train_sample.columns)}")
for col in train_sample.columns:
    dtype = train_sample[col].dtype
    n_null = train_sample[col].isnull().sum()
    null_str = f" [{n_null} null]" if n_null > 0 else ""
    print(f"  {col:<20} {str(dtype):<15}{null_str}")
print(f"\nFirst 20 rows:")
print(first_rows.to_string())

# Stats for sensor columns
print(f"\n--- SENSOR STATS (full train) ---")
for col in ['enmo', 'anglez']:
    vals = train_sample[col]
    print(f"\n{col}:")
    print(f"  count={len(vals):,}, nulls={vals.isnull().sum()}")
    print(f"  mean={vals.mean():.4f}, std={vals.std():.4f}")
    print(f"  min={vals.min():.4f}, 25%={vals.quantile(0.25):.4f}, "
          f"50%={vals.median():.4f}, 75%={vals.quantile(0.75):.4f}, max={vals.max():.4f}")

# Check sampling rate
print(f"\n--- SAMPLING RATE ---")
first_series = train_sample[train_sample['series_id'] == train_sample['series_id'].iloc[0]]
first_series_ts = pd.to_datetime(first_series['timestamp'])
time_diffs = first_series_ts.diff().dropna()
print(f"Time between steps (first series): {time_diffs.mode().values[0]}")
print(f"Mean time diff: {time_diffs.mean()}")
print(f"Steps per 5 seconds: estimated 1 step per 5s = 12 steps/min = 720 steps/hr = 17,280 steps/day")

# Match events to series
print(f"\n--- SERIES OVERLAP ---")
train_series_ids = set(train_sample['series_id'].unique())
event_series_ids = set(events['series_id'].unique())
print(f"Train series IDs: {len(train_series_ids)}")
print(f"Event series IDs: {len(event_series_ids)}")
print(f"Series in both: {len(train_series_ids & event_series_ids)}")
print(f"Series in events but not train: {len(event_series_ids - train_series_ids)}")
print(f"Series in train but not events: {len(train_series_ids - event_series_ids)}")

del train_sample

print(f"\n{'='*70}")
print("INSPECTION COMPLETE")
print(f"{'='*70}")
