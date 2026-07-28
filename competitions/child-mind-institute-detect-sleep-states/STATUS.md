# Child Mind Institute — Detect Sleep States

**Date**: 2026-02-19

## Competition Info
- URL: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states
- Problem: Time series event detection (detect sleep onset and wakeup from accelerometer data)
- Metric: Event Detection Average Precision (EDAP) — AP with tolerance-based matching
- **Code competition** — requires Kaggle notebook submission
- Train: 127.9M rows across 277 series (941MB parquet)
- Test: Hidden (placeholder test has only 3 series with 150 steps each)

## Key EDA Findings
- Two sensor channels: enmo (movement intensity, 0-11.4, right-skewed) and anglez (wrist angle, -90 to 90)
- Sampling rate: 1 step per 5 seconds = 17,280 steps/day
- Recording duration: 2-83 days per series (mean 26.7)
- 14,508 labeled events (7,254 onset + 7,254 wakeup), 34% have null step (no annotation)
- Mean sleep duration: 8.65 hours, onset peaks at 2-3am, wakeup at 11am
- **Sleep onset signal**: enmo drops 5x (0.032→0.007), anglez_std drops from 32→14
- **Wakeup signal**: enmo increases 6x (0.007→0.041), anglez_std increases from 25→33
- 132/277 series have >1h zero-enmo runs (device-off periods)
- Inter-series variation: anglez_mean CV=1.17 (high), enmo_mean CV=0.30

## Feature Engineering
- 27 features from 1-minute downsampled windows (12 steps per bin)
- Base features: enmo_mean/std/max/min/range, anglez_mean/std/max/min/range, anglez_abs_mean
- Rolling features: 15/30/60-minute rolling means of enmo_mean, anglez_mean, anglez_std
- Change features: 30/60-minute diffs of enmo_mean and anglez_std
- Time features: hour, hour_sin, hour_cos

## Experiment Results

| # | Model | Local EDAP (1hr tol) | Notes |
|---|-------|---------------------|-------|
| 1 | LightGBM Baseline | 0.3126 (mean AP) | 1-min downsample, peak detection, GroupKFold |

Detailed tolerance analysis:
- 30min: onset=0.103, wakeup=0.354, mean=0.228
- 1hr: onset=0.170, wakeup=0.455, mean=0.313
- 3hr: onset=0.329, wakeup=0.554, mean=0.441
- 6hr: onset=0.445, wakeup=0.614, mean=0.529

## Submissions
- No Kaggle submission yet (code competition — requires notebook)
- Local validation file: `lgbm_baseline_20260219_154909.csv`

## Model Details
- **Baseline**: LightGBM binary classifiers (separate for onset/wakeup)
- scale_pos_weight=100 for extreme imbalance (0.3% positive)
- GroupKFold by series_id (5 folds)
- Peak detection with NMS (min_distance=60 min) for event extraction
- Top features: anglez_std_diff30, anglez_std_roll60, anglez_std_roll30, hour

## Lessons Learned
- Wakeup (AP=0.45) is significantly easier to detect than onset (AP=0.17) at 1hr tolerance
- anglez_std change is the most predictive feature — sleep = stable wrist angle
- Model converges in only 2 iterations due to extreme imbalance — needs better sampling strategy
- Time of day (hour) is a strong feature — priors on typical sleep schedules help
- Code competition format prevents direct CSV submission

## Potential Improvements
- Better sampling: use negative hard mining instead of scale_pos_weight
- Finer temporal resolution (5-step = 25s windows instead of 1-min)
- 1D CNN / Transformer for sequence modeling
- More features: rolling kurtosis, zero-crossing rate, spectral features
- Post-processing: enforce onset-wakeup alternation and realistic sleep durations
- Ensemble with different window sizes
- Build Kaggle notebook for actual submission
