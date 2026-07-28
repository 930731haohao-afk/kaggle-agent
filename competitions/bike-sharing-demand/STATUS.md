# Bike Sharing Demand — Competition Status

## Competition Info
- **URL**: https://www.kaggle.com/c/bike-sharing-demand
- **Problem**: Predict hourly bike rental count from weather and time features
- **Metric**: RMSLE (Root Mean Squared Logarithmic Error, minimize)
- **Data**: 10,886 train rows (hourly, 2011–2012), 6,493 test rows
- **Train/test split**: Train = days 1–19 of each month, Test = days 20–31

## Current Best Score
| CV RMSLE | Public LB RMSLE |
|----------|-----------------|
| 0.317    | 0.419           |

## Pipeline Summary

### Features (13 total)
Extracted from 9 raw columns. Key engineered features:
- `hour_workingday` — hour × 100 + workingday (dominant feature, captures commute vs. leisure)
- `rush_hour` — flag for peak commute hours on workdays
- Datetime extractions: `hour`, `dayofweek`, `month`, `year`, `day`
- Dropped: `atemp` (0.985 corr with temp), `is_weekend`, `season`, `windspeed_zero` (zero importance)

### Models
All trained on `log1p(count)` with GroupKFold-5 by year-month.

| Rank | Model            | CV RMSLE |
|------|------------------|----------|
| 1    | CatBoost         | 0.317    |
| 2    | XGBoost          | 0.322    |
| 3    | LightGBM-default | 0.325    |
| 4    | LightGBM-tuned   | 0.327    |
| 5    | Ridge            | 0.877    |
| 6    | Mean predictor   | 1.419    |

### Submission
- **Method**: Simple average of CatBoost + XGBoost + LightGBM, retrained on full training data
- **Post-processing**: `expm1` to reverse log transform, clip >= 0, round to int
- **File**: `submissions/submission_ensemble3_cv0.317_20260222_114643.csv`

## Key Observations
1. `hour_workingday` interaction is 12× more important than any other feature — bimodal commute pattern on workdays vs. midday plateau on weekends
2. All 3 GBMs scored within 0.01 of each other — feature engineering matters more than model choice
3. CV (0.317) vs. LB (0.419) gap suggests GroupKFold-by-month is slightly optimistic for this day-of-month split

## Potential Improvements
- **Separate casual/registered models** — Train two models, sum predictions (known trick for this competition)
- **Cyclical encoding** — sin/cos for hour and month
- **Better CV** — Temporal holdout matching the day-of-month split
- **Optuna tuning** — Systematic hyperparameter search
- **Ensemble weights** — Optimize blending weights instead of simple average

## Files
```
competitions/bike-sharing-demand/
├── config.yaml                 # Competition metadata
├── experiments.json            # All experiment logs (7 entries)
├── STATUS.md                   # This file
├── data/
│   ├── train.csv               # Raw train data
│   ├── test.csv                # Raw test data
│   ├── sampleSubmission.csv    # Submission format reference
│   ├── train_processed.parquet # Engineered features (train)
│   └── test_processed.parquet  # Engineered features (test)
├── scripts/
│   ├── eda.py                  # Exploratory data analysis
│   ├── feature_engineering.py  # Feature pipeline
│   ├── train.py                # Model training with CV
│   └── submit.py               # Ensemble submission generation
└── submissions/
    └── submission_ensemble3_cv0.317_20260222_114643.csv
```
