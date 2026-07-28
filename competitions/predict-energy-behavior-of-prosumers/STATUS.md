# Predict Energy Behavior of Prosumers Competition Memory

**Date**: 2026-02-19

## Competition Info
- URL: https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers
- Problem: Time series regression (predict energy production and consumption of Estonian prosumers)
- Metric: MAE (Mean Absolute Error)
- Train: 2,018,352 rows (Sep 2021 – May 2023, ~638 days)
- 69 prediction units (county × is_business × product_type), hourly granularity
- Supplementary data: client info, historical weather, forecast weather, electricity prices, gas prices
- **Code competition** — requires Kaggle notebook submission via time-series API

## Key EDA Findings
- Target highly right-skewed (mean=275, median=31, skew=7.7) — log1p reduces skew to 0.09
- 17.4% zero targets (mostly production at night/cloudy days)
- Production peaks at noon (solar-driven, mean=284 MWh), consumption peaks at 8am (537 MWh)
- Strong seasonality: production high in summer, consumption high in winter
- Lag-24 autocorrelation ~0.96 — yesterday's same-hour value is the strongest predictor
- Weekdays 22% higher than weekends
- County 0 (Harjumaa/Tallinn) dominates with 7x the mean target of smallest county
- Business consumption 5.6x residential consumption
- `installed_capacity` correlates 0.83 with consumption target
- Solar radiation correlates 0.59 with production target
- Weather only covers ~49% of train rows (49/112 grid points mapped to counties)
- Train/test distributions consistent, no leakage detected
- Growing trend — both production and consumption increasing over time

## Feature Engineering
- 56 features total
- **Lag features**: target_lag_1, _2, _3, _24, _48, _168 + diffs
- **Rolling stats**: 24h/168h rolling mean/std, same-hour-7d rolling mean
- **Time features**: hour, dayofweek, month, day, is_weekend + cyclical (sin/cos) encodings
- **Client**: eic_count, installed_capacity (matched via data_block_id)
- **Weather**: county-aggregated historical (temperature, solar radiation, cloud cover, etc.)
- **Prices**: hourly electricity price, daily electricity stats, gas prices
- **Interactions**: capacity_per_eic, solar_x_capacity, temp_deviation

## Experiment Results

| # | Model | Validation MAE | Notes |
|---|-------|---------------|-------|
| 1 | Lag-24 baseline | 86.5208 | Yesterday's same-hour value |
| 2 | LightGBM single | 35.5532 | 827 iterations, all segments |
| 3 | LightGBM split (prod+cons) | 38.2550 | Separate models, worse than single |
| — | Local test (example files) | 40.4229 | End-to-end pipeline validated |

## Submissions
- `lgbm_submission_20260219_143821.csv` — local validation only (code competition)
- No Kaggle notebook submitted yet

## Model Details
- **Best model**: Single LightGBM (MAE objective, lr=0.05, num_leaves=255, 827 boosting rounds)
- Validation: Time-series split (blocks 0-549 train, 550-637 validation)
- Single model outperformed separate production/consumption models
- Top features by importance:
  - Production: target_roll_mean_same_hour_7d, target_lag_24, target_lag_1, solar radiation
  - Consumption: target_lag_1, target_lag_2, target_lag_24, hour, eic_count, electricity prices

## Lessons Learned
- Single model outperformed split prod/cons models — shared learning helps
- Lag features dominate importance — this is fundamentally an autoregressive problem
- Weather only partially matched (~49%) due to station-county mapping gaps
- County 0 (Harjumaa) contributes most error (MAE=149) due to large scale
- Peak error at noon — solar production is the hardest to predict
- 59% improvement over naive lag-24 baseline

## Potential Improvements
- Integrate forecast weather data (745MB, 48h ahead) instead of only historical
- Fill weather gaps by using nearest-station interpolation for unmapped counties
- Per-county or per-prediction-unit models for high-variance segments
- XGBoost / CatBoost ensemble
- Hyperparameter tuning with Optuna
- More lag features (lag at same hour same dayofweek, monthly lags)
- Target log1p transformation during training
- Build Kaggle notebook for actual submission via time-series API
