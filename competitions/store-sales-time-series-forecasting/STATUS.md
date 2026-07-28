# Store Sales - Time Series Forecasting Status

**Date**: 2026-02-10

## Competition Info
- URL: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- Problem: Time series regression (predict daily grocery store sales)
- Metric: RMSLE (Root Mean Squared Logarithmic Error)
- Train: 3,000,888 rows (2013-01-01 to 2017-08-15), Test: 28,512 rows (2017-08-16 to 2017-08-31)
- 1,782 series: 54 stores x 33 product families (Ecuador-based retailer)
- Supplementary data: stores, oil prices, holidays, transactions

## Key EDA Findings
- Target highly skewed: mean=358, median=11, 31.3% zeros
- Top families: GROCERY I (3,777), BEVERAGES (2,386), PRODUCE (1,349)
- Oil prices volatile: range 26-111, mean 68
- Ecuador has many national/regional/local holidays affecting sales
- Sales relatively stable across 2017 months (~465-490 avg)

## Feature Engineering
- 33 features (used training data from 2015+ for efficiency)
- Merged store metadata: city, state, store type, cluster
- Oil prices: forward-filled daily, merged to combined data
- Holidays: national holiday flag, any holiday/event flag
- Date features: year, month, day, DOW, DOY, week, quarter, is_weekend, month_start/end
- Cyclical encoding: sin/cos for month and day-of-week
- Trend: days_since_start
- Categorical encoding: family, city, state, store type
- Series aggregates: store-family monthly mean, DOW mean, overall mean
- Store-level and family-level means
- Recent lags: 7/14/28-day mean per store-family

## Experiment Results

| # | Model | CV RMSLE | CV Std |
|---|-------|----------|--------|
| 1 | LightGBM | 0.5911 | 0.1167 |
| 2 | XGBoost | 0.6041 | 0.0996 |

Plus LightGBM trained on full data (no CV score).

Ensemble: LGBM_cv (0.3) + XGB_cv (0.2) + LGBM_full (0.5)

## Submissions
- `submission_ensemble_0.5911_20260210_150841.csv` → Public LB: **0.41453**

## Lessons Learned
- Public LB (0.415) much better than CV mean (0.591) — the most recent CV split (0.427) was closest, suggesting model improves with more recent training data
- log1p/expm1 transform essential for RMSLE optimization on skewed sales data
- Store-family aggregate features are crucial for capturing hierarchy
- 3M rows is large — used 2015+ subset to balance compute time vs information

## Potential Improvements
- More granular holiday features (local/regional holidays per store city/state)
- Earthquake event (April 2016) as special feature — known to impact Ecuador retail
- Payroll day features (15th and end of month)
- Oil price rolling averages and momentum
- Transactions data as feature (though not available for test period)
- Per-family or per-store models instead of single global model
- Fourier features for complex seasonality
- Exponential smoothing / ARIMA per-series for baseline blending
