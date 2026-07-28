# Forest Cover Type Prediction Status

**Date**: 2026-02-10

## Competition Info
- URL: https://www.kaggle.com/competitions/forest-cover-type-prediction
- Problem: 7-class classification (predict forest cover type from cartographic variables)
- Metric: Accuracy
- Train: 15,120 rows (perfectly balanced: 2,160 per class), Test: 565,892 rows
- Features: 54 (10 continuous + 4 wilderness area + 40 soil type, all numeric, no missing values)

## Key EDA Findings
- Perfectly balanced training set (2,160 samples per class)
- Unusual train/test ratio (15K vs 566K) — high overfitting risk
- Constant columns: Soil_Type7, Soil_Type15 (all zeros) — dropped
- Elevation is the strongest discriminator (range: 2,223 for Type 4 to 3,363 for Type 7)
- All features already numeric, no missing values

## Feature Engineering
- 67 features total (from 52 after dropping 2 constant columns)
- Distance: Euclidean distance to hydrology, road+fire sum/diff
- Hillshade: mean of 3 hillshade features, pairwise diffs
- Aspect: sin/cos cyclical encoding
- Elevation: interactions with vertical/horizontal distance to hydrology
- Log transforms: on 3 skewed distance features
- Aggregations: wilderness type and soil type as single ordinal columns

## Experiment Results

| # | Model | CV Accuracy | CV Std |
|---|-------|-------------|--------|
| 1 | RandomForest | 0.8767 | 0.0055 |
| 2 | ExtraTrees | 0.8860 | 0.0035 |
| 3 | LightGBM | 0.8888 | 0.0046 |
| 4 | XGBoost | 0.8819 | 0.0060 |

Ensemble weights: RF (0.2) + ExtraTrees (0.2) + LightGBM (0.3) + XGBoost (0.3)

## Submissions
- `submission_ensemble_0.8888_20260210_142851.csv` → Public LB: **0.77510**, Private LB: **0.77510**

## Lessons Learned
- CV accuracy (~0.889) vs Public LB (0.775) — significant gap likely due to small train set (15K) vs large test set (566K)
- XGBoost requires 0-indexed class labels (had to shift Cover_Type from 1-7 to 0-6)
- LightGBM best single model on CV; ExtraTrees second best
- Tree-based models all perform similarly (87-89% CV accuracy range)

## Potential Improvements
- Increase number of estimators (1000+) and tune more aggressively
- Add more feature interactions (elevation x wilderness, soil x distance)
- Try stacking ensemble instead of simple weighted average
- Neural network approaches (TabNet, deep embeddings)
- Semi-supervised learning leveraging the large unlabeled test set
- More aggressive hyperparameter tuning with Optuna/Bayesian optimization
