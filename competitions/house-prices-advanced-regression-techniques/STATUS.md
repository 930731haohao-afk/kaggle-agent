# House Prices Competition Status

**Date**: 2026-02-10

## Competition Info
- URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- Problem: Regression (predict residential house sale prices)
- Metric: RMSLE (Root Mean Squared Log Error)
- Train: 1460 rows (1458 after outlier removal), Test: 1459 rows
- Features: 80 original (38 numerical, 43 categorical)

## Key EDA Findings
- Target (SalePrice) right-skewed (1.88) — log1p transform reduces skew to 0.12
- Most NA values mean "feature not present" (no pool, no garage, etc.)
- Top predictors: OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64), TotalBsmtSF (0.61)
- Multicollinearity: GarageCars<->GarageArea, TotalBsmtSF<->1stFlrSF, GrLivArea<->TotRmsAbvGrd
- 2 outliers removed (GrLivArea >4000 sqft with low price, IDs 524 and 1299)
- 19 features with |skew| > 1.0 — log-transformed
- Categoricals with large price ranges: Neighborhood, ExterQual, KitchenQual
- No significant train-test distribution shift

## Feature Engineering
- 88 features total (from 80 original)
- Missing values: categoricals filled with "None", numericals with 0, LotFrontage by neighborhood median
- Ordinal encoding for quality features (ExterQual, BsmtQual, KitchenQual, etc.)
- Label encoding for remaining categoricals
- New features: TotalSF, TotalPorchSF, TotalBath, HouseAge, RemodAge, GarageAge, IsRemodeled, IsNewHouse, HasPool, HasGarage, HasBsmt, Has2ndFlr, HasFireplace, HasMasVnr
- Log1p applied to 51 skewed features
- Dropped low-value features: Utilities, PoolArea, PoolQC, MiscVal, MiscFeature

## Experiment Results

| # | Model | CV RMSLE | CV Std |
|---|-------|----------|--------|
| 1 | MeanPredictor | 0.3996 | 0.0000 |
| 2 | Ridge | 0.1185 | 0.0093 |
| 3 | Lasso | 0.1175 | 0.0096 |
| 4 | ElasticNet | 0.1173 | 0.0097 |
| 5 | LightGBM | 0.1230 | 0.0067 |
| 6 | XGBoost | 0.1177 | 0.0059 |
| 7 | WeightedEnsemble | 0.1141 | 0.0070 |

Ensemble weights: Ridge (0.15) + Lasso (0.15) + LightGBM (0.35) + XGBoost (0.35)

## Submissions
- `submission_ensemble_0.1141_20260210_135754.csv` → Public LB: **0.12613**

## Lessons Learned
- CV RMSLE (0.1141) vs LB (0.1261) — reasonable gap, ensemble generalizes well
- Linear models (Lasso, ElasticNet) are competitive with tree models on this dataset
- Weighted ensemble of diverse model types (linear + tree) outperforms any single model
- Proper missing value handling is critical — most NAs are informative, not random
- Log-transforming the target is essential for this competition (RMSLE metric)
- Ordinal encoding quality features preserves order information vs. one-hot encoding

## Potential Improvements
- Hyperparameter tuning with Optuna for each model
- Stacking (meta-learner on OOF predictions) instead of simple weighted average
- Target encoding for high-cardinality categoricals (Neighborhood)
- Feature selection to reduce from 88 to most impactful features
- Try CatBoost (native categorical handling)
- Polynomial features for top predictors (OverallQual * GrLivArea)

---

## 5-Stage Ablation Ladder (2026-07-20, huangweihaohuang account)

Faithful Stage 1→5 ladder via `competitions/run_stage_ladder.py` with bespoke Ames features
(`scripts/features_ladder.py`: ordinal quality ladders, TotalSF/TotalBath/TotalPorch,
HouseAge/RemodAge/GarageAge, Has* flags, OverallQual×TotalSF, LotFrontage by-neighborhood median).
Shared folds, seed=42; target modelled in log1p space.

| Stage | 1 baseline | 2 skill | 3 +linear-iter | 4 +tree-search | 5 +idea-inject |
|---|---|---|---|---|---|
| CV RMSLE | 0.1318 | 0.1261 | 0.1227 | **0.1226** | 0.1226 |

- **Champion submitted → Public LB 0.12221** (`ladder_stage5_rmsle_0.12260.csv`) — **beats the old
  bespoke LB 0.12613** by 0.0039; LB even came in slightly better than CV. A genuine win.
- Ladder monotonic; **Stage 5 = Stage 4** → idea injection null (constructive no-op).
