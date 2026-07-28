# home-data-for-ml-course — Competition Status

**Competition**: Housing Prices Competition for Kaggle Learn Users
**Task**: Regression — predict house sale prices (Ames, Iowa)
**Metric**: MAE (Mean Absolute Error) on SalePrice
**Dataset**: 1460 train / 1459 test, 80 features
**Best Public LB**: 12,615.63

## Experiments

| # | Name | Models | CV MAE | Public LB | Notes |
|---|------|--------|--------|-----------|-------|
| 1 | lgb_xgb_ensemble_v1 | LGB + XGB | — | 13,657.73 | Basic ordinal encoding, interactions |
| 2 | 5model_stack_v2 | LGB+XGB+Ridge+ENet+GBR | 14,200 | 13,658.12 | Unconstrained weights, negative weights overfit |
| 3 | constrained_4model_stack_v3 | Ridge(34%) + GBR(66%) | 13,293 | 12,615.63 | Outlier removal, skew correction, Huber loss |

## Key Features
- 107 engineered features: ordinal quality encoding, area interactions, quality*area, log transforms, ratios
- Outlier removal: 2 houses with GrLivArea > 4000 and low price
- Log1p on 66 highly skewed features
- GBR with Huber loss most robust to outliers
- Neighborhood target encoding (median price from train)

## What Worked
- **Outlier removal** (2 extreme GrLivArea houses): biggest single improvement
- **Skew correction** (log1p on skewed features): helps linear models
- **GBR with Huber loss**: robust to remaining outliers, best individual model
- **Ridge + GBR blend**: Ridge provides regularization, GBR provides nonlinearity
- **Constrained positive weights**: more stable than unconstrained optimization

---

## 5-Stage Ablation Ladder (2026-07-20, huangweihaohuang account)

Faithful Stage 1→5 ladder via `competitions/run_stage_ladder.py` with bespoke Ames features
(shared module with house-prices). Shared folds, seed=42; target modelled in log1p space.

| Stage | 1 baseline | 2 skill | 3 +linear-iter | 4 +tree-search | 5 +idea-inject |
|---|---|---|---|---|---|
| CV MAE | 15551 | 14873 | 14386 | **14321** | 14321 |

- **Champion submitted → Public LB 12907** (`ladder_stage5_mae_14321.29312.csv`) — ahead of the old
  v1/v2 (13658) but **behind the old best v3 (12616)**. Ladder monotonic; Stage 5 = Stage 4 (null).
- **Why behind v3 (diagnosed):** this ladder is **GBDT-only**, but home-data's winning recipe is
  **linear-model-heavy + MAE-robust** — old v3 = Ridge(34%) + GBR(66%) with outlier removal + Huber.
  Diagnostic: on identical features an L1(MAE)-objective LGB scores CV MAE 15038 vs default-L2 15934,
  so the generic ladder's default regression objective is also mildly metric-misaligned for MAE.
- **To pass 12616:** add Ridge/ElasticNet members + an L1-objective GBDT + `GrLivArea>4000` outlier
  removal (not yet done).
