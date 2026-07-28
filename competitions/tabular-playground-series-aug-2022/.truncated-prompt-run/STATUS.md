# tabular-playground-series-aug-2022 — Competition Status

**FINAL: LogisticRegression(C=0.02) on 5 features, GroupKFold(product_code) OOF AUC 0.59131 (with per-product rank normalization).**
**Submission: `submission.csv` (= `submissions/submission_lr_rankpp_0.59131.csv`), 20,775 rows, probs rank-normalized per test product.**
**Key: train products A–E vs test F–I disjoint — simple LR generalizes better than LGBM (0.5906 vs 0.5842 raw OOF); rank-pp adds +0.0007.**

## Competition Info
- Predict product failure (binary), metric ROC-AUC, 26,570 train / 20,775 test.
- Train product_code A–E, test F–I — **disjoint groups**; CV = leave-one-product-out GroupKFold(5).
- Missing values grow m3→m17 (1.4%→8.6%); loading ~1%.

## Current Best Score
| Model | CV (OOF AUC, rank-pp) |
|---|---|
| LR(C=0.02), [loading, measurement_17, m3_na, m5_na, measurement_2] | **0.59131** |
| rank-blend LR+LGB (w_lgb=0.05) | 0.59133 (noise-level, rejected) |
| LGBM shallow, full features | 0.58415 (raw) |

## Pipeline Summary
### Features (`scripts/features.py`)
- measurement_17 imputed per product via HuberRegressor on its 4 most-correlated measurements; other numerics per-product median.
- NA indicators: m3_na (missing = lower fail rate 0.160 vs 0.213), m5_na (higher 0.254), na_count.
- Per-product z-scaling (feature-only stats, leakage-free). Raw loading beat log-loading after per-product scaling.

### Models
- LR sweep (8 subsets × C grid): winner `[loading, measurement_17, m3_na, m5_na, measurement_2]`, C=0.02.
- LGBM (shallow, deterministic, 10 threads) clearly worse — overfits per-product noise, test products unseen.
- Blend LR+LGB: +0.00002 — rejected per noise-level-blend prior.

### Submission
- `scripts/make_submission.py`: full-train LR refit, per-product rank normalization on test predictions.
- Validated vs sample_submission: shape, id order, [0,1] range, no NaN.

## Key Observations
- AUC pooled across disjoint products → per-product rank normalization consistently +0.0007.
- Attributes constant within product_code — useless for cross-product generalization.
- 7 experiments in `experiments.json`.

## Potential Improvements
- Winner-style KNN imputation of all measurements; small feature-set XGB linear booster; calibrated per-product blending.
