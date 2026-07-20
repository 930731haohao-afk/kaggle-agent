# playground-series-s6e7 — Predicting Student Health Risk (LIVE, deadline 2026-07-31)

**Task:** 3-class classification (health_condition: at-risk 86% / unhealthy 8% / fit 6%)
**Metric:** balanced accuracy · **Account:** huangweihaohuang · **Date:** 2026-07-20

## Result
- **Blend CV (balanced accuracy): 0.94994 → Public LB 0.94939** (gap -0.0006, honest CV).
- Models (class-balanced, 5-fold StratifiedKFold, native categoricals, determinism on):
  LGBM 0.94969 · XGBoost 0.94970 · CatBoost 0.94939 (OOF, prior-adjusted).
- Blend weights: cat 0.4 / xgb 0.4 / lgbm 0.2; minority-class prob multiplier x1.5.
- Submission: submissions/s6e7_blend_v1.csv (295,753 rows).

## Key decisions
- **balanced accuracy => class balancing MANDATORY** (class_weight/auto_class_weights/sample_weight).
  This is the OPPOSITE of experience.md's AUC insight (there, removing imbalance weighting helped) —
  did NOT blind-apply the prior; reasoned from the metric.
- **Per-class probability adjustment** tuned on OOF (scale minority-class posteriors) to maximize
  balanced accuracy — predicts more minorities than their true prior (correct for this metric).
- Categorical NaN filled as "missing" string (CatBoost requires it; LGBM/XGB handle natively).
