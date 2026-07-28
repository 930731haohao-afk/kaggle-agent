# playground-series-s6e7 — Predicting Student Health Risk (LIVE, deadline 2026-07-31)

**Task:** 3-class classification (health_condition: at-risk 86% / unhealthy 8% / fit 6%)
**Metric:** balanced accuracy · **Account:** huangweihaohuang · **Date:** 2026-07-20

## Result (best = v3b)
- **v3b blend CV 0.95005 → Public LB 0.94985** — current best submission.
- **v1 blend CV 0.94994 → Public LB 0.94939** — first submission.
- **Real-LB progress: 0.94939 → 0.94985 (+0.00046)** via a wider model pool + finer weight search.
- v2 (Optuna-tuned LGBM + seed-bag, CV 0.94998) written but NOT submitted (superseded by v3b).

## Optimization trajectory (ceiling analysis)
- **v1**: LGBM+XGB+CatBoost class-balanced 5-fold blend + per-class prior-adjust. CV 0.94994.
- **v2**: Optuna-tune strongest (LGBM) on fold-0 proxy, add tuned + seed-bag to pool, re-blend.
  CV 0.94998 (+0.00004) — decision-opt at ceiling, GBDT-family diversity exhausted.
- **v3/v3b**: add STRUCTURALLY DIVERSE non-GBDT learners — a GPU **MLP** (class-weighted CE) and
  **ExtraTrees** (bagging) — then re-blend 7 members. CV 0.95005 (+0.00007 over v2).
  - MLP solo 0.94861, ET solo 0.94704 (both valid, GBDT-level) but the blend gives them only
    **1.1% + 5.0% weight** → non-GBDT diversity contributes ~0 to CV. **Ceiling confirmed.**
  - Yet on the real LB v3b beat v1 by +0.00046 (finer weight tuning + the small diverse share help).
- Models are all near-identical on OOF (0.9470–0.9498); the metric ceiling is ~0.950 here.
- Submissions: s6e7_blend_v1.csv, s6e7_blend_v3b.csv (295,753 rows each).

## Key decisions
- **balanced accuracy => class balancing MANDATORY** (class_weight/auto_class_weights/sample_weight).
  This is the OPPOSITE of experience.md's AUC insight (there, removing imbalance weighting helped) —
  did NOT blind-apply the prior; reasoned from the metric.
- **Per-class probability adjustment** tuned on OOF (scale minority-class posteriors) to maximize
  balanced accuracy — predicts more minorities than their true prior (correct for this metric).
- Categorical NaN filled as "missing" string (CatBoost requires it; LGBM/XGB handle natively).
- **Numeric NaN breaks the MLP** (7 NUM cols have 1–11% NaN): GBDTs eat NaN natively, but an MLP
  fed NaN inputs diverges to NaN weights → degenerate 0.333 output. Fix = per-fold median impute +
  missing-indicator features (missingness is informative) + grad clipping. Only then is the NN a
  fair diversity test.
