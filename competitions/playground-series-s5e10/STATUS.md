# Playground Series S5E10 — Predicting Road Accident Risk

**Date**: 2026-02-10 (Feb prior-season run) / 2026-07-07 (cross-season unit 3/5)

## Competition Info
- URL: https://www.kaggle.com/competitions/playground-series-s5e10
- Problem: Regression (predict road accident risk, 0-1)
- Metric: RMSE (minimize)
- Train: 517,754 rows, Test: 172,585 rows
- Target: `accident_risk` (0-1, mean=0.352, std=0.166, on a 0.01 grid, 98 unique values)

## Key EDA Findings
- No missing values, no train/test shift (every feature < 0.8% mean shift), 10,774 exact
  feature-duplicate rows (label noise floor, but tiny relative to 517K)
- curvature strongest predictor (Pearson 0.544 with accident_risk)
- speed_limit second (0.431); num_reported_accidents moderate (0.214)
- num_lanes / road_signs_present / school_season near-zero correlation (|r| < 0.01)
- Continuous i.i.d. target, no time/group structure -> plain shuffled KFold

## Cross-season protocol (unit 3/5)
- The Feb-2026 run (experiments.json exp 1-4) already trained LGB / XGB / a second LGBM
  + an OOF weight-searched ensemble on a 27-feature engineered set with
  KFold(5, shuffle, seed=42): CV RMSE 0.056074, Public LB 0.05558, Private 0.05583.
  Kept verbatim as the stage-1 record (reuse equivalent Feb records, don't retrain).
  Honest caveat: Feb's features were already EDA-engineered (not raw), and its third
  member was a second LightGBM rather than CatBoost.
- July run reuses the SAME folds (sklearn KFold split depends only on n_samples + seed),
  so all July scores are directly comparable to Feb's.

## Four-stage run (July, all scores 5-fold clipped-OOF RMSE, same folds throughout)
- Stage 2 (kaggle-agent skill six stages): LGB+XGB+CAT blend, OOF RMSE 0.056027.
- Stage 3 (linear self-iteration, 3 rounds):
  - r1 prior checks (BOTH resolved by controlled same-folds comparison first):
    (A) s3e9 "explicit product interactions add nothing for GBDT" prior CONFIRMED here
        — 12 base features (0.056061) beat the 27-feature set (0.056069) per solo LGB, so
        the 15 interaction columns were dropped from the pool's LGB family. Contradicts
        Feb's STATUS claim that interactions are key.
    (B) s3e14 snap-to-0.01-grid prior REJECTED for RMSE — snapping the stage-2 blend OOF
        raised RMSE (0.056095 vs clip 0.056027); squared error wants the conditional mean,
        not a grid point. Post-processing stays clip-only. r1 blend 0.056017.
  - r2 Optuna (fold-0 proxy, direct-metric objective, 40 trials 630s): tuned LGB 0.055998
    added to pool; 4-way blend 0.055982.
  - r3 seed-bag (tuned LGB seed=2024): 0.055991 solo; 5-way blend 0.055976. Stopped after
    3 rounds (diminishing returns).
- Stage 4 (tree search v3): first tree-search run for this comp; root = strongest cached
  solo (LGB_tuned_seed2024, 0.055991), digit-for-digit re-verified. Winner = node #29,
  a 14-member kitchen-sink mega-blend of the 21-solo pool, OOF RMSE **0.055968** (search
  stopped on patience: 20 evals no improvement post-burst, 39 evaluated nodes / 60 budget,
  wall 3308.1s). Logged as experiments.json exp #15; see REPORT.md for the full ladder.
- Four-stage ladder (same folds): 0.056074 -> 0.056027 -> 0.055976 -> 0.055968 (1->4 +0.19%).

## Submissions
- Feb: `submission_ensemble_0.0561_20260210_152814.csv` -> Public 0.05558, Private 0.05583.
- July best (stage 4): OOF reproduction gate PASSED (scripts/06_rebuild_tree_best.py) — all
  21 members retrained max|dOOF|~1e-15, recovered blend RMSE 0.055968 == target, weights match.
  Submission `sub_tree_best_0.05597_20260707_182744.csv` built; late-submission attempt
  returned 403 (competition closed) -> recorded CV-only.

## Lessons Learned (this run)
- Feb's "interactions are key" did not survive a controlled on/off check at this metric —
  priors must be re-verified per comp before trust (same discipline as s4e11's
  is_unbalance rejection).
- Extremely stable CV (fold std ~0.0001); the whole four-stage ladder moves within a
  ~0.0001 RMSE band, so gains are real but small — an honest "last-mile" competition.
