# Playground Series S6E1 — Predicting Student Exam Scores

**Date**: 2026-02-10 (Feb prior-season run) / 2026-07-07 (cross-season unit 4/5)

## Competition Info
- URL: https://www.kaggle.com/competitions/playground-series-s6e1
- Problem: Regression (predict student exam scores)
- Metric (this cross-season unit, per config.yaml): **R2 (maximize)**
- Train: 630,000 rows, Test: 270,000 rows
- Target: `exam_score` (range 19.6–100, mean 62.51, std 18.92, 805 unique values)

## Key EDA Findings
- No missing values, no feature-duplicate rows, no train/test shift (every numeric
  feature < 0.04% mean shift)
- study_hours strongest predictor by far (Pearson 0.762267 with exam_score)
- class_attendance second (0.360954); sleep_hours moderate (0.167410)
- age near-zero (0.010472); gender minimal effect (Feb STATUS)
- Continuous i.i.d. target, no time/group structure -> plain shuffled KFold

## Cross-season protocol (unit 4/5)
- The Feb-2026 run (experiments.json exp 1-4) already trained LGB / XGB / a second LGBM
  + an OOF weight-searched ensemble on a 22-feature engineered set with
  KFold(5, shuffle, seed=42): OOF R2 0.786414. Kept verbatim as the stage-1 record
  (reuse equivalent Feb records, don't retrain). Honest caveat: Feb's features were
  already EDA-engineered (not raw), and its third member was a second LightGBM rather
  than CatBoost.
- July run reuses the SAME folds (sklearn KFold split depends only on n_samples + seed),
  so all July scores are directly comparable to Feb's.
- Reproducibility: all July training is bit-reproducible cross-process (LightGBM
  `deterministic=True` + `force_row_wise=True`, all models `num_threads=16`) so the
  stage-4 OOF-reproduction gate holds digit-for-digit.

## Four-stage run (July, all scores 5-fold clipped-OOF R2, same folds throughout)
- Stage 2 (kaggle-agent skill six stages): LGB+XGB+CAT blend, OOF R2 0.786793.
- Stage 3 (linear self-iteration, 3 rounds):
  - r1 prior checks (BOTH resolved by controlled same-folds comparison first):
    (A) s3e9 "explicit product interactions add nothing for GBDT" prior REJECTED here
        — the 22-feature set (0.785938) BEAT the 11 base features (0.785844) per solo
        LGB (delta -0.000094), so interactions genuinely help on this data; the
        22-feature set was kept. This VINDICATES Feb's STATUS claim that interactions
        matter, and is the exact opposite of s5e10 where the same prior was confirmed.
    (B) s3e1/s3e11 clip-to-range prior ADOPTED (harmless safety net): clipping the
        stage-2 blend OOF to [0,100] moved R2 by +0.000000 (raw 0.786792 -> clip
        0.786793); kept as the only post-processing. r1 blend 0.786794.
  - r2 Optuna (fold-0 proxy, direct-metric objective, 40 trials 715.9s): tuned LGB
    0.786647 added to pool; 4-way blend 0.786998.
  - r3 seed-bag (tuned LGB seed=2024): 0.786613 solo; 5-way blend 0.787060. Stopped
    after 3 rounds (diminishing returns).
- Stage 4 (tree search v3): first tree-search run for this comp; root = strongest cached
  solo (LGB_tuned, 0.786647), digit-for-digit re-verified. Winner = node #27, a 21-member
  kitchen-sink mega-blend of the 21-solo pool (17 non-zero weights), OOF R2 **0.787183**
  (search stopped on patience: 20 evals no improvement post-burst, 46 evaluated nodes /
  60 budget, wall 4592.9s). Logged as experiments.json exp #15; see REPORT.md for the
  full ladder.
- Four-stage ladder (same folds): 0.786414 -> 0.786793 -> 0.787060 -> 0.787183 (1->4 +0.0978%).

## Submissions
- Feb: submission got a real Kaggle leaderboard, but the LB was scored on **RMSE**
  (Public 8.70380, Private 8.72876) — a DIFFERENT metric from this unit's R2, so it is
  not numerically comparable to the R2 ladder; it only confirms the Feb pipeline
  produced valid ranked test predictions.
- July best (stage 4): OOF reproduction gate PASSED (scripts/06_rebuild_tree_best.py) — all
  21 members retrained max|dOOF|=0 (bit-identical), recovered blend R2 0.787183 == target,
  weights match. Submission `sub_tree_best_0.78718_20260707_221451.csv` built; late-submission
  attempt returned 403 (competition closed) -> recorded CV-only.

## Lessons Learned (this run)
- Feb's "interactions are key" claim SURVIVED a controlled on/off check at this metric
  (interactions help by 0.000094 R2) — the exact opposite outcome to s5e10, where the same
  s3e9 prior was confirmed. Same prior, opposite verdict per comp: priors must be
  re-verified, and both confirmations and rejections are logged honestly.
- LightGBM's timing-dependent auto row-wise/col-wise histogram choice makes training
  non-reproducible across processes (root retrain vs cached OOF differed by max 0.87 per
  sample at first); pinning force_row_wise + num_threads fixed it to 0.0, which the strict
  gate requires.
- Extremely stable CV (fold-level, tiny spread); the whole four-stage ladder moves within a
  ~0.0008 R2 band, so gains are real but small — an honest "last-mile" competition.
