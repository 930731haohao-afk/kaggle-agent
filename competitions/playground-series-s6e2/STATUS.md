# Playground Series S6E2 — Predicting Heart Disease

**Date**: 2026-02 (Feb prior-season run) / 2026-07 (cross-season unit 5/5, final)

## Competition Info
- URL: https://www.kaggle.com/competitions/playground-series-s6e2
- Problem: Binary classification (Heart Disease Presence/Absence)
- Metric (this cross-season unit, per config.yaml): **ROC-AUC (maximize)** — submit
  raw positive-class probabilities, never hard 0/1 labels
- Train: 630,000 rows, Test: 270,000 rows, 13 raw UCI numeric features
- Target: `Heart Disease` (Presence -> 1 / Absence -> 0), positive rate 0.44834

## Key EDA Findings
- No missing values, no feature-duplicate rows, fairly balanced (imbalance ratio 1.23)
- Thallium strongest raw predictor (point-biserial 0.605776), then Chest pain type
  (0.460684), Exercise angina (0.441864), Max HR (-0.440985), Number of vessels fluro
  (0.438604), ST depression (0.430641)
- BP near-zero (-0.005181), FBS over 120 (0.033570), Cholesterol (0.082753) weak
- Binary classification, no time/group structure -> StratifiedKFold on the class label

## Cross-season protocol (unit 5/5, the final cross-season unit)
- The Feb-2026 run's v1 ensemble (experiments.json exp 1) trained LGB+XGB + an OOF
  weight-searched ensemble on a 34-feature engineered set with StratifiedKFold(5,
  shuffle, seed=42): OOF AUC 0.95498. Kept verbatim as the stage-1 record (reuse the
  fold-identical Feb ensemble, don't retrain). StratifiedKFold's split depends only on
  the target + seed, so July's folds reproduce Feb v1's exactly and all July scores are
  directly comparable.
- Feb's later variants (v2 10-fold, v3 5-seed x 5-fold, v4 10-seed x 10-fold, v5
  58-feature) used NON-canonical CV schemes and are kept as context only (never ladder
  rungs — never compare scores across CV schemes). Feb's best real Kaggle submission was
  v3, Public LB AUC 0.95332.
- Reproducibility: all July training is bit-reproducible cross-process (LightGBM
  `deterministic=True` + `force_row_wise=True`, all models `num_threads=16`) so the
  stage-4 OOF-reproduction gate holds digit-for-digit — pinned from the very first run
  (the s6e1 hard-won fix, applied here from the start; verified: 3-model cross-process
  max|diff|=0 before the pipeline began).

## Four-stage run (July, all scores 5-fold OOF AUC, same folds throughout)
- Stage 2 (kaggle-agent skill six stages): LGB+XGB+CAT blend, OOF AUC 0.955197 (LGB
  0.955009, XGB 0.955048, CAT 0.955151; blend weights LGB 0.15 / XGB 0.25 / CAT 0.60).
- Stage 3 (linear self-iteration, 3 rounds):
  - r1 prior checks (BOTH resolved by controlled same-folds comparison first):
    (A) s3e9 "explicit product/threshold columns add nothing for GBDT" + s3e7/s3e14
        "too many features regress" prior CONFIRMED here — the 13 raw UCI features
        (0.955238) BEAT the full 33-feature set (0.955009) per solo LGB (delta
        +0.000229), so the 20 engineered columns do NOT help; base13 adopted for the
        LGB pool member. This is the OPPOSITE outcome to s6e1 (where interactions
        helped) and the SAME as s5e10 — same prior, opposite verdict per comp.
    (B) s3e3/s4e1 AUC prior "adding class-imbalance weighting hurts a ranking metric":
        is_unbalance=True (0.954993) vs default (0.955009), delta -0.000016 -> prior
        CONFIRMED (weighting hurts AUC), default unweighted kept — a third large-sample
        confirmation after s3e3 and s4e1. r1 blend 0.955310.
  - r2 Optuna (fold-0 proxy, direct-AUC objective, 40 trials 515.1s): tuned base13 LGB
    0.955501 (shallow depth-4, strong regularization) added to pool; 4-way blend
    0.955501 (tuned LGB dominates).
  - r3 seed-bag (tuned LGB seed=2024, 0.955501): 5-way blend 0.955510. Stopped after 3
    rounds (diminishing returns).
- Stage 4 (tree search v3): first tree-search run for this comp; root = strongest cached
  solo (LGB_tuned_seed2024, 0.955501), digit-for-digit re-verified. Winner = node #43, a
  37-member kitchen-sink mega-blend of the full solo pool (8 non-zero weights, dominant
  member node #14 weight 0.3523), OOF AUC **0.955529** (search stopped on the 60-node
  hard budget cap, 60 evaluated nodes, wall 7958.9s, 0 failed, 0 dedup). Logged as
  experiments.json exp 16; see REPORT.md for the full ladder.
- Four-stage ladder (same folds): 0.95498 -> 0.955197 -> 0.955510 -> 0.955529
  (stage 1->4 +0.0575%).

## Submissions
- Feb: v3 got a real Kaggle leaderboard on the SAME AUC metric (Public LB 0.95332) —
  it confirms the Feb pipeline produced valid ranked test predictions, but is on a
  different (multi-seed) CV scheme so it is not a numeric anchor for this single-fold
  July ladder.
- July best (stage 4): OOF-reproduction gate via scripts/06_rebuild_tree_best.py — all
  37 members retrained, digit-for-digit OOF check, weight-search replay, then a
  raw-probability submission. Late-submission attempt expected to return 403
  (competition closed) -> recorded CV-only.

## Lessons Learned (this run)
- The s3e9 "explicit interactions add nothing" prior got its OPPOSITE verdict vs s6e1:
  here the 13 raw UCI features BEAT the 33-feature engineered set by 0.000229 AUC, so
  interactions were rejected (same as s5e10). Feb built 20 engineered columns and
  believed them essential; a clean same-folds check showed the raw columns alone score
  higher for LGB. Same prior, opposite outcomes across comps — priors must be
  re-verified, both confirmations and rejections logged honestly.
- The AUC imbalance-weighting prior (s3e3/s4e1) held a third time on a large sample:
  is_unbalance hurt AUC by 0.000016. Small effect (data is fairly balanced), consistent
  direction.
- Determinism pinned from the start (the s6e1 fix) meant the stage-4 gate passed without
  a mid-run scramble: the mega-blend's members reproduce bit-identically cross-process.
- Extremely stable CV: the whole four-stage ladder moves within a ~0.0006 AUC band, so
  gains are real but small — an honest "last-mile" competition, and the mega-blend still
  extracted the final 0.000019 over the linear stage's best blend.

---

**Final Status** (cross-season unit 5/5 complete): four-stage ladder
0.95498 -> 0.955197 -> 0.955510 -> 0.955529, mega-blend winner, gate-verified, CV-only
(competition closed).
