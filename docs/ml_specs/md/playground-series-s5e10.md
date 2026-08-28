# ML Specification Report — playground-series-s5e10

### Road Accident Risk Prediction · our from-scratch agent (GBDT pool + tree-search v3)

> *Figures grounded in the competition's `facts.json`, `config.yaml`, `eda_summary.json`, `STATUS.md`, and `experiments_tree_v3.json`, with benchmark and timing figures from the external `s5e10` record.*

## Overview

The task is to predict a continuous road accident-risk score (0–1) from road and environmental features (metric: RMSE, minimize). We solve it with a from-scratch pipeline — a pool of gradient-boosted trees (LightGBM / XGBoost / CatBoost) refined through a four-stage ladder that ends on a 60-node-budget tree search. Our champion, tree-search node #29 (a 14-member kitchen-sink mega-blend of the 21-solo pool), scores **CV OOF RMSE 0.055968** (submission tagged 0.05597); the NVIDIA reproduce-agent, copying a public XGBoost-residuals kernel, reaches 0.05609 — a **dead heat** (our number is marginally lower, but the two are called a tie), with NVIDIA's small structural edge coming from a physics-prior trick.


**Why it matters.** Quantifying road-segment accident risk informs infrastructure investment, speed-limit setting, and safer routing, with the goal of reducing injuries and fatalities.

---

## Data

**Purpose of Data.** Predict a road segment's accident-risk score — a continuous regression Playground Series episode, target `accident_risk` ∈ [0, 1]. **Data Format** is clean **tabular CSV**, a mix of numeric and low-cardinality categorical columns. **Data Volume** is **517,754 training rows / 172,585 test rows**, with **12 raw predictive features** (8 numeric, 4 categorical) once `id` and the target are removed.

**Data Quality** is high: **no missing values** in either split, **no train↔test shift** (every feature moves < 0.8% in mean between splits), and no high-collinearity feature pairs. The one wrinkle is **10,774 exact feature-duplicate rows** in train — a small label-noise floor, tiny relative to 517K. The target is continuous with **98 unique values on a 0.01 grid** (mean 0.352377, std 0.166417, median 0.34, skew 0.378418), which motivates a plain regression treatment. The strongest EDA signals, all consistent with road-safety intuition, were:

- **curvature** is the dominant predictor (Pearson r ≈ +0.544, Spearman ≈ +0.547).
- **speed_limit** is second (r ≈ +0.431); **num_reported_accidents** is moderate (r ≈ +0.214).
- **lighting = night** notably raises risk (≈ 0.47 vs ≈ 0.30 in daylight).
- **num_lanes, road_signs_present, school_season** carry near-zero linear signal (|r| < 0.01) — candidate noise columns.

**Annotation Guidelines.** The label is `accident_risk`, a continuous score in [0, 1]; submissions are **real-valued predictions** scored by RMSE — a squared-error target that wants the conditional mean, not a hard class or a grid point.

**Feature Set.** The pipeline models the **12 raw base features** directly (GBDTs handle the cardinality-3 categoricals natively). A Feb prior-season run had used a 27-feature engineered set (12 base + 15 explicit interaction columns); a controlled same-folds check in July found the **12 base features beat the 27-feature set** (solo LGB 0.056061 vs 0.056069), so the 15 interaction columns were dropped from the pool's LGB family — confirming the s3e9 "explicit product interactions add nothing for GBDT" prior and contradicting Feb's claim that interactions were key.

| Group | Features |
|-------|----------|
| Numeric (8) | `num_lanes`, `curvature`, `speed_limit`, `road_signs_present`, `public_road`, `holiday`, `school_season`, `num_reported_accidents` |
| Categorical (4, cardinality 3) | `road_type`, `lighting`, `weather`, `time_of_day` |
| Dropped after on/off check | 15 interaction columns from the Feb 27-feature set |

**Splitting strategy.** A single canonical **shuffled KFold 5-Fold CV (seed = 42)**, chosen because the target is a continuous i.i.d. score with no time or group structure. The same fold partition is **reused identically across the Feb and July runs** — the sklearn KFold split depends only on `n_samples` + seed — so every score in this report is directly comparable. The leak-free discipline at stake is **fold-safe target encoding**: any target-derived encoding is computed on the model's own fold partition (TE-fold == model-fold), never on held-out rows — the same technique the benchmark reproduction is scored on (`repro_type` = fold-safe TE). Here the categoricals are cardinality-3, so encoding impact is minor, but the safeguard is honored throughout. The leaderboard is closed (late submission returns `403`), so July scores are **CV-only**; the Feb prior-season submission (CV 0.056074 → Public 0.05558 / Private 0.05583, gap ≈ 0.0002–0.0003) anchors CV trustworthiness.

## Models & Architecture

**Purpose of Architecture.** A continuous risk regressor that minimizes RMSE. **Architecture Type** is a **gradient-boosted decision-tree ensemble** — LightGBM as the primary learner, with XGBoost and CatBoost members, combined by a weighted blend; the champion is a **14-member mega-blend** discovered at tree-search node #29, drawn from a 21-solo candidate pool.

Each member takes the same **Input Format** — a numeric feature matrix with the four cardinality-3 categoricals handled natively — of **Input Dimension 12 base features** per row (a 1-D vector; there is no spatial/sequence structure).

**Architecture Description.** Each GBDT member is an additive ensemble of decision trees grown by histogram-based, leaf-wise gradient boosting (the tuned LGB backbone: `num_leaves` = 230, `max_depth` = 11, `min_child_samples` = 10, `feature_fraction` ≈ 0.874, `bagging_fraction` ≈ 0.757). The blend layer sits on top as a convex weighted average of the members' out-of-fold predictions, with weights found by Dirichlet sampling (k = 800, seed = 42) plus coordinate-ascent refinement, minimizing clipped OOF RMSE. **Model Complexity** is best expressed as trees × leaves rather than a dense parameter count; the champion blends 14 such GBDT members. Its weight mass is concentrated: the dominant member is **node #15 (LGB), weight 0.3359** (solo RMSE 0.055987), with node #17 (0.0984), node #6 (0.0898) and node #19 (0.0892) next, tailing into two CatBoost and two XGBoost members at small weights.

## Training procedures

Training proceeds as a **four-stage ladder** on the shared 5-fold split, each stage measured on the same clipped-OOF RMSE scale:

| Stage | Configuration | OOF RMSE |
|-------|---------------|---------:|
| 1 | Feb record: LGB / XGB / 2nd-LGBM + OOF weight-searched ensemble (27-feature set) | 0.056074 |
| 2 | skill pipeline: LGB + XGB + CAT blend (12 base features) | 0.056027 |
| 3 | linear self-iteration, 3 rounds (prior checks → Optuna tune → seed-bag), 5-way blend | 0.055976 |
| 4 | tree-search `harness_v3` (60-node budget, node #29 = 14-member mega-blend) | **0.055968** |

Stage 3 unfolds in three rounds: **r1** resolved two priors by controlled same-folds comparison — the interaction-column prior (confirmed, above) and the s3e14 snap-to-0.01-grid prior (**rejected**: snapping the blend OOF *raised* RMSE to 0.056095 vs 0.056027, so post-processing stays clip-only) — closing at blend 0.056017; **r2** ran Optuna (fold-0 proxy, direct-metric objective, 40 trials, 630 s), adding a tuned LGB (0.055998) for a 4-way blend at 0.055982; **r3** seed-bagged the tuned LGB (seed = 2024, solo 0.055991) for a 5-way blend at 0.055976, then stopped on diminishing returns.

The **Loss Function** is squared error (LightGBM `regression` / `rmse` objective) with model selection on RMSE. The **Optimization Algorithm** is gradient boosting (histogram, leaf-wise) — **not** SGD/ADAM — with blend weights optimized by Dirichlet search + coordinate-ascent. The **Learning Rate** is the boosting shrinkage (tuned LGB `learning_rate` ≈ 0.034); there is **no Learning Rate Scheduler** (*N/A — GBDT convergence is governed by CV early-stopping, not a schedule*) and **no Batch Size** (*N/A — full-dataset histogram boosting, not mini-batched*).

**Training Duration** for the stage-4 tree search was **~3,308 s** of wall-clock over **39 evaluated nodes** (of a 60-node budget; the search stopped on patience — 20 evaluations without improvement post-burst); the Optuna tune was a separate 40 trials / 630 s, and a solo LGB member fits in ~70–90 s. **Training Memory** was not explicitly capped — a 518K × 12 matrix is trivial on the single arm64 CPU machine, peak not separately measured. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue here is **cross-season experience injection** — this is unit 3/5 of a cross-season sweep, carrying distilled priors (s3e9 interactions, s3e14 snap-to-grid) that are re-verified per competition. **Data Augmentation** is *N/A (tabular)*; the analogues are feature selection and seed-bagging.

**Reproducibility Standards** are strict: fixed seeds (fold seed = 42; Optuna and seed-bag seeds logged), LightGBM determinism flags, and a digit-for-digit reproduction gate (`06_rebuild_tree_best.py`) that retrains all 21 pool members with `max|ΔOOF| ≈ 1e-15` versus cached OOFs and recovers blend RMSE 0.055968 == target with matching weights. This was a **native v3 run — no replay was needed** (faithful replay is native), so the champion is the search's own committed output rather than a reconstruction.

## Inference procedures

**Decision Threshold** is *N/A* — RMSE is a squared-error regression metric, so we submit raw real-valued predictions and never threshold. The only post-processing is **clip to [0, 1]** (the snap-to-0.01-grid alternative was tested and rejected, as it raised RMSE). **Inference Duration** was not separately profiled; GBDT scoring of the 172,585 test rows is sub-second per member on CPU, and **Inference Memory** is negligible (tree inference is lightweight, not capped).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict the test segments' accident-risk scores. **Performance Metrics.** RMSE (minimize) is the sole competition metric; our stage-4 champion reaches **CV OOF RMSE 0.055968** (submission file tagged 0.05597), improving 0.056074 → 0.056027 → 0.055976 → 0.055968 across the four stages (+0.19% stage-1→4). The CV is extremely stable — fold std ≈ 0.0001, so the whole ladder moves within a ~0.0001 RMSE band; gains are real but small, an honest "last-mile" competition. The Feb prior-season submission (Public 0.05558 / Private 0.05583) is reference only, as the July run is CV-only.

**Performance Benchmarking.** The natural comparator is the **NVIDIA reproduce-agent**, which copies the strongest public kernel:

| Agent | Approach | RMSE | Note |
|-------|----------|-----:|------|
| NVIDIA | reproduces `cdeotte/xgb-boosting-over-residuals-cv-0-05595` | 0.05609 | physics-prior trick |
| Our agent | from-scratch GBDT pool + tree-search v3 | **0.05597** | tie |

The result is a **dead heat** (`winner` = tie): our champion's 0.05597 is marginally lower (better) than NVIDIA's 0.05609, but the two are declared even. The benchmark comparison is at the **technique** level — specifically fold-safe target encoding, the leak-free discipline both agents honor — and the small structural difference is that **NVIDIA's edge comes from a physics-prior trick** in the copied residual-boosting kernel. On this last-mile regression problem the two approaches converge, unlike the older churn-style episodes where a well-optimized public kernel retained a clearer edge.

---

*Fields marked "not recorded": training peak memory (not separately measured); inference duration and memory (not separately profiled); leaderboard for the July run (competition closed — CV-only).*

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions were re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run were produced by the separate credentialed operator scoring step and are recorded in `benchmark_results/rerun_three_way.csv`; the figures below revise nothing in the sections above.

- **Lane**: done 2026-08-17, 72 min — audit flagged raw library opens; transcript timeline shows all of them strictly after the submission was written and validated, so the score stands; the lane's post-run hand write-back to the shared knowledge library was reverted before any later lane ran.
- re-run local CV: **RMSE 0.055971** (honest 0.055973), 29-member NNLS blend selected by leave-fold-out honest score.
- Public / private leaderboard for this re-run (`benchmark_results/rerun_three_way.csv`): **Public 0.05554 / Private 0.05576**; `winner_priv` = **mine** (AIDE 0.05579, NVIDIA 0.05588).
