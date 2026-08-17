# ML Specification Report — playground-series-s6e1

### Student Exam-Score Prediction · our from-scratch agent (GBDT pool + tree-search v3)

> *Figures grounded in the competition's `facts.json`, `eda_summary.json`, `STATUS.md`, and `experiments_tree_v3.json`, with benchmarking and training-time fields from the external-facts record.*

## Overview

The task is to predict students' exam scores from study habits and demographics (metric: R²). We solve it
with a from-scratch pipeline — a pool of gradient-boosted trees (LightGBM / XGBoost / CatBoost) refined by a four-stage
ladder and a 60-node tree search that ends on a 21-member kitchen-sink mega-blend. Our champion scores
**CV R² 0.787183**; the NVIDIA reproduce-agent, running a core GBM copy of a public kernel, reaches 0.7863 — a
statistical **tie**. The honest context is that the public leaderboard's best solution is a neural net that our GBDT
agent does not reproduce, so the head-to-head here is GBM-vs-GBM.


**Why it matters.** Predicting exam outcomes from study and lifestyle factors helps educators flag at-risk students early and target support where it matters.

---

## Data

**Purpose of Data.** Predict a student's continuous `exam_score` from study habits and demographics — a
regression Playground Series episode. **Data Format** is clean **tabular CSV**, a mix of numeric and low-cardinality
categorical columns. **Data Volume** is **630,000 training rows / 270,000 test rows**, with 11 raw predictive features
once `id` and the target are removed (4 numeric, 7 categorical).

**Data Quality** is high: **no missing values** in either split, **0 duplicate rows**, and no train↔test mean shift
(every numeric feature drifts < 0.04%). The target `exam_score` is continuous and near-symmetric — range **19.6–100**,
**mean 62.51, std 18.92, 805 unique values**, skew −0.048 — so no log transform is warranted. The strongest EDA signals,
all consistent with domain intuition, were:

- **study_hours** is the dominant predictor by far (Pearson **0.762267** / Spearman 0.770 with `exam_score`).
- **class_attendance** is second (Pearson **0.360954**); **sleep_hours** is moderate (**0.167410**).
- **age** is effectively noise (Pearson 0.010472) and gender shows minimal effect (Feb STATUS).
- All categoricals are low-cardinality (2–7 levels) with **no unseen levels in test**, so encoding is safe.

**Annotation Guidelines.** The label is `exam_score` ∈ [19.6, 100], a continuous score. Submissions are predicted
scores scored by R² — no thresholding or class decision is involved.

**Feature Set.** From the 11 raw columns we engineer **22 features**, grouped below:

| Group | Features |
|-------|----------|
| Base numeric | `age`, `study_hours`, `class_attendance`, `sleep_hours` |
| Ordinal-encoded | `sleep_quality_ord`, `facility_rating_ord`, `exam_difficulty_ord` |
| Categorical-encoded | `gender_enc`, `course_enc`, `internet_enc`, `study_method_enc` |
| Interactions | `study_x_attendance`, `study_x_sleep_quality`, `study_x_facility`, `attendance_x_sleep`, `method_x_hours`, `sleep_x_quality` |
| Polynomial | `study_sq`, `attendance_sq` |
| Composite | `total_effort`, `study_method_score`, `sleep_deficit` |

**Splitting strategy.** A single canonical **shuffled 5-Fold KFold CV (seed = 42)**, shared across all stages so scores
are directly comparable. The target is continuous and i.i.d. with no time or group structure, so plain shuffled KFold is
the correct scheme (the validation hint confirms this). Because a sklearn KFold split depends only on `n_samples` + seed,
the July run reuses the **exact same folds as the February prior-season run**, making every score on the ladder mutually
comparable. The leaderboard is closed (late submission returns `403`), so all scores here are **CV-only**; the Feb
submission — scored on RMSE, a different metric — confirms the pipeline produces valid ranked predictions but is not
numerically comparable to this R² ladder.

## Models & Architecture

**Purpose of Architecture.** A regressor that maximizes R² on `exam_score`. **Architecture Type** is a
**gradient-boosted decision-tree ensemble** — LightGBM as the primary learner, with XGBoost and CatBoost members and
seed-bagged / Optuna-tuned LGB variants, combined by a weighted blend; the champion is a **21-member kitchen-sink
mega-blend** (17 non-zero weights) discovered at tree-search node #27.

Each member takes the same **Input Format** — a numeric feature matrix with categoricals ordinal-/label-encoded — of
**Input Dimension 22 features** per row (a 1-D vector; there is no spatial/sequence structure).

**Architecture Description.** Each GBDT member is an additive ensemble of decision trees grown by histogram-based,
leaf-wise gradient boosting (a representative Optuna-tuned LGB: `num_leaves` = 251, `max_depth` = 6, direct-metric
tuned). The blend layer sits on top as a convex weighted average of the members' out-of-fold predictions, with weights
found by Dirichlet sampling (k = 800, seed = 42) plus coordinate-ascent refinement. **Model Complexity** is therefore
best expressed as trees × leaves rather than a dense parameter count: the untuned pool LGB averages ~1,428 boosting
iterations, and the champion blends 21 such members (17 receiving non-zero weight, the largest being node #16 at
weight 0.1126).

## Training procedures

Training proceeds as a **four-stage ladder** on the shared 5-fold split, each stage adding capability and measured on the
same clipped-OOF R² scale:

| Stage | Configuration | OOF R² |
|-------|---------------|-------:|
| 1 | Feb weighted ensemble (LGB / XGB / 2nd-LGB on the 22-feature set) | 0.786414 |
| 2 | skill pipeline: LGB + XGB + CAT blend (simplex grid weights) | 0.786793 |
| 3 | linear self-iteration (prior checks / Optuna fold-0-proxy tune / seed-bag), 5-way blend | 0.787060 |
| 4 | tree-search `harness_v3` (60-node budget, node #27 = 21-member mega-blend) | **0.787183** |

The 1→4 climb is +0.0978%. Two priors were resolved by controlled same-folds comparison in stage 3: (A) the s3e9
"explicit interactions add nothing for GBDT" prior was **rejected here** — the 22-feature set (0.785938) beat the 11
base features (0.785844) per solo LGB, so interactions genuinely help, vindicating Feb's claim and the exact opposite of
s5e10; (B) clip-to-[0,100] was **adopted** as a harmless safety-net post-process (raw 0.786792 → clip 0.786793).

The **Loss Function** is squared error (LightGBM `regression`/L2 objective) with model selection on R². The
**Optimization Algorithm** is gradient boosting (histogram, leaf-wise) — **not** SGD/ADAM — with blend weights optimized
by Dirichlet sampling + coordinate ascent. The **Learning Rate** is the boosting shrinkage (the Optuna-tuned LGB used
`learning_rate` ≈ 0.03017); there is **no Learning Rate Scheduler** (*N/A — GBDT convergence is governed by CV
early-stopping, not a schedule*) and **no Batch Size** (*N/A — full-dataset histogram boosting, not mini-batched*).

**Training Duration** for the tier-4 search was ~**4,540 s** of wall-clock over **46 model fits** (the external-facts
training-time field); the harness's own log records 4,592.9 s over 46 evaluated nodes of the 60-node budget, and the
stage-3 Optuna tune was 40 trials / 715.9 s on a fold-0 proxy. **Training Memory** was not explicitly capped — a
630k × 22 float matrix is modest on the single arm64 CPU machine; peak not separately measured ("not recorded").
**Transfer Learning** is *N/A (no pretrained weights)*; its analogue here is **cross-season record reuse** — the Feb
stage-1 ensemble is kept verbatim and folded into the July ladder. **Data Augmentation** is *N/A (tabular)*; the
analogues are feature engineering (the 22 features) and seed-bagging (tuned LGB seed = 2024).

**Reproducibility Standards** are strict: fixed seeds (fold seed = 42; Optuna and seed-bag seeds logged) and LightGBM
determinism flags (`deterministic=True`, `force_row_wise=True`, fixed `num_threads=16` across all models), which were
required because LightGBM's timing-dependent row-wise/col-wise histogram choice otherwise made training non-reproducible
across processes (max 0.87 per-sample drift before pinning). This run is a **native v3 run — no faithful replay was
needed**; the stage-4 OOF-reproduction gate (`06_rebuild_tree_best.py`) retrained all 21 members at max|dOOF| = 0
(bit-identical), recovering blend R² 0.787183 with member weights matching digit-for-digit.

## Inference procedures

**Decision Threshold** is *N/A* — R² is a regression metric, so we submit predicted scores directly (only clipped to the
observed [0, 100] range as a safety net) and never threshold. **Inference Duration** was not separately profiled
("not recorded"); GBDT scoring of the 270,000 test rows is sub-second per member on CPU, and **Inference Memory** is
negligible (tree inference is lightweight, not capped).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict the test students' exam scores. **Performance Metrics.** R² is the sole
competition metric for this cross-season unit; our tier-4 champion reaches **CV clipped-OOF R² 0.787183**, climbing
0.786414 → 0.786793 → 0.787060 → 0.787183 across the four stages. The February prior-season submission received a real
Kaggle leaderboard, but it was scored on **RMSE** (Public 8.70380 / Private 8.72876) — a different metric — so it is not
numerically comparable to the R² ladder and only confirms the pipeline emitted valid ranked predictions. The R²
leaderboard value itself is **not recorded** (competition closed to late submission; recorded CV-only).

**Performance Benchmarking.** The natural comparator is the **NVIDIA reproduce-agent**, which runs a core GBM copy of a
public kernel:

| Agent | Approach | R² | Note |
|-------|----------|---:|------|
| Our agent | from-scratch GBDT pool + tree-search | **0.78718** | tie |
| NVIDIA | reproduces `jiaoyouzhang/student-scores` (core, GBM only) | 0.7863 | tie |

The two agents are effectively **tied** (our 0.78718 vs NVIDIA 0.7863). The honest caveat is that the public
leaderboard's best solution is a **neural net**, which the GBDT reproduce-agent does not reproduce — so this comparison is
strictly GBM-vs-GBM, not a contest against the public state of the art. Within that GBM regime, this is an honest
"last-mile" competition: the whole four-stage ladder moves inside a ~0.0008 R² band with extremely stable fold-level CV,
so the gains are real but small, and neither agent's GBM pipeline separates from the other.

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions are being re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run are **TBD** until the separate credentialed operator scoring step; the figures below are the lane's own local CV only and revise nothing in the sections above.

- **Lane**: lane **running** (started 2026-08-17 15:26); re-run CV and LB: TBD.
