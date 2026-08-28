# ML Specification Report — playground-series-s3e5
### Wine Quality (ordinal) Prediction · our from-scratch agent (GBDT pool + tree-search v2)

> *Figures grounded in the competition's `facts.json`, `eda_summary.json`, `STATUS.md`, and `experiments_tree_v2.json`.*

## Overview

The task is to predict wine quality on an ordinal 3–8 scale (metric: quadratic weighted kappa, QWK).
We solve it with a from-scratch pipeline — a pool of gradient-boosted regressors (LightGBM / XGBoost / CatBoost)
whose continuous output is discretized by an OptimizedRounder, grown through a multi-tier ablation and a 22-node
tree search. Our benchmarked champion scores **CV QWK 0.5677** (exp #8, 6-way blend), climbing to 0.57066 once the
tree search adds a boundary-push member. This beats the **NVIDIA reproduce-agent**, which copies Grandmaster
`rsakata/optimize-qwk-by-lgb` verbatim and reaches only 0.5478 — our agent ahead by ~3.6%, a rare case where the
from-scratch agent tops a Grandmaster kernel.


**Why it matters.** Automating sensory quality scoring from physicochemical measurements gives winemakers objective, scalable quality control; the ordinal target also makes this a clean testbed for QWK-optimized modelling.

---

## Data

**Purpose of Data.** Predict the ordinal quality score (integers 3–8) of a wine from its physicochemical
measurements — a Playground Series ordinal-regression episode scored by QWK. **Data Format** is clean, all-numeric
**tabular CSV**: 11 continuous physicochemical features, no categorical columns. **Data Volume** is **2,056 training
rows / 1,372 test rows**, with 11 raw predictive features once `Id` (a plain row index) and the target are removed.

**Data Quality** is high on the usual axes — **no missing values** in either split, **0 duplicate rows** (feature-only
or full), and **no covariate shift** (largest train↔test mean difference 2.1%, on citric acid). The defining modelling
wrinkle is **severe class imbalance**: quality 3 has only 12 rows (0.6%) and quality 8 only 39 rows (1.9%), while
classes 5 and 6 dominate (839 + 778 = 79% combined) — an imbalance ratio of ~69.9×. The strongest EDA signals were:

- **Alcohol** is the dominant predictor (Spearman r ≈ +0.504); **sulphates** next (r ≈ +0.457).
- **Volatile acidity** (r ≈ −0.248) and **total sulfur dioxide** (r ≈ −0.227) are the strongest negative signals.
- **Mild collinearity** only: fixed acidity vs citric acid (r ≈ 0.696), vs pH (r ≈ −0.674), vs density (r ≈ 0.616);
  free vs total SO₂ (r ≈ 0.638) — both sides kept, as trees handle collinearity natively.
- **Modeling-head decision**: regression, not multiclass — QWK penalizes squared class distance, so a continuous
  regressor respects the ordering and lets the abundant mid-classes (5/6) inform placement of the data-starved
  extremes (3/8). Reconfirmed this run: a diverse multiclass-EV member scored only 0.5204 and got zero blend weight.

**Annotation Guidelines.** The label is `quality` ∈ {3, 4, 5, 6, 7, 8}. Submissions are **integer quality
predictions** scored by QWK — an agreement metric on discrete classes, so the continuous model output must be
rounded to integers before scoring.

**Feature Set.** From the 11 raw columns we engineer **21 features** (11 → 21), all domain-informed ratios and
interactions:

| Group | Features |
|-------|----------|
| SO₂ structure | `free_so2_ratio`, `bound_so2` |
| Acidity structure | `fixed_to_volatile_acid`, `citric_to_volatile_acid`, `total_acid`, `acid_ph_ratio` |
| Alcohol interactions | `alcohol_x_sulphates`, `alcohol_to_density`, `alcohol_x_density` |
| Fermentation | `sugar_to_alcohol` |

**Splitting strategy.** A single canonical **5-fold StratifiedKFold on the raw `quality` label (seed = 42, shuffle)**,
held fixed across all 12 experiments so scores are directly comparable. The leaderboard was **not attempted** (no
Kaggle credentials on this machine), so every score here is **CV-only** — no Public/Private LB numbers exist.

## Models & Architecture

**Purpose of Architecture.** An ordinal regressor that maximizes QWK on wine quality. **Architecture Type** is a
**gradient-boosted decision-tree ensemble** — LightGBM / XGBoost / CatBoost regression members combined by a convex
weighted blend, whose continuous output is mapped to integer classes by an OptimizedRounder; the benchmarked
champion is a **6-way blend** (exp #8), and the tree search extends it to a **3-way blend** (node #17).

Each member takes the same **Input Format** — an all-numeric feature matrix (no categorical encoding needed) — of
**Input Dimension 21 features** per row (a 1-D vector; no spatial/sequence structure).

**Architecture Description.** Each GBDT member is an additive ensemble of regression trees grown by histogram-based
gradient boosting; the two strongest members (LGB_tuned, CAT_tuned) are produced by Optuna with the objective set
directly to **post-rounder QWK on the trial's own OOF vector** (dodging the raw-vs-rounded discretization trap). The
blend layer sits on top as a convex weighted average of members' out-of-fold predictions, with weights found by
Dirichlet random search (k = 800) plus coordinate-ascent refinement, all scored on post-rounder QWK. **Model
Complexity** is best expressed as trees × leaves rather than a dense parameter count — a tuned member's exact tree
count is not recorded in the logs, and the champion blends up to 6 such members (the tree-search champion, 3).

## Training procedures

Training proceeds as a **multi-tier ablation** on the shared 5-fold split, each tier measured on the same
post-rounder OOF scale:

| Tier | Configuration | OOF QWK |
|------|---------------|--------:|
| 1 | generic 3-model blend, raw 11 features, naive rounding (exp #1) | 0.47871 |
| 2 | 21 features + regression blend + OptimizedRounder (exp #3) | 0.52687 |
| 3 | Dirichlet weight search + Optuna-tuned LGB & CAT pool, 6-way blend (exp #8) | **0.56769** |
| 4 | tree-search `harness_v2` (22 nodes, node #17 = 3-way blend) | 0.57066 |

The **Loss Function** for each GBDT member is regression L2, but model selection and Optuna tuning are driven by the
final **post-rounder QWK** (never the raw regression score — the s3e16 lesson). The **Optimization Algorithm** is
gradient boosting (histogram-based) — **not** SGD/ADAM — with blend weights optimized by Dirichlet + coordinate-ascent
search. The **Learning Rate** is the boosting shrinkage (LGB/CAT `learning_rate`), set per member by Optuna; the exact
tuned value is *not recorded* in the logs. There is **no Learning Rate Scheduler** (*N/A — GBDT convergence is governed
by CV, not a schedule*) and **no Batch Size** (*N/A — full-dataset histogram boosting, not mini-batched*).

**Training Duration** for the tree search was ~**697 s of wall-clock over 40 evaluated fits** (the v1 run; the v2 run
was 793.7 s / 22 nodes); each Optuna tune was 40 trials / 600 s timeout, and the full 5-fold CV is cheap because the
dataset is tiny (2,056 rows). **Training Memory** was not explicitly capped — a 2,056 × 21 float matrix is trivial on
the single arm64 CPU machine; peak was *not recorded*. **Transfer Learning** is *N/A (no pretrained weights)*; its
analogue here is **cross-competition experience/prior injection** — distilled lessons (s3e14 keep-don't-replace, s3e16
decide-on-post-rounder) and the boundary-push prior (P18) that seeded the tree search's winning LGBBOUND member.
**Data Augmentation** is *N/A (tabular)*; the analogues are feature engineering (the 21 features) and seed-bagging
(both tried; seed-bags gave zero weight here).

**Reproducibility Standards** are strict: fixed fold seed = 42, LightGBM determinism, and a resumable tree state
(`experiments_tree_v2.json`) whose root and solo seeds reload from the v1 cache with bit-level agreement (|diff| < 4e-6,
zero retraining). This being a native tree-search run, a **faithful replay is not needed** (the state file is the
reproducible artifact).

## Inference procedures

**Decision Threshold** is central here because QWK scores discrete classes: the regressor's continuous output is
mapped to integers by an **OptimizedRounder** whose cutpoints are fit (Nelder-Mead) on the OOF vector to maximize
post-rounder QWK, then applied with `clip[3, 8]`. The benchmarked champion (exp #8) uses cutpoints
**[3.574, 4.586, 5.609, 6.174, 7.628]**; the tree-search champion (node #17) searches its cutpoints jointly with the
blend weights inside each node's metric function (specific node-#17 cutpoint values *not recorded* as a static list). A
nested (leave-fold-out) cutpoint diagnostic showed the full-OOF cutpoint overfit risk is modest and shrinking as the
pool matured (gap 0.01644 on the 5-way blend → 0.00377 on the 6-way). **Inference Duration** was not separately
profiled; GBDT scoring of the 1,372 test rows is sub-second per member on CPU, and **Inference Memory** is negligible
(tree inference is lightweight, not capped).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict the integer quality class of each test wine. **Performance Metrics.**
QWK is the sole competition metric; our benchmarked champion (exp #8) reaches **CV OOF QWK 0.5677**, along the
progression 0.47871 → 0.52687 → 0.56769, and the tree search adds a further **+0.00297 to 0.57066** (node #17). All
figures are **CV-only** — the run was never submitted (no Kaggle credentials), so no Public/Private LB numbers exist.

**Performance Benchmarking.** The natural comparator is the **NVIDIA reproduce-agent**, which copies the strongest
public kernel verbatim:

| Agent | Approach | CV QWK | Note |
|-------|----------|-------:|------|
| Our agent | from-scratch GBDT pool + tree-search | **0.5677** | winner |
| NVIDIA | reproduces `rsakata/optimize-qwk-by-lgb` (verbatim) | 0.5478 | −3.6% |

Our agent **beats the NVIDIA reproduce-agent by +0.0199 (~3.6% relative)** — and does so against a kernel authored by
a Kaggle **Grandmaster** (rsakata), which makes this one of the clearer wins for the originate-from-scratch approach
over verbatim reproduction. The margin widens further (to 0.57066, ~+4.2%) once the tree search's boundary-push blend
member is included. The honest caveats: this is a CV-only comparison (neither side has an LB number on this machine),
and the champion's cutpoints are fit on full OOF — though the nested-cutpoint diagnostic bounds that overfit risk as
small and shrinking.

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions are being re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run are **TBD** until the separate credentialed operator scoring step; the figures below are the lane's own local CV only and revise nothing in the sections above.

- **Lane**: done 2026-08-14, 46 min.
- re-run local CV: **QWK 0.57638** (honest leave-fold-out re-measurement 0.57198), six shallow-CatBoost blend with OptimizedRounder cutpoints (node #24).
- Public / private leaderboard for this re-run: **TBD** (pending operator scoring).
