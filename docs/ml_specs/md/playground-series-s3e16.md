# ML Specification Report — playground-series-s3e16

### Crab Age Prediction · our from-scratch agent (GBDT pool + tree-search)

> *Figures grounded in the competition's `facts.json`, `eda_summary.json`, `STATUS.md`, and `experiments_tree_v3.json`.*

## Overview

The task is to predict the age of crabs from physical measurements (metric: MAE, lower is better). We
solve it with a from-scratch pipeline — a pool of gradient-boosted trees (LightGBM / XGBoost / CatBoost) plus diverse
variants, refined by a tree search whose weight optimisation is decided directly on the **rounded** integer metric. Our
champion is an 8-member blend (tree-search node #15) scoring **rounded OOF MAE 1.33563** on 5-fold CV, submitted to the
real leaderboard at Public 1.34315 / Private 1.33859. Against the NVIDIA reproduce-agent (which verbatim-copies a public
XGBoost kernel scoring 1.3387), our agent's 1.3356 wins — the lower MAE, so **our agent is the winner** on this episode.


**Why it matters.** Crab age normally requires destructive manual inspection; predicting it from external measurements supports sustainable fisheries management and harvesting decisions.

---

## Data

**Purpose of Data.** Predict the integer age of a crab (`Age`) from morphometric measurements — a **regression**
Playground Series episode scored by mean absolute error. **Data Format** is clean **tabular CSV**: one categorical
column (`Sex` ∈ {I = infant, M, F}) and seven continuous size/weight measures. **Data Volume** is **74,051 training
rows / 49,368 test rows**, with 8 raw predictive features once `id` and the target are removed.

**Data Quality** is high: **no missing values** in either split, **0 duplicate rows**, and near-identical train↔test
distributions (all per-feature mean shifts < 0.4%, so no covariate shift). The one raw-data wrinkle is that **`Height`
= 0 in 24 training rows** (physically invalid) — treated as missing and imputed with the median. Severe collinearity
is the defining structural feature: `Length`↔`Diameter` r = 0.989, and `Weight ≈ Shucked + Viscera + Shell` (r = 0.993).
The target is a right-skewed integer count (**range 1–29, mean 9.97, std 3.18, skew 1.09**, 28 unique values), which is
exactly what makes rounding of the regression output matter (see the Inference section). The strongest EDA signals were:

- **`Sex='I'` (infant)** → mean age 7.6 vs M 10.9 / F 11.3 — a strong signal captured by an `is_infant` flag.
- **`Shell Weight`** is the best-correlated feature with Age (Spearman 0.74); `Shucked Weight` the weakest (0.61).
- **Collinearity** among all size/weight columns (0.86–0.99), motivating a feature-pruned member that drops `Weight`.
- **Integer, right-skewed target** — rounding OOF predictions to the nearest integer improves MAE materially.

**Annotation Guidelines.** The label is `Age`, a positive integer count. Submissions are real-valued regression
predictions scored by MAE; because the ground truth is integer-valued, the post-processed predictions are **rounded to
the nearest integer and clipped to ≥ 1** before submission (see Decision Threshold).

**Feature Set.** From the 8 raw columns we engineer **24 features**, grouped below:

| Group | Features |
|-------|----------|
| Raw | `Length`, `Diameter`, `Height`, `Weight`, `Shucked Weight`, `Viscera Weight`, `Shell Weight` |
| Flags | `Height_was_zero`, `is_infant`, `is_male`, `is_female` |
| Weight ratios | `Shucked Weight_ratio`, `Viscera Weight_ratio`, `Shell Weight_ratio`, `weight_resid` (= Weight − Σparts), `parts_sum` |
| Size ratios | `diam_len`, `height_len`, `height_diam` |
| Derived | `volume`, `density`, `shell_density`, `meat_to_shell`, `shucked_to_shell` |

**Splitting strategy.** A single canonical **5-fold StratifiedKFold on binned `Age` (ages ≥ 20 merged into one bin),
shuffle, seed = 42**, shared across all members so scores are directly comparable. Fold MAE is very stable (1.34–1.37),
and the CV scheme is validated against the real leaderboard twice: both real submissions show the same-sign,
same-order-of-magnitude CV↔LB gap (rounded OOF ~0.003–0.0075 better than LB), so **CV is trustworthy**.

## Models & Architecture

**Purpose of Architecture.** A regressor that minimises MAE on the integer crab-age target. **Architecture Type** is a
**gradient-boosted decision-tree ensemble** — LightGBM (L1 objective), XGBoost (`reg:absoluteerror`) and CatBoost (MAE)
as the core learners, extended with diverse variants (Optuna-tuned LGB, a seed-bagged copy, a boundary-pushed low-LR
LGB, a Tweedie-objective LGB, and a feature-pruned LGB that drops the collinear `Weight`), combined by a weighted blend.
The champion is an **8-member blend** discovered at tree-search node #15.

Each member takes the same **Input Format** — a numeric feature matrix with `Sex` one-hot encoded — of **Input Dimension
24 features** per row (a 1-D vector; there is no spatial/sequence structure).

**Architecture Description.** Each GBDT member is an additive ensemble of decision trees grown by histogram-based
gradient boosting (representative LGB: `num_leaves` = 63, `n_estimators` up to 3000 with early stopping,
`learning_rate` = 0.02). The blend layer sits on top as a convex weighted average of the members' out-of-fold
predictions, with weights found by a Dirichlet (k = 800) plus coordinate-ascent search scored **directly on the rounded
MAE** (`metric_fn = MAE(round(clip(blend)))`). The champion places dominant weight on CatBoost (**0.5061**), followed by
the feature-pruned LGB (0.2662) and the boundary-pushed LGB (0.1117). **Model Complexity** is best expressed as
trees × leaves rather than a dense parameter count: each LGB member is up to ~3000 × 63 leaves, and the champion blends
8 such members.

## Training procedures

Training proceeds from a linear-champion baseline through a tree search on the shared 5-fold split, each stage measured
on the same OOF scale (all scores are rounded MAE unless noted):

| Stage | Configuration | Rounded OOF MAE |
|-------|---------------|----------------:|
| Baseline | LGB/XGB/CAT 0.6/0.3/0.1 blend, coarse simplex grid (exp #1) | 1.33812 |
| Phase B | + Optuna-tuned LGB / + seed-bagged LGB (raw improved, rounded worsened) | 1.33850 → 1.33893 |
| Tree search | reproduce 3-way with k=800+coordinate-ascent (node #9) | 1.33640 |
| Tree search | + LGB_tuned / seed2024 / LGBBOUND / TWEEDIE / FEATPRUNE members | 1.33635 → 1.33567 |
| **Champion** | **8-way blend, node #15 (weights decided on rounded MAE)** | **1.33563** |

The headline training finding is a genuine **raw-vs-rounded inversion**: node #15 has raw OOF MAE 1.35712 (worse than the
baseline's 1.35589) yet rounded OOF MAE 1.33563 (better than 1.33812) — the cleanest demonstration that raw and rounded
MAE are different optimisation targets here, and every decision must be made on the rounded number.

The **Loss Function** is per-member absolute-error / MAE (LightGBM `regression_l1`, XGBoost `reg:absoluteerror`,
CatBoost `MAE`), with model and blend selection on rounded MAE. The **Optimization Algorithm** is gradient boosting
(histogram-based) — **not** SGD/ADAM — with blend weights optimised by Dirichlet sampling + coordinate ascent. The
**Learning Rate** is the boosting shrinkage (representative LGB `learning_rate` = 0.02; the boundary-pushed member uses
0.005); there is **no Learning Rate Scheduler** (*N/A — GBDT convergence is governed by CV early-stopping, not a
schedule*) and **no Batch Size** (*N/A — full-dataset histogram boosting, not mini-batched*).

**Training Duration** for the tree search was **~1,409 s of wall-clock over 24 evaluated node fits**; individual solo
members fit in ~24–37 s each (CatBoost 24 s, LGB 31 s, XGB 37 s). **Training Memory** was **not recorded** — a
74k × 24 float matrix is trivial on the single arm64 CPU machine, peak not separately measured. **Transfer Learning**
is *N/A (no pretrained weights)*; its analogue here is **cross-competition experience injection** — the distilled
"Optuna fold-proxy tune → add to pool → seed-bag" recipe carried in from earlier episodes (which, notably, this episode
showed does *not* reliably transfer to a rounded-integer-MAE metric). **Data Augmentation** is *N/A (tabular)*; the
analogues are feature engineering (the 24 features) and seed-bagging.

**Reproducibility Standards** are strict: a fixed fold seed (42), logged Optuna/seed-bag seeds, and a rebuild gate that
retrains every champion member from scratch and requires digit-for-digit OOF reproduction. The **faithful replay
reproduced 24/24 member fits exactly (Δ = 0)**, and the blend reproduces 1.33563 exactly (after recovering the
full-precision weight vector — the 4-decimal stored weights alone drift the rounded score to 1.335796).

## Inference procedures

**Decision Threshold** is the **integer round/clip rule**: raw blended predictions are **rounded to the nearest integer
and clipped to ≥ 1** (`round(clip(blend))`) before submission — this is the load-bearing post-processing step, since the
target is integer-valued and rounding is what converts the raw-MAE ordering into the far better rounded-MAE score. The
champion submission's integer predictions fall in [4, 19] (train range [1, 29]). **Inference Duration** was **not
recorded**; GBDT scoring of the 49,368 test rows is sub-second per member on CPU, and **Inference Memory** is negligible
(tree inference is lightweight, not capped).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict the integer age of each test crab and minimise MAE. **Performance Metrics.**
MAE is the sole competition metric; our tree-search champion reaches **rounded OOF MAE 1.33563**, improving on the
linear baseline's 1.33812 by 0.00249 (−0.19% relative). Both real leaderboard checks confirm CV trustworthiness:

| # | Source | Rounded OOF MAE | Public LB | Private LB |
|---|--------|----------------:|----------:|-----------:|
| 1 | Linear champion (LGB/XGB/CAT 0.6/0.3/0.1, exp #1) | 1.33812 | 1.34356 | 1.34075 |
| 2 | Tree-search best (node #15, 8-way blend, exp #5) | **1.33563** | **1.34315** | **1.33859** |

Submission #2 beats #1 on **both** boards (Public −0.00041, Private −0.00216).

**Performance Benchmarking.** The natural comparator is the **NVIDIA reproduce-agent**, which verbatim-reproduces a
strong public kernel:

| Agent | Approach | MAE | Note |
|-------|----------|----:|------|
| Our agent | from-scratch GBDT pool + tree-search | **1.3356** | winner |
| NVIDIA | verbatim reproduces `pandeyg0811/mae-1-337-with-featureengineering-xgb` | 1.3387 | −0.0031 behind |

Our agent wins by 0.0031 MAE on this episode — the from-scratch pool with a rounded-metric-aware tree search edges out
the reproduced public XGBoost kernel. The decisive levers were (1) precise blend-weight search decided directly on the
rounded metric, and (2) genuinely diverse pool members — a feature-pruned LGB (dropping the r=0.993 collinear `Weight`)
and a boundary-pushed low-LR LGB — that a verbatim kernel copy does not explore.

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions were re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run were produced by the separate credentialed operator scoring step and are recorded in `benchmark_results/rerun_three_way.csv`; the figures below revise nothing in the sections above.

- **Lane**: done 2026-08-14, 45 min.
- re-run local CV: **MAE 1.334958** (5-fold StratifiedKFold on binned Age, post-processed), unweighted blend chosen over the tree champion to avoid refit optimism.
- Public / private leaderboard for this re-run (`benchmark_results/rerun_three_way.csv`): **Public 1.33738 / Private 1.33822**; `winner_priv` = **mine** (NVIDIA 1.34001, AIDE 1.34224).
