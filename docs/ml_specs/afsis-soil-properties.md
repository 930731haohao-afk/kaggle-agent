# ML Specification Report — afsis-soil-properties

### Africa Soil Property Prediction Challenge · our from-scratch agent (kernel-methods spectral pool + tree-search v3)

> *Figures grounded in the competition's `config.yaml`, `dossier.json`, `STATUS.md`, `experiments.json`, `experiments_tree_v3.json`, `experiments_tree_v3_run1_diluted_blend.json`, and `scripts/state/final.json`, with leaderboard and benchmark figures from the study's frozen three-way score table.*
> *Version discipline: this competition ran 2026-07-28 under the pre-v5 (v3-frozen) pipeline, the same configuration as the rest of the frozen 18-competition baseline — Stage 0.5 did not yet exist. The cited `dossier.json` was produced by the retrospective problem-identification sweep of 2026-07-29/30 and was NOT consumed by this run; it is cited only as a descriptive record of the task.*

## Overview

The task is to predict five physical and chemical soil properties — Ca, P, pH, SOC, Sand — from mid-infrared spectral measurements of African soil samples (metric: MCRMSE, the mean columnwise RMSE over the five targets, minimize). We solve it with a from-scratch pipeline built for the p ≫ n regime: a pool of kernel ridge / SVR models over five spectral preprocessing variants (plus PLS and LightGBM-on-PCA for diversity), validated by a nested GroupKFold over hidden site groups and refined by an 80-node tree search. Our champion, tree-search node #73 — a per-target greedy (Caruana forward-selection-with-replacement) blend of all 64 solo nodes — scores **honest CV MCRMSE 0.444076** (0.428772 when the blend weights are fit in-sample) and, on the real leaderboard, **Public 0.45207 / Private 0.49517**, ranking **659/1233 (percentile 46.6)**. Among the three benchmark agents ours is the **leaderboard winner**: AIDE posts the best local CV (0.40664) but finishes behind us on both boards (Private 0.49861, percentile 45.6), and NVIDIA's kernel reproduction trails far behind (Private 0.69855, percentile 18.8).

**Why it matters.** Sub-Saharan African soils are chronically under-surveyed because wet-chemistry assays are slow and expensive; predicting nutrient status from cheap infrared spectroscopy enables soil mapping at continental scale, informing fertilizer recommendations, land management, and food-security policy.

---

## Data

**Purpose of Data.** Predict five continuous soil-property targets per sample — a spectral multi-target tabular regression in a p ≫ n regime (the dossier's task family). **Data Format** is **tabular CSV** dominated by an ordered signal: **3,578 spectral columns** (mid-infrared absorbance, wavenumbers `m599.76` to `m7497.96`), **15 spatial covariates** (BSAN…TMFI — anonymized satellite/terrain rasters), and **1 categorical** (`Depth`: Topsoil/Subsoil), 3,594 features in all. **Data Volume** is **1,157 training rows / 727 test rows** — tiny n against huge p, which drives every modelling choice below.

**Data Quality** is high on the surface — **zero nulls**, all features pre-standardized — but the structure hides two traps. The targets are pre-standardized yet heavily right-skewed (skew P 7.45, Ca 4.71, SOC 2.45, pH 0.87, Sand 0.39), so squared error is driven by a handful of extreme rows. And the rows are not i.i.d.: the strongest EDA signals were

- **580 distinct spatial signatures across 1,157 train rows** — 565 sites each contribute a Topsoil/Subsoil pair, so a plain shuffled KFold puts two near-duplicate samples of the same site in different folds and leaks location.
- **Zero train↔test site overlap**, and test locations sit systematically far from train in the 15-d covariate space (train→train nearest-neighbour distance median 0.306 vs test→train median 0.956; 46.7% of test locations exceed the train 95th percentile) — test is drawn from held-out landscapes, not the same clusters.
- **Kernelization is worth ~0.032 MCRMSE** over linear ridge (0.475432 vs 0.507098) — the spectra→property map is nonlinear.
- The spatial covariates alone are weak (KRR-RBF on spatial only: 0.726751) — the spectra carry the signal.
- **P is by far the hardest target**: honest per-target RMSE 0.917442, versus 0.321491 (Ca), 0.338487 (pH), 0.321772 (SOC), 0.321187 (Sand) — one column dominates the achievable headroom, motivating per-target models and per-target blend weights.

**Annotation Guidelines.** The labels are five continuous, pre-standardized laboratory soil measurements (no de-standardization key is provided); submissions are **real-valued predictions for all five columns**, scored by MCRMSE — an L2-family metric that wants conditional means, not classes.

**Feature Set.** The saturated **CO2 band (2352–2380 cm⁻¹, 15 columns) is dropped**; the remaining spectra are served in **five preprocessing variants** — raw, SNV, SG-d1+SNV, SG-d2+SNV, SG-d1(w11)+SNV (Savitzky–Golay derivatives + standard-normal-variate) — each solo model picking one variant, plus z-scored spatial covariates (with a tunable `spatial_weight` knob) and binary `Depth`.

**Splitting strategy.** The load-bearing decision. Validation is a **nested GroupKFold(5)** (seed = 42) with the group key = the 580 spatial-signature site groups recovered from the 15-covariate tuples, so a site's Topsoil/Subsoil pair can never straddle folds — mirroring the held-out-landscape train→test relation. Each target's ridge alpha is selected by an **inner GroupKFold(4) that only ever sees outer-train rows**, making every solo score honest by construction. Blend scores are honest only after the weights are **re-fit leave-one-outer-fold-out**: the champion reads 0.428772 with in-sample weights but **0.444076** honestly — a measured weight-fitting optimism of **+0.015304**, squarely inside the +0.0146…+0.0186 band the experience library predicted.

## Models & Architecture

**Purpose of Architecture.** A five-target soil-property regressor that minimizes MCRMSE under a held-out-landscape split. **Architecture Type** is a **regularized kernel-methods ensemble** — kernel ridge regression (RBF, polynomial-2, Laplacian kernels) and SVR-RBF as the core learners, chosen over GBDT for the p ≫ n regime, with PLS regression and LightGBM-on-PCA as structurally different decorrelating members. The champion is a **per-target greedy blend of 64 solo nodes** discovered at tree-search node #73.

Each member takes the same **Input Format** — one spectral preprocessing variant concatenated with the weighted spatial covariates and Depth — a 1-D row vector per sample (**Input Dimension** ≈ 3,563 spectral columns after the CO2-band drop, + 15 spatial + Depth; the spectra are treated as an ordered signal at preprocessing time, then flattened for the kernel).

**Architecture Description.** Each kernel member fits five independent per-target regressions (MCRMSE weights columns equally and difficulty varies wildly, so no multi-output head). **Model Complexity** is governed by the kernel matrix, n × n = 1,157², not a parameter count — regularization (per-target alpha, kernel width, `spatial_weight`) does the work. The best solo is **node #42, KRR-RBF on SNV spectra with `spatial_weight` 0.25, honest 0.469086**; the blend layer on top assigns per-target weights by Caruana greedy forward selection (60 rounds) over out-of-fold predictions. Per-target weights matter a lot: 0.433071 with per-target weights vs 0.452840 with one shared weight vector over the same 16 members.

## Training procedures

Training proceeds as the standard ladder — naive floor, linear protocol, then tree search — every score on the same nested-GroupKFold scale:

| Stage | Configuration | MCRMSE |
|-------|---------------|-------:|
| naive floor | per-fold target mean | 1.024945 |
| linear baseline | Ridge (linear kernel) on SG-d1 spectra + spatial | 0.507098 |
| solo pool (16 configs) | best = SVR-RBF on SG-d1 + spatial (tree root) | 0.471939 |
| linear-stage blend | 15-member per-target Dirichlet + coordinate ascent (weights in-sample) | 0.433219 |
| tree search v3 (node #73) | 64-member per-target greedy kitchen-sink blend (weights in-sample) | 0.428772 |
| **champion, honest** | same blend, weights re-fit leave-one-outer-fold-out | **0.444076** |

The tree search (harness_v3) evaluated **80 nodes (67 solo / 13 blend) in 74 s**, first beating the linear-stage best at **evaluation 17**. The budget was raised from the default 60 to 80 because 15 of the first 16 nodes are zero-compute cached-OOF re-imports of the linear pool; the search stopped on the **hard budget cap (80/80)** with the mandatory explore burst (injected at n_eval = 70) still improving — the burst's kitchen-sink blend *is* the champion. The sanity gate fired once (burst seed #71, linear kernel on raw spectra, 0.686914 against a bound of 0.549675 — lineage plateaued immediately); the cost guard never fired; 19 lineages plateaued and 12 dedup rejections were logged.

**The one real finding is a negative result.** The first tree-search run re-seeded only 4 of the 16 linear-pool solos, and its kitchen-sink blend scored **0.449717 — worse than the linear blend it was meant to beat**. Re-seeding the entire linear pool as first-generation lineages, with no other change, moved the champion to 0.428772. The weight-search method is second-order: at 64 members greedy 0.428772, Dirichlet 0.428899, bagged greedy 0.430371 — **pool diversity, not weight search, explains essentially the whole gap** (run-1 tree archived as `experiments_tree_v3_run1_diluted_blend.json`, experiment #6).

The **Loss Function** family is L2 throughout, per the dossier's metric trap ("RMSE family, not MAE") — kernel ridge minimizes squared error in closed form; the SVR members' ε-insensitive loss is admitted for diversity, with model selection always on MCRMSE. The **Optimization Algorithm** is closed-form kernel solves / quadratic programming — **not** SGD/ADAM — with blend weights by greedy forward selection. There is **no Learning Rate** and **no Scheduler** (*N/A — closed-form and QP fits, no gradient-descent schedule*; the lone LightGBM-on-PCA member uses standard boosting shrinkage) and **no Batch Size** (*N/A — full-matrix kernel fits, not mini-batched*).

**Training Duration** for the tree-search stage was **74 s of wall-clock over 80 evaluated nodes** (kernel fits at n = 1,157 are cheap; no blend approached the 45 s cost-guard threshold); linear-stage and total pipeline wall-clock were not recorded. **Training Memory** was not explicitly capped — the dominant object is the 1,157² kernel matrix, trivial on the arm64 CPU machine, peak not recorded. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue is **cross-competition experience injection** — the TASK-SPECTRAL and TASK-BLEND-DECOR priors and the +0.0146…+0.0186 weight-optimism band carried in from the experience library and verified here. **Data Augmentation** is *N/A (tabular/spectral)*; the analogue is the five spectral preprocessing variants, which manufacture the view diversity the blend feeds on.

**Reproducibility Standards**: fixed seed 42 for the fold construction, per-target alphas selected deterministically inside the nested inner loop, and the tree root re-imports the linear-stage cached OOF **digit-verified** against the original run; the full backtrack log, dedup log, and the negative-result run-1 tree are all preserved in the workspace JSONs.

## Inference procedures

**Decision Threshold** is *N/A* — MCRMSE is a regression metric, so we submit raw real-valued predictions for all five targets and never threshold. No post-processing is applied. The submission (`submission.csv`, 727 rows) was verified column-for-column and PIDN-order-identical to `sample_submission.csv`. **Inference Duration** was not separately profiled — kernel scoring of 727 test rows against 1,157 support rows is sub-second per member — and **Inference Memory** is negligible (not capped).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict Ca, P, pH, SOC, and Sand for the 727 held-out-landscape test samples. **Performance Metrics.** MCRMSE (minimize) is the sole competition metric; our champion's honest local number is **CV MCRMSE 0.444076**, improving 1.024945 → 0.507098 → 0.471939 → 0.433219 (in-sample) → 0.428772 (in-sample) down the ladder, with the honest leave-fold-out re-fit as the headline. On the real leaderboard the same submission scores **Public 0.45207 / Private 0.49517**, rank **659/1233 (percentile 46.6)** — and the honest grouped-CV number (0.444076) tracks the public score (0.45207) closely, vindicating the nested-GroupKFold design. The champion's 0.000127 margin over the runner-up blend is noise; the gaps to the 16-member blends (~0.0043) and to the best solo (~0.040) are not.

**Performance Benchmarking.** Per the benchmark set's standing convention, **between-agent claims use the leaderboard numbers**; each agent's local CV is its own internal view under its own scheme (grouped and nested for us, unknown and not comparable for the others) and is **never used for cross-agent ranking**. The frozen three-way table:

| Agent | Approach | Local CV | Public LB | Private LB | Percentile |
|-------|----------|---------:|----------:|-----------:|-----------:|
| Our agent | from-scratch kernel pool + tree-search v3 | 0.444076 | 0.45207 | **0.49517** | **46.6** |
| AIDE | open-source AIDE agent | 0.40664 | 0.45636 | 0.49861 | 45.6 |
| NVIDIA | reproduces a public kernel | 0.48493 | 0.61276 | 0.69855 | 18.8 |

**Our agent wins on the leaderboard** (`lb_winner` = mine): Private 0.49517 vs AIDE's 0.49861 and NVIDIA's 0.69855, and the same ordering holds on the public board. AIDE is the `local_winner` (0.40664) — but its local edge does not survive contact with the private test, while our honest grouped CV sits much nearer its own leaderboard outcome; the site-group split policy that made our local number *look* worse is exactly what made it *transfer*. NVIDIA's copied kernel collapses on this comp (percentile 18.8), the recurring failure mode of vote-based kernel selection on older research competitions.

---

*Fields marked "not recorded": linear-stage and total pipeline wall-clock; training peak memory; inference duration and memory; target physical units and standardization constants (not provided in the data). STATUS.md was written before benchmark scoring and marks the run OOF-only; the leaderboard figures above come from the study's frozen three-way score table.*

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions are being re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run are **TBD** until the separate credentialed operator scoring step; the figures below are the lane's own local CV only and revise nothing in the sections above.

- **Lane**: done 2026-08-14, 24 min, audit clean.
- re-run local CV: **MCRMSE 0.470503** honest (landscape-grouped 5-fold, weights refit per fold; same blend 0.455779 in-sample).
- Public / private leaderboard for this re-run: **TBD** (pending operator scoring).
