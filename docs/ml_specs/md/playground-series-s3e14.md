# ML Specification Report — playground-series-s3e14

### Wild Blueberry Yield Prediction · our from-scratch agent (GBDT pool + tree-search v3)

> *Figures grounded in the competition's `facts.json`, `eda_summary.json`, `STATUS.md`, and `experiments_tree_v3.json`.*

## Overview

The task is to predict wild-blueberry `yield` from field/environmental measurements (metric: MAE, lower
is better). We solve it with a from-scratch pipeline — a pool of gradient-boosted trees (LightGBM / XGBoost / CatBoost
plus a regularized LGB) refined by linear iteration and a 60-node harness-v3 tree search that ends on a 34-member
"kitchen-sink" mega-blend. Our submission-ready linear champion scores **CV OOF MAE 340.59891** (≈340.6), and the
tree-search research champion (node #44, OOF-only) reaches a lower **340.35572**. The NVIDIA reproduce-agent, copying a
public first-place kernel, reaches **337.28** — NVIDIA ahead by ~3.3 MAE; its true edge was OOF-stacking, which we did
not reproduce.


**Why it matters.** Yield prediction from pollination and weather variables informs planting, pricing, and resource allocation in specialty-crop agriculture.

---

## Data

**Purpose of Data.** Predict the continuous `yield` of wild-blueberry plots — a tabular regression Playground Series
episode (Aygün et al. Nature 2026 benchmark, Season 3 Episode 14). **Data Format** is clean **tabular CSV**, entirely
numeric. **Data Volume** is **15,289 training rows / 10,194 test rows**, with **16 raw predictive features** once `id`
and the target are removed (there are **no categorical columns**).

**Data Quality** is high: **no missing values** in either split. The sources disagree on exact-duplicate rows —
`eda_summary.json` reports **14 duplicate rows** in train while `STATUS.md` reports **7 exact duplicate rows** (both
describe them as harmless); the discrepancy is noted rather than resolved. Train↔test means differ by **<1% on every
column** (largest shift `RainingDays` −0.82%), so there is no covariate shift. The strongest EDA signals, all
consistent with blueberry-biology intuition, were:

- **Fruit-biology block dominates**: `fruitset` (r = +0.886), `seeds` (r = +0.869), `fruitmass` (r = +0.826) are by
  far the strongest predictors — prime candidates for interaction/product terms.
- **Rain suppresses yield**: `RainingDays` (r = −0.477) and `AverageRainingDays` (r = −0.484) are strongly negative;
  the two are near-collinear (r = 0.991).
- **`clonesize`** is a moderate negative discrete driver (r = −0.383, only 6–8 distinct values).
- **Temperature columns are redundant and weak**: the six Upper/Lower TRange columns are near-perfectly collinear
  (pairwise r ≥ 0.998) yet individually near-zero-signal (|r| ≈ 0.02) — condensed and later mostly dropped.
- **Pollinator columns** (`honeybee`, `bumbles`, `andrena`, `osmia`) are individually weak (|r| 0.07–0.20); `honeybee`
  is extremely right-skewed (skew ≈ 41.6).

**Annotation Guidelines.** The label is `yield` ∈ ℝ, a continuous float over **[1945.53, 8969.40]** (mean 6025.19,
std 1337.06, skew −0.29, only 776 distinct values but **not** integer-valued — `is_integer_valued = false`,
`log_transform_candidate = false`). Submissions are **real-valued yield predictions** scored by MAE.

**Feature Set.** From the 16 raw columns we engineer up to **27 features** (iteration 1), later pruned to a **21-feature**
working set (iteration 2, after a LGB gain-importance probe dropped near-zero features), grouped below:

| Group | Features |
|-------|----------|
| Raw (kept) | `clonesize`, `honeybee`, `bumbles`, `andrena`, `osmia`, `fruitset`, `fruitmass`, `seeds`, `RainingDays`, `AverageRainingDays`, `MaxOfUpperTRange`, `MinOfUpperTRange`, `MaxOfLowerTRange` |
| Fruit-biology interactions | `fruitset_x_seeds`, `fruitset_x_fruitmass`, `fruitmass_x_seeds`, `fruit_triple`, `seeds_per_fruitset` |
| Temperature (condensed) | `temp_spread`, `temp_avg` (`temp_avg` dropped in iter 2) |
| Pollinator index | `total_pollinators`, `honeybee_share` |
| Rain / scale | `rain_intensity` (dropped iter 2), `log_clonesize` (dropped iter 2) |
| Dropped noise (iter 2) | `temp_avg`, `log_clonesize`, `rain_intensity`, `MinOfLowerTRange`, `AverageOfLowerTRange`, `AverageOfUpperTRange` |

**Splitting strategy.** A single canonical **5-fold KFold CV (shuffle, seed = 42)**, shared across every linear
iteration and every tree-search node so scores are directly comparable. Stratification is unnecessary — the target is
continuous and i.i.d. with no groups/time structure and no train↔test shift (validation hint: *"continuous i.i.d.
target → standard KFold"*). The Kaggle leaderboard was not used (no credentials in this environment), so **all scores
here are CV-OOF only**; no public/private LB numbers exist for this run.

## Models & Architecture

**Purpose of Architecture.** A regression model that minimizes MAE on wild-blueberry yield. **Architecture Type** is a
**gradient-boosted decision-tree ensemble** — LightGBM as the primary learner, with XGBoost, CatBoost (native
categorical handling of the low-cardinality env columns), and a regularized LGB member, combined by a weighted blend
plus snap-to-grid post-processing. The submission-ready champion is a **5-member simplex blend** (linear exp #7); the
overall lowest-OOF champion is the **34-member harness-v3 mega-blend** discovered at tree-search node #44.

Each member takes the same **Input Format** — a purely numeric feature matrix (no encoding needed; CatBoost optionally
treats the low-cardinality env columns as native categoricals) — of **Input Dimension 21 features** per row in the
working configuration (a 1-D vector; no spatial/sequence structure).

**Architecture Description.** Each GBDT member is an additive ensemble of decision trees grown by histogram-based
gradient boosting under an MAE (`regression_l1`) objective. The blend layer sits on top as a convex weighted average of
the members' out-of-fold predictions, with weights found by grid-simplex search (linear iterations) or by v3's k=800
Dirichlet sampling + coordinate-ascent search (tree search); snap-to-grid is applied to the blended output.
**Model Complexity** is best expressed as trees × leaves rather than a dense parameter count — a representative tuned LGB
member used `num_leaves` = 63, `max_depth` = 5 (Optuna box, exp #4); exact per-member tree/leaf counts for the root
members are **not recorded**. The submission-ready champion blends 5 such members; the tree-search champion blends 34,
of which 28 retained weight > 0.005.

## Training procedures

Training proceeds as a staged progression on the shared 5-fold split, each stage measured on the same OOF scale:

| Stage | Configuration | OOF MAE |
|-------|---------------|--------:|
| Baseline (exp #1) | generic 3-model LGB+XGB+CAT blend (16 raw features) | 341.40782 |
| Iteration 1 (exp #2) | 27 features + fruit-biology interactions, 3-model blend + snap-to-grid | 340.95856 |
| Iteration 2 (exp #3) | pruned to 21 features + native-cat CatBoost, grid-simplex blend + snap (beat Ridge stack 344.04) | 340.71180 |
| Phase B (exp #5) | + Optuna-tuned LGB **added** as 4th member, 4-way simplex + snap | 340.62702 |
| Phase B (exp #7, linear champion) | + seed-2024 LGB as 5th member, 5-way simplex + snap | **340.59891** |
| Tree search v3 (exp #8, node #44) | 34-member explore-burst mega-blend, k=800 weight search + snap (OOF-only) | **340.35572** |

The **Loss Function** is mean absolute error (LightGBM `regression_l1` / MAE objective), with model selection directly
on OOF MAE. The **Optimization Algorithm** is gradient boosting (histogram-based) — **not** SGD/ADAM — with blend
weights optimized by grid-simplex (linear) and Dirichlet + coordinate-ascent (tree v3). The **Learning Rate** is the
boosting shrinkage; the Optuna-tuned LGB used `learning_rate` = 0.0154 (exp #4), while the root members' learning rates
are **not recorded** individually. There is **no Learning Rate Scheduler** (*N/A — GBDT convergence is governed by CV
early-stopping, not a schedule*) and **no Batch Size** (*N/A — full-dataset histogram boosting, not mini-batched*).

**Training Duration** for the tree-search v3 run was **~2,338 s wall (≈39 min) over 60 node evaluations** (the hard
budget cap); ~12 min of that was wasted on 6 catastrophic DART evals (solo OOF ~6,100+ MAE) that contributed nothing.
The linear Phase-B rounds were ~3–6 min each; the Optuna LGB tune was 36 fold-0-proxy trials in 293 s.
**Training Memory** was **not recorded** — a 15k × 21 float matrix is trivial on the single arm64 CPU machine, peak not
separately measured. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue here is **cross-season
experience injection** — s3e7's "don't re-tune the second model, add tuned variants to the pool" lesson was carried in
and directly shaped Phase B. **Data Augmentation** is *N/A (tabular)*; the analogues are feature engineering (the 21–27
features) and seed-bagging.

**Reproducibility Standards** are strict: a fixed 5-fold KFold (shuffle, seed = 42) shared across all scripts and the
tree, with solo-node OOF scores digit-verified against the STATUS ledger to 5 decimals (root asserted == 342.02154).
The tree-search champion came from a **native harness-v3 run — no replay needed** (`v3_native = true`); the run resumes
deterministically from `experiments_tree_v3.json` at seed 42. Blend weight search is deterministic at seed 42. Honest
caveat carried in-source: the 34-dim mega-blend weights were fit on the full OOF with no nested validation, so the
burst-beats-exploit *direction* is robust but the 5th decimal is not.

## Inference procedures

**Decision Threshold** is *N/A* — this is regression, not classification, and the target is not integer-valued
(`is_integer_valued = false`), so no thresholding or rounding is applied. The one output transform is **snap-to-grid**:
final predictions are snapped to the **nearest observed train `yield` value**. This was validated empirically (the EDA
noted the target is *not* a coarse integer grid, so snapping was tested rather than assumed) and won consistently by a
small margin — e.g. raw blend 340.65207 → snapped 340.59891 in the linear champion, and raw 340.57853 → snapped
340.35572 in the tree-search champion (where snap-to-grid was applied inside the metric function on every candidate
weight vector). **Inference Duration** was **not recorded**; GBDT scoring of the 10,194 test rows is sub-second per
member on CPU. **Inference Memory** is negligible and **not recorded** (tree inference is lightweight). Note the
tree-search champion is an **OOF-only research result** — no test predictions were generated for it, so it has no
submission file; the submission-ready deliverable is the linear 5-way blend (340.59891).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict continuous wild-blueberry yield and minimize MAE. **Performance Metrics.**
MAE is the sole competition metric; our submission-ready linear champion reaches **CV OOF MAE 340.59891** (≈**340.6**,
the reported `our_best`), improving 341.40782 → 340.95856 → 340.71180 → 340.62702 → 340.59891 across the linear stages,
and the tree-search v3 mega-blend pushes further to **340.35572** (OOF-only). No leaderboard scores exist — this run was
never submitted to Kaggle.

**Performance Benchmarking.** The natural comparator is the **NVIDIA reproduce-agent**, which copies the strongest
public kernel:

| Agent | Approach | CV MAE | Note |
|-------|----------|-------:|------|
| NVIDIA | reproduces `sergiosaharovskiy/ps-s3e14-first-place-winning-solution` (core) | **337.28** | winner |
| Our agent | from-scratch GBDT pool + tree-search v3 | 340.6 | −3.3 MAE |

NVIDIA leads by ~3.3 MAE. The honest caveat: the public kernel's real edge was **OOF-stacking**, which our
reproduction did **not** capture (`repro_type = core`) — so this is a comparison of a well-optimized public first-place
solution against our from-scratch pool, and the gap here reflects that missing stacking layer rather than a difference
in base-model quality. On this S3-era yield problem the well-optimized public kernel retains a clear edge; the
reproduce-vs-originate gap narrows on newer seasons, analyzed separately in the four-axis evaluation.

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions were re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run were produced by the separate credentialed operator scoring step and are recorded in `benchmark_results/rerun_three_way.csv`; the figures below revise nothing in the sections above.

- **Lane**: done 2026-08-14, 27 min.
- re-run local CV: **MAE 340.95936** leave-fold-out honest (340.82815 fitted-OOF), LGB+XGB blend snapped to the observed target grid (node #69).
- Public / private leaderboard for this re-run (`benchmark_results/rerun_three_way.csv`): **Public 343.43434 / Private 333.18611**; `winner_priv` = **nvidia** (NVIDIA 330.71616, AIDE 332.31556).
