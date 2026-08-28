# ML Specification Report — playground-series-s3e14
### Wild Blueberry Yield Prediction · AIDE agent (20-step tree search, Claude Sonnet via shim)

> *Figures grounded in the run's `facts_aide.json`, the agent's own `report.md` and `best_solution.py`
> (run `2-optimal-ruddy-dogfish`), its `journal.json` tree, and `config.yaml` for task metadata.
> Dataset shapes were measured directly from the run's own `input/` copy of the competition files.*

## Overview

The task is to predict wild blueberry `yield` for 10,194 test rows from pollination, fruit-set and weather
covariates (metric: MAE, minimized). AIDE searched for a solution over **20 steps**, of which **18 returned a
score and 2 timed out**, improving from a first LightGBM draft at **CV MAE 342.0642** to a champion at
**CV MAE 339.8059** at **step 11** — a three-seed, three-model blend of LightGBM (with native categorical
columns), CatBoost and XGBoost.

```
342.0642 − 339.8059 = 2.2583
```

This is the run in the batch where AIDE behaved most like a careful practitioner and least like a lucky one.
It failed rarely (2 of 20 steps), it recognised that fold-to-fold MAE variance was the binding constraint and
attacked it with multi-seed averaging, and it spent its last eight steps honestly probing alternatives —
stratified folds, a log target, cheaper budgets — none of which beat step 11. But careful is not the same as
best: on the private leaderboard AIDE scores **332.31556** (71.2nd percentile, rank 541/1877), edging past our
own agent by 0.12 while finishing 1.60 behind the NVIDIA reproduce-agent. `facts_aide.json` records
`local_winner = nvidia` and `lb_winner = nvidia`.

**Why it matters.** Wild blueberry yield depends on pollinator activity and weather during a narrow bloom
window, so a model that maps those conditions to yield is directly useful for agronomic planning. As an agent
benchmark, this episode tests a different skill from a model-family discovery: here every reasonable model
family lands in the same place, and the only remaining lever is variance reduction.

---

## Data

**Purpose of Data.** Predict the continuous `yield` for each field record — a **regression** task scored by
MAE, minimized (`config.yaml`: `evaluation_metric: mae`, `optimization_direction: minimize`,
`target_column: yield`). **Data Format** is **tabular CSV**, entirely numeric. **Data Volume**, measured from
the run's `input/` directory, is **15,289 training rows × 18 columns / 10,194 test rows × 17 columns**, leaving
**16 raw predictive features** once `id` and `yield` are removed. This is by far the smallest training set of
the four competitions in this batch (s3e11, s3e14, s3e16, s3e19), which matters: it is the reason fold variance
dominates everything else. Other reports in this folder document smaller sets still — s3e3 has 1,677 training rows
and s3e5 has 2,056.

**Data Quality** is clean — **zero missing values** in either split — with one small wrinkle: **7 duplicate
rows** exist in train once `id` is excluded. AIDE never checked for them and never mentions them; at 7 rows out
of 15,289 the omission is harmless, but it is characteristic of a loop that has no EDA phase.

The same absence produced a second, more consequential misreading. **The agent's `report.md` motivates its
target-transform experiment by saying `log1p(yield)` was tested "to address potential right-skew in the
target" — but the measured skew of `yield` is −0.2912, i.e. mildly *left*-skewed**, over a range
[1945.53061, 8969.40184] with mean 6025.1940 and 776 distinct values. The experiment was run anyway at step 17
and returned 339.9262, worse than the untransformed champion, so measurement corrected the wrong premise. Still,
one of AIDE's 20 steps was spent testing a hypothesis that thirty seconds of `describe()` would have ruled out.

A third observation belongs on the record as a **data-hygiene caveat**. The run's `input/` directory contains a
file named `oof_v4.npz`, an out-of-fold prediction cache that is not part of the official competition data.
AIDE found it, inspected it, and at **step 5** merged a matching column into a `HistGradientBoostingRegressor`
solution. That node scored **347.4488** — worse than the step-0 baseline of 342.0642 — and the file appears
nowhere in the champion's code, which was verified to contain no reference to `oof_v4`. The champion is
therefore uncontaminated, but the stray file should not have been in the input tree, and a version of this run
where that node had scored well would have been unusable.

**Annotation Guidelines.** The label is `yield`, a continuous float. Submissions are predicted yields scored by
MAE; no thresholding, rounding or class decision is involved.

**Feature Set.** From the 16 raw columns the champion builds **29 features**:

| Group | Features |
|-------|----------|
| Raw | the 16 original columns |
| Multiplicative interactions | `fruitset_fruitmass`, `fruitset_seeds`, `fruitmass_seeds` |
| Polynomial | `fruitset_sq`, `fruitmass_sq`, `seeds_sq` |
| Pollinator aggregate | `total_bees` (honeybee + bumbles + andrena + osmia) |
| Temperature spreads | `temp_range_avg`, `temp_range_max`, `temp_range_min` |
| Rain | `rain_intensity` (`AverageRainingDays` × `RainingDays`) |
| Ratios | `fruitset_per_clonesize`, `seeds_per_fruitmass` |

The champion additionally selects **13 of the 16 raw columns as native LightGBM categoricals** — `clonesize`,
the four pollinator counts, the six temperature-range columns, and the two rain-day columns — under a
cardinality test of ≤ 30 distinct values in *both* train and test. All 13 candidates pass. CatBoost and XGBoost
in the same blend receive the plain 29-column float matrix, so the categorical treatment is a LightGBM-only
source of ensemble diversity.

**Splitting strategy.** **5-Fold KFold (shuffle)** repeated over **three seeds (42, 52, 62)**, with the fold
seed set from the bagging seed so each repeat is a genuinely different partition. Out-of-fold predictions are
averaged across the three repeats before the blend weights are fitted. AIDE adopted this after observing in its
step-9 plan that "individual fold MAEs vary substantially (335–358)" — a 23-point spread on a metric where the
entire run only moved 2.26 points. It also tested **StratifiedKFold on quantile-binned `yield`** at step 13 as
an alternative variance control; it scored 340.2264 and was rejected. The run was **submitted to Kaggle**:
public 342.44918, private 332.31556.

## Models & Architecture

**Purpose of Architecture.** A regressor minimizing MAE on blueberry yield. **Architecture Type** is a
**three-member gradient-boosted-tree blend under multi-seed repeated cross-validation**, combined by a
grid-searched convex weighting. Every member is trained with an L1/MAE objective, so the whole stack optimizes
the competition metric directly rather than a squared-error proxy.

**Input Format** is a 29-column numeric matrix; the LightGBM member additionally receives 13 of those columns
typed as pandas `category`. **Input Dimension** is 29 features per row, a flat vector.

**Architecture Description.** The three members are:

- **LightGBM** — `objective="regression_l1"`, `metric="mae"`, `learning_rate=0.03`, `num_leaves=31`,
  `max_depth=-1`, `min_data_in_leaf=20`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`,
  up to 2,000 boosting rounds with 100-round early stopping, and `categorical_feature` set to the 13 columns.
- **CatBoost** — `loss_function="MAE"`, `learning_rate=0.03`, `depth=6`, up to 3,000 iterations with 100-round
  early stopping, on the plain float matrix.
- **XGBoost** — `objective="reg:absoluteerror"`, `learning_rate=0.03`, `max_depth=6`, up to 3,000 estimators,
  `subsample=0.8`, `colsample_bytree=0.8`, 100-round early stopping, also on the plain float matrix.

The blend layer is a **brute-force simplex grid search at 0.02 resolution** over the three-way weight
simplex, fitted on the seed-averaged OOF predictions against MAE. AIDE's own `report.md` records the selected
weights as approximately **0.80 LightGBM / 0.16 CatBoost / 0.04 XGBoost** — a distribution that makes the
XGBoost member very nearly decorative and is consistent with the report's own finding that "LightGBM
consistently outperformed CatBoost and XGBoost as a standalone model."

**Model Complexity** is best expressed as members × repeats × folds: **3 models × 3 seeds × 5 folds = 45 fitted
boosted-tree ensembles**, each up to 2,000–3,000 trees of depth 6 (or 31 leaves) before early stopping. Notably,
AIDE performed **no hyperparameter search at all** — its own Future Work section concedes that "no systematic
hyperparameter search (e.g. Bayesian optimization/Optuna) was performed for any model; fixed 'reasonable'
parameters were used throughout." Every parameter above is a first-guess value that survived because nothing
tested it.

## Training procedures

### The search trajectory

The `journal.json` tree has **5 root drafts (steps 0, 1, 5, 6, 7)** and 15 descendant nodes. The champion sits
at depth 5 along **0 → 2 → 3 → 9 → 10 → 11**. Only two steps failed, both `TimeoutError`.

| Step | Parent | What AIDE tried | CV MAE |
|------|--------|-----------------|-------:|
| 0 | — | LightGBM L1 + engineered interactions (first draft) | 342.0642 |
| 1 | — | ExtraTreesRegressor on ratio features (draft) | 364.8458 |
| 2 | 0 | + CatBoost, simple average blend | 341.2719 |
| 3 | 2 | grid-searched optimal blend weight | 341.2 |
| 4 | 3 | Ridge stacker over OOF preds + raw features | 341.2 |
| 5 | — | `HistGradientBoosting` (`absolute_error`) + `oof_v4` column (draft) | 347.4488 |
| 6 | — | LightGBM with Huber loss (draft) | 341.7924 |
| 7 | — | XGBoost `reg:absoluteerror` (draft) | 342.5979 |
| 8 | 3 | + per-fold isotonic clipping of predictions | 341.2848 |
| 9 | 3 | multi-seed 5-fold repeats, LGB+CatBoost | 339.9835 |
| 10 | 9 | + XGBoost as a third member | 339.9689 |
| **11** | **10** | **+ native categorical columns for LightGBM (champion)** | **339.8059** |
| 12 | 11 | extend categorical treatment to CatBoost | `TimeoutError` |
| 13 | 11 | StratifiedKFold on quantile-binned target | 340.2264 |
| 14 | 12 | fix: cut seeds/iterations to fit the budget | 341.3068 |
| 15 | 11 | 2-seed reduced budget | 340.5208 |
| 16 | 11 | 3-seed, lighter per-model budgets | 339.8649 |
| 17 | 11 | `log1p(yield)` target transform | 339.9262 |
| 18 | 11 | restore full fidelity at 3 seeds | `TimeoutError` |
| 19 | 18 | fix: trim the loop again | 341.4234 |

The shape of this search is a **long shallow plateau followed by one real idea**. Steps 0 through 8 — nine
nodes covering LightGBM, ExtraTrees, HistGradientBoosting, XGBoost, Huber loss, two blend variants, a Ridge
stacker and isotonic clipping — move the score from 342.0642 to 341.2, a gain of under one MAE point, and two
of them (steps 3 and 4) tie exactly at 341.2. The genuine breakthrough is **step 9**, where AIDE stopped
varying the model and started varying the *split*:

```
341.2000 − 339.9835 = 1.2165
1.2165 / 2.2583 = 0.539
```

Multi-seed repeated CV alone delivered 54% of the run's entire improvement, more than every model-family and
blending experiment combined. The remaining two gains are small: adding XGBoost (339.9689) and giving LightGBM
its categorical columns (339.8059).

After step 11 the search **fans out and finds nothing**. Eight consecutive nodes — 12 through 19 — all take
step 11 or its descendants as parent, and the best of them is step 16 at 339.8649:

```
339.8649 − 339.8059 = 0.0590
```

Forty percent of the step budget bought a result 0.059 MAE *worse* than what already existed. What that fan-out
actually reveals is a **budget ceiling rather than a modelling ceiling**. The champion configuration is
expensive — 45 model fits at full iteration budgets — and AIDE's own plans say so repeatedly. Steps 12 and 18
both attempted to enrich it (categorical CatBoost; restored fidelity) and both hit the 30-minute wall; steps
14, 15, 16 and 19 are all explicitly *reduced-budget* retreats, trading accuracy for completion. The run
finished pinned against its compute limit, with its best node sitting close enough to the wall that no
enrichment of it could be evaluated at all.

Total execution time was **13,556.3 s (3.77 h)** across 20 steps, mean 677.8 s, max 1,800 s — the most
expensive run of the four in this batch:

```
13556.3 / 3600 = 3.77
```

The failure rate was low, and so was the waste:

```
2 / 20 = 0.10
2 × 1800 = 3600
3600 / 13556.3 = 0.266
```

Two timeouts consumed 27% of wall clock. The champion node itself took **1,158.7 s** (from `journal.json`),
already two-thirds of the way to the ceiling.

### Specification fields

The **Loss Function** is **L1 / mean absolute error**, applied natively by all three members
(`regression_l1`, CatBoost `MAE`, `reg:absoluteerror`) — an exact match to the competition metric, and one of
the better decisions in the run, since a squared-error surrogate would have mis-weighted the noisy tail that
drives this dataset's fold variance. The **Optimization Algorithm** is gradient boosting (leaf-wise for
LightGBM, symmetric for CatBoost, level-wise for XGBoost) — **not** SGD/ADAM — with the blend weights found by
exhaustive grid search rather than by gradient descent. The **Learning Rate** is the boosting shrinkage, `0.03`
for all three members. There is **no Learning Rate Scheduler** (*N/A — convergence is governed by 100-round
early stopping against each fold's validation split, not a schedule*) and **no Batch Size** (*N/A —
full-dataset boosting, not mini-batched*).

**Training Duration** is 13,556.3 s across the whole search, of which the champion accounts for 1,158.7 s.
**Training Memory** was **not recorded** — no instrumentation, no cap. **Transfer Learning** is *N/A (no
pretrained weights)*; AIDE's analogue is its **journal-conditioned prompt**, which is unusually visible in this
run: the plans for steps 11, 15, 16 and 17 all open with "Looking at the results history…" and cite the exact
prior CV numbers they are trying to beat. **Data Augmentation** is *N/A (tabular)*; the analogues are the 13
engineered features and, more importantly, the **three-seed repeated CV**, which functions as augmentation of
the *validation* signal rather than of the training data.

**Reproducibility Standards** are partial. Seeds are explicit (`[42, 52, 62]`, each driving both the KFold
partition and the three learners), the blend grid is deterministic at 0.02 resolution, and the champion script
re-runs as written. The search that produced it is not reproducible — node proposals come from LLM sampling —
and there is no replay gate or cached-node verification. The submitted file came from a direct rerun of the
best node.

## Inference procedures

**Decision Threshold** is **N/A** — MAE on a continuous target, so raw predictions are submitted with no
rounding, clipping or snapping. **Post-processing** in the champion is nil: test predictions from each of the
45 fits are accumulated with weight `1 / (n_seeds × n_splits)`, combined by the three fitted blend weights, and
written straight to `submission.csv` against `sample_submission.csv`'s `id` column. AIDE did test one
post-processing idea — per-fold isotonic clipping at step 8 — and it scored 341.2848 against a running best of
341.2, so it was dropped. **Inference Duration** and **Inference Memory** were **not recorded**; AIDE times
whole scripts, not their scoring phase.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict `yield` for every test row and minimize MAE.

**Performance Metrics.** The champion reaches **CV MAE 339.8059** locally, **342.44918** on the public
leaderboard and **332.31556** on the private leaderboard, placing **rank 541/1877, 71.2nd percentile**. The
local estimate is *pessimistic* relative to private — the private split is simply easier:

```
339.8059 − 332.31556 = 7.49034
```

and the public-to-private spread is wide in the same direction:

```
342.44918 − 332.31556 = 10.13362
```

A 10-point public/private gap on a metric where the whole 20-step search moved 2.26 points is worth stating
plainly: **the leaderboard noise on this competition is several times larger than the signal AIDE was
optimizing**. Local CV over three seeds was the more trustworthy instrument, and it is to AIDE's credit that it
selected on CV rather than chasing the public score. The scored trajectory was
342.0642 → 341.2 → 339.9835 → 339.9689 → **339.8059**.

**Performance Benchmarking.** All three agents attacked this competition independently; the private-LB figures
below are the only cross-lane numbers used in this report.

| Agent | Approach | Private LB MAE | Note |
|-------|----------|---------------:|------|
| NVIDIA | reproduce-agent distilling a public kernel | 330.71616 | winner |
| **AIDE** | 20-step tree search → 3-seed × 3-model L1 blend | **332.31556** | second |
| Our agent | from-scratch pipeline | 332.4356 | third |

```
332.4356 − 332.31556 = 0.12004
332.31556 − 330.71616 = 1.5994
```

AIDE finishes 0.12 ahead of our own agent — a difference far inside the fold-to-fold spread of 335–358 that
AIDE itself measured, and therefore not a meaningful separation — but a real 1.60 behind NVIDIA.
`facts_aide.json` records `local_winner = nvidia` and `lb_winner = nvidia`.

The honest reading of the gap is that **AIDE never tuned anything**. It found the right loss (L1), the right
variance control (multi-seed repeats) and a sensible member set, and then ran out of budget before it could
search hyperparameters — its own Future Work section names this as the first missing piece, and its last eight
steps were spent fighting the 30-minute ceiling rather than optimizing inside it. The NVIDIA lane, by
reproducing an already-tuned public solution, starts where AIDE's search was heading and could not afford to
go. On a small, noisy dataset where every model family plateaus within one MAE point, inherited tuning is worth
more than additional search, and that is precisely the margin visible here.
