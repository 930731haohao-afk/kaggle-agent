# ML Specification Report — playground-series-s3e7
### Hotel Reservation Cancellation — Binary Classification · AIDE agent (20-step tree search, Claude Sonnet via shim)

> *All scores are taken from the run's `facts_aide.json`. Tree structure, per-step plans and per-node execution times come from `logs/2-adamant-sapphire-harrier/journal.json`; hyperparameters from `best_solution.py` and the run's `config.yaml`; metric/target metadata from the competition `config.yaml`. Dataset shapes are counted directly from the run's own `input/` directory.*

## Overview

The task is to predict whether a hotel reservation is cancelled (`booking_status`) from booking-record fields, scored
by **ROC AUC (maximize)**. AIDE ran **20 steps — 16 scored, 4 buggy** — and finished with a champion **local OOF AUC of
0.89963** at **step 19, the last step of the budget**. The champion is a repeated 2×5-fold stratified
LightGBM + XGBoost + CatBoost blend over 24 features with grid-searched weights.

The defining fact of this episode is how little the search achieved. AIDE's very first draft — a plain LightGBM with
five ratio features — scored **0.8977**, and twenty steps later the champion scored 0.89963:

```
0.89963 − 0.8977 = 0.00193        (0.2150 % relative improvement)
```

On the leaderboard: **public 0.90934, private 0.90109, rank 225/680 (67.1st percentile)**, third of three lanes
(my agent 0.90257, NVIDIA 0.90458), with the score table recording `local_winner = nvidia` and `lb_winner = nvidia`.
The gap to the winner (0.00349) is larger than everything AIDE's twenty steps bought.

**Why it matters.** Cancellation forecasting drives overbooking policy and revenue management, and this dataset is the
largest of the five in this batch (42,100 training rows), which changes the agent's economics: every experiment costs minutes
instead of seconds, and a search that mostly re-tests near-identical GBDT blends spends its budget without moving the
metric. This is the episode where AIDE's fixed 20-step budget and its 1800 s per-node timeout collide most sharply.

---

## Data

**Purpose of Data.** Predict a binary cancellation label from reservation attributes — party composition, stay length,
lead time, arrival date, market segment, room and meal type, price and prior-booking history — with `booking_status`
as target, `id` as key, scored by ROC AUC. **Data Format** is **integer-coded tabular CSV**: all columns arrive
numeric, with several of them semantically categorical. **Data Volume**, counted from the run's `input/` directory, is
**42,100 training rows × 19 columns** and **28,068 test rows × 18 columns** — the largest dataset in this batch
(s3e1, s3e3, s3e5, s3e7, s3e9), ahead of s3e1's 37,137 rows. The `input/` directory contained only `train.csv`, `test.csv` and `sample_submission.csv`, so this
is a clean from-raw-data search.

**Data Quality** was never profiled (**no EDA artifact exists**). One defect is visible in the champion's code:
bookings with zero guests or zero nights exist, and every price ratio guards against them with `.replace(0, 1)` in the
denominator — a silent imputation that turns a division by zero into "price per one unit" rather than a missing value.
`prev_cancel_ratio` uses the same guard. Missing-value counts, duplicate counts, arrival-date validity and train↔test
drift were never measured by this run.

**Annotation Guidelines.** The label is binary, and because the metric is AUC the submission carries **scores, not
decisions** — only the ordering of the 28,068 test bookings is graded, and no threshold is applied anywhere in the
pipeline.

**Feature Set.** The champion's matrix is 24 columns wide:

```
19 (raw columns) − 2 (id, booking_status) = 17 raw features
17 + 7 (engineered) = 24 features
```

| Group | Features |
|-------|----------|
| Party and stay (raw) | `no_of_adults`, `no_of_children`, `no_of_weekend_nights`, `no_of_week_nights`, `required_car_parking_space`, `no_of_special_requests` |
| Booking context (raw) | `lead_time`, `arrival_year`, `arrival_month`, `arrival_date`, `type_of_meal_plan`, `room_type_reserved`, `market_segment_type`, `avg_price_per_room` |
| Prior history (raw) | `repeated_guest`, `no_of_previous_cancellations`, `no_of_previous_bookings_not_canceled` |
| Engineered (7, AIDE step 0) | `total_nights`, `total_guests`, `price_per_person`, `price_per_night`, `total_prev_bookings`, `prev_cancel_ratio`, `is_weekend_only` |

All seven engineered features come from the *first draft* and survive unchanged to the champion — AIDE never improved
its feature set in twenty steps. It tried twice and failed both times: cyclical sin/cos encodings of arrival month and
date plus a `lead_time × special_requests` interaction scored 0.89802 at step 9, and an out-of-fold target encoding of
`market_segment_type × room_type_reserved` scored 0.89826 at step 11, both below the 0.89861 incumbent.

Seven columns are declared categorical (`market_segment_type`, `room_type_reserved`, `type_of_meal_plan`,
`arrival_year`, `arrival_month`, `repeated_guest`, `required_car_parking_space`) and each learner receives them in its
own idiom: pandas `category` dtype for LightGBM, `category` dtype with shared train/test levels and
`enable_categorical=True` for XGBoost, and string casts with explicit `cat_features` indices for CatBoost. This is
one of the few genuinely careful pieces of engineering in the run.

**Splitting strategy.** **`StratifiedKFold(n_splits=5, shuffle=True)` repeated twice with seeds 42 and 123** — ten
fits per algorithm — matching the harness's `k_fold_validation: 5`. OOF vectors are accumulated as
`prediction / n_repeats` and test predictions as `prediction / (n_splits × n_repeats)`, so both are 2-repeat averages
over identical fold assignments across the three algorithms, which is what makes the OOF weight search valid.
Repeated CV was introduced only at the final step; every earlier node used a single 5-fold split. The leaderboard was
exercised: the champion's submission was scored publicly and privately.

## Models & Architecture

**Purpose of Architecture.** Maximize AUC — rank 28,068 bookings by cancellation risk. **Architecture Type** is a
**three-member gradient-boosted-tree blend**: LightGBM, XGBoost and CatBoost, combined by a convex weighted average
whose weights are grid-searched at 0.05 resolution on the averaged out-of-fold predictions. No neural model ever
executed; AIDE drafted PyTorch MLPs with categorical embeddings three separate times and all three failed to import.

**Input Format** is three parallel copies of the same 24-column table, differing only in how the seven categorical
columns are typed. **Input Dimension** is **24 features per row**.

**Architecture Description.** LightGBM carries AIDE's only successful hyperparameter tuning pass (step 12):
`learning_rate` 0.02, `num_leaves` 63, `max_depth` −1, `min_data_in_leaf` 15, `feature_fraction` 0.8,
`bagging_fraction` 0.8, `bagging_freq` 5, `lambda_l1` 0.5, `lambda_l2` 1.0, up to 3000 boosting rounds with early
stopping at 150. XGBoost uses 2000 estimators, `learning_rate` 0.03, `max_depth` 6, `subsample` 0.8,
`colsample_bytree` 0.8, `min_child_weight` 1, `tree_method="hist"`, early stopping 100 — the untuned configuration
from step 2, because the dedicated XGBoost tuning pass at step 17 (0.89869) failed to beat its parent. CatBoost uses
2000 iterations, `learning_rate` 0.03, `depth` 6, native `cat_features`, early stopping 100, likewise untuned after
its own tuning pass at step 18 (0.89862) failed. Step 19's analysis records the selected blend weights as **0.40 LGB /
0.45 XGB / 0.15 CatBoost**.

**Model Complexity** in fits:

```
3 algorithms × 2 repeats × 5 folds = 30 model fits
```

on 42,100 rows, with per-model ceilings of 3000 (LightGBM), 2000 (XGBoost) and 2000 (CatBoost) trees before early
stopping. The journal records 313.3 s of execution for this node — the largest of any scored node in the run except
step 18's 360.3 s.

## Training procedures

### Search trajectory

| Step | Parent | What AIDE changed | OOF AUC |
|-----:|-------:|-------------------|--------:|
| 0 | — | draft: LightGBM + 7 engineered features, 5-fold | 0.8977 |
| 1 | — | draft: RandomForest on the same features | 0.88709 |
| 2 | 0 | + XGBoost, weight-searched 2-way blend | 0.89849 |
| 3 | 2 | + CatBoost, 3-way weight grid | 0.89861 |
| 4 | 3 | + out-of-fold target encoding of three categoricals | 0.89852 |
| 5 | — | draft: PyTorch MLP | *buggy — ModuleNotFoundError* |
| 6 | — | draft: PyTorch MLP with embeddings | *buggy — ModuleNotFoundError* |
| 7 | — | draft: PyTorch MLP with embeddings (third attempt) | *buggy — ModuleNotFoundError* |
| 8 | 3 | logistic stacking meta-learner vs linear blend | 0.89861 |
| 9 | 3 | + cyclical month/date encodings, log lead time | 0.89802 |
| 10 | 3 | + monotone constraints on `lead_time`, all three models | *buggy — TimeoutError (1800 s)* |
| 11 | 3 | + target encoding of `market_segment × room_type` | 0.89826 |
| 12 | 3 | LightGBM tuning: 63 leaves, L1/L2, lower LR | 0.89874 |
| 13 | 7 | repair → single HistGradientBoosting | 0.89463 |
| 14 | 10 | repair → 3-fold, 2 models, fixed 0.5/0.5 weights | 0.89681 |
| 15 | 5 | repair → single HistGradientBoosting | 0.8953 |
| 16 | 6 | repair → single HistGradientBoosting | 0.89397 |
| 17 | 12 | XGBoost tuning pass | 0.89869 |
| 18 | 12 | CatBoost tuning pass | 0.89862 |
| 19 | 12 | repeated 2×5-fold CV for all three models (**champion**) | **0.89963** |

The tree opened with the configured **5 root drafts** (steps 0, 1, 5, 6, 7) and stayed **shallow — maximum depth 4**,
with the champion also at depth 4 on the lineage 0 → 2 → 3 → 12 → 19. Step 3 was the hub, spawning **six children**
(4, 8, 9, 10, 11, 12) of which exactly one improved on it. **Only 5 of the 16 scored steps set a new incumbent**
(steps 0, 2, 3, 12, 19) — the lowest hit rate of the five runs in this batch, and three of those five are the first three steps.

**The plateau is the finding.** Ten of the sixteen scored nodes land between 0.89802 and 0.89963 — every one of them a
variation on the same three-model blend — and the run's entire improvement, 0.00193, is smaller than the 0.00349 that
separates AIDE from the NVIDIA lane on the private board. In substance the search tested seven distinct ideas
against the step-3 blend (target encoding, stacking, cyclical features, monotone constraints, interaction target
encoding, and per-model tuning for each of the three learners) and six of them lost. Only the last one, repeated CV,
worked, and AIDE's own report is candid about what that means: the improvement headroom it had been chasing "was CV
noise rather than true signal".

**Four steps, a fifth of the budget, produced no metric at all.** Steps 5, 6 and 7 are three independent PyTorch MLP
drafts, all killed instantly by `ModuleNotFoundError: No module named 'torch'` — the same failure AIDE repeated three
times without its draft policy learning from the first. Their three repairs (13, 15, 16) all reverted to
scikit-learn `HistGradientBoostingClassifier` and all three scored below the step-3 incumbent (0.89463, 0.8953,
0.89397), so the entire neural-diversity branch — three drafts plus three repairs, six of twenty steps — contributed
nothing. The fourth buggy step is the timeout described next.

**And one step consumed half the compute budget for nothing.** Step 10 attempted monotone constraints on `lead_time`
across all three learners and hit the harness's wall-clock ceiling:

```
1800 / 3641.1 = 49.44 % of the run's total execution time
```

spent on a node that produced no submission and no metric. Its repair at step 14 over-corrected by cutting to 3 folds,
dropping CatBoost and fixing the blend weights at 0.5/0.5 simultaneously — a confounded experiment that scored 0.89681
and told AIDE nothing about monotone constraints, which were never retried.

One piece of context for reading "step 0": this journal is the second leg of the run, started with `initial_journal`
pointing at a 5-node journal from an earlier leg, so AIDE's memory at step 0 already contained five prior attempts on
this competition — which partly explains why the first draft was already within 0.00193 AUC of the final champion.

### Optimization specification

The **Loss Function** is binary cross-entropy — LightGBM `objective="binary"`, XGBoost `binary:logistic`, CatBoost
`Logloss` — with early-stopping and model selection both driven by **AUC** (`metric="auc"` / `eval_metric="AUC"`).
The **Optimization Algorithm** is histogram-based gradient boosting, with a brute-force 0.05-step grid search over the
3-simplex as the second-stage optimizer for blend weights. The **Learning Rate** is the boosting shrinkage: 0.02 for
the tuned LightGBM, 0.03 for XGBoost and CatBoost. There is **no Learning Rate Scheduler** (*N/A — convergence is
governed by early stopping at patience 150/100/100, not a schedule*) and **no Batch Size** (*N/A — full-dataset
histogram boosting*).

**Training Duration** for the whole search was **3641.1 s of executed wall-clock across 20 nodes** (mean 182.1 s, max
1800 s — the timeout), the most expensive of the five runs in this batch. The champion node alone took 313.3 s per the journal.
**Training Memory Consumption Limits** are **not recorded**; the harness caps wall-clock, not memory.

**Transfer Learning** is *N/A (no pretrained weights; the competition forbids them)*; its analogue is **journal
memory**, which on this episode reads as an accurate but unproductive record — step 12's plan correctly summarizes
that "feature-engineering tweaks and additional target encodings have not consistently beaten the 0.8986 plateau", and
then proposes more of the same kind of change. **Data Augmentation** is *N/A (tabular)*; the analogues are the seven
engineered ratio features and the **2-repeat fold bagging** introduced at step 19, which was the only lever in the
last seven steps that worked.

**Reproducibility Standards** are moderate and unverified: fold seeds fixed at 42 and 123, each model's seed set to
the repeat seed, blend weights determined by an exhaustive deterministic grid. But **AIDE runs no replay or rebuild
gate** — 0.89963 is what one execution printed, and the champion was never re-run to confirm it.

## Inference procedures

**Decision Threshold** is *N/A* — the metric is AUC, so raw probabilities are submitted and no cut-off is applied.
**Post-processing is absent**: for each algorithm the test probability is the mean of its ten fold models
(accumulated as `predict_proba / (n_splits × n_repeats)`), the three means are combined with the searched weights
0.40 / 0.45 / 0.15, and the result is written directly. There is no calibration, no rank transform and no clipping.
One fragility is worth recording: unlike the other four episodes in this batch, this script does **not** reindex against
`sample_submission.csv` — it only reorders columns with `submission[sample_sub.columns]` and relies on `test.csv`
already being in submission order. That held here (the file scored normally), but it is an unchecked assumption.

**Inference Duration** is **not recorded** separately — test scoring happens inside each fold's loop and is folded
into the 313.3 s champion node. **Inference Memory Consumption Limits** are **not recorded**.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Score all 28,068 test reservations by cancellation risk and submit a two-column
CSV matching `sample_submission.csv`. **Performance Metrics.** ROC AUC, maximized. AIDE's local trajectory runs
0.8977 → 0.89849 → 0.89861 → 0.89874 → 0.89963 across the draft, the two-way blend, the three-way blend, the LightGBM
tuning pass and the repeated-CV champion — five incumbent values spanning 0.00193 in total. Local and leaderboard agree
closely:

```
0.90109 (private) − 0.89963 (local OOF) = 0.00146
0.90934 (public)  − 0.90109 (private)   = 0.00825
```

The local estimate is very slightly conservative, which is the expected behaviour of a 2-repeat OOF average on 42,100
rows: with this much data the validation signal is reliable, and AIDE's problem here was not measurement but the
absence of anything worth measuring. The submission placed **225/680, the 67.1st percentile**.

**Performance Benchmarking.** All three lanes submitted to the same private leaderboard:

| Lane | Approach | Private AUC | Note |
|------|----------|------------:|------|
| NVIDIA | reproduce-agent, public-kernel replication | **0.90458** | `lb_winner` |
| My agent | from-scratch GBDT pool + tree search | 0.90257 | ahead of AIDE by 0.00148 |
| AIDE | 20-step tree search, repeated-CV 3-model GBDT blend | 0.90109 | last |

```
0.90458 − 0.90109 = 0.00349
0.90257 − 0.90109 = 0.00148
```

All three lanes are within 0.00349 of each other, so this episode separates the agents by very little — but AIDE is
third, and the reason is legible in its own log. Of twenty steps it spent six on a neural branch that could not
import, one on a timeout that returned nothing, and nine on variations of a blend that had already converged by
step 3. What actually improved the score was one hyperparameter pass and one variance-reduction pass. The same total
compute spent on re-running the timed-out monotone-constraint idea at a feasible budget, or on any structural
hypothesis rather than another blend permutation, would at least have produced evidence; as executed, 15 of the 20
steps failed to improve the incumbent and 4 of those returned no metric at all. `facts_aide.json` records both `local_winner` and `lb_winner` as
`nvidia`; on this episode the two agree.
