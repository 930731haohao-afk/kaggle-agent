# ML Specification Report — playground-series-s3e9
### Concrete Compressive Strength — Regression · AIDE agent (20-step tree search, Claude Sonnet via shim)

> *All scores are taken from the run's `facts_aide.json`. Tree structure, per-step plans, per-node analyses and execution times come from `logs/2-benevolent-crazy-tarantula/journal.json`; hyperparameters from `best_solution.py` and the run's `config.yaml`; metric/target metadata from the competition `config.yaml`. Dataset shapes are counted directly from the run's own `input/` directory.*

## Overview

The task is to predict the compressive `Strength` of a concrete mix from its component masses and curing age, scored
by **RMSE (minimize)**. AIDE searched it over **20 steps — 19 scored, 1 buggy** — moving a LightGBM draft at
**12.1737** to a champion **local OOF RMSE of 12.0526** at **step 19, the last step of the budget**. The champion is a
single CatBoost regressor with leaf-wise (`Lossguide`) tree growth, curing age treated as a *categorical* feature, and
repeated 3×5-fold cross-validation — no ensemble at all, because every blend AIDE tried collapsed back onto pure
CatBoost.

On the leaderboard: **public 11.86657, private 12.30667, rank 427/767 (44.5th percentile)**, third of three lanes
(my agent 12.29871, NVIDIA 12.22913), with the score table recording `local_winner = nvidia` and `lb_winner = nvidia`.
The episode's defining number is not the ranking but the validation gap: AIDE's local estimate was optimistic by
0.25407 RMSE, by far the largest local-to-leaderboard discrepancy of the five runs in this batch (s3e1, s3e3, s3e5,
s3e7, s3e9), and the one hypothesis that would have explained it was raised inside the run and abandoned in the same
sentence.

**Why it matters.** Concrete strength prediction is a genuine engineering-qualification problem — mix design, curing
schedules, structural sign-off — and the dataset carries a physical structure (curing age takes a handful of standard
milestone values) that rewards representation choices over model choices. AIDE found exactly that: its two best moves
were both about *how age is represented*, not about which learner or how many.

---

## Data

**Purpose of Data.** Predict a continuous compressive strength from eight mixture and curing variables, with
`Strength` as target and `id` as key, scored by RMSE. **Data Format** is small, clean, all-numeric **tabular CSV**.
**Data Volume**, counted from the run's `input/` directory, is **5,407 training rows × 10 columns** and
**3,605 test rows × 9 columns** — eight raw predictive features (`CementComponent`, `BlastFurnaceSlag`,
`FlyAshComponent`, `WaterComponent`, `SuperplasticizerComponent`, `CoarseAggregateComponent`,
`FineAggregateComponent`, `AgeInDays`). The `input/` directory contained only `train.csv`, `test.csv` and
`sample_submission.csv`, so this is a clean from-raw-data search.

**Data Quality** was never profiled (**no EDA artifact exists**), and this is the episode where that omission is most
consequential. AIDE's feature code divides by `CementComponent`, `WaterComponent`, `EffectiveBinder` and
`CoarseAggregateComponent` with no zero-guards and never faulted, so those columns are strictly positive in both
splits — but that is inference from the absence of a traceback, not measurement. More importantly, at step 15 AIDE
explicitly raised the hypothesis that near-duplicate rows leak across folds, and then dropped it in the same plan:
*"since duplicate-based leakage is unverified, I instead focus on a more robust, atomic improvement"*. It never
returned to it, and it never wrote the four lines of code that would have checked. Given the 0.25407 local-to-private
gap documented below, that unwritten check is the most valuable experiment this run did not run.

**Annotation Guidelines.** The label is continuous strength in the dataset's units; submissions are real-valued
predictions scored by RMSE, with no threshold, rounding or ranking involved.

**Feature Set.** The champion's matrix is 20 columns wide:

```
10 (raw columns) − 2 (id, Strength) = 8 raw features
8 + 12 (engineered) = 20 features
```

| Group | Features |
|-------|----------|
| Raw mixture and age (8) | `CementComponent`, `BlastFurnaceSlag`, `FlyAshComponent`, `WaterComponent`, `SuperplasticizerComponent`, `CoarseAggregateComponent`, `FineAggregateComponent`, `AgeInDays` |
| Mix ratios (6) | `WaterCementRatio`, `CementWaterRatio`, `BinderAggregateRatio`, `SuperplasticizerCementRatio`, `FineCoarseRatio`, `WaterEffectiveBinderRatio` |
| Binder aggregates (3) | `TotalBinder` (cement + slag + fly ash), `EffectiveBinder` (cement + 0.8 × slag + 0.5 × fly ash, a reactivity weighting), `AggregateTotal` |
| Age representations (3) | `LogAge` (log1p), `SqrtAge`, **`AgeCat`** (`AgeInDays` cast to string and declared a CatBoost categorical) |

The feature set is domain-reasoned rather than generated — `EffectiveBinder`'s 0.8 / 0.5 pozzolanic weights are AIDE's
own chemistry argument — and it is stable from step 6 onward. The single most valuable column is `AgeCat`, introduced
at step 17 on the argument that curing age takes a small number of standard milestone values (1, 3, 7, 28, 90, 365
days) with non-smooth strength jumps between them, so a native categorical split beats any smooth transform of the
same variable. That one change moved 12.0636 → 12.0579, and it is the reason the champion's lineage exists.

**Splitting strategy.** **`KFold(n_splits=5, shuffle=True)` repeated three times with seeds 42, 142 and 242** — the
harness's `k_fold_validation: 5` with a 3-repeat wrapper — plus a separate single-split `KFold(5, random_state=42)`
used for the `max_leaves` screening pass. Plain KFold, not stratified or grouped: no grouping variable was ever
considered, which is precisely the choice the abandoned duplicate-leakage hypothesis would have questioned. The
reported metric is the RMSE of the **repeat-averaged** OOF vector, not the mean of per-repeat RMSEs; step 19's
analysis records the three individual repeat RMSEs as 12.0457, 12.1046 and 12.0961, so the headline 12.0526 is lower
than two of the three runs it summarizes — a legitimate ensembling effect (the submission is likewise a 3-repeat
average) but worth stating plainly. The leaderboard was exercised: the champion's submission was scored publicly and
privately.

## Models & Architecture

**Purpose of Architecture.** Minimize RMSE on a small, physically structured tabular regression problem.
**Architecture Type** is a **single CatBoost regressor** — not an ensemble. This is a finding rather than a default:
AIDE tried blending CatBoost with LightGBM (step 11) and with XGBoost (step 14), and in both cases the OOF-optimal
convex weight converged to **1.00 on CatBoost**, so two full steps were spent proving that the other libraries added
no complementary signal. The earlier LightGBM+XGBoost blends (steps 3–4) were both worse than a plain CatBoost draft.

**Input Format** is a mixed-dtype matrix: nineteen float columns plus one string column (`AgeCat`) passed to CatBoost
through `cat_features`. **Input Dimension** is **20 features per row**.

**Architecture Description.** CatBoost regressor with `grow_policy="Lossguide"` and `max_leaves` 31 (selected by
screening {16, 31, 64}), `learning_rate` 0.05, `l2_leaf_reg` 5.0, `bagging_temperature` 0.0, `random_strength` 1.0,
`iterations` 2000, `loss_function="RMSE"`, early stopping 100 with `use_best_model=True`, and `cat_features` pointing
at `AgeCat`. Every one of those values is the product of an AIDE search step: `depth`/`learning_rate`/`l2_leaf_reg` at
step 13, `bagging_temperature`/`random_strength` at step 15, the categorical age at step 17, and the switch from
CatBoost's default symmetric trees to leaf-wise growth at step 19 — the last on the argument that symmetric levels
cannot express asymmetric splits around a categorical age boundary.

**Model Complexity** in fits:

```
3 max_leaves candidates × 5 screening folds = 15 fits
3 repeats × 5 folds                        = 15 fits
                                     total = 30 CatBoost fits
```

each with a 2000-iteration ceiling at 31 leaves before early stopping. Note that the screening pass selects
`max_leaves` on the same training data that the final repeated CV then scores, so the reported metric carries a small
selection optimism on top of everything else.

## Training procedures

### Search trajectory

| Step | Parent | What AIDE changed | OOF RMSE |
|-----:|-------:|-------------------|---------:|
| 0 | — | draft: LightGBM + water/binder ratios, log age | 12.1737 |
| 1 | — | draft: RandomForest + ExtraTrees average | 13.1352 |
| 2 | 0 | 5-seed bagged LightGBM | 12.1331 |
| 3 | 2 | + XGBoost, fixed 50/50 blend | 12.1072 |
| 4 | 3 | grid-searched blend weight (34 % LGB / 66 % XGB) | 12.1056 |
| 5 | — | draft: CatBoost, default hyperparameters | 12.0721 |
| 6 | — | draft: CatBoost + extended ratio/age feature set | 12.0682 |
| 7 | — | draft: KNN regressor, tuned k | 13.2831 |
| 8 | 6 | + 3-seed bagging inside each fold | *buggy — TimeoutError (1800 s)* |
| 9 | 8 | repair: 2 seeds, drop the redundant full-data retrain | 12.0754 |
| 10 | 6 | repeated 3×5-fold CV | 12.0675 |
| 11 | 10 | + LightGBM blend — optimal weight → 1.00 CatBoost | 12.0675 |
| 12 | 10 | + age × chemistry interaction terms | 12.0745 |
| 13 | 10 | hyperparameter grid: depth 5, lr 0.05, l2 5.0 | 12.0636 |
| 14 | 13 | + XGBoost blend — optimal weight → 1.00 CatBoost | 12.0636 |
| 15 | 13 | wider grid incl. `bagging_temperature`, `random_strength` | 12.0636 |
| 16 | 13 | log1p target transform | 12.2845 |
| 17 | 13 | **`AgeInDays` as a native categorical feature** | 12.0579 |
| 18 | 17 | + leakage-safe KNN target-mean feature | 12.0655 |
| 19 | 17 | `grow_policy=Lossguide`, `max_leaves` 31 (**champion**) | **12.0526** |

The tree opened with the configured **5 root drafts** (steps 0, 1, 5, 6, 7) and stayed **shallow — maximum depth 4** —
with the champion at depth 4 on the lineage 6 → 10 → 13 → 17 → 19. Both of the first two drafts' branches were
abandoned once the CatBoost drafts landed: nothing descended from step 0 after step 4. **10 of the 19 scored steps set
a new incumbent.** Total improvement:

```
12.1737 − 12.0526 = 0.1211        (0.9948 % relative reduction)
```

Under 1 % from twenty steps, and the distribution of that gain is lopsided. The two late structural moves —
categorical age at step 17 and leaf-wise growth at step 19 — account for a tenth of it, while simply switching library
from LightGBM to CatBoost in the opening drafts accounts for most of the rest:

```
12.0636 − 12.0526 = 0.0110        (steps 17 and 19 combined)
12.1737 − 12.0682 = 0.1055        (LightGBM draft → CatBoost draft, steps 0 → 6)
0.1211  − 0.0110  = 0.1101        (everything except the two late moves)
```

**One timed-out node consumed nine tenths of the compute.** Step 8 asked for 3-seed bagging inside each of five
folds — 15 CatBoost fits at 2000 iterations, plus three redundant full-data retrains — and hit the harness's
wall-clock ceiling:

```
1800 / 1991.7 = 90.38 % of the run's total execution time
```

for a node that produced no metric and no submission. Every other node in this run is cheap (mean 99.6 s including the
timeout; the champion itself took 23.0 s per the journal), so in practice AIDE bought 19 experiments for
`1991.7 − 1800 = 191.7` seconds of useful compute and then spent half an hour discovering that 18 CatBoost fits do not
fit in 30 minutes. The repair at
step 9 correctly identified all three causes (too many seeds, too many iterations, a redundant retrain) and scored
12.0754 — but seed bagging as an *idea* was never retested at a feasible budget, so the run never learned whether it
helps.

**Two steps were spent proving a negative twice.** Steps 11 and 14 both add a second library to the blend, and both
report an optimal weight of 1.00 on CatBoost — identical incumbent scores of 12.0675 and 12.0636 respectively. AIDE's
memory recorded the first result and its plan for step 14 still argued that "prior experiments showed CatBoost +
XGBoost/LightGBM blends were complementary", which is a misreading of its own log. Step 15 adds a third such
repetition — a wider hyperparameter grid that rediscovered the previous grid's optimum and reported 12.0636 again — so
**steps 11, 14 and 15 all returned exactly their parent's score**, three of twenty steps that moved nothing.

One piece of context for reading "step 0": this journal is the second leg of the run, started with `initial_journal`
pointing at a 5-node journal from an earlier leg, so AIDE's memory at step 0 already contained five prior attempts on
this competition.

### Optimization specification

The **Loss Function** is squared error — CatBoost `loss_function="RMSE"` — matching the competition metric exactly,
so training loss, early-stopping criterion and model selection are the same quantity throughout. AIDE tested the one
alternative that a right-skewed positive target invites: step 16 trained on `log1p(Strength)` and inverted with
`expm1`, scoring 12.2845, comfortably worse, and the original scale was kept. The **Optimization Algorithm** is
CatBoost's ordered gradient boosting, here with leaf-wise (`Lossguide`) rather than symmetric tree growth. The
**Learning Rate** is the boosting shrinkage, `learning_rate` 0.05, selected by the step-13 grid. There is **no Learning
Rate Scheduler** (*N/A — convergence is governed by early stopping at patience 100, not a schedule*) and **no Batch
Size** (*N/A — full-dataset boosting*).

**Training Duration** for the whole search was **1991.7 s of executed wall-clock across 20 nodes** (mean 99.6 s, max
1800 s — the timeout). Excluding that one node, the entire twenty-step search is a few minutes of compute.
**Training Memory Consumption Limits** are **not recorded**.

**Transfer Learning** is *N/A (no pretrained weights; the competition forbids them)*; its analogue is **journal
memory**, which drove the productive lineage — step 13's plan cites the failed interaction-feature experiment as its
reason to tune hyperparameters instead, and step 19's cites the categorical-age result as its reason to try
asymmetric growth. It also drove the unproductive repetition at step 14. **Data Augmentation** is *N/A (tabular)*; the
analogues are the twelve engineered chemistry/age features and **repeat bagging** (3 repeats, seeds 42 / 142 / 242,
with each fold's model seed set to `seed + fold`).

**Reproducibility Standards** are moderate and unverified: repeat seeds derived deterministically as `42 + repeat × 100`,
per-fold model seeds as `seed + fold`, screening seed 42, and a hard `assert` that the submission's id order matches
`sample_submission.csv`. But **AIDE runs no replay or rebuild gate** — 12.0526 is what one execution printed, and the
champion was never re-run. The `max_leaves` screening pass in particular is re-executed on every run and could select
a different value.

## Inference procedures

**Decision Threshold** is *N/A* — RMSE is a continuous regression metric, so raw real-valued predictions are
submitted. **Post-processing is absent**: the test prediction is the mean over 3 repeats of the mean over 5 folds of
each fold model's output, written straight to CSV. There is no clipping to the observed strength range, no
non-negativity floor and no calibration — notable because strength is a physically non-negative quantity, so nothing
in the pipeline prevents a negative prediction. The only guard in the script is the id-order assertion against
`sample_submission.csv`, which is stricter than the reindex used in the other four episodes in this batch and is the one piece of
defensive engineering in the run.

**Inference Duration** is **not recorded** separately — each fold model scores the test set immediately after fitting,
so it is included in the champion node's 23.0 s. **Inference Memory Consumption Limits** are **not recorded**.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict `Strength` for each of the 3,605 test mixes and submit a two-column CSV
in `sample_submission.csv` order. **Performance Metrics.** RMSE, minimized. AIDE's local trajectory runs
12.1737 → 12.0721 → 12.0682 → 12.0675 → 12.0636 → 12.0579 → 12.0526 across the LightGBM draft, the CatBoost drafts,
repeated CV, the hyperparameter grid, categorical age and leaf-wise growth. The leaderboard tells a different story:

```
12.30667 (private) − 12.0526  (local OOF) = 0.25407
12.30667 (private) − 11.86657 (public)    = 0.4401
```

The 0.25407 local-to-private gap is more than twice the total improvement the search achieved (0.1211), which means
the 5-fold protocol was measuring something systematically easier than the private test set — and the 0.4401 spread
between public and private is the largest relative shake-out in this batch. Two explanations are consistent with the
evidence and neither was tested: the near-duplicate leakage AIDE raised and dropped at step 15, and ordinary
small-sample variance on a 3,605-row test split cut into public and private halves. The submission placed
**427/767, the 44.5th percentile**.

**Performance Benchmarking.** All three lanes submitted to the same private leaderboard:

| Lane | Approach | Private RMSE | Note |
|------|----------|-------------:|------|
| NVIDIA | reproduce-agent, public-kernel replication | **12.22913** | `lb_winner` |
| My agent | from-scratch GBDT pool + tree search | 12.29871 | ahead of AIDE by 0.00796 |
| AIDE | 20-step tree search, single Lossguide CatBoost | 12.30667 | last |

```
12.30667 − 12.22913 = 0.07754
12.30667 − 12.29871 = 0.00796
```

AIDE finishes third by 0.00796 from the my-agent lane — a hair — and 0.07754 from NVIDIA. What separates it is not
modelling quality: a single well-tuned CatBoost with a categorical age representation is a defensible answer to this
competition, and AIDE reached it by genuine search rather than replication. What separates it is that its validation
protocol told it 12.0526 when the truth was 12.30667, so every late-stage decision — leaf-wise growth over symmetric,
31 leaves over 16 or 64, keeping the KNN feature out — was made on differences of 0.01 measured by an instrument with
0.25407 of bias. The run also had the compute to check: it used 1991.7 s of which 1800 s went to one dead node, so a
duplicate audit and a grouped-CV re-run would have cost minutes. `facts_aide.json` records both `local_winner` and
`lb_winner` as `nvidia`; on this episode the two agree.

The my-agent figure in the table above is that lane's pre-re-run score. On the current benchmark record
(`benchmark_results/rerun_three_way.csv`) its final-architecture re-run scores private 12.33192, so AIDE is second of
the three rather than third; `winner_priv` is still `nvidia`.
