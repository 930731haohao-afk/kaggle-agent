# ML Specification Report — playground-series-s3e16
### Crab Age Prediction · AIDE agent (20-step tree search, Claude Sonnet via shim)

> *Figures grounded in the run's `facts_aide.json`, the agent's own `report.md` and `best_solution.py`
> (run `2-solid-bizarre-lobster`), its `journal.json` tree, and `config.yaml` for task metadata.
> Dataset shapes were measured directly from the run's own `input/` copy of the competition files.*

## Overview

The task is to predict a crab's integer `Age` from eight physical measurements for 49,368 test rows (metric:
MAE, minimized). AIDE searched for a solution over **20 steps** and — uniquely among the four runs in this
batch — **every one of them executed and returned a score; there were zero buggy nodes**. It improved from a
first LightGBM draft at **CV MAE 1.3597** to a champion at **CV MAE 1.3369** at **step 17**: a three-model
blend combined by a quantile-regression stacker and then calibrated by per-boundary rounding thresholds.

```
1.3597 − 1.3369 = 0.0228
```

The interesting result here is *where* that improvement came from. It was not the models. Every base learner
AIDE tried — LightGBM, XGBoost, CatBoost, a PyTorch MLP, random forest plus ElasticNet — landed in a band a
few thousandths wide. The entire gain came from noticing that `Age` is an integer and that a continuous
regressor's output must be rounded, and then from tuning *where* the rounding boundaries sit. One step did most
of the work:

```
1.3582 − 1.3406 = 0.0176
0.0176 / 0.0228 = 0.772
```

Step 8 — "round the predictions" — delivered 77% of the run's total improvement in a single 22-second node.

This produced AIDE's best percentile of the batch, **rank 124/1431 at the 91.4th percentile** with a private
LB of **1.34224**, yet it still finishes last of the three agents: our own agent scores 1.33859 and NVIDIA
1.34001, and `facts_aide.json` records `local_winner = mine` and `lb_winner = mine`. The gap is traceable to a
specific self-inflicted wound described below — the champion's headline 1.3369 is an **in-sample-calibrated**
number that AIDE's own honest nested-CV runs had already shown to be optimistic, and it selected on it anyway.

**Why it matters.** Ageing a crab conventionally requires killing and sectioning it; a model that infers age
from external measurements supports non-destructive stock assessment. As an agent benchmark, this episode
isolates a skill that has nothing to do with model choice: recognizing that the loss is defined on integers and
that the last transformation before submission is worth more than the first ten modelling decisions.

---

## Data

**Purpose of Data.** Predict integer `Age` for each crab — a **regression** task scored by MAE, minimized
(`config.yaml`: `evaluation_metric: mae`, `optimization_direction: minimize`, `target_column: Age`).
**Data Format** is **tabular CSV** with one categorical column. **Data Volume**, measured from the run's
`input/` directory, is **74,051 training rows × 10 columns / 49,368 test rows × 9 columns**, leaving **8 raw
predictive features**: `Sex` (I / M / F) plus seven continuous size and weight measurements.

**Data Quality** is clean: **zero missing values** in either split and **zero duplicate rows** once `id` is
excluded. The defining structural property is the target itself. **`Age` is an integer** taking 28 distinct
values over the range [1, 29], with mean 9.9678 and skew 1.0929 — genuinely right-skewed, so unlike the sibling
runs in this batch the `log1p` transform AIDE applied rests on a correct premise. The integer-valued target is
the single most exploitable fact about this dataset, because MAE on integers is minimized by an integer
prediction, and no continuous regressor produces one.

AIDE ran no EDA step and never printed a target histogram; it reached the integer insight at step 8 through
what its own plan calls "the hypothesis flagged in memory that rounding predictions" would help — i.e. from
prior knowledge in the prompt context rather than from looking at the data. It took eight steps to get there.

One source conflict is worth surfacing. **The agent's `report.md` describes the calibration stage as
"coordinate descent over ~8 individual age-bin decision boundaries", but the champion's code derives the
boundary count as `max_age − min_age`**, which on the measured target is:

```
29 − 1 = 28
```

The report understates the fitted parameter count by more than threefold, which matters because the whole
overfitting question in this run turns on how many free parameters the calibrator has.

**Annotation Guidelines.** The label is `Age`, a positive integer. Submissions are integer ages scored by MAE.
The champion writes integers to the submission file and clips them to `[1, 29]` — the observed training range —
before saving.

**Feature Set.** From the 8 raw columns the champion builds **19 features**:

| Group | Features |
|-------|----------|
| Raw | the 7 continuous measurements |
| Categorical | `Sex_I`, `Sex_M`, `Sex_F` (one-hot, with explicit zero-fill for any level absent from a split) |
| Geometry | `Volume` (Length × Diameter × Height), `Density` (Weight / Volume) |
| Weight ratios | `Weight_Ratio_Shucked`, `Weight_Ratio_Viscera`, `Weight_Ratio_Shell` |
| Weight consistency | `Weight_Sum` (component sum), `Weight_Diff` (`Weight` − `Weight_Sum`) |
| Shape ratios | `Length_Diameter_Ratio`, `Height_Diameter_Ratio` |

This set was fixed at step 0 and never meaningfully changed for the rest of the run. AIDE did try one
expansion — degree-2 interaction-only polynomial features across the seven raw measurements — and its own
report records that it "did not improve base-model MAE beyond the ~1.359–1.361 floor already reached", which
it attributes correctly to tree splits already representing those interactions implicitly.

**Splitting strategy.** A single canonical **5-Fold KFold (shuffle, `random_state=42`)** for the base learners,
held constant across every node of the search so the 20 scored nodes are mutually comparable. The stacker uses
a **second, independent 5-fold split (`random_state=123`)** over the OOF prediction matrix, which is correct
practice — fitting the meta-learner on the same folds that generated its inputs would leak. What is *not*
protected by the same discipline is the calibration stage; see the trajectory section. The run was **submitted
to Kaggle**: public 1.3392, private 1.34224.

## Models & Architecture

**Purpose of Architecture.** A predictor of integer crab age minimizing MAE. **Architecture Type** is a
**three-stage stack**: (1) three gradient-boosted-tree base learners on a `log1p(Age)` target, (2) a
median-regression meta-learner over their out-of-fold predictions, and (3) a **learned discretizer** that maps
the continuous stacked output onto integers by 28 individually-fitted decision boundaries. Stage 3 is the part
that carries the result, and it is not a model in the usual sense at all.

**Input Format** for stage 1 is a 19-column float matrix; for stage 2 it is a 3-column matrix of base-model OOF
predictions in original age units; for stage 3 it is a single scalar per row. **Input Dimension** is therefore
19 → 3 → 1 across the stack.

**Architecture Description.**

- **Stage 1, base learners.** LightGBM (`objective="regression_l1"`, `num_leaves=31`, `learning_rate=0.05`,
  `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`, `seed=42`, up to 1,000 rounds with 50-round
  early stopping); CatBoost (`loss_function="MAE"`, `depth=6`, `learning_rate=0.05`, up to 2,000 iterations,
  50-round early stopping, `random_seed=42`); XGBoost (`objective="reg:absoluteerror"`, `max_depth=6`,
  `learning_rate=0.05`, up to 2,000 estimators, `subsample=0.8`, `colsample_bytree=0.8`, 50-round early
  stopping, `random_state=42`). All three train on `log1p(Age)` and are inverted with `expm1` before the stack.
- **Stage 2, meta-learner.** `QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")` — a median/LAD
  regression, chosen at step 17 specifically because the earlier OLS stacker
  (`LinearRegression(positive=True)`) minimizes squared error and therefore introduced a systematic negative
  bias that the calibrator then had to undo. Fitted across 5 meta-folds and averaged.
- **Stage 3, calibrator.** Three candidate strategies are scored on the OOF predictions and the best is
  selected: raw continuous output clipped to `[1, 29]`; a single global additive offset searched over
  `[-0.5, 0.5]` at 0.01 resolution followed by rounding; or **per-boundary threshold calibration**, a
  three-pass coordinate descent that moves each of the 28 age-bin boundaries within ±0.35 of its nominal
  half-integer position at 0.02 resolution, subject to monotonicity against its neighbours.

**Model Complexity** is **3 models × 5 folds = 15 fitted boosted-tree ensembles** plus 5 meta-learner fits plus
**28 free threshold parameters**. That last count is the crux: 28 scalars fitted by direct search against the
same 74,051-row OOF vector they are evaluated on.

## Training procedures

### The search trajectory

The `journal.json` tree has **5 root drafts (steps 0, 1, 5, 6, 7)** and 15 descendant nodes, and it is the
**deepest search of the batch** — the champion sits at depth 8 along
**0 → 2 → 3 → 4 → 8 → 10 → 12 → 13 → 17**. All 20 nodes executed successfully.

| Step | Parent | What AIDE tried | CV MAE |
|------|--------|-----------------|-------:|
| 0 | — | LightGBM L1 on engineered ratio/volume features (first draft) | 1.3597 |
| 1 | — | RandomForest + ElasticNet 70/30 blend (draft) | 1.416 |
| 2 | 0 | + CatBoost, fixed 50/50 blend | 1.3585 |
| 3 | 2 | grid-searched optimal blend weight | 1.3583 |
| 4 | 3 | `log1p(Age)` target transform | 1.3582 |
| 5 | — | XGBoost `reg:absoluteerror` (draft) | 1.3585 |
| 6 | — | PyTorch MLP, L1 loss, BatchNorm/Dropout (draft) | 1.3669 |
| 7 | — | LightGBM median regression, diversified features (draft) | 1.3594 |
| 8 | 4 | **round predictions to the nearest integer** | 1.3406 |
| 9 | 8 | test whether naive rounding is optimal | 1.3406 |
| 10 | 8 | + XGBoost as a third blend member | 1.3402 |
| 11 | 10 | question the single global offset | 1.3402 |
| 12 | 10 | OLS (`positive=True`) stacking meta-learner | 1.3386 |
| 13 | 12 | per-boundary threshold calibration by coordinate descent | 1.3374 |
| 14 | 13 | re-run the coordinate descent on the same data | 1.3386 |
| 15 | 13 | Yeo-Johnson target transform instead of `log1p` | 1.3388 |
| 16 | 13 | **honest nested-CV** evaluation of single-offset calibration | 1.3396 |
| **17** | **13** | **quantile-regression stacker replacing the biased OLS (champion)** | **1.3369** |
| 18 | 17 | honest nested-CV evaluation of the calibration | 1.3412 |
| 19 | 17 | seed-bagged base models, 2 seeds × 5 folds | 1.3391 |

The run divides cleanly at step 8. **Steps 0–7 are a flat plateau of eight nodes**, spanning five different
model families and both target transforms, all confined to 1.3582–1.3669 except the random-forest draft. AIDE's
report reaches the right conclusion about them: "the underlying signal-to-noise ceiling for this feature set
was reached early, and further architectural search yielded diminishing returns."

**Step 8 breaks it in one move** by rounding to integers, and the search's remaining twelve steps are spent
almost entirely on refining *that* — better blend weights (1.3402), a stacker instead of a grid search
(1.3386), per-boundary thresholds instead of a single offset (1.3374), an unbiased stacker (1.3369). Model-side
experiments continued in parallel and continued to be worth nothing: seed-bagging the base learners at step 19
scored 1.3391, worse than the champion, and AIDE's report concludes that "base-model variance was not the
binding constraint."

The uncomfortable part of this trajectory is that **AIDE diagnosed its own overfitting and then selected the
overfitted node anyway**. Steps 16 and 18 exist precisely to measure the calibration honestly — refit the
thresholds on 4/5 of the OOF vector, evaluate on the held-out fifth — and they returned **1.3396** and
**1.3412**. The report states the finding explicitly: "naive in-sample calibration reporting was optimistic by
roughly 0.001–0.003 MAE", and its own conclusion names the honest band as "**≈ 1.3386–1.3412**". Yet AIDE's
node-selection rule takes the lowest reported metric, and the lowest reported metric is the in-sample number
from step 17. The private leaderboard then confirmed the honest estimates and refuted the headline:

```
1.34224 − 1.3369 = 0.00534
```

Private is worse than the champion's local score by 0.00534, and worse even than the *pessimistic* end of
AIDE's own honest band:

```
1.34224 − 1.3412 = 0.00104
```

This is the run's central weakness, and it is a scoring-protocol failure rather than a modelling one. AIDE
performed the correct experiment, recorded the correct answer in prose, and had no mechanism to let that answer
override a lower number in its journal.

Compute was the cheapest of the batch after s3e19. Total execution time was **1,860.2 s (31.0 minutes)** across
20 steps, mean 93.0 s, max 393.9 s:

```
1860.2 / 60 = 31.0
```

With zero failures, **none of that time was wasted on crashes**:

```
0 / 20 = 0.00
```

The champion node itself took **325.0 s** (from `journal.json`). Nothing in this run came close to the harness
timeout — the entire 20-step search cost less wall clock than a single timed-out step in the s3e11 or s3e14
runs.

### Specification fields

The **Loss Function** is **L1 / mean absolute error** at every stage: `regression_l1` for LightGBM, `MAE` for
CatBoost, `reg:absoluteerror` for XGBoost, and a median (0.5-quantile) objective for the stacker — the last
chosen deliberately at step 17 to keep the stack's objective aligned with the metric after the OLS stacker's
squared-error bias was diagnosed. The calibrator optimizes MAE directly by search. The **Optimization
Algorithm** is gradient boosting for stage 1 (**not** SGD/ADAM), interior-point linear programming (`highs`)
for the quantile stacker, and **coordinate descent** for the 28 thresholds. The **Learning Rate** is the
boosting shrinkage, `0.05` for all three base learners. There is **no Learning Rate Scheduler** (*N/A —
convergence is governed by 50-round early stopping per fold, not a schedule*) and **no Batch Size**
(*N/A — full-dataset boosting; the one mini-batched model in the run, the step-6 PyTorch MLP, scored 1.3669 and
was abandoned*).

**Training Duration** is 1,860.2 s across the whole search, of which the champion accounts for 325.0 s.
**Training Memory** was **not recorded** — no instrumentation, no cap. **Transfer Learning** is *N/A (no
pretrained weights)*; AIDE's analogue is its **journal-conditioned prompt**, and this run shows it working
unusually well — all fifteen non-draft plans open with "Building on…", most of them citing the exact prior best
CV score they intend to beat, and the step-8 breakthrough is explicitly attributed to "the hypothesis flagged
in memory". **Data Augmentation** is
*N/A (tabular)*; the analogues are the 11 engineered features and the step-19 seed-bagging experiment, which
failed to help.

**Reproducibility Standards** are good at the script level and absent at the search level. All seeds are fixed
(`random_state=42` for the base folds and all three learners, `123` for the meta-folds), the offset and
threshold grids are deterministic, and the champion re-runs as written. The search itself is not reproducible —
node proposals come from LLM sampling. There is no replay gate. The one reproducibility issue that *does* bite
is the calibration's in-sample fit, discussed above: the champion's reported metric is not an out-of-sample
quantity, and nothing in the harness flags that.

## Inference procedures

**Decision Threshold** is, unusually for a regression task, **the entire story** — this pipeline has 28 of
them. After the three base learners predict `log1p(Age)` and are inverted by `expm1`, and after the quantile
stacker combines them, the continuous output is clipped to `[1, 29]` and then discretized by
`np.searchsorted` against the 28 fitted thresholds, each nominally at a half-integer boundary
(1.5, 2.5, … 28.5) but shifted by up to ±0.35 by the coordinate descent. The result is clipped again to
`[1, 29]`, cast to `int`, floored at 1, and reindexed onto `sample_submission.csv`'s `id` order.

Note that the champion script does not hard-code the per-boundary strategy: it scores all three calibration
strategies on the OOF vector and takes whichever is lowest, so the exact submitted mapping depends on that
comparison. **Post-processing is therefore not a fixed step but a selected one**, and it is selected on the
same data it is fitted on. **Inference Duration** and **Inference Memory** were **not recorded**; AIDE times
whole scripts, not their scoring phase.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict integer `Age` for every test row and minimize MAE.

**Performance Metrics.** The champion reports **CV MAE 1.3369** locally, and scores **1.3392 public /
1.34224 private**, placing **rank 124/1431, 91.4th percentile** — AIDE's strongest placement across this
batch. The scored trajectory was 1.3597 → 1.3585 → 1.3582 → 1.3406 → 1.3402 → 1.3386 → 1.3374 → **1.3369**, but
that sequence is misleading after step 12, because from there onward the reported numbers are in-sample
calibration fits. The private result confirms the honest nested-CV estimates (1.3396 at step 16, 1.3412 at step
18) rather than the headline, and the public-to-private drift is in the usual direction:

```
1.34224 − 1.3392 = 0.00304
```

**Performance Benchmarking.** All three agents attacked this competition independently; the private-LB figures
below are the only cross-lane numbers used in this report.

| Agent | Approach | Private LB MAE | Note |
|-------|----------|---------------:|------|
| Our agent | from-scratch pipeline | 1.33859 | winner |
| NVIDIA | reproduce-agent distilling a public kernel | 1.34001 | second |
| **AIDE** | 20-step tree search → 3-model stack + 28-threshold calibration | **1.34224** | third |

```
1.34224 − 1.33859 = 0.00365
1.34224 − 1.34001 = 0.00223
```

`facts_aide.json` records `local_winner = mine` and `lb_winner = mine`. AIDE finishes third by 0.00365 and
0.00223 — margins comparable in size to the very optimism its own nested-CV steps measured (0.001–0.003 MAE).
That is the honest summary of this run: **AIDE found the right idea and then over-fitted it.** Rounding to
integers was the correct and decisive insight, worth 77% of everything the search achieved, and no comparator
lane found anything better on the modelling side either — all three agents sit within 0.004 MAE of one another,
on a dataset where every model family plateaus at ~1.358 before calibration. What separated them was restraint
in the last stage. A calibrator with a single global offset, honestly validated, would have been the safer
submission; AIDE had that number (1.3396) in its own journal and did not select it.
