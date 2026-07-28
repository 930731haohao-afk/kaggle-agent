# ML Specification Report — playground-series-s6e7

### Student Health-Risk Prediction (3-class) · AIDE agent (20-step tree search, Claude Sonnet via shim)

> *Scores grounded exclusively in `facts_aide.json` for run `2-rational-cryptic-manatee`; pipeline
> configuration read from AIDE's own `best_solution.py` and `report.md`, step-level intent from
> `journal.json`, and task metadata from `competitions/playground-series-s6e7/config.yaml`.*

## Overview

The task is to assign each of 295,753 test subjects to one of three health classes — at-risk / unhealthy /
fit — from lifestyle and physiological features, scored by **balanced accuracy**. This run was executed by
**AIDE**, a tree-search coding agent: each step writes one complete `solution.py`, runs it in a sandbox under
a 30-minute wall clock, reads back the printed metric or the traceback, and mutates. Over **20 steps** it
produced **17 scored solutions and 3 crashes**, and its champion — found at **step 17** — is a multinomial
**logistic regression stacker** over logit-transformed base-model probabilities, scoring **CV balanced
accuracy 0.94981** locally and **0.94982** on the public leaderboard, for **rank 895 / 2542 (64.8th
percentile)**.

**A structural caveat governs this entire report and must be stated before anything else.** The input
directory handed to AIDE for this competition did not contain only `train.csv`, `test.csv` and
`sample_submission.csv`. It also contained **seven pre-computed out-of-fold / test probability archives** —
`oof_cat.npz`, `oof_et.npz`, `oof_lgbm.npz`, `oof_lgbm_seed2024.npz`, `oof_lgbm_tuned.npz`, `oof_mlp.npz`,
`oof_xgb.npz` — i.e. the OOF predictions of a seven-model base ensemble that AIDE did not train. AIDE noticed
them at step 0 and, entirely reasonably, built a stacker on top of them; it **never trained a single base
model from raw features in the whole run**. Everything below therefore describes a *meta-learning* run, not an
end-to-end modelling run, and the comparison against the other lanes for this competition is not a
like-for-like agent comparison. This is a lane-isolation failure in the data staging, and it is reported here
rather than papered over.

**Why it matters.** Screening a large population into health-risk tiers from cheap lifestyle signals is a
triage problem where the minority classes are the ones that matter — which is exactly why the metric is
balanced accuracy and not accuracy. Per `config.yaml` the class prior is **at-risk 0.859 / unhealthy 0.084 /
fit 0.058**, so a majority-class predictor scores the 0.333 floor; every point above that comes from
correctly surfacing the two small classes.

---

## Data

**Purpose of Data.** Classify `health_condition` into three levels. **Data Format** is **tabular CSV** for the
raw features plus **seven NumPy `.npz` archives** of pre-computed probabilities. **Data Volume**, per
`config.yaml`, is **690,088 training rows / 295,753 test rows**; the raw feature set is 7 numeric columns
(`sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`,
`water_intake`) and 6 categorical columns (`diet_type`, `stress_level`, `sleep_quality`,
`physical_activity_level`, `smoking_alcohol`, `gender`).

**Data Quality** is, in this lane, almost entirely irrelevant — and that is itself the finding. Because AIDE
worked on the stacked probability matrix, it never inspected missingness, distributions, or train↔test shift
in the raw columns; there is no EDA artifact of any kind, since the agent's loop has no EDA stage. The one
piece of data-quality work it actually did was a schema-discovery bug hunt: at step 6 the script crashed with
a `ValueError` because AIDE's key-matching heuristic looked for `.npz` keys containing substrings like "oof"
or "prob", while the archives in fact use the bare keys `"oof"` and `"test"`. Step 8 diagnosed and fixed this
by matching on array shape (`shape[0] == n_train` / `== n_test`) rather than key name — and that shape-based
resolver is what survives into the champion script.

The defining wrinkle of the task, **severe three-class imbalance**, was handled correctly and early:
`class_weight="balanced"` at step 2 lifted the score from 0.92475 to 0.94971, by far the largest single gain
of the run. AIDE also tested the obvious alternative — post-hoc prior correction, dividing predicted
probabilities by empirical class frequencies — at step 4, and it collapsed the score to 0.8973, because it
double-corrects an imbalance that `class_weight` has already removed. That is a genuine controlled ablation,
and it is one of only two in the run.

**Annotation Guidelines.** The label is `health_condition` ∈ {at-risk, unhealthy, fit}, integer-encoded with
`LabelEncoder` for modelling and inverse-transformed back to the original strings for submission. The
submission is a **hard class label** per row, not a probability — balanced accuracy is computed on the argmax
decision.

**Feature Set.** The champion's feature matrix is not the competition's features at all:

| Group | Content | Width |
|-------|---------|------:|
| Base-model probabilities | 7 archives × 3 class probabilities, resolved by row-count shape match | 21 |
| Transform | `logit(p) = log(p / (1 − p))` with `p` clipped to `[1e-6, 1 − 1e-6]` | 21 |
| Scaling | `StandardScaler` fitted on the logit matrix | 21 |
| Raw lifestyle features | **none** — tested at step 18, rejected | 0 |
| Consensus statistics | **none** — tested at step 12, rejected | 0 |

The **logit transform is the single most important modelling decision in the run**: raw probabilities are
bounded and pile up near 0 and 1, whereas their log-odds are unbounded and roughly linear in the quantity a
logistic meta-learner is actually fitting. Step 2 (balanced weights, raw probabilities) scored 0.94971; step 3
(balanced weights, logit features) scored 0.94979. Every subsequent enrichment was rejected on the same
folds: appending the raw lifestyle features scored 0.94973 (step 18), appending per-class mean/std/min/max and
per-model entropy scored 0.94967 (step 12), and concatenating raw and logit representations into 42 columns
scored 0.94972 (step 19). All three lost to the plain 21-column logit stack.

**Splitting strategy.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for the reported metric,
with an inner **3-fold** stratified search over the regularization strength `C` to keep tuning inside the
30-minute box. This is the run's clearest methodological weakness and AIDE's own report names it: the `C`
chosen by the inner search and the score reported by the outer 5-fold evaluation come from the same training
data, so the reported 0.94981 is mildly optimistic as a generalization estimate. The competition was still
open at submission time (deadline 2026-07-31 per `config.yaml`), so `facts_aide.json` carries a **public LB
score but no private LB score**.

## Models & Architecture

**Purpose of Architecture.** Combine seven frozen base-model probability vectors into a single three-class
decision that maximizes balanced accuracy. **Architecture Type** is a **multinomial (softmax) logistic
regression meta-learner** — a linear model, not an ensemble of trees, and not a neural network. AIDE reached
this by elimination rather than assumption: it tried LightGBM as meta-learner (0.87968), an `MLPClassifier`
with 32 hidden units (0.87862), LDA (0.91463), QDA (0.94146), a LogReg+LDA 50/50 blend (0.92467), an
alpha-tuned LogReg+QDA blend (0.94979) and `LinearSVC` with hinge loss (0.94471). **Every non-linear and every
alternative-linear meta-learner lost to plain logistic regression**, and the tuned blend search drove its own
mixing weight to the pure-LogReg endpoint.

**Input Format** is a dense float64 matrix of standardized log-odds; **Input Dimension is 21** (7 models × 3
classes). **Architecture Description**: `LogisticRegression(max_iter=1000, class_weight="balanced",
C=best_C, solver="lbfgs", n_jobs=-1)`, where `best_C` is selected from the grid {0.01, 0.03, 0.1, 0.3, 1.0}
by the inner 3-fold search (AIDE's report records 0.01 as the winner — the most heavily regularized point on
the grid, which is consistent with 21 near-collinear features on 690k rows). **Model Complexity** is trivially
small:

```
3 classes × 21 features + 3 intercepts = 66 parameters
```

Sixty-six parameters sitting on top of seven frozen base models. The champion node is step 17, and it carries
no blend weights of its own — the "ensembling" all happened upstream in base models AIDE did not build.

## Training procedures

### The search trajectory

Of the **20 steps** in the budget, **17 produced a score and 3 crashed**:

```
20 − 17 = 3 buggy steps
```

The failures split into one logic error and two budget overruns (`buggy_exc_types: {"ValueError": 1,
"TimeoutError": 2}`), at the three steps absent from the metric trajectory — **steps 6, 10 and 14**. Step 6
is the `ValueError` (the `.npz` key-matching bug described above, fixed at step 8). Steps 10 and 14 are the
timeouts, each hitting the 1800 s harness wall (`exec_time_max_s` = 1800.1 s), so together they burned

```
2 × 1800 = 3600 s
3600 / 6312.7 ≈ 0.5703  →  57.0% of all execution time
6312.7 / 3600 ≈ 1.75 h total
```

**More than half of this run's compute produced no score at all.** Both timeouts have the same root cause
class — trying to make a linear model expensive. Step 10 expanded the 21 logit features into pairwise
interactions with `PolynomialFeatures(degree=2, interaction_only=True)`, taking the design matrix to 231
columns on 690,088 rows and stalling L-BFGS; step 14 ran an exhaustive 25-combination elastic-net grid with
the `saga` solver on the same data. AIDE diagnosed both correctly and repaired both — step 15 shrank the
search space and switched to a single train/validation split for the coarse stage, and step 17 simply deleted
the polynomial expansion.

And that is the run's most striking structural fact: **the champion is a bug-fix node.** Step 17's plan is not
an improvement proposal, it is a post-mortem ("the root cause of the timeout is the `PolynomialFeatures`
step … the fix is to drop it"), and the stripped-down repair it produced — 21 logit features, standardized,
L2, `lbfgs`, `C=0.01` — turned out to be the best solution AIDE found. The agent's best idea was the removal
of its own worst idea.

| Step | What AIDE changed | CV balanced accuracy |
|------|-------------------|---------------------:|
| 0 | LogReg stacker on raw base probabilities (baseline) | 0.92475 |
| 1 | LightGBM meta-learner + raw lifestyle features | 0.87968 |
| 2 | **`class_weight="balanced"` + tuned `C`**, raw probabilities | 0.94971 |
| 3 | **logit transform of the 21 probability features** | 0.94979 |
| 4 | post-hoc prior correction on the argmax | 0.8973 |
| 5 | LDA meta-learner | 0.91463 |
| 6 | QDA meta-learner | *ValueError* (`.npz` key matching) |
| 7 | `MLPClassifier`, 32 hidden units | 0.87862 |
| 8 | bug fix: shape-based `.npz` resolution → QDA runs | 0.94146 |
| 9 | LogReg + LDA 50/50 blend | 0.92467 |
| 10 | pairwise interaction expansion (21 → 231 features) | *TimeoutError* |
| 11 | LogReg + QDA alpha-tuned blend | 0.94979 |
| 12 | + per-class consensus statistics and entropy | 0.94967 |
| 13 | L1 penalty via `saga` | 0.9498 |
| 14 | elastic-net, 25-combination `saga` grid | *TimeoutError* |
| 15 | timeout repair: shrunken search, holdout coarse stage | 0.94974 |
| 16 | `LinearSVC`, hinge loss | 0.94471 |
| 17 | **timeout repair: drop interactions, scale, L2 `lbfgs`, `C=0.01`** | **0.94981** |
| 18 | + raw lifestyle features appended to the logit stack | 0.94973 |
| 19 | raw + logit dual representation (42 features) | 0.94972 |

The trajectory spans a wide band because of the failed model-family excursions —

```
0.94981 − 0.87862 = 0.07119
```

— but the *useful* range is minute. The two decisions that mattered both landed by step 3, and their combined
effect is

```
0.94981 − 0.92475 = 0.02506
```

After step 3 the search is flat to the fourth decimal: step 3 reached 0.94979, and the remaining **fourteen
scored steps bought**

```
0.94981 − 0.94979 = 0.00002
```

That is an honest plateau. AIDE spent roughly three quarters of its budget confirming, across six distinct
meta-learner families and four feature-enrichment variants, that a tuned linear model on 21 logit features
could not be beaten — a negative result that is genuinely informative, but a very expensive way to obtain it.
The agent had no mechanism to notice it had converged and stop.

### Training configuration

The **Loss Function** is multinomial cross-entropy (softmax log-loss) with **balanced class weights**, i.e.
each class's contribution is reweighted by the inverse of its frequency — the single change that took the run
from 0.92475 to 0.94971, and mandatory here because the metric is balanced accuracy on an 0.859 / 0.084 /
0.058 prior. The **Optimization Algorithm** is **L-BFGS**, a full-batch quasi-Newton method, with
`max_iter=1000` for the final fits and `max_iter=500` during the inner `C` search; regularization is L2 with
strength `1/C`. There is **no Learning Rate** in the usual sense (*N/A — L-BFGS selects its own step length by
line search; the tunable analogue is the regularization strength `C`, grid-searched over
{0.01, 0.03, 0.1, 0.3, 1.0}*), **no Learning Rate Scheduler** (*N/A — convergence is governed by the L-BFGS
tolerance and `max_iter`, not a schedule*), and **no Batch Size** (*N/A — L-BFGS consumes all 690,088 rows
per iteration; the run is full-batch by construction*).

**Training Duration** across the whole search was **6,312.7 s** of execution time over 20 nodes, mean
**315.6 s**, max **1,800.1 s** (the harness cap). Per-node wall time for the champion is not carried in
`facts_aide.json` — **not recorded**. **Training Memory** was never instrumented — **not recorded**; the
working set is a 690,088 × 21 float64 matrix plus the seven source archives.

**Transfer Learning** is the field where this competition is genuinely anomalous. Formally it is *N/A (no
pretrained weights)*, and AIDE has **no cross-competition memory** — every run starts cold, its only
continuity being the within-run journal that lets each plan quote its incumbent's score ("the best result so
far (0.94979)…"). But in substance, the seven pre-computed `.npz` archives *are* a transfer of learned
representations from outside the run, and AIDE consumed them. It is the reason a 66-parameter linear model
reaches the 64.8th percentile. **Data Augmentation** is *N/A (tabular)*; the analogues AIDE tried — feature
enrichment from the raw columns, cross-model consensus statistics, dual raw+logit encoding, pairwise
interactions — were tested at steps 18, 12, 19 and 10 respectively and **all four were rejected or timed out**.

**Reproducibility Standards** are adequate for a deterministic model and no more. `random_state=42` is fixed
on both the inner 3-fold and outer 5-fold splits; logistic regression under L-BFGS is deterministic given its
inputs, so there is no seed-sensitivity to control. But there is **no rebuild or reproduction gate** — AIDE
never retrains its champion to verify the cached score — and, decisively, **the run is not reproducible from
the competition data alone**: it depends on seven `.npz` artifacts produced outside the run, whose provenance
and generating code are not part of this lane. Re-running `best_solution.py` against a clean competition
download would fail at the first `np.load`.

## Inference procedures

**Decision Threshold** is *N/A in the binary sense* — the submission requires a hard three-class label, and
the rule used is a plain **argmax over the softmax outputs** of the final meta-model. AIDE explicitly tested
the alternative: step 4's prior-corrected argmax (dividing by empirical class frequencies before taking the
maximum) scored 0.8973 against the 0.94979 of the uncorrected rule, so the naive argmax was retained. No
per-class threshold search was ever run; AIDE's own report lists it as future work.

**Post-processing** is `LabelEncoder.inverse_transform` back to the original class strings, then column
reordering to match `sample_submission.csv`. Note that unlike the CV loop, the **final model is refit once on
all 690,088 training rows** (not a fold ensemble) before predicting the test set — so the submitted
predictions come from a single logistic regression, while the reported 0.94981 comes from five fold models.
**Inference Duration** and **Inference Memory** were not profiled — **not recorded**; scoring 295,753 rows
through a 66-parameter linear model is negligible.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Assign each of the 295,753 test subjects to one of three health classes,
scored by balanced accuracy (the unweighted mean of per-class recall).

**Performance Metrics.** AIDE's champion scores **CV balanced accuracy 0.94981** locally and **0.94982** on
the public leaderboard, for **rank 895 / 2542** = the **64.8th percentile**. The CV-to-LB agreement is as
close as this metric can plausibly get:

```
0.94982 − 0.94981 = 0.00001
```

which says the 5-fold stratified estimate is essentially unbiased here — unsurprising with 690k training rows
and 295k test rows, where both estimates are tightly concentrated. The competition was still open, so
`facts_aide.json` carries **no private LB score** and no percentile shift can be checked.

The full arc of the run, from first scored solution to champion, is

```
0.94981 − 0.92475 = 0.02506
```

of which the balanced class weighting and the logit transform account for essentially all — both landed by
step 3, and the following fourteen scored steps added 0.00002.

**Performance Benchmarking.** A numeric three-way table **cannot be given for this competition.**
`facts_aide.json`'s score table records the verdict fields `local_winner: mine` and `lb_winner: mine (pub)` —
i.e. the from-scratch agent won both the local and the public-LB comparison — but it does **not** carry
`mine_priv` or `nvidia_priv` values for s6e7, and inventing comparator numbers is not permitted. What can be
said is:

| Agent | Approach | Public LB balanced accuracy | Note |
|-------|----------|----------------------------:|------|
| AIDE | 20-step search → LogReg stacker on 21 logit features | 0.94982 | rank 895 / 2542, 64.8th pct |
| From-scratch agent | not carried in `facts_aide.json` | — | recorded as `lb_winner: mine (pub)` |
| NVIDIA | not carried in `facts_aide.json` | — | no figure available |

Two honest qualifications close this out. First, the comparison is **not like-for-like**: AIDE was handed a
seven-model OOF ensemble in its input directory and spent all twenty steps stacking it, so what is being
compared is one agent's end-to-end pipeline against another agent's meta-learner over borrowed base models.
Second, AIDE's own contribution to the score is small and quickly exhausted — two good decisions by step 3,
then a fourteen-step plateau worth 0.00002, with 57.0% of its compute lost to two self-inflicted timeouts on a
problem whose winning model has 66 parameters.
