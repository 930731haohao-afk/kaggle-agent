# ML Specification Report — playground-series-s4e11
### Depression Prediction · our from-scratch agent (GBDT pool + tree-search v3)

> *Figures grounded in the competition's `facts.json`, `eda_summary.json`, `STATUS.md`, and `experiments_tree_v3.json`.*

## Overview

The task is to classify 93,800 survey respondents as depressed or not from demographic/lifestyle
features (metric: Accuracy). We solve it with a from-scratch pipeline — a pool of gradient-boosted trees
(LightGBM / XGBoost / CatBoost) refined by a four-tier ablation and a 60-node tree search that ends on an 8-member
blend. Because Accuracy needs a hard 0/1 label, every tier also optimizes a probability **decision threshold**. Our
champion scores **CV Accuracy 0.940235**; the NVIDIA reproduce-agent, distilling the essence of a public
XGB/CatBoost/LGB voting kernel, reaches 0.9399 — a dead heat, with our agent nudging ~0.0003 ahead of NVIDIA.
The third benchmark agent edges past both: the frozen three-way score table
(`benchmark_results/three_way_scores.csv`) records `aide_local` **0.94055** against our `mine_local` 0.940235, with
`local_winner` set to **aide**.


**Why it matters.** Early identification of depression risk from survey and lifestyle factors can route scarce mental-health resources to those most likely to benefit — as a screening aid, not a diagnosis.

---

## Data

**Purpose of Data.** Predict whether a survey respondent is depressed — a binary-classification Playground Series
episode. **Data Format** is clean **tabular CSV**, a mix of numeric and categorical columns. **Data Volume** is
**140,700 training rows / 93,800 test rows**, with 18 raw predictive features once `id` and the target are removed.

**Data Quality** is high on the structural axis — **0 duplicate rows** and no train↔test mean shift (all eight numeric
features within ±0.52%) — but two real wrinkles shape modelling. First, **class imbalance**: only **18.2% of respondents
are depressed** (`Depression`=1; 25,567 / 140,700), which drives the decision-threshold tuning below. Second, the
missing values are **structured, not random**: the *Working Professional or Student* split creates disjoint feature
blocks. Academic Pressure / CGPA / Study Satisfaction are ~80% null (112,802–112,803 rows) because they only apply to
students, while Work Pressure / Job Satisfaction are ~20% null (27,910–27,918 rows) because they only apply to
professionals. The strongest EDA signals, all consistent with domain intuition, were:

- **Age** is the dominant predictor (Pearson r ≈ **−0.56**); younger respondents (students) are far more depressed.
- The **Working Professional vs Student** split is the #1 structural feature: **students (19.8% of rows) have a 58.6%
  depression rate vs professionals (80.2%) at 8.2%**.
- **Suicidal thoughts** is highly predictive: *Yes* → 31.8% depression vs *No* → 4.9%.
- **Academic Pressure** correlates strongly with depression among students (r ≈ +0.48); **Financial Stress** correlates
  in both groups (r ≈ +0.23).
- High-cardinality text columns are noisy: **City** (98 values) and **Degree** (115 values) carry unseen categories in
  test, and *Sleep Duration* / *Dietary Habits* have rare junk categories (e.g. "50-75 hours", "Mealy").

**Annotation Guidelines.** The label is `Depression` ∈ {0, 1} (1 = depressed). Submissions are **hard 0/1 labels**
scored by Accuracy — a thresholded decision, not a raw ranking.

**Feature Set.** From the 18 raw columns we engineer **28 features**, grouped below:

| Group | Features |
|-------|----------|
| Binary encodings | `is_student`, `suicidal_thoughts`, `family_history`, `is_male` |
| Age | `age_decade`, `is_young` |
| Cleaned categoricals | Sleep Duration → 4 groups, Dietary Habits → 3 groups + other |
| Interactions | `pressure` (academic+work), `satisfaction`, `stress_x_pressure`, `suicidal_x_student`, `age_x_pressure` |
| Label-encoded high-cardinality | `City`, `Profession`, `Degree` |
| Structured-NaN handling | absent professional/student blocks filled with 0 |

**Splitting strategy.** A single canonical **Stratified 5-Fold CV (seed = 42)**, shared across all four tiers so scores
are directly comparable. A deliberate leakage audit — carried over from the s4e1 surname target-encoding lesson —
confirmed the safeguard is unnecessary here: the feature set uses **only label encoding** for the high-cardinality
categoricals (City / Profession / Degree) and **no target encoding of any kind**, so there is no analogous fold-safety
bug to correct and this run is honestly comparable to the February baseline. The **leaderboard for this run is not
recorded** (`facts.json` `leaderboard: null`); scores here are **CV-only**, with the February prior-season submission
(CV 0.93738 → Public LB 0.94093 → Private LB 0.93939) anchoring CV trustworthiness.

## Models & Architecture

**Purpose of Architecture.** A binary depression classifier that maximizes thresholded Accuracy. **Architecture Type**
is a **gradient-boosted decision-tree ensemble** — LightGBM as the primary learner, with XGBoost and CatBoost members,
combined by a weighted blend; the champion is an **8-member blend** discovered at tree-search node #26.

Each member takes the same **Input Format** — a numeric feature matrix with categoricals encoded (binary / label /
cleaned-group) — of **Input Dimension 28 features** per row (a 1-D vector; there is no spatial/sequence structure).

**Architecture Description.** Each GBDT member is an additive ensemble of decision trees grown by histogram-based,
leaf-wise gradient boosting. Two representative LightGBM configurations appear: the February baseline (`num_leaves` = 63,
~1266 boosting iterations) and the Optuna-tuned member (`num_leaves` = 48, `max_depth` = 3, `learning_rate` ≈ 0.070,
`min_child_samples` = 51). The blend layer sits on top as a convex weighted average of the members' out-of-fold
predictions, with weights found by Dirichlet sampling (k=800, seed=42) plus coordinate-ascent refinement, maximizing
OOF threshold-optimized Accuracy. **Model Complexity** is therefore best expressed as trees × leaves rather than a dense
parameter count; the champion blends 8 such members, dominated by node #10 (an LGB variant) at weight 0.603.

## Training procedures

Training proceeds as a **four-tier ablation** on the shared 5-fold split, each tier adding capability and measured on
the same OOF scale:

| Tier | Configuration | OOF Accuracy |
|------|---------------|-------------:|
| 1 | generic 3-model blend (18 raw label-encoded columns) | 0.93955 |
| 2 | skill pipeline: 28 features, 3-model LGB+XGB+CAT blend | 0.939488 |
| 3 | + linear iteration (is_unbalance ablation / Optuna 50-trial tune / seed-bag), 5-way blend | 0.9399 |
| 4 | + tree-search `harness_v3` (60 nodes, node #26 = 8-member blend) | **0.940235** |

(Note the honest non-monotonicity: the generic tier-1 blend on raw columns edged the tier-2 engineered blend by
~0.0001; the engineered features paid off only once the tier-3/tier-4 tuning and search were layered on.)

The **Loss Function** is binary log-loss (LightGBM `binary` objective) with model selection on threshold-optimized
Accuracy; the `is_unbalance` class-weighting was ablated and *removing* it was better (Δ −0.001230 when enabled) — a
notable result, since this was the first test of the AUC-era `is_unbalance` prior on a *threshold* metric, and the prior
direction transferred. The **Optimization Algorithm** is gradient boosting (histogram, leaf-wise) — **not** SGD/ADAM —
with blend weights optimized by Dirichlet sampling / coordinate-ascent. The **Learning Rate** is the boosting shrinkage
(LGB `learning_rate` = 0.05 baseline, ≈ 0.070 for the Optuna-tuned member); there is **no Learning Rate Scheduler**
(*N/A — GBDT convergence is governed by CV early-stopping, not a schedule*) and **no Batch Size** (*N/A — full-dataset
histogram boosting, not mini-batched*).

**Training Duration** for the tier-4 search was ~1,607 s of wall-clock over 60 evaluated nodes (the search log records
1,656.1 s total, 0 failed / 0 dedup rejections, `stop_reason` = "hard budget cap reached (60/60 evaluated nodes)"); the
Optuna tune was 50 TPE trials / 437.7 s, and the February baseline LGB fit in ~1266 iterations / 171.5 s. **Training
Memory** was not explicitly capped — a 140.7k × 28 float matrix is trivial on the single arm64 CPU machine, peak not
separately measured. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue here is **cross-season
experience injection** — the `is_unbalance` / Optuna fold-proxy tune / seed-bag priors were carried in from earlier
seasons' `experience.md`, with the `is_unbalance` prior tested for the first time on a threshold metric here. **Data
Augmentation** is *N/A (tabular)*; the analogues are feature engineering (the 28 features) and seed-bagging (Optuna-tuned
LGB re-fit at seed 2024).

**Reproducibility Standards** are strict: fixed seeds (fold seed = 42; Optuna and seed-bag seeds logged) and an OOF
digit-reproduction gate that passed before the submission was built — all 8 champion members were retrained from their
tree configs to **max|ΔOOF| ≤ 5.53e-14** vs the search's cached OOFs (well under the 1e-9 tolerance), the weight-search
replay recovered blend Accuracy 0.940235 exactly, and the decision threshold was re-derived at 0.49. This was a **native
v3 run (no replay needed)**.

## Inference procedures

**Decision Threshold** is **0.49** — the OOF-optimized probability cut the champion node #26 blend re-derived at
submission time. Unlike the s4e1 AUC task (which submits raw probabilities), Accuracy requires a hard 0/1 label, so the
93,800 test probabilities were thresholded at 0.49 before writing the submission. (For reference, the February baseline
had tuned a higher 0.75 cut on a single is_unbalance LGB; the blended, no-imbalance champion settled near 0.49.)
**Inference Duration** was not separately profiled; GBDT scoring of the 93,800 test rows is sub-second per member on CPU,
and **Inference Memory** is negligible (tree inference is lightweight, not capped).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Classify each test respondent as depressed or not. **Performance Metrics.**
Accuracy is the sole competition metric; our tier-4 champion reaches **CV OOF Accuracy 0.940235**, climbing 0.93955
(tier 1) → 0.939488 (tier 2) → 0.9399 (tier 3) → 0.940235 (tier 4). Reference baselines: a majority-class predictor
scores 0.81829, and the February prior-season run posted CV 0.93738 / Public LB 0.94093 / Private LB 0.93939 — reference
only, as this run is CV-only and its **own leaderboard result is not recorded**.

**Performance Benchmarking.** The natural comparator is the **NVIDIA reproduce-agent**, which distills the essence of the
strongest public kernel:

| Agent | Approach | CV Accuracy | Note |
|-------|----------|------------:|------|
| AIDE | open-source AIDE agent | **0.94055** | `local_winner` |
| Our agent | from-scratch GBDT pool + tree-search | 0.940235 | ahead of NVIDIA |
| NVIDIA | reproduces (essence) `mayukh18/feature-eng-cv-voting-xgb-catb-lgb` | 0.9399 | — |

Our agent and NVIDIA are effectively **tied** (our 0.9402 vs NVIDIA 0.9399, +~0.0003 to our agent), and AIDE sits
fractionally ahead of both as the frozen table's `local_winner` (0.94055) — all three inside a ~0.0007 band. Unlike the
earlier S3/S4-era churn problem (s4e1) where the well-optimized public kernel kept a small edge, here our from-scratch
pipeline matches and marginally exceeds the reproduce-agent — consistent with the newer-season pattern where the
reproduce-vs-originate gap collapses to a dead heat.

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions were re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run were produced by the separate credentialed operator scoring step and are recorded in `benchmark_results/rerun_three_way.csv`; the figures below revise nothing in the sections above.

- **Lane**: done 2026-08-14, 81 min.
- re-run local CV: **accuracy 0.940810** (5-fold StratifiedKFold, OOF-fitted threshold 0.480), 4-member weighted blend.
- Public / private leaderboard for this re-run (`benchmark_results/rerun_three_way.csv`): **Public 0.94195 / Private 0.94107**; `winner_priv` = **mine** (NVIDIA 0.94076, AIDE 0.94056).
