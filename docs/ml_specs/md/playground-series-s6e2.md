# ML Specification Report — playground-series-s6e2

### Heart Disease Prediction · our from-scratch agent (GBDT pool + tree-search v3)

> *Figures grounded in the competition's `facts.json`, `config.yaml`, `eda_summary.json`, `STATUS.md`, and `experiments_tree_v3.json`.*

## Overview

The task is to rank 270,000 patients by heart-disease probability (metric: ROC-AUC) on a synthetic
dataset derived from UCI Heart Disease. We solve it with a from-scratch pipeline — a pool of gradient-boosted trees
(LightGBM / XGBoost / CatBoost) refined by a four-stage ladder and a 60-node tree search that ends on a 37-member
mega-blend. Our champion scores **CV OOF AUC 0.955529** (submission file rounds to 0.95553); the NVIDIA
reproduce-agent, copying a public XGBoost/LightGBM/CatBoost k-fold kernel, reaches **0.9554** — effectively a **tie**,
with our agent nudging fractionally ahead.


**Why it matters.** Cardiovascular disease is a leading cause of death worldwide; risk-ranking from routine clinical features supports triage and preventive screening — as decision support, not diagnosis.

---

## Data

**Purpose of Data.** Predict whether a patient has heart disease (Presence vs Absence) — a binary-classification
Playground Series episode (S6E2). **Data Format** is clean **tabular CSV**, all numeric columns. **Data Volume** is
**630,000 training rows / 270,000 test rows**, with **13 raw predictive features** once `id` and the target are removed.

**Data Quality** is high: **no missing values** in either split, **0 duplicate rows**, and only modest train↔test mean
shift (largest ≈ +0.83% on `Number of vessels fluro`). The class balance is mild — **44.834% positive** (`Heart Disease`
= Presence), an **imbalance ratio of 1.23** — which keeps the metric (AUC) clean and makes class-weighting largely
unnecessary (confirmed in training). The strongest EDA signals, all consistent with cardiology intuition, were:

- **Thallium** is the dominant raw predictor (point-biserial r ≈ +0.606).
- **Chest pain type** (r ≈ +0.461), **Exercise angina** (r ≈ +0.442), **Max HR** (r ≈ −0.441), **Number of vessels
  fluro** (r ≈ +0.439), and **ST depression** (r ≈ +0.431) form the strong mid-tier.
- **Slope of ST** (r ≈ +0.415) and **Sex** (r ≈ +0.342) contribute moderately.
- **BP** is essentially uninformative (r ≈ −0.005); **FBS over 120** (r ≈ +0.034) and **Cholesterol** (r ≈ +0.083) are
  weak — a notable contrast with textbook risk-factor expectations.

**Annotation Guidelines.** The label is `Heart Disease` ∈ {Absence, Presence}, mapped to {0, 1} (1 = Presence).
Submissions are **positive-class probabilities** scored by AUC — a ranking, not a hard 0/1 decision.

**Feature Set.** Feb's focused **33-feature** set (13 raw UCI columns + 20 engineered) is carried into the July run and
used for the XGBoost/CatBoost pool members; the 20 engineered columns were, however, **rejected for the LGB member** by a
same-folds prior check (see the Training section), so the champion LGB family runs on the **13 raw features** alone:

| Group | Features |
|-------|----------|
| Raw UCI (13) | `Age`, `Sex`, `Chest pain type`, `BP`, `Cholesterol`, `FBS over 120`, `EKG results`, `Max HR`, `Exercise angina`, `ST depression`, `Slope of ST`, `Number of vessels fluro`, `Thallium` |
| Thresholds / bins | `Age_decade`, `BP_high`, `Chol_high`, `Thallium_7` |
| Heart-rate derived | `HR_reserve`, `HR_pct_max` |
| Interactions | `Age_x_MaxHR`, `Age_x_STdep`, `Age_x_Vessels`, `Chol_x_Age`, `BP_x_Chol`, `BP_x_Age`, `STdep_x_Slope`, `STdep_x_MaxHR`, `Vessels_x_Thallium`, `Angina_x_STdep`, `ChestPain_x_Angina` |
| Composite / ratios | `Risk_score`, `Chol_per_Age`, `BP_per_Age` |

**Splitting strategy.** A single canonical **Stratified 5-Fold CV (seed = 42, shuffle)**, shared across all four stages
so scores are directly comparable. Because `StratifiedKFold`'s split depends only on the target vector and seed, the
July folds reproduce the Feb v1 folds **byte-identically**, letting the reused Feb anchor sit on the same ladder. The
leaderboard is closed (late submission returns `403`), so all scores here are **CV-only**; Feb's best real submission
(v3, on a non-canonical multi-seed scheme) posted Public LB AUC 0.95332 as a loose sanity anchor.

## Models & Architecture

**Purpose of Architecture.** A binary heart-disease classifier that maximizes ROC-AUC. **Architecture Type** is a
**gradient-boosted decision-tree ensemble** — LightGBM as the primary learner, with XGBoost and CatBoost pool members,
combined by a weighted blend; the champion is a **37-member mega-blend** discovered at tree-search node #43.

Each member takes the same **Input Format** — a numeric feature matrix (no categorical encoding needed; all columns are
already numeric) — of **Input Dimension 13 features** for the champion LGB family (33 for the XGB/CAT members); a 1-D
vector with no spatial/sequence structure.

**Architecture Description.** Each GBDT member is an additive ensemble of decision trees grown by histogram-based,
leaf-wise gradient boosting (representative stage-2 LGB: ~646 boosting iterations; the Optuna-tuned champion LGB uses
`num_leaves` = 102 with a shallow `max_depth` = 4 and strong regularization). The blend layer sits on top as a convex
weighted average of members' out-of-fold predictions, with weights found by Dirichlet sampling plus coordinate-ascent
refinement. **Model Complexity** is therefore best expressed as trees × leaves rather than a dense parameter count: the
champion blends 37 pool members, though the weight search kept only **8 non-zero members — all LGB-family variants** of
the tuned base-13 LGB, so in practice the mega-blend is an intra-LGB-family variance reduction.

## Training procedures

Training proceeds as a **four-stage ladder** on the shared 5-fold split, each stage measured on the same OOF scale:

| Stage | Configuration | OOF AUC |
|-------|---------------|--------:|
| 1 | Feb v1 LGB+XGB anchor (34 features, fold-identical, reused not retrained) | 0.95498 |
| 2 | skill pipeline: 33-feature LGB+XGB+CAT blend (weights 0.15 / 0.25 / 0.60) | 0.955197 |
| 3 | linear iteration: base-13 prior check + `is_unbalance` ablation + Optuna 40-trial tune + seed-bag, 5-way blend | 0.955510 |
| 4 | tree-search `harness_v3` (60 nodes, node #43 = 37-member mega-blend) | **0.955529** |

Two experience-library **priors were re-verified by controlled same-folds comparison** in stage 3: (A) "explicit
product/threshold columns add nothing for GBDT" (s3e9) was **confirmed** — the 13 raw features (0.955238) beat the full
33-feature LGB (0.955009) by +0.000229, the *opposite* verdict to s6e1 but the *same* as s5e10; (B) "class-imbalance
weighting hurts a ranking metric" (s3e3/s4e1) was **confirmed a third time** — `is_unbalance`=True (0.954993) lost to the
default (0.955009) by −0.000016. The **Loss Function** is binary log-loss (LightGBM `binary` objective) with model
selection on AUC; the default **unweighted** objective was kept. The **Optimization Algorithm** is gradient boosting
(histogram, leaf-wise) — **not** SGD/ADAM — with blend weights optimized by Dirichlet search + coordinate-ascent. The
**Learning Rate** is the boosting shrinkage (tuned champion LGB `learning_rate` ≈ 0.0171); there is **no Learning Rate
Scheduler** (*N/A — GBDT convergence is governed by CV early-stopping, not a schedule*) and **no Batch Size** (*N/A —
full-dataset histogram boosting, not mini-batched*).

**Training Duration** for the stage-4 tree search was **~7,901 s of wall-clock over 60 evaluated nodes** (0 failed, 0
dedup rejections; the search stopped on the 60-node hard budget cap); the stage-3 Optuna tune was 40 trials / 515.1 s on
a fold-0 proxy. **Training Memory** was not explicitly capped — a 630k × 33 float matrix is comfortable on the single
arm64 CPU machine; peak was not separately recorded. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue
here is **cross-season experience injection** — distilled priors from s3e3/s3e9/s5e10/s6e1 carried in and each re-verified
on this data. **Data Augmentation** is *N/A (tabular)*; the analogues are feature engineering and seed-bagging.

**Reproducibility Standards** are strict: fixed seeds (fold seed = 42; Optuna seed and seed-bag seed = 2024 logged),
LightGBM determinism flags (`deterministic=True`, `force_row_wise=True`, fixed `num_threads`=16 across all models),
pinned from the very first run (the hard-won s6e1 fix). This was a **native v3 run — no replay needed**; the stage-4
OOF-reproduction gate (`06_rebuild_tree_best.py`) retrained all 37 members and matched cached OOFs digit-for-digit
(max|ΔOOF| = 0), recovering the blend AUC 0.955529 and the stored weights member-for-member.

## Inference procedures

**Decision Threshold** is *N/A* — AUC is a ranking metric, so we submit raw positive-class probabilities and never
threshold (a 0.5 cut would only be needed if hard labels were required). **Inference Duration** was not separately
profiled; GBDT scoring of the 270,000 test rows is sub-second per member on CPU, and **Inference Memory** is negligible
(tree inference is lightweight, not capped).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Rank the test patients by heart-disease probability. **Performance Metrics.** ROC-AUC
is the sole competition metric; our stage-4 champion reaches **CV OOF AUC 0.955529** (submission file `0.95553`),
climbing 0.95498 → 0.955197 → 0.955510 → 0.955529 across the four stages (stage 1→4 ≈ +0.0575%). The whole ladder moves
within a ~0.0006 AUC band — an honest "last-mile" competition where gains are real but small, and the mega-blend still
extracted a final +0.000019 over the linear stage's best blend. The Feb v3 prior-season submission (non-canonical
multi-seed CV) posted Public LB 0.95332 — reference only, as this run is CV-only.

**Performance Benchmarking.** The natural comparator is the **NVIDIA reproduce-agent**, which copies the strongest public
kernel:

| Agent | Approach | CV AUC | Note |
|-------|----------|-------:|------|
| NVIDIA | reproduces `kospintr/heart-xgb-lightgbm-catb-baseline-k-fold` (verbatim-ish, CPU) | 0.9554 | tie |
| Our agent | from-scratch GBDT pool + tree-search v3 | **0.95553** | nudges ahead |

The result is effectively a **dead heat**, with our from-scratch agent fractionally ahead of the reproduced public
kernel — consistent with the newer-season (S5–S6) pattern where the reproduce-vs-originate gap collapses, in contrast to
the small public-kernel edge seen on older S3-era problems. Both numbers are leak-free.

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions are being re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run are **TBD** until the separate credentialed operator scoring step; the figures below are the lane's own local CV only and revise nothing in the sections above.

- **Lane**: lane **queued**; re-run CV and LB: TBD.
