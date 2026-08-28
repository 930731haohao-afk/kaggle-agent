# ML Specification Report — cat-in-the-dat

### Categorical Feature Encoding Challenge · our from-scratch agent (sparse-OHE linear pool + tree-search v3)

> *Figures grounded in the competition's `STATUS.md`, `config.yaml`, `dossier.json`, `experiments.json`, `experiments_tree_v3.json`, and `headless_run.log`, with leaderboard and cross-agent figures from the study's frozen three-way score table.*
> *Version discipline: this competition ran 2026-07-28 under the pre-v5 (v3-frozen) pipeline, the same configuration as the rest of the frozen 18-competition baseline — Stage 0.5 did not yet exist. The cited `dossier.json` was produced by the retrospective problem-identification sweep of 2026-07-29/30 and was NOT consumed by this run; it is cited only as a descriptive record of the task.*

## Overview

The task is binary classification from **23 exclusively categorical features** — the 2019 "cat-in-the-dat" challenge, explicitly designed as an encoding benchmark (metric: ROC-AUC, maximize). We solve it with a from-scratch pipeline — sparse one-hot + regularized logistic regression as the primary learner family, with LightGBM target-encoded and native-categorical members, refined through a linear protocol and a full 60-node `harness_v3` tree search. Our champion, tree-search node #55 (a kitchen-sink mega-blend of all 38 solos, 17 nonzero weights, LR-family-dominated), scores **CV OOF AUC 0.803272** (honest leave-fold-out weight re-fit 0.803231). On the real leaderboard it reaches **Public 0.80801 / Private 0.80241 — rank 362/1341, percentile 73.1** — winning the three-way benchmark on both local CV and the leaderboard: AIDE lands a razor-thin second (Private 0.80217), and the NVIDIA reproduce-agent trails far behind (Private 0.77084).


**Why it matters.** Categorical encoding is the unavoidable first decision of virtually every industrial tabular problem — IDs, product codes, hashed keys. This competition isolates that decision from everything else, so the answer it yields (when a sparse linear model beats GBDT, and how to target-encode without leaking the label) transfers directly to real-world pipelines.

---

## Data

**Purpose of Data.** Predict a binary `target` from categorical columns only — an i.i.d. tabular binary-classification benchmark with no numeric features at all. **Data Format** is clean **tabular CSV**. **Data Volume** is **300,000 training rows / 200,000 test rows**, with **23 raw predictive features** once `id` and the target are removed; the positive rate is **0.30588**.

**Data Quality** is high: **no missing values**, **no duplicate rows**, and strong evidence the test set is an i.i.d. split of the same synthetic population — `id` is a contiguous row counter (train 0–299999, test 300000–499999), there is no date column, and category level sets are near-identical across splits (all low-cardinality columns identical; nom_8 has 4 unseen test levels, nom_9 87 unseen out of 11,981). The strongest EDA signals were:

- **The ordinals' alphabetical order ≈ their target-rate order** (correlations 0.994 / 0.994 / 0.971 for ord_3/ord_4/ord_5) — so alphabetical integer mapping is a valid ordinal encoding.
- **High-cardinality nominals are opaque 9-char hex hashes** (nom_5–nom_9, up to 11,981 levels) with no external semantics — signal lives entirely in the encoding, and external data is useless despite being permitted.
- **day/month are anonymized cycles** (1–7, 1–12) with no year anchor — no absolute timeline exists.

**Annotation Guidelines.** The label is `target` ∈ {0, 1}; submissions must be **probabilities/scores, not hard labels**, ranked by ROC-AUC — a rank metric, so calibration and thresholds are irrelevant by construction.

**Feature Set.** All 23 raw columns are modeled, in two representations: a **sparse one-hot matrix over train+test union categories (16,552 columns)** for the linear members, and an **integer-encoded frame** (ordinals mapped by their documented orders, plus sin/cos of day/month) for the GBDT members, with **fold-safe target encoding (smoothing 20, nested 4-subfold on train rows)** over nom_0–9 / ord_5 / day / month for the TE variants.

| Group | Columns | Cardinalities |
|-------|---------|---------------|
| Binary (5) | `bin_0`–`bin_4` (bin_3 T/F, bin_4 Y/N → 0/1) | 2 each |
| Nominal low-card (5) | `nom_0`–`nom_4` | 3, 6, 6, 6, 4 |
| Nominal high-card (5) | `nom_5`–`nom_9` (hex hashes) | 222, 522, 1220, 2215, 11981 |
| Ordinal (6) | `ord_0`–`ord_5` (documented orders) | 3, 5, 6, 15, 26, 192 |
| Cyclical (2) | `day`, `month` | 7, 12 |

**Splitting strategy.** A single canonical **StratifiedKFold 5-fold CV (shuffled, seed = 42)** — the data is i.i.d. with no time or group structure, and stratifying on the 30.6% positive rate is cheap insurance. The dominant leakage risk here is **not the split but the encoders**: any target/likelihood encoding must be fit inside each training fold only (out-of-fold TE), never on the full training set — the discipline every TE member honors. The run itself was **leaderboard-blind** (OOF-only, closed-era benchmark run); the Public/Private figures quoted in Evaluation come from the study's late-submission scoring, not from any in-run feedback.

## Models & Architecture

**Purpose of Architecture.** A binary probability ranker that maximizes ROC-AUC. **Architecture Type** is unusual for this report set: the primary learner is **L2-regularized logistic regression on the sparse one-hot matrix** — on an all-categorical target with no continuous interactions, the linear model on OHE beats GBDT outright — with LightGBM members (fold-safe TE and native-categorical variants) kept for blend diversity. The champion is a **38-member kitchen-sink mega-blend** discovered at tree-search node #55, with **17 nonzero weights**.

Each linear member takes the same **Input Format** — the CSR one-hot design matrix — of **Input Dimension 16,552 sparse columns** per row; GBDT members consume the 23-column integer/TE frame (a 1-D vector; no spatial/sequence structure).

**Architecture Description.** The backbone member is LogisticRegression (`C` = 0.1, solver = lbfgs, `max_iter` = 2000, `tol` = 1e-5) on the full OHE matrix, solo OOF AUC 0.803103; the LRC lineage's grid refinement found the C optimum flat in [0.1, 0.15] with C = 0.12 best (0.803169). The blend layer is a convex weighted average of members' out-of-fold predictions, weights found by Dirichlet sampling (k = 800, cost-guard-coarsened to k = 200 on large blends) plus coordinate ascent. **Model Complexity** is dominated by the linear members (~16.5K weights each); the champion's weight mass is concentrated in the LR family — **lr_C0.12 at 0.3723**, lr_C0.1_root 0.1658, lr_C0.2 0.1403, with lgb_te_reg at 0.0732 and lr_C0.13 at 0.0447 — GBDT members contribute only a thin diversity sliver.

## Training procedures

Training proceeds as a **staged ladder** on the shared StratifiedKFold split, each stage measured on the same OOF AUC scale:

| Stage | Configuration | OOF AUC |
|-------|---------------|--------:|
| linear solo | LR on full sparse OHE (16,552 cols), C = 0.1 | 0.803103 |
| GBDT comparators | LGBM + fold-safe TE 0.797328 · LGBM native-categorical 0.776214 | — |
| linear blend | greedy blend, LR C=0.1 (0.75) + C=0.2 (0.25) | 0.803169 |
| tree search `harness_v3` (60/60 nodes, node #55) | 38-member kitchen-sink mega-blend | **0.803272** |

The tree search ran five lineages from the digit-verified root: **LRC** (C grid 0.07–0.5; best C = 0.12 → 0.803169), **LRPAIRS** (pairwise interaction crosses — **hurt everywhere**, best 0.796539 vs the 0.803103 base: explicit crosses damage a saturated sparse LR), **LGBTE** (tuning peaked at 0.798865 — the GBDT ceiling here is ~0.799), **SEEDBAG**, and **BLEND** (first ensemble seed beat the linear best at eval #10, 0.803226 > 0.803169). The champion came from the mandatory explore-burst mega-blend; both burst long-shots passed the sanity gate (deep LGB 0.797445, near-unregularized LR C=1.0 on crosses 0.770255 — both far below, as predicted).

The **Loss Function** is log-loss (logistic regression; LightGBM binary objective), with model selection on OOF AUC. The **Optimization Algorithm** is L-BFGS for the linear members and histogram gradient boosting for the LGBM members — **not** SGD/ADAM — with blend weights optimized by Dirichlet search + coordinate ascent. The **Learning Rate** is *N/A for the champion's dominant members* (L-BFGS has no learning-rate hyperparameter); the LGBM members use boosting shrinkage 0.05 (pool) / 0.03 (regularized variant). There is **no Learning Rate Scheduler** (*N/A — no SGD-style optimizer in play*) and **no Batch Size** (*N/A — full-dataset solvers, not mini-batched*).

**Training Duration** for the full pipeline was **~75 min of wall-clock** (headless run: EDA → prep → linear protocol → 60/60-node tree search); the root retrain took 12.5 s, and the cost guard coarsened every large blend evaluation (k = 800 → 200 whenever an eval exceeded the 45 s threshold; ~46–60 s observed on the mid-size blends, up to ~129 s on the 38-member mega-blends — always logged, never silent). **Training Memory** was not explicitly capped — a 300K × 16.5K sparse CSR matrix is modest on the single arm64 CPU machine, peak not recorded. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue is **cross-competition experience injection** — the TASK-CAT-ONLY prior ("encoding is the whole game; linear-on-OHE competes with GBDT") set the pool design, and the afsis lesson dictated reseeding the full linear pool as cached-OOF first-generation nodes. **Data Augmentation** is *N/A (tabular)*; the analogues are the dual encoding representations and seed-bagging (LGB seeds 42/2024/777/4000–4004).

**Reproducibility Standards** are strict: fixed fold seed 42 with a persisted fold assignment (`folds.npy`), LightGBM determinism flags with 10 threads, and a **digit-verified root** — the tree search retrained the linear-stage best from scratch and matched the cached 0.803103 exactly before any search step. The honest ledger records dedup_rejections = 10, the full cost-guard and backtrack logs, and the stop reason (hard budget cap, 60/60 evaluated nodes). This was a native v3 run — the champion is the search's own committed output.

## Inference procedures

**Decision Threshold** is *N/A* — ROC-AUC is a rank metric, so we submit raw predicted probabilities and never threshold; calibration and any monotone post-processing are provable no-ops under AUC, so **no post-processing is applied**. The unseen test levels (4 in nom_8, 87 in nom_9) are handled by `handle_unknown="ignore"` in the OHE (all-zero row block) and the global-prior fallback in TE. **Inference Duration** was not separately profiled; scoring the 200,000 test rows is seconds per member on CPU, and **Inference Memory** is negligible (sparse dot products and tree traversal, not capped). The submission (`submission.csv`, 200K rows, id/target) was format-validated against `sample_submission.csv`.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Rank the 200,000 test rows by probability of `target` = 1. **Performance Metrics.** ROC-AUC (maximize) is the sole competition metric; our champion reaches **CV OOF AUC 0.803272**, improving 0.803103 → 0.803169 → 0.803272 across the ladder, with the honest leave-fold-out weight re-fit at **0.803231** (in-sample weight optimism +0.000041). The tree search's gain over the linear blend is +0.000103 — noise-level per-fold — so its real value here was **negative-result mapping**: interactions dead, GBDT ceiling ~0.799, C optimum flat, confirming plain OHE+LR is essentially this feature space's ceiling at this budget.

**Performance Benchmarking.** Per the set's standing convention, **between-agent claims use the real leaderboard numbers; each agent's local CV describes only its own pipeline's internal view** (schemes differ across agents and are not comparable). The three-way comparison, from the study's frozen score table:

| Agent | Local CV AUC | Public LB | Private LB | Percentile |
|-------|-------------:|----------:|-----------:|-----------:|
| Our agent | 0.803272 | 0.80801 | **0.80241** | 73.1 |
| AIDE | 0.803255 | 0.80790 | 0.80217 | 70.3 |
| NVIDIA | 0.77478 | 0.77641 | 0.77084 | 32.6 |

**Our agent wins on both axes** (local winner and leaderboard winner), placing **362/1341 (percentile 73.1)** on the final standings with Private 0.80241. The instructive structure is the two-tier result: our agent and AIDE independently converge on the same sparse-OHE + logistic-regression solution and finish within 0.00024 private AUC of each other — evidence that this really is the encoding-benchmark's ceiling for a from-scratch pipeline — while the NVIDIA reproduce-agent's copied-kernel approach lands ~0.032 private AUC lower with a percentile of 32.6. On a competition built to test encoding judgment, deriving the right encoding from the data beat replaying someone else's notebook by a wide margin, and the tiny CV→LB gap on our side (0.803272 local vs 0.80241 private) shows the StratifiedKFold estimate was honest.

---

*Fields marked "not recorded": training peak memory (not separately measured); inference duration and memory (not separately profiled). The run itself was leaderboard-blind (OOF-only); all LB figures come from the study's frozen three-way score table.*

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions are being re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run are **TBD** until the separate credentialed operator scoring step; the figures below are the lane's own local CV only and revise nothing in the sections above.

- **Lane**: done 2026-08-14, 140 min, audit clean (1 minor).
- re-run local CV: **ROC-AUC 0.803272** (nested leave-fold-out), 5-member blend from a 70-node v3 tree search; dossier dispatched 8/8 typed operators.
- Public / private leaderboard for this re-run: **TBD** (pending operator scoring).
