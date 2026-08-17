# ML Specification Report — conway-s-reverse-game-of-life

### Reverse Conway's Game of Life · our from-scratch agent (delta-conditioned residual CNN + synthetic-board engine + tree-search v3)

> *Figures grounded in the competition's `config.yaml`, `dossier.json`, `STATUS.md`, `experiments.json`, and `experiments_tree_v3.json` (plus `scripts/cnn_lib.py` for the training loop), with leaderboard and benchmark figures from the study's frozen three-way score table.*
> *Version discipline: this competition ran 2026-07-28 under the pre-v5 (v3-frozen) pipeline, the same configuration as the rest of the frozen 18-competition baseline — Stage 0.5 did not yet exist. The cited `dossier.json` was produced by the retrospective problem-identification sweep of 2026-07-29/30 and was NOT consumed by this run; it is cited only as a descriptive record of the task.*

## Overview

The task is an inverse problem on a known dynamical system: given `delta` ∈ {1..5} and a 20×20 binary board *after* `delta` Conway Game-of-Life steps ("stop"), predict the board `delta` steps *earlier* ("start"). The metric is MAE over 400 binary cells × 50,000 test rows — with 0/1 targets and 0/1 predictions this is exactly the per-cell error rate (minimize). We solve it with a from-scratch pipeline whose decisive lever is problem structure: the forward rule (B3/S23, dead boundary) was verified exactly, which unlocks **unlimited synthetic training data**, exact D4 symmetry augmentation/TTA, and a convolutional architecture whose locality matches the rule. Our champion, **cnn_v2** (a delta-conditioned 96-channel depth-8 residual CNN trained on 120k synthetic boards per fold, finished with per-delta decision thresholds), scores **OOF MAE 0.110352** locally and **0.10771 public / 0.10875 private** on the leaderboard — **rank 3/144, percentile 98.6** — beating both comparison agents (AIDE private 0.11065, NVIDIA private 0.11189) on local CV *and* on the leaderboard.


**Why it matters.** Reversing a many-to-one deterministic system is the canonical structured-output inverse problem — the same shape as deconvolution, de-noising, and trajectory reconstruction — and it tests whether an agent can recognize and exploit exact generative structure (a free simulator, symmetry groups, locality) instead of defaulting to generic tabular ML.

---

## Data

**Purpose of Data.** Predict the 400 binary `start` cells from `delta` + the 400 `stop` cells — operationally 400 correlated per-cell binary predictions per row. **Data Format** is **tabular CSV encoding a 2-D grid**: train has 802 columns (`id`, `delta`, `start.1..400`, `stop.1..400`), test has 402 (`id`, `delta`, `stop.1..400`); columns are row-major indices of the 20×20 board (verified by forward-simulation match). **Data Volume** is **50,000 train rows / 50,000 test rows**, with `delta` near-uniform over 1..5 in both files (train counts 9,880–10,110 per delta).

**Data Quality** is high: all cells are 0/1, no missing values, no empty boards, and test rows are i.i.d. draws from the same board generator (no time, group, or entity structure; train/test ids are separate namespaces). The strongest EDA signals, all consequences of the Life dynamics, were:

- **The forward rule is exactly known**: B3/S23 with a dead (zero-padded) boundary reproduces `stop` from `start` for 4,000/4,000 sampled rows (toroidal wrap manages only 1,201/4,000 — the board is finite, not a torus). Stage-1 EDA re-confirmed 2,000/2,000 on its own sample.
- **Density decays with delta**: start density 0.1434 overall; stop density 0.1377 @ d1 → 0.1187 @ d5. Edge rows/cols are sparser (0.104 vs 0.148 interior) — the dead boundary matters.
- **~10.6% of rows are fixed points** (422/4,000 sampled have start == stop — still-lifes under the finite board).
- **Baseline floors**: all-zero MAE 0.145128; per-delta heuristic (stop @ d1, zero @ d2–5) 0.139585; stop-as-start is 0.15509 overall and beats all-zero *only* at delta = 1 (0.117) — every candidate must clear the per-delta floor, not the overall one.

**Annotation Guidelines.** The labels are the 400 binary `start` cells; submissions must be **hard 0/1** — under MAE a calibrated probability is strictly worse than thresholding, and the 0.5 threshold is the MAE-optimal decision rule for a per-cell probability, so thresholds are tuned only inside the metric function.

**Feature Set.** The CNN members consume the raw grid — no tabular feature engineering stage exists. The LGB member uses hand-cut local features. The third "feature" is really a data engine: a **synthetic board generator** (density U(0.01, 0.99) → 5 warm-up steps → record start → `delta` steps → stop, discard empties) whose output matches the official train marginals almost exactly (density percentiles identical to 4 decimals).

| Member | Input representation |
|--------|----------------------|
| cnn_v1 / cnn_v2 | 6-channel 20×20 tensor: stop board + 5 one-hot delta planes (features learned) |
| lgb_v1 | per-cell 7×7 stop-board window + row/col position, per delta |

**Splitting strategy.** A single canonical **5-fold KFold stratified by delta (seed = 42)** — delta is the only structural covariate and difficulty rises steeply with it, so every fold must carry the same delta mix; rows are otherwise i.i.d. with no groups or time. The leak-free discipline specific to this competition: **synthetic boards may enter training folds but never a validation fold** — the 50k official rows are the only validation source, otherwise CV measures the simulator instead of the competition distribution. This rule is honored throughout. The leaderboard was closed (2014 deadline), so the pipeline run itself was **CV-only with no LB feedback**; the leaderboard figures in this report come from the study's later frozen three-way scoring.

## Models & Architecture

**Purpose of Architecture.** A per-cell probability model of the pre-image board, thresholded to hard 0/1 under MAE. **Architecture Type** is a **delta-conditioned fully-convolutional residual CNN** — convolutional locality matches the rule's 3×3 light cone, and zero padding matches the verified dead boundary — with a per-cell LightGBM as a decorrelating blend member and a per-delta weight + threshold blend layer on top. The champion is **cnn_v2 standing alone**: the per-delta Dirichlet + coordinate-ascent weight search over {cnn_v1, cnn_v2, lgb_v1} collapsed to pure cnn_v2 at every delta.

Each CNN takes the same **Input Format** — a 6×20×20 tensor (stop plane + 5 broadcast one-hot delta planes) — so **Input Dimension** is 2,400 values per row with explicit 2-D spatial structure; one network handles all five deltas (per-delta separate CNNs were tested and lose, 0.11848 vs 0.11575 at search grade).

**Architecture Description.** cnn_v2 is a residual stack of **96 channels × depth 8** convolutional blocks with BatchNorm and 400 per-cell sigmoid outputs (cnn_v1 is the ch64 predecessor). lgb_v1 is a per-delta, per-cell LightGBM (`n_estimators` = 150, `num_leaves` = 63, `min_child_samples` = 100, `learning_rate` = 0.1, deterministic + `force_row_wise` flags) — solo OOF MAE 0.122521, useful only as a candidate member. **Model Complexity**: a dense parameter count was not recorded; complexity is expressed as channels × depth (96 × 8 residual convs), with capacity governed jointly by the 120k-boards-per-fold synthetic training set.

## Training procedures

Training proceeds as a **baseline → solo → tree-search → full-grade ladder** on the shared stratified 5-fold split:

| Stage | Configuration | OOF MAE |
|-------|---------------|--------:|
| exp #1 | all-zero baseline | 0.145128 |
| exp #2 | per-delta heuristic (stop @ d1, zero @ d2–5) | 0.139585 |
| exp #5 | lgb_v1 (per-cell per-delta LightGBM) | 0.122521 |
| exp #4 | cnn_v1 (ch64 depth8, lr 2e-3, 20 ep, 100k synth/fold, D4 aug + 8× TTA) | 0.111928 |
| exp #7 | tree search v3, 23 nodes on fold-0 proxy (root 0.112929 → best node 7: 0.112899) | *(fold-0, not comparable)* |
| exp #8 | **cnn_v2 full-grade** (ch96, lr 3e-3, 18 ep, 120k synth/fold) 0.11039 → per-delta thresholds | **0.110352** |

The **Loss Function** is per-cell binary cross-entropy with logits (`cnn_lib.py`), with model selection on *thresholded MAE*, never on loss. The **Optimization Algorithm** is **AdamW** (weight decay 1e-4), the **Learning Rate** is 2e-3 for cnn_v1 and **3e-3 for cnn_v2** — the single best lever the tree search found — under a **OneCycleLR scheduler**, with **Batch Size 512**. Unlike the GBDT episodes in this set, the DL-specific fields are all real here, none N/A.

The **tree search (harness_v3)** ran on a **fold-0 proxy** (12.5k boards = 5M cells; full 5-fold CNN evals, ~8.4 min each, are unaffordable per node), root = digit-verified cached reuse of cnn_v1 fold-0 (0.112929). It deviated from the 60-node default by design and by intervention: budget 40 → 30, patience 12 → 5, burst forced at n_eval = 18 (CNN solos cost 5–7 min vs ~3 planned — logged in `backtrack_log`), finishing at **23 evaluated nodes** across ARCH / PERDELTA / SEEDBAG / LGB / BLEND lineages. Findings: **ch96 + lr 3e-3 is the best CNN** (0.113895 vs base 0.115752 at search grade); depth12 consistently hurts at the 10-epoch budget; kitchen-sink 12-member blends (0.11294–0.11298) lose to the lean blend; best node 7 added per-delta weights + a per-delta threshold sweep (LGB weight only 0.006–0.011, zero at delta = 1). Node 5 (6th eval) tied the linear-protocol blend; node 7 (8th eval) surpassed it; dedup rejections 6.

One lever was tested and **rejected** (exp #3): forward-consistency refinement — greedy cell flips minimizing forward mismatch plus a CNN prior — drives mismatch 0.099 → 0.068 but *worsens* MAE at every delta (0.11273 → 0.11732 at λ=0.2, 0.11647 at λ=0.5 on 3k fold-0 boards). Reverse Life is many-to-one: per-cell marginals are MAE-optimal, a single consistent pre-image is not. The REFINE lineage was dropped from the search.

**Training Duration**: search-grade CNN solos cost 186–428 s each (per-node `wall_s`); the full-grade cnn_v2 5-fold wall-clock was not recorded. **Training Memory** was not recorded. **Transfer Learning** is *N/A (no pretrained weights — every network is trained from scratch)*; its analogue is cross-competition prior injection (the afsis per-target prior motivated per-delta weights and thresholds as first-class blend options). **Data Augmentation** is genuinely present, twice over: unlimited **synthetic boards** (100k/fold for cnn_v1, 120k/fold for cnn_v2) from the verified simulator, and **exact D4 dihedral augmentation** (the rule commutes with all 8 square symmetries under a dead boundary).

**Reproducibility Standards**: fixed seeds throughout (folds/CNN/LGB seed 42, seed-bag member 777), LightGBM `deterministic` + `force_row_wise` flags, and the tree-search root verified digit-for-digit against the cached cnn_v1 fold-0 OOF before any node was expanded.

## Inference procedures

**Decision Threshold** is a real, tuned quantity here (unlike the regression episodes): predictions must be hard 0/1, and the champion applies **per-delta thresholds {d1: 0.51, d2–5: 0.49}** to cnn_v2's probabilities, fit on full OOF, shaving 0.11039 → 0.110352. Inference also applies **8× D4 test-time augmentation** (averaging the 8 exact symmetry transforms). **Inference Duration** and **Inference Memory** were not recorded. The final `submission.csv` was validated against `sampleSubmission.csv` — columns, id order, binary values, live-cell rate 0.072.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict the 400 start cells for all 50,000 test boards. **Performance Metrics.** MAE (minimize) is the sole competition metric; our champion reaches **OOF MAE 0.110352** (folds 0.111139 / 0.110576 / 0.110343 / 0.109182 / 0.110519), improving 0.145128 → 0.139585 → 0.122521 → 0.111928 → 0.110352 across the ladder — a 24% error reduction over the all-zero floor. On the real test set the champion scores **0.10771 public / 0.10875 private**; local CV was slightly pessimistic, consistent with the synthetic-only-in-train discipline.

**Performance Benchmarking.** Per the set's standing convention, **between-agent claims use the leaderboard numbers**; each agent's local CV describes its own pipeline's internal view only and is not compared across agents. The frozen three-way table:

| Agent | Approach | Local (own CV) | Public LB | Private LB | Percentile |
|-------|----------|---------------:|----------:|-----------:|-----------:|
| Our agent | from-scratch delta-conditioned CNN + synthetic data + tree search | 0.110352 | 0.10771 | **0.10875** | **98.6** |
| AIDE | open-source AIDE agent | 0.112181 | 0.10960 | 0.11065 | 97.9 |
| NVIDIA | reproduce-agent (public-kernel replay) | 0.11331 | 0.11094 | 0.11189 | 96.5 |

**Our agent wins on both axes** (`local_winner` = mine, `lb_winner` = mine): private-LB margin 0.0019 over AIDE and 0.00314 over NVIDIA, landing at **rank 3/144 (percentile 98.6)** against the final standings — the strongest placement in this sweep's grid-structured episode. (One source conflict, surfaced for honesty: `config.yaml` records 141 teams; the frozen table's 3/144 denominator is quoted verbatim.) The edge is attributable to structure exploitation: verifying the forward rule bought unlimited in-distribution training data, exact symmetry augmentation, and a boundary-correct architecture — levers a kernel-replay agent inherits only if the copied kernel had them, and which the leaderboard rewarded in the same order as our local CV predicted.

---

*Fields marked "not recorded": CNN parameter count; full-grade cnn_v2 training wall-clock; training peak memory; inference duration and memory. The pipeline run was CV-only (LB closed 2014); leaderboard figures come from the study's frozen three-way scoring.*
