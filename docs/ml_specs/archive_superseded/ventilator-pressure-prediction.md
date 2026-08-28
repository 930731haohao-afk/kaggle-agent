# ML Specification Report — ventilator-pressure-prediction

### Ventilator Airway-Pressure Prediction (breath time series) · our from-scratch agent (BiLSTM seq2seq — deep-learning fallback lane)

> *Figures grounded in the competition's `facts.json` (built by `benchmark_infra/collect_dl_facts.py` from `config.yaml`, `experiments.json`, `cv_result.json`, `STATUS.md`, `headless_run2.log`, `scripts/train.py`), with grading figures from the offline MLE-bench records in `benchmark_results/run2/`. This is a **special (deep-learning fallback) lane**: grading is **offline MLE-bench** against a held-out private split — there is no live Kaggle leaderboard, and the framework's DL fields (learning-rate schedule, batch size, epochs, augmentation, transfer learning) are native here rather than translated from GBDT.*

## Overview

The task is to predict the airway pressure in a ventilator's respiratory circuit at each of the 80 timesteps of a breath, given the time series of control inputs (lung attributes R and C, commanded inflow `u_in`, expiratory-valve state `u_out`); the metric is **MAE computed only on inspiratory-phase rows (`u_out == 0`), minimized**. The GBDT tree-search pipeline does not apply to per-timestep sequence regression, so the agent fell back to a from-scratch deep-learning recipe: a bidirectional LSTM sequence-to-sequence model trained per breath. The champion is a **BiLSTM-4x256 over 53 per-timestep features, 5-fold ensemble aggregated by median and snapped to the pressure grid**, scoring **CV OOF masked MAE 0.19222** (raw 0.19361) and grading **0.16525** on the offline MLE-bench private split — no medal (bronze needs 0.1364, median is 0.1638), but by far the best of the three benchmarked agents (AIDE 0.34591, NVIDIA 0.40093).

**Why it matters.** Mechanically ventilating a sedated patient means clinicians hand-tune pressure control against an unseen lung; a model that accurately simulates circuit pressure for a given control input is a building block for ventilator controllers that adapt to individual lungs, easing clinician load and reducing ventilator-induced injury.

---

## Data

**Purpose of Data.** Each breath is an 80-timestep time series; the model must output the circuit `pressure` at every timestep, but only the inspiratory phase (`u_out == 0`) is scored — a sequence-to-sequence regression task. **Data Format** is **tabular CSV in long format**: one row per timestep carrying `breath_id`, lung attributes `R` and `C`, `time_step`, `u_in`, `u_out`, and (train only) the target `pressure`; rows group into fixed-length 80-step breaths. **Data Volume**: the graded test split has **603,600 scored rows** (the deliverable submission's verified row count), i.e. 7,545 breaths:

```
603600 rows / 80 timesteps per breath = 7545 test breaths
```

The train-split row/breath count was **not recorded** in this lane's structured records (no `eda_summary.json` was produced under the fallback's time budget).

**Data Quality.** Missing-value / duplicate profiling was likewise **not recorded**. The defining wrinkles are structural rather than statistical: (1) **only the inspiratory phase is scored**, so both the loss and the validation metric must mask expiratory rows; and (2) the target is not truly continuous — training pressures lie on a **950-value grid**, which the pipeline exploits at post-processing. One recorded diagnostic: the five fold models disagree wildly (std 2.57) **only in the unscored expiratory region**; their scored-region spread is 0.115.

**Annotation Guidelines.** The label is the recorded circuit `pressure` at each timestep; a submission is a real-valued prediction per row, scored by masked MAE — an L1 target that wants the conditional median, which later motivates the median fold-aggregator.

**Feature Set.** The model consumes **53 per-timestep engineered features** — run1's validated 41 plus 12 new ones (R×C cross one-hot 9, `u_in` cummax, cummean, reverse-cumsum) — all z-normalized into a `(n_breaths, 80, 53)` tensor:

| Group | Features (from `scripts/train.py`) |
|-------|------------------------------------|
| Raw inputs | `u_in`, `u_out`, `time_step`, Δt |
| Lung attributes | log1p of R, C, R·C; one-hots of R and C levels; 9 R×C cross one-hots |
| Cumulative / integral | `u_in` cumsum, time-integral (and C-scaled), cummax, cummean, reverse-cumsum |
| Local context | lags & leads of `u_in` at offsets 1–4 with their deltas; finite-difference derivative |
| Breath-level | `u_in` mean/max/std/first-value broadcast; `u_out` cumsum and switch-timing |

**Splitting strategy.** A **5-fold KFold on breaths** (the grouping unit — all 80 timesteps of a breath stay in one fold, preventing within-breath leakage), fold seed 42 (`SEED = 42` in `scripts/train.py`); validation replicates the scored mask exactly. All 5 folds were trained. Because this is an offline MLE-bench lane there is **no public/private live leaderboard**: the single offline grade (0.16525) is the only external anchor, and it came in *better* than the honest CV (0.19222) — the direction of that gap is noted honestly but is not diagnosed in the lane's records.

## Models & Architecture

**Purpose of Architecture.** A per-timestep pressure regressor minimizing masked MAE. **Architecture Type** is a **bidirectional LSTM sequence-to-sequence network** — a native deep-learning architecture, not a GBDT translation. The champion is a **5-fold ensemble of a single architecture** (five fold models aggregated by element-wise median), not a heterogeneous blend.

**Input Format / Dimension.** A `(n_breaths, 80, 53)` float tensor — 80 timesteps × 53 z-normalized per-timestep features per breath; the sequence axis is real structure (unlike the tabular lanes' 1-D vectors).

**Architecture Description.** An input projection Linear→LayerNorm→SiLU lifts each timestep into the hidden width, followed by a **4-layer bidirectional LSTM with hidden size 256**, and a 2-layer head emitting one pressure per timestep. The loss is masked L1 (scored region only), so training optimizes exactly what the metric measures. **Model Complexity**: the dense parameter count was **not recorded**; the architecture is fully specified by layers = 4, hidden = 256, bidirectional, over 53 input channels.

## Training procedures

Training was a **two-iteration linear ladder** (tree search deliberately skipped — see below), both iterations on the same task and metric:

| Stage | Configuration | Masked MAE |
|-------|---------------|-----------:|
| run1 (archived) | same BiLSTM recipe, 41 features, budget-bound to 1 fold × 100 epochs | 0.24004 OOF (single fold); offline grade 0.2315 |
| run2 (champion) | +12 features (53 total), bf16 + GPU-resident data, 5 folds × 260 epochs, median + snap | **0.19222** OOF snapped (raw 0.19361) |

Fold validation MAEs: 0.19432 / 0.19599 / 0.19258 / 0.18934 / 0.19583.

The **Loss Function** is **masked L1** on `u_out == 0` rows only. The **Optimization Algorithm** is **AdamW** (weight decay 1e-4 in `scripts/train.py`) with gradient-norm clipping at 5. The **Learning Rate** is 0.002 (peak), driven by a native **Learning Rate Scheduler: OneCycleLR over 260 epochs**. The **Batch Size** is **1024 breaths**, with **bf16 autocast** mixed precision.

**Training Duration** was **~2.6 h** wall-clock for the full 5-fold × 260-epoch run. The run log credits feasibility to throughput engineering, not architecture: bf16 autocast + the whole dataset resident on GPU + `cudnn.benchmark` cut epoch time from 46.8 s to 6.9 s (~6.8×), turning run1's "1 fold × 100 epochs" budget into "5 folds × 260 epochs". The foreground-only constraint was handled by **checkpoint/resume chunking**: 18 sequential ~9-min runs (`--chunk-seconds 520`, exit 3 = resume). **Training Memory**: **not recorded**.

**Transfer Learning** (native DL field): **none — trained from scratch**; the competition's rules forbid pretrained models (`pretrained_models_allowed: false`), so unlike the other two special lanes there is no pretrained backbone here. **Data Augmentation** (native DL field): **none recorded** in the lane's records.

**Reproducibility Standards.** Fixed seed 42; per-epoch seeded shuffle replay makes the chunked training **fully deterministic across restarts**; features and per-fold test predictions are cached under `scripts/work/`. **Tree search was deliberately skipped**: one evaluation node here is a full LSTM training run (~30 min/fold, ~2.6 h for an honest 5-fold CV), so a ~60-node search cannot fit the 4 h budget — the skill's documented fallback (single linear iteration, run1 → run2) applied instead.

## Inference procedures

**Decision Threshold** is *N/A* — a regression metric; raw per-row values are submitted. Two post-processing steps are load-bearing: (1) **median across the 5 fold models' test predictions** — the MAE-optimal aggregator (scored-region fold spread std 0.115; mean-vs-median absolute difference 0.037); then (2) **snap to the 950-value training pressure grid**, validated on OOF (0.19361 → 0.19222, a −0.0014 gain). The submission has 603,600 rows with id order verified against `sample_submission.csv`. **Inference Duration / Inference Memory**: **not recorded**.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict test-breath pressures on the scored inspiratory mask. **Performance Metrics.** Masked MAE (minimize): run1 0.24004 (single-fold OOF) → run2 **0.19222** honest 5-fold OOF (a −0.048 CV improvement); offline MLE-bench grade **0.16525** against thresholds gold 0.1119 / silver 0.1338 / bronze 0.1364 / median 0.1638 — **no medal, worse than median** (0.16525 > 0.1638). The lane's own record notes the medal threshold was "likely still out of reach" and that public solutions place ~0.15 territory at deeper models (5×384), second seeds, and ~300 epochs.

**Performance Benchmarking.** Grading is **offline MLE-bench** (all three agents scored on the same held-out private split — not a live leaderboard):

| Agent | Approach | Masked MAE (offline grade) | Note |
|-------|----------|---------------------------:|------|
| Our agent | from-scratch BiLSTM-4x256 seq2seq, 5-fold median + grid snap | **0.16525** | best of three, no medal |
| AIDE | (approach not recorded in the grade record) | 0.34591 | no medal |
| NVIDIA | (approach not recorded in the grade record) | 0.40093 | no medal |

Our fallback recipe more than halves both comparators' error. The honest caveats: nobody medalled — the recorded medal thresholds sit well below all three scores — and the comparison is grade-only (the comparators' pipelines are not described in the run2 grade records).

---

*Fields marked "not recorded": train-split volume and missing/duplicate profiling (no EDA summary artifact); model parameter count; training peak memory; inference duration and memory; NVIDIA/AIDE approach details. Grading is offline MLE-bench — no public/private live leaderboard exists for this lane.*
