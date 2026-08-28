# ML Specification Report — us-patent-phrase-to-phrase-matching

### Patent Phrase Semantic-Similarity Scoring (NLP pair regression) · our from-scratch agent (DeBERTa-v3 fine-tunes + NNLS blend — deep-learning fallback lane)

> *Figures grounded in the competition's `facts.json` (built by `benchmark_infra/collect_dl_facts.py` from `config.yaml`, `experiments.json`, `scripts/*_cv.json`, `scripts/ensemble_run2_decision.json`, `STATUS.md`, `headless_run2.log`), with grading figures from the offline MLE-bench records in `benchmark_results/run2/`. This is a **special (deep-learning fallback) lane**: grading is **offline MLE-bench** against a held-out private split — there is no live Kaggle leaderboard, and the framework's DL fields (backbone, learning-rate schedule, batch size, epochs, transfer learning) are native here rather than translated from GBDT.*

## Overview

The task is to score the semantic similarity of a pair of patent phrases (`anchor`, `target`) within a CPC classification context, against human-rated labels in [0, 1]; the metric is the **Pearson correlation coefficient (maximize)**. Pair-similarity NLP is outside the GBDT tree-search pipeline's home turf, so the agent fell back to fine-tuning pretrained transformer encoders. The champion is an **NNLS blend of four members — deberta-v3-large at two seeds, deberta-v3-base, and a LightGBM lexical baseline** — scoring **honest leave-fold-out Pearson r 0.83184** (full-OOF 0.83206) and grading **0.86658** on the offline MLE-bench private split: a **silver medal** (silver threshold 0.863, gold 0.87). The NVIDIA reproduce-agent grades 0.8616 (bronze, exactly on its threshold) and AIDE 0.65821 (no medal) — **our agent wins the three-way comparison**.

**Why it matters.** Patent examination hinges on deciding whether a claimed phrase means the same thing as prior-art phrasing — "television set" vs "TV set", but judged inside a technical domain. Automated context-aware similarity scoring helps examiners and inventors search a corpus of millions of documents where verbatim matching fails.

---

## Data

**Purpose of Data.** Each row is a phrase pair — an `anchor` phrase, a `target` phrase, and a CPC context code — with a human-rated similarity `score` in [0, 1] to regress. **Data Format** is **tabular CSV of short text fields** plus a categorical context code. **Data Volume**: the graded test split has **3,648 pairs** (the deliverable submission's verified row count); the train-split row count was **not recorded** in this lane's structured records (no `eda_summary.json` was produced under the fallback's time budget).

**Data Quality.** Missing-value / duplicate profiling was **not recorded**. The defining wrinkle, established by the run's own honesty check, is the **train/test anchor overlap**: test anchors are 100% seen in train (the *pairs* are unseen), so a GroupKFold-by-anchor CV measures a strictly harder generalization problem than the graded one — the run recorded "true score likely ≥ CV", and the offline grade confirmed it (0.86658 graded vs 0.83184 CV).

**Annotation Guidelines.** The label is a human-rated similarity score in [0, 1]; a submission is a real-valued prediction per pair, scored by Pearson correlation — a pure linear-association metric, so raw (clipped) scores are submitted with no thresholding or calibration.

**Feature Set.** Two representations feed the members:

| Group | Representation |
|-------|----------------|
| Transformer input (3 DeBERTa members) | `anchor [SEP] target [SEP] <CPC section title> <context code>`, max_len 96 |
| Lexical features (LGBM baseline, 14 features) | `cos_char`, `cos_word`, `jaccard`, `n_common`, `len_a`, `len_t`, `nw_a`, `nw_t`, `len_ratio`, `a_in_t`, `t_in_a`, `same_first_word`, `context_code`, `section` |

**Splitting strategy.** The canonical scheme is **GroupKFold(4) grouped by anchor** — chosen because anchors repeat across many rows, and grouping prevents the model from scoring an anchor it has memorized. GroupKFold is seed-independent, so all transformer OOFs share identical folds and every blend score is computed on a consistent partition. The LGBM baseline used a StratifiedKFold(5) on score (seed 42) — a non-grouped, optimistic solo estimate, honestly flagged in the records; it still earns blend weight on the grouped-DeBERTa folds. **LB status**: offline MLE-bench grade only — no live public/private leaderboard.

## Models & Architecture

**Purpose of Architecture.** Maximize Pearson r between predicted and human similarity scores. **Architecture Type** is **fine-tuned pretrained transformer encoders + a lexical GBDT, combined by NNLS** — the DL fields are native. The champion is the **4-member NNLS blend** `nnls_all`, with weights:

| Member | Backbone | Seed | Solo OOF r | NNLS weight |
|--------|----------|-----:|-----------:|------------:|
| deberta_large_s1337 | `microsoft/deberta-v3-large` | 1337 | 0.8213 | 0.3659 |
| deberta_large_s42 | `microsoft/deberta-v3-large` | 42 | 0.8197 | 0.2618 |
| deberta_base_s42 | `microsoft/deberta-v3-base` | 42 | 0.8060 | 0.2531 |
| lgbm lexical baseline | LightGBM (14 features) | 42 | 0.6113 | 0.1192 |

**Input Format / Dimension.** Transformer members consume the tokenized pair-plus-context string truncated to **max_len 96**; the LGBM member consumes a 14-dimensional lexical feature vector.

**Architecture Description.** Each DeBERTa member is the pretrained encoder with a **mean-pooling head** over token embeddings and a **BCEWithLogits** objective on the [0, 1] score (sigmoid output). The blend layer is a **non-negative least squares (NNLS)** fit on OOF predictions, with the winning candidate chosen by **honest leave-fold-out refit** (weights refit on 3 folds, scored on the held-out fold). **Model Complexity**: parameter counts were **not recorded**; the members are fully identified by their published backbone names.

## Training procedures

Training proceeded as a **member ladder plus a blend-candidate search** (tree search deliberately skipped — see below), all scored as OOF Pearson r:

| Stage | Configuration | OOF r |
|-------|---------------|------:|
| 1 | LGBM char/word TF-IDF + lexical (safety baseline) | 0.6113 |
| 2 | deberta-v3-base, seed 42 | 0.8060 |
| 3 | deberta-v3-large, seed 42 | 0.8197 |
| 4 | deberta-v3-large, seed 1337 (seed bagging) | 0.8213 |
| 5 | NNLS blend of all 4 (honest leave-fold-out) | **0.83184** |

The blend stage evaluated 9 candidates by honest score: solo members (above), mean_larges 0.82598, mean_all_deberta 0.83015, nnls_larges 0.82597, nnls_deberta 0.83003, and the winner **nnls_all 0.83184** (full-OOF 0.83206).

The **Loss Function** is **BCEWithLogitsLoss** on the [0, 1] score, with model selection on Pearson r. The **Optimization Algorithm** is **fused AdamW** (weight decay 0.01); blend weights by NNLS. The **Learning Rate** (native) is **1e-5 (large) / 2e-5 (base)** for the encoder body with a separate **head LR of 0.0001**. The **Learning Rate Scheduler** (native) is **cosine decay with 10% warmup** — the recipe from the project's deberta-v3-large collapse memory, smoke-probed before committing (val_r 0.64 @ 40 steps) and delivering 8/8 clean large folds with zero collapses. The **Batch Size** (native) is **32**, for **3 epochs** per fold, under **bf16 autocast over fp32 master weights**.

**Training Duration**: **~2.5 h** for the whole run; a 4-fold deberta-v3-large fine-tune takes ~35 min. **Training Memory**: **not recorded**.

**Transfer Learning** (native DL field): **the load-bearing ingredient** — `microsoft/deberta-v3-base` and `microsoft/deberta-v3-large` pretrained checkpoints, explicitly allowed by the rules (`pretrained_models_allowed: true`); this is the run's main source of injected prior knowledge. **Data Augmentation** (native DL field): **none recorded**; the variance-reduction analogue actually used is **transformer seed bagging** (the same large recipe at seeds 42 and 1337, solos 0.8197 / 0.8213).

**Reproducibility Standards.** Fixed seeds (42 / 1337); GroupKFold's seed-independence guarantees identical folds across all members; all training jobs were foreground-waited (no orphaned background runs). The honest-vs-full-OOF gap of the chosen blend is only 0.0002, evidence the weight fitting is not overfitting the OOF. **Tree search was deliberately skipped**: each meaningful node is a ~35-min transformer fine-tune, so a ~60-node search is ~35 GPU-hours against the ~4 h budget; the light blend-level candidate search (9 candidates, honest scoring) served as the Stage-4 loop.

## Inference procedures

**Decision Threshold** is *N/A* — Pearson r is a correlation metric; raw real-valued predictions are submitted. The only post-processing is a **clip to [0, 1]** (the label's range). The submission (`submissions/ensemble_run2_nnls_all.csv`, promoted to `submission.csv`) has 3,648 rows, format-verified against `sample_submission.csv`. **Inference Duration / Inference Memory**: **not recorded**.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict test pairs' similarity scores. **Performance Metrics.** Pearson r (maximize): 0.6113 → 0.8060 → 0.8197 → 0.8213 → **0.83184** across the five stages; offline MLE-bench grade **0.86658** against thresholds gold 0.87 / silver 0.863 / bronze 0.8616 / median 0.851 — a **silver medal**, above median. Versus the archived run1 (OOF 0.8158, offline grade 0.8575), run2 gains +0.016 OOF, driven by the large backbone under the stable recipe plus seed bagging. The CV-to-grade gap (0.83184 → 0.86658) is consistent with the recorded anchor-overlap analysis: the grouped CV is conservative relative to the graded distribution.

**Performance Benchmarking.** Grading is **offline MLE-bench** (all three agents scored on the same held-out private split — not a live leaderboard):

| Agent | Approach | Pearson r (offline grade) | Medal |
|-------|----------|--------------------------:|-------|
| Our agent | from-scratch DeBERTa-v3 fine-tunes + NNLS blend | **0.86658** | **silver** |
| NVIDIA | (approach not recorded in the grade record) | 0.8616 | bronze |
| AIDE | (approach not recorded in the grade record) | 0.65821 | none |

Our agent is the only silver in the three-way; NVIDIA lands exactly on the bronze threshold (0.8616), and AIDE finishes below median. The honest caveats: the comparison is grade-only (the comparators' pipelines are not described in the run2 grade records), and our margin over NVIDIA rests on the honest blend rather than any single member — the best solo (0.8213 CV) would not have been separated from a well-executed reproduction.

---

*Fields marked "not recorded": train-split volume and missing/duplicate profiling (no EDA summary artifact); model parameter counts; training peak memory; inference duration and memory; NVIDIA/AIDE approach details. Grading is offline MLE-bench — no public/private live leaderboard exists for this lane.*
