# ML-Spec Report Structure — field checklist (顏佐榕 5-section framework)

Distilled from the 15 committed reports in `docs/ml_specs/`. Fill every field; use
`not recorded` for unavailable, `N/A (reason)` for DL-only fields under a GBDT pipeline.
Every number must trace to `facts.json` / the grounded sources (Hard Rule 1).

## Header

```
# ML Specification Report — playground-series-<comp>
### <Task title> · our from-scratch agent (<architecture, e.g. GBDT pool + tree-search v3>)

> *Figures grounded in the competition's `facts.json`, `eda_summary.json`, `STATUS.md`,
> `experiments_tree_v3.json`(and `config.yaml` / benchmark record where used).*
```

## 1. Overview  (= 目的 (甲) what + why)

- One paragraph: the task (with the metric + direction), the from-scratch approach, the
  **champion CV score**, and a one-line vs-NVIDIA outcome (winner / tie / behind + margin).
- **"Why it matters."** — a real, non-canned sentence on the problem's real-world stakes.

## 2. Data  (= 資料規格)

- **Purpose of Data** — what each row is and what is predicted (task type).
- **Data Format** — tabular CSV; numeric / categorical mix.
- **Data Volume** — N train / N test rows; # raw predictive features (after dropping id/target).
- **Data Quality** — missing values, duplicate rows, train/test covariate shift, and the
  **defining modelling wrinkle** (imbalance / top-coding / label noise / low signal / …).
  Surface source conflicts honestly.
- **Strongest EDA signals** — a short bulleted list (top correlations, key categorical
  effects, collinearity, leakage check).
- **Annotation Guidelines** — the label definition + range; what a submission is (prob /
  real value / integer) and how it is scored.
- **Feature Set** — a grouped table (Raw / Ratios / Log / Geo / Encoded / Interactions /
  Dropped) with the engineered feature count; note the champion's trimmed set if different.
- **Splitting strategy** — the canonical CV scheme (KFold / StratifiedKFold / bin-stratified
  / GroupKFold / LOYO / TimeSeriesSplit) + seed, **why** it fits the data, fold-safety of any
  target encoding, and the LB status (CV-only vs a real LB anchor).

## 3. Models & Architecture  (= 模型規格)

- **Purpose of Architecture** — the objective (minimize/maximize the metric).
- **Architecture Type** — GBDT ensemble (LGB/XGB/CAT) + blend; name the champion shape
  (solo / N-way blend / mega-blend) and its **tree-search node id**.
- **Input Format / Input Dimension** — numeric feature matrix; # features; note "1-D vector,
  no spatial/sequence structure".
- **Architecture Description** — representative member hyperparameters (num_leaves, depth,
  learning_rate, …); the blend layer (convex weighted average of OOF; weight-search method).
- **Model Complexity** — express as **trees × leaves**, not a dense parameter count; champion
  member weights (which members carry the mass).
- For **structural / non-GBDT champions** (e.g. s3e20 empirical-Bayes): say so; mark the
  DL/GBDT mapping N/A and describe the structural knobs instead.

## 4. Training procedures  (= 訓練規格)

- **Staged trajectory table** — each stage/tier → configuration → OOF score (the
  "breakthrough" trail ending at the champion).
- **Loss Function** — per-member objective (squared error / log-loss / L1 / RMSLE-on-log1p /
  post-rounder metric …); note model selection is on the (post-processed) competition metric.
- **Optimization Algorithm** — gradient boosting (histogram / leaf-wise / ordered) — **not**
  SGD/ADAM; blend-weight search method (grid / Dirichlet+coordinate-ascent / NNLS).
- **Learning Rate** — the boosting shrinkage value(s).
- **Learning Rate Scheduler** — `N/A (GBDT convergence is governed by CV early-stopping, not
  a schedule)`.
- **Batch Size** — `N/A (full-dataset histogram boosting, not mini-batched)`.
- **Training Duration** — wall-clock + # evaluated nodes/fits; Optuna trial count/time.
- **Training Memory** — `not recorded` (state the matrix size is trivial if true).
- **Transfer Learning** — `N/A (no pretrained weights)`; analogue = cross-competition
  experience-library / prior injection (name what transferred and whether it helped).
- **Data Augmentation** — `N/A (tabular)`; analogues = feature engineering + seed-bagging.
- **Reproducibility Standards** — fixed seeds (fold seed = 42), LightGBM determinism flags,
  and the replay / `06_rebuild_tree_best.py` OOF bit-reproduction gate result.

## 5. Inference procedures  (= 推論程序)

- **Decision Threshold** — the load-bearing post-process for threshold metrics (Accuracy →
  OOF-tuned cutoff; QWK → OptimizedRounder cutpoints; rounded-MAE → round+clip); or `N/A`
  for ranking (AUC) / squared-error (RMSE/R²) that submit raw values.
- **Other post-processing** — clip to range / snap-to-grid (note if tested & rejected).
- **Inference Duration** / **Inference Memory** — usually `not recorded` / negligible; note
  if the champion was OOF-only (no test predictions / no submission file).

## 6. Evaluation & Benchmarking  (= 評估指標)

- **Task for Performance Evaluation** — restate the prediction target.
- **Performance Metrics** — the metric + the stage progression (baseline → … → champion),
  relative improvement; state CV-only if no LB.
- **Performance Benchmarking** — the vs-NVIDIA reproduce-agent table:

  | Agent | Approach | <metric> | Note |
  |-------|----------|---------:|------|
  | NVIDIA | reproduces `<kernel>` (verbatim / essence / technique) | <score> | winner / tie |
  | Our agent | from-scratch GBDT pool + tree-search | <score> | winner / tie / behind |

  Then a paragraph with the honest framing (de-leaking audit if relevant, unified CV scheme,
  why the gap exists or collapses). Omit the table if no benchmark record exists.
