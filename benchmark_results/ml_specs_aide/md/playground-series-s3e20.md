# ML Specification Report — playground-series-s3e20

### Rwanda CO₂ Emission Forecasting from Satellite Data · AIDE agent (20-step tree search, Claude Sonnet via shim)

> *Metrics, step counts and timings are taken verbatim from the run's `facts_aide.json` (derived from `journal.json` for run `2-khaki-hamster-of-progress`). Pipeline description comes from AIDE's own auto-generated `report.md` and from `best_solution.py`; task metadata from `config.yaml`. The few figures that exist only in a node's plan/analysis text are attributed inline.*

## Overview

The task is to predict weekly CO₂ `emission` at Rwandan monitoring sites from satellite-derived pollutant readings and spatiotemporal metadata (metric: RMSE, minimize; target `emission`, id `ID_LAT_LON_YEAR_WEEK`). It is solved here not by a hand-built pipeline but by **AIDE**, a tree-search coding agent: at each step it writes a complete solution script, executes it in a sandbox, reads back either the printed validation metric or the traceback, and mutates the script accordingly. Given a **20-step budget**, AIDE's champion appeared at **step 15** with **holdout RMSE 21.17676** — an optimally weighted XGBoost + LightGBM blend built on leak-free location and location×week target encodings, a per-location year-trend extrapolation, a recency-weighted location-week encoding, cyclical week features, and rotated coordinates.

**No leaderboard verification was possible.** The submission attempt returned HTTP 403 with `RulesAcceptanceRequired` — Kaggle's late-submission path is closed for this account on this competition — so `facts_aide.json`'s score table carries a local metric and nothing else. This report therefore states the local holdout RMSE only; no public or private LB number exists for this run and none is estimated. The score table's `local_winner` field records **nvidia**, but the `mine_priv` / `nvidia_priv` columns that carry the three-way comparison in the other episodes are **absent here**, so no numeric agent-versus-agent statement can be made — only the recorded winner label.

**Why it matters.** Rwanda has almost no ground-based emission monitoring network; the practical question behind this dataset is whether freely available satellite retrievals can stand in for physical sensors well enough to track where and when emissions rise. Because the test period (2022) sits strictly after the training period (2019–2021), the competition is really a test of forward extrapolation under sensor drift, not of interpolation — which is exactly the regime a monitoring programme would face in deployment.

---

## Data

**Purpose of Data.** Predict a monitoring site's weekly CO₂ emission level — a continuous regression problem with a genuine temporal split. **Data Format** is **tabular CSV**, dominated by numeric satellite-sensor columns plus four spatiotemporal keys. **Data Volume**: the competition config records **75 features**, mostly satellite sensor readings (Cloud, Aerosol, UV, SO₂, CO, NO₂ measurement blocks), with training spanning **2019–2021** and test covering **2022**, and `week_no` running 0–52. Exact row counts are **not recorded** in this run's sources.

**Data Quality** is the defining difficulty. The config records **heavy block-structured missingness** — up to **99.4% missing in `UvAerosolLayerHeight`** — and a **heavily right-skewed target** (median 45.6, mean 81.9, max 3167.8). AIDE performed **no imputation at all**, relying on LightGBM/XGBoost/CatBoost native missing-value handling throughout; that is a defensible choice for tree models but means the sparsest sensor blocks contribute almost nothing. The skew was attacked once, at step 19, with a `log1p` target transform, and the search recorded the run's **worst post-baseline score, 25.65625** — the transform was abandoned. AIDE's own report attributes the failure to the mismatch between an RMSE evaluation on the original scale and a log-scale loss, plus `expm1` back-transformation bias in the high-emission tail.

**Annotation Guidelines.** The label is `emission`, a non-negative continuous quantity; submissions are real-valued predictions scored by RMSE, so the target of estimation is the conditional mean and the only structural constraint honoured is non-negativity.

**Feature Set.** The champion feeds the model every raw column except the id and the constructed location key, plus five engineered families:

| Group | Features |
|-------|----------|
| Raw spatiotemporal | `latitude`, `longitude`, `year`, `week_no` |
| Raw satellite sensors | the remaining pollutant/cloud/aerosol/UV columns, passed through with native NaN handling |
| Cyclical | `week_sin`, `week_cos` (period 52) |
| Rotated coordinates | `rot_{15,30,45,60}_x` and `rot_{15,30,45,60}_y` (8 columns) |
| Leak-free target encodings | `loc_target_enc`, `loc_week_target_enc`, `loc_year_trend`, `loc_week_recency_enc` |
| Constructed key (excluded from the matrix) | `loc_id` = latitude/longitude rounded to 3 dp |

The rotated coordinates are the single change that produced the champion: axis-aligned tree splits cannot cheaply isolate diagonal spatial groupings, and per the step-15 plan text AIDE added them explicitly to expose "diagonal spatial clusters" to the splitter. A coarser `region_id` (1 dp, ≈11 km) encoding and a location-relative pollutant-anomaly family were both tried and both landed slightly *above* the peak (21.43136 at step 18 and 21.49149 at step 17), so neither entered the champion.

**Splitting strategy.** A **single time-based holdout**: train on `year < 2021`, validate on `year == 2021`. This is the correct structural choice — random or grouped K-fold would leak future weeks of the same site into training and understate the forward-extrapolation gap the 2022 test set actually imposes. All target encodings are computed strictly from prior-period data (year-by-year expanding windows inside train, full-train statistics for test) with a layered location-week → location → global-mean fallback, so the encoding discipline is leak-free. The honest cost is **variance**: one validation year, no repeats, and every score in this report — including the 21.17676 champion — is a single-sample estimate. AIDE's own report flags the symptom directly, noting that the optimal XGB/LGBM blend weight swung across near-identical feature sets, which is what an over-fitted single-split weight search looks like. **Leaderboard status: none.** The competition returned 403 on submission, so the run is holdout-only and unverified against any Kaggle split.

## Models & Architecture

**Purpose of Architecture.** A non-negative continuous emission regressor minimizing RMSE under temporal extrapolation. **Architecture Type** is a **two-member gradient-boosted decision-tree blend** — XGBoost (depth-wise, histogram) plus LightGBM (leaf-wise) — combined by a scalar weight grid-searched on the 2021 holdout. CatBoost was evaluated (step 6, RMSE 24.44646) and later offered a slot in a three-model simplex search, where per AIDE's report the optimizer drove its weight to exactly zero; it is absent from the champion.

Both members take the same **Input Format** — a dense numeric matrix with NaNs left in place — of **Input Dimension**: the 75 config-recorded raw features minus the id and `loc_id`, plus 2 cyclical, 8 rotated-coordinate, and 4 target-encoding columns. The exact final column count is **not recorded** (`feature_cols` is computed dynamically from the training frame).

**Architecture Description.** The XGBoost member is `XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", random_state=42)` with `early_stopping_rounds=50` on the holdout; the LightGBM member mirrors it at `n_estimators=2000, learning_rate=0.03, max_depth=7, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42`, early-stopped at 50 rounds. **Model Complexity** is best read as trees × leaves rather than a dense parameter count: each member is capped at 2000 trees of depth 7, with LightGBM additionally leaf-capped at `num_leaves` = 63, and the effective tree count set by early stopping and then reused as a fixed `n_estimators` for the full-data refit. The blend layer is a one-dimensional grid over `w ∈ [0, 1]` in steps of 0.01 minimizing holdout RMSE on clipped predictions; per the step-15 node analysis, LightGBM was the stronger single learner and the grid settled on **w = 0.19** for XGBoost — i.e. the champion is roughly four-fifths LightGBM.

## Training procedures

There is no hand-designed stage ladder here. The "training procedure" is the **search itself**: 20 sequential nodes, each a complete rewrite-execute-analyse cycle.

### Search trajectory

AIDE ran **all 20 steps to completion with zero buggy nodes** — no exceptions, no timeouts, an empty `buggy_exc_types` map. That is the cleanest of the four runs in this batch and reflects how cheap this dataset is: **462.0 s** of total execution across 20 nodes, **mean 23.1 s**, **max 78.1 s**, nowhere near the 30-minute per-node cap that dominated the larger episodes.

| Step | Holdout RMSE | Node (per AIDE's report) |
|-----:|-------------:|--------------------------|
| 0 | 27.62991 | LightGBM baseline on raw features |
| 1 | 26.85647 | XGBoost + location TE + cyclical week |
| 2 | 24.57833 | location × week target encoding |
| 3 | 24.69419 | count-weighted Bayesian shrinkage of that encoding |
| 4 | 22.87344 | per-location linear year-trend extrapolation |
| 5 | 26.81841 | per-(location, week) year trend |
| 6 | 24.44646 | CatBoost with location-cluster categorical |
| 7 | 25.02996 | exponential-decay recency weighting |
| 8 | 22.15451 | linear-decay 2-year recency location-week encoding |
| 9 | 22.29519 | recency encoding on location only |
| 10 | 22.283 | global week-of-year encoding |
| 11 | 22.29519 | 3-year lookback recency encoding |
| 12 | 21.38273 | 50/50 XGB + LGBM average |
| 13 | 21.36361 | grid-searched blend weight |
| 14 | 21.36379 | 3-model simplex adding CatBoost |
| **15** | **21.17676** | **rotated coordinates + optimal blend — champion** |
| 16 | 22.18691 | drop all raw pollutant columns |
| 17 | 21.49149 | location-relative pollutant anomalies |
| 18 | 21.43136 | coarse regional target encoding |
| 19 | 25.65625 | `log1p` target transform |

The search improved the metric substantially over its own starting point:

```
27.62991 − 21.17676 = 6.45315
6.45315 / 27.62991 = 0.23355  →  23.4% RMSE reduction from first scored node to champion
```

Most of that came from three ideas, all data-representation rather than model choices: location×week target encoding (step 2), per-location year-trend extrapolation (step 4), and linear-decay recency weighting (step 8). Ensembling contributed a further step down at 12–13, and rotated coordinates supplied the last increment:

```
21.36379 − 21.17676 = 0.18703   (rotated coordinates, step 14 → step 15)
22.15451 − 21.17676 = 0.97775   (best single model, step 8 → blended champion, step 15)
```

**The honest weakness is what happened after step 15.** The champion arrived with a quarter of the budget still unspent, and **none of the last four nodes beat it** — 22.18691, 21.49149, 21.43136, 25.65625. AIDE spent 20% of its budget on ablations that all regressed, and closed the run on its second-worst score. The search also shows no cost-awareness pressure to stop: with no plateau detector, it simply kept mutating. Set against a single-year holdout whose blend weight AIDE itself observed to be unstable, a 0.18703 improvement at step 15 is inside the noise band one should expect from that validation design, and the run has no leaderboard to adjudicate it.

### Specification fields

The **Loss Function** is squared error — LightGBM's default `regression` (L2) objective and XGBoost's `reg:squarederror` — with model selection and early stopping both on RMSE, so loss and metric are aligned. The **Optimization Algorithm** is gradient boosting (histogram-based; leaf-wise for LightGBM, depth-wise for XGBoost) — **not** SGD or ADAM — with the blend weight found by exhaustive 1-D grid search. The **Learning Rate** is the boosting shrinkage, **0.03** for both members. There is **no Learning Rate Scheduler** (*N/A — GBDT convergence is governed by early stopping on the holdout, not a schedule*) and **no Batch Size** (*N/A — full-dataset histogram boosting, not mini-batched*).

**Training Duration** for the whole search was **462.0 s** (mean **23.1 s**/node, max **78.1 s**); the champion node itself is one of the cheaper ones. **Training Memory** is **not recorded** — AIDE does not instrument peak RSS. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue in an AIDE run is **within-run journal memory** — each node's plan is conditioned on the accumulated record of prior nodes and their metrics, which is why step 15 explicitly builds on "the strongest prior result … RMSE 21.36". Crucially, that memory does **not** cross competitions: AIDE begins every episode from the raw task description with no distilled priors. **Data Augmentation** is *N/A (tabular)*; the analogues here are feature engineering (rotated coordinates, cyclical encodings) and target encoding.

**Reproducibility Standards** are moderate. `random_state=42` is fixed in both learners and the encodings are deterministic, so a re-run of `best_solution.py` should reproduce the pipeline; but there is **no digit-for-digit rebuild gate** in the AIDE lane — the artifact of record is the emitted script, re-runnable end to end, not a verified reconstruction. The blend weight is selected on the same 2021 holdout used to report 21.17676, so the champion number is **selection-optimistic by construction**, and with no leaderboard there is nothing to measure that optimism against.

## Inference procedures

**Decision Threshold** is *N/A* — RMSE regression, so raw real-valued predictions are submitted with no thresholding. **Post-processing** is a **clip at zero** (`np.clip(pred, 0, None)`), applied to each member's predictions and again to the blend, enforcing the physical non-negativity of an emission; no rounding or grid-snapping is used. The final models are **refit on the full 2019–2021 training set** using the early-stopped iteration counts learned on the holdout, test-time encodings are computed from full-train statistics targeting 2022, and the submission is merged onto `sample_submission.csv`'s ID order to guarantee row alignment. **Inference Duration** and **Inference Memory** are **not recorded**.

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Predict weekly 2022 emission at each Rwandan site. **Performance Metrics.** RMSE (minimize) is the sole metric; AIDE's champion reaches **holdout RMSE 21.17676** at step 15, against a first scored node of **27.62991** — the 23.4% reduction computed above. The distribution of the 20 scored nodes is wide (27.62991 down to 21.17676), which reflects a search that genuinely explored rather than a plateau: unlike the tightly clustered Playground episodes, here representation choices moved the metric by whole units of RMSE.

**Performance Benchmarking.** This is the one competition in the batch with **no leaderboard row at all**. The submission returned `403 PERMISSION_DENIED / RulesAcceptanceRequired`, so there is no public LB, no private LB, no percentile and no rank for AIDE here, and no estimate is substituted. `facts_aide.json`'s score table contains exactly two usable fields:

| Field | Value |
|-------|-------|
| `aide_local` (RMSE, holdout) | 21.17676 |
| `local_winner` | nvidia |

The three-way comparison available elsewhere in this sweep — `mine_priv` / `nvidia_priv` against `aide_priv` — **does not exist for this competition**; those keys are simply absent. So the only defensible statement is the recorded one: on the local metric the run record names **NVIDIA** the winner, and AIDE's own local figure is 21.17676. Any numeric margin, any percentile or rank of the kind quoted in the other episodes of this sweep, and any claim about which agent would have placed higher on Kaggle's private split would be fabrication. The comparison for this episode is unresolved, and should be read as such.

---

*Fields marked "not recorded": exact train/test row counts and final feature-matrix width; training peak memory; inference duration and memory; the entire leaderboard row (Kaggle returned 403 — holdout-only, no public/private LB, no percentile, no rank); and the `mine_priv` / `nvidia_priv` figures needed for a numeric three-way benchmark.*
