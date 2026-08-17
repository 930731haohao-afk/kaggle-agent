# ML Specification Report — tabular-playground-series-sep-2022

### Book Sales Forecast (2021 daily panel) · our from-scratch agent (structural-ridge + LGB-ratio blend, GDP level arm)

> *Figures grounded in the competition's `STATUS.md`, `config.yaml`, `dossier.json`, `experiments.json`, `injection_ledger.json`, and `artifacts/tree_v3.json`, with leaderboard figures from the study's frozen three-way score table (see also `/home/tjyen/ai_agents/aideml-runs/FIRING_CLASS_THREE_WAY.md`).*

## Overview

The task is to forecast daily book sales (`num_sold`) for 6 EU countries × 2 stores × 4 products across all of 2021, trained on 2017–2020 (metric: SMAPE, minimize). We ran the full v5 pipeline — Stage 0.5 dossier with external-data injection (World Bank GDP per capita, per-country holiday calendars), a five-member model pool under time-holdout CV, a 60-node tree search, and a post-search blend audit — joining the frozen baseline 2026-07-31 as a firing-class extension (the original frozen set is 18 comps; `config.yaml` notes it is reported separately). Our champion is an **equal-weight 4-member cross-family blend** (2 × `ridge_struct` + 2 × `lgb_ratio`) at **fold2019 SMAPE 4.66570**, with the 2021 country level set by a GDP-proportional arm. On the real leaderboard this is the set's clearest failure case: **Public 28.32738 / Private 28.54859 (percentile 14.6)** versus **NVIDIA 4.77458 / 5.48351 (percentile 80.3, the winner)** and **AIDE 5.78998 / 9.72095 (percentile 48.3)**. The within-year shapes were right; the one decision the data could not test — which country-level regime 2021 follows — was bet wrong, and `STATUS.md` had flagged that regime as undecidable from data *before* submission.

**Why it matters.** Multi-country retail forecasting drives print-run, inventory, and staffing decisions a full year ahead; and this episode isolates the hardest, most consequential piece of that job — extrapolating the cross-country *level* when the training data's final year sits in a different regime — making it a stress test of how a pipeline behaves when its validation scheme literally cannot see the deciding variable.

---

## Data

**Purpose of Data.** Forecast units sold per day for every (date × country × store × product) series — a tabular time-series regression on a daily panel. **Data Format** is clean **tabular CSV** with four raw predictive columns (`date`, `country`, `store`, `product`) once `row_id` and the target `num_sold` are removed. **Data Volume** is **48 complete daily series** (6 countries × 2 stores × 4 products): train covers 2017-01-01 → 2020-12-31 (1,461 days, 48 × 1,461 = 70,128 rows) and test is all of 2021 (365 days, 48 × 365 = **17,520 test rows**, matching the submission file). A workspace note: the local data symlinks were circular/dead at session start, so official data was re-downloaded fresh via the Kaggle API.

**Data Quality** is high in the small — complete daily series, integer targets with min 19 (so no SMAPE zero trap) — but broken in the large: **2020 is a synthetic full-year equal-share regime**. All six country totals sit at 641,998–644,443 (share = 1/6 ± 0.005) from **January** 2020 onward — pre-COVID months included — so "COVID" is a superficial explanation; the generator equalized the whole year, and 2021's regime is therefore **unknowable from the data alone**. The strongest EDA signals:

- **GDP carries the country level exactly** in normal years: log-log elasticity of `num_sold` on GDP per capita is 0.996–1.003 with R² 0.9994 per year, 2017–2019.
- **Multiplicative structure is near-exact**: the store factor is 0.2575 constant across all years/countries; product day-of-year share curves are country/store-independent (max dev 0.0022) and year-stable (corr > 0.95) — except "Kaggle for Kids", which follows a **biennial cycle** (2017 ≈ 2019, 2018 ≈ 2020; min corr 0.526).
- **Day-of-week factors are shared across all dimensions** (0.943–1.14 weekend lift), with an end-of-year spike Dec 27–Jan 10 of up to 2×.
- The **noise floor is ≈ 4.2 SMAPE** (residual vs a rolling-median × dow reference) — the champion's 4.67 sits close above it.

**Annotation Guidelines.** The label is `num_sold` ∈ ℤ⁺ (integer daily units, min 19); submissions are sales forecasts scored by SMAPE, with the injected postprocess (**round-to-int, clip ≥ 1**) applied *inside* the metric function so every fold and blend sees it (TASK-MAE-METRIC prior).

**Feature Set.** The GBDT members use ~20–23 engineered features; the champion `lgb_ratio` uses 21:

| Group | Features |
|-------|----------|
| Calendar | `doy365`, `dow`, `is_weekend`, `month`, `day`, `odd_year` |
| Holidays (external, per-country) | `is_holiday`, `days_since_hol` (0..10), `days_until_hol` (0..5) |
| Cyclical | `doy_sin1..4`, `doy_cos1..4` (champion members use K = 2) |
| Categoricals | `country`, `store`, `product` |
| Trend | `year_c` (centered on 2019) |
| GDP (external; `lgb_join` arm only) | `gdp_pc`, `log_gdp` (World Bank, join on country × year, lag = 0) |

The `ridge_struct` family instead uses an explicit structural design: store/product/dow dummies, product Fourier terms, Kids-parity Fourier, EOY day dummies, and per-name holiday offsets. All 7 dossier-proposed injection operators were realized (`injection_ledger.json`); `sample_weight` for 2020 was *revised* by EDA (shapes intact, only the level broken — so no blanket down-weighting), and the `join_feature` GDP arm was raced and lost.

**Splitting strategy.** **Time holdout by year**: primary fold2019 (train 2017–18 → validate 2019), secondary fold2018, with shuffled KFold **forbidden** by the injected `split_policy` (TASK-TS-FUTURE prior: 4.5× optimism bias measured on s3e19). fold2020 is regime-broken — every member scores ~31–32 SMAPE on it — so it is excluded from level-form selection and used only for shape robustness. The leaderboard was not consulted during the run; all in-run scores are CV-only, with the single frozen submission made under benchmark conditions.

## Models & Architecture

**Purpose of Architecture.** A daily-sales forecaster that minimizes SMAPE under a strictly-future test window with an undecidable level regime. **Architecture Type** is a **two-family shape model plus a separated level arm**: (1) `ridge_struct` — L2-regularized linear regression on the explicit multiplicative-structure design above (champion members: alpha 2.7 and alpha 0.9, `fourier_k_global` = 2, `fourier_k_product` = 2, `kids_parity_k` = 0, EOY window day 355 → 10, per-name holiday windows −5..+10, dow × store interactions on); (2) `lgb_ratio` — LightGBM boosted trees on the GDP-normalized log-ratio target with `year_c` trend (champion members: `num_leaves` = 32, `learning_rate` = 0.025, K = 2 Fourier; the second adds `min_child_samples` = 10 for a diversity seed). The champion is the **equal-weight blend of these 4 members** (tree-search nodes #47, #48, #66, #67).

**Input Format / Dimension**: a per-row feature vector (21 features for `lgb_ratio`; a structural dummy/Fourier design matrix for ridge) — 1-D, with temporal information encoded as calendar/Fourier features. The decisive **shape/level decoupling**: models predict the within-year *shape*, and the 2021 country *level* is reconstructed as **k_hat 15.3434 × gdp_pc(country, 2021) / 365** (the GDP-proportional arm), with an equal-share arm raced in parallel. **Model Complexity** is trees × leaves for the LGB members and a sparse dummy design for ridge — small models; the intelligence is in the decomposition, not the capacity.

## Training procedures

Training proceeds through the linear members/blend protocol, then the tree search and its post-search blend audit, all on the shared fold2019/fold2018 holdouts:

| Stage | Configuration | fold2019 SMAPE |
|-------|---------------|---------------:|
| members_v1 best solo | `ridge_struct` (shape × GDP level) | 4.79072 |
| blend_v1, honest gate | 5-member greedy blend, weights fit on fold2018 | 4.76621 |
| tree search v3 best solo (node #66) | `lgb_ratio` {leaves 32, lr 0.025, K = 2} | 4.7388 |
| blend audit — **champion** equal4 [#47, #48, #66, #67] | equal weights, unfitted | **4.66570** |

The tree search (harness_v3, 60-node budget, judged worth it: cheap evals, many untuned structural knobs) evaluated 60/60 nodes (stop reason: hard budget cap), beat the linear best solo at eval 4 (node #3, 4.78717) and the linear honest blend at eval 40 (node #47, 4.74379). Its biggest structural find was **`kids_parity_k` = 0**: dropping the EDA-motivated biennial-parity Fourier terms improved 4.7683 → 4.74379 (fold2018 5.67 → 5.51) — EDA-true structure still failed the CV gate. Five lineages plateaued ([3, 30, 31, 2, 1]), the backtrack log holds 16 entries, dedup rejected 10 proposals (+ 5 burned placeholders), the sanity gate fired twice (both correctly), and the cost guard never fired. The blend audit ran Dirichlet sampling (k = 800) + coordinate ascent over the 60-member pool: in-sample best 4.65383 (kitchen-sink) and fitted pair weights (w#47 = 0.85 → 4.70484) were both **rejected by the honest gate** (weights fit on fold2018) in favor of unfitted equal weights over a small cross-family set — the honest-protocol variant scores 4.67082, and the 4.6657-vs-4.6708 difference is treated as noise band.

The **Loss Function** is L2 — closed-form ridge on the structural design, and LightGBM regression on the log GDP-ratio target — with model selection on SMAPE after level reconstruction and the round/clip postprocess. The **Optimization Algorithm** is ridge's closed-form solve plus histogram, leaf-wise gradient boosting — **not** SGD/ADAM — with blend weights from greedy forward search (linear round) and Dirichlet + coordinate ascent (audit). The **Learning Rate** is the boosting shrinkage (0.025 in champion members); there is **no Learning Rate Scheduler** (*N/A — GBDT convergence is governed by iterations, not a schedule*) and **no Batch Size** (*N/A — full-dataset fitting, not mini-batched*).

**Training Duration** wall-clock was **not recorded** in the artifacts; fits on a 70,128-row panel are cheap on the single arm64 CPU machine, and **Training Memory** was likewise not recorded. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue is the **dossier prior injection** — matched priors TASK-TS-FUTURE, TASK-TS-CALENDAR, TASK-MAE-METRIC, all 7 proposed operators realized — with the knowledge library **pinned to a pre-competition commit** so no experience bullet cites this competition. **Data Augmentation** is *N/A (tabular)*; the analogues are the external-data joins (GDP, holidays) and the structural feature engineering.

**Reproducibility Standards**: the tree-search root was digit-verified (`ridge_struct` 4.79072 == members_v1), and one **driver bug is honestly disclosed** — an off-by-one node-id guess made every in-search blend node fail (solo track unaffected), so the blend track was re-run post-search via `scripts/blend_audit.py` using the same procedure the harness would have applied. Final-fit decisions that CV cannot test are documented rather than scored: ridge members train on all 4 years (2020 shapes intact under shape normalization), `lgb_ratio` members on 2017–2019 only (2020 ratios regime-broken — e.g. Poland ratio 39.9 vs 15.4), and k_hat = 15.3434 is the mean of the 2017/18/19 cross-country k means (no trend; 2020 k invalid).

## Inference procedures

**Decision Threshold** is *N/A* — SMAPE is a regression forecast metric; the only postprocess is the injected **round-to-int + clip ≥ 1**, applied identically inside every metric call. Inference reconstructs each 2021 prediction as shape × GDP-arm level; the predicted 2021 daily country levels are BE 2171, FR 1838, DE 2201, IT 1549, PL 783, ES 1295 (≈ +12% over 2019, tracking 2021 nominal GDP recovery). The primary submission is `submission.csv` (17,520 rows); the raced **equal-share arm** is preserved at `submissions/submission_equalshare_arm.csv` (explicitly NOT primary). **Inference Duration** and **Inference Memory** were not separately profiled (linear + small-GBDT scoring of 17,520 rows is trivial).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Forecast 2021 daily sales for all 48 series. **Performance Metrics.** SMAPE (minimize) is the sole metric; the champion equal4 blend scores **fold2019 SMAPE 4.66570** (honest-protocol variant 4.67082; best solo 4.7388), close above the ≈ 4.2 noise floor. Known caveats were logged *before* the leaderboard was seen: the level estimator is validated on only 2 fold observations (k error +2.4% on 2019, −2% on 2018), member selection partially touched fold2019, the honest level-forecast cost is ≈ 0.24 SMAPE vs oracle level, and — decisively — the 2021 regime (GDP-proportional vs equal-share) is **not decidable from data**, with the GDP arm chosen on dossier priors (elasticity 1.000, R² 0.9994).

**Performance Benchmarking.** Per the set's standing convention, **between-agent claims use the real leaderboard numbers below; local CV describes the pipeline's internal view only.** From the study's frozen three-way score table:

| Agent | Approach | Public SMAPE | Private SMAPE | Percentile |
|-------|----------|-------------:|--------------:|-----------:|
| NVIDIA | reproduce-agent (public-kernel port) | **4.77458** | **5.48351** | 80.3 |
| AIDE | open-source from-scratch agent | 5.78998 | 9.72095 | 48.3 |
| Our agent | v5 pipeline (dossier + external-data injection) | 28.32738 | 28.54859 | 14.6 |

**NVIDIA wins** (`lb_winner` = nvidia); our agent finishes last, having hit its known failure mode: a **CV→LB decoupling** in which local 4.67 became public 28.3 because the champion bet the wrong 2021 country-level regime. The pub→priv movement is nearly flat (28.32738 → 28.54859), consistent with a systematic level bias rather than overfitting — the shape modeling held; the untestable regime bet did not. AIDE, by contrast, degrades sharply from public to private (5.78998 → 9.72095), while NVIDIA's ported kernel holds up — this is a competition where a mature public solution exists, which is exactly where the reproduce strategy is strongest. The methodological lesson the workspace records is not "the CV was wrong" — the time-holdout scheme, the honest weight gate, and the raced equal-share arm were all correct discipline — but that **when a decision is flagged as undecidable from data, the prior that breaks the tie carries the whole leaderboard outcome**.

---

*Fields marked "not recorded": training wall-clock and peak memory; inference duration and memory; leaderboard was not consulted during the run (CV-only until the single frozen benchmark submission).*

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions are being re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run are **TBD** until the separate credentialed operator scoring step; the figures below are the lane's own local CV only and revise nothing in the sections above.

- **Lane**: lane **queued**; re-run CV and LB: TBD.
