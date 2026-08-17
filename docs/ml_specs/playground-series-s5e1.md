# ML Specification Report — playground-series-s5e1

### Forecasting Sticker Sales · our from-scratch agent (level/shape structural Ridge + honest-gated tree-search v3)

> *Figures grounded in the competition's `STATUS.md`, `config.yaml`, `dossier.json`, `experiments.json`, `experiments_tree_run2.json`, `final_selection.json`, `injection_ledger.json`, and the join logs, with leaderboard figures from the study's frozen three-way benchmark table (`FIRING_CLASS_THREE_WAY.md`).*

## Overview

The task is to forecast daily sticker sales (`num_sold`) for 90 retail series — 6 countries × 3 stores × 5 products — across all of 2017–2019, trained on 2010–2016 history (metric: MAPE, minimize). We solve it with the full v5 pipeline: a dossier-primed run with external-data injection (World Bank GDP per capita, national holiday calendars), whose decisive move is **level/shape decoupling** — a country-year *level* anchored on GDP-per-capita times a convergence-ratio rule, and a structural Ridge *shape* model fit on `log(y / level)`. Our champion is the **solo structural Ridge `ridge_level_conv50`**, scoring **CV MAPE 6.9281** (pooled rounded over expanding-year folds; raw-clip1 6.9719). A 60-node tree search found **no honest improvement** — every fitted blend collapsed under leave-year-out reweighting — so the pre-registered linear champion was submitted. On the frozen three-way benchmark our submission is the **best private-leaderboard score of the three agents** (private MAPE 0.12752, percentile 76.5, vs NVIDIA 0.14594 / AIDE 0.17917), with the paired verdict confirming mine > AIDE and the other pairs statistically indistinguishable.


**Why it matters.** Multi-country retail demand forecasting drives inventory, logistics, and revenue planning; as a three-year future-window extrapolation it also tests whether an agent can recognize that the cross-country level must be carried by an external macroeconomic covariate rather than extrapolated by a tree model.

---

## Data

**Purpose of Data.** Forecast daily units sold per (date × country × store × product) series — a tabular time-series regression Playground Series episode, target `num_sold`. **Data Format** is clean **tabular CSV** with four raw predictive columns (`date`, `country`, `store`, `product`) once `id` and the target are removed. **Data Volume** is **230,130 training rows (2010-01-01 → 2016-12-31, 2,557 days) / 98,550 test rows (2017-01-01 → 2019-12-31, 1,095 days)**, a complete 90-series daily panel on both sides.

**Data Quality** has one structural wrinkle: **3.85% of train targets are null**, and EDA shows this is **censoring of low values, not missing-at-random** — Kenya's cheapest series has non-null min = p10 = median = 5 with the null rate falling monotonically as demand grows (0.94 in 2010 → 0.08 in 2016), concentrated on weekdays; a Canada series is analogously censored at ~200; 2 series are all-null. Policy: nulls are dropped from fitting and excluded from CV scoring (MAPE is undefined against NaN). The strongest EDA signals, all pointing at a structured multiplicative panel, were:

- **Log-additive structure**: additive main effects on `log(y)` reach **R² = 0.979**, rising to **0.989** with a country × product interaction — strongly structured synthetic data (the tpsjan22 regime, not s3e19).
- **Store share is constant** (std 0.00035); **product share follows sinusoids** with 104-week and 52-week periods — bounded Fourier features, no extrapolation hazard.
- **GDP proportionality is converging**: the cross-country dispersion of `total(country, year) / gdp_pc` collapses **0.185 (2010) → 0.049 (2016)**, with every country's ratio trending toward ~1.0 — the regime the level anchor exploits.
- **A 2015 regime shift**: yearly totals fell **−10.9% in 2015 while GDP grew**; fold-2015 is the hardest year for every model.
- **Seasonality**: dow multipliers Mon–Thu 0.947 / Fri 1.00 / Sat 1.05 / Sun 1.157 (two country groups by dow profile), a Christmas/New-Year ramp to 1.30× around Dec 25–Jan 5, and a minor mid-May bump (doy 127).

**Annotation Guidelines.** The label is `num_sold`, an integer daily count; submissions are **continuous forecasts** scored by MAPE — a relative-error regression metric, so the multiplicative (log-space) treatment and denominator-side protection matter more than raw squared error.

**Feature Set.** From the four raw columns plus two external joins the pipeline builds `features.parquet` (26 features for the GBDT arm) and a wider explicit ridge design:

| Group | Features |
|-------|----------|
| Categoricals | `country_code`, `store_code`, `product_code` |
| Calendar / cyclical | `dow`, `doy`, `month`, `doy_sin1..4` / `doy_cos1..4`, `c104_sin1..2` / `c104_cos1..2`, `c52_sin1..2` / `c52_cos1..2` |
| External joins | `gdp_pc` / `log_gdp_pc` (World Bank per-capita, `merge_year_safe` lag = 0, snapshot 2026-07-13); `is_holiday`, `days_to_holiday`, `days_from_holiday` (holidays pkg 0.101, 6 countries, 10,530 holiday rows) |
| Ridge design (champion) | country/store/product main effects + c×p + s×p + c×dow interactions, Fourier (doy 8, product×104w 2, product×52w 2), per-holiday-name × offset(−5..+10) dummies |

Both joins are leak-audited: the GDP join matched **328,680 / 328,680 rows** (0 unmatched, leakage check passed); holidays are a deterministic future calendar (no leakage possible).

**Splitting strategy.** The dossier pinned this before EDA: the test window is a pure **three-year future** (2017–2019 after 2010–2016), so validation is **expanding-year folds — train < y, validate == y, for y ∈ {2014, 2015, 2016}** — and shuffled KFold is *forbidden* (a 4.5× optimism bias was measured for it on s3e19). The scheme is deterministic (no fold seed needed), and the pooled score is computed with the submission post-processing (clip + round-to-int) *inside* the metric so every search decision optimizes the submitted quantity. Fold-2015, the regime-shift year, dominates the pooled score for every model (10.3–11.9 vs ~5.1–5.5 for 2014/2016), and the honest gates below turn on exactly that fold. All in-run decisions were **CV-only** (no leaderboard feedback during the run); the finished submission was then scored once on the frozen three-way benchmark.

## Models & Architecture

**Purpose of Architecture.** A daily-sales forecaster that minimizes MAPE under future-window extrapolation. **Architecture Type** is — unusually for this report set — **not a GBDT blend but a solo structural linear model**: a two-part **level/shape decomposition**. The *level* `level(country, year)` is an anchor equal to `gdp_pc × conv50 ratio` — each country's sales-to-GDP-per-capita ratio carried forward under the `conv50` rule, i.e. 50% continued convergence toward the common cross-country ratio (the convergence fraction sweep 25/75/100 scored 7.258 / 7.542 / 8.404 against conv50's family best, a clean U-shape). The *shape* is a **Ridge regression (α = 1.0) on `log(y / level)`** over the explicit design above. A LightGBM pool (log-target, ratio-target, MAPE-weighted, and level-anchor variants) was raced throughout and populated the blend pools, but every blend was rejected by the honest gate.

Each model takes the same **Input Format** — a numeric feature row per (date × series) — of **Input Dimension 26 features** in the GBDT arm; the ridge design width is not separately recorded. **Model Complexity** for the champion is that of a closed-form linear solve (one coefficient per design column) — no trees, no iterative training; parameter count not recorded.

## Training procedures

Training proceeds as a **baseline race → level/shape iteration → blend gate → tree search → honest final selection** ladder, all on the identical expanding-year folds:

| Stage | Configuration | pooled MAPE |
|-------|---------------|------------:|
| 3 — baselines (exp #1–5) | best: `lgb_join` (26 features, log target, GDP join); also lgb_ratio 11.808, lgb_mapew 11.085, ridge_struct 13.675, ridge_cyc 11.086; naive 12.228 | 9.195 |
| iter — level/shape decoupling (exp #6–9) | LGB shape on `log(y/level)`, level rules last/avg2/trend/conv → 8.061 / 10.007 / 7.955 / **7.099** | 7.099 |
| iter — conv sweep + ridge shape (exp #11–15) | conv25/75/100 7.258/7.542/8.404; `ridge_level_conv50` raw 6.9719 → **rounded 6.9281** | **6.9281** |
| blend (exp #16) | greedy Caruana 3-member blend — in-sample 6.858, honest leave-year-out 7.619 → **REJECTED** | (rejected) |
| tree search v3 (exp #17) | 60 nodes: in-sample best 6.764 (38-member mega-blend) → **no honest improvement** | (rejected) |
| final honest selection (exp #18) | root solo + clip ≥ 5 + round-to-int | **6.9281** |

The **Loss Function** is L2 on the log-space shape target (`log(y/level)`), with model selection on rounded pooled MAPE; the metric-mimicking alternative — L1 with 1/y sample weights (`lgb_mapew`) — was raced and lost badly (11.085 vs 9.195 for plain L2-on-log), settling the dossier's open objective question by CV. The **Optimization Algorithm** for the champion is a closed-form ridge solve (α = 1.0) — **not** SGD/ADAM and not even boosting; the LGB pool members use histogram gradient boosting. There is **no Learning Rate** for the champion (*N/A — closed-form linear solve*), **no Learning Rate Scheduler**, and **no Batch Size** (*N/A — full-dataset fitting, not mini-batched*).

**Training Duration**: the tree-search stage ran **945.8 s of wall-clock over 60 evaluated nodes** (48 solo / 12 blend; stop = hard 60-node budget cap); the root champion refits in **9.2 s**. Runs were foreground, seeds fixed, threads = 10. **Training Memory** was not explicitly capped — a 230K-row panel is trivial on the single arm64 CPU machine, peak **not recorded**. **Transfer Learning** is *N/A (no pretrained weights)*; its analogue here is the **v5 dossier priming** — four matched task priors (TASK-TS-FUTURE, TASK-TS-CALENDAR, TASK-MAE-METRIC, TASK-BLEND-DECOR) injected the split policy, the GDP ratio/join race, the holiday windows, and the postprocess-inside-metric rule before any model was fit (injection ledger: 8 proposed / 3 realized data-layer / 5 consumed configs / 0 unrealized). **Data Augmentation** is *N/A (tabular)*; the analogues are the two external-data joins and the Fourier harmonics.

**Reproducibility Standards** are strict: fixed seeds throughout, a fresh tree-search evaluator + cache built for this run (lane isolation — the archived s5e1 caches and evaluators from the arms study were never touched, and `experience.md` contained no self-citing bullets to exclude), and the search root **digit-verified at 6.928121** against the linear stage before searching. The search's honest bookkeeping is fully logged: evals-to-match-linear-best = 1 (the root itself), 0 dedup rejections, 0 cost-guard firings, the burst sanity gate fired 2× (EXPL_CTRYTREND 7.520, EXPL_MAPEW 11.054 — both plateaued at seed), and 17 backtrack entries.

## Inference procedures

**Decision Threshold** is *N/A* — MAPE is a regression forecast metric, so we submit continuous predictions and never threshold. Post-processing is **clip ≥ 5** (the train minimum; a denominator-side MAPE protection, OOF 6.9279 vs 6.9281 for clip ≥ 1) then **round-to-int** (`num_sold` is a count; rounding is worth 6.9719 → 6.9281 on the champion and was applied inside the metric for all decisions). `submission.csv` is the champion's full-train test prediction — all 98,550 rows. **Inference Duration** was not separately profiled (a linear predict over 98,550 rows is sub-second) and **Inference Memory** is negligible (not capped).

## Evaluation & Benchmarking

**Task for Performance Evaluation.** Forecast 2017–2019 daily sticker sales for all 90 series. **Performance Metrics.** MAPE (minimize) is the sole competition metric; our champion reaches **CV MAPE 6.9281** (pooled rounded, expanding-year folds; raw-clip1 6.9719, whose per-fold scores are 5.104555 / 10.294318 / 5.533216). The decisive quality control is the **honest gate**: the greedy blend's in-sample 6.858 collapsed to 7.619 under leave-year-out reweighting, and the final LOFO audit ranked every alternative behind the pre-registered root — root_solo **6.9281** / eqw_top3 7.1170 / best_solo_lofo 7.3070 / blend_refit 7.4875 / mega_refit 7.5753. The tree search's in-sample 6.764 was therefore rejected; its 15.8 minutes bought a *negative* result — the third competition in a row confirming that in a structure-dominated task with a homogeneous pool, fitted-weight blends overfit the few validation folds.

**Performance Benchmarking.** Between-agent claims use the **frozen leaderboard numbers**; the local CV above describes the pipeline's internal view only (and is quoted in percentage points, whereas Kaggle reports MAPE as a fraction — different unit conventions, not directly comparable digits). All three agents submitted to the real leaderboard, joining the frozen baseline 2026-07-31:

| Agent | Public MAPE | Private MAPE | Percentile rank |
|-------|------------:|-------------:|----------------:|
| Our agent (mine) | 0.10236 | **0.12752** | **76.5** |
| NVIDIA | **0.05726** | 0.14594 | 69.3 |
| AIDE | 0.12573 | 0.17917 | 48.1 |

**Our agent takes the private leaderboard** (`lb_winner` = mine on private; paired verdict **M > A**, the other pairs statistically indistinguishable). The shape of the result is instructive: NVIDIA's reproduction posts a far better *public* score (0.05726) but falls to 0.14594 on *private* — a public→private shake-up — while our CV-only, honest-gated submission degrades least and finishes at percentile 76.5, the best final standing of the three. The caveat is stated plainly: only the mine-vs-AIDE gap survives the paired significance test, so the headline claim vs NVIDIA is "ahead on private, not statistically separable", not a proven win. This run used the full v5 pipeline — dossier priming plus external-data injection with the knowledge library pinned; the dossier's unresolved join-vs-ratio form question (LB historically picked `join_feature`, CV picks the level/ratio form) stays open here, since the level-anchor form's CV margin was large (6.93 vs 9.16 rounded) and it is what was submitted.

---

*Fields marked "not recorded": ridge design width / parameter count; training peak memory; inference duration and memory; per-stage wall-clock outside the tree search (only the 945.8 s search and the 9.2 s root refit are logged).*

---

## Final-Architecture Re-Run (v5 pipeline, Stage 0.5 in-lane) — 2026-08

All 20 baseline competitions are being re-run in an isolated root (`~/benchruns/myagent-rerun`) under one frozen architecture (`docs/rerun_manifest.json`): every lane runs the **full v5 pipeline including the Stage 0.5 problem dossier** (dossier.json written in-lane before modeling), holds no credentials, and is checked by a per-lane transcript audit. Leaderboard scores for the re-run are **TBD** until the separate credentialed operator scoring step; the figures below are the lane's own local CV only and revise nothing in the sections above.

- **Lane**: done 2026-08-15, 113 min.
- re-run local CV: **cross-arm blend MAPE 5.840798%** (ratio-arm alone 5.886112) — weights and global scale fitted on the same OOF rows they score, so optimistic relative to held-out; lane's own caveat recorded.
- Public / private leaderboard for this re-run: **TBD** (pending operator scoring).
