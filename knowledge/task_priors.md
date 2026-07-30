# Task-Level Priors ([TASK-*]) — consumed by Stage 0.5 (Problem Dossier)

Task-TYPE knowledge (never competition-specific solutions). Every entry carries a trigger,
an action, and evidence. Evidence discipline follows `experience.md`: entries with score
evidence cite `competition, score A → score B`; literature-general entries without our own
score evidence are tagged [GEN] and count as unvalidated priors (idea-bank strength) until
a competition confirms them.

---

## TASK-TS-FUTURE — test set is a future time window
- **Trigger**: date/time column present AND test period strictly after train period (or
  description says forecasting).
- **Action**: time-based split mandatory (`year < y` vs `year == y`, or TimeSeriesSplit);
  shuffled KFold forbidden. Assess macro external data (see whitelist) — target series that
  depend on country/region economics usually need them. **Critically, when the test window
  lies outside the training range the covariate must carry the level, not decorate the
  features: use the `ratio_target` / `log_offset` operators, never `join_feature` alone.**
  A GBDT is piecewise-constant and cannot extrapolate; pair the ratio target with a
  `trend_term` for the drift that is common to all series.
- **Evidence**: s3e19 controlled split experiment 4.56 (shuffled) vs 20.41 (time split) —
  4.5× optimism bias; s3e19 outcome: all three agents 48.3–50.5 in the bimodal lower mode,
  top 4.67, accepted solutions join per-country GDP. tps-jan-2022: GDP-per-capita join,
  CV SMAPE 4.1793 vs 6.1551 (AIDE, official data only); the GBDT form of that recipe
  (`target = log(num_sold / gdp_pc)`) measured −2.6 SMAPE (8.43 → 5.80) and ablating the
  linear `year_c` drift term cost 4.83 → 6.02 on the 2017 fold (tpsjan22 exp #2/#3).
  Counter-evidence for the weak form: s3e19 v5 joined `gdp_pc` as a plain feature and the
  model predicted the 2022 level at 0.96× 2021 while the data implied +5..+33% growth —
  48.24 SMAPE, because a feature outside the training range cannot be extrapolated.

## TASK-TS-CALENDAR — daily/weekly retail-like series
- **Trigger**: daily-resolution series with country/store/product keys.
- **Action**: calendar features (day-of-week, month, holidays, Black-Friday-type events);
  holiday calendar join from whitelist.
- **Evidence**: tps-jan-2022 — AIDE discovered calendar + Black-Friday features from official
  data alone and reached 6.1551; combined with GDP (see TASK-TS-FUTURE) the family reaches ~4.2.

## TASK-CAT-ONLY — purely categorical features
- **Trigger**: all (or nearly all) feature columns categorical, incl. high-cardinality.
- **Action**: encoding is the whole game — target encoding with CV-safe folds, ordinal
  mappings for ordered categories, pairwise feature crosses; GBDT native categorical handling
  as a strong baseline.
- **Evidence**: cat-in-the-dat — my-agent private AUC 0.80241 (PR 73.1) vs NVIDIA 0.77084
  (PR 32.6, stale kernels); s3e11 — AIDE's only outright win came from a node that
  accidentally enabled native categorical handling.

## TASK-SPECTRAL — wide numeric spectra (columns ≫ rows)
- **Trigger**: thousands of ordered numeric columns (wavelengths/frequencies), few rows.
- **Action**: derivative/smoothing transforms (Savitzky–Golay-type), dimensionality reduction,
  per-target models for multi-target; strong regularization (rows are scarce).
- **Evidence**: afsis-soil-properties — 1,157 × 3,600; my-agent MCRMSE 0.49517 (PR 46.6) with
  an 80-node search; tree search on this comp: 0.44817 → 0.444076 CV while wall time fell
  32 → 17 min.

## TASK-STRUCT-OUT — structured multi-output with known generative rules
- **Trigger**: output is a grid/sequence governed by explicit rules (e.g., cellular automata).
- **Action**: exploit the structure directly (per-cell models, local neighborhoods,
  rule-informed features) rather than flat multi-output regression.
- **Evidence**: conway-s-reverse-game-of-life — my-agent MAE 0.10875, PR 98.6 (best three-way).

## TASK-MAE-METRIC — MAE-family evaluation metric
- **Trigger**: metric is MAE / SMAPE / quantile-like.
- **Action**: L1 objective (not L2) for training; rounding/threshold post-processing inside
  the metric function so every blend candidate sees it.
- **Evidence**: experience.md entry (s3e14/s3e16 family); postprocessing-in-metric rule from
  tree-search v2 (see `docs/tree_search_versions_note.pdf`).

## TASK-BLEND-DECOR — variance-heavy blend pool [transferable archetype]
- **Trigger**: blend pool dominated by deep/low-regularization members (variance-dominated).
- **Action**: add one shallow, heavily regularized, diverse-library member (bias-dominated)
  as a decorrelator before re-optimizing weights; for RMSE-family metrics verify weights with
  the NNLS closed form (Dirichlet search approximation error ~1e-5 can eat the real gain).
- **Evidence**: prior-wiring experiments — the archetype took the largest LLM-member weight in
  s6e1 (+3e-5 R²) and s4e1 (+7e-5 AUC), and s5e10's NNLS diagnostic reused it
  (`docs/prior_wiring_findings.md`).

## TASK-IMBALANCED [GEN] — heavily imbalanced binary target
- **Trigger**: positive rate ≲ 2%.
- **Action**: stratified folds; rank-based metrics need no resampling — calibrate only if the
  metric is probability-sensitive; beware accuracy-style thresholds.
- **Evidence**: none of ours yet — [GEN], unvalidated.

---

## External-data whitelist (Stage 0.5 / Plan C)

Only these sources may be proposed by the dossier; each join must state a leakage rule.

| Source | Content | Join key | Leakage rule |
|---|---|---|---|
| World Bank Open Data | 8 macro indicators: GDP family (level/per-capita/growth), population, CPI, unemployment, urbanization, internet penetration | country × year | only values dated ≤ the prediction year |
| Public holiday calendars (e.g., `holidays` pkg) | national holidays | country × date | deterministic future — no leakage |
| ISO country/region tables | codes, geo groupings | country | static — no leakage |
| Exchange rates (ECB SDMX) | monthly reference rates vs EUR, 12 currencies | currency × month | only values dated ≤ prediction month |
| OWID COVID (compact) | cases/deaths per million (raw + smoothed) | country × date | only values dated ≤ row date; relevant to 2020+ windows; ~178 MB one-time cache, download must be explicitly allowed |

**Rules**: (1) join keys must exist in the official data or be derivable from it; (2) no
future information relative to the prediction row's date — implemented as `lag` in
`external_data/join.py`: `lag=0` allows same-period values (historical-solution convention),
`lag=1` restricts to strictly earlier periods; state which lag a dossier assumes; (3) the join
script must log the source snapshot date; (4) competition rules must permit external data —
check before use.
