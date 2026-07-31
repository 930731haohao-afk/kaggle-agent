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
  depend on country/region economics usually need them. **When the test window lies outside
  the training range, emit BOTH forms as separate arms — `join_feature` (covariate as a plain
  feature) and `ratio_target` / `log_offset` (covariate carries the level) — and select between
  them by a rule fixed before any score is seen. Do not assume either form wins.** The
  reasoning that a piecewise-constant GBDT cannot extrapolate a feature is sound and predicts
  `ratio_target`; the leaderboard has twice said otherwise (below). Pair whichever form is used
  with a `trend_term` for the drift common to all series.
- **Evidence**: s3e19 controlled split experiment 4.56 (shuffled) vs 20.41 (time split) —
  4.5× optimism bias; s3e19 outcome: all three agents scored 48.3–50.5 in the lower mode of a
  bimodal leaderboard. (The clause naming that competition's accepted solutions was struck on
  2026-07-30: Stage 0.5 forbids competition-specific solution knowledge as an input, and a prior
  library that carries it launders exactly the input class the isolation protocol excludes. The
  split experiment and the tps-jan-2022 self-experiment below carry this prior on their own.)
  tps-jan-2022: GDP-per-capita join,
  CV SMAPE 4.1793 vs 6.1551 (AIDE, official data only); the GBDT form of that recipe
  (`target = log(num_sold / gdp_pc)`) measured −2.6 SMAPE (8.43 → 5.80) and ablating the
  linear `year_c` drift term cost 4.83 → 6.02 on the 2017 fold (tpsjan22 exp #2/#3).
  Counter-evidence for the weak form: s3e19 v5 joined `gdp_pc` as a plain feature and the
  model predicted the 2022 level at 0.96× 2021 while the data implied +5..+33% growth —
  48.24 SMAPE, because a feature outside the training range cannot be extrapolated. Using the
  `ratio_target` operator instead, on the identical configuration, gave CV 10.148 → 7.793.
  **Bound on the recipe (held-out evidence):** on tps-sep-2022 the same operator *hurt*
  (11.348 → 11.807; 11.515 with the 2020 COVID year down-weighted), and the proportionality
  diagnostic predicted it — cross-country dispersion is ~0.01 in 2017–2019 but 0.466 in 2020.
  Run the diagnostic and RECORD its verdict — but do not gate on it: see below.
- **The proportionality diagnostic was falsified as a form predictor (2026-07-31, pre-registered
  test on s5e1).** The diagnostic said s5e1 was the worst violation in the class (dispersion
  0.05–0.21, broken 6 of 7 years) and a prediction was committed IN ADVANCE that `ratio_target`
  would therefore lose to `join_feature` by more than on s3e19/sep-2022. The opposite happened:
  ratio_target won decisively (private MAPE 0.12417 vs 0.15626, paired-test significant). Per the
  falsifier's own pre-committed terms, **this diagnostic is demoted from a gate to a recorded
  note** — it measures something real about the data but does not predict which operator form
  wins.
- **Leaderboard verdict on the form, across three competitions (updated 2026-07-31).**
  s3e19: `join_feature` 48.497 beats `ratio_target` 52.073 (1-year horizon). tps-sep-2022:
  `join_feature` 23.390 beats `ratio_target` 24.091 (1-year horizon). s5e1: `ratio_target`
  0.12417 beats `join_feature` 0.15626 (**3-year horizon**). All six external arms beat their
  no-external baselines significantly — the injection result itself replicates at n=3. But the
  form verdict SPLIT, and every local signal tested is now 0-for-3 at predicting it: CV picked
  ratio/baseline/featurejoin on the three comps while the LB said featurejoin/featurejoin/ratio,
  and the held-out-year protocol was wrong on all three probes. Consequence: **race both arms,
  always; no local signal or diagnostic currently earns the right to choose for you.**
  Post-hoc observation, NOT validated: the two 1-year comps went to `join_feature`, the one
  3-year comp to `ratio_target` — horizon length may be the real selector. Treat this exactly
  as sceptically as the proportionality story that just died: it is now formally registered as
  H-HORIZON in `docs/preregistrations/horizon_length_form_selector.md`, whose protocol is
  BINDING on the next firing-class competition — copy the prediction into that competition's
  dossier and commit it before any model is fitted or any leaderboard consulted.

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
