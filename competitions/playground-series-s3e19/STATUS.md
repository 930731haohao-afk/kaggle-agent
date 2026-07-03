# STATUS — playground-series-s3e19 (Forecast Mini-course Sales)

- **Task**: regression, predict daily `num_sold` per (date × country × store × product); metric **SMAPE** (minimize); id col `id`
- **Data**: 136,950 train rows (2017-01-01 → 2021-12-31, 1,826 days) / 27,375 test rows (all of 2022, 365 days); 75 series (5 countries × 3 stores × 5 products), every series complete daily, no missing, no dups
- **Rules**: no external data, no pretrained, no internet; daily submission limit 5

## Progress
- [x] Stage 0 Setup — config.yaml, data present (from batch prep)
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py` (20 features)
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost on log1p target, TimeSeriesSplit)
- [x] Stage 4 Reflexion diagnostic — `scripts/diagnostic_kfold.py` (CV-scheme gap analysis)
- [x] Stage 5 Submission generated — `submissions/sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv`
- [ ] Submitted to Kaggle leaderboard — **not submitted** (unattended batch run, no credentials used)

## Current best (local CV only, no LB yet)
| Experiment | CV scheme | Blend SMAPE |
|-|-|-|
| #1 generic baseline | random 5-fold | 5.31891 |
| #2 engineered | TimeSeriesSplit 5-fold | 10.17540 |
| #3 diagnostic (same features as #2) | random 5-fold | 4.28142 |
| #7 Phase B final (6-way + Optuna-tuned LGB) | TimeSeriesSplit 5-fold | **10.01946** |

**Apples-to-apples read**: under the same random-KFold scheme as the baseline, the engineered
pipeline improves SMAPE 5.31891 → 4.28142 (−19.5%). The 10.17540 number is not a regression —
it is a *different, more honest yardstick*: TimeSeriesSplit forces each fold to extrapolate into
strictly-later dates, mirroring the real train(2017–21) → test(2022) gap. Random KFold interleaves
validation days inside seen seasonal cycles and is therefore optimistic for this task. Expect the
public LB to land between the two numbers, likely closer to the time-based one.

## EDA key findings (`scripts/eda.py`)
1. Test period (2022) is entirely **after** train (2017–2021), zero overlap → time-based validation is the honest scheme; documented decision to use `TimeSeriesSplit(n_splits=5)` over unique dates.
2. 75 complete daily series; store & product shares of yearly totals are **rock-stable** across years (Kagglazon ≈ 0.691 every year; products within ±0.005) → strong multiplicative structure, categoricals carry it.
3. Country shares **drift** (Argentina 0.0999 → 0.0663; Canada 0.3052 → 0.3244) → country × time interaction matters; hardest part of extrapolation.
4. Strong seasonality: weekend uplift (Sun mean 188.4 vs Mon–Thu ≈ 157), Dec/Jan peak (Dec 183.5, Jan 175.7 vs Apr 156.9); Jan-1 spike (216.7 vs 165.5 overall mean).
5. Target right-skewed (skew 1.747) but log1p nearly symmetric (−0.224) → train on log1p, invert with expm1.
6. Yearly totals non-monotonic (2020 COVID-ish dip, 2021 rebound) → no naive linear trend feature.

## Feature engineering (20 features, `scripts/features.py`)
Calendar: year, month, day, dow, day_of_year, weekofyear, quarter.
Flags: is_weekend, is_month_start, is_month_end, is_new_year.
Cyclical: sin/cos of month, dow, day_of_year.
Categoricals: country/store/product label-encoded, fed natively as categoricals to all 3 models.
Target transform: log1p(num_sold) → expm1 at prediction time.

## Model results
### Experiment #2 — TimeSeriesSplit 5-fold (expanding window on unique dates)
| Model | OOF SMAPE |
|-------|-----------|
| LightGBM | 10.17758 |
| XGBoost | 10.46000 |
| CatBoost | 10.73064 |
| **Blend 0.9/0.0/0.1 (OOF grid search)** | **10.17540** |

Fold 1 (only 2017 data to train on) is worst for all models (LGB 12.43); later folds 8–11. XGB weight searched to 0 again — LGB dominates, consistent with the baseline (#1) and prior batch competitions.

### Experiment #3 — diagnostic, same features, shuffled KFold
| Model | OOF SMAPE |
|-------|-----------|
| LightGBM | 4.31618 |
| XGBoost | 4.33201 |
| CatBoost | 4.32390 |
| **Blend 0.4/0.2/0.4** | **4.28142** |

Under interpolation all three models are near-identical and blending helps; under extrapolation LGB clearly leads. Kept the **time-based weights (0.9/0.0/0.1)** for the submission because the test set is a pure future-year extrapolation.

## Reflexion note (why #2 "looked worse" than baseline)
Score went from 5.32 → 10.18 which initially reads as a failure. Hypothesis tested by #3: the jump is caused by the CV scheme, not the features/models. Confirmed — identical features/models under the baseline's random-KFold scheme score 4.28 (< 5.32). Lesson reinforced: **never compare scores across CV schemes**; log the scheme with every experiment (v2 schema `cv.strategy` field does this).

## Submission
- Current best file: `submissions/sub_6way_optuna_blend_10.01946_20260704_002039.csv` (Phase B round 4)
- Method: LGB seeds 42/2024/7 + CAT seeds 42/2024 + Optuna-tuned LGB, all retrained on
  100% of train (log1p target), blended 0/0.1/0.4/0/0/0.5, clipped at 0
- Prediction sanity: mean 178.2 / max 1432.9 vs train mean 165.5 / max 1380 — plausible for a growing 2022
- Previous submission (pre-Phase B): `submissions/sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv`
- Not submitted to Kaggle (no credentials in this unattended run)

## Phase B self-improvement iteration (2026-07-04)

**Baseline going in**: 10.17540 (TimeSeriesSplit, exp #2, LGB 0.9/XGB 0/CAT 0.1).

### Round 1 — ratio decomposition as standalone member (exp #4) — REJECTED
Implemented the highest-EV untried idea: `total(date) x country_share(country,year,
linear-trend-extrapolated, renormalized) x combo_share_within_country(country,store,
product, pooled)`. Pre-check motivating this: additive OLS of log1p(num_sold) on
country+store+product alone (**no date at all**) explains **R²=0.974** of target
variance — strong evidence for the multiplicative structure.
Despite that, the standalone RD model scored OOF SMAPE **14.75038** (TimeSeriesSplit),
far worse than LGB's 10.17758 — weight search gave it **0.0** weight in a 3-way blend
with LGB+CAT (XGB dropped, see below). Blend only nudged to 10.17489 via a finer
weight-grid (0.05 vs 0.1 step) on LGB/CAT, **not** from RD.

Follow-up oracle diagnostics (fed the ACTUAL daily total, isolating the share-model
error) showed the share decomposition alone gets SMAPE 10.04 (flat country share) to
10.84 (trend-extrapolated country share) — i.e. **roughly on par with GBDT**, not
better. Swapping the harmonic-linear total-forecast for a small LightGBM on the
aggregated daily series didn't help either (daily-total OOF SMAPE ~9.6-10%, R²=0.41
in log space) — the 5-year, COVID-disrupted grand-total series (2017 4.45M → 2018
4.72M → 2019 4.52M → **2020 4.09M dip** → **2021 4.88M rebound**) has no reliable
extrapolable trend signal with only 4-5 yearly points. **Conclusion: unlike s3e20,
the structural/ratio signal here does NOT beat GBDT** — GBDT already captures the
categorical structure natively (native cat_features / label-encoding splits are
already ~as good as the pooled share ratios) and the real bottleneck (grand-total
1-year-ahead extrapolation under a volatile short series) affects any method equally.
Also notable: **linear trend extrapolation of the drifting country share is worse
than a flat (last-value) share** (10.84 vs 10.04 SMAPE with oracle total) — a
counter-intuitive but real result given how few, noisy yearly points are available.

Also dropped **XGBoost** from the pool per instruction (weight-searched to 0 in both
exp #2 and #3 already).

### Round 2 — seed bagging (exp #5) — IMPROVED
Pivoted to a cheap, previously-validated cross-competition pattern (see
`knowledge/experience.md`): add a second LGB seed (2024) alongside the original
LGB (seed 42) and CAT (seed 42), 3-way weight search.
**Result: 10.17489 → 10.16605** (weights LGB_s42 0.65 / LGB_s2024 0.30 / CAT 0.05).
This is also better than the original submitted best (10.17540). New submission:
`submissions/sub_lgb_seedbag_cat_blend_10.16605_20260704_000957.csv`.

### Round 3 — extended seed bagging (exp #6) — IMPROVED
Added a third LGB seed (7) and a second CAT seed (2024): 5-way weight search over
LGB(s42,s2024,s7) + CAT(s42,s2024).
**Result: 10.16605 → 10.15721** (weights LGB_s42 0.4 / LGB_s2024 0.2 / LGB_s7 0.3 /
CAT_s42 0.1 / CAT_s2024 0.0). Diminishing but real seed-bagging gains, consistent
with the experience.md pattern. Submission:
`submissions/sub_lgb3seed_cat2seed_blend_10.15721_20260704_001501.csv`.

### Round 4 — Optuna LGB tuning, fold-5 proxy (exp #7) — IMPROVED (largest gain)
Optuna TPE, 40 trials in 42.0s, objective = SMAPE on **fold 5 only** — chosen over
the usual fold-0 proxy because TimeSeriesSplit folds are not interchangeable and
fold 5 has the largest training window / regime closest to the real
train(2017–21)→test(2022) gap. Best params: lr 0.0371, num_leaves 20,
min_child_samples 38, subsample 0.61, colsample_bytree 0.90, reg_alpha 0.001,
reg_lambda 0.43 — shallower/simpler than the hand-set config (num_leaves 63),
consistent with "extrapolation rewards regularization".
Tuned LGB re-validated on the full 5-fold OOF: solo **10.14833** (new best single
model, vs LGB_s42 10.17758). Added as a NEW member (not a replacement) to the
round-3 pool; 6-way weight search → weights {LGB_s42 0, LGB_s2024 0.1, LGB_s7 0.4,
CAT_s42 0, CAT_s2024 0, **LGB_tuned 0.5**}.
**Result: 10.15721 → 10.01946.** Submission:
`submissions/sub_6way_optuna_blend_10.01946_20260704_002039.csv`.

**Honest caveat**: fold 5 is both the Optuna objective and 1/5 of the OOF used for
the blend weight search, so LGB_tuned's stellar fold-5 score (8.13 vs its fold-4
11.90) is partially selected-on, and 10.01946 is likely somewhat optimistic
relative to a fully nested protocol. Directionally the gain is real (LGB_tuned
also wins or ties folds 2–3, which were not tuned on: 8.00/10.13 vs LGB_s42
8.05/10.17), but the true expected 2022 SMAPE is probably between ~10.05 and
~10.15.

### Phase B summary
| Round | Change | TimeSeriesSplit blend SMAPE |
|-|-|-|
| — | baseline (exp #2, submitted config) | 10.17540 |
| 1 | + ratio-decomposition member (exp #4) | 10.17489 (RD weight 0 → rejected) |
| 2 | seed bagging: + LGB s2024 (exp #5) | 10.16605 |
| 3 | + LGB s7, + CAT s2024 (exp #6) | 10.15721 |
| 4 | + Optuna fold-5-proxy tuned LGB (exp #7) | **10.01946** |

Net improvement: 10.17540 → 10.01946 (−0.15594 SMAPE, −1.53% relative), all under
the identical TimeSeriesSplit 5-fold scheme (same folds, same seed 42). Stopped at
4 rounds (protocol max). Current best submission:
`submissions/sub_6way_optuna_blend_10.01946_20260704_002039.csv` (not submitted to
Kaggle — unattended run, no credentials).

## Potential improvements (untried / deprioritized after Phase B)
- Ratio-decomposition (tried, rejected — see Round 1 above). Do not retry without a
  materially better total-forecasting method for the aggregate series.
- Nested fold-proxy validation for the tuned LGB (train fold-5-tuned config, but
  weight-search on folds 1–4 only) to firm up the 10.019 estimate.
- Per-country GDP-style regressors or explicit country-share extrapolation — likely
  low value given Round 1's finding that trend-extrapolated shares underperform flat.
- Fourier terms with multiple harmonics; holiday calendars per country (external data
  disallowed → build from date patterns only).
- Seed-bag the tuned LGB (experience.md warns this may be ineffective for
  directly-tuned configs — s3e5 counterexample — but cheap to test).

## Files
```
competitions/playground-series-s3e19/
├── config.yaml
├── STATUS.md
├── experiments.json            # 7 experiments (v2 schema for #2-#7)
├── data/                       # train/test/sample + *_processed.csv
├── scripts/
│   ├── eda.py
│   ├── features.py
│   ├── train.py                # main pipeline (time-based CV + submission)
│   ├── diagnostic_kfold.py     # reflexion diagnostic (no submission)
│   ├── train_ratio.py          # Phase B round 1: ratio decomposition (rejected)
│   ├── train_seedbag.py        # Phase B round 2: LGB seed bagging
│   ├── train_seedbag2.py       # Phase B round 3: extended seed bagging
│   └── train_optuna.py         # Phase B round 4: Optuna fold-5-proxy tuned LGB
└── submissions/
    ├── sub_generic_5.31891_20260703_121406.csv
    ├── sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv
    ├── sub_lgb_cat_rd_blend_10.17489_20260704_000355.csv
    ├── sub_lgb_seedbag_cat_blend_10.16605_20260704_000957.csv
    ├── sub_lgb3seed_cat2seed_blend_10.15721_20260704_001501.csv
    └── sub_6way_optuna_blend_10.01946_20260704_002039.csv   # current best
```

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e19/scripts/eda.py
uv run python3 competitions/playground-series-s3e19/scripts/features.py
uv run python3 competitions/playground-series-s3e19/scripts/train.py
uv run python3 competitions/playground-series-s3e19/scripts/diagnostic_kfold.py  # optional diagnostic
```
