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
| #2 engineered (submission) | TimeSeriesSplit 5-fold | 10.17540 |
| #3 diagnostic (same features as #2) | random 5-fold | **4.28142** |

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
- File: `submissions/sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv`
- Method: LGB+XGB+CAT retrained on 100% of train (log1p target), blended 0.9/0.0/0.1, clipped at 0
- Prediction sanity: mean 178.2 / max 1389.6 vs train mean 165.5 / max 1380 — plausible for a growing 2022
- Not submitted to Kaggle (no credentials in this unattended run)

## Potential improvements (untried)
- Ratio-decomposition approach: forecast total daily sales, then apply (near-constant) store/product shares and smoothed country-share trend — the share stability found in EDA suggests this could beat pure GBDT.
- Per-country GDP-style regressors or explicit country-share extrapolation for the drifting country mix.
- Fourier terms with multiple harmonics; holiday calendars per country (external data disallowed → build from date patterns only).
- Blend weight search on the *last* time fold only (closest to the 2022 regime).

## Files
```
competitions/playground-series-s3e19/
├── config.yaml
├── STATUS.md
├── experiments.json            # 3 experiments (v2 schema for #2, #3)
├── data/                       # train/test/sample + *_processed.csv
├── scripts/
│   ├── eda.py
│   ├── features.py
│   ├── train.py                # main pipeline (time-based CV + submission)
│   └── diagnostic_kfold.py     # reflexion diagnostic (no submission)
└── submissions/
    ├── sub_generic_5.31891_20260703_121406.csv
    └── sub_lgb_xgb_cat_blend_10.17540_20260703_194908.csv
```

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e19/scripts/eda.py
uv run python3 competitions/playground-series-s3e19/scripts/features.py
uv run python3 competitions/playground-series-s3e19/scripts/train.py
uv run python3 competitions/playground-series-s3e19/scripts/diagnostic_kfold.py  # optional diagnostic
```
