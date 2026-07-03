# STATUS — playground-series-s3e14 (Wild Blueberry Yield)

- **Task**: regression, predict `yield`; metric **MAE** (minimize); id col `id`
- **Data**: 15,289 train / 10,194 test; 16 raw features (5 pollinator/clonesize env vars, 6 collinear
  temperature-range columns, RainingDays/AverageRainingDays, fruitset/fruitmass/seeds); no missing values,
  7 exact duplicate rows in train (harmless)
- **Rules**: no external data, no pretrained models, no internet; daily submission limit 5

## Progress
- [x] Stage 0 Setup — config.yaml, data already present (baseline-only acceptance test ran earlier)
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py`
- [x] Stage 3 Modeling + CV — `scripts/train.py` (iteration 1), `scripts/train_v2.py` (iteration 2, self-improvement)
- [x] Stage 4 Evaluation — reflexion between iteration 1 and 2 (see below)
- [x] Stage 5 Submission generated — `submissions/sub_blend_v2_340.71180_20260703_193959.csv`
- [ ] Submitted to Kaggle leaderboard — **not submitted** (no Kaggle credentials in this environment; per task
      instructions this run does not submit)

## Current best score (OOF, no LB available)
| Experiment | Model | OOF MAE |
|------------|-------|---------|
| #1 (prior baseline) | generic LGB+XGB+CAT blend | 341.40782 |
| #2 (iteration 1) | feature-engineered blend + snap-to-grid | 340.95856 |
| #3 (iteration 2, **best**) | pruned features + native-cat CatBoost + snap-to-grid | **340.71180** |

Improvement over baseline: 341.40782 − 340.71180 = 0.69602 (~0.20% relative).

## EDA key findings (`scripts/eda.py`)
1. Target `yield` is continuous, roughly symmetric (skew ≈ −0.16), range [1945.5, 8969.4]. Only 776 of
   15,289 rows have distinct yield values, but this comes from shared combinations of low-cardinality env
   vars × continuous fruit measures — **not** a coarse rounding grid like s3e16's integer Age. Rounding to
   integers is meaningless here; the only discrete-grid idea worth testing is snapping predictions to the
   nearest *observed train yield value*, which was tested empirically (see below).
2. `fruitset`, `fruitmass`, `seeds` are the dominant predictive block (r = 0.83–0.89 with yield) — strong
   candidates for interaction/product terms.
3. `clonesize`, `RainingDays`/`AverageRainingDays` are low-cardinality (6–8 distinct values) and negatively
   correlated with yield (r = −0.38 to −0.48).
4. The 6 Upper/Lower TRange columns are near-perfectly collinear with each other (pairwise r ≥ 0.999) and
   individually weak (|r| ≈ 0.02) — redundant, condensed into `temp_spread`/`temp_avg` (later found
   near-zero importance and dropped in iteration 2).
5. Pollinator columns (`honeybee`, `bumbles`, `andrena`, `osmia`) are individually weak (|r| 0.07–0.20);
   combined into a `total_pollinators` index and `honeybee_share`.
6. No missing values; train/test feature means differ by <1% on every column → no covariate shift → plain
   5-fold KFold (shuffle) is an appropriate validation scheme (no stratification needed, unlike s3e16's
   discrete Age target).

## CV scheme
- **5-fold KFold**, shuffle, seed=42 (i.i.d. target, no groups/time structure, no train/test shift).

## Feature engineering
- Iteration 1 (`scripts/features.py`, 27 features = 16 raw + 11 engineered): fruit-biology interactions
  (`fruitset_x_seeds`, `fruitset_x_fruitmass`, `fruitmass_x_seeds`, `fruit_triple`, `seeds_per_fruitset`),
  condensed temperature (`temp_spread`, `temp_avg`), pollinator index (`total_pollinators`,
  `honeybee_share`), `rain_intensity`, `log_clonesize`.
- Iteration 2 (`scripts/train_v2.py`, 21 features): a quick LGB feature-importance probe showed
  `temp_avg`, `log_clonesize`, `rain_intensity`, `MinOfLowerTRange`, `AverageOfLowerTRange`,
  `AverageOfUpperTRange` had near-zero gain importance → dropped as noise.

## Model results (OOF MAE, 5-fold KFold)
| Experiment | LGB | XGB | CAT | Blend method | Blend OOF | Post-processing | Final OOF |
|------------|-----|-----|-----|--------------|-----------|------------------|-----------|
| #2 (iter 1) | 342.10905 | 342.53852 | 344.07199 | grid simplex (0.5/0.25/0.25) | 341.02061 | snap-to-grid | **340.95856** |
| #3 (iter 2) | 342.02154 | 342.21778 | 343.60196 (native-cat) | grid simplex (0.45/0.25/0.3), beat Ridge stack (344.04375) | 340.75961 | snap-to-grid | **340.71180** |

Experiment log: `experiments.json` (#1 baseline, #2, #3), all logged via `log_experiment_v2`.

## Self-improvement notes (Reflexion)
- Iteration 1 → 2 delta was small and expected (−0.247): pruning noise features and giving CatBoost native
  categorical handling of the low-cardinality env columns both nudged OOF down slightly; neither was a
  dramatic gain because `fruitset`/`fruitmass`/`seeds` already dominate the signal.
- Ridge (non-negative) stacking on OOF predictions was tried as an alternative to the grid-searched simplex
  blend in iteration 2 and **underperformed** (344.04375 vs 340.75961) — kept the simplex blend. With only
  3 base models, a constrained grid search over the simplex is hard to beat with a learned meta-model on
  this little data per model.
- Snap-to-grid post-processing (snap predictions to the nearest observed train yield value) won in both
  iterations by a small but consistent margin (~0.05), confirming the EDA hypothesis was worth testing even
  though the target has no coarse integer grid.
- Stopped after 2 iterations (within the 1–2 iteration budget for this run): improvement is decelerating and
  the dominant signal (fruit-biology block) has already been exploited via interactions.

## Next ideas (not yet tried)
- Optuna tuning of LGB/XGB/CatBoost hyperparameters (time budget in this run went to feature/blend
  experiments instead).
- Target transform (log or Box-Cox) given the wide yield range, then evaluate MAE in original units.
- Quantile regression heads or per-clonesize-bucket separate models (clonesize is a strong discrete driver).

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e14/scripts/eda.py
uv run python3 competitions/playground-series-s3e14/scripts/train.py     # iteration 1
uv run python3 competitions/playground-series-s3e14/scripts/train_v2.py  # iteration 2 (best)
```
No Kaggle submission was made (no credentials in this environment).
