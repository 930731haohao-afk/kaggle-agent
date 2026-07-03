# STATUS — playground-series-s3e1 (California Housing)

- **Task**: regression, predict `MedHouseVal`; metric **RMSE** (minimize); id col `id`
- **Data**: 37,137 train / 24,759 test; 8 raw numeric features (MedInc, HouseAge, AveRooms,
  AveBedrms, Population, AveOccup, Latitude, Longitude); no missing values, no duplicate rows
- **Rules**: no external data, no pretrained models, no internet; daily submission limit 5

## Progress
- [x] Stage 0 Setup — config.yaml, data present
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py`
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost, RMSE objective, weight-searched blend)
- [x] Stage 3b Iteration probe — `scripts/tune_geo_te.py` (geo target-encoding, not adopted)
- [x] Stage 5 Submission generated — `submissions/sub_lgb_xgb_cat_blend_0.55877_20260703_182947.csv`
- [x] Stage 6 Self-improvement iteration (Phase B, 4 rounds) — new best submission
      `submissions/sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv`
- [ ] Submitted to Kaggle leaderboard — not performed this run (no Kaggle credentials touched; run is offline/unattended)

## Current Best Score
| | OOF RMSE |
|-|----------|
| Baseline (exp 1, generic 8-feature blend) | 0.56166 |
| Blend (exp 2, engineered features) | 0.55877 |
| **5-way blend (exp 7: +Optuna-tuned LGB, +seed-bag, +KNN/coastal geo feats)** | **0.55709** |

Improvement over exp-2 blend: 0.55877 − 0.55709 = 0.00168 (≈0.30% further relative RMSE reduction).
Improvement over exp-1 baseline: 0.56166 − 0.55709 = 0.00457 (≈0.81% relative reduction).
No public/private LB available — submission was not made to Kaggle this run.

## EDA key findings
1. Target `MedHouseVal` is right-skewed (skew 0.97) and **top-coded at 5.00001** for 4.92% of
   train rows — a known California-Housing artifact (true value ≥5.00001 gets capped). This
   caps achievable RMSE regardless of modeling; noted as an inherent data limitation, not fixed.
2. `HouseAge` is also top-coded at 52 for 5.57% of rows.
3. `MedInc` is by far the strongest linear predictor (Pearson r=0.70); `AveRooms` next (r=0.37).
   Individually, `Latitude` (r=-0.116) and `Longitude` (r=-0.057) look weak, but a quick 5-fold
   LGB on raw features ranked **Longitude and Latitude as the top-2 importance features** —
   geo signal is non-linear (location-based, not distance-from-origin-based).
4. `Latitude`/`Longitude` are strongly anti-correlated with each other (r=-0.937), as expected
   for California's NW–SE coastline geography.
5. Outlier features: `AveRooms` max 28.8 (train)/56.3 (test), `AveOccup` max 503 (train)/230
   (test), `AveBedrms` max 5.9 (train)/10.5 (test) — long right tails driven by a handful of
   very low-occupancy blocks.
6. No missing values in train or test; no duplicate rows; no train/test feature-identical
   overlap.
7. Train vs. test distributions closely match (largest mean difference: `Population` at 1.12%,
   all others <0.5%) — no covariate shift, so no reweighting/adversarial-validation needed.
8. Quick sanity 5-fold LGB on raw 8 features alone already reached OOF RMSE 0.56693, close to
   the exp-1 blend (0.56166), confirming the dataset has good baseline signal and this run's
   gains come primarily from geo/ratio feature engineering rather than model choice.

## CV scheme (fixed first — matters more than the model)
- **Standard 5-fold KFold** (shuffle=True, seed=42), no stratification/grouping.
- Justification: target is continuous with no natural strata, no group or time structure in
  the data (each row is an independent census block), and train/test distributions are
  near-identical (see EDA finding 7) — plain KFold is the textbook-correct choice here.
- Fold RMSE range for the winning blend components was ~0.545–0.588 (naturally noisier than
  a large-N classification task at this scale, but no fold is a systematic outlier).

## Feature engineering (24 features total: 8 raw + 16 engineered)
- **Ratios**: `households` (= Population/AveOccup), `bedroom_ratio` (AveBedrms/AveRooms),
  `rooms_per_person` (AveRooms/AveOccup) — normalizes by implied household count.
- **Log1p transforms**: AveRooms, AveBedrms, Population, AveOccup, households — tames the
  heavy right-tail outliers found in EDA.
- **Geo**: Euclidean distance to 5 major CA population centers (LA, SF, San Diego, Sacramento,
  San Jose) + `dist_nearest_city`; `lat_long` interaction term; `geo_cluster` (KMeans, k=25,
  fit on train coordinates only, applied to test).
- New-feature correlation with target: `rooms_per_person` was the strongest new signal
  (r=0.452), followed by `dist_nearest_city` (r=-0.344) and `log_AveRooms` (r=0.342).
- Validated: no NaNs introduced, all features numeric, no train/test column mismatch, no
  suspiciously-perfect (leakage-level) target correlation.

## Model results (OOF RMSE, 5-fold KFold, engineered features)
| Model | OOF RMSE | Notes |
|-------|----------|-------|
| LightGBM (RMSE objective) | 0.56109 | n_estimators=2000 w/ early stopping, lr=0.03 |
| XGBoost (reg:squarederror) | 0.56297 | max_depth=7, lr=0.03 |
| CatBoost (RMSE loss) | 0.56188 | depth=8, lr=0.03 |
| **Blend (weight-searched, grid step 0.05)** | **0.55877** | weights LGB=0.45 / XGB=0.15 / CAT=0.40 |

Total training wall time (3 models × 5 folds): 45.5s — well within the 15-minute budget.

## Iteration round 1 (self-improvement: reflexion on a surprising near-null result)
**Hypothesis**: since raw Lat/Long were the top-2 importance features in the EDA sanity model,
a KFold-safe smoothed target-encoding of a finer (50-cluster) geo grouping should beat the
coarse 25-cluster categorical id used in the main blend.

**Result**: LGB-only OOF RMSE 0.56102 vs. 0.56109 for the exp-2 LGB component — a -0.00007
delta, i.e. noise-level, not a real improvement.

**Decision**: reverted / not adopted. Trees already extract the non-linear geo signal directly
from raw Latitude/Longitude splits plus the distance-to-city features; an explicit target
encoding of a coarse spatial grouping adds no incremental information here. Kept exp 2's
feature set as final.

**Next steps considered but not pursued this run** (time-boxed; diminishing returns expected):
- Optuna tuning of LGB/CatBoost hyperparameters (main lever left is likely feature granularity,
  not hyperparameters, given how close the 3 base models already score).
- Quantile/Huber loss variants to reduce sensitivity to the top-coded target values.
- Finer geo clustering (k=50-100) used as a raw feature (not target-encoded) rather than k=25.

Experiment log: `experiments.json` (#1 baseline, #2 winning blend, #3 rejected TE probe).

## Phase B self-improvement iteration (4 rounds, all improved — stopped at protocol's 2–4 round upper bound)

| Round | exp # | Change | OOF RMSE | Delta vs prior |
|-------|-------|--------|----------|-----------------|
| 1 | 4 | Optuna TPE fold-0-proxy tuning of LGB (50 trials, 74.3s), added as 4th blend member (not replacing original LGB) | 0.557977 | -0.00079 vs exp 2 (0.558768) |
| 2 | 5 | Seed bagging: same tuned-LGB hyperparams, random_state 42→2024, added as 5th member | 0.557859 | -0.00012 vs round 1 |
| 3 | 6 | Probe: +knn_mean_dist_10 (KNN k=10 mean distance, combined train+test coords) +coastal_dist (distance to 12 CA coastline anchor points); LGB-only probe | 0.560444 (solo LGB) | -0.00065 vs exp-2 LGB (0.56109) — above 0.0005 noise threshold, adopted |
| 4 | 7 | Full retrain of the 5-member pool on the 26-feature set (round-3 features added) | **0.557088** | -0.00077 vs round 2 |

**Total improvement this iteration**: 0.558768 (exp 2) → 0.557088 (exp 7) = -0.00168 (≈0.30% relative).
Combined with the original feature-engineering win: 0.56166 (exp 1 baseline) → 0.557088 = -0.004572
(≈0.81% relative reduction overall).

**Reflexion — Round 3 vs. Round-1-era exp #3**: exp #3 showed that *target-encoding* a coarse
geo grouping adds nothing once trees already split on raw Lat/Long (-0.00007, noise). Round 3
tested a different lever — *non-target-encoded* geometric features (KNN local-density distance,
coastal distance) — and got a real gain (-0.00065). Lesson: "geo target-encoding is noise-level
here" does not generalize to "no more geo feature engineering is possible" — the two are
different mechanisms (grouped target leakage-safe averaging vs. raw distance geometry).

**Training time**: Optuna search 74.3s + full-CV revalidation 13.2s + 3-model recompute 45.2s
(round 1) + 9.7s (round 2 seed bag) + geo probe (~seconds) + 67.2s (round 4 full retrain) — total
well under the 30-minute budget.

New best submission: `submissions/sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv`.
Not submitted to Kaggle this run (no credentials touched, unattended run).

Experiment log additions: `experiments.json` #4–#7. Scripts: `scripts/tune_lgb_optuna.py` (round 1),
`scripts/seed_bag_round2.py` (round 2), `scripts/geo_knn_probe.py` (round 3 probe),
`scripts/round4_full_retrain.py` (round 4).

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e1/scripts/eda.py
uv run python3 competitions/playground-series-s3e1/scripts/features.py
uv run python3 competitions/playground-series-s3e1/scripts/train.py
# optional iteration probe (not adopted, informational only):
uv run python3 competitions/playground-series-s3e1/scripts/tune_geo_te.py
# Phase B iteration (current best pipeline):
uv run python3 competitions/playground-series-s3e1/scripts/tune_lgb_optuna.py
uv run python3 competitions/playground-series-s3e1/scripts/seed_bag_round2.py
uv run python3 competitions/playground-series-s3e1/scripts/geo_knn_probe.py
uv run python3 competitions/playground-series-s3e1/scripts/round4_full_retrain.py
```

No Kaggle submission was made this run (unattended weekend batch — credentials untouched by
instruction). To submit the generated file manually:
```bash
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e1 \
  -f competitions/playground-series-s3e1/submissions/sub_round4_geofeat_5way_blend_0.55709_20260703_212811.csv \
  -m "5-way blend (Optuna-tuned LGB + seed bag + KNN/coastal geo features), OOF 0.557088"
```
