# STATUS — playground-series-s3e7 (Hotel Reservation Cancellation)

- **Task**: binary classification, predict `booking_status` (1 = cancelled); metric **ROC-AUC** (maximize); id col `id`
- **Data**: 42,100 train / 28,068 test; 17 raw features, all already integer/float-encoded
  (meal plan, room type, market segment are pre-label-encoded); no missing values, no
  duplicate rows/ids, no train/test id overlap
- **Rules**: no external data, no pretrained models, no internet; daily submission limit 5
- **Class balance**: 60.8% not-cancelled / 39.2% cancelled (mild imbalance) → StratifiedKFold

## Progress
- [x] Stage 0 Setup — config.yaml, data present; `experiments.json` seeded with 1
      generic-baseline entry (blend AUC 0.89882) to beat
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py` (two variants, see below)
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost, 5-fold StratifiedKFold,
      OOF weight-searched blend)
- [x] Stage 4 Evaluation/iteration — 1 reflexion-driven iteration (feature trimming)
- [x] Stage 5 Submission generated — `submissions/sub_blend_0.89939_20260703_190223.csv`
- [x] **Phase B self-improvement iteration (2026-07-03 evening)** — 3 rounds, new best
      OOF 0.899893 (exp 6); final submission `submissions/sub_blend_0.89989_20260703_205132.csv`
- [ ] Not submitted to Kaggle leaderboard (no credentials in this run; local CV only)

## Best result vs baseline
| | OOF ROC-AUC | vs baseline (0.89882) |
|-|-------------|------------------------|
| Generic baseline blend (exp 1, raw 17 features) | 0.89882 | — |
| Iter1: 17 raw + 14 engineered features (exp 2) | 0.89788 | −0.00094 |
| Iter2: 17 raw + 8 trimmed engineered features (exp 3) | 0.89939 | +0.00057 |
| Phase B R1: Optuna-tuned LGB + hand-set XGB/CAT (exp 4) | 0.899891 | +0.00107 |
| Phase B R2: + Optuna-tuned XGB (exp 5, REJECTED) | 0.899722 | +0.00090 |
| **Phase B R3: exp-4 bases, 0.01-grid prob blend (exp 6, FINAL)** | **0.899893** | **+0.00107** |

CV is local-only this run (no Kaggle submission), so no CV↔LB gap to report.

## Phase B self-improvement iteration (3 rounds, same CV: 5-fold StratifiedKFold seed=42)
- **Round 1 (exp 4, KEPT)**: Optuna-tuned LGB (50 trials, TPE, fold-0 proxy objective —
  a first attempt tuning on full 5-fold per trial blew a 25-min timeout with zero results;
  fold-0 tuning took 300s). Winner is shallow + regularized: max_depth=3, lr=0.068,
  reg_alpha=2.14, colsample=0.53 vs hand-set depth-unlimited num_leaves=63/lr=0.03.
  LGB solo 0.898824 → 0.899215; blend 0.899395 → **0.899891** (+0.000496).
- **Round 2 (exp 5, REJECTED)**: same recipe applied to XGB (50 trials, fold-0 proxy).
  XGB solo improved 0.898765 → 0.898860, but blend REGRESSED to 0.899722 (−0.000169):
  tuned XGB converged to shallow trees (depth 4, colsample 0.5) similar to tuned LGB,
  losing ensemble diversity. Kept hand-set XGB.
- **Round 3 (exp 6, FINAL)**: ensemble-stage-only change on exp-4 base models.
  Fine 0.01 weight grid 0.899893 (weights LGB 0.54/XGB 0.40/CAT 0.06); rank-average
  blend 0.899884 (worse). Delta +0.000002 vs exp 4 — noise-level. Counted together
  with Round 2 as two consecutive rounds without meaningful improvement → STOP.
- Scripts: `scripts/optuna_lgb.py` (R1), `scripts/optuna_xgb.py` (R2),
  `scripts/blend_refine.py` (R3). Total Phase B wall time ≈ 45 min (25 of which was
  the aborted full-5-fold tuning attempt).

## EDA key findings
1. `lead_time` is the single strongest predictor (single-feature AUC 0.73, |corr| 0.375):
   cancellation rate rises monotonically from 11% (≤7 days) to 68% (>180 days lead time).
2. `no_of_special_requests`, `repeated_guest`, `required_car_parking_space` are strongly
   *protective*: repeated guests cancel only 0.9% of the time vs 40% for new guests.
3. `market_segment_type` matters a lot: segment 1 (likely "Online") cancels 50.5% of the
   time vs 1.6% for segment 4.
4. `avg_price_per_room` has a non-monotonic relationship with target (rises then dips in
   the top price quintile) — kept raw, not binned.
5. No missing values, no duplicate rows/ids, no train/test id overlap, no train/test
   distribution shift (all normalized shifts < 0.02), no leakage (all single-feature
   AUCs < 0.85, max was `lead_time` at 0.73 — a legitimately strong, non-leaky signal).
6. `arrival_date`/`arrival_month` have near-zero raw correlation with target (0.003, 0.008).

## CV scheme (fixed first)
- **5-fold StratifiedKFold on `booking_status` directly** (binary target, no binning
  needed), shuffle, seed=42. Fold AUCs stable across all 3 models (0.893–0.901 range).

## Feature engineering — two iterations
**Iter1 (14 engineered, on top of 17 raw)**: total_nights, total_guests, has_children,
weekend_ratio, price_per_person, price_per_night, lead_time_log, prior_cancel_rate,
has_prior_history, month_sin/cos, date_sin/cos, no_special_and_price. **Result: OOF 0.89788,
below the 0.89882 generic baseline** (delta −0.00094) — a flat/slightly-negative signal.

**Iter2 (reflexion)**: Hypothesis — the cyclical month/date encodings and the
special_requests×price interaction were adding noise rather than signal, since their
underlying raw columns (`arrival_date`, `arrival_month`) have near-zero raw correlation
with the target and GBMs can already learn interactions/splits natively. Trimmed to 8
engineered features: total_nights, total_guests, has_children, weekend_ratio,
price_per_person, lead_time_log, prior_cancel_rate, has_prior_history. **Result: OOF
0.89939, now +0.00057 over baseline** — hypothesis confirmed, kept as final.

Per the self-improvement decision framework, +0.00057 is a small-but-real "flat→positive"
signal (not a plateau — only 2 iterations run); stopped after iter2 given the 15-minute
training budget and diminishing expected returns from a 3rd micro-tweak.

## Model results (OOF AUC, 5-fold, Phase A iter2 feature set — 25 features)
| Model | OOF AUC | time |
|-------|---------|------|
| LightGBM (binary, num_leaves=63, lr=0.03) | 0.89882 | 30.3s |
| XGBoost (binary:logistic, depth=6, lr=0.03) | 0.89876 | 31.2s |
| CatBoost (Logloss/AUC, depth=7, lr=0.03) | 0.89671 | 47.1s |
| Blend 0.5/0.4/0.1 (weight-searched, grid step 0.05) | 0.89939 | — |

Total training time (all 3 models, iter2 run): ~109s. Well within the 15-minute budget
across both iterations combined (~3m37s wall time for train.py × 2 runs).

Experiment log: `experiments.json` (#1 generic baseline, #2 iter1, #3 iter2,
#4 Phase B R1/tuned-LGB, #5 Phase B R2/rejected, #6 Phase B R3/final).

## Model results (OOF AUC, 5-fold, Phase B final — exp 6)
| Model | OOF AUC | notes |
|-------|---------|-------|
| LightGBM (Optuna-tuned: depth=3, lr=0.068, reg_alpha=2.14) | 0.899215 | strongest single |
| XGBoost (hand-set: depth=6, lr=0.03 — kept for diversity) | 0.898765 | |
| CatBoost (hand-set: depth=7, lr=0.03) | 0.896709 | |
| **Blend 0.54/0.40/0.06 (prob, 0.01 grid)** | **0.899893** | FINAL |

## Next ideas (not tried — future work)
- Target/frequency encoding for `market_segment_type` given its large rate spread (1.6–50.5%).
- CatBoost native categorical handling for the 3 label-encoded categorical cols
  (meal_plan 4 lv / room_type 7 lv / market_segment 5 lv) — CAT is the weakest base
  model (0.8967) and only carries 0.06 blend weight, so expected blend gain is small.
- Optuna-tune CAT (same fold-0 proxy recipe) — but see the Round-2 diversity caveat.
- Threshold-free ranking metric (AUC) means no post-processing/threshold tuning applies.

## Files
```
competitions/playground-series-s3e7/
├── config.yaml
├── experiments.json        # 6 entries (see above)
├── STATUS.md                # this file
├── data/                    # train.csv, test.csv, sample_submission.csv
├── scripts/
│   ├── eda.py
│   ├── features.py          # build_features() + feature_columns(variant="trimmed"|"full")
│   ├── train.py              # LGB/XGB/CAT, 5-fold CV, weight-search blend, logs experiment
│   ├── optuna_lgb.py          # Phase B R1: Optuna LGB (fold-0 proxy) + blend → exp 4
│   ├── optuna_xgb.py          # Phase B R2: Optuna XGB (rejected) → exp 5
│   └── blend_refine.py        # Phase B R3: fine-grid + rank blend → exp 6 (final)
└── submissions/
    ├── sub_generic_0.89882_20260703_120542.csv   # pre-existing baseline
    ├── sub_blend_0.89788_20260703_185938.csv     # iter1 (kept for audit trail)
    ├── sub_blend_0.89939_20260703_190223.csv     # iter2 (Phase A best)
    ├── sub_blend_0.89989_20260703_203932.csv     # Phase B R1 (exp 4)
    └── sub_blend_0.89989_20260703_205132.csv     # Phase B R3 (exp 6) — FINAL best local CV
```

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e7/scripts/eda.py
uv run python3 competitions/playground-series-s3e7/scripts/train.py        # Phase A pipeline
uv run python3 competitions/playground-series-s3e7/scripts/optuna_lgb.py    # Phase B R1
uv run python3 competitions/playground-series-s3e7/scripts/blend_refine.py  # Phase B R3 (final)
# Not submitted to Kaggle in this run (no credentials available). To submit:
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])")
uv run kaggle competitions submit -c playground-series-s3e7 \
  -f competitions/playground-series-s3e7/submissions/sub_blend_0.89989_20260703_205132.csv \
  -m "Optuna-tuned LGB + XGB + CAT blend (0.54/0.40/0.06), OOF 0.899893"
```
