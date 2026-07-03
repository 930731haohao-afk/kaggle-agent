# STATUS — playground-series-s3e5 (Wine Quality, ordinal)

- **Task**: ordinal regression head, predict `quality` (3–8); metric **quadratic_weighted_kappa (QWK)**, maximize; id col `Id`
- **Data**: 2,056 train / 1,372 test; 11 raw physicochemical features, no missing values, no duplicates
- **Rules**: no external data, no pretrained models, no internet; daily submission limit 5 (not used — no credentials on this machine)

## Progress
- [x] Stage 0 Setup — config.yaml already present (generic-baseline entry #1 pre-existing)
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py` (11 → 21 features)
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost regression, 5-fold StratifiedKFold)
- [x] Self-improvement iteration (first pass) — optimized-rounder threshold tuning vs naive rounding
- [x] **Phase B self-improvement iteration (this run)** — `scripts/iterate.py`, 4 real rounds + 2 nested-cutpoint diagnostics
- [x] Stage 5 Submission generated — `submissions/sub_iterate_0.56769_20260703_231704.csv` (new best)
- [ ] Submitted to Kaggle leaderboard — not attempted (no Kaggle credentials configured in this environment; local CV only)

## Current best score
| | OOF QWK (post-rounder) | Public LB | Private LB |
|-|---------|-----------|------------|
| Generic baseline blend (exp #1) | 0.47871 | — | — |
| Regression blend, naive round (exp #2) | 0.47191 | — | — |
| Regression blend, optimized rounder (exp #3, prior best) | 0.52687 | — | — |
| + finer Dirichlet weight search (exp #4) | 0.52986 | — | — |
| + Optuna-tuned LGB added to pool (exp #5) | 0.56293 | — | — |
| + seed-bag LGB_tuned (exp #6, no gain) | 0.56293 | — | — |
| + Optuna-tuned CAT added to pool (exp #8) | **0.56769** | — | — |
| + seed-bag CAT_tuned (exp #9, no gain) | 0.56716 | — | — |
| + LGB multiclass EV-decode member (exp #11, no gain) | 0.56769 | — | — |

**New best: 0.56769** (exp #8). **Delta vs prior best 0.52687: +0.04082** (≈7.7% relative). **Delta vs
original generic baseline 0.47871: +0.08898** (≈18.6% relative). Not submitted — no Kaggle
credentials available in this session, so no Public/Private LB numbers exist yet.

## Phase B iteration (this run) — protocol and outcome
Ran `competitions/playground-series-s3e5/scripts/iterate.py` step by step, deciding keep/reject on
the **final post-rounder QWK only** (per knowledge/experience.md s3e16 lesson — never decide on raw
member OOF). Same CV throughout: 5-fold StratifiedKFold(shuffle=True, random_state=42) on `quality`.

1. **Round 0 (methodology, no new model)**: exp#3 used a coarse 0.1-step grid over the 3-model
   simplex for the blend weight search. Replaced with Dirichlet random search (800 draws) +
   coordinate-ascent refinement, scored on post-rounder QWK. Same 3 members, same folds.
   **0.52687 → 0.52986** (+0.00299). Logged as exp #4.
2. **Round 1**: Optuna (TPE sampler, 40 trials, 600s timeout, full 5-fold CV — data is tiny so the
   full CV is affordable, no fold-0 proxy needed) tuning LightGBM with the objective set to the
   **post-rounder QWK computed directly on that trial's own OOF vector** (not raw RMSE) — this
   dodges the s3e16 "discretization boundary" trap by optimizing the actual final metric. Best
   single-model LGB_tuned post-rounder QWK: 0.56244 (vs orig LGB 0.50547). Added as a 4th pool
   member (kept, not replaced, per s3e14 lesson); weight search moved ~98% of blend weight onto
   LGB_tuned. **0.52986 → 0.56293** (+0.03307). Logged as exp #5.
3. **Round 2 (seed-bag)**: added a second random_state=2024 copy of LGB_tuned to the pool; weight
   search gave it **zero** weight. **No improvement** (0.56293 = 0.56293). Logged as exp #6
   (non-improving round #1).
4. **Diagnostic — nested cutpoints** (exp #7, not a model change): compared the current practice
   (OptimizedRounder cutpoints fit on the full 5-fold OOF vector) against honest nested
   (leave-fold-out) cutpoint fitting on the exp#5/6 5-way blend. Full-OOF QWK 0.56293 vs nested
   QWK 0.54649 (gap +0.01644) — modest, addresses the overfit-risk flag from the prior run's
   STATUS.md.
5. **Round 3**: same Optuna full-CV direct-post-rounder-QWK recipe applied to CatBoost (the
   strongest baseline member). Best single-model CAT_tuned post-rounder QWK: 0.56466. Added as a
   6th pool member (kept, not replaced); weight search moved ~95% weight onto CAT_tuned.
   **0.56293 → 0.56769** (+0.00476). Logged as exp #8 (**new overall best**).
6. **Round 4 (seed-bag)**: second random_state=2024 copy of CAT_tuned. **No improvement**
   (0.56716 < 0.56769). Logged as exp #9 (non-improving round #1 of the final pair).
7. **Diagnostic — nested cutpoints, re-run on the exp#8 6-way blend** (exp #10, not a model
   change): full-OOF QWK 0.56769 vs nested QWK 0.56393 (gap +0.00377) — **even tighter** than the
   5-way blend's gap (0.01644), i.e. the risk shrank as the pool matured. Fold-level cutpoints are
   very stable across folds — full-OOF cutpoint fitting is safe here.
8. **Round 5 (diverse ordinal head)**: LGB multiclass classifier (`class_weight="balanced"`,
   targeting the extreme-class 3/8 data starvation) with expected-value decode
   (`sum(class_i * P(class_i))`) added as a 7th pool member. Single-model post-rounder QWK 0.52040
   (worse than every tuned regression member); weight search gave it **zero** weight. **No
   improvement** (0.56769 = 0.56769). Logged as exp #11 (non-improving round #2 of the final pair).

**STOP reached**: 2 consecutive non-improving rounds (exp #9 seed-bag, exp #11 multiclass-head) per
protocol, on top of an already-substantial gain. Final champion: **exp #8, 6-way blend (LGB/XGB/CAT
originals + LGB_tuned + LGB_tuned_seed2024 + CAT_tuned), weights LGB 0.05 / CAT_tuned 0.95 (rest
≈0), OptimizedRounder(cutpoints=[3.574, 4.586, 5.609, 6.174, 7.628]), OOF QWK = 0.56769**.

## EDA key findings
1. Severe class imbalance: quality 3 = 0.6% (12 rows), quality 8 = 1.9% (39 rows); 5 and 6 dominate (79% combined). Imbalance ratio 69.9x.
2. `alcohol` (Spearman r=0.504) and `sulphates` (r=0.457) are by far the strongest single-feature predictors; `volatile acidity` (r=-0.248) and `total sulfur dioxide` (r=-0.227) next.
3. No missing values, no duplicate rows (feature-only or full), no leakage — `Id` is a plain row index.
4. Train/test feature distributions are near-identical (largest mean diff 2.1%, citric acid) — no covariate shift.
5. Mild collinearity: fixed acidity vs citric acid (r=0.696), vs pH (r=-0.674), vs density (r=0.616); free vs total SO2 (r=0.638). Kept both sides — tree models handle collinearity natively.
6. **Modeling-head decision** (justified in `scripts/eda.py` §7): regression, not multiclass classification. QWK penalizes squared distance between predicted/true class; a regressor's continuous output respects ordering and lets abundant mid-classes (5/6) inform placement of the data-starved extremes (3/8) — reconfirmed this run: the diverse multiclass-EV member (round 5 above) scored 0.52040, well below the tuned regression members, and got zero blend weight.

## CV scheme
- **5-fold StratifiedKFold on the raw `quality` label** (6 classes), shuffle, seed=42. Unchanged
  across all 11 experiments (comparability requirement for this iteration).

## Feature engineering (11 → 21 features, unchanged this run)
Domain-informed ratios/interactions on top of the raw columns: `free_so2_ratio`, `bound_so2` (bound vs free SO2); `fixed_to_volatile_acid`, `citric_to_volatile_acid`, `total_acid`, `acid_ph_ratio` (acidity structure); `alcohol_x_sulphates`, `alcohol_to_density`, `alcohol_x_density` (the two strongest raw predictors combined); `sugar_to_alcohol` (fermentation-completeness proxy).
No feature changes this iteration — all gains came from hyperparameter tuning (Optuna, direct
post-rounder-QWK objective) and finer blend weight search.

## Concerns
- Test-set predictions from the new best (exp #8) contain **no quality-3, 4, or 8 rows** — only
  5/6/7 (589/439/344 of 1372). This is narrower than the prior best's 4/5/6/7 spread. Expected
  given the 12/39-row training support for the true extremes and a global regression signal, but
  worth flagging more strongly than before: the tuned CatBoost member (95% of blend weight) may be
  smoothing predictions harder than the original blend. If the private test set has meaningful
  quality-4 support this could cost QWK there even though OOF says otherwise (OOF also has very
  little quality-4 support to detect this).
- Optimized-rounder cutpoints are fit on the full 5-fold OOF vector (standard practice). This run's
  nested-cutpoint diagnostic (exp #7, #10) shows the overfit risk is modest and shrinking
  (0.01644 → 0.00377 gap as the pool matured) — this addresses and downgrades the concern flagged
  in the prior STATUS.md.
- Not submitted to Kaggle (no credentials in this session) — CV-only result; no CV↔LB gap can be
  assessed yet.

## Potential improvements (not attempted, time-boxed out)
- True ordinal-regression loss (e.g. CORAL/CORN) — the simpler multiclass+EV-decode proxy tried
  this run did not help, but a proper ordinal loss is a different mechanism and untested.
- Class-weight tweaks specifically on the regression heads for extreme classes 3/8 (only tried on
  the multiclass diagnostic member, not on LGB/XGB/CAT regressors directly).
- A small stacking meta-model on OOF predictions (Ridge stacking has failed to beat simplex weight
  search in other competitions per knowledge/experience.md — expected low EV here too).

## Files
```
competitions/playground-series-s3e5/
├── config.yaml
├── STATUS.md
├── experiments.json          (11 experiments)
├── facts.json                (regenerated via kaggle-report collect.py)
├── REPORT.md / REPORT.pdf    (regenerated, facts.best = exp #8)
├── data/ (train.csv, test.csv, sample_submission.csv, train_processed.csv, test_processed.csv)
├── scripts/
│   ├── eda.py
│   ├── features.py
│   ├── train.py               (original 3-model + naive-vs-optimized-rounder pipeline)
│   ├── iterate.py             (Phase B: Optuna tuning, pool growth, seed-bag, nested-cutpoint diagnostics)
│   └── cache/*.npz            (per-member OOF/pred checkpoints, resumable)
└── submissions/
    ├── sub_generic_0.47871_20260703_120341.csv        (pre-existing baseline)
    ├── sub_blend_optround_0.52687_20260703_184829.csv (prior best)
    └── sub_iterate_0.56769_20260703_231704.csv         (NEW BEST, this run)
```

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e5/scripts/eda.py
uv run python3 competitions/playground-series-s3e5/scripts/features.py
uv run python3 competitions/playground-series-s3e5/scripts/train.py
# Phase B iteration (see REPORT.md §8 for the full ordered command sequence):
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py base
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_lgb
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r1_pool
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_cat
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r2_pool
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r3_multiclass
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py submit
```
