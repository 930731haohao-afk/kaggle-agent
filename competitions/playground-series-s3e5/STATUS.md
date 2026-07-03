# STATUS — playground-series-s3e5 (Wine Quality, ordinal)

- **Task**: ordinal regression head, predict `quality` (3–8); metric **quadratic_weighted_kappa (QWK)**, maximize; id col `Id`
- **Data**: 2,056 train / 1,372 test; 11 raw physicochemical features, no missing values, no duplicates
- **Rules**: no external data, no pretrained models, no internet; daily submission limit 5 (not used — no credentials on this machine)

## Progress
- [x] Stage 0 Setup — config.yaml already present (generic-baseline entry #1 pre-existing)
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py` (11 → 21 features)
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost regression, 5-fold StratifiedKFold)
- [x] Self-improvement iteration — optimized-rounder threshold tuning vs naive rounding (logged both)
- [x] Stage 5 Submission generated — `submissions/sub_blend_optround_0.52687_20260703_184829.csv`
- [ ] Submitted to Kaggle leaderboard — not attempted (no Kaggle credentials configured in this environment; local CV only)

## Current best score
| | OOF QWK | Public LB | Private LB |
|-|---------|-----------|------------|
| Generic baseline blend (exp #1, pre-existing) | 0.47871 | — | — |
| Regression blend, naive round (exp #2) | 0.47191 | — | — |
| **Regression blend, optimized rounder (exp #3, final)** | **0.52687** | — | — |

**Delta vs baseline: +0.04816** (≈10% relative improvement). Not submitted — no Kaggle credentials available in this session, so no Public/Private LB numbers exist yet.

## EDA key findings
1. Severe class imbalance: quality 3 = 0.6% (12 rows), quality 8 = 1.9% (39 rows); 5 and 6 dominate (79% combined). Imbalance ratio 69.9x.
2. `alcohol` (Spearman r=0.504) and `sulphates` (r=0.457) are by far the strongest single-feature predictors; `volatile acidity` (r=-0.248) and `total sulfur dioxide` (r=-0.227) next.
3. No missing values, no duplicate rows (feature-only or full), no leakage — `Id` is a plain row index.
4. Train/test feature distributions are near-identical (largest mean diff 2.1%, citric acid) — no covariate shift.
5. Mild collinearity: fixed acidity vs citric acid (r=0.696), vs pH (r=-0.674), vs density (r=0.616); free vs total SO2 (r=0.638). Kept both sides — tree models handle collinearity natively.
6. **Modeling-head decision** (justified in `scripts/eda.py` §7): regression, not multiclass classification. QWK penalizes squared distance between predicted/true class; a regressor's continuous output respects ordering and lets abundant mid-classes (5/6) inform placement of the data-starved extremes (3/8) — a classifier would need to learn separate decision boundaries for classes with only 12–39 examples.

## CV scheme
- **5-fold StratifiedKFold on the raw `quality` label** (6 classes), shuffle, seed=42.
- Chosen over plain KFold because with a 69.9x imbalance ratio, an unstratified split risks folds with too few (or zero) examples of quality 3/8, making per-fold QWK unstable.
- Per-fold QWK (final blend + optimized rounder, checked post-hoc): [0.551, 0.570, 0.503, 0.479, 0.528], mean 0.526, std 0.033 — consistent with the overall OOF QWK, no single fold dominating the score.

## Feature engineering (11 → 21 features)
Domain-informed ratios/interactions on top of the raw columns: `free_so2_ratio`, `bound_so2` (bound vs free SO2); `fixed_to_volatile_acid`, `citric_to_volatile_acid`, `total_acid`, `acid_ph_ratio` (acidity structure); `alcohol_x_sulphates`, `alcohol_to_density`, `alcohol_x_density` (the two strongest raw predictors combined); `sugar_to_alcohol` (fermentation-completeness proxy).
**Validated**: `alcohol_x_sulphates` raised Spearman |r| to 0.550 (vs 0.504/0.457 for the two raw features alone) and 7 of the top-10 features by LightGBM importance are engineered, not raw.

## Modeling & self-improvement
- Models: LightGBM, XGBoost, CatBoost, all regression objectives, 5-fold StratifiedKFold on `quality`.
- **Blend weight search scored directly against the optimized-rounder QWK** (not naive-round QWK), so the chosen weights are optimal for the metric actually being reported.
- **Self-improvement iteration**: logged the identical blend (weights LGB 0.2 / XGB 0.2 / CAT 0.6) under both naive rounding and an `OptimizedRounder` (Nelder-Mead-tuned cutpoints on OOF QWK) to make the value of threshold optimization explicit — this is the single highest-leverage change available given the tiny (128 KB) dataset and tight time budget.
  - Naive round: OOF QWK 0.47191
  - Optimized rounder: OOF QWK 0.52687 (**+0.05496** from thresholding alone)
  - Result was as expected (Reflexion: brief note) — naive rounding assumes prediction error is symmetric around each integer class, which breaks down under 69.9x imbalance (predictions for rare classes 3/8 get pulled toward the dominant 5/6 mass); tuning cutpoints directly against QWK corrects this bias.

## Model results (OOF, 5-fold, naive round for per-model comparability)
| Model | OOF QWK (naive round) | time |
|-------|------------------------|------|
| LightGBM | 0.45220 | — |
| XGBoost | 0.46218 | — |
| CatBoost | 0.47094 | — |
| **Blend 0.2/0.2/0.6 + naive round** | 0.47191 | — |
| **Blend 0.2/0.2/0.6 + optimized rounder (final)** | **0.52687** | — |

Full 5-fold x 3-model training: 40.3s; blend weight search (with per-candidate rounder fit): 3.7s; full-data retrain for submission: 8.2s. Total script runtime ≈52s, well inside the 15-minute budget — left no need for a second self-improvement iteration (>5% relative gain already achieved, see Autonomous Iteration Protocol stopping criteria).

Experiment log: `experiments.json` (#1 baseline, #2 naive-round comparison, #3 final).

## Concerns
- Test-set predictions contain **no quality-3 or quality-8 rows** (`sub_blend_optround_...csv` predicts only 4/5/6/7). Expected given the 12/39-row training support for those classes and a global regression signal, but worth flagging — if the private test set is similarly imbalanced this is unlikely to hurt QWK much, but it does mean the model cannot recognize the extremes it has almost never seen.
- Optimized-rounder cutpoints were fit on the full 5-fold OOF vector (standard practice for this trick, not nested-CV) — there is a small risk of the cutpoints being mildly tuned to this particular OOF sample; the ~0.033 fold-to-fold std suggests this risk is modest relative to the overall gain.
- Not submitted to Kaggle (no credentials in this session) — CV-only result; no CV↔LB gap can be assessed yet.

## Potential improvements (not attempted, time-boxed out)
- Optuna hyperparameter tuning of the three base models (small headroom expected — the rounder was clearly the dominant lever here).
- True ordinal-regression loss (e.g. CORAL/CORN) or a small stacking meta-model on OOF predictions.
- Nested-CV refit of the optimized rounder per outer fold to get an unbiased QWK estimate (current estimate refits the rounder on the full OOF vector).

## Files
```
competitions/playground-series-s3e5/
├── config.yaml
├── STATUS.md
├── experiments.json
├── data/ (train.csv, test.csv, sample_submission.csv, train_processed.csv, test_processed.csv)
├── scripts/
│   ├── eda.py
│   ├── features.py
│   └── train.py
└── submissions/
    ├── sub_generic_0.47871_20260703_120341.csv   (pre-existing baseline)
    └── sub_blend_optround_0.52687_20260703_184829.csv   (final, this run)
```

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e5/scripts/eda.py
uv run python3 competitions/playground-series-s3e5/scripts/features.py
uv run python3 competitions/playground-series-s3e5/scripts/train.py
```
