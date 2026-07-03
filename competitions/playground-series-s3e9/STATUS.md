# STATUS — playground-series-s3e9 (Concrete Compressive Strength)

- **Task**: regression, predict `Strength`; metric **RMSE** (minimize); id col `id`
- **Data**: 5,407 train / 3,605 test; 8 numeric features (cement/slag/fly-ash/water/
  superplasticizer/coarse-agg/fine-agg components + curing `AgeInDays`); no missing, no dups
  on full rows, but ~56% of train **feature-rows** duplicate another row with a different
  measured `Strength` (label noise, not leakage)
- **Rules**: no external data, no pretrained, no internet; daily submission limit 5
- **Not submitted to Kaggle this run** (no credentials available in this session — see
  Reproduce section for the submit command once credentials exist)

## Progress
- [x] Stage 0 Setup — `config.yaml`, data present (from a prior session)
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py`
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost, RMSE objective)
- [x] Stage 4 Self-improvement iteration — interaction features tried, reverted (see below)
- [x] Stage 5 Submission generated (local only) — `submissions/sub_blend_12.07347_20260703_191124.csv`
- [ ] Submitted to Kaggle leaderboard — pending (no API credentials this session)

## Current best score (local CV only — no LB yet)
| | OOF RMSE |
|-|----------|
| Blend (weighted) | **12.07347** |

Baseline to beat (experiment #1, generic untuned blend): **12.54287**.
Improvement: 12.54287 − 12.07347 = 0.46940 (3.74% relative).

## EDA key findings (`scripts/eda.py`)
1. `AgeInDays` is the dominant driver (Pearson 0.334, Spearman 0.604 with raw values) but
   heavily right-skewed (skew 2.75); `log1p(AgeInDays)` correlates far more strongly and
   near-linearly with `Strength` (Pearson 0.558, Spearman 0.604) — concrete strength
   develops roughly log-linearly with curing time.
2. `BlastFurnaceSlag` and `FlyAshComponent` are supplementary cementitious materials that
   are zero in 58.6% / 72.6% of rows respectively (only used in some mixes, not missing data).
3. Classic water/cement ratio (Pearson −0.151) is a weaker signal than **water/total-binder**
   ratio (binder = Cement + Slag + FlyAsh), which reaches Pearson −0.227 — slag and fly ash
   also consume water in hydration, so the naive cement-only ratio is domain-incomplete.
4. No missing values in train or test; train/test numeric distributions are close
   (largest mean shift: `AgeInDays` −5.02%, `BlastFurnaceSlag` −4.79%) — no strong
   covariate shift.
5. **~56% of train feature-rows (excluding target) are exact duplicates of another train
   row**, each with a *different* measured `Strength` — this is measurement/synthetic
   resampling noise, not leakage, and it caps the achievable CV score (irreducible label
   noise) while punishing high-capacity models that try to memorize specific rows.
6. This directly explains the baseline anomaly noted in `experiments.json` #1: with
   default (unregularized, many-leaf) hyperparameters, LGB (RMSE 13.21) and XGB (RMSE
   12.98) overfit this noisy small dataset (5,407 rows / 8 features) far more than
   CatBoost's ordered-boosting regularization (RMSE 12.54), so the OOF weight search
   zeroed both of them out and CatBoost took 100% of the blend.

## CV scheme
- **5-fold StratifiedKFold on `Strength` deciles** (`pd.qcut(y, 10)`), shuffle, seed=42 —
  chosen over plain KFold for more stable folds on a small (5.4k-row), noisy regression target.

## Feature engineering (22 features, `scripts/features.py`)
Raw 8 components + `log_age`, `sqrt_age`; zero-inflation flags `has_slag` / `has_flyash` /
`has_superplasticizer`; total cementitious `binder`; ratios `water_binder_ratio`,
`water_cement_ratio`, `sp_binder_ratio`, `agg_binder_ratio`, `fine_coarse_ratio`,
`slag_cement_ratio`, `flyash_cement_ratio`; `total_mass` (sum of all components).

## Model results (OOF RMSE, 5-fold)
| Model | OOF RMSE | time |
|-------|----------|------|
| LightGBM (regularized: num_leaves=15, depth=5, L1=2, L2=4) | 12.11061 | 7s |
| XGBoost (regularized: depth=4, min_child_weight=8, L1=2, L2=4) | 12.12086 | 8s |
| CatBoost (depth=6, l2_leaf_reg=6) | 12.07459 | 3s |
| **Weighted blend (LGB 0.15 / XGB 0.0 / CAT 0.85)** | **12.07347** | — |

The fix for the "LGB/XGB get 0 weight" anomaly was **regularization + features**, not
just features alone: with the same 22 engineered features but default (unregularized)
depth/leaves, LGB/XGB still overfit; explicit shallow depth + higher L1/L2 on both
brought LGB's OOF down to a level where the weight search now assigns it 15% (still 0%
for XGB — CatBoost remains the strongest individual learner on this data).

## Self-improvement iteration (Stage 4, experiment #3 → reverted)
**Hypothesis**: adding explicit interaction terms (`binder × log_age`,
`cement × log_age`, `water_binder_ratio × log_age`) should help because concrete
hydration strength is physically a joint function of binder amount and curing time.

**Result**: OOF RMSE got *worse* (12.09483 vs 12.07347, +0.02136) — a surprise given
the domain rationale, so treated as a Reflexion trigger.

**Self-critique**: tree-based models (LGB/XGB/CatBoost) already learn multiplicative
interactions natively via sequential splits; adding the explicit product columns only
increased dimensionality/collinearity on an already-small, noisy dataset (5,407 rows,
now 25 features) without adding information the trees couldn't already extract —
net effect was more overfitting surface, not more signal.

**Decision**: reverted (`scripts/features.py` documents the negative result inline).
Re-ran with the reverted feature set (experiment #4) and reproduced the byte-identical
prediction file and exact same 12.07347 OOF RMSE, confirming determinism and that
experiment #2's result was not a fluke. The worse (exp #3) submission file was deleted
from `submissions/` after logging to keep only the two best-CV submission files
(exp #2 / exp #4, identical) and the baseline for comparison.

## Key observations
- Small-data + label-noise story (56% duplicate feature-rows with differing targets)
  is the single most important fact about this competition — it bounds how much any
  model/feature engineering can improve RMSE and rewards regularization over capacity.
- No CV↔LB comparison available yet (not submitted to Kaggle this run).

## Potential improvements (not yet tried)
- Optuna tuning of CatBoost depth/l2_leaf_reg (currently hand-set) — small headroom
  expected given the label-noise ceiling, but untried.
- A denoising approach: for exact-duplicate feature-rows, replace the target with the
  group mean before training (removes label noise at the cost of some info) — risky,
  would need careful CV-fold-safe implementation to avoid leakage.
- Stacking meta-model on the 3 base OOF predictions instead of a linear weight search.

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e9/scripts/eda.py
uv run python3 competitions/playground-series-s3e9/scripts/train.py
# submit (needs a valid Kaggle token; none available in this session):
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e9 \
  -f competitions/playground-series-s3e9/submissions/sub_blend_12.07347_20260703_191124.csv \
  -m "<msg>"
```

## Files
```
competitions/playground-series-s3e9/
├── config.yaml
├── STATUS.md
├── experiments.json          # 4 experiments: #1 baseline, #2 best, #3 reverted, #4 confirm
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── scripts/
│   ├── eda.py
│   ├── features.py
│   └── train.py
└── submissions/
    ├── sub_generic_12.54287_20260703_120632.csv       # baseline
    ├── sub_blend_12.07347_20260703_191007.csv          # current best (exp #2, not submitted)
    └── sub_blend_12.07347_20260703_191124.csv          # identical reproduction (exp #4)
```
