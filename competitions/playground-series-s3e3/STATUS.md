# STATUS — playground-series-s3e3 (Employee Attrition)

- **Task**: binary classification, predict `Attrition`; metric **ROC-AUC** (maximize); id col `id`
- **Data**: 1,677 train / 1,119 test; 33 raw features (25 numeric + 8 categorical); no missing, no dups
- **Rules**: no external data, no pretrained, no internet; daily submission limit 5

## Progress
- [x] Stage 0 Setup — config.yaml, data present, generic baseline (exp #1, 0.81624) pre-logged
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py`
- [x] Stage 3 Modeling + CV — `scripts/train.py` (v1), `scripts/train_v2.py` (v2)
- [x] Self-improvement iteration (Reflexion) — v2 fixes v1 regression, beats baseline (exp #3, 0.83292)
- [x] Phase B self-improvement iteration (4 rounds, exp #4–#7) — new best 0.83814
- [x] Stage 5 Submission generated — `submissions/sub_blend_0.83814_20260703_233303.csv`
- [ ] Not submitted to Kaggle leaderboard (unattended run — submission generation only, per instructions)

## Experiment trajectory (Verifiable Rewards)
| exp | model | OOF ROC-AUC | Δ vs baseline (0.81624) |
|-----|-------|-------------|--------------------------|
| #1 | generic LGB+XGB+CAT blend (no FE) | 0.81624 | baseline |
| #2 | engineered features + imbalance weighting (scale_pos_weight/class_weights) | 0.81901 | +0.00277 |
| #3 | engineered features, no imbalance weighting, lighter LGB (v2) | 0.83292 | +0.01668 |
| #4 | LGB Optuna-tuned, full 5-fold CV AUC objective (50 trials, solo) | 0.83731 | +0.02107 |
| #5 | 5-way blend: LGB_orig+XGB+CAT+LGB_tuned+LGB_tuned_seedbag2024 | 0.83778 | +0.02154 |
| #6 | CatBoost native cat_features retry (diagnostic, solo) | 0.81426 | +0.01802 (still weakest member) |
| #7 | **6-way blend incl. CAT_native, rank-average** | **0.83814** | **+0.02190** |

**Best: exp #7, `scripts/iterate_round4_add_catnative_blend.py` → `submissions/sub_blend_0.83814_20260703_233303.csv`**
(prior best exp #3: `scripts/train_v2.py` → `submissions/sub_blend_0.83292_20260703_183913.csv`)

## EDA key findings
1. Target imbalanced: 88.1% class 0 / 11.9% class 1 (attrition) → StratifiedKFold required.
2. No missing values, no duplicate rows, no unseen test categories.
3. `EmployeeCount`, `StandardHours`, `Over18` are constant (zero variance) → dropped.
4. Strongest categorical signal: `OverTime` (22.0% attrition if Yes vs 8.8% if No) and
   `MaritalStatus` (Single 19.8% vs Divorced 4.9%).
5. Strongest numeric correlations (all negative, i.e. protective): `StockOptionLevel`,
   `Age`, `JobInvolvement`, `TotalWorkingYears`, `JobLevel`, tenure features, `MonthlyIncome`.
6. Tenure features mutually correlated 0.75–0.79 (`YearsAtCompany`/`YearsWithCurrManager`/
   `YearsInCurrentRole`); `JobLevel`↔`MonthlyIncome` r=0.91. No single-feature leakage (AUC>0.75) found.
7. Small n (1677 rows) → high per-fold AUC variance (0.79–0.89) is expected, not a bug.

## CV scheme
- **5-fold StratifiedKFold** on `Attrition`, shuffle=True, seed=42 (fixed across all experiments
  for comparability).

## Feature engineering (48 features, from 33 raw)
Dropped 3 constant columns. Label + frequency encoding for 7 categoricals (`BusinessTravel`,
`Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`). Added: tenure
ratios (`role_tenure_ratio`, `mgr_tenure_ratio`, `promo_ratio`, `company_tenure_ratio`); income
features (`income_per_joblevel`, `income_per_year_worked`); satisfaction composite
(`satisfaction_avg`, `satisfaction_min`); `overtime_joblevel` interaction; `age_at_join`,
`companies_per_year`.

## Model results (OOF ROC-AUC, 5-fold)
### exp #2 (v1: engineered features + imbalance weighting)
| Model | OOF AUC |
|-------|---------|
| LGB | 0.81901 |
| XGB (scale_pos_weight) | 0.79196 |
| CAT (class_weights) | 0.77663 |
| Blend (weight search) | 0.81901 (LGB weight=1.0) |

### exp #3 (v2: same features, imbalance weighting removed, lighter LGB) — **BEST**
| Model | OOF AUC | time |
|-------|---------|------|
| **LGB** (num_leaves=7, reg_alpha=1.0, reg_lambda=2.0) | **0.83292** | 1.1s |
| XGB (no scale_pos_weight) | 0.80763 | 2.8s |
| CAT (no class_weights, depth=5, l2=8.0) | 0.76268 | 1.9s |
| **Blend (weight search)** | **0.83292** (LGB weight=1.0) | — |

## Reflexion note (self-improvement, applied)
exp #2's XGB/CAT scores regressed *below* their own generic-baseline per-model scores
(XGB 0.80511→0.79196, CAT 0.80874→0.77663) despite only adding feature engineering +
imbalance weighting on top. Hypothesis: ROC-AUC is a ranking metric, not threshold-sensitive,
so `scale_pos_weight`/`class_weights` mainly perturb the loss landscape/regularization rather
than helping ranking — a bad trade on a noisy 1677-row dataset with 48 partly-correlated
engineered features. exp #3 removed all imbalance weighting and tightened LGB regularization
(num_leaves 15→7, reg_alpha 0.5→1.0, reg_lambda 1.0→2.0); LGB OOF jumped 0.81901→0.83292.
XGB also improved (0.79196→0.80763) but CAT did not recover (0.76268) — CatBoost's default
regularization/depth combo appears to overfit this data regardless of weighting; it was
correctly zeroed out by the weight search in both experiments and is not used in the final
blend. Given time budget, this was **one iteration** and was stopped after a clear win
(+2.05% relative over baseline) rather than pushed further — plateau-detection heuristic
recommends banking the gain rather than over-searching a 1677-row dataset.

## Phase B self-improvement iteration (2026-07-03, exp #4–#7)

4 rounds, one change each, all on the same StratifiedKFold(5, seed=42) CV. Stopped after
round 4 on a plateau-detection basis (gains diminishing: +0.0044 → +0.0005 → +0.0004) rather
than pushing further on a 1677-row dataset.

### Round 1 (exp #4): Optuna LGB, full 5-fold CV AUC objective (solo)
Data is tiny enough (1677 rows) that a full 5-fold CV per Optuna trial is cheap (50 trials,
TPE, 111.3s total) — no fold-0 proxy needed, per experience.md's tiny-data recipe (previously
only validated for QWK/rounder objectives; this iteration confirms it transfers to a
continuous rank metric like AUC too). Best params: shallower than exp #3's already-shallow
LGB (num_leaves 3 vs 7, max_depth 4, reg_alpha 0.068, reg_lambda 0.0012 — lighter L1/L2 but
compensated by fewer leaves/higher min_child_samples=60). Solo LGB OOF: 0.832925 → 0.837305
(+0.00438).

### Round 2 (exp #5): add tuned LGB to pool + seed bag
Added round-1 tuned LGB to the exp #3 pool (LGB_orig/XGB/CAT unchanged) rather than replacing
it, plus a seed=2024 bag of the same tuned hyperparameters. 5-way weight search:
LGB_orig=0.3, LGB_tuned=0.7, everything else (XGB/CAT/seed-bag)=0. Blend: 0.837305 → 0.837776
(+0.00047). The seed-bag copy itself got zero weight — consistent with the s3e5 "seed bagging
on an already-directly-optimized model gives little extra diversity" pattern in experience.md.

### Round 3 (exp #6): CatBoost native categorical retry (diagnostic)
Resolved the long-standing open item: rebuilt a CatBoost-only feature set that keeps the 7
categorical columns as raw strings passed via `cat_features` (no label/freq pre-encoding),
with `allow_writing_files=False` + explicit `thread_count=4` (s3e11 lesson — catboost_info
file writes can stall sandboxed background runs). Swept 3 depth/reg configs; best
(depth=3, l2_leaf_reg=16.0, lr=0.05) gave OOF 0.814259 — a large jump from exp #3's
label/freq-encoded CatBoost (0.762684), i.e. native categorical handling **partially rescues**
CatBoost (+0.0516), but it remains well below LGB (0.837305) and below the 0.80 "worth
keeping in the pool" bar is cleared, so it got one more trial in round 4.

### Round 4 (exp #7): add CAT_native to pool, prob-blend vs rank-average
Added CAT_native to the round-2 5-way pool (6-way). Weight search still zeroed CAT_native
(weight 0) even with native handling — confirms CatBoost's weakness on this ~1.7k-row dataset
is structural (not merely an encoding artifact), matching exp #2/#3's earlier verdict. Compared
prob-space weight-search blend (0.837776, same as round 2 — CAT_native contributes nothing in
prob space) vs rank-average using the same weights (0.838140) — rank-average won by +0.000364.
New best: **0.838140** (submission `sub_blend_0.83814_20260703_233303.csv`).

## Next ideas (if resumed)
- CatBoost: native cat_features handling closed most of the gap (0.7627→0.8143) but it is now
  a *closed* open item — do not retry again; treat "CatBoost structurally weak on this
  ~1.7k-row dataset" as the durable verdict (see knowledge/experience.md).
- Try target encoding (out-of-fold, leakage-safe) for high-cardinality `JobRole`/`EducationField`
  as an alternative to label+freq encoding for LGB/XGB.
- Feature selection: drop redundant raw tenure columns (keep engineered ratios only) — untested,
  flagged as a hypothesis in EDA but not verified.
- Small-n dataset: consider RepeatedStratifiedKFold (e.g. 5×3) for more stable OOF estimates
  before trusting further micro-tuning.
- Rank-average vs prob-blend gap (+0.000364) is small — worth re-checking on a future iteration
  whether it holds up or is noise-level (cf. s3e7's noise-level blend-layer differences).

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e3/scripts/eda.py
uv run python3 competitions/playground-series-s3e3/scripts/train_v2.py                              # exp #3, 0.83292
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round1_optuna_lgb.py              # exp #4, 0.83731
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round2_pool_seedbag.py            # exp #5, 0.83778
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round3_catboost_native.py         # exp #6, 0.81426 (diagnostic)
uv run python3 competitions/playground-series-s3e3/scripts/iterate_round4_add_catnative_blend.py     # exp #7, 0.83814 (best)
```
