# STATUS — playground-series-s3e3 (Employee Attrition)

- **Task**: binary classification, predict `Attrition`; metric **ROC-AUC** (maximize); id col `id`
- **Data**: 1,677 train / 1,119 test; 33 raw features (25 numeric + 8 categorical); no missing, no dups
- **Rules**: no external data, no pretrained, no internet; daily submission limit 5

## Progress
- [x] Stage 0 Setup — config.yaml, data present, generic baseline (exp #1, 0.81624) pre-logged
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py`
- [x] Stage 3 Modeling + CV — `scripts/train.py` (v1), `scripts/train_v2.py` (v2, best)
- [x] Self-improvement iteration (Reflexion) — v2 fixes v1 regression, beats baseline
- [x] Stage 5 Submission generated — `submissions/sub_blend_0.83292_20260703_183913.csv`
- [ ] Not submitted to Kaggle leaderboard (unattended run — submission generation only, per instructions)

## Experiment trajectory (Verifiable Rewards)
| exp | model | OOF ROC-AUC | Δ vs baseline (0.81624) |
|-----|-------|-------------|--------------------------|
| #1 | generic LGB+XGB+CAT blend (no FE) | 0.81624 | baseline |
| #2 | engineered features + imbalance weighting (scale_pos_weight/class_weights) | 0.81901 | +0.00277 |
| #3 | **engineered features, no imbalance weighting, lighter LGB (v2)** | **0.83292** | **+0.01668** |

**Best: exp #3, `scripts/train_v2.py` → `submissions/sub_blend_0.83292_20260703_183913.csv`**

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

## Next ideas (if resumed)
- CatBoost still underperforms baseline; investigate cat_features native handling (currently
  label/freq-encoded like the trees) rather than assuming it needs the same regularization as LGB.
- Try target encoding (out-of-fold, leakage-safe) for high-cardinality `JobRole`/`EducationField`
  as an alternative to label+freq encoding.
- Feature selection: drop redundant raw tenure columns (keep engineered ratios only) — untested,
  flagged as a hypothesis in EDA but not verified.
- Small-n dataset: consider RepeatedStratifiedKFold (e.g. 5×3) for more stable OOF estimates
  before trusting further micro-tuning.

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e3/scripts/eda.py
uv run python3 competitions/playground-series-s3e3/scripts/train_v2.py   # best (exp #3, 0.83292)
```
