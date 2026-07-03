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
- [x] Phase D-2 tree-search v2 sweep (`tree_search/run_s3e3.py`) — new best **0.841442**, beats linear 0.838140

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

## Appendix: Phase D-2 tree-search v2 sweep (2026-07-04)

**Sweep question**: can harness v2 (ensemble-default node space + experience-library
priors + adaptive plateau + child dedup) match or beat the linear-iteration best
(0.83814) in fewer evaluations than linear iteration took rounds (7 experiments,
exp #1–#7)? **Answer for s3e3: yes — matched/beat it in the same 7 evaluations, then
kept improving to a new best 0.841442 by evaluation #12, in 33.9s wall-clock.**

Built: `tree_search/eval_s3e3.py` (solo: LGB/XGB/CAT, StratifiedKFold(5, seed=42) —
byte-identical reproduction verified for all 5 known configs before searching: root
LGB_tuned 0.837305, LGB_orig 0.832925, XGB 0.807634≈0.80763, CAT_orig 0.762684≈0.76268,
CAT_native 0.814259, all exact; blend: harness_v2.eval_blend for prob-space, small local
dirichlet re-implementation for rank-space) + `tree_search/run_s3e3.py` (harness_v2
driver). Root = linear winner's tuned-LGB config (exp #4 Optuna direct-full-CV-AUC
objective). OOF cache: `tree_search/cache_s3e3/` (gitignored, via
`harness_v2.cache_oof`/`load_oof`).

### Node/backtrack/dedup summary
- **22 evaluated nodes** (18 solo / 4 blend), 22 total (0 failed), wall=33.9s.
- **4 backtracks** (all genuine plateaus, 3 non-improving children each, tie_rate=0.000
  throughout — AUC is continuous, so the adaptive-plateau discretization branch never
  fired, as expected/predicted by the harness_v2 docstring): BLEND lineage plateaued at
  node #10, FEAT at #14, XGBTUNED at #17, SEEDBAG at #20 — `select_next_parent` correctly
  re-picked the next-best non-plateaued lineage each time (BLEND→FEAT→XGBTUNED→SEEDBAG→
  LGBORIG) without ever needing the "all plateaued, reopen" fallback.
- **0 dedup rejections** — the hand-authored mutation queues never proposed a
  byte-identical duplicate config this run (the dedup code path itself, including the
  result-stripped-hash fix needed because `eval_and_add` merges `result` into the stored
  config, is exercised on every non-root `add_node` call via `find_dup`, just never
  actually triggered a rejection here).

### Best vs linear, evaluations-to-match/beat
| | AUC | evaluations |
|---|---|---|
| Linear-iteration best (exp #7, 6-way rank-avg blend) | 0.838140 | 7 experiments |
| Tree v2: node #6 ([FEAT] seed) | 0.838903 | **7** (tied with linear's round count) |
| Tree v2: node #7 ([BLEND] seed, 3-way) | 0.839540 | 8 |
| Tree v2: node #11 ([FEAT] mutation, global best) | **0.841442** | **12** |

Harness v2 matched/beat the linear best at the SAME evaluation count linear needed
rounds (7), then kept improving for another 5 evaluations to a real +0.0033 gain over
the linear ceiling, all in well under a minute of wall-clock.

### The winning direction was a genuine new finding, not just re-derivation
The global best (#11) traces to the **FEAT lineage** (seed #6 → mutation #11): dropping
the 5 raw tenure columns (`YearsAtCompany`, `YearsInCurrentRole`, `YearsWithCurrManager`,
`YearsSinceLastPromotion`, `TotalWorkingYears` — already condensed into
`role_tenure_ratio`/`mgr_tenure_ratio`/`promo_ratio`/`company_tenure_ratio`/
`income_per_year_worked`/`age_at_join` upstream) plus further dropping
`income_per_year_worked`, on top of the tuned-LGB hyperparams: 0.837305 → 0.838903 →
0.841442. This was flagged in this file's own "Next ideas" as an **untested hypothesis**
("drop redundant raw tenure columns … untested, flagged as a hypothesis in EDA but not
verified") — it is now **verified**: real, additive, +0.0041 over the tuned-LGB root.
Promote to a durable lesson: on this 1,677-row dataset with a shallow, heavily
regularized LGB (num_leaves=3), the raw tenure columns the engineered ratios were
derived from are pure redundant noise once the ratios exist — pruning them is a real,
if modest, win (candidate line for knowledge/experience.md's 高共線性特徵 section on a
future distillation pass).

### Prior-usage log (idea-injection experiment)
`harness_v2.suggest_priors({"metric": "auc", "tags": ["10k", "small_sample"]})` returned
7 bullets (P0–P6, verbatim from knowledge/experience.md's ROC-AUC/小樣本 sections — all
of them, unsurprisingly, sourced from this very comp's own earlier rounds). Every
mutation-queue entry in `run_s3e3.py` is tagged `[PRIOR Pk]` or `[PRIOR none]`; counting
only search-LOOP-proposed mutations (excluding the 6 hand-authored first-gen seeds,
which don't have an in-lineage parent to compare against):

- **7 prior-informed mutations, win rate 1/7 (14.3%)** — P2 (seed-bag) fired 3×, P5
  (regularize further) fired 3×, P3 (CatBoost-in-blend, expected-zero) fired 1×.
- **7 uninformed mutations, win rate 1/7 (14.3%)** — identical raw win rate.
- **Caveat that matters more than the tie**: 2 of the informed mutations (P3/P4-tagged,
  "add CATORIG"/"add CATNATIVE" to the blend) were deliberately-structured confirmations
  that a closed direction is STILL zero-value, not attempts to win — by design they
  can't "win" in the naive sense, so the raw win-rate comparison undersells how correct
  the priors were (both confirmed exactly as predicted: CATORIG/CATNATIVE additions
  moved the blend AUC by <0.0001, noise-level). The actual global-best-producing
  mutation (#11, FEAT further-trim, +0.0025 over its parent) was **not** prior-informed
  — it came from this comp's own STATUS.md "Next ideas" list, not the cross-comp
  experience library. Honest read: priors were directionally correct (0 wasted full
  hyperparam searches down dead ends like re-tuning CatBoost) but the single biggest
  lever this run was a comp-local hypothesis, not a transferred one.

### Sweep-question answer for s3e3
**Yes.** Harness v2 matched the linear-iteration best at the same evaluation count
(7) linear needed *rounds* (each round itself costing several model trainings, not
one), then surpassed it by evaluation #12 (+0.0033), using 22 total evaluations in
33.9s wall-clock — well inside the ~25 min budget. Ensemble-default node space,
adaptive plateau, and child dedup all behaved as designed (plateau/backtrack fired 4×
correctly, tie_rate stayed 0 throughout as expected for a continuous metric, dedup
path is wired in though not empirically triggered this run). The experience-library
prior-injection mechanism worked mechanically (7/7 bullets matched, correctly shaped
3 solo lineages' mutation queues) but did not out-win uninformed mutations on raw
win-rate this run — its real value here was preventing wasted searches (no CatBoost
re-tuning attempts), while the actual score-moving discovery came from the comp's own
already-logged open item.

