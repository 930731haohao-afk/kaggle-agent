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
- [x] Stage 4b Phase B self-improvement iteration (4 rounds, exp #5–#8, see below) —
  new best via seed-bagged pool (dup-smoothing and Optuna-tuned-LGB-solo did not help)
- [x] Stage 5 Submission generated (local only) — `submissions/sub_blend_12.07003_20260703_235120.csv`
- [ ] Submitted to Kaggle leaderboard — pending (no API credentials this session)

## Current best score (local CV only — no LB yet)
| | OOF RMSE |
|-|----------|
| Blend (7-way weighted, exp #8) | **12.07003** |

Baseline to beat (experiment #1, generic untuned blend): **12.54287**.
Improvement: 12.54287 − 12.07003 = 0.47284 (3.77% relative).
Phase B iteration gain over previous best: 12.07347 − 12.07003 = 0.00344.

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

## Phase B self-improvement iteration (exp #5–#8, 2026-07-03 evening)
Four rounds, one isolated change each, same folds/seed/features as exp #2 throughout
(5-fold StratifiedKFold on Strength deciles, seed 42 — scores directly comparable).

| Round | Change | OOF RMSE | Verdict |
|-------|--------|----------|---------|
| 1 (exp #5) | Fold-safe duplicate-group target smoothing (train targets → group-mean Strength of exact raw-feature duplicates within the training fold; validation/metric untouched) | 12.08122 | worse (+0.00775), reverted |
| 2 (exp #6) | Optuna-tuned LGB (TPE 60 trials, full-5fold-CV objective, 371.6s) added to pool as 4th member | 12.07347 | tie — tuned LGB solo 12.12074 worse than orig 12.11061, weight search gave it 0 |
| 3 (exp #7) | Seed-bagged the tuned LGB (2nd random_state=1042) as 5th member | **12.07143** | improved (−0.00204); weights {CAT 0.773, LGB_tuned_seed2 0.227} |
| 4 (exp #8) | Seed-bagged the dominant CatBoost (seed 1042) + 3rd tuned-LGB seed (2042); 7-way pool | **12.07003** | improved (−0.00140); weights {CAT 0.478, CAT_seed2 0.265, LGB_tuned_seed2 0.127, LGB_tuned_seed3 0.129} |

**Round 1 self-critique (dup-smoothing failed — why)**: GBDT with squared loss already
fits the *conditional mean* of duplicated feature rows implicitly (the squared-loss
minimizer over identical inputs is their target mean), so explicit group-mean smoothing
adds no information; what it does change is the effective per-group sample weighting
and it biases the training signal for groups split across folds (a group's fold-local
mean is a noisy estimate of its true mean). Net effect: all three models slightly worse
(LGB 12.11061→12.12277, XGB 12.12086→12.11109*, CAT 12.07459→12.08207; *XGB alone
improved, but not enough to help the blend). The label-noise diagnosis was right; the
implied fix was already priced in by the loss function. Precision on the diagnosis:
2,401/5,407 rows (44.4%) are non-first duplicates; ~56% of rows belong to some
duplicate group (both figures describe the same structure).

**Rounds 2–4 takeaway**: on this label-noise-bounded data, the entire Phase B gain
came from *seed bagging* (variance reduction), not from tuning or denoising. Optuna's
best full-CV solution (num_leaves 11, depth 3, lr 0.0198, weak L1/L2) solo scored
worse than the hand-regularized LGB and earned 0 blend weight — yet its *seed variants*
earned 25.6% combined weight, and seed-bagging CatBoost (77%→ split 47.8/26.5) gave
the largest single-round gain. Consistent with the noise-ceiling story: at the ceiling,
averaging independently-seeded models is the only "free" direction left.

## Key observations
- Small-data + label-noise story (56% duplicate feature-rows with differing targets)
  is the single most important fact about this competition — it bounds how much any
  model/feature engineering can improve RMSE and rewards regularization over capacity.
- Phase B confirmed the ceiling operationally: denoising (dup-smoothing) and
  hyperparameter tuning both failed to beat it; only seed-bagging/averaging moved it.
- No CV↔LB comparison available yet (not submitted to Kaggle this run).

## Potential improvements (not yet tried)
- Stacking meta-model on the base OOF predictions instead of a linear weight search
  (note: Ridge stacking lost to simplex search on s3e14 — low EV).
- More seeds in the bag (returns are visibly diminishing: −0.00204 → −0.00140).
- GroupKFold on duplicate-groups as a CV-honesty diagnostic (untried; would change
  the CV scheme so scores would not be comparable to the current series).

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e9/scripts/eda.py
uv run python3 competitions/playground-series-s3e9/scripts/train.py            # 3-way best (12.07347)
# Phase B iteration (in order; round 4 consumes round 2/3's npz checkpoint):
uv run python3 competitions/playground-series-s3e9/scripts/train_dup_smooth.py     # round 1 (12.08122, negative)
uv run python3 competitions/playground-series-s3e9/scripts/train_optuna_pool.py    # rounds 2+3 (12.07347 / 12.07143)
uv run python3 competitions/playground-series-s3e9/scripts/train_round4_seedbag.py # round 4 → best 12.07003 + submission
# submit (needs a valid Kaggle token; none available in this session):
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e9 \
  -f competitions/playground-series-s3e9/submissions/sub_blend_12.07003_20260703_235120.csv \
  -m "<msg>"
```

## Files
```
competitions/playground-series-s3e9/
├── config.yaml
├── STATUS.md
├── experiments.json          # 8 experiments: #1 baseline, #2 best-3way, #3 reverted,
│                             # #4 confirm, #5 dup-smooth (neg), #6 +tuned LGB (tie),
│                             # #7 +seed-bag LGB (12.07143), #8 7-way seed-bag best (12.07003)
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── scripts/
│   ├── eda.py
│   ├── features.py
│   ├── train.py                  # 3-way blend (exp #2/#4)
│   ├── train_dup_smooth.py       # Phase B round 1 (negative result, kept for record)
│   ├── train_optuna_pool.py      # Phase B rounds 2+3
│   ├── train_round4_seedbag.py   # Phase B round 4 (current best)
│   └── _round*.npz               # OOF/pred checkpoints (regenerable, not committed)
└── submissions/
    ├── sub_generic_12.54287_20260703_120632.csv       # baseline
    ├── sub_blend_12.07347_20260703_191007.csv          # exp #2 3-way best
    ├── sub_blend_12.07347_20260703_191124.csv          # identical reproduction (exp #4)
    └── sub_blend_12.07003_20260703_235120.csv          # CURRENT BEST (exp #8, not submitted)
```

## Phase C-2a — Tree-search prototype (2026-07-04)

Ran the ERA-inspired (Aygün et al. 2026) candidate-tree search prototype (計畫書
Stage-4 先遣實驗) against s3e9, using `tree_search/harness.py` (generic
selection/plateau/backtrack engine) + `tree_search/eval_s3e9.py` (per-comp evaluator,
reuses the *exact* same 22-feature set and 5-fold `StratifiedKFold` decile scheme,
seed=42, as `scripts/train.py` — single-model configs only, LGB/XGB/CAT, no blending).
Driver: `tree_search/run_s3e9.py`. Full tree persisted to `experiments_tree.json`.

**Run stats**: 20/20 nodes evaluated (0 failed), wall time **125.3s** (well under the
~40 min budget; per-node cost 2.0–17.0s, cheaper than the 20–60s design target).
Root = s3e9's current hand-regularized single-LGB config (reproduced byte-for-byte:
12.11061, matching the table above). 5 first-generation lineages fanned out from
root: **CAT** (model swap), **XGB** (model swap), **FEAT** (feature-subtraction),
**ROBUST** (huber/fair loss objective swap — untested direction), **REG** (further
capacity/regularization nudges).

**Backtracks: 4 genuine plateau→backtrack events** (not faked — logged in
`experiments_tree.json`'s `search_state.backtrack_log`): CAT plateaued after 3
non-improving children → switched to REG (2nd-best lineage) → REG plateaued →
switched to FEAT (3rd-best) → FEAT plateaued → switched to XGB (4th-best) → XGB
plateaued at the node budget. ROBUST (5th-best, worst-scoring lineage) never got a
turn as "active" — it only received its 2 seed-adjacent children while other
lineages were active, since it never surfaced as best-scoring.

**Best tree-search node**: `#1`, the **CatBoost seed itself** (depth=6, l2=6),
OOF RMSE **12.07459** — i.e. no proposed mutation across all 20 nodes beat the
already-known-best single-model config from the earlier linear iteration. Every one
of CAT's own 3 children (regularize more / shrink depth / add bagging_temperature)
tied or lost to the parent; every REG, FEAT, and XGB child scored 12.10–12.11 range
(worse than CAT, none beating the 12.11061 LGB reference either in most cases).

**Comparisons** (all CV-only, no LB):
| Reference | RMSE | vs tree-search best (12.07459) |
|-----------|------|------|
| Single-model root (regularized LGB solo, this run's root) | 12.11061 | tree search **-0.03602** better (rediscovers the already-known CAT solo score, not a new gain) |
| Linear-iteration 7-way seed-bagged blend (exp #8) | **12.07003** | tree search **+0.00456** worse — expected: this harness evaluates single models only, it cannot reach the blend/seed-bagging space that produced the actual best score |

**Most productive mutation direction**: none, strictly — the tree search did not
discover anything better than a config linear iteration already knew about. If
forced to rank, "model-type switch to CatBoost" (the very first branch) was the only
lineage worth exploring further; every genuinely *new* mutation (huber/fair loss,
feature subtraction, deeper regularization, XGB nudges) moved the score the wrong
way. The clearest and most useful negative finding: **swapping to a robust loss
(huber/fair) made LGB substantially worse** (12.11061 → 12.16–12.22), the opposite
of the a-priori hypothesis that a loss less sensitive to residual magnitude would
help under the 56%-duplicate-row label-noise ceiling — worth remembering as a
"tried, don't retry" data point alongside dup-group smoothing and interaction terms.

**Harness design deviations from the brief** (documented in `harness.py`'s
docstring): (1) a lineage that exhausts its per-node expansion budget
(`MAX_CHILDREN_PER_NODE=3`) is also treated as "plateaued" even without 3
non-improving children, so the search never stalls; (2) if literally every lineage
is plateaued the flags are cleared once as a fallback. Neither triggered in this
run (plain 3-non-improvement plateaus accounted for all 4 backtracks).

**Feasibility verdict for Stage 4**: the harness mechanics are solid and cheap
(crash-safe JSON-after-every-node, deterministic, ~6s/node average, adaptive
backtracking worked exactly as designed) and are ready to scale up; but to actually
beat this competition's current best score, Stage 4 needs the search space widened
beyond single-model nodes — e.g. an explicit "ensemble/seed-bag of two existing tree
nodes" node type — since on this particular label-noise-capped dataset the real
historical gains (Phase B, 12.07347→12.07003) came entirely from seed-bagging /
blending, a move this single-model-per-node prototype structurally cannot make.
