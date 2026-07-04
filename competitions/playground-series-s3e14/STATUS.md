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
- [x] Phase B self-improvement iteration (2026-07-03 evening) — 4 rounds, `scripts/train_v3.py`–`train_v6.py`
      (exp #4–#7, see "Phase B iteration rounds" below)
- [x] Stage 5 Submission generated — best: `submissions/sub_blend_v6_340.59891_20260703_211503.csv`
- [ ] Submitted to Kaggle leaderboard — **not submitted** (no Kaggle credentials in this environment; per task
      instructions this run does not submit)

## Current best score (OOF, no LB available)
| Experiment | Model | OOF MAE |
|------------|-------|---------|
| #1 (prior baseline) | generic LGB+XGB+CAT blend | 341.40782 |
| #2 (iteration 1) | feature-engineered blend + snap-to-grid | 340.95856 |
| #3 (iteration 2) | pruned features + native-cat CatBoost + snap-to-grid | 340.71180 |
| #5 (Phase B round 2) | 4-way blend (+Optuna-tuned LGB as extra member) + snap | 340.62702 |
| #7 (Phase B round 4, **best**) | 5-way blend (+seed-2024 LGB) + snap | **340.59891** |

Improvement over baseline: 341.40782 − 340.59891 = 0.80891 (~0.24% relative).
Phase B improvement over previous best: 340.71180 − 340.59891 = 0.11289.

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

## Phase B iteration rounds (2026-07-03 evening, exp #4–#7)
One change per round, same 5-fold KFold seed 42; stopped after round 4 (protocol max, gains decelerating).

| Round | Script | Change (vs previous best config) | OOF MAE | Verdict |
|-------|--------|----------------------------------|---------|---------|
| 1 | `train_v3.py` (exp #4) | Optuna fold-0-proxy tuned LGB (50-trial recipe from experience.md) **replaces** original LGB | 340.82694 | ✗ worse — tuned LGB better solo (342.02154→341.68775) but blend degraded (340.75961→340.95316): diversity loss |
| 2 | `train_v4.py` (exp #5) | keep original LGB **and add** tuned LGB as 4th base model, 4-way simplex grid | **340.62702** | ✓ new best (weights LGB 0.35 / LGB_TUNED 0.2 / XGB 0.2 / CAT 0.25) |
| 3 | `train_v5.py` (exp #6) | nested OOF isotonic calibration of blend output before snap | 340.62702 (fallback) | ✗ isotonic itself 346.99717 — badly worse, auto-rejected; result = round-2 output |
| 4 | `train_v6.py` (exp #7) | add seed-diversified LGB (identical params, seed 2024) as 5th base model | **340.59891** | ✓ new best (weights 0.2/0.15/0.2/0.25/0.2; snap 340.65207→340.59891) |

Key Phase B lessons:
- A better-tuned single model can *hurt* the blend if it replaces a diverse member — s3e7's "don't re-tune
  the second model" lesson generalizes: tuned variants should be **added** to the pool, not swapped in.
- Isotonic calibration of a GBDT blend on MAE is strongly negative here (+6.3 MAE) — the blend is already
  well-calibrated in the median sense; monotone recalibration only adds variance.
- Seed bagging (5th member, seed 2024) gave a small but real gain (340.62702 → 340.59891).
- OOF/test prediction matrix cached in `data/oof_v4.npz` for cheap future blend experiments.

## Next ideas (not yet tried)
- Target transform (log or Box-Cox) given the wide yield range, then evaluate MAE in original units.
- Quantile regression heads or per-clonesize-bucket separate models (clonesize is a strong discrete driver).
- More seed bags (3–5 seeds per model family) — round 4 suggests small further gains available.

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e14/scripts/eda.py
uv run python3 competitions/playground-series-s3e14/scripts/train.py     # iteration 1
uv run python3 competitions/playground-series-s3e14/scripts/train_v2.py  # iteration 2
uv run python3 competitions/playground-series-s3e14/scripts/train_v3.py  # Phase B round 1 (Optuna, exp #4)
uv run python3 competitions/playground-series-s3e14/scripts/train_v4.py  # Phase B round 2 (exp #5; writes data/oof_v4.npz)
uv run python3 competitions/playground-series-s3e14/scripts/train_v5.py  # Phase B round 3 (isotonic, exp #6; needs oof_v4.npz)
uv run python3 competitions/playground-series-s3e14/scripts/train_v6.py  # Phase B round 4 (exp #7, best; needs oof_v4.npz)
```
No Kaggle submission was made (no credentials in this environment).

## Appendix — Tree search with ensemble node space (Phase C-2b, 2026-07-04)

ERA-inspired candidate-tree search (`tree_search/run_s3e14.py` + `tree_search/eval_s3e14.py` +
`tree_search/harness.py`), second prototype run. Innovation vs the s3e9 first run: the node
space has TWO kinds — `solo` (one model config; its OOF/test matrix is cached to
`tree_search/cache_s3e14/` at eval time) and `blend` (Dirichlet- or grid-simplex weight search
over cached member OOF matrices + snap-to-grid, **no retraining → <1 s per node**). This
directly fixes the s3e9 verdict that a single-model-only node space cannot reach where linear
iteration won. Same 5-fold KFold seed 42 folds as all linear-iteration scripts (verified: solo
seeds reproduce STATUS numbers to 5 decimals — 342.02154 / 342.21778 / 343.60196 / 341.68775).

### Headline result
| | Linear iteration (Phase A+B) | Tree search (this run) |
|---|---|---|
| Best OOF MAE | 340.59891 (exp #7, 5-way blend) | **340.52635** (node #11, 6-way blend) |
| Evaluations to reach its best | ~7 experiments × 3–5 model trainings each | 12 evaluations (8 solo trainings + 4 blend evals) |
| Evaluations to match/beat 340.59891 | — | **9** (node #8, 4-way blend, 340.59485) |
| Total training wall time | ~4 rounds × 3–6 min (Phase B alone) | **593 s (9.9 min)** for all 20 nodes |
| Backtracks | n/a (linear) | 2 (plateau rule, both after best was found) |

Tree best **340.52635 beats linear best 340.59891 by 0.073** (~0.02% relative — small but real
on this metric scale, and 2.6× larger than linear Phase B's final round-4 gain of 0.028).

### Winning node (#11)
6-way blend of cached solo OOFs: root LGB (#0) + XGB (#2) + CAT (#1) + Optuna-tuned LGB (#3) +
seed-2024 LGB (#4) + **18-feature trimmed LGB (#5)** — Dirichlet weight search
(w ≈ 0.11/0.19/0.24/0.14/0.18/0.15), raw 340.64596 → snap 340.52635. The 6th member (drop the
3 remaining raw temp-range cols on top of the 21-feature set; solo 342.02283, i.e. no solo
gain over root) was never tried by linear iteration — its value is pure blend diversity,
exactly the kind of node only reachable once the tree can expand into ensemble space.

Sanity anchor: node #10 (grid_simplex method-swap probe on the 5-way pool) reproduced linear
exp #7 **exactly** — raw 340.65207, snap 340.59891, weights 0.2/0.2/0.25/0.15/0.2 — confirming
the two runs are fold-for-fold comparable, and isolating the Dirichlet search (2×3000 samples
+ concentration refinement) as the source of the finer-weight edge (#9: 340.54904 vs
#10: 340.59891 on identical members).

### Score-vs-evaluations curve (global best after each evaluation)
```
eval  1 (root solo LGB)      342.02154
eval  4 (tuned-LGB seed)     341.68775   <- best any solo node ever reaches
eval  8 (3-way blend seed)   340.69300   <- first blend node
eval  9 (4-way +tunedLGB)    340.59485   <- BEATS linear best (340.59891)
eval 10 (5-way +seedbag)     340.54904
eval 12 (6-way +FEAT)        340.52635   <- final best; evals 13-20 never improved on it
```
Node-efficiency verdict: linear iteration needed 2 Phase-A iterations + 4 Phase-B rounds (each
a full multi-model pipeline) to reach 340.59891; the tree passed that score at evaluation 9 of
20 (~5.7 min wall), and each ensemble improvement after the solo pool existed cost <1 s.

### Search trace (20 evaluated nodes: 12 solo / 8 blend, 0 failed)
- Root + 6 solo seeds (CAT/XGB/LGBTUNED/SEEDBAG/FEAT/REG) populate the OOF cache (~5 min).
- BLEND lineage seeded (3-way mirror of exp #3) → immediately best → harness keeps pulling on
  it: +tuned (→340.59485), +seedbag (→340.54904), method-swap probe (grid_simplex, 340.59891),
  +FEAT (→**340.52635**), remove-weakest probe (340.63637 — confirms even the lowest-weight
  member pulls its weight), +REG 7-way (340.60446, REG weight ~0.003 — the weight search
  correctly rejects the weak solo), fallback re-run (unchanged).
- Backtrack #1 at node 14 (BLEND: 3 non-improving children) → LGBTUNED lineage: seed-7 variant
  341.71805, finer-lr 341.68861, seed fallback 341.95945 → backtrack #2 at node 17 → FEAT
  lineage: 2 further trims (342.28883 / 342.26410, both worse) → 20-node budget reached.
- Both backtracks fired *after* the global best was already found — the plateau rule correctly
  spent the residual budget probing solo space for new diversity sources rather than
  over-expanding a saturated blend lineage.

### Verdict vs the s3e9 run
The C-2b hypothesis is confirmed: with ensemble nodes in the search space, tree search matched
linear iteration's Phase-B result in 9 evaluations and then exceeded it with a blend
composition (6-way incl. a differently-featured member) that the linear process never
proposed. Caching solo OOFs at eval time is the enabling mechanism — blend nodes are ~50×
cheaper than solo nodes, so ensemble-space exploration is nearly free once the pool exists.
Full tree: `experiments_tree.json`; OOF caches regenerable via the run script, gitignored.

Reproduce: `uv run python3 tree_search/run_s3e14.py` (delete `experiments_tree.json` and
`tree_search/cache_s3e14/` first for a from-scratch run; the script resumes otherwise).

## Appendix — Phase F-2 harness v3 validation run (2026-07-04)

**Validation question** (same as the s3e7 F-2 run): does harness_v3's DEFAULT automatic
policy run end-to-end with no regression vs the v1-proto tree, and does the policy
machinery do real work? **Answer: yes on both. New best OOF MAE 340.35572 (−0.171 vs the
v1-proto tree's 340.52635, −0.243 vs linear's 340.59891), and again the single biggest
move came from the phase machine's mandatory explore-burst kitchen-sink mega-blend.**

Built: `tree_search/run_s3e14_v3.py` (driver; `eval_s3e14.py` reused byte-for-byte
unmodified) → `experiments_tree_v3.json` (prior `experiments_tree.json` untouched).
Root + the 6 v1-proto solo seeds (nodes 0–6) reused from `cache_s3e14/` via the
digit-verify path (fresh MAE recompute from cached OOF must equal the stored score to
5dp; root asserted == 342.02154). Blend nodes evaluated fresh via v3's k=800 +
coordinate-ascent search (never v1's k=3000 hand-rolled search), snap-to-grid INSIDE
the metric_fn for every candidate weight vector.

### v3 vs v1-proto vs linear 對照
| | OOF MAE | evals | best found at |
|---|---|---|---|
| Linear iteration (exp #7, 5-way blend) | 340.59891 | ~7 experiments | — |
| Tree v1-proto (harness v1, C-2b) | 340.52635 | 20 | eval 11 |
| **Tree v3 (this run)** | **340.35572** | 60 (cap) | **eval 44** |

Curve: beat linear at eval 9 (4-way, 340.55371), beat the v1-proto tree at eval 11
(5-way, 340.51119 — one eval later than v1 needed, on the same member set), then the
exploit phase ground down to 340.45150 by eval 16 (7-way minus XGB, node #15), 22 more
exploit evals bought nothing, and the burst's 34-member mega-blend (#44, eval 44) took
it to **340.35572** (raw 340.57853 → snap 340.35572). Unlike s3e7 (where the weight
search zeroed 29/38 members), here 28/34 members kept weight >0.005 — including 4 CAT
variants (0.22 combined), both burst survivors EXPL_CATDEEP (solo 345.16, w=.051) and
EXPL_REGDEEP (solo 341.99, w=.045): on this noisier target, breadth-of-averaging itself
is the signal, one more confirmation that blend contribution ≠ solo score.

### Policy behavior (v3 features, honestly scored)
- **Phase machine**: exploit evals 1–38 (BLEND exhausted its mutation space first —
  forced backtrack — then all 8 solo lineages 3-strike-plateaued in sequence);
  `explore_burst` auto-fired at eval 38 → 5 long-shot solos + mega-blend; **stopped at
  the hard cap 60/60** (`stop_reason`: "hard budget cap reached"),
  evals_since_burst_improve = 16 < patience 20 at the cap. Idle tail 16 evals ≤ 20
  target. Same shape as s3e7: the burst improving the global best resets the patience
  counter, so the patience stop can only fire when a burst FAILS — on both F-2 comps
  the burst paid off, and the numeric backstop (cap) is what ended the run. The
  patience path itself was verified end-to-end only in a small-budget driver smoke
  (stop_reason "2 evals without improvement post-burst (patience=2)") — in-the-wild
  evidence for it is still pending a comp whose burst is a dud.
- **Burst payoff**: 340.45150 → 340.35572 (−0.096), 100% of the post-exploit gain; the
  5 long-shot solos were all solo-worse (DART catastrophically so — see ledger) yet 2
  of them entered the winning blend.
- **Boundary-push (feature 4)**: `boundary_candidates(LGBTUNED params vs its train_v3
  Optuna box)` returned [] — correct, no tuned param sits within 5% of its box edge
  here (nearest: reg_lambda at ~13% in log-space). The check ran and honestly found
  nothing; s3e7's F-2 run is where it fires for real (max_depth on the box edge).
- **Dedup-consumes-budget / reopen-blend / cost guard**: none fired (0 rejections — the
  authored queues never re-proposed a duplicate; no solo ever beat a blend; the
  34-member MAE blend cost 1.6s ≪ 45s). All three verified live on s3e7 or in unit
  tests instead.
- **Prior usage**: informed 4 (win 75% — the P9/P2 add-to-pool moves) vs uninformed 5
  (win 20%); consistent with the D-3 finding that priors excel at ensemble mechanics.

### Honest operational ledger
- Sum of eval wall: 2338s (39 min) over 60 evals; the first process hit the driver's
  own 35-min wall guard at 59/60 and a 2-minute resume recorded the final eval + the
  hard-cap stop. The overshoot is DART's fault: 6 DART evals at 107–140s each (vs
  ~40-70s typical) — DART's MAE objective run produced garbage solos (6144–6544 OOF
  MAE, ~18× worse than baseline; dart + early-stopping-disabled + regression_l1 do not
  mix at this n_estimators) and burned ~12 min for zero value. Backlog note: burst
  seeds need a per-seed wall/score sanity gate.
- Nodes 0–8 (root, 6 seeds, 3-way + 4-way blends) were first evaluated during a
  pre-flight smoke of this driver's plumbing (node #8's mutation string carries a
  leftover "[test]" tag) and the real run resumed on top — every config and score is a
  genuine evaluation through the identical code path (reuse digit-verified against the
  v1-proto tree; blend search deterministic at seed=42).
- Same OOF-weight-overfit caveat as every tree run: 34-dim weights fit on the full OOF,
  no nested validation; the burst-beats-exploit direction is robust, the 5th decimal
  is not.

Reproduce: `uv run python3 tree_search/run_s3e14_v3.py` (resumes from
`experiments_tree_v3.json`).
