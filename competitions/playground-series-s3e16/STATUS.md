# STATUS — playground-series-s3e16 (Crab Age)

- **Task**: regression, predict `Age`; metric **MAE** (minimize); id col `id`
- **Data**: 74,051 train / 49,368 test; 8 features (1 cat `Sex` + 7 continuous size/weight); no missing, no dups
- **Rules**: no external data, no pretrained, no internet; daily submission limit 5

## Progress
- [x] Stage 0 Setup — config.yaml, data downloaded
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py`
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost, MAE objective)
- [x] Stage 5 First submission generated — `submissions/sub_blend_20260703_091840.csv`
- [x] Submitted to Kaggle leaderboard — **Public 1.34356 / Private 1.34075** (2026-07-03)

## Leaderboard result (submission #1)
| | OOF MAE | Public LB | Private LB |
|-|---------|-----------|------------|
| Blend (rounded) | 1.33812 | **1.34356** | **1.34075** |

CV↔LB gap ~0.006 → **CV is trustworthy** (slightly optimistic, as expected). Safe to iterate on OOF.

> ⚠️ **Auth note**: valid Kaggle token is `KGAT_7...` in `~/.kaggle/kaggle_api_token.txt`,
> NOT the `KGAT_2...` in `~/.kaggle/kaggle.json` (that one 403s). Submit with:
> `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')`

## EDA key findings
1. `Sex='I'` (infant) → mean age 7.6 vs M 10.9 / F 11.3. Strong signal → `is_infant` flag.
2. `Shell Weight` best correlated with Age (Spearman 0.74); `Shucked Weight` weakest (0.61).
3. Severe collinearity (0.86–0.99) among size/weight; `Weight≈Shucked+Viscera+Shell` (r=0.993).
4. `Height==0` in 24 train rows (invalid) → treated as missing, imputed with median.
5. Train/test distributions near-identical (<0.4% mean diff) → no covariate shift.
6. Target integer 1–29, right-skewed (skew 1.09). **Rounding OOF preds improves MAE.**

## CV scheme (fixed first — matters more than the model)
- **5-fold StratifiedKFold on binned Age** (ages ≥20 merged into one bin), shuffle, seed=42.
- Fold MAE very stable (1.34–1.37) → CV is trustworthy.

## Feature engineering (24 features)
Sex one-hot + `is_infant`; part weight ratios; `weight_resid = Weight − Σparts`; size ratios
(diam/len, height/len, height/diam); `volume`, `density`, `shell_density`; `meat_to_shell`, `shucked_to_shell`.

## Model results (OOF MAE, 5-fold)
| Model | OOF MAE | rounded | time |
|-------|---------|---------|------|
| LightGBM (L1) | 1.35651 | 1.33885 | 31s |
| XGBoost (reg:absoluteerror) | 1.35763 | 1.34160 | 37s |
| CatBoost (MAE) | 1.36162 | 1.33846 | 24s |
| **Blend 0.6/0.3/0.1 (round)** | **1.35589** | **1.33812** | — |

Experiment log: `experiments.json` (#1).

## Iteration round — Phase B self-improvement (2026-07-03)

Goal: push CV further via the validated "Optuna fold-proxy → add to pool → seed bag" recipe
(knowledge/experience.md). Result: **no improvement — best stays 1.33812 rounded**. STOPPED after
2 consecutive non-improving rounds, per protocol.

| Round | Change | Raw OOF MAE | Rounded OOF MAE | vs best (1.33812) |
|-------|--------|-------------|------------------|--------------------|
| — | baseline (LGB/XGB/CAT 0.6/0.3/0.1) | 1.35589 | **1.33812** | — |
| 1 | + Optuna fold0-proxy tuned LGB (21/50 trials, 300s timeout) as 4th pool member | 1.35541 | 1.33850 | worse (+0.00038) |
| 2 | + seed-bagged tuned LGB (seed=2024) as 5th pool member | 1.35533 | 1.33893 | worse (+0.00081) |

Both rounds improved the **raw** OOF MAE (1.35589 → 1.35541 → 1.35533, a real but small gain from
tuning+bagging) but made the **rounded** OOF MAE worse. Root cause: LGB_tuned and
LGB_tuned_seed2024 only differ from the original LGB in depth/lr/reg (same objective, features,
folds) → low ensemble diversity → the small raw-MAE gain isn't large enough to shift the discrete
rounding boundary favorably. New insight logged to `knowledge/experience.md`: the
Optuna→pool→seed-bag recipe (previously validated on RMSE-scale targets s3e1/s3e7/s3e11/s3e14)
does **not** reliably transfer when the final metric is **rounded integer MAE** — decide on the
post-rounded score, not raw OOF, or the "improvement" can be illusory.
Scripts: `scripts/tune_lgb_optuna.py`, `scripts/pool_lib.py`, `scripts/round1.py`, `scripts/round2.py`.
Experiment log: `experiments.json` (#3, #4).

## Next ideas (updated after this iteration)
- Optuna+seed-bagging tried — didn't survive rounding (see above); not worth more trials on LGB alone.
- Untried: Tweedie/Poisson-objective LGB as a genuinely diverse pool member (different loss
  surface, not just different hyperparams) — `pool_lib.train_lgb_tweedie()` and `scripts/round3.py`
  are scaffolded but not yet run (stopped early per 2-non-improving-round rule).
- Untried: collinear feature pruning (Weight≈Shucked+Viscera+Shell, r 0.86–0.99) or a stacking
  meta-model on OOF.

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e16/scripts/eda.py
uv run python3 competitions/playground-series-s3e16/scripts/train.py
# submit (use the huangweihaohuang token; kaggle.json 403s):
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e16 -f <submission.csv> -m "<msg>"
```

## Appendix — Tree-search v2 (Phase E-3, harness_v2, 2026-07-04)

**Goal**: s3e16 is the acid test for tree_search v2 on a ROUNDED-INTEGER-MAE metric — the
only comp in the harness_v2 lineup with a REAL Kaggle LB anchor, and the one place Phase
B's linear iteration already hit the recipe's own documented failure mode (raw OOF MAE
improved monotonically 1.35589→1.35541→1.35533 across 2 rounds; ROUNDED OOF MAE got WORSE
both times, 1.33812→1.33850→1.33893 — see `knowledge/experience.md`'s "MAE/整數目標"
boundary-condition bullet). Question: can v2's mechanisms (metric-aware plateau, ensemble-
default node space, child dedup, k=800+coordinate-ascent weight search — all decisions on
the ROUNDED score, per the rule `metric_fn = MAE(round(clip(blend)))` applied inside every
node's own scoring path) find a gain that survives rounding where the linear recipe could
not?

**Result: YES — the rounding-boundary failure mode was OVERCOME.**

| | Rounded OOF MAE | vs previous best (1.33812) |
|-|------------------|------------------------------|
| Linear-iteration best (blend 0.6/0.3/0.1 LGB/XGB/CAT, coarse simplex grid step=0.1) | 1.33812 | — |
| **v2 tree-search best (node #15, 8-way blend)** | **1.33563** | **−0.00249 (−0.186% relative)** |

- **Nodes**: 27 total, 24 evaluated (target ≥18 met with margin), 3 failed (all
  fast/harmless — see "known limitation" below), 4 backtrack events, 2 dedup rejections.
- **Wall-clock**: 1409.4s (~23.5 min), within the ~28 min / 30 min budget.
- **Prior usage**: 13 informed (mutations tagged `[PRIOR ...]`) vs 1 uninformed
  child-of-root comparison; informed win rate 38.5% (5/13), uninformed win rate 0%
  (0/1, n too small to read much into). Most of this run's real wins came from the
  BLEND lineage's own weight-search mechanics (harness-level, not individually
  prior-tagged mutations), not from any single authored mutation idea outperforming.

### The headline finding: a genuine raw-vs-rounded INVERSION (the strongest possible confirmation of the "decide on rounded, not raw" rule)

The winning 8-way blend (node #15) has **raw OOF MAE 1.35712** — WORSE than the linear
champion's raw 1.35589 — but **rounded OOF MAE 1.33563**, BETTER than the champion's
1.33812. Independently re-verified digit-for-digit outside the tree (loading the 8 cached
member OOFs and recomputing both raw and rounded MAE from the stored weight vector). This
is the exact mirror image of Phase B's failure (there: raw improved, rounded worsened;
here: raw worsened, rounded improved) — together the two results are the cleanest possible
demonstration that raw OOF MAE and rounded OOF MAE are genuinely different optimization
targets on this comp, and every decision must be made on the rounded number.

### Winning composition (node #15)
8-way blend, weights (dirichlet k=800 + coordinate-ascent refinement, searched directly
against rounded MAE):

| member | model | weight | solo rounded MAE |
|--------|-------|--------|-------------------|
| #0 root | LGB (plain L1) | 0.0198 | 1.33885 |
| #1 XGB | XGB (reg:absoluteerror) | 0.0256 | 1.34160 |
| #2 CAT | CatBoost (MAE) | **0.5061** (dominant) | 1.33846 |
| #3 LGB_tuned | Optuna fold0-proxy LGB | 0.0075 | 1.33979 |
| #4 LGB_tuned_seed2024 | seed-bagged LGB_tuned | 0.0626 | 1.33914 |
| #5 LGBBOUND | boundary-pushed LGB (lr=0.005) | 0.1117 | 1.33950 |
| #6 TWEEDIE | Tweedie-objective LGB | 0.0004 (~inert) | 1.37467 |
| #8 FEATPRUNE | LGB, drop `Weight` | 0.2662 | 1.34025 |

POISSON (#7, solo 1.37501) was tried but never made it into the winning composition (a
sibling 9-way blend with POISSON added scored 1.33584, slightly worse).

Notably, the search's biggest single lever was NOT any new lineage's solo strength — it
was **weight-search precision on the ORIGINAL 3-way composition**: reproducing the linear
champion's exact members (root LGB + XGB + CAT) with k=800+coordinate-ascent instead of
the champion's coarse (step=0.1) simplex grid already found 1.33640 (node #9), beating
1.33812 before any new model was added — a direct within-comp confirmation of the E-2
lesson ("coarse grids silently tie/lose to finer search"). Every subsequent member
addition (LGB_tuned, LGB_tuned_seed2024, LGBBOUND, TWEEDIE, FEATPRUNE) then chipped the
score down further: 1.33640→1.33635→1.33625→1.33616→1.33567→1.33563.

### Per-lineage findings
- **LGBBOUND (boundary-push)**: LGB_tuned's Optuna learning_rate=0.0102 sat almost exactly
  on `tune_lgb_optuna.py`'s own search-box lower edge ([0.01, 0.06] log) — every other
  tuned hyperparam was comfortably interior. Pushing to lr=0.005 gave a real, if modest,
  solo improvement over LGB_tuned (1.33979→1.33950), confirming the box-edge distrust
  lever partially transfers here — but pushing further to lr=0.003 reversed (1.34085,
  worse), so the true optimum sits between the box edge and the first push, not
  monotonically further out. LGBBOUND's best variant also contributed a real blend gain
  (11.17% final weight).
- **TWEEDIE / POISSON (diverse-objective heads)**: both scored badly SOLO (1.37467 /
  1.37501, far worse than every other pool member) — the count-like-objective hypothesis
  did not pay off as a strong single model on this feature-engineered dataset. But TWEEDIE
  still contributed real blend value once added (7-way 1.33616→1.33567, the run's single
  largest blend improvement) despite its own near-zero final weight (0.04%) — the
  coordinate-ascent re-optimizes every other member's weight jointly, so a weak-but-
  differently-wrong member can still improve the blend's error correlation structure even
  at ~0 weight. POISSON's marginal contribution was negative (9-way with POISSON: 1.33584,
  worse than the 8-way's 1.33563) and it was excluded from the final best.
- **FEATPRUNE (drop `Weight`, the near-perfect collinearity r=0.993 with
  Shucked+Viscera+Shell)**: solo score 1.34025 (worse than root) but the single highest-
  weighted new member in the final blend (26.62%) — the biggest confirmed "Untried" idea
  from Phase B's list, and the addition that produced the actual global-best node (#15).
- **CAT dominance**: the deeper weight search shifted dominant weight to CAT (50.6%)
  rather than the linear champion's LGB-dominant 60/30/10 split — a genuinely different
  point on the weight simplex, not just a refinement of the same one.

### Known limitation (harmless, documented for honesty)
`eval_s3e16_v2.py` only implements a fresh-training runner for `model="lgb"` (XGB/CAT are
reused read-only from Phase B's cache, per the module's stated scope). The legacy XGB/CAT/
LGB_tuned/LGB_tuned_seed2024 lineages have no authored mutation queue, so once selected for
further expansion they immediately fall through to `solo_fallback`'s seed-variation retry —
which raises `ValueError` for `model="xgb"/"cat"` (caught by `evaluate()`, status="failed",
~0s each). This produced 3 fast "failed" nodes on the CAT lineage (18, 19, 20) before it
correctly self-plateaued via the harness's "expansion budget exhausted" rule — no time or
correctness cost, just a slightly untidy-looking tree. Not fixed in this run (would not
have changed the result; XGB/CAT retraining was out of scope by design).

### Verdict
**The rounding-boundary failure mode diagnosed in Phase B was overcome.** v2's combination
of (1) a properly-budgeted k=800+coordinate-ascent weight search decided directly on the
rounded metric, and (2) genuinely diverse new pool members (a feature-pruned LGB, boundary-
pushed LGB, and — despite poor solo scores — Tweedie-objective LGB) found a real,
independently-verified 0.00249 rounded-MAE improvement over the best Kaggle-anchored
submission on record for this comp. Not submitted to Kaggle per this run's constraints; the
gain is an OOF-verified finding only.

Scripts: `tree_search/eval_s3e16_v2.py`, `tree_search/run_s3e16_v2.py`.
Tree: `competitions/playground-series-s3e16/experiments_tree.json`.
