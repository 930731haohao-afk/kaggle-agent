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
- [x] **Phase D-3 tree-search v2 sweep (2026-07-04)** — `tree_search/run_s3e7.py`, new best
      OOF **0.900242** (+0.000349 over linear best; see appendix at end of this file)
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

## Appendix: Phase D-3 tree-search v2 sweep (2026-07-04)

**Sweep question**: can harness v2 (ensemble-default node space + experience-library
priors + adaptive plateau + child dedup) match or beat the linear-iteration best
(0.899893) in fewer evaluations than linear iteration took rounds (6 experiments,
exp #1–#6)? s3e7 was flagged as the HARDEST sweep case (linear squeezed only +0.0005
total; suspected near-zero headroom). **Answer for s3e7: split. Evals-to-match: NO —
matching took 8 evaluations (> linear's 6), and the "match" was literally the blend
seed reproducing linear's own 3-model trio. Headroom: YES — the tree then found a real
+0.000349 beyond the linear ceiling (0.900242 by evaluation #13, wall 405.8s), via two
levers the linear run never tried (seed-bagging the tuned LGB; a deliberately-diverse
deep XGB).** The suspected near-zero headroom was real for solo models (no solo mutation
beat the root) but NOT for the ensemble pool.

Built: `tree_search/eval_s3e7.py` (solo: LGB/XGB/CAT on the iter2 trimmed 25-feature set
via scripts/features.py unmodified, StratifiedKFold(5, seed=42) — digit-for-digit
reproduction verified for all 3 known configs before searching: root LGB_tuned 0.899215,
XGB_hand 0.898765, CAT_hand 0.896709, all exact; the blend seed then reproduced exp #6's
0.899893 with weights 0.5397/0.4021/0.0582 ≈ the linear 0.54/0.40/0.06) +
`tree_search/run_s3e7.py` (harness_v2 driver). Root = linear winner's strongest solo
(exp #4 Optuna fold-0-proxy-tuned LGB). OOF cache: `tree_search/cache_s3e7/` (gitignored).

### Node/backtrack/dedup summary
- **22 evaluated nodes** (13 solo / 9 blend), 22 total (0 failed), wall=405.8s.
- **3 backtracks**: 1 forced (BLEND lineage's mutation space genuinely exhausted — queue
  done and every remaining candidate a duplicate) + 2 genuine 3-strike plateaus (SEEDBAG
  at node #18, FEATPRUNE at #21). tie_rate peaked at 0.10 < 0.15 threshold, so the
  adaptive-plateau discretization branch never fired (AUC is continuous — as predicted).
- **Dedup**: 0 rejections in the final run — but only because the FIRST full run of this
  driver exposed a live-lock the dedup safety-net alone cannot prevent: after the blend
  queue exhausted, the old fallback re-proposed the same "add member #4" config 73
  consecutive times (every one correctly rejected as a duplicate of node #14, zero
  compute wasted) until the iteration safety cap killed the run at 15/22 nodes.
  **Harness-level lesson: rejection-only dedup is necessary but not sufficient — the
  mutation PROPOSER must consult `find_dup` before proposing, and blend member lists
  must be hashed order-insensitively (same member SET == same search space).** Both
  fixes are in run_s3e7.py (`blend_fallback` pre-checks every candidate; `strip_result`
  sorts members), plus a forced-backtrack path when a lineage's proposal space is
  genuinely exhausted.

### Best vs linear, evaluations-to-match/beat
| | OOF AUC | evaluations |
|---|---|---|
| Linear-iteration best (exp #6, prob 0.01-grid blend) | 0.899893 | 6 experiments |
| Tree v2: node #7 (BLEND seed = linear's trio reproduced) | 0.899893 | 8 (match, not fewer) |
| Tree v2: node #8 (+SEEDBAG member) | 0.900054 | 9 (first genuine beat) |
| Tree v2: node #13 (global best: 5-way rank blend, CAT removed) | **0.900242** | **13** |

Winning chain: #7 trio 0.899893 → #8 +LGB_seed2024 0.900054 → #9 +FEATPRUNE 0.900057 →
#11 +XGB_deep 0.900224 → #12 prob→rank 0.900238 → #13 −CAT(w=0.007) **0.900242**.
Final members/weights (rank space): LGB_tuned .057 / XGB_hand .134 / LGB_seed2024 .506 /
LGB_featprune .014 / XGB_deep .288.

### What actually moved it — and what didn't (honest ledger)
- **Seed-bagging the Optuna-tuned LGB** (seed 42→2024, untried by linear): solo 0.899448
  even beat the root 0.899215; +0.000161 as blend member. The 4-comp "add-to-pool +
  seed-bag" recipe transfers to s3e7; the linear run simply stopped before this step.
- **Deliberate-diversity deep XGB** (depth 8, lr 0.05 — pushed AWAY from the tuned-LGB
  optimum in direct response to exp #5's diversity-loss rejection): worst-but-one solo
  (0.898199) yet the second-largest single blend gain (+0.000167, 0.27–0.29 weight).
  Confirms exp #5's lesson from the constructive side: blend contribution ≠ solo score.
- **Feature pruning did NOT transfer to s3e7** (s3e3's winning lever, this brief's
  specific hypothesis): dropping the 2 measured-weakest engineered features
  (prior_cancel_rate gain-importance 0.000, has_children 59.5) LOST to the intact root
  solo (0.899146 vs 0.899215), and both further-prune and restore-one mutations stayed
  below root. The 8-feature iter2 set appears already pruned to its optimum — but the
  pruned variant still earned 1–8% blend weight as a cheap diversity member.
- **Regularization nudges: all 5 lost** (REGNUDGE seed + queue, SEEDBAG reg push,
  XGBHAND nudge never reached). 42k rows is not the <10k regime the
  regularize-over-capacity prior is validated on.
- **rank vs prob stayed noise-level** (±0.00002 both directions across #12/#15),
  re-confirming P18 — though this time rank happened to sit on top.
- **Removing the near-zero-weight CAT member cost nothing and gained +0.000004** (P16
  confirmed); re-adding REGNUDGE (#14) and swapping back to prob (#15) both lost.

### Prior-usage log (idea-injection experiment)
`suggest_priors({"metric":"auc","tags":["特徵","共線性","ensemble","optuna"]})` returned
20 bullets (P0–P19; the Optuna/ensemble sections are rich for this comp family). Among
the 14 search-loop mutations (first-gen seeds excluded, no in-lineage parent):
- **8 prior-informed, win rate 5/8 = 62.5%** (P2 seed-bag ×2: 1W1L; P9 add-to-pool ×3:
  2W1L; P18 rank-swap 1W; P16 remove-weakest 1W; P3 further-prune 1L).
- **6 uninformed, win rate 1/6 = 16.7%** (fallbacks and hand-nudges).
- Unlike s3e3 (where informed tied uninformed at 14.3% and the best move came from the
  comp's own next-ideas), on s3e7 the experience-library priors materially outperformed:
  every blend-layer prior (P9/P16/P18) did exactly what its evidence line predicted, and
  the two biggest gains (#8, #11) were P2/P9-shaped moves. Small n; but two sweeps in,
  the pattern is "priors excel at ensemble mechanics, comp-specific feature insight
  still has to be earned locally" (s3e7's P3 pruning transfer failed, s3e3's did too in
  reverse — its winner was local, not transferred).

## Appendix: Phase F-2 harness v3 validation run (2026-07-04)

**Validation question**: does harness_v3's DEFAULT automatic policy (budget/phase machine
+ auto-stop + dedup-consumes-budget + reopen-blend trigger + boundary-push + k=800+
coordinate-ascent blend search + blend-cost guard) run end-to-end in the wild with no
regression vs the v2 sweep — and does the policy machinery do real work? **Answer: yes on
both. New best OOF 0.900455 (+0.000213 over v2's 0.900242, +0.000562 over linear), and
the ENTIRE gain beyond the v2-reproduction plateau came from the phase machine's
mandatory explore burst (kitchen-sink mega-blend), the exact mechanism E-5 predicted.**

Built: `tree_search/run_s3e7_v3.py` (driver; `eval_s3e7.py` reused byte-for-byte
unmodified) → `experiments_tree_v3.json` (prior `experiments_tree.json` untouched).
Root + the 6 v2 solo seeds + 1 v2 node were reused from `cache_s3e7/` via the
digit-verify path (recomputed AUC from cached OOF must equal the v2 tree's stored score
to 6dp; root asserted == 0.899215 before anything else ran). 44 solos trained fresh.

### v3 vs v2 vs linear 對照
| | OOF AUC | evals | best found at |
|---|---|---|---|
| Linear iteration (exp #6) | 0.899893 | 6 experiments | — |
| Tree v2 (harness_v2, D-3) | 0.900242 | 22 | eval 13 |
| **Tree v3 (this run)** | **0.900455** | 60 (cap) | **eval 46** |

Winning node #48: rank-space Dirichlet(k=800)+coordinate-ascent blend over the FULL
38-member solo pool (injected as the explore burst's kitchen-sink long-shot, node #44
prob-space 0.900432 → #48 rank swap 0.900455). Weight search zeroed 29/38 members; the
9 survivors: SEEDBAG .358, XGBDIV family .459 (4 nodes), **EXPL_BOUND2 .114 (the
boundary-push-derived depth-2 LGB — solo only 0.897541, yet 3rd-largest weight)**, root
.038, FEATPRUNE-child .018, XGBHAND-child .013. Confirms (at 38-member scale) the v2
lesson that blend contribution ≠ solo score. Honest caveat shared with every prior tree
run: weights are fit on the full OOF (no nested validation), so ~0.0002-level gains
carry OOF-weight-overfit risk; the direction (burst mega-blend > hand-grown blend) is
the robust part, the 6th decimal is not.

### Policy behavior (the v3 features, honestly scored)
- **Phase machine**: exploit evals 1–39 (all 9 first-gen lineages 3-strike-plateaued in
  sequence, pre-burst best 0.900054 = v2's 4-way SEEDBAG blend reproduced at eval 11);
  `explore_burst` auto-fired at eval 39 → 5 long-shot solos (DART/extra-trees/deep-CAT/
  lossguide-XGB/depth2-LGB, all solo-worse as expected) + the mega-blend (the winner);
  stopped at the **hard cap 60/60** (`stop_reason`: "hard budget cap reached"),
  `evals_since_burst_improve`=13 < patience 20 at cap — the burst improving the best at
  evals 45/46 reset the patience counter, so the numeric backstop did its job. Idle
  tail: 14 evals (46→60), within the ≤~20 target.
- **Burst payoff**: +0.000401 over the pre-burst plateau (0.900054 → 0.900455). Without
  the burst this run would have ended a v2-regression; with it, a v2-beat. Second comp
  (after E-5 s3e3-scale) where the mandatory burst supplied every post-plateau gain.
- **Boundary-push (feature 4)**: `boundary_candidates()` flagged the root's max_depth=3
  sitting on its Optuna box's own low edge [3,12] → auto-seeded BOUNDARYPUSH (depth 2,
  solo 0.896441, worse) and the burst's paired-compensation variant EXPL_BOUND2
  (depth 2 + lr 0.1, solo 0.897541) — which earned 0.114 weight in the winning blend.
  The mutation type paid rent via diversity, not solo score.
- **Dedup-consumes-budget (feature 2)**: fired 3× (nodes #45–47, `status=failed`
  placeholders burned under the root after 2-in-a-row dedup rejections of replayed
  burst-seed proposals during a crash-resume). Worked as designed; zero evaluated-node
  budget wasted.
- **Reopen-blend-on-solo-breakthrough (feature 3)**: never fired — correct, no solo node
  ever became global best after blends existed (blends led from eval 10 onward).
- **Cost guard (feature 6)**: never fired — the 38-member AUC blend cost 11.4s < 45s
  threshold. k=800+ascent (feature 5) was the default throughout.
- **Prior usage**: informed 7 (win 28.6%) vs uninformed 3 (33.3%) — small n, no edge
  this run; the two decisive moves (burst injection, mega-blend) were harness-policy
  moves, not experience-library transfers.

### Honest operational ledger (things that went wrong)
- Sum of eval wall over the 60 nodes: 1593s (26.6 min) — within budget. But the run
  needed 3 restarts: (1) a `KeyError` on the burst's mega-blend lineage (driver bug:
  `propose_child` dispatched blend-vs-solo on the literal name "BLEND"; fixed to
  dispatch on the lineage's first-gen node `kind`), (2) module-level driver state
  (`LINEAGE_NAMES`/burst flag/node results) not surviving resume (fixed: node_results
  moved into `search_state`, burst state re-derived from the tree), and (3) one
  EXPL_CATDEEP CatBoost eval (depth 9, bagging_temperature 2.0, seed 4001) hung 28 min
  at 313% CPU — `signal.alarm`-based timeouts cannot interrupt a native fit() that
  never returns to Python bytecode; killed + resumed, the retry completed in 54.8s.
  Lesson for the harness backlog: eval timeouts need a subprocess boundary, not SIGALRM.
- All three fixes are driver-level; `harness_v3.py` itself needed zero changes.

Reproduce: `uv run python3 tree_search/run_s3e7_v3.py` (resumes from
`experiments_tree_v3.json`; delete it + the non-v2 `cache_s3e7/solo_*.npz` entries for
a from-scratch run).
