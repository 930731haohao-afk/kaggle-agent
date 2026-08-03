# Tree Search — the advanced Stage 4 loop

## Contents
1. [When to switch to tree search](#1-when-to-switch-to-tree-search)
2. [Node-space design (by competition type)](#2-node-space-design-by-competition-type)
3. [Budget and stopping rules](#3-budget-and-stopping-rules)
4. [Driver script checklist](#4-driver-script-checklist)
5. [Priors](#5-priors)
6. [Honest reporting rules](#6-honest-reporting-rules)
7. [Quick start](#7-quick-start)

This document is the **advanced Stage 4 loop** that follows [05_evaluation.md](05_evaluation.md) (the linear iteration protocol) — it does not replace it, but takes over once the conditions are met. Evidence: `docs/tree_search_prototype.md` (a full report over 15 runs / 10 competitions) and `docs/scaling_experiment.md` (the Phase E-5 scaling curve).

## 1. When to switch to tree search

**Trigger condition**: Stage 3 already has a baseline solo model and has produced at least **1 blend** (manually or automatically) — i.e. the linear iteration protocol in 05_evaluation.md has run at least its first round. Do not skip linear iteration before this — tree search's node-space needs at least a solo pool and a known blend concept to have something to mutate.

**Why (evidence)**: `docs/tree_search_prototype.md` §10 — over 15 runs / 10 competitions, tree search scores **9 wins, 1 exact tie (s3e9), 0 losses** against the linear-iteration best score. The only v1 loss across three comps (s3e9) was root-caused to a node space structurally missing an ensemble; after v2 added the ensemble-default node space it flipped to an exact tie. In other words, tree search is not "a different way to train" but "systematically automating and exhaustively searching the mutations linear iteration used to find by hand (boundary-push, seed-bagging, kitchen-sink blend)."

**When to still use linear iteration (fallback)**:
- The first iteration round (no baseline + blend yet) — see [05_evaluation.md](05_evaluation.md).
- Tiny data / extremely expensive evaluation, where a ~60-node budget is unaffordable.
- The exploratory EDA phase (tree search is a convergence tool for "once you know what to optimize," not an exploration tool).

## 2. Node-space design (by competition type)

| Competition type | Node-space design | Basis |
|---|---|---|
| General tabular (GBDT-dominant) | **solo + blend dual-track (default)**: a `kind="solo"` node is a single-model configuration (model/params/features); a `kind="blend"` node is a weighted mix of existing solo nodes. This is the v2/v3 default schema, directly applicable in 8 of 10 comps. | `harness_v2.py` node schema; §6 "ensemble-default node space" |
| Structure-dominant (e.g. s3e20: grouped historical means beat GBDT outright) | **structural configs**: `kind="solo"` but `params` are structural knobs (window size, anomalous-year downweighting, joint shift amount) rather than GBDT hyperparameters. First confirm whether the Phase-B residual diagnostic already shows "features carry no structured predictable signal about the target beyond structure" — if so, do not waste budget forcing a GBDT node-space. | `eval_s3e20_v2.py`; §9 E-4 |
| Discretized / post-rounding decision metrics (e.g. QWK-after-rounder, rounded MAE) | metric_fn **must include the postprocessing** (rounding/OptimizedRounder); decisions are always made on the **post-processed** score, never on the raw OOF score — otherwise you hit the "raw improves, rounded gets worse" mirror-failure mode. | §9 E-3 (s3e16): the winning combination had worse raw MAE than the linear champion but better rounded MAE |

Regardless of type, **every node's `config` field must be the canonical/hashable "stored form"** (so the hash the dedup relies on is meaningful) — see the `core(cfg)` convention in `run_s3e7_v3.py`: drop output fields such as `result`/`want_importance`, and sort a blend's `members` before storing.

### Sign convention: every stored `score` is lower-is-better

The harness has exactly one score convention and it is not configurable: **`node["score"]` is always lower-is-better.** `harness.py`'s node-schema docstring states it ("lower-is-better metrics assumed throughout this harness; flip sign before calling in if the comp metric is maximize-better"), and `global_best` / `select_next_parent` are literally `min(...)` over `score`; `eval_blend`'s `metric_fn` and the `plateaued` / boundary / sanity-gate logic all inherit the same assumption.

So for a **maximize** metric (AUC, accuracy, F1, QWK, R², MAP@k) the driver must **negate before storing**:

```python
score = round(-auc, 6)                     # stored on the node: lower-is-better
hv3.add_node(tree, parent_id, mutation, stored_cfg, score, "evaluated", wall_s)
result["auc"] = round(auc, 6)              # human-readable value lives in result, NOT in score
metric_fn = lambda vec: -ev.auc(ev._y, vec)   # blend weight search minimizes too
```

For a **minimize** metric (RMSE, MAE, log_loss, MAPE) store it as-is — no flip. Say which case applies in the evaluator's module docstring, as `eval_s3e7.py` (flip) and `eval_s3e1.py` / `eval_s5e10.py` (no flip) each do.

Getting this wrong does not raise: a maximize metric stored unflipped makes `min()` return the **worst** node, so the search reports its own least-good candidate as the winner and then spends its whole budget mutating that lineage. Check it on the root — the root's stored score must be the negation of the number reported in `result` whenever the comp metric is maximize-better (2026-08-03 audit).

## 3. Budget and stopping rules

`harness_v3.py` already encodes these rules as default constants; the driver need not reinvent them:

- **Total budget of 60 evaluated nodes** (`DEFAULT_TOTAL_BUDGET`) = exploit phase ~35–40 + a mandatory explore burst of 5–8 long-shot lineages. The number 60 comes from the mean best/total ratio of 0.63 (range 0.10–1.00) over the 15 runs, leaving ample headroom.
- **The explore burst must not be skipped**, even if the exploit phase "looks converged." Three independent lines of evidence (E-5's s3e3-scale, F-2's s3e7, F-2's s3e14) show that **all** post-exploit gains came from a mandatorily-injected kitchen-sink mega-blend — not once from a hand-written long-shot solo lineage alone. `update_phase`/`should_stop` automatically flip the phase machine to `"explore_burst"`; once the driver sees this phase it must inject at least one kitchen-sink blend of "the entire current solo pool," plus 1–2 long-shot solos in contrasting directions.
- **Stopping rule**: stop after **15–20** consecutive evaluations (v3 default 20) since the burst began without refreshing the global best; regardless of the phase machine, the **60-node numeric cap is the backstop**. Honest caveat: the auto-stop patience-counter path **never actually fired** in either F-2 live run (the burst kept improving, the counter kept getting reset, and the search ended on the numeric cap) — it has only been validated end-to-end in one small-budget smoke test, so there is no live "the burst is a dud" data point yet.
- **Boundary-push is a first-class mutation type**: `boundary_candidates()` automatically flags hyperparameters stuck at the edge of the declared search space (`edge_frac=0.05`) and proposes candidates pushing outward. Four independent comps (s3e11/s3e5/s3e16/s3e7) confirm this is often one of the single largest levers — call it; don't rely on manually re-reading the Optuna trial table.

## 4. Driver script checklist

Using `tree_search/run_s3e7_v3.py` as the template (already wired with the three Phase H-1 productionization features), a new competition's driver **must**:

1. **Digit-for-digit root verification**: the root node's configuration must be bit-identical to that comp's known-best solo (usually from linear iteration or a previous harness version); immediately after evaluating it, `assert round(auc_or_score, 6) == <known_value>` and abort on failure — do not let drift in the root's own data/features silently contaminate the whole tree.
2. **OOF cache reuse** (optional but recommended): if an old tree (v2 or linear iteration) already has cached OOF for the identical configuration, reload it and recompute the metric, accepting the reuse only when it matches to 6 decimal places (digit-verified), otherwise fall back to real training — saves the compute of redundant training without sacrificing correctness.
3. **Resume-state contract** (H-1 feature 7, `save_search_state`/`load_search_state`): the driver's own runtime state (lineage name mapping, burst-injected flags, per-node result cache, dedup offset) all goes into `tree["search_state"]["driver_state"]`, **never** in module-level Python globals. Always save with `hv3.save_search_state(tree, path)` and load with `hv3.load_search_state(path)` — these entry points first run `validate_state` as a self-check, giving a clear diagnostic when a resume breaks instead of a KeyError surfacing three iterations later. Root cause: 1 of the 3 restart runs in the s3e7 F-2 validation came directly from module-level state not surviving a resume.
4. **Subprocess-level evaluation timeout** (H-1 feature 8, `eval_solo_subprocess`): a solo node's training call always goes through `hv3.eval_solo_subprocess(eval_module_path, config, timeout_s, node_id=...)` rather than calling `ev.evaluate(...)` directly — `signal.alarm` cannot interrupt a native `fit()` (CatBoost/LightGBM/XGBoost don't return control to Python bytecode); only an OS-level subprocess boundary can force a kill. Root cause: one CatBoost training in s3e7 F-2 (depth 9, bagging_temperature 2.0) hung for 28 minutes at 313% CPU in a native call.
5. **Burst-seed sanity gate** (H-1 feature 9, `apply_burst_seed_sanity_gate`): call it immediately after each explore-burst long-shot seed is evaluated; if its score falls outside a sanity band of "3× the root↔global-best gap" (with an absolute-value floor to avoid everything failing when the gap is 0), immediately mark that lineage plateaued so it can't keep spawning children. Root cause: a DART long-shot seed in s3e14 F-2 had an OOF MAE 18× the baseline, and the driver still spent ~12 minutes (6 evaluations) training its children before giving up.

## 5. Priors

Call `harness_v2.suggest_priors(comp_meta)` (inherited unchanged in v3) to keyword-match against `knowledge/experience.md`, returning verbatim evidence-cited bullets, with no LLM call. Query it once before proposing each lineage's first mutation.

**`comp_meta` MUST carry `comp`** — `{"comp": "<competition-slug>", "metric": ..., "tags": [...], "data_type": ...}`. Without it the call raises `ValueError`: the slug drives self-exclusion, which drops every bullet whose 證據 cites the competition being solved. That filter is the whole reason a re-run cannot improve its score by reading its own recorded answer, so it is required rather than optional. (Matching is token-boundary, so `s3e1` no longer swallows `s3e19`/`s3e11` evidence — 2026-08-03.)

**Priors set the floor, comp-local insight sets the ceiling** (a repeated finding across the 9-comp D+E sweep): priors consistently and correctly avoid wasting compute on known dead ends, but the single largest lever that actually opens up a score gap in each comp almost always comes from that comp's own EDA/comp-local insight or a structural change to the search mechanism itself (tenure-prune, top-code clip, auto_scale, depth-boundary-push, LGBBOUND, YEARWEIGHTS/JOINT), not from an experience-library hit itself. So: **use priors to prune dead ends, but don't stop looking for comp-local new mutation directions just because the priors didn't hit**.

After each search, if a new validated insight is found, write it back into `knowledge/experience.md` (consistent with the existing convention of the linear iteration protocol).

## 6. Honest reporting rules

Every completed tree-search run must report:

- **evals-to-beat**: the evaluation at which it matched/surpassed the linear-iteration best score (the driver should compute `evals_to_match_linear_best` mechanically, like `run_s3e7_v3.py`, not estimate from impression).
- **backtracks / plateau log**: `tree["search_state"]["backtrack_log"]` and the `plateaued` lineage list, attached verbatim without summarizing away the detail.
- **dedup rejection count**: `DEDUP_REJECTIONS`/`dedup_rejections` — too many means the node-space design is too narrow and the mutation queue is worth revisiting.
- **cost-guard / sanity-gate trigger records**: both default to "a trigger must leave an explicit warning," never silent — list each one when reporting, and say so explicitly even when "neither fired this run."
- **OOF-only caveat**: unless this run's final weights/postprocessing parameters have been validated on the real Kaggle Public/Private LB, you must state "this conclusion holds on the OOF score only and has not been LB-validated" — only 1 of the 15 historical runs (s3e16/E-3) has a real LB anchor; all the rest are OOF-only.
- **small-sample / overfit caveat**: if a key decision behind the final score (weights, postprocessing hyperparameters) was fit directly against the same OOF/CV folds used for evaluation, and the folds are few (e.g. 3-fold LOYO) or the gap is tiny in magnitude (<1% relative improvement), state plainly that this may be CV noise rather than real signal (cf. s3e20's 2.17% blend weight, s3e19's fold-5 double-dip).

## 7. Quick start

```python
import sys; sys.path.insert(0, "tree_search")
import harness_v3 as hv3

tree = hv3.load_search_state(TREE_PATH) if os.path.exists(TREE_PATH) else hv3.new_tree(COMP)
hv3.init_budget(tree)  # 60 / EXPLORE_BURST 5-8 / PATIENCE 20, unless there is a special reason to override

# SIGN: node scores are lower-is-better (§2). Maximize metric -> store -metric and keep the
# readable value in result[...]; minimize metric -> store as-is. Wrong sign = min() selects
# the worst node, silently.

# root + first-generation solo seeds + at least one blend seed; the root must assert digit-for-digit verification
# main loop:
while not hv3.should_stop(tree):
    phase = tree["search_state"]["budget"]["phase"]
    if phase == "explore_burst" and not burst_injected:
        inject_explore_burst(tree, ...)          # follow each seed with apply_burst_seed_sanity_gate
        continue
    parent_id, lineage_id = hv3.select_next_parent(tree)
    if parent_id is None: break
    child_cfg, mutation = propose_child(tree, parent_id, lineage_id)  # includes the suggest_priors query
    r = hv3.eval_solo_subprocess("tree_search/eval_<comp>.py", child_cfg, timeout_s=200, node_id=nid)
    ...
    hv3.save_search_state(tree, TREE_PATH)
```

Full runnable example: `tree_search/run_s3e7_v3.py`. Full API list: the `tree_search/harness_v3.py` module docstring.
