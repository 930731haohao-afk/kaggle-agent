"""tree_search/run_s3e16_v2.py — drives the harness_v2 (Phase E-3) candidate-tree search
for playground-series-s3e16 (Crab Age, regression, ROUNDED-INTEGER MAE). This is the
FIRST tree_search v2 build on a metric with a rounding-boundary and the ONLY comp in the
harness_v2 lineup with a REAL Kaggle LB anchor (Public 1.34356 / Private 1.34075 on OOF
1.33812), making it the acid test the task brief names it: can v2's mechanisms (metric-
aware plateau, ensemble-default node space, child dedup, k=800+coord-ascent weight
search) find a gain that SURVIVES rounding, where Phase B's linear iteration (STATUS.md,
scripts/round1.py/round2.py) already hit the recipe's own documented failure mode
(raw OOF MAE improved monotonically 1.35589->1.35541->1.35533, ROUNDED OOF MAE got WORSE
both times 1.33812->1.33850->1.33893 -- knowledge/experience.md's "MAE/整數目標" boundary-
condition bullet)?

--- Budget strategy ---
Root (LGB, the linear-best blend's dominant 60%-weight member) + 4 Phase-B pool members
(XGB, CAT, LGB_tuned, LGB_tuned_seed2024) are REUSED from scripts/cache/*.npz
(eval_s3e16_v2.load_legacy_solo, load + digit-for-digit raw-AND-rounded MAE recompute,
~0s each, verified BEFORE any node is added -- see verify_digit_for_digit()). The freed
training budget goes to genuinely new territory:

  1. LGBBOUND -- BOUNDARY-PUSH on the Optuna box scripts/tune_lgb_optuna.py searched for
     LGB_tuned (21/50 trials, 300s timeout). Checked every one of its 7 tuned hyperparams
     against its own search box (see eval_s3e16_v2.py's module docstring for the full
     table): learning_rate=0.0102 landed almost EXACTLY on the box's own LOWER bound
     ([0.01, 0.06] log-scale, ~2% of the log-range from the edge) -- every other tuned
     param (num_leaves, min_child_samples, subsample, colsample_bytree, reg_alpha,
     reg_lambda) sits comfortably interior (22%-85% of its range). This is the s3e11/s3e5
     boundary-push lever applied to the ONE param that actually saturates here
     [PRIOR: "超參調校(Optuna)" section's boundary-condition bullet is this exact
     failure mode already observed on this comp's own pool -- pushing the ONE saturated
     axis further, rather than repeating "nudge everything a little", is the direct
     response to that diagnosis].

  2. TWEEDIE -- diverse-objective probe (STATUS.md's own "Untried" list, scripts/
     pool_lib.train_lgb_tweedie was scaffolded but never run): Tweedie loss models a
     right-skewed, non-negative, count-like target (Age, skew 1.09) with a genuinely
     different loss surface than L1/MAE, not just nudged hyperparams on the same
     objective -- directly targets the diagnosed root cause of Phase B's failure
     (insufficient OOF-space diversity among pool members).

  3. POISSON -- a second diverse-objective probe in the same spirit as TWEEDIE (Age is a
     positive integer count), tested as an honest sibling rather than assuming Tweedie is
     the only count-like head worth trying.

  4. FEATPRUNE -- collinear feature-pruned member (STATUS.md's other "Untried" idea):
     Weight ~= Shucked+Viscera+Shell (r=0.993, EDA finding #3) means `Weight` itself is
     redundant given `parts_sum` (already in the 24-feature set, literally the sum of the
     three parts) and `weight_resid` (Weight minus that sum) -- dropping raw `Weight`
     tests whether the tree models are wasting splits on a column whose information is
     already fully recoverable from two other columns already present, per experience.md's
     "高共線性特徵" section precedent (s3e14 pruned near-perfect-collinear TRange columns
     for a real gain).

  5. BLEND -- composition search seeded at the linear champion's OWN exact composition
     (root LGB + XGB + CAT, dirichlet+coord-ascent weight search on k=800 as an internal
     reproduction check of the 1.33812 anchor), then grows by adding LGB_tuned,
     LGB_tuned_seed2024, and the best variant found in each of the 4 new lineages above,
     plus a remove-weakest probe [PRIOR: "Ensemble/後處理" section -- zero/near-zero-
     weight members cost nothing to keep in the pool, let the weight search decide].

Blend-cost note: unlike eval_s3e5_v2.py's QWK-after-OptimizedRounder (a scipy.optimize
call per candidate weight vector), s3e16's metric_fn is `MAE(clip(round(vec), 1, None))`
-- pure numpy, no optimizer call -- so a k=800+coordinate-ascent blend node here costs
low-single-digit seconds, not ~60s. The real budget constraint in THIS run is entirely
solo-node training time (each LGB fresh-train is roughly 60-150s depending on how low
`learning_rate` is pushed, since lower lr needs more boosting rounds before early
stopping(150) fires) -- the run's mutation queues are sized accordingly (2-3 authored
mutations per new lineage, not a long queue) so backtrack/dedup budget still exists
without risking the ~28 min wall-clock ceiling.

--- Idea-injection experiment (recommendation #4) ---
harness_v2.suggest_priors(COMP_META) is called once at startup; COMP_META's keywords hit
"### MAE/整數目標" (metric="mae"), "### 高共線性特徵" (tag "共線性"), "## 超參調校(Optuna)"
(tag "optuna"), "## Ensemble/後處理" (tag "ensemble") -- including, critically, the exact
boundary-condition bullet describing THIS comp's own Phase-B failure mode. Every mutation
description tags itself `[PRIOR Pk]` when directly shaped by one of those bullets, or
`[PRIOR none]` otherwise, so prior-usage win-rate can be computed post-hoc exactly as
run_s3e5_v2.py/run_s3e9_v2.py do.

v1: there is no v1 tree-search build for s3e16 (Phase B's iteration was the plain
linear-iteration script pattern, scripts/round1.py/round2.py/round3.py on harness-less
ad hoc code) -- this run writes to competitions/playground-series-s3e16/
experiments_tree.json (plain name, no "_v1"/"_v2" suffix needed per the task brief).
"""
import copy
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import eval_s3e16_v2 as ev  # noqa: E402

COMP = "playground-series-s3e16"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 24        # task brief: >=18 evaluated nodes; margin for backtracks/dedup
MAX_WALL_S = 28 * 60      # hard stop, matches the ~30 min training budget
EVAL_TIMEOUT_S = 240
LINEAR_BEST_ROUNDED = 1.33812   # linear-iteration's best (STATUS.md exp #1, rounded OOF MAE)
LINEAR_BEST_RAW = 1.35589

COMP_META = {"metric": "mae", "tags": ["共線性", "optuna", "ensemble"]}
# matches "### MAE/整數目標", "### 高共線性特徵", "## 超參調校(Optuna)", "## Ensemble/後處理"

DEDUP_REJECTIONS = []
_dedup_offset = {}


def dc(cfg):
    return copy.deepcopy(cfg)


def strip_result(cfg):
    out = {k: v for k, v in cfg.items() if k != "result"}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def find_dup(tree, cfg):
    target = hv2.config_hash(strip_result(cfg))
    for n in tree["nodes"]:
        if hv2.config_hash(strip_result(n["config"])) == target:
            return n["id"]
    return None


# ---------------------------------------------------------------------------
# Root + 4 legacy pool members: EXACT reproduction of Phase B's pool (scripts/pool_lib.py
# + scripts/round1.py/round2.py), reused via eval_s3e16_v2.load_legacy_solo (Phase B's
# scripts/cache/*.npz, read-only, digit-for-digit verified on BOTH raw and rounded MAE).
# ---------------------------------------------------------------------------
LGB_BASE_PARAMS = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                        learning_rate=0.02, num_leaves=63, min_child_samples=40,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                        reg_alpha=1.0, reg_lambda=2.0)
CAT_BASE_PARAMS = dict(loss_function="MAE", eval_metric="MAE", iterations=4000,
                        learning_rate=0.03, depth=7, l2_leaf_reg=5.0)
XGB_BASE_PARAMS = dict(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.02,
                        max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                        reg_alpha=1.0, reg_lambda=2.0)
LGB_TUNED_PARAMS = dict(
    learning_rate=0.010194593774108831, num_leaves=61, min_child_samples=30,
    subsample=0.8094065093571273, colsample_bytree=0.7216637100997246,
    reg_alpha=0.1131044878505013, reg_lambda=2.563832295704767, n_estimators=3000,
)

ROOT_CONFIG = {"kind": "solo", "model": "lgb", "params": dict(LGB_BASE_PARAMS, random_state=42),
               "features": {"drop": []}}
ROOT_MUTATION = ("root: s3e16 pool's dominant member -- plain L1-objective LGB, 60% weight "
                 "in the linear-champion 0.6/0.3/0.1 blend (STATUS.md exp #1, OOF MAE "
                 "raw 1.35651 / rounded 1.33885), REUSED from scripts/cache/LGB.npz "
                 "(verified digit-for-digit on BOTH raw and rounded MAE, no retrain)")

LEGACY_SPECS = {
    # name -> (expected_raw, expected_rounded, config)
    "XGB": (1.35763, 1.34160, {"kind": "solo", "model": "xgb",
                                "params": dict(XGB_BASE_PARAMS, random_state=42),
                                "features": {"drop": []}}),
    "CAT": (1.36162, 1.33846, {"kind": "solo", "model": "cat",
                                "params": dict(CAT_BASE_PARAMS, random_seed=42),
                                "features": {"drop": []}}),
    "LGB_tuned": (1.35583, 1.33979, {"kind": "solo", "model": "lgb",
                                      "params": dict(LGB_TUNED_PARAMS, random_state=42),
                                      "features": {"drop": []}}),
    "LGB_tuned_seed2024": (1.35604, 1.33914,
                           {"kind": "solo", "model": "lgb",
                            "params": dict(LGB_TUNED_PARAMS, random_state=2024),
                            "features": {"drop": []}}),
}
LEGACY_DESCS = {
    "XGB": "vanilla-hyperparam XGB (reg:absoluteerror), OOF raw 1.35763 / rounded 1.34160, "
           "30% weight in linear-champion blend, REUSED from scripts/cache/XGB.npz",
    "CAT": "vanilla-hyperparam CatBoost (MAE loss), OOF raw 1.36162 / rounded 1.33846, "
           "10% weight in linear-champion blend, REUSED from scripts/cache/CAT.npz",
    "LGB_tuned": "Optuna fold0-proxy tuned LGB (21/50 trials, 300s timeout), OOF raw "
                 "1.35583 / rounded 1.33979 -- Phase B round1: added to blend, ROUNDED "
                 "MAE got WORSE (1.33812->1.33850) despite raw improving, REUSED from "
                 "scripts/cache/LGB_tuned.npz [PRIOR: 超參調校 boundary-condition bullet "
                 "-- this IS the failure-mode member]",
    "LGB_tuned_seed2024": "seed-bagged LGB_tuned (same hyperparams, random_state=2024), "
                          "OOF raw 1.35604 / rounded 1.33914 -- Phase B round2: added, "
                          "ROUNDED MAE got WORSE AGAIN (1.33812->1.33893), 2nd "
                          "consecutive non-improving round -> Phase B STOPPED here, "
                          "REUSED from scripts/cache/LGB_tuned_seed2024.npz [PRIOR: same "
                          "boundary-condition bullet]",
}


# ---------------------------------------------------------------------------
# NEW first-generation lineages: LGBBOUND, TWEEDIE, POISSON, FEATPRUNE (real training).
# ---------------------------------------------------------------------------
def seed_lgbbound():
    p = dict(LGB_TUNED_PARAMS, random_state=42, learning_rate=0.005)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("BOUNDARY-PUSH (s3e11/s3e5 lever): LGB_tuned's Optuna learning_rate=0.0102 "
            "landed almost EXACTLY on tune_lgb_optuna.py's own search box lower bound "
            "([0.01,0.06] log) -- every other tuned param sits interior (22%-85% of its "
            "range, see eval_s3e16_v2.py module docstring's full table). Push past the "
            "edge, 0.0102->0.005, all other tuned hyperparams unchanged [PRIOR: 超參調校 "
            "section's boundary-condition bullet -- direct response to the diagnosed "
            "failure mode by attacking the ONE axis that actually saturated, not another "
            "generic nudge]")
    return cfg, desc


def seed_tweedie():
    p = dict(LGB_TUNED_PARAMS, random_state=42)
    p.pop("learning_rate")
    p.update(objective="tweedie", tweedie_variance_power=1.3, metric="mae",
             learning_rate=0.02)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("DIVERSE-OBJECTIVE probe (STATUS.md's own 'Untried' list, scripts/pool_lib."
            "train_lgb_tweedie scaffolded but never run): Tweedie loss "
            "(variance_power=1.3) targets Age's right-skewed (skew 1.09), non-negative, "
            "count-like distribution with a genuinely DIFFERENT loss surface than every "
            "other pool member's L1/MAE objective -- attacks Phase B's diagnosed root "
            "cause (insufficient OOF-space diversity among near-duplicate LGB variants) "
            "directly, rather than another hyperparam nudge on the same objective "
            "[PRIOR: MAE/整數目標 section -- rounding is the decision metric; a genuinely "
            "different loss surface is the honest way to test whether MORE diversity "
            "(not more raw-MAE optimization on the same surface) is what crosses the "
            "rounding boundary]")
    return cfg, desc


def seed_poisson():
    p = dict(LGB_BASE_PARAMS, random_state=42)
    p.update(objective="poisson", metric="mae")
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("Second DIVERSE-OBJECTIVE probe, honest sibling to TWEEDIE (not cherry-"
            "picking only one count-like head): Poisson objective, Age's positive-"
            "integer-count nature is the textbook Poisson use case, plain base "
            "hyperparams (not the tuned LGB_tuned set, to isolate the objective-change "
            "effect from the boundary-push effect already tested by LGBBOUND) [PRIOR: "
            "same MAE/整數目標 diversity rationale as TWEEDIE]")
    return cfg, desc


def seed_featprune():
    p = dict(LGB_BASE_PARAMS, random_state=42)
    cfg = {"kind": "solo", "model": "lgb", "params": p,
           "features": {"drop": ["Weight"]}}
    desc = ("COLLINEAR FEATURE-PRUNED probe (STATUS.md's other 'Untried' idea): "
            "Weight ~= Shucked Weight + Viscera Weight + Shell Weight (r=0.993, EDA "
            "finding #3) and the 24-feature set ALREADY carries both `parts_sum` (the "
            "literal sum of those 3 parts) and `weight_resid` (Weight minus that sum) -- "
            "raw `Weight` is fully algebraically recoverable from 2 columns already "
            "present, so dropping it tests whether it is pure redundancy or genuinely "
            "load-bearing (e.g. via nonlinear tree splits that the ratio/residual "
            "features don't expose as directly) [PRIOR: 高共線性特徵 section -- s3e14's "
            "precedent that pruning near-perfect-collinear columns can be a real gain, "
            "not just noise]")
    return cfg, desc


NEW_SOLO_SEEDS = [("LGBBOUND", seed_lgbbound), ("TWEEDIE", seed_tweedie),
                  ("POISSON", seed_poisson), ("FEATPRUNE", seed_featprune)]


# ---------------------------------------------------------------------------
# Per-lineage authored SOLO mutation queues (small -- 1-2 items, fallback covers the rest)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"] = dict(c["params"])
    c["params"].update(params_update)
    return c


LGBBOUND_QUEUE = [
    lambda c: (_bump(c, learning_rate=0.003),
               "LGBBOUND: push further past the box edge, 0.005->0.003 -- brackets "
               "whether the boundary-push gain (if any) keeps growing or has already "
               "peaked [PRIOR: 超參調校 boundary-condition]"),
    # NOTE: a 3rd queue item (n_estimators 3000->5000 at lr=0.003, to rule out
    # early_stopping(150) truncating convergence) was drafted but DROPPED after an
    # empirical timing check: the lr=0.005 seed alone took 217s/5-fold (measured before
    # this run started), so lr=0.003 is expected ~250-300s and a 5000-estimator variant
    # risks 350s+ -- with 4 new lineages sharing a ~28min wall-clock budget, this run
    # trims LGBBOUND to 2 real trainings (seed + 1 queue item) rather than 3, and lets
    # solo_fallback's seed-variation fill any remaining expansion budget on this lineage
    # cheaply if the search loop still wants to expand it further.
]
TWEEDIE_QUEUE = [
    lambda c: (_bump(c, tweedie_variance_power=1.5),
               "TWEEDIE: variance_power 1.3->1.5 (closer to pure count/Poisson-like "
               "variance-to-mean scaling) -- brackets the variance_power axis [PRIOR: "
               "MAE/整數目標 diversity rationale]"),
    lambda c: (_bump(c, tweedie_variance_power=1.1),
               "TWEEDIE: variance_power 1.3->1.1 (closer to Gaussian/L2-like scaling), "
               "other bracket direction [PRIOR: MAE/整數目標 diversity rationale]"),
]
POISSON_QUEUE = [
    lambda c: (_bump(c, learning_rate=0.03, num_leaves=31),
               "POISSON: shallower/faster variant (num_leaves 63->31, lr 0.02->0.03) -- "
               "cheaper second point on this objective's hyperparam surface [PRIOR: "
               "MAE/整數目標 diversity rationale]"),
]
FEATPRUNE_QUEUE = [
    lambda c: (dict(c, features={"drop": ["Weight", "parts_sum"]}),
               "FEATPRUNE: also drop parts_sum (drop both sides of the near-identity "
               "Weight<->parts_sum+weight_resid relationship) -- tests whether the "
               "residual alone (weight_resid) already carries the decomposition's "
               "useful signal [PRIOR: 高共線性特徵 section]"),
    lambda c: (dict(c, features={"drop": ["Shucked Weight_ratio", "Viscera Weight_ratio",
                                          "Shell Weight_ratio"]}),
               "FEATPRUNE: alternate collinearity axis -- drop the 3 weight-proportion "
               "ratio columns (each is Weight-normalized, i.e. Weight appears on both "
               "sides of every ratio) instead of raw Weight itself [PRIOR: 高共線性特徵]"),
]
SOLO_QUEUES = {"LGBBOUND": LGBBOUND_QUEUE, "TWEEDIE": TWEEDIE_QUEUE,
               "POISSON": POISSON_QUEUE, "FEATPRUNE": FEATPRUNE_QUEUE,
               # legacy-reused lineages: no authored queue -> immediate solo_fallback
               "XGB": [], "CAT": [], "LGB_tuned": [], "LGB_tuned_seed2024": []}


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    c["params"] = dict(c["params"])
    c["params"]["random_state"] = 7000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): same "
            f"config, alt random_state={7000 + attempt} [PRIOR none -- seed-bagging an "
            f"already-tuned model gave zero/negative rounded gain in Phase B (round2); "
            f"used here only as a last-resort filler, never counted on for a win]")
    return c, desc


# ---------------------------------------------------------------------------
# BLEND lineage
# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    all_names = list(LEGACY_SPECS) + [n for n, _ in NEW_SOLO_SEEDS] + ["BLEND"]
    out = {}
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name in all_names:
                if name not in out and n["mutation"].startswith(f"[{name}]"):
                    out[name] = n["id"]
    return out


def solo_pool(tree):
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["XGB"], ids["CAT"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    desc = (f"ensemble seed: 3-way blend root LGB(#{tree['root_id']}) + XGB(#{ids['XGB']}) "
            f"+ CAT(#{ids['CAT']}) -- reproduces the linear champion's EXACT composition "
            f"(0.6/0.3/0.1) as an internal check that k=800+coord-ascent weight search on "
            f"THIS module's own metric_fn recovers 1.33812 digit-for-digit before "
            f"exploring new territory")
    return cfg, desc


def _mk_add_named(name):
    def fn(tree, parent_cfg):
        seed_ids = _lineage_seed_ids(tree)
        if name not in seed_ids:
            return None
        add_id = seed_ids[name]
        if add_id in parent_cfg["members"]:
            return None
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        desc = (f"BLEND: add {name} solo member (#{add_id}) -> {len(child['members'])}-way, "
                f"dirichlet+coord-ascent weight search on rounded MAE [PRIOR: Ensemble/"
                f"後處理 -- weight search arbitrates, no cost to trying]")
        return child, desc
    return fn


def _blend_add_lineage_best(lineage_name):
    def fn(tree, parent_cfg):
        seed_ids = _lineage_seed_ids(tree)
        if lineage_name not in seed_ids:
            return None
        lineage_id = seed_ids[lineage_name]
        best = None
        for n in tree["nodes"]:
            if (n["status"] == "evaluated" and n["id"] != tree["root_id"]
                    and n["config"].get("kind") == "solo"
                    and hv2.lineage_of(tree, n["id"]) == lineage_id):
                if best is None or n["score"] < best["score"]:
                    best = n
        if best is None or best["id"] in parent_cfg["members"]:
            return None
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [best["id"]]
        child.pop("result", None)
        desc = (f"BLEND: add {lineage_name}'s best variant so far (#{best['id']}, "
                f"rounded MAE={best['score']:.5f}) -> {len(child['members'])}-way -- new "
                f"territory (Phase B never tested this member as a blend candidate) "
                f"[PRIOR: {lineage_name} lineage's own prior tag]")
        return child, desc
    return fn


def _blend_remove_weakest(tree, parent_cfg):
    weights = (parent_cfg.get("result") or {}).get("weights")
    members = parent_cfg["members"]
    if not weights or len(members) <= 2:
        return None
    weak_id = min(members, key=lambda m: weights.get(str(m), 1.0))
    child = dc(parent_cfg)
    child["members"] = [m for m in members if m != weak_id]
    child.pop("result", None)
    desc = (f"BLEND: remove lowest-weight member #{weak_id} -> leaner "
            f"{len(child['members'])}-way, test whether it was truly adding value "
            f"[PRIOR: Ensemble/後處理 section -- deleting continuously-zero-weight "
            f"members is zero-cost]")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("LGB_tuned"),                    # -> 4-way
    _mk_add_named("LGB_tuned_seed2024"),            # -> 5-way, reproduces Phase B's own pool
    _blend_add_lineage_best("LGBBOUND"),            # -> new territory
    _blend_add_lineage_best("TWEEDIE"),             # -> new territory
    _blend_add_lineage_best("POISSON"),             # -> new territory
    _blend_add_lineage_best("FEATPRUNE"),           # -> new territory
    _blend_remove_weakest,                          # leaner-blend probe
]


def blend_fallback(tree, parent_cfg, attempt):
    pool = solo_pool(tree)
    unused = [nid for nid in pool if nid not in parent_cfg["members"]]
    if unused:
        add_id = unused[0]
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            desc = (f"BLEND fallback: add next-best unused pool member #{add_id} -> "
                    f"{len(child['members'])}-way [PRIOR none]")
            return child, desc
    weights = (parent_cfg.get("result") or {}).get("weights")
    members = parent_cfg["members"]
    if weights and len(members) > 2:
        weak_id = min(members, key=lambda m: weights.get(str(m), 1.0))
        child = dc(parent_cfg)
        child["members"] = [m for m in members if m != weak_id]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            return child, f"BLEND fallback: remove weakest #{weak_id} [PRIOR none]"
    return None  # genuinely exhausted -> forced backtrack


# ---------------------------------------------------------------------------
def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES[lineage_id]
    idx = hv2.lineage_size(tree, lineage_id) - 1 + _dedup_offset.get(lineage_id, 0)
    if name == "BLEND":
        result = None
        if idx < len(BLEND_QUEUE):
            result = BLEND_QUEUE[idx](tree, parent_node["config"])
        if result is None:
            attempt = max(idx - len(BLEND_QUEUE), 0)
            result = blend_fallback(tree, parent_node["config"], attempt)
        if result is None:
            return None, None
        child_cfg, desc = result
    else:
        queue = SOLO_QUEUES[name]
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


LINEAGE_NAMES = {}


# ---------------------------------------------------------------------------
def reuse_legacy(tree, mutation, cfg, expected_raw, expected_rounded, cache_name, is_root=False):
    nid = hv2.next_id(tree)
    oof, pred, raw, rounded = ev.load_legacy_solo(cache_name, expected_raw, expected_rounded)
    hv2.cache_oof(ev.CACHE_DIR, nid, oof, pred=pred, y=ev._y, mae_raw=raw, mae_rounded=rounded)
    result = {"n_feats": None, "mae_raw": round(raw, 5), "mae_rounded": round(rounded, 5),
              "reused_from": f"scripts/cache/{cache_name}.npz"}
    stored_cfg = {**cfg, "result": result}
    if is_root:
        real_nid = hv2.add_root(tree, mutation, stored_cfg, round(rounded, 5), "evaluated", 0.0)
    else:
        real_nid, dup = hv2.add_node(tree, tree["root_id"], mutation, stored_cfg,
                                      round(rounded, 5), "evaluated", 0.0, allow_duplicate=True)
        assert dup is None
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    hv2.save(tree, TREE_PATH)
    return real_nid, result


def eval_and_add(tree, parent_id, mutation, child_cfg, is_root=False, lineage_id=None):
    dup_id = None if is_root else find_dup(tree, child_cfg)
    if dup_id is not None:
        DEDUP_REJECTIONS.append(dict(mutation=mutation, dup_id=dup_id))
        if lineage_id is not None:
            _dedup_offset[lineage_id] = _dedup_offset.get(lineage_id, 0) + 1
        return None, dup_id, None

    nid = hv2.next_id(tree)
    r = ev.evaluate(child_cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
    stored_cfg = child_cfg
    if r["status"] == "evaluated" and r.get("result") is not None:
        stored_cfg = {**child_cfg, "result": r["result"]}
    if is_root:
        real_nid = hv2.add_root(tree, mutation, stored_cfg, r["score"], r["status"], r["wall_s"])
    else:
        real_nid, add_dup = hv2.add_node(tree, parent_id, mutation, stored_cfg, r["score"],
                                          r["status"], r["wall_s"], allow_duplicate=True)
        assert add_dup is None
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    hv2.save(tree, TREE_PATH)
    return real_nid, None, r


def score_of(r):
    if r is None:
        return None
    return r.get("score")


# ---------------------------------------------------------------------------
_PRIOR_TAG_RE = re.compile(r"\[PRIOR([^\]]*)\]")


def prior_usage_summary(tree):
    by_id = {n["id"]: n for n in tree["nodes"]}
    informed, uninformed = [], []
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["parent_id"] is None or n["parent_id"] == tree["root_id"]:
            continue
        m = _PRIOR_TAG_RE.search(n["mutation"])
        if not m:
            continue
        tag = m.group(1).strip()
        parent = by_id.get(n["parent_id"])
        if parent is None or parent["score"] is None or n["score"] is None:
            continue
        won = n["score"] < parent["score"]  # lower rounded MAE is better
        (informed if tag != "none" else uninformed).append(dict(node_id=n["id"], tag=tag, won=won))

    def rate(lst):
        return (sum(1 for x in lst if x["won"]) / len(lst)) if lst else None

    return dict(informed=informed, uninformed=uninformed,
                informed_win_rate=rate(informed), uninformed_win_rate=rate(uninformed))


def verify_digit_for_digit():
    print("=== digit-for-digit Phase-B cache verification (BEFORE searching) ===")
    checks = [("LGB (root)", 1.35651, 1.33885)] + \
             [(name, raw, rounded) for name, (raw, rounded, _) in LEGACY_SPECS.items()]
    cache_name_map = {"LGB (root)": "LGB", **{n: n for n in LEGACY_SPECS}}
    for name, raw, rounded in checks:
        cache_name = cache_name_map[name]
        oof, pred, r, rd = ev.load_legacy_solo(cache_name, raw, rounded)
        print(f"  {name} (scripts/cache/{cache_name}.npz): recomputed raw={r:.6f} "
              f"rounded={rd:.6f} vs historical raw={raw} rounded={rounded} -> OK "
              f"(|diff raw|={abs(r - raw):.2e}, |diff rounded|={abs(rd - rounded):.2e})")
    print("=== verification passed ===\n")


def main():
    t_start = time.time()
    verify_digit_for_digit()

    if os.path.exists(TREE_PATH):
        tree = hv2.load(TREE_PATH)
        print(f"Resuming existing tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv2.new_tree(COMP)
        priors = hv2.suggest_priors(COMP_META)
        tree["priors"] = priors
        print(f"suggest_priors({COMP_META}) -> {len(priors)} bullets:")
        for i, p in enumerate(priors):
            print(f"  P{i}: {p[:200]}")

    def n_evaluated():
        return sum(1 for n in tree["nodes"] if n["status"] == "evaluated")

    # --- root: REUSED from scripts/cache/LGB.npz ---
    if not tree["nodes"]:
        nid, r = reuse_legacy(tree, ROOT_MUTATION, ROOT_CONFIG, 1.35651, 1.33885, "LGB", is_root=True)
        print(f"[root] #{nid} rounded_MAE={r['mae_rounded']} wall_s=0.0 (REUSED, verified "
              f"digit-for-digit)")

    # --- 4 legacy pool members: REUSED from scripts/cache/*.npz ---
    for name, (expected_raw, expected_rounded, cfg) in LEGACY_SPECS.items():
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        mutation = f"[{name}] {LEGACY_DESCS[name]}"
        nid, r = reuse_legacy(tree, mutation, cfg, expected_raw, expected_rounded, name)
        print(f"[{name} seed] #{nid} rounded_MAE={r['mae_rounded']} wall_s=0.0 (REUSED, "
              f"verified digit-for-digit)")

    # --- 4 NEW first-generation SOLO lineages (real training) ---
    for name, seed_fn in NEW_SOLO_SEEDS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} rounded_MAE={score_of(r)} status={r['status']} "
              f"wall_s={r['wall_s']}")

    # --- BLEND lineage seed (needs root + XGB + CAT ids) ---
    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} rounded_MAE={score_of(r)} status={r['status']} "
              f"wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = list(LEGACY_SPECS) + [n for n, _ in NEW_SOLO_SEEDS] + ["BLEND"]
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in all_names:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name

    # --- adaptive tree-search loop ---
    iterations = 0
    while n_evaluated() < TARGET_NODES and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > 100:
            print("Safety cap on iterations reached, stopping.")
            break
        parent_id, lineage_id = hv2.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted (no more eligible expansions). Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        if child_cfg is None:
            st = tree["search_state"]
            if lineage_id not in st["plateaued"]:
                st["plateaued"].append(lineage_id)
                st["backtrack_log"].append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=(f"lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)}'s "
                            f"mutation proposals exhausted -> forced backtrack")))
            hv2.save(tree, TREE_PATH)
            print(f"FORCED BACKTRACK: lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"mutation space exhausted; plateaued={tree['search_state']['plateaued']}")
            continue
        nid, dup, r = eval_and_add(tree, parent_id, mutation, child_cfg, lineage_id=lineage_id)
        if nid is None:
            print(f"DEDUP REJECTED <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"(identical to existing node #{dup}); retrying with next mutation-queue slot")
            continue
        gb = hv2.global_best(tree)
        gb_score = gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"rounded_MAE={score_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best={gb_score} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']} "
              f"| tie_rate={hv2.tie_rate(tree):.3f} "
              f"| elapsed={time.time() - t_start:.0f}s")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    gb_score = gb["score"] if gb else None
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    n_reused = sum(1 for n in tree["nodes"] if n["status"] == "evaluated"
                   and (n["config"].get("result") or {}).get("reused_from"))
    prior_stats = prior_usage_summary(tree)
    tree["prior_usage_summary"] = {
        "n_informed": len(prior_stats["informed"]), "n_uninformed": len(prior_stats["uninformed"]),
        "informed_win_rate": prior_stats["informed_win_rate"],
        "uninformed_win_rate": prior_stats["uninformed_win_rate"],
    }
    tree["dedup_rejections"] = DEDUP_REJECTIONS
    tree["final_tie_rate"] = round(hv2.tie_rate(tree), 4)
    hv2.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend, "
          f"{n_reused} reused-from-Phase-B-cache), {len(tree['nodes'])} total, "
          f"wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} rounded_MAE={gb_score} mutation={gb['mutation'][:150]}")
    print(f"Linear-iteration best (rounded): {LINEAR_BEST_ROUNDED} (v2 tree "
          f"{'BEAT' if gb_score < LINEAR_BEST_ROUNDED else ('MATCHED' if gb_score == LINEAR_BEST_ROUNDED else 'did NOT beat')} it)")
    print(f"Final tie_rate: {hv2.tie_rate(tree):.3f}")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for e in tree["search_state"]["backtrack_log"]:
        print(" ", e)
    print(f"Dedup rejections ({len(DEDUP_REJECTIONS)}):")
    for e in DEDUP_REJECTIONS:
        print(" ", e)
    print(f"Prior usage: informed={len(prior_stats['informed'])} "
          f"(win rate={prior_stats['informed_win_rate']}), "
          f"uninformed={len(prior_stats['uninformed'])} "
          f"(win rate={prior_stats['uninformed_win_rate']})")


if __name__ == "__main__":
    main()
