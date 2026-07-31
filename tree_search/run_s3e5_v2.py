"""tree_search/run_s3e5_v2.py — drives the harness_v2 (Phase E-2) candidate-tree search
for playground-series-s3e5 (Wine Quality, ordinal regression, QWK, DISCRETIZED metric).

--- Why this run exists ---
v1 (tree_search/run_s3e5.py + eval_s3e5.py, harness.py) TIED here: its best node #11
(4-way blend: root LGB_tuned + CAT_tuned + XGB + FEAT-17feat, weights
0.129/0.645/0.018/0.208) scored 0.56766, 0.00003 short of the linear-iteration's 0.56769
-- STATUS.md's own appendix calls this "the cutpoint-boundary noise gap" and closes with
"樹搜尋在離散化指標上機制全部成立...但未超越". harness_v2's two Stage-4 recommendations
were BUILT reading exactly that v1 s3e5 lesson: the metric-aware adaptive plateau (§6.2 --
tie_rate(tree) > 0.15 -> ADAPTIVE_PLATEAU_STREAK=5 + neutral-tie bookkeeping) and the
ensemble-default node space (§6.1 -- kind:"solo"|"blend" as a first-class schema, not a
hand-rolled convention). This run is THEIR first real test: does re-running the same
comp/data/CV/rounder discipline on harness_v2 flip the 3-way tie among {v1 tree, linear
iteration, v2 tree}?

v1's own experiments_tree.json is left completely untouched; this run writes to its own
experiments_tree_v2.json. v1's own cache_s3e5/*.npz is READ-ONLY here (see
eval_s3e5_v2.py's module docstring for the digit-for-digit reuse mechanism); this run's
own new/re-cached OOFs live in cache_s3e5/v2/ (a subdirectory, so no v1 cache file can
ever be overwritten).

--- Budget strategy ---
Root + 6 first-gen solo seeds are REUSED from v1's cache_s3e5/solo_{0,2,3,4,5,6,7}.npz
(eval_s3e5_v2.load_v1_solo, load + digit-for-digit QWK recompute, ~0s each, verified
BEFORE any node is added -- see verify_digit_for_digit()). The freed training budget goes
to genuinely new territory the task brief calls out:

  1. LGBBOUND -- BOUNDARY-PUSH on the root's Optuna-tuned LGB hyperparams (the "s3e11
     lever": Optuna optima can sit exactly on the search box's edge, meaning the true
     optimum may lie OUTSIDE the box that was searched). Checked the linear run's actual
     Optuna space (scripts/iterate.py's tune_lgb, trial.suggest_* bounds) before writing
     this: max_depth=3 landed EXACTLY on the box's lower bound (searched [3,10]);
     min_child_samples=40 landed EXACTLY on the box's upper bound (searched [3,40]);
     reg_lambda=0.00104 landed essentially ON the box's lower bound (searched [1e-3,10]
     log-scale). Three separate hyperparameters saturated their box edges -- a much
     stronger boundary signal than CatBoost's tuned params show (see CATBOUND below) --
     so this lineage pushes each one further past its Optuna box edge, individually and
     combined [PRIOR P18 -- s3e11's capacity/regularization-reversal finding is the
     direct precedent for "don't trust an Optuna optimum sitting on the box's own edge";
     P8/P10 -- this comp's small sample (~2k rows) independently predicts MORE
     regularization should keep helping, consistent with 2 of the 3 saturated params
     (max_depth down, min_child_samples up) pushing in the more-regularized direction].

  2. CATBOUND -- the same boundary-push question applied to the tuned CatBoost, as an
     honest CONTRAST/confirm-or-deny probe: CAT's own Optuna box
     (depth=4 in [3,8], learning_rate=0.0237 in [0.01,0.15] log, l2_leaf_reg=6.46 in
     [0.01,20] log, min_data_in_leaf=27 in [1,40]) shows NO parameter sitting exactly on
     an edge -- the tuned solution is interior. Expectation going in: this lineage should
     find LESS to gain than LGBBOUND, since there is no saturated-edge signal to exploit
     [PRIOR P18 -- same lever, deliberately tested on a member where the box-edge
     precondition does NOT hold, to avoid cherry-picking only the positive case].

  3. FOLDROUNDER -- the task brief's "per-fold-averaged cutpoints as an alternative
     rounder inside a node variant" lever, seeded as a genuine lineage (root's exact LGB-
     tuned config, `rounder`:"fold_avg" instead of the default "full_oof"; retrains --
     cheap, ~2s -- rather than reusing a cached OOF, so its mutation queue can explore
     normally). STATUS.md's own exp #10 diagnostic already ran the identical computation
     BY HAND once, on the linear run's 6-way blend, and found it WORSE than full-OOF
     fitting (0.56449 vs 0.56769) -- the honest prior here is flat-to-worse, not assumed
     to help [PRIOR P4/P5 -- nested/full-OOF overfit-risk diagnostics, the direct
     precedent for this exact alternative-rounder computation].

  4. BLEND -- composition search seeded at the task brief's explicit suggestion ("blend
     composition around the two tuned members"): 2-way (root LGB_tuned + CAT_tuned) is
     seeded first (mirrors the linear champion's OWN effective composition -- its 6-way
     blend's weight search gave 95% to CAT_tuned + 5% to LGB, i.e. only 2 members
     actually mattered), then grows by adding XGB, FEAT-17 (reproducing v1's own best
     4-way composition as an internal check), LGBNUDGE, CATORIG, LGBORIG, and the best
     LGBBOUND variant once trained -- plus a remove-weakest probe and one rounder="
     fold_avg" toggle on the best composition found [PRIOR P19 -- zero/near-zero-weight
     members cost nothing to keep in the pool, let the weight search decide].

Blend-cost mitigation (task brief: "coarser weight grid ... if 45s/blend threatens
budget -- document what you choose") -- REVISED after an empirical check found the
"obviously safe" coarser choice actively cost the win. See eval_s3e5_v2.py's
evaluate_blend docstring for the full k-sweep: k=200 (~15s/node) reproduced v1's own best
4-member composition but landed on a WORSE point of the weight simplex (0.56601 vs v1's
0.56766 on IDENTICAL OOF data -- confirmed a pure search-quality gap, not a data bug);
k=800 (~60s/node, matching v1's own dirichlet-draw budget, PLUS a coordinate-ascent
refinement stage v1's own weight_search() had that harness_v2.eval_blend's dirichlet-only
mode lacks by default) found 0.56874 on that SAME composition -- beating both v1
(0.56766) and the linear iteration (0.56769). Given ample wall-clock headroom (a full
22-node pass at k=200 used only ~90s against a ~28 min budget), this run uses k=800
throughout rather than a coarser setting: the actual budget risk in this comp was never
"45s x N blend nodes blows the 30 min ceiling" (it structurally cannot, num_blend_nodes is
single digits), it was under-searching a metric surface where the winning region of the
weight simplex is narrow.

--- Idea-injection experiment (recommendation #4) ---
harness_v2.suggest_priors(COMP_META) is called once at startup; COMP_META's keywords hit
"### QWK/序數目標" (metric="qwk", tag="序數目標"), "### 小樣本(<10k 列)" (tag "小樣本"),
"## 超參調校(Optuna)" (tag "optuna"), "## Ensemble/後處理" (tag "ensemble") -- 20 bullets
(P0-P19) total. Every mutation description tags itself `[PRIOR Pk]` when directly shaped
by one of those bullets, or `[PRIOR none]` otherwise, so prior-usage win-rate can be
computed post-hoc exactly as run_s3e9_v2.py/run_s3e11.py do.
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
import eval_s3e5_v2 as ev  # noqa: E402

COMP = "playground-series-s3e5"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree_v2.json")
TARGET_NODES = 22        # task brief: >=18 evaluated nodes; margin for backtracks/dedup
MAX_WALL_S = 28 * 60      # hard stop, matches the ~30 min training budget
EVAL_TIMEOUT_S = 120
LINEAR_BEST = 0.56769     # linear-iteration's best (STATUS.md exp #8, full precision --
                          # experiments.json stores "score": 0.56769 exactly, no hidden decimals)
V1_TREE_BEST = 0.56766    # v1 (harness.py) tree-search best (node #11, 4-way blend)

COMP_META = {"metric": "qwk", "tags": ["序數目標", "小樣本", "optuna", "ensemble"]}
# matches "### QWK/序數目標", "### 小樣本(<10k 列)", "## 超參調校(Optuna)",
# "## Ensemble/後處理" -- 20 bullets (P0-P19), see module docstring for which are cited.

DEDUP_REJECTIONS = []
_dedup_offset = {}


def dc(cfg):
    return copy.deepcopy(cfg)


def strip_result(cfg):
    """hv2.add_node's own built-in dedup compares the FULL stored config, which by the
    time a node is stored already has the evaluator's `result` merged in (weights,
    cutpoints, ...) -- two semantically-identical proposals would never hash equal that
    way. This driver's own dedup check strips "result" first, and sorts blend member
    lists (order-independent duplicate detection), before hashing."""
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
# Root + 6 legacy pool members: EXACT reproduction of v1's root + first-gen solo seeds
# (tree_search/run_s3e5.py), reused via eval_s3e5_v2.load_v1_solo (v1's cache_s3e5/*.npz,
# read-only, digit-for-digit verified).
# ---------------------------------------------------------------------------
LGB_TUNED_PARAMS = dict(
    num_leaves=57, max_depth=3, learning_rate=0.016850482169027743, min_child_samples=40,
    subsample=0.9520104938890754, colsample_bytree=0.7120571432335079,
    reg_alpha=0.7955110174517752, reg_lambda=0.001041891430753302, n_estimators=345,
)
CAT_TUNED_PARAMS = dict(
    depth=4, learning_rate=0.023711050140388417, l2_leaf_reg=6.458015909315451,
    iterations=333, random_strength=1.2907627214036907, min_data_in_leaf=27,
)

ROOT_CONFIG = {"kind": "solo", "model": "lgb", "params": dict(LGB_TUNED_PARAMS, random_state=42),
               "features": {"drop": []}, "rounder": "full_oof"}
ROOT_MUTATION = ("root: s3e5 current-best single learner -- Optuna direct-post-rounder-"
                 "QWK-tuned LGB (STATUS.md exp #5 origin), solo QWK 0.56244, REUSED from "
                 "v1's cache_s3e5/solo_0.npz (verified digit-for-digit, no retrain) "
                 "[PRIOR P3 -- Optuna-direct-final-metric objective, the biggest lever "
                 "already baked into this config]")

LEGACY_SPECS = {
    # name -> (v1_node_id, expected historical QWK, config)
    "CAT": (2, 0.56466, {"kind": "solo", "model": "cat",
                          "params": dict(CAT_TUNED_PARAMS, random_seed=42),
                          "features": {"drop": []}, "rounder": "full_oof"}),
    "XGB": (3, 0.50847, {"kind": "solo", "model": "xgb",
                         "params": dict(objective="reg:squarederror", n_estimators=1500,
                                        learning_rate=0.03, max_depth=5, subsample=0.8,
                                        colsample_bytree=0.7, reg_lambda=1.0,
                                        min_child_weight=5, random_state=42),
                         "features": {"drop": []}, "rounder": "full_oof"}),
    "LGBNUDGE": (4, 0.56027, {"kind": "solo", "model": "lgb",
                              "params": dict(LGB_TUNED_PARAMS, random_state=42,
                                             n_estimators=500, learning_rate=0.012),
                              "features": {"drop": []}, "rounder": "full_oof"}),
    "FEAT": (5, 0.5566, {"kind": "solo", "model": "lgb",
                         "params": dict(LGB_TUNED_PARAMS, random_state=42),
                         "features": {"drop": ["fixed_to_volatile_acid", "citric_to_volatile_acid",
                                               "sugar_to_alcohol", "acid_ph_ratio"]},
                         "rounder": "full_oof"}),
    "CATORIG": (6, 0.52517, {"kind": "solo", "model": "cat", "params": dict(random_seed=42),
                             "features": {"drop": []}, "rounder": "full_oof"}),
    "LGBORIG": (7, 0.50547, {"kind": "solo", "model": "lgb", "params": dict(random_state=42),
                             "features": {"drop": []}, "rounder": "full_oof"}),
}
LEGACY_DESCS = {
    "CAT": "Optuna-tuned CatBoost, solo QWK 0.56466, REUSED from v1 cache_s3e5/solo_2.npz "
           "[PRIOR P3 -- direct-QWK Optuna objective, second-strongest single learner]",
    "XGB": "vanilla/untuned XGB, solo QWK 0.50847, REUSED from v1 cache_s3e5/solo_3.npz "
           "[PRIOR P19 -- kept as a free blend-diversity source regardless of solo strength]",
    "LGBNUDGE": "second independent LGB-tuned exploration point (n_estimators 345->500, lr "
                "0.0169->0.012), solo QWK 0.56027, REUSED from v1 cache_s3e5/solo_4.npz "
                "[PRIOR P3]",
    "FEAT": "root's tuned-LGB hyperparams on a 17-feature subset (drop 4 weakest-"
            "correlation ratios), solo QWK 0.5566, REUSED from v1 cache_s3e5/solo_5.npz "
            "-- v1's own BEST blend (node #11, 0.56766) included this exact member "
            "[PRIOR P3]",
    "CATORIG": "vanilla/untuned CatBoost, solo QWK 0.52517, REUSED from v1 "
               "cache_s3e5/solo_6.npz [PRIOR P19]",
    "LGBORIG": "vanilla/untuned LGB, solo QWK 0.50547, REUSED from v1 cache_s3e5/solo_7.npz "
               "[PRIOR P19]",
}


# ---------------------------------------------------------------------------
# NEW first-generation lineages: LGBBOUND, CATBOUND, FOLDROUNDER (real training).
# ---------------------------------------------------------------------------
def seed_lgbbound():
    p = dict(LGB_TUNED_PARAMS, random_state=42, max_depth=2)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []},
           "rounder": "full_oof"}
    desc = ("BOUNDARY-PUSH (the s3e11 lever, task brief): root's Optuna-tuned max_depth=3 "
            "landed EXACTLY on the tune_lgb search box's own lower bound ([3,10], "
            "scripts/iterate.py) -- push past it, max_depth 3->2 -- if the box itself "
            "clipped the true optimum, going further should keep helping [PRIOR P18 -- "
            "Optuna optima sitting on a search-box edge is exactly s3e11's precedent for "
            "distrust; P8/P10 -- this comp's small sample independently predicts more "
            "regularization (shallower trees) should help]")
    return cfg, desc


def seed_catbound():
    p = dict(CAT_TUNED_PARAMS, random_seed=42, depth=3)
    cfg = {"kind": "solo", "model": "cat", "params": p, "features": {"drop": []},
           "rounder": "full_oof"}
    desc = ("BOUNDARY-PUSH contrast probe: CAT's tuned depth=4 sits well INSIDE its own "
            "Optuna box ([3,8]) -- unlike LGB's 3 saturated params, no CAT hyperparameter "
            "landed on an edge, so the honest expectation is LESS to gain here. Depth "
            "4->3 (the nearest-to-boundary axis) is tested anyway as a confirm-or-deny "
            "control, deliberately NOT cherry-picking only the positive (LGBBOUND) case "
            "[PRIOR P18 -- same lever, tested where the precondition does not hold]")
    return cfg, desc


def seed_foldrounder():
    cfg = {"kind": "solo", "model": "lgb", "params": dict(LGB_TUNED_PARAMS, random_state=42),
           "features": {"drop": []}, "rounder": "fold_avg"}
    desc = ("ALTERNATIVE-ROUNDER probe (task brief: 'per-fold-averaged cutpoints as an "
            "alternative rounder inside a node variant'): root's EXACT LGB-tuned config, "
            "only `rounder` flipped full_oof->fold_avg (fit OptimizedRounder separately "
            "on each fold's own OOF slice, average the 5 cutpoint vectors, decode the "
            "full OOF with that average). STATUS.md's own exp #10 already ran this exact "
            "computation by hand on the linear run's 6-way blend and found it WORSE "
            "(0.56449 vs 0.56769) -- honest prior is flat-to-worse on a solo, retrain is "
            "cheap (~2s) so worth confirming at this run's own pool maturity too [PRIOR "
            "P4/P5 -- nested/full-OOF cutpoint-overfit diagnostics, direct precedent for "
            "this exact alternative]")
    return cfg, desc


NEW_SOLO_SEEDS = [("LGBBOUND", seed_lgbbound), ("CATBOUND", seed_catbound),
                  ("FOLDROUNDER", seed_foldrounder)]


# ---------------------------------------------------------------------------
# Per-lineage authored SOLO mutation queues (small -- 1-2 items, fallback covers the rest)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"] = dict(c["params"])
    c["params"].update(params_update)
    return c


LGBBOUND_QUEUE = [
    lambda c: (_bump(c, min_child_samples=60),
               "LGBBOUND: 2nd saturated param -- min_child_samples=40 landed EXACTLY on "
               "tune_lgb's own upper bound ([3,40]); push past it, 40->60 [PRIOR P18/P8]"),
    lambda c: (_bump(c, reg_lambda=1e-5),
               "LGBBOUND: 3rd saturated param -- reg_lambda=0.00104 landed essentially ON "
               "tune_lgb's lower bound ([1e-3,10] log); push further toward 0, ->1e-5 "
               "[PRIOR P18]"),
    lambda c: (_bump(c, max_depth=2, min_child_samples=60, reg_lambda=1e-5),
               "LGBBOUND: COMBINE all 3 boundary-pushes at once (max_depth=2, "
               "min_child_samples=60, reg_lambda=1e-5) -- tests whether the individual "
               "pushes compound or interfere [PRIOR P18]"),
]
CATBOUND_QUEUE = [
    lambda c: (_bump(c, depth=2),
               "CATBOUND: push depth further (3->2) since depth=3 (1st push) is still "
               "interior-ish -- brackets whether ANY gain exists on this axis at all "
               "[PRIOR P18 control]"),
]
FOLDROUNDER_QUEUE = [
    lambda c: (_bump(c, max_depth=2),
               "FOLDROUNDER: combine with LGBBOUND's max_depth=2 push -- does "
               "fold_avg rounder change which hyperparameter direction wins, or is the "
               "rounder choice orthogonal to model tuning [PRIOR P18/P4]"),
]
SOLO_QUEUES = {"LGBBOUND": LGBBOUND_QUEUE, "CATBOUND": CATBOUND_QUEUE,
               "FOLDROUNDER": FOLDROUNDER_QUEUE,
               # legacy-reused lineages: no authored queue -> immediate solo_fallback
               "CAT": [], "XGB": [], "LGBNUDGE": [], "FEAT": [], "CATORIG": [], "LGBORIG": []}


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    c["params"] = dict(c["params"])
    seed_key = "random_state" if c["model"] in ("lgb", "xgb") else "random_seed"
    c["params"][seed_key] = 6000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): same "
            f"config, alt {seed_key}={6000 + attempt} [PRIOR P15 -- seed-bagging an "
            f"already direct-QWK-Optuna-tuned model gave ZERO gain twice in the linear "
            f"run; used here only as a last-resort filler, never counted on for a win]")
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
    """All evaluated kind=='solo' node ids with rounder=='full_oof' (blend composition
    stays in one rounder space; the fold_avg rounder is tested as its own dedicated
    toggle mutation, see _blend_rounder_toggle, not mixed member-by-member), best
    (highest QWK == lowest -QWK score) first."""
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"
             and n["config"].get("rounder", "full_oof") == "full_oof"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["CAT"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet",
           "rounder": "full_oof"}
    desc = (f"ensemble seed: 2-way blend of root LGB_tuned(#{tree['root_id']}) + "
            f"CAT_tuned(#{ids['CAT']}) -- task brief's explicit suggestion 'blend "
            f"composition around the two tuned members', which also mirrors the linear "
            f"champion's OWN effective composition (its 6-way weight search put 95% on "
            f"CAT_tuned + 5% on LGB, i.e. only 2 members actually mattered) [PRIOR P3]")
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
                f"dirichlet weight search [PRIOR P19 -- weight search arbitrates, no cost "
                f"to trying]")
        return child, desc
    return fn


def _blend_add_lgbbound_best(tree, parent_cfg):
    seed_ids = _lineage_seed_ids(tree)
    if "LGBBOUND" not in seed_ids:
        return None
    lineage_id = seed_ids["LGBBOUND"]
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
    desc = (f"BLEND: add LGBBOUND's best variant so far (#{best['id']}, QWK="
            f"{-best['score']:.5f}) -> {len(child['members'])}-way -- new territory (v1 "
            f"never tested boundary-pushed hyperparams as a blend member) [PRIOR P18]")
    return child, desc


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
            f"[PRIOR P19]")
    return child, desc


def _blend_rounder_toggle(tree, parent_cfg):
    if parent_cfg.get("rounder") == "fold_avg":
        return None  # already tried on this exact member set
    child = dc(parent_cfg)
    child["rounder"] = "fold_avg"
    child.pop("result", None)
    desc = ("BLEND: ALTERNATIVE-ROUNDER toggle full_oof->fold_avg on the current best "
            "composition -- STATUS.md's exp #10 found this WORSE by hand on the linear "
            "run's 6-way blend (0.56449 vs 0.56769); honest re-check at this run's own "
            "pool maturity [PRIOR P4/P5]")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("XGB"),               # -> 3-way
    _mk_add_named("FEAT"),              # -> 4-way, reproduces v1's own best composition
    _mk_add_named("LGBNUDGE"),          # -> 5-way
    _mk_add_named("CATORIG"),           # -> 6-way
    _mk_add_named("LGBORIG"),           # -> 7-way
    _blend_add_lgbbound_best,           # -> new territory
    _blend_remove_weakest,              # leaner-blend probe
    _blend_rounder_toggle,              # alternative-rounder probe
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
    if parent_cfg.get("rounder") != "fold_avg":
        child = dc(parent_cfg)
        child["rounder"] = "fold_avg"
        child.pop("result", None)
        if find_dup(tree, child) is None:
            return child, "BLEND fallback: alternative-rounder toggle [PRIOR P4/P5]"
    weights = (parent_cfg.get("result") or {}).get("weights")
    members = parent_cfg["members"]
    if weights and len(members) > 2:
        weak_id = min(members, key=lambda m: weights.get(str(m), 1.0))
        child = dc(parent_cfg)
        child["members"] = [m for m in members if m != weak_id]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            return child, f"BLEND fallback: remove weakest #{weak_id} [PRIOR P19]"
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
def reuse_legacy(tree, mutation, cfg, v1_node_id, expected_qwk, is_root=False):
    nid = hv2.next_id(tree)
    oof, pred, score, coef = ev.load_v1_solo(v1_node_id, expected_qwk)
    hv2.cache_oof(ev.CACHE_DIR, nid, oof, pred=pred, y=ev._y, qwk=score)
    result = {"n_feats": None, "qwk": round(score, 5),
              "cutpoints": [round(float(c), 4) for c in coef], "rounder": "full_oof",
              "reused_from": f"v1 cache_s3e5/solo_{v1_node_id}.npz"}
    stored_cfg = {**cfg, "result": result}
    if is_root:
        real_nid = hv2.add_root(tree, mutation, stored_cfg, round(-score, 5), "evaluated", 0.0)
    else:
        real_nid, dup = hv2.add_node(tree, tree["root_id"], mutation, stored_cfg,
                                      round(-score, 5), "evaluated", 0.0, allow_duplicate=True)
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


def qwk_of(r):
    if r is None:
        return None
    if r.get("result") and "qwk" in r["result"]:
        return r["result"]["qwk"]
    return -r["score"] if r.get("score") is not None else None


# ---------------------------------------------------------------------------
_PRIOR_TAG_RE = re.compile(r"\[PRIOR (P\d+(?:/P\d+)*|none)")


def prior_usage_summary(tree):
    by_id = {n["id"]: n for n in tree["nodes"]}
    informed, uninformed = [], []
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["parent_id"] is None or n["parent_id"] == tree["root_id"]:
            continue
        m = _PRIOR_TAG_RE.search(n["mutation"])
        if not m:
            continue
        tag = m.group(1)
        parent = by_id.get(n["parent_id"])
        if parent is None or parent["score"] is None or n["score"] is None:
            continue
        won = n["score"] < parent["score"]
        (informed if tag != "none" else uninformed).append(dict(node_id=n["id"], tag=tag, won=won))

    def rate(lst):
        return (sum(1 for x in lst if x["won"]) / len(lst)) if lst else None

    return dict(informed=informed, uninformed=uninformed,
                informed_win_rate=rate(informed), uninformed_win_rate=rate(uninformed))


def verify_digit_for_digit():
    print("=== digit-for-digit v1-cache verification (BEFORE searching) ===")
    checks = [("ROOT(LGB_tuned)", 0, 0.56244)] + [(name, v1id, exp) for name, (v1id, exp, _) in LEGACY_SPECS.items()]
    for name, v1id, expected in checks:
        _oof, _pred, score, _coef = ev.load_v1_solo(v1id, expected)
        print(f"  {name} (v1 #{v1id}): recomputed QWK={score:.6f} vs historical {expected} "
              f"-> OK (|diff|={abs(score - expected):.2e})")
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

    # --- root: REUSED from v1's cache_s3e5/solo_0.npz ---
    if not tree["nodes"]:
        nid, r = reuse_legacy(tree, ROOT_MUTATION, ROOT_CONFIG, 0, 0.56244, is_root=True)
        print(f"[root] #{nid} QWK={r['qwk']} wall_s=0.0 (REUSED, verified digit-for-digit)")

    # --- 6 legacy pool members: REUSED from v1's cache_s3e5/*.npz ---
    for name, (v1id, expected, cfg) in LEGACY_SPECS.items():
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        mutation = f"[{name}] {LEGACY_DESCS[name]}"
        nid, r = reuse_legacy(tree, mutation, cfg, v1id, expected)
        print(f"[{name} seed] #{nid} QWK={r['qwk']} wall_s=0.0 (REUSED, verified digit-for-digit)")

    # --- 3 NEW first-generation SOLO lineages (real training) ---
    for name, seed_fn in NEW_SOLO_SEEDS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} QWK={qwk_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # --- BLEND lineage seed (needs root + CAT ids) ---
    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} QWK={qwk_of(r)} status={r['status']} wall_s={r['wall_s']}")

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
        gb_qwk = -gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"QWK={qwk_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_QWK={gb_qwk} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']} "
              f"| tie_rate={hv2.tie_rate(tree):.3f}")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    gb_qwk = -gb["score"] if gb else None
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
          f"{n_reused} reused-from-v1-cache), {len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} QWK={gb_qwk} mutation={gb['mutation'][:150]}")
    print(f"v1 tree-search best: {V1_TREE_BEST} (v2 "
          f"{'BEAT' if gb_qwk > V1_TREE_BEST else ('MATCHED' if gb_qwk == V1_TREE_BEST else 'did not beat')} it)")
    print(f"Linear-iteration best: {LINEAR_BEST} (v2 "
          f"{'BEAT' if gb_qwk > LINEAR_BEST else ('MATCHED' if gb_qwk == LINEAR_BEST else 'did not beat')} it)")
    print(f"Final tie_rate: {hv2.tie_rate(tree):.3f}  (adaptive plateau threshold=0.15)")
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
