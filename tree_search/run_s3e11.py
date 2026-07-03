"""tree_search/run_s3e11.py — drives the harness_v2 (Phase D-6, FINAL sweep comp)
candidate-tree search for playground-series-s3e11 (Media Campaign Cost, RMSLE, minimize).

Sweep question: v2 is 4/4 BEAT on the prior comps (s3e3 #12, s3e7 #13, s3e1 #9, s3e19
#22 despite the TimeSeriesSplit-CV curveball). s3e11 is the SCALE test: 360,336 train
rows (vs 1.7k-137k in every prior comp), where a solo eval genuinely costs 30-100s of
wall-clock (not the sub-20s of smaller comps) -- does the harness still deliver a win
under a node budget that MUST be blend-heavy rather than solo-heavy to fit ~30-35 min?

--- Phase D-6 scale lever (this run's own addition, no prior comp needed it) ---
competitions/playground-series-s3e11/scripts/cache/*.npz already holds the linear
iteration's best 5-way pool (LGB, CAT_orig, CAT_tuned, CAT_tuned_seed2024,
CAT_tuned_seed7 -- iterate2.py's own checkpoint cache) fully trained. Root + all 4
first-generation SOLO seeds below are REUSED directly from that cache via
eval_s3e11.load_legacy_solo() (load + digit-for-digit RMSLE recompute, asserted against
the historical STATUS.md value, NO retraining) -- this is the "verify root reproduces
digit-for-digit BEFORE searching" check the task brief asks for, done essentially for
free instead of costing ~10-12 min of redundant retraining. The freed budget goes to
THREE new solo variants (one more CAT seed, one deliberately-diverse deep LGB, one
feature-subtraction probe) plus blend-heavy weight-search exploration over the resulting
8-member solo pool -- no retraining is needed for any blend node, ever.

Per the task brief's explicit instruction: the auto_scale global-multiplier postprocess
idea from run_s3e19.py does NOT transfer here and is deliberately NOT re-tested --
auto_scale exists to correct a systematic level/scale mismatch that TimeSeriesSplit's
train/validate-on-later-dates asymmetry can introduce; plain KFold OOF has no such
structural level gap (every fold sees the same underlying distribution), so there is no
reason to expect a flat multiplier to help, and burning eval budget "just to check" isn't
worth it at this data scale where every eval matters more than usual.

--- Idea-injection experiment (recommendation #4) ---
harness_v2.suggest_priors(COMP_META) is called once at startup; COMP_META's keywords hit
knowledge/experience.md's "### RMSLE" (metric="rmsle" -- s3e11 is itself the sole
evidence source for that section), "### 低訊號資料(全部原始特徵 |r|≤0.11)" (tag
"低訊號" -- s3e11's own store_te finding), "## 超參調校(Optuna)" (tag "optuna" -- s3e11
is itself the 4th-competition validation of the fold-proxy-tune -> pool -> seed-bag
recipe AND the "large data wants MORE capacity" reversal finding), "## Ensemble/後處理"
(tag "ensemble"). Every mutation queue entry tags its description with `[PRIOR Pk]` when
directly shaped by one of those returned bullets, or `[PRIOR none]` otherwise.

--- Known constraints from the task brief (STATUS.md + experiments.json) ---
- Do NOT re-test per-combo feature means after store_te (R4, STATUS.md: harmful,
  0.295715->0.296200) -- store_te already exhausts the group-key's target signal, a
  non-target group aggregate on the same key is pure redundancy+noise for a tree model.
- Do NOT re-test XGB (weight-searched to 0 twice, dropped from the pool per R1).
- DO test (this run's own extensions, per the task brief's budget strategy): a 4th CAT
  seed (STATUS.md stopped at 3 seeds citing diminishing gains, -0.000067 on the 3rd --
  honest test of whether a 4th is genuinely exhausted or still marginally positive),
  a deliberately-diverse deep/high-capacity LGB variant (large-data-wants-capacity
  finding applied to LGB, which the linear run never tuned at all), a feature-SUBTRACTION
  probe (drop the two weakest engineered ratio features, mirroring experience.md's
  "features過多反而退步" pattern -- untested in the SUBTRACTION direction here, R4 only
  ever tested ADDING more features).

--- Digit-for-digit reproduction note ---
eval_s3e11.py's load_legacy_solo() loads scripts/cache/{lgb,cat_orig,cat_tuned,
cat_tuned_seed2024,cat_tuned_seed7}.npz and recomputes RMSLE via THIS module's own
rmsle_from_log on each, asserting agreement with STATUS.md's historical values (LGB
0.29661, CAT_orig 0.29618, CAT_tuned 0.29579, CAT_tuned_s2024 0.29591, CAT_tuned_s7
0.29578) to within 2e-4 BEFORE any node is added to the tree -- verified as part of
building this driver (see main()'s startup log).

--- v2-specific plumbing vs run_s3e1.py/run_s3e19.py --- (identical convention — see
those files' docstrings for the full rationale) `find_dup` compares result-STRIPPED
config hashes (with blend members SORTED) before spending compute on an eval; rejections
are logged to DEDUP_REJECTIONS and bump a per-lineage retry offset.

Root: CAT_tuned (STATUS.md exp #5-#8, Optuna fold-0-proxy-tuned CatBoost) -- solo OOF
RMSLE 0.29579, the single strongest member in the linear 5-way pool, picked as ROOT to
match the established convention from run_s3e1.py/run_s3e3.py/run_s3e7.py/run_s3e19.py
(root = the Optuna-found config itself).
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
import eval_s3e11 as ev  # noqa: E402

COMP = "playground-series-s3e11"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 24        # phase 1 ran to 20 (aim >=18, mostly blend per the scale budget);
                         # phase-2 resume bumps to 24 for the BLEND2 lineage below --
                         # phase 1's strongest discovery (#17, depth-12 CAT solo
                         # 0.295461, better than every phase-1 blend) landed AFTER the
                         # BLEND lineage had already plateaued at node #16, so it never
                         # entered any blend; BLEND2 re-blends the updated solo pool
                         # around it at pure weight-search cost (no retraining)
MAX_WALL_S = 30 * 60     # hard stop, matches the ~30-35 min training budget
EVAL_TIMEOUT_S = 300
LINEAR_BEST = 0.295648   # linear-iteration's best (exp #8, 5-way tuned-CAT-family blend, STATUS.md)
LINEAR_ROUNDS = 8        # STATUS.md exp #1-#8 (baseline + base + engineered + Phase B R1-R5)

COMP_META = {"metric": "rmsle", "tags": ["低訊號", "optuna", "ensemble"]}
# matches "### RMSLE", "### 低訊號資料(全部原始特徵 |r|≤0.11)", "## 超參調校(Optuna)",
# "## Ensemble/後處理"

DEDUP_REJECTIONS = []  # (attempted_mutation, existing_dup_id) log for the report
_dedup_offset = {}      # lineage_id -> extra propose_child idx offset from rejections


def dc(cfg):
    return copy.deepcopy(cfg)


def strip_result(cfg):
    """See run_s3e1.py's identical helper docstring: eval_and_add merges the evaluator's
    OUTPUT `result` dict into the stored config for record-keeping, which would make
    hv2.add_node's own built-in dedup never fire. This driver's own dedup check strips
    "result" first so it compares only the part of the config that was actually
    PROPOSED. Blend member lists are additionally SORTED before hashing so two blends
    over the same member SET reached in different append-order are recognized as
    semantic duplicates."""
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
# Root + reused-pool params: exact reproduction of STATUS.md exp #5-#8's 5-way pool
# ---------------------------------------------------------------------------
CAT_TUNED_PARAMS = dict(
    depth=10, learning_rate=0.08243442179862394, l2_leaf_reg=5.4822780685788235,
    min_data_in_leaf=33, random_strength=0.09160286047373326, random_seed=42,
)
ROOT_CONFIG = {
    "kind": "solo", "model": "cat", "params": dc(CAT_TUNED_PARAMS),
    "features": {"drop": []},
}
ROOT_MUTATION = ("root: s3e11 linear-winner's strongest solo config -- Optuna fold-0-"
                 "proxy-tuned CatBoost (40 trials, 319.3s, STATUS.md exp #5/Phase B "
                 "Round 2), solo OOF RMSLE 0.29579 (best single model in the linear "
                 "5-way pool, weight 0.4 in the final blend), REUSED from "
                 "scripts/cache/cat_tuned.npz (verified digit-for-digit, no retrain) "
                 "[PRIOR: 4-competition-validated 'Optuna fold-proxy tune -> pool -> "
                 "seed bag' recipe, s3e11 is itself one of the validating comps AND the "
                 "'large data wants MORE capacity' reversal finding's evidence source]")

LGB_ORIG_PARAMS = dict(n_estimators=2000, learning_rate=0.04, num_leaves=63,
                       min_child_samples=60, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
                       random_state=42)
CAT_ORIG_PARAMS = dict(depth=8, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42)

LEGACY_SPECS = {
    # name -> (legacy_cache_filename, expected historical RMSLE, config)
    "LGB_ORIG": ("lgb", 0.29661,
                 {"kind": "solo", "model": "lgb", "params": dc(LGB_ORIG_PARAMS),
                  "features": {"drop": []}}),
    "CAT_ORIG": ("cat_orig", 0.29618,
                 {"kind": "solo", "model": "cat", "params": dc(CAT_ORIG_PARAMS),
                  "features": {"drop": []}}),
    "CAT_S2024": ("cat_tuned_seed2024", 0.29591,
                  {"kind": "solo", "model": "cat",
                   "params": dict(dc(CAT_TUNED_PARAMS), random_seed=2024),
                   "features": {"drop": []}}),
    "CAT_S7": ("cat_tuned_seed7", 0.29578,
               {"kind": "solo", "model": "cat",
                "params": dict(dc(CAT_TUNED_PARAMS), random_seed=7),
                "features": {"drop": []}}),
}
LEGACY_DESCS = {
    "LGB_ORIG": ("model-type: hand-set LGB (exp #4-#8's original, non-Optuna params) -- "
                 "solo OOF 0.29661, weight 0 in every blend since R2 but kept for "
                 "diversity per experience.md's 'weight search often zeroes weak "
                 "members, that is a feature not a bug' [PRIOR none -- STATUS.md base "
                 "model, verbatim, REUSED from scripts/cache/lgb.npz]"),
    "CAT_ORIG": ("model-type: hand-set CatBoost (exp #4-#8's original, non-Optuna "
                 "params) -- solo OOF 0.29618, weight 0 in the final blend but the "
                 "2nd-strongest un-tuned base model [PRIOR none -- STATUS.md base "
                 "model, verbatim, REUSED from scripts/cache/cat_orig.npz]"),
    "CAT_S2024": ("seed variation: Optuna-tuned CatBoost, random_seed 42->2024 -- "
                  "STATUS.md R3's 2nd seed-bag member, solo OOF 0.29591, weight 0.2 in "
                  "the final 5-way blend [PRIOR P8 -- seed bagging is the cheapest "
                  "reliable post-tuning residual gain, 4-competition-validated recipe, "
                  "REUSED from scripts/cache/cat_tuned_seed2024.npz]"),
    "CAT_S7": ("seed variation: Optuna-tuned CatBoost, random_seed 42->7 -- STATUS.md "
               "R5's 3rd seed-bag member, solo OOF 0.29578 (essentially tied with the "
               "tuned root for strongest single model), weight 0.4 in the final blend "
               "[PRIOR P8 -- seed bagging, same evidence as CAT_S2024, REUSED from "
               "scripts/cache/cat_tuned_seed7.npz]"),
}


# ---------------------------------------------------------------------------
# Three NEW first-generation SOLO lineages -- the only ones needing real training among
# the first-gen seeds (root + 4 legacy members above cost ~0s each).
# ---------------------------------------------------------------------------
def seed_cat_s99():
    p = dict(dc(CAT_TUNED_PARAMS), random_seed=99)
    cfg = {"kind": "solo", "model": "cat", "params": p, "features": {"drop": []}}
    desc = ("UNTRIED per STATUS.md 'seed bagging saw diminishing gains (-0.000067 on "
            "the 3rd seed), stopped there' -- test a 4TH seed of the same tuned-CAT "
            "recipe (99, distinct from 42/2024/7) to check whether the pool is "
            "genuinely exhausted or still has marginal juice; honest test, not assumed "
            "to help [PRIOR P8 -- seed bagging recipe, extending s3e11's own 3-seed "
            "evidence one seed further]")
    return cfg, desc


def seed_deeplgb():
    p = dc(LGB_ORIG_PARAMS)
    p.update(num_leaves=255, min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
             reg_alpha=0.05, reg_lambda=0.1, learning_rate=0.05)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("model-variant: DELIBERATELY diverse deep/high-capacity/light-reg LGB "
            "(num_leaves 63->255, reg_lambda 1.0->0.1, reg_alpha 0.5->0.05, "
            "min_child_samples 60->20) -- direct application of s3e11's OWN Optuna "
            "finding that at 360k-row scale CatBoost's optimum reverses the usual "
            "small-data reg-favoring pattern (depth 10, lr 0.082, i.e. MORE capacity), "
            "tested here on LGB (which the linear run never tuned at all) as a "
            "deliberate diversity source now that XGB is banned [PRIOR P6 -- direct "
            "test of the large-data-wants-capacity direction on a model family that "
            "was never itself tuned]")
    return cfg, desc


def seed_featvariant():
    p = dc(CAT_ORIG_PARAMS)
    cfg = {"kind": "solo", "model": "cat", "params": p,
           "features": {"drop": ["children_away", "cars_per_child"]}}
    desc = ("feature-SUBTRACTION probe: drop children_away/cars_per_child (the two "
            "weakest engineered ratio features -- thin, noisy per-row ratios with no "
            "group structure, unlike store_te/amenity_count/sales_ratio/"
            "weight_per_case which either carry the main store_te signal or are "
            "well-populated physical ratios) from the hand-set CAT_orig baseline -- "
            "tests whether trimming redundant/weak engineered columns helps, mirroring "
            "experience.md's '特徵過多反而退步' pattern in the SUBTRACTION direction "
            "(R4 only ever tested feature ADDITION and that was harmful) [PRIOR P value "
            "-- 'features過多反而退步' from s3e7/s3e14/s3e9, untested as subtraction on "
            "s3e11's own engineered feature set]")
    return cfg, desc


NEW_SOLO_SEED_SPECS = [("CAT_S99", seed_cat_s99), ("DEEPLGB", seed_deeplgb),
                       ("FEATVARIANT", seed_featvariant)]
ALL_SOLO_NAMES = list(LEGACY_SPECS) + [n for n, _ in NEW_SOLO_SEED_SPECS]


# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    out = {}
    all_names = ALL_SOLO_NAMES + ["BLEND", "BLEND2"]
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name in all_names:
                if name not in out and n["mutation"].startswith(f"[{name}]"):
                    out[name] = n["id"]
    return out


def _best_node_in_lineage(tree, lineage_id):
    best = None
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["id"] == tree["root_id"]:
            continue
        if hv2.lineage_of(tree, n["id"]) == lineage_id:
            if best is None or n["score"] < best["score"]:
                best = n
    return best


def _id_for(tree, name):
    if name == "ROOT":
        return tree["root_id"]
    lineage_id = _lineage_seed_ids(tree)[name]
    best = _best_node_in_lineage(tree, lineage_id)
    return best["id"] if best else lineage_id


def solo_pool(tree):
    """All evaluated kind=='solo' node ids, best (lowest RMSLE) first."""
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["LGB_ORIG"], ids["CAT_ORIG"], ids["CAT_S2024"], ids["CAT_S7"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    desc = (f"ensemble seed: 5-way blend of root CAT_tuned(#{tree['root_id']}) + "
            f"LGB_ORIG(#{ids['LGB_ORIG']}) + CAT_ORIG(#{ids['CAT_ORIG']}) + "
            f"CAT_S2024(#{ids['CAT_S2024']}) + CAT_S7(#{ids['CAT_S7']}) CACHED OOF "
            f"(all 5 REUSED from scripts/cache, log1p-space), dirichlet weight search "
            f"-- reproduces the linear-iteration's exp #8 5-member pool under "
            f"harness_v2's ensemble machinery, no retraining [PRIOR none -- generic "
            f"ensemble-default node-space seed, the recommendation #1 mechanism itself]")
    return cfg, desc


def seed_blend2(tree):
    """Phase-2 resume lineage (see TARGET_NODES comment): re-blend the top of the
    UPDATED solo pool -- phase 1's global best #17 (depth-12 tuned-CAT, 0.295461) and
    its two seed variants were only discovered after the BLEND lineage plateaued, so no
    phase-1 blend ever saw them. Members = the 7 best solos by OOF RMSLE (computed
    dynamically from the tree, so this is resume-safe), which at phase-2 start is
    {#17 depth12_s7, #18 depth12_s3000, #19 depth12_s3001, #0 root CAT_tuned,
    #4 CAT_S7, #5 CAT_S99, #6 DEEPLGB} -- the depth-12 family plus the phase-1
    winners' diversity anchors."""
    members = solo_pool(tree)[:7]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    desc = (f"ensemble phase-2 seed: 7-way blend of the current top-7 solo pool "
            f"{members} (includes the depth-12 CAT family found AFTER the BLEND "
            f"lineage plateaued -- phase 1's best solo #17 never entered any blend), "
            f"dirichlet weight search over CACHED OOF, no retraining [PRIOR P8/P11 -- "
            f"add-to-pool-not-replace applied to the newly discovered strongest "
            f"family; weight search arbitrates]")
    return cfg, desc


def _mk_add_named(name, prior_tag):
    def fn(tree, parent_cfg):
        add_id = _id_for(tree, name)
        if add_id in parent_cfg["members"]:
            return None
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        desc = (f"BLEND: add {name} solo member (#{add_id}) -> {len(child['members'])}-way, "
                f"dirichlet weight search {prior_tag}")
        return child, desc
    return fn


def _blend_method_swap(tree, parent_cfg):
    child = dc(parent_cfg)
    cur = child.get("weight_search", "dirichlet")
    child["weight_search"] = "grid_simplex" if cur == "dirichlet" else "dirichlet"
    child.pop("result", None)
    n_members = len(child["members"])
    if child["weight_search"] == "grid_simplex" and n_members > 5:
        return None  # grid_simplex only supported up to 5 members (harness_v2 constraint)
    desc = (f"BLEND: weight-SEARCH-METHOD comparison {cur}->{child['weight_search']} on "
            f"the same {n_members} members -- grid_simplex is exhaustive (0.05-step "
            f"simplex, similar spirit to the linear run's own 0.1-step grid) vs "
            f"dirichlet's random+refine search [PRIOR none -- generic "
            f"ensemble-mechanism re-check]")
    return child, desc


def _blend_remove_weakest(tree, parent_cfg):
    prior = parent_cfg.get("result") or {}
    weights = prior.get("weights")
    members = parent_cfg["members"]
    if not weights or len(members) <= 2:
        return None
    idx = int(np.argmin(weights))
    weak_id = members[idx]
    child = dc(parent_cfg)
    child["members"] = [m for m in members if m != weak_id]
    child.pop("result", None)
    desc = (f"BLEND: remove lowest-weight member #{weak_id} (w={weights[idx]:.3f}) -> "
            f"leaner {len(child['members'])}-way, test whether it was truly adding "
            f"value or just being tolerated by the weight search [PRIOR P14 -- "
            f"removing (near-)zero-weight members is zero-cost, per experience.md "
            f"Ensemble/後處理; s3e11's own R1 already validated this for XGB]")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("CAT_S99", "[PRIOR P8 -- add-to-pool-not-replace, untried 4th-seed "
                             "idea, weight search arbitrates whether it helps at zero "
                             "extra blend-side cost]"),
    _mk_add_named("DEEPLGB", "[PRIOR none -- deliberate-diversity LGB as blend member, "
                             "large-data-wants-capacity finding applied via LGB instead "
                             "of a banned second model family]"),
    _mk_add_named("FEATVARIANT", "[PRIOR none -- feature-subtraction member, let weight "
                                 "search decide whether the leaner feature set adds "
                                 "value even if its solo score is worse]"),
    _blend_method_swap,
    _blend_remove_weakest,
]


def blend_fallback(tree, parent_cfg, attempt):
    """Post-queue blend mutations, each find_dup-checked HERE before being returned (see
    run_s3e1.py's identical rationale: never live-lock re-proposing the same duplicate).
    s3e11 has no postprocess lever (no top-coding/auto_scale analog here), so the last
    resort is simply exhausting solo_pool additions/removals; once that's dry, return
    None -> forced backtrack (a genuine plateau, expected and logged)."""
    pool = solo_pool(tree)
    for add_id in [nid for nid in pool if nid not in parent_cfg["members"]]:
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            desc = (f"BLEND fallback: add next-best unused pool member #{add_id} -> "
                    f"{len(child['members'])}-way [PRIOR none -- fallback]")
            return child, desc
    # try removing the single weakest-weighted member as a leaner alternative first
    prior = parent_cfg.get("result") or {}
    weights = prior.get("weights")
    members = parent_cfg["members"]
    if weights and len(members) > 2:
        idx = int(np.argmin(weights))
        child = dc(parent_cfg)
        child["members"] = [m for i, m in enumerate(members) if i != idx]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            desc = (f"BLEND fallback: remove weakest member #{members[idx]} (all "
                    f"additions duplicate) [PRIOR none -- fallback]")
            return child, desc
    return None  # genuinely exhausted -> forced backtrack


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    seed_key = "random_state" if c["model"] == "lgb" else "random_seed"
    c["params"] = dc(c["params"])
    c["params"][seed_key] = 3000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): "
            f"same config, alt {seed_key}={3000 + attempt} [PRIOR none -- fallback]")
    return c, desc


# ---------------------------------------------------------------------------
# Per-lineage authored mutation queues (SOLO): fn(parent_config) -> (child_config, desc)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"].update(params_update)
    return c


LGB_ORIG_QUEUE = [
    lambda c: (_bump(c, reg_alpha=1.5, reg_lambda=2.0),
               "LGB_ORIG: reg_alpha 0.5->1.5, reg_lambda 1.0->2.0 -- push MORE "
               "regularization (opposite direction from DEEPLGB) to bracket which "
               "capacity direction this un-tuned baseline prefers [PRIOR P6 -- direct "
               "test of capacity-direction]"),
    lambda c: (_bump(c, num_leaves=31, min_child_samples=100),
               "LGB_ORIG: num_leaves 63->31, min_child_samples 60->100 -- shallower/"
               "more regularized variant [PRIOR P6]"),
]

CAT_ORIG_QUEUE = [
    lambda c: (_bump(c, depth=10, learning_rate=0.08),
               "CAT_ORIG: depth 8->10, learning_rate 0.05->0.08 -- push toward "
               "CAT_tuned's own Optuna-found optimum from the un-tuned baseline "
               "[PRIOR P6 -- large-data-wants-capacity, this comp's own Optuna finding]"),
    lambda c: (_bump(c, l2_leaf_reg=6.0),
               "CAT_ORIG: l2_leaf_reg 3.0->6.0 -- regularization nudge [PRIOR none]"),
]

CAT_S2024_QUEUE = [
    lambda c: (_bump(c, l2_leaf_reg=8.0),
               "CAT_S2024: l2_leaf_reg 5.48->8.0, push L2 further while keeping "
               "seed=2024 [PRIOR none -- generic hand-nudge]"),
]

CAT_S7_QUEUE = [
    lambda c: (_bump(c, depth=12),
               "CAT_S7: depth 10->12, push capacity further while keeping seed=7 "
               "[PRIOR P6 -- continue the large-data-wants-capacity direction]"),
]

CAT_S99_QUEUE = [
    lambda c: (_bump(c, learning_rate=0.06),
               "CAT_S99: learning_rate 0.0824->0.06, slower-lr variant of the 4th-seed "
               "probe [PRIOR none -- generic hand-nudge]"),
]

DEEPLGB_QUEUE = [
    lambda c: (_bump(c, num_leaves=400, colsample_bytree=0.7),
               "DEEPLGB: push capacity further -- num_leaves 255->400, "
               "colsample_bytree 0.85->0.7 [PRIOR P6 -- continue the "
               "large-data-wants-capacity direction]"),
    lambda c: (_bump(c, learning_rate=0.07, subsample=0.7),
               "DEEPLGB: push diversity direction further -- lr 0.05->0.07, subsample "
               "0.85->0.7 [PRIOR none -- this run's own diversity-seeking direction, "
               "mirrors run_s3e19.py's DEEPLGB_QUEUE pattern]"),
]

FEATVARIANT_QUEUE = [
    lambda c: (dict(c, features={"drop": ["weight_per_case", "sales_ratio"]}),
               "FEATVARIANT: alternate subset -- drop weight_per_case/sales_ratio "
               "(the physical-ratio pair) instead, keep children_away/cars_per_child, "
               "to isolate WHICH engineered ratios (if any) are the redundant ones "
               "[PRIOR none -- continue the feature-reduction probe, opposite subset "
               "choice from the seed]"),
]

SOLO_QUEUES = {
    "LGB_ORIG": LGB_ORIG_QUEUE, "CAT_ORIG": CAT_ORIG_QUEUE, "CAT_S2024": CAT_S2024_QUEUE,
    "CAT_S7": CAT_S7_QUEUE, "CAT_S99": CAT_S99_QUEUE, "DEEPLGB": DEEPLGB_QUEUE,
    "FEATVARIANT": FEATVARIANT_QUEUE,
}
LINEAGE_NAMES = {}  # lineage_id (node id of the first-gen node) -> short name


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES[lineage_id]
    idx = hv2.lineage_size(tree, lineage_id) - 1 + _dedup_offset.get(lineage_id, 0)
    if name.startswith("BLEND"):  # BLEND and BLEND2 share the same mutation queue
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
        if idx < len(queue) and queue[idx] is not None:
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


# ---------------------------------------------------------------------------
def reuse_legacy(tree, mutation, cfg, legacy_name, expected_score, is_root=False):
    """Adds a node WITHOUT calling ev.evaluate() -- loads the already-trained OOF/pred
    from scripts/cache/<legacy_name>.npz via ev.load_legacy_solo (which itself asserts
    digit-for-digit consistency with the historical STATUS.md score), caches it into
    this driver's own harness_v2 cache dir under the predicted node id, and records
    wall_s=0.0 with a `reused_from` marker in the stored result. Mirrors eval_and_add's
    contract (predicted-id assertion, tree save) but skips training entirely."""
    nid = hv2.next_id(tree)
    oof, pred, score = ev.load_legacy_solo(legacy_name, expected_score)
    ev.hv2.cache_oof(ev.CACHE_DIR, nid, oof, pred=pred, rmsle=score)
    result = {"n_feats": len(ev.ALL_FEATURES) - len(cfg.get("features", {}).get("drop", [])),
              "rmsle": round(score, 6), "reused_from": f"scripts/cache/{legacy_name}.npz"}
    stored_cfg = {**cfg, "result": result}
    if is_root:
        real_nid = hv2.add_root(tree, mutation, stored_cfg, round(score, 6), "evaluated", 0.0)
    else:
        real_nid, dup = hv2.add_node(tree, tree["root_id"], mutation, stored_cfg,
                                      round(score, 6), "evaluated", 0.0, allow_duplicate=True)
        assert dup is None
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    hv2.save(tree, TREE_PATH)
    return real_nid, result


def eval_and_add(tree, parent_id, mutation, child_cfg, is_root=False, lineage_id=None):
    """Returns (nid_or_None, dup_id_or_None, result_or_None). On dedup rejection, nid is
    None, dup_id is the pre-existing node's id, no eval was run."""
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


def rmsle_of(r):
    if r is None:
        return None
    if r.get("result") and "rmsle" in r["result"]:
        return r["result"]["rmsle"]
    return r.get("score")


# ---------------------------------------------------------------------------
# Prior-usage analysis (idea-injection experiment): "win" = child's score beat its
# direct parent's score (lower RMSLE = improvement, no sign flip).
# ---------------------------------------------------------------------------
_PRIOR_TAG_RE = re.compile(r"\[PRIOR (P\d+|none)")


def prior_usage_summary(tree):
    by_id = {n["id"]: n for n in tree["nodes"]}
    informed, uninformed = [], []
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["parent_id"] is None or n["parent_id"] == tree["root_id"]:
            continue  # skip root and first-gen seeds (no in-lineage parent to compare against)
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


def main():
    t_start = time.time()
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

    # --- root: REUSED from scripts/cache/cat_tuned.npz (digit-for-digit verified) ---
    if not tree["nodes"]:
        nid, r = reuse_legacy(tree, ROOT_MUTATION, ROOT_CONFIG, "cat_tuned", 0.29579, is_root=True)
        print(f"[root] #{nid} RMSLE={r['rmsle']} wall_s=0.0 (REUSED, verified digit-for-digit)")

    # --- 4 legacy pool members: REUSED from scripts/cache/*.npz ---
    for name, (legacy_name, expected, cfg) in LEGACY_SPECS.items():
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        mutation = f"[{name}] {LEGACY_DESCS[name]}"
        nid, r = reuse_legacy(tree, mutation, cfg, legacy_name, expected)
        print(f"[{name} seed] #{nid} RMSLE={r['rmsle']} wall_s=0.0 (REUSED, verified digit-for-digit)")

    # --- 3 NEW first-generation SOLO lineages (real training) ---
    for name, seed_fn in NEW_SOLO_SEED_SPECS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} RMSLE={rmsle_of(r)} status={r['status']} wall_s={r['wall_s']}")

    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} RMSLE={rmsle_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # --- phase-2 lineage: BLEND2 over the updated solo pool (see TARGET_NODES note) ---
    already_blend2 = any(n["mutation"].startswith("[BLEND2]")
                         for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend2:
        cfg, desc = seed_blend2(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND2] {desc}", cfg)
        print(f"[BLEND2 seed] #{nid} RMSLE={rmsle_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = ALL_SOLO_NAMES + ["BLEND", "BLEND2"]
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in all_names:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name

    # --- adaptive tree-search loop ---
    iterations = 0
    while n_evaluated() < TARGET_NODES and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > 80:
            print("Safety cap on iterations reached, stopping.")
            break
        parent_id, lineage_id = hv2.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted (no more eligible expansions). Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        if child_cfg is None:  # lineage's mutation space exhausted -> forced backtrack
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
        gb_rmsle = gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"RMSLE={rmsle_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_RMSLE={gb_rmsle} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    gb_rmsle = gb["score"] if gb else None
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
    # first evaluated-node index (1-based) at which global_best first reached/beat LINEAR_BEST
    evals_to_match = None
    running_best = None
    for i, n in enumerate([n for n in tree["nodes"] if n["status"] == "evaluated"], start=1):
        s = n["score"]
        running_best = s if running_best is None else min(running_best, s)
        if running_best <= LINEAR_BEST and evals_to_match is None:
            evals_to_match = i
    tree["evals_to_match_linear_best"] = evals_to_match
    hv2.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend, "
          f"{n_reused} reused-from-legacy-cache), {len(tree['nodes'])} total, "
          f"wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} RMSLE={gb_rmsle} mutation={gb['mutation']}")
    print(f"Linear-iteration best: {LINEAR_BEST} over {LINEAR_ROUNDS} experiments "
          f"(tree {'BEAT' if gb_rmsle < LINEAR_BEST else ('MATCHED' if gb_rmsle == LINEAR_BEST else 'did not beat')} it)")
    print(f"Evals to match/beat linear best: {evals_to_match}")
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
