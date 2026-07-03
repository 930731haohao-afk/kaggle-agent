"""tree_search/run_s3e1.py — drives the harness_v2 (Phase D-4) candidate-tree search for
playground-series-s3e1 (California Housing, RMSE, minimize).

Sweep question (same for every Phase D comp): can harness v2 (ensemble-default node
space + experience-library priors + adaptive plateau + child dedup) match or beat the
linear-iteration best (0.557088, exp #7, 5-way weight-searched blend on 26 features) in
FEWER evaluations than linear iteration took rounds (7 experiments across 4 Phase-B
self-improvement rounds)? Sweep so far: s3e3 BEAT (eval#12), s3e7 BEAT (+0.00035,
eval#13; priors 62.5% win-rate).

--- Idea-injection experiment (recommendation #4) ---
harness_v2.suggest_priors(COMP_META) is called once at startup; COMP_META's keywords hit
knowledge/experience.md's "### RMSE/極偏態目標" (metric="rmse"), "### 地理座標資料" (tag
"地理" — s3e1's OWN geo-target-encoding-is-noise-level lesson AND its geo-feature-
engineering-still-has-headroom counter-lesson), "## 超參調校(Optuna)" (tag "optuna",
s3e1 is itself one of the 3-competition-validated "fold-proxy tune -> pool -> seed bag"
recipe's evidence sources), "## Ensemble/後處理" (tag "ensemble"). Every mutation queue
entry tags its description with `[PRIOR Pk]` when directly shaped by one of those
returned bullets, or `[PRIOR none]` otherwise.

--- Known constraints from the task brief (STATUS.md) ---
- Target top-coded at exactly 5.00001 for 4.92% of train rows (EDA finding #1) -- this
  caps achievable RMSE and motivates two new node kinds this run tests specifically:
  postprocess.clip (cheap OOF-level clip to [y.min(), 5.00001], applied INSIDE metric_fn
  for blends per harness_v2.eval_blend's contract) and a "ceiling_hybrid" solo model
  (ONE ambitious node: two-stage classifier-gates-regressor, see eval_s3e1.py's
  _run_ceiling_hybrid docstring for the honest-failure-is-a-valid-outcome framing).
- Geo target-encoding is noise-level here (exp #3, -0.00007) -- NOT re-tested. Geo
  feature engineering itself (KNN density + coastal distance, exp #6/#7) is the real,
  validated lever and is already baked into the fixed 26-feature set every node here
  trains on (no feature-ADDITION mutations attempted; feature-SUBTRACTION probes are
  in scope per the brief's "feature-subset probes" idea).

--- Digit-for-digit reproduction note ---
eval_s3e1.py prefers loading the linear run's own train_processed_v2.csv/
test_processed_v2.csv (byte-identical feature values) over recomputing features fresh,
because LightGBM's histogram split search proved (empirically, while wiring this up)
chaotically sensitive to ~1e-13-level floating point differences at num_leaves=121 --
recomputing from raw CSVs reproduced the SAME logic but shifted LGB_TUNED_SEED2024's OOF
RMSE from 0.558812 to 0.558552, a real ~0.00026 delta from pure float noise. Loading the
existing processed_v2 CSVs reproduces every STATUS.md base-model score exactly (verified
digit-for-digit: LGB 0.56073, XGB 0.56133, CAT 0.56106, LGB_TUNED 0.55887,
LGB_TUNED_SEED2024 0.55881, 5-way dirichlet blend 0.557085 vs linear's grid-search
0.557088) before this driver started searching.

--- v2-specific plumbing vs run_s3e14.py/run_s3e5.py (v1) --- (identical convention to
run_s3e3.py/run_s3e7.py — see those files' docstrings for the full rationale) `find_dup`
compares result-STRIPPED config hashes (with blend members SORTED) before spending
compute on an eval; rejections are logged to DEDUP_REJECTIONS and bump a per-lineage
retry offset.

Root: LGB, Optuna fold-0-proxy-tuned (STATUS.md exp #4, Phase B Round 1) — solo OOF
0.558866 (rounds to 0.55887), tied for strongest solo in the linear pool with its own
seed-bagged sibling (LGB_TUNED_SEED2024, 0.558812, weight 0.2037 in this run's dirichlet
blend) to within 0.00005 -- picked as ROOT (not the seed variant) to match the
established root convention from run_s3e3.py/run_s3e7.py (root = the Optuna-found
config itself; the seed-bagged copy is a first-generation SEEDBAG lineage, same as
those two runs).
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
import eval_s3e1 as ev  # noqa: E402

COMP = "playground-series-s3e1"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 22        # aim >=18 evaluated nodes (mix of solo + blend)
MAX_WALL_S = 30 * 60     # hard stop, matches the ~30 min training budget
EVAL_TIMEOUT_S = 200
LINEAR_BEST = 0.557088   # linear-iteration's best (exp #7, 5-way grid-search blend, STATUS.md)
LINEAR_ROUNDS = 7        # STATUS.md exp #1-#7 (baseline + TE probe + 4 Phase-B rounds)

COMP_META = {"metric": "rmse", "tags": ["地理", "optuna", "ensemble"]}
# matches "### RMSE/極偏態目標", "### 地理座標資料", "## 超參調校(Optuna)", "## Ensemble/後處理"

DEDUP_REJECTIONS = []  # (attempted_mutation, existing_dup_id) log for the report
_dedup_offset = {}      # lineage_id -> extra propose_child idx offset from rejections


def dc(cfg):
    return copy.deepcopy(cfg)


def strip_result(cfg):
    """See run_s3e3.py's identical helper docstring: eval_and_add merges the evaluator's
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
# Root: exact reproduction of STATUS.md exp #4/#7 (Optuna fold-0-proxy-tuned LGB)
# ---------------------------------------------------------------------------
LGB_TUNED_PARAMS = dict(
    learning_rate=0.011615865989246453, num_leaves=121, max_depth=10,
    min_child_samples=82, subsample=0.6523068845866853,
    colsample_bytree=0.5488360570031919, reg_alpha=0.5456725485601477,
    reg_lambda=0.057624872164786026, n_estimators=2000, random_state=42,
    verbosity=-1, objective="rmse",
)
ROOT_CONFIG = {
    "kind": "solo", "model": "lgb", "params": dc(LGB_TUNED_PARAMS),
    "features": {"drop": []},
}
ROOT_MUTATION = ("root: s3e1 linear-winner's strongest solo config -- Optuna fold-0-"
                 "proxy-tuned LGB (50 trials, 74.3s, STATUS.md exp #4/Phase B Round 1), "
                 "solo OOF RMSE 0.558866 (rounds to 0.55887), weight 0.2607 in this "
                 "run's 5-way dirichlet blend (0.557085, matching linear's grid-search "
                 "0.557088) [PRIOR: 3-competition-validated 'Optuna fold-proxy tune -> "
                 "pool -> seed bag' recipe, s3e1 is itself one of the validating comps]")

LGB_ORIG_PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                       min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1,
                       objective="rmse")
XGB_ORIG_PARAMS = dict(n_estimators=2000, learning_rate=0.03, max_depth=7,
                       min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                       objective="reg:squarederror", tree_method="hist", verbosity=0)
CAT_ORIG_PARAMS = dict(iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                       random_seed=42, loss_function="RMSE", verbose=False)


# ---------------------------------------------------------------------------
# Seven first-generation SOLO lineage seeds.
# ---------------------------------------------------------------------------
def seed_lgborig():
    cfg = {"kind": "solo", "model": "lgb", "params": dc(LGB_ORIG_PARAMS), "features": {"drop": []}}
    desc = ("model-variant: exp #2/#7's original (non-Optuna) LGB, unchanged from "
            "STATUS.md base models -- solo OOF 0.56073, cheap diversity source, weight "
            "0.1927 in the linear pool [PRIOR none -- STATUS.md base model, verbatim]")
    return cfg, desc


def seed_xgborig():
    cfg = {"kind": "solo", "model": "xgb", "params": dc(XGB_ORIG_PARAMS), "features": {"drop": []}}
    desc = ("model-type: hand-set XGB, unchanged from STATUS.md exp #2/#7 base models -- "
            "solo OOF 0.56133, weakest learner but kept for diversity (weight 0.0788 in "
            "this run's blend) [PRIOR P17 -- weight search often zeroes/near-zeroes weak "
            "members, that's a feature not a bug; kept as blend-diversity seed]")
    return cfg, desc


def seed_catorig():
    cfg = {"kind": "solo", "model": "cat", "params": dc(CAT_ORIG_PARAMS), "features": {"drop": []}}
    desc = ("model-type: hand-set CatBoost, unchanged from STATUS.md exp #2/#7 base "
            "models -- solo OOF 0.56106, weight 0.264 in the linear pool (2nd-highest "
            "solo weight after root) [PRIOR none -- STATUS.md base model, verbatim]")
    return cfg, desc


def seed_seedbag():
    p = dc(LGB_TUNED_PARAMS)
    p["random_state"] = 2024
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("seed variation: LGB_tuned params, random_state 42->2024 -- STATUS.md exp #5's "
            "5th blend member, solo OOF 0.558812 (marginally the single strongest solo in "
            "the whole pool, tied with root to within 0.00005) [PRIOR P8 -- seed bagging "
            "is the cheapest reliable post-tuning residual gain, s3e1 is itself one of "
            "the 3-competition-validated recipe's evidence sources]")
    return cfg, desc


def seed_regnudge():
    p = dc(LGB_TUNED_PARAMS)
    p["reg_alpha"] = 1.2
    p["num_leaves"] = 80
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("regularization nudge on the tuned-LGB hyperparams: reg_alpha 0.546->1.2, "
            "num_leaves 121->80 (push further regularized/shallower along a direction "
            "Optuna itself found competitive at max_depth=10) -- 37k rows is medium "
            "scale, testing whether s3e11's 'large data wants MORE capacity not less' "
            "finding or s3e7's 'small/medium data wants regularization' finding transfers "
            "here [PRIOR P6 -- direct test of which capacity-direction prior applies at "
            "this data scale]")
    return cfg, desc


def seed_ceiling():
    cfg = {"kind": "solo", "model": "ceiling_hybrid",
           "params": {"reg_params": dc(LGB_TUNED_PARAMS),
                      "clf_params": {"num_leaves": 31, "learning_rate": 0.05,
                                     "n_estimators": 500, "random_state": 42},
                      "mode": "soft", "threshold": 0.5},
           "features": {"drop": []}}
    desc = ("model-type: TWO-STAGE ceiling-aware hybrid (this run's ONE ambitious node "
            "per the brief) -- binary classifier for 'target>=5.00001' (OOF clf_auc "
            "~0.964) soft-blends its base LGB_tuned regressor toward the ceiling, "
            "weighted by predicted cap-probability. Honest test, not assumed to help: "
            "only 4.92% of rows are truly capped, so a less-than-precise classifier can "
            "easily net NEGATIVE by nudging uncapped rows upward too [PRIOR none -- this "
            "run's own ambitious probe, no experience.md precedent for a 2-stage "
            "ceiling-classifier]")
    return cfg, desc


def seed_xgbdiv():
    p = dc(XGB_ORIG_PARAMS)
    p["max_depth"] = 9
    p["learning_rate"] = 0.06
    p["colsample_bytree"] = 0.6
    p["reg_lambda"] = 0.3
    cfg = {"kind": "solo", "model": "xgb", "params": p, "features": {"drop": []}}
    desc = ("model-variant: XGB pushed DEEPER/higher-lr/lighter-reg (depth 7->9, lr "
            "0.03->0.06, colsample 0.8->0.6, reg_lambda 1.0->0.3) -- deliberately "
            "diversifying AWAY from LGB_tuned's optimum rather than re-tuning toward it "
            "(s3e7's winning lever: exp #5 showed tuning a 2nd model toward the same "
            "objective converges structure and REGRESSES the blend via lost diversity) "
            "[PRIOR none -- direct application of s3e7's diversity-preservation lesson "
            "to a comp where it was never tried]")
    return cfg, desc


SOLO_SEED_SPECS = [("LGBORIG", seed_lgborig), ("XGBORIG", seed_xgborig),
                    ("CATORIG", seed_catorig), ("SEEDBAG", seed_seedbag),
                    ("REGNUDGE", seed_regnudge), ("CEILING", seed_ceiling),
                    ("XGBDIV", seed_xgbdiv)]


# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    out = {}
    all_names = [n for n, _ in SOLO_SEED_SPECS] + ["BLEND"]
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
    """All evaluated kind=='solo' node ids (includes ceiling_hybrid), best (lowest RMSE)
    first."""
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["LGBORIG"], ids["XGBORIG"], ids["CATORIG"], ids["SEEDBAG"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    desc = (f"ensemble seed: 5-way blend of root LGB_tuned(#{tree['root_id']}) + "
            f"LGBORIG(#{ids['LGBORIG']}) + XGBORIG(#{ids['XGBORIG']}) + "
            f"CATORIG(#{ids['CATORIG']}) + SEEDBAG(#{ids['SEEDBAG']}) CACHED OOF, "
            f"dirichlet weight search -- reproduces the linear-iteration's exp #7 "
            f"5-member pool under harness_v2's ensemble machinery, no retraining "
            f"[PRIOR none -- generic ensemble-default node-space seed, the "
            f"recommendation #1 mechanism itself]")
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
    desc = (f"BLEND: weight-SEARCH-METHOD comparison {cur}->{child['weight_search']} on "
            f"the same {len(child['members'])} members -- grid_simplex is exhaustive "
            f"(0.05-step simplex, matches the linear run's own grid search exactly) vs "
            f"dirichlet's random+refine search; re-check whether either method finds a "
            f"meaningfully different optimum [PRIOR none -- generic ensemble-mechanism "
            f"re-check, RMSE has no rank-vs-prob analog since it's not a ranking metric]")
    return child, desc


def _blend_clip_toggle(tree, parent_cfg):
    child = dc(parent_cfg)
    cur_pp = child.get("postprocess") or {}
    cur_clip = bool(cur_pp.get("clip", False))
    child["postprocess"] = {"clip": not cur_clip}
    child.pop("result", None)
    desc = (f"BLEND: TOP-CODE-AWARE postprocess toggle clip={cur_clip}->{not cur_clip} -- "
            f"clip blended OOF to [y.min(), 5.00001] BEFORE scoring (applied inside "
            f"metric_fn per harness_v2.eval_blend's contract, so the weight search itself "
            f"is clip-aware, not just the winner) -- STATUS.md's linear run only ever "
            f"clipped the final TEST submission, never the OOF used to select/score "
            f"models; solo-level clip test showed a real +0.00018 gain (0.558866-> "
            f"0.558683 on root alone) [PRIOR none -- this run's own top-code-aware node, "
            f"direct response to STATUS.md EDA finding #1]")
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
            f"leaner {len(child['members'])}-way, test whether it was truly adding value "
            f"or just being tolerated by the weight search [PRIOR P14 -- removing "
            f"(near-)zero-weight members is zero-cost, per experience.md Ensemble/後處理]")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("REGNUDGE", "[PRIOR P8 -- add-to-pool-not-replace, weight search "
                               "arbitrates at zero cost]"),
    _mk_add_named("XGBDIV", "[PRIOR none -- deliberate-diversity XGB as blend member, "
                             "s3e7's winning lever applied here]"),
    _blend_clip_toggle,
    _mk_add_named("CEILING", "[PRIOR none -- ambitious-node member, let weight search "
                              "decide whether the ceiling hybrid adds value in blend "
                              "even if its solo score is worse]"),
    _blend_method_swap,
    _blend_remove_weakest,
]


def blend_fallback(tree, parent_cfg, attempt):
    """Post-queue blend mutations, each find_dup-checked HERE before being returned (see
    run_s3e7.py's identical rationale: never live-lock re-proposing the same duplicate)."""
    pool = solo_pool(tree)
    for add_id in [nid for nid in pool if nid not in parent_cfg["members"]]:
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            desc = (f"BLEND fallback: add next-best unused pool member #{add_id} -> "
                    f"{len(child['members'])}-way [PRIOR none -- fallback]")
            return child, desc
    child = dc(parent_cfg)
    cur_pp = child.get("postprocess") or {}
    child["postprocess"] = {"clip": not bool(cur_pp.get("clip", False))}
    child.pop("result", None)
    if find_dup(tree, child) is None:
        desc = "BLEND fallback: all member additions duplicate; clip toggle instead [PRIOR none -- fallback]"
        return child, desc
    return None  # genuinely exhausted -> forced backtrack


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    if c["model"] == "ceiling_hybrid":
        c["params"] = dc(c["params"])
        c["params"]["threshold"] = round(max(0.1, 0.5 - 0.1 * (attempt + 1)), 2)
        desc = (f"fallback threshold sweep (queue exhausted): ceiling_hybrid threshold "
                f"-> {c['params']['threshold']} [PRIOR none -- fallback]")
        return c, desc
    seed_key = "random_state" if c["model"] in ("lgb", "xgb") else "random_seed"
    c["params"][seed_key] = 3000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): same "
            f"config, alt {seed_key}={3000 + attempt} [PRIOR none -- fallback]")
    return c, desc


# ---------------------------------------------------------------------------
# Per-lineage authored mutation queues (SOLO): fn(parent_config) -> (child_config, desc)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"].update(params_update)
    return c


LGBORIG_QUEUE = [
    lambda c: (_bump(c, reg_alpha=0.5, reg_lambda=0.5),
               "LGBORIG: reg_alpha/reg_lambda 0.1->0.5 -- regularization nudge on the "
               "un-tuned baseline [PRIOR none -- generic hand-nudge]"),
    lambda c: (_bump(c, num_leaves=31, min_child_samples=40),
               "LGBORIG: num_leaves 63->31, min_child_samples 20->40 -- shallower/more "
               "regularized variant of the un-tuned baseline [PRIOR none]"),
]

XGBORIG_QUEUE = [
    lambda c: (_bump(c, reg_alpha=0.5, reg_lambda=2.0),
               "XGBORIG: reg_alpha 0.1->0.5, reg_lambda 1.0->2.0 -- regularization nudge "
               "[PRIOR none -- generic hand-nudge]"),
    lambda c: (_bump(c, max_depth=5),
               "XGBORIG: max_depth 7->5 -- shallower variant [PRIOR none]"),
]

CATORIG_QUEUE = [
    lambda c: (_bump(c, l2_leaf_reg=6.0),
               "CATORIG: l2_leaf_reg 3.0->6.0 -- regularization nudge [PRIOR none]"),
    lambda c: (_bump(c, depth=6),
               "CATORIG: depth 8->6 -- shallower CatBoost variant [PRIOR none]"),
]

SEEDBAG_QUEUE = [
    lambda c: (_bump(c, random_state=777),
               "SEEDBAG: second seed variant (2024->777) [PRIOR P8 -- add-to-pool + "
               "seed-bag, diminishing but real gains]"),
    lambda c: (_bump(c, reg_lambda=0.15),
               "SEEDBAG: reg_lambda 0.0576->0.15, push L2 further while keeping the "
               "seed=2024 bag [PRIOR none -- generic hand-nudge]"),
]

REGNUDGE_QUEUE = [
    lambda c: (_bump(c, reg_alpha=2.5, num_leaves=50),
               "REGNUDGE: push further -- reg_alpha 1.2->2.5, num_leaves 80->50 [PRIOR "
               "P6 -- continue whichever capacity-direction the first nudge favored]"),
    lambda c: (_bump(c, min_child_samples=120),
               "REGNUDGE: min_child_samples 82->120 -- regularize via leaf-size instead "
               "of leaf-count [PRIOR P6 -- same direction, different knob]"),
]

CEILING_QUEUE = [
    lambda c: (dict(c, params=dict(c["params"], mode="hard")),
               "CEILING: mode soft->hard (discrete override instead of confidence-"
               "weighted nudge) -- test whether the soft blend's presumably-gentler "
               "errors on misclassified rows actually matter vs a hard cutover [PRIOR "
               "none -- this run's own probe]"),
    lambda c: (dict(c, params=dict(c["params"], threshold=0.3)),
               "CEILING: classifier threshold 0.5->0.3 -- more permissive capped-row "
               "flagging (lower precision but higher recall on the true 4.92% capped "
               "rows) [PRIOR none]"),
]

XGBDIV_QUEUE = [
    lambda c: (_bump(c, random_state=2024),
               "XGBDIV: seed variation (random_state 42->2024) on the deliberate-"
               "diversity params [PRIOR P8 -- add-to-pool + seed-bag]"),
    lambda c: (_bump(c, learning_rate=0.08, subsample=0.7),
               "XGBDIV: push diversity direction further -- lr 0.06->0.08, subsample "
               "0.8->0.7 [PRIOR none -- this run's own diversity-seeking direction]"),
]

SOLO_QUEUES = {"LGBORIG": LGBORIG_QUEUE, "XGBORIG": XGBORIG_QUEUE, "CATORIG": CATORIG_QUEUE,
               "SEEDBAG": SEEDBAG_QUEUE, "REGNUDGE": REGNUDGE_QUEUE,
               "CEILING": CEILING_QUEUE, "XGBDIV": XGBDIV_QUEUE}
LINEAGE_NAMES = {}  # lineage_id (node id of the first-gen node) -> short name


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
        if idx < len(queue) and queue[idx] is not None:
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


# ---------------------------------------------------------------------------
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


def rmse_of(r):
    if r is None:
        return None
    if r.get("result") and "rmse" in r["result"]:
        return r["result"]["rmse"]
    return r.get("score")


# ---------------------------------------------------------------------------
# Prior-usage analysis (idea-injection experiment): "win" = child's score beat its
# direct parent's score (lower RMSE = improvement, no sign flip -- unlike the AUC comps).
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

    # --- root ---
    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} RMSE={rmse_of(r)} wall_s={r['wall_s']}")

    # --- seed the 7 first-generation SOLO lineages ---
    for name, seed_fn in SOLO_SEED_SPECS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} RMSE={rmse_of(r)} status={r['status']} wall_s={r['wall_s']}")

    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} RMSE={rmse_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = [n for n, _ in SOLO_SEED_SPECS] + ["BLEND"]
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
        gb_rmse = gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"RMSE={rmse_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_RMSE={gb_rmse} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    gb_rmse = gb["score"] if gb else None
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
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

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} RMSE={gb_rmse} mutation={gb['mutation']}")
    print(f"Linear-iteration best: {LINEAR_BEST} over {LINEAR_ROUNDS} experiments "
          f"(tree {'BEAT' if gb_rmse < LINEAR_BEST else ('MATCHED' if gb_rmse == LINEAR_BEST else 'did not beat')} it)")
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
