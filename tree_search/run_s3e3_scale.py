"""tree_search/run_s3e3_scale.py — Phase E-5 SCALING experiment: how does tree-search
quality scale with node budget? Extends the SAME harness_v2 (Phase D-2) search regime
that produced competitions/playground-series-s3e3/experiments_tree.json (22 nodes, best
AUC 0.841442) to an 80-evaluated-node budget, in a FRESH tree
(competitions/playground-series-s3e3/experiments_tree_scale.json) so the D-2 tree/run
script are never touched.

Design (see docs/scaling_experiment.md for the write-up):
  - Same root (Optuna-tuned LGB), same 6 first-generation SOLO lineage seeds, same
    BLEND seed as run_s3e3.py's D-2 run -- imported directly from run_s3e3 (module `d2`
    below) so configs are BYTE-IDENTICAL, which is what makes cache-reuse verification
    meaningful (recommendation below).
  - Cache reuse: run_s3e3.py already cached these 7 nodes' OOF vectors under
    tree_search/cache_s3e3/. This run redirects its OWN caching to a SEPARATE directory
    (tree_search/cache_s3e3_scale/) so it can never overwrite/corrupt the D-2 cache, but
    for the 7 identical-config seed nodes it copies the D-2 cache file over after
    DIGIT-VERIFYING it (recompute AUC from the cached OOF array, compare to the D-2
    tree's stored score to 6 decimals) -- skipping a real retrain for those 7 nodes.
  - Mutation policy: same per-lineage authored queues as D-2 (imported from run_s3e3),
    EXTENDED with boundary-push / joint-move / FEATPRUNE items (the exploit phase, evals
    1-40) and, once the budget crosses 40 evaluated nodes, six new EXPLORE-phase
    long-shot lineages seeded directly off the root (extreme regularization, contrarian
    CatBoost depth, aggressive joint feature pruning, and two "kitchen-sink" blends of
    every solo node evaluated so far -- one at k=800 dirichlet draws instead of D-2's
    k=1500, one in rank-space). Every node's mutation string carries a
    "[PHASE exploit|explore]" tag (in addition to the existing "[PRIOR ...]" tag D-2
    already used) so the phase is machine-extractable straight from the tree JSON.
  - Per-node cumulative-best curve: every evaluated node appends an entry to
    tree["curve"] (eval_index, node_id, lineage, phase, auc, cum_best_auc,
    is_new_global_best, wall_s, cum_wall_s) so docs/scaling_experiment.md's curve table
    is read directly off the tree JSON, no recomputation needed.

Budget: TARGET_NODES=80 evaluated nodes, MAX_WALL_S hard stop (same order as D-2's).
"""
import copy
import os
import shutil
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2       # noqa: E402
import eval_s3e3 as ev         # noqa: E402
import run_s3e3 as d2          # noqa: E402 -- the D-2 driver; we only call generic
                                # tree->tree helper functions from it (never d2.main(),
                                # never d2.eval_and_add / d2.TREE_PATH), so the D-2 tree
                                # file is never touched by this script.

COMP = "playground-series-s3e3"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_scale.json")
D2_TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree.json")

D2_CACHE_DIR = os.path.join(_HERE, "cache_s3e3")           # D-2's cache -- READ ONLY here
SCALE_CACHE_DIR = os.path.join(_HERE, "cache_s3e3_scale")  # this run's own cache dir
ev.CACHE_DIR = SCALE_CACHE_DIR  # redirect eval_s3e3's module-global cache dir at runtime
                                # (no file edit -- see module docstring) so nothing this
                                # script evaluates can ever collide with/overwrite a D-2
                                # cache file.

TARGET_NODES = 80
MAX_WALL_S = 25 * 60
EVAL_TIMEOUT_S = 120
EXPLOIT_CUTOFF = 40   # evals 1-40 tagged "exploit", 41-80 tagged "explore"
LINEAR_BEST = 0.838140
D2_BEST = 0.841442    # run_s3e3.py / experiments_tree.json global best (node #11)

COMP_META = d2.COMP_META

DEDUP_REJECTIONS = []
_dedup_offset = {}
LINEAGE_NAMES = {}
CURVE = []            # tree["curve"]: per-eval cumulative-best log
_explore_injected = {"done": False}


def dc(cfg):
    return copy.deepcopy(cfg)


def n_evaluated(tree):
    return sum(1 for n in tree["nodes"] if n["status"] == "evaluated")


# ---------------------------------------------------------------------------
# Digit-verified cache reuse for the 7 seed nodes (root + 6 solo lineages).
# BLEND seed (#7) is always re-evaluated fresh: blend nodes are <1s (no training) and
# aren't cached anyway, so there's nothing to reuse.
# ---------------------------------------------------------------------------
def _load_d2_tree():
    if not os.path.exists(D2_TREE_PATH):
        return None
    return hv2.load(D2_TREE_PATH)


_D2_TREE = _load_d2_tree()


def try_reuse_from_d2(node_id: int, config: dict):
    """If `config` (a solo node) is byte-identical to some D-2 node's config AND that
    D-2 node has a cached OOF whose recomputed AUC matches the D-2 tree's stored score
    to 6 decimals, copy the cache file into SCALE_CACHE_DIR under `node_id` and return a
    ready-made eval result dict WITHOUT calling ev.evaluate (skips the retrain). Returns
    None if no digit-verified match exists (caller should fall back to a real eval)."""
    if _D2_TREE is None or config.get("kind") != "solo":
        return None
    target_hash = hv2.config_hash(d2.strip_result(config))
    for n in _D2_TREE["nodes"]:
        if hv2.config_hash(d2.strip_result(n["config"])) != target_hash:
            continue
        src = os.path.join(D2_CACHE_DIR, f"solo_{n['id']}.npz")
        if not os.path.exists(src):
            return None
        d = np.load(src, allow_pickle=True)
        oof = d["oof"]
        recomputed_auc = round(float(ev.auc(ev._y, oof)), 6)
        d2_auc = round(-n["score"], 6)
        if abs(recomputed_auc - d2_auc) >= 1e-6:
            return None  # not digit-verified -- fold assignment or config drifted, don't trust it
        os.makedirs(SCALE_CACHE_DIR, exist_ok=True)
        dst = os.path.join(SCALE_CACHE_DIR, f"solo_{node_id}.npz")
        shutil.copyfile(src, dst)
        result = dict(n["config"].get("result") or {})
        result["auc"] = recomputed_auc
        result["reused_from_d2_node"] = n["id"]
        result["reuse_digit_verified"] = True
        return dict(status="evaluated", score=round(-recomputed_auc, 6), wall_s=0.0,
                    result=result, error=None)
    return None


# ---------------------------------------------------------------------------
# k=800 blend weight-search variant (a boundary-push on D-2's hardcoded k=1500):
# hv2.eval_blend's k isn't exposed through eval_s3e3.evaluate_blend's config dispatch,
# so this calls harness_v2's generic ensemble machinery directly instead of routing
# through ev.evaluate -- still reusing eval_s3e3.py's `auc`/`_y`/CACHE_DIR machinery,
# just with a different weight_search budget.
# ---------------------------------------------------------------------------
def eval_blend_k(config, node_id=None, k=800):
    t0 = time.time()
    members = config["members"]

    def neg_auc(vec):
        return -ev.auc(ev._y, vec)

    best_w, best_neg, _oofs = hv2.eval_blend(ev.CACHE_DIR, members, neg_auc, weight_search="dirichlet", k=k)
    best_score = -best_neg
    wall = time.time() - t0
    result = dict(members=members, weights=[round(float(w), 4) for w in best_w],
                  method=f"dirichlet_k{k}", space="prob", auc=round(best_score, 6))
    return dict(status="evaluated", score=round(-best_score, 6), wall_s=round(wall, 1),
                result=result, error=None)


def run_eval(child_cfg, node_id):
    if child_cfg.get("kind") == "blend" and child_cfg.get("weight_search", "").startswith("dirichlet_k"):
        k = int(child_cfg["weight_search"].replace("dirichlet_k", ""))
        return eval_blend_k(child_cfg, node_id=node_id, k=k)
    return ev.evaluate(child_cfg, node_id=node_id, timeout_s=EVAL_TIMEOUT_S)


# ---------------------------------------------------------------------------
# EXPLOIT-phase queue extensions: D-2's authored queues + boundary-push / joint-move /
# FEATPRUNE items that push further along the SAME directions D-2 already found working
# (see knowledge/experience.md PRIOR P5: small-sample + correlated features -> regularize
# over capacity) instead of proposing brand-new hypotheses.
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"].update(params_update)
    return c


LGBORIG_QUEUE_EXT = d2.LGBORIG_QUEUE + [
    lambda c: (_bump(c, reg_lambda=8.0),
               "LGBORIG boundary-push: reg_lambda 4.0->8.0, further along the known-good "
               "L2 direction [PRIOR P5]"),
    lambda c: (_bump(c, num_leaves=4, min_child_samples=80),
               "LGBORIG boundary-push: num_leaves 5->4, min_child_samples 50->80 "
               "[PRIOR P5]"),
]

XGBTUNED_QUEUE_EXT = d2.XGBTUNED_QUEUE + [
    lambda c: (_bump(c, max_depth=1),
               "XGBTUNED boundary-push: max_depth 2->1, maximal-shallow stumps [PRIOR P5]"),
    lambda c: (_bump(c, subsample=0.5, colsample_bytree=0.6, reg_lambda=3.0),
               "XGBTUNED joint-move: subsample 0.688->0.5 + colsample 0.83->0.6 + "
               "reg_lambda 0.84->3.0 together [PRIOR P5]"),
]

SEEDBAG_QUEUE_EXT = d2.SEEDBAG_QUEUE + [
    lambda c: (_bump(c, reg_lambda=0.05),
               "SEEDBAG boundary-push: reg_lambda 0.01->0.05, further along the tuned L2 "
               "direction [PRIOR P5]"),
    lambda c: (_bump(c, num_leaves=2, reg_alpha=0.15),
               "SEEDBAG joint-move: num_leaves 3->2 + reg_alpha 0.068->0.15 together "
               "[PRIOR P5]"),
]

_LE_COLS = [f"{c}_le" for c in
            ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole",
             "MaritalStatus", "OverTime"]]
_RATE_COLS = ["DailyRate", "HourlyRate", "MonthlyRate"]

FEAT_QUEUE_EXT = d2.FEAT_QUEUE + [
    lambda c: (dict(c, features={"drop": c["features"]["drop"] + _RATE_COLS}),
               f"FEATPRUNE: drop the 3 noisy *Rate columns ({', '.join(_RATE_COLS)}) on "
               "top of the current drop list [PRIOR none -- untested]"),
    lambda c: (dict(c, features={"drop": c["features"]["drop"] + _LE_COLS}),
               "FEATPRUNE joint-move: drop all 7 redundant *_le label-encoded columns, "
               "keep the *_freq encoding of the same categoricals only [PRIOR none -- "
               "untested, generic collinear-encoding hygiene]"),
]

SOLO_QUEUES_EXT = {"LGBORIG": LGBORIG_QUEUE_EXT, "XGBTUNED": XGBTUNED_QUEUE_EXT,
                   "CATORIG": d2.CATORIG_QUEUE, "CATNATIVE": d2.CATNATIVE_QUEUE,
                   "SEEDBAG": SEEDBAG_QUEUE_EXT, "FEAT": FEAT_QUEUE_EXT}


def _blend_k800(tree, parent_cfg):
    child = dc(parent_cfg)
    child["weight_search"] = "dirichlet_k800"
    child.pop("result", None)
    desc = (f"BLEND boundary-push: re-run dirichlet weight search at k=800 (vs the "
            f"k=1500 default) on the same {len(child['members'])} members -- does a "
            f"smaller search budget change the optimum? [PRIOR none]")
    return child, desc


def _blend_joint_add2(tree, parent_cfg):
    pool = d2.solo_pool(tree)
    unused = [nid for nid in pool if nid not in parent_cfg["members"]]
    if len(unused) < 2:
        return None
    child = dc(parent_cfg)
    child["members"] = parent_cfg["members"] + unused[:2]
    child.pop("result", None)
    desc = (f"BLEND joint-move: add 2 unused pool members at once (#{unused[0]}, "
            f"#{unused[1]}) -> {len(child['members'])}-way in one step [PRIOR none]")
    return child, desc


BLEND_QUEUE_EXT = d2.BLEND_QUEUE + [_blend_k800, _blend_joint_add2]
BLEND_QUEUES = {"BLEND": BLEND_QUEUE_EXT}


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES.get(lineage_id, f"L{lineage_id}")
    idx = hv2.lineage_size(tree, lineage_id) - 1 + _dedup_offset.get(lineage_id, 0)
    kind = parent_node["config"].get("kind", "solo")

    if kind == "blend":
        queue = BLEND_QUEUES.get(name, [])
        result = queue[idx](tree, parent_node["config"]) if idx < len(queue) else None
        if result is None:
            attempt = max(idx - len(queue), 0)
            result = d2.blend_fallback(tree, parent_node["config"], attempt)
        child_cfg, desc = result
    else:
        queue = SOLO_QUEUES_EXT.get(name, [])
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = d2.solo_fallback(parent_node["config"], idx - len(queue))

    phase = "exploit" if n_evaluated(tree) < EXPLOIT_CUTOFF else "explore"
    return child_cfg, f"[{name}] [PHASE {phase}] {desc} (parent=#{parent_id})"


# ---------------------------------------------------------------------------
# EXPLORE-phase long-shot lineages: injected once, directly under root, the first time
# n_evaluated() crosses EXPLOIT_CUTOFF. Each represents a hypothesis the exploit-phase
# authored queues never test: a contrarian direction (deeper/less-regularized CatBoost,
# despite it being a "closed item" per D-2's P3/P4), an even-more-aggressive joint
# feature prune, and two "kitchen-sink" blends of every solo node the search has found
# so far (one narrow-k weight search, one rank-space).
# ---------------------------------------------------------------------------
def explore_seeds(tree):
    pool = d2.solo_pool(tree)  # best-first solo node ids evaluated so far
    seeds = []

    p = dc(d2.LGB_TUNED_PARAMS)
    p.update(num_leaves=3, reg_lambda=10.0, reg_alpha=1.0, min_child_samples=100, random_state=99)
    seeds.append(("EXTREMEREG", {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}},
                  "EXPLORE long-shot: LGB pushed FAR beyond every boundary-push tried in the "
                  "exploit phase (num_leaves=3, reg_lambda=10, reg_alpha=1.0, "
                  "min_child_samples=100) -- is there a regularization sweet spot the "
                  "incremental boundary-pushes overshot past? [PRIOR none]"))

    px = dict(objective="binary:logistic", n_estimators=2000, eval_metric="auc",
              early_stopping_rounds=100, random_state=99, learning_rate=0.02, max_depth=1,
              min_child_weight=13, subsample=0.9, colsample_bytree=0.9,
              reg_alpha=0.002, reg_lambda=0.84)
    seeds.append(("XGBSHALLOWWIDE", {"kind": "solo", "model": "xgb", "params": px, "features": {"drop": []}},
                  "EXPLORE long-shot: XGB contrarian direction -- maximal-shallow "
                  "(max_depth=1) but LESS subsampled (0.9/0.9) and slower learning_rate "
                  "(0.02) than every XGBTUNED queue item tried so far [PRIOR none]"))

    pc = dict(loss_function="Logloss", eval_metric="AUC", iterations=2000,
              learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=99)
    seeds.append(("CATDEEP", {"kind": "solo", "model": "cat", "params": pc,
                              "features": {"drop": [], "native_cat": True}},
                  "EXPLORE long-shot / late-backtrack test: CatBoost native, depth 3->6 "
                  "(much deeper, much less L2) -- directly contradicts [PRIOR P3/P4]'s "
                  "'CatBoost is structurally weak here, do not retry' verdict; kept as an "
                  "explicit test of whether that closed-item call was premature at a "
                  "larger node budget"))

    drop_all = (["YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager",
                 "YearsSinceLastPromotion", "TotalWorkingYears"] + _RATE_COLS + _LE_COLS)
    seeds.append(("FEATKITCHEN", {"kind": "solo", "model": "lgb", "params": dc(d2.LGB_TUNED_PARAMS),
                                  "features": {"drop": drop_all}},
                  f"EXPLORE long-shot joint feature-prune: drop all {len(drop_all)} "
                  "redundant/noisy columns identified across the whole exploit phase "
                  "(5 raw tenure cols + 3 *Rate cols + 7 *_le cols) IN ONE MOVE, instead "
                  "of the incremental single/pairwise drops tried so far [PRIOR none]"))

    if len(pool) >= 3:
        seeds.append(("KITCHENBLEND", {"kind": "blend", "members": list(pool),
                                       "weight_search": "dirichlet_k800", "space": "prob"},
                      f"EXPLORE long-shot: kitchen-sink blend of ALL {len(pool)} solo "
                      f"nodes evaluated so far (ids {pool}), k=800 weight search -- does "
                      "including every weak/closed-item/boundary-push learner ever beat "
                      "the lean curated D-2-style blend? [PRIOR none]"))
        seeds.append(("RANKKITCHEN", {"kind": "blend", "members": list(pool),
                                      "weight_search": "dirichlet", "space": "rank"},
                      f"EXPLORE long-shot: same {len(pool)}-node kitchen-sink pool, "
                      "rank-space dirichlet search instead of prob-space -- re-checks "
                      "STATUS.md's rank-vs-prob 'possibly noise' finding at kitchen-sink "
                      "scale [PRIOR none]"))
    return seeds


_parent_dup_streak = {}


# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, child_cfg, is_root=False, lineage_id=None,
                  t_start=None):
    """Local (this-run-only) equivalent of run_s3e3.py's eval_and_add: writes to
    TREE_PATH (experiments_tree_scale.json), never d2.TREE_PATH. Tries digit-verified
    D-2 cache reuse for solo nodes before spending real compute; logs a curve entry for
    every evaluated node.

    Dead-end guard: dedup rejections don't consume MAX_CHILDREN_PER_NODE budget in
    harness_v2 (no node is ever added), so a mutation fn that deterministically
    regenerates the same config every call (e.g. blend_fallback's "pool fully included,
    queue exhausted" branch once a blend lineage has absorbed every solo node) makes
    select_next_parent re-pick that same dead parent forever. Two consecutive dedup
    rejections against the SAME parent burns a real (failed-status) child slot so the
    parent eventually hits MAX_CHILDREN_PER_NODE and the harness naturally moves on."""
    dup_id = None if is_root else d2.find_dup(tree, child_cfg)
    if dup_id is not None:
        DEDUP_REJECTIONS.append(dict(mutation=mutation, dup_id=dup_id))
        if lineage_id is not None:
            _dedup_offset[lineage_id] = _dedup_offset.get(lineage_id, 0) + 1
        if not is_root and parent_id is not None:
            streak = _parent_dup_streak.get(parent_id, 0) + 1
            _parent_dup_streak[parent_id] = streak
            if streak >= 2:
                hv2.add_node(
                    tree, parent_id,
                    f"[dead-end placeholder] mutation fn for parent #{parent_id} regenerated "
                    f"an identical existing config {streak} times in a row (dedup-rejected "
                    f"every time) -- burning this expansion slot so the search moves on "
                    f"instead of spinning forever",
                    {"kind": "placeholder", "dead_end_parent": parent_id, "streak": streak},
                    None, "failed", 0.0, allow_duplicate=True)
                hv2.save(tree, TREE_PATH)
                _parent_dup_streak[parent_id] = 0
        return None, dup_id, None
    if parent_id is not None:
        _parent_dup_streak[parent_id] = 0

    nid = hv2.next_id(tree)
    r = try_reuse_from_d2(nid, child_cfg)
    if r is None:
        r = run_eval(child_cfg, nid)
    stored_cfg = child_cfg
    if r["status"] == "evaluated" and r.get("result") is not None:
        stored_cfg = {**child_cfg, "result": r["result"]}

    if is_root:
        real_nid = hv2.add_root(tree, mutation, stored_cfg, r["score"], r["status"], r["wall_s"])
    else:
        real_nid, add_dup = hv2.add_node(tree, parent_id, mutation, stored_cfg, r["score"],
                                          r["status"], r["wall_s"], allow_duplicate=True)
        assert add_dup is None
    assert real_nid == nid

    if r["status"] == "evaluated":
        auc_val = r["result"]["auc"] if r.get("result") and "auc" in r["result"] else -r["score"]
        prior_best = max((e["auc"] for e in CURVE), default=-1.0)
        cum_best = max(prior_best, auc_val)
        is_new_best = (len(CURVE) == 0) or (auc_val > prior_best + 1e-9)
        phase = ("exploit" if "[PHASE exploit]" in mutation else
                 "explore" if "[PHASE explore]" in mutation else "seed")
        CURVE.append(dict(
            eval_index=len(CURVE) + 1, node_id=nid,
            lineage=LINEAGE_NAMES.get(lineage_id, "ROOT" if is_root else str(lineage_id)),
            phase=phase,
            auc=auc_val, cum_best_auc=round(cum_best, 6), is_new_global_best=is_new_best,
            wall_s=r["wall_s"], cum_wall_s=round(time.time() - t_start, 1) if t_start else None,
            reused_from_d2=bool(r.get("result", {}).get("reuse_digit_verified")) if r.get("result") else False,
        ))
    tree["curve"] = CURVE
    hv2.save(tree, TREE_PATH)
    return real_nid, None, r


def auc_of(r):
    if r is None:
        return None
    if r.get("result") and "auc" in r["result"]:
        return r["result"]["auc"]
    return -r["score"] if r.get("score") is not None else None


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = hv2.load(TREE_PATH)
        CURVE.extend(tree.get("curve", []))
        print(f"Resuming existing SCALE tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv2.new_tree(COMP)
        priors = hv2.suggest_priors(COMP_META)
        tree["priors"] = priors
        tree["design_note"] = ("Phase E-5 scaling experiment: fresh tree, same root+seeds+"
                                "mutation policy as D-2 (experiments_tree.json), 80-node "
                                "budget, exploit (evals 1-40) / explore (evals 41-80) "
                                "phase-tagged.")
        print(f"suggest_priors({COMP_META}) -> {len(priors)} bullets")

    # --- root ---
    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, d2.ROOT_MUTATION, d2.ROOT_CONFIG, is_root=True, t_start=t_start)
        print(f"[root] #{nid} AUC={auc_of(r)} wall_s={r['wall_s']} reused={r.get('result',{}).get('reuse_digit_verified')}")

    # --- 6 first-generation SOLO lineages (byte-identical to D-2) ---
    for name, seed_fn in d2.SOLO_SEEDS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] [PHASE seed] {desc}", cfg, t_start=t_start)
        print(f"[{name} seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"reused={r.get('result',{}).get('reuse_digit_verified')}")

    # --- BLEND seed ---
    already_blend = any(n["mutation"].startswith("[BLEND]") for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = d2.seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] [PHASE seed] {desc}", cfg, t_start=t_start)
        print(f"[BLEND seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")

    all_names = [n for n, _ in d2.SOLO_SEEDS] + ["BLEND"]
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in all_names:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name
            if n["id"] not in LINEAGE_NAMES:
                # explore-phase long-shot lineages seeded later (see below): name is
                # embedded as "[NAME]" prefix same convention, just not in all_names yet
                tag = n["mutation"].split("]", 1)[0].lstrip("[")
                LINEAGE_NAMES[n["id"]] = tag

    iterations = 0
    while n_evaluated(tree) < TARGET_NODES and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > 600:
            print("Safety cap on iterations reached, stopping.")
            break

        already_explore_seeded = any(n["mutation"].startswith(f"[{nm}]")
                                      for n in tree["nodes"] if n["parent_id"] == tree["root_id"]
                                      for nm in ("EXTREMEREG", "XGBSHALLOWWIDE", "CATDEEP",
                                                 "FEATKITCHEN", "KITCHENBLEND", "RANKKITCHEN"))
        if not _explore_injected["done"] and not already_explore_seeded and n_evaluated(tree) >= EXPLOIT_CUTOFF:
            print(f"\n--- exploit phase saturated at {n_evaluated(tree)} evaluated nodes; "
                  f"injecting EXPLORE long-shot lineages ---")
            for name, cfg, desc in explore_seeds(tree):
                if any(n["mutation"].startswith(f"[{name}]")
                       for n in tree["nodes"] if n["parent_id"] == tree["root_id"]):
                    continue  # already seeded (resume-safety)
                nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] [PHASE explore] {desc}",
                                            cfg, t_start=t_start)
                if nid is None:
                    print(f"EXPLORE seed [{name}] dedup-rejected (identical to existing #{dup}), skipping")
                    continue
                LINEAGE_NAMES[nid] = name
                print(f"[{name} EXPLORE seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")
                if n_evaluated(tree) >= TARGET_NODES:
                    break
            _explore_injected["done"] = True
            continue
        elif already_explore_seeded:
            _explore_injected["done"] = True

        parent_id, lineage_id = hv2.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted. Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        nid, dup, r = eval_and_add(tree, parent_id, mutation, child_cfg, lineage_id=lineage_id, t_start=t_start)
        if nid is None:
            print(f"DEDUP REJECTED <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"(identical to existing node #{dup})")
            continue
        gb = hv2.global_best(tree)
        gb_auc = -gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_AUC={gb_auc} (#{gb['id'] if gb else '-'}) n_eval={n_evaluated(tree)}")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    gb_auc = -gb["score"] if gb else None
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    prior_stats = d2.prior_usage_summary(tree)
    tree["prior_usage_summary"] = {
        "n_informed": len(prior_stats["informed"]), "n_uninformed": len(prior_stats["uninformed"]),
        "informed_win_rate": prior_stats["informed_win_rate"],
        "uninformed_win_rate": prior_stats["uninformed_win_rate"],
    }
    tree["dedup_rejections"] = DEDUP_REJECTIONS
    tree["curve"] = CURVE
    tree["budget"] = dict(target_nodes=TARGET_NODES, exploit_cutoff=EXPLOIT_CUTOFF, total_wall_s=round(total_wall, 1))
    hv2.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated(tree)} evaluated ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} AUC={gb_auc} mutation={gb['mutation'][:120]}")
    print(f"D-2 best: {D2_BEST} (tree {'BEAT' if gb_auc > D2_BEST else 'did not beat'} it)")
    print(f"Linear-iteration best: {LINEAR_BEST} (tree {'BEAT' if gb_auc > LINEAR_BEST else 'did not beat'} it)")
    print(f"Dedup rejections: {len(DEDUP_REJECTIONS)}")


if __name__ == "__main__":
    main()
