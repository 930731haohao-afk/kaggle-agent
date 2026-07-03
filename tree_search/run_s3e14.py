"""tree_search/run_s3e14.py — drives the ERA-inspired candidate-tree search for
playground-series-s3e14 (Phase C-2b: TWO node kinds, solo + blend/ensemble).

This is the fix for s3e9's key lesson (tree_search/run_s3e9.py's verdict): a
single-model-only node space cannot reach where linear iteration won on s3e14 (a 5-way
blend, OOF MAE 340.59891) because solo models plateau around ~342-343.6 no matter how
they're mutated. So this run seeds SIX solo lineages first (model-type/hyperparam/
feature variety, to populate a diverse pool of cheaply-cached OOF matrices), then seeds
a BLEND lineage that weight-searches over CACHED member OOF predictions (tree_search/
eval_s3e14.py's evaluate_blend, no retraining -> blend nodes cost ~0-1s instead of
20-60s) and mutates by adding/removing members or swapping the weight-search method.

Which lineage/node actually gets expanded next, and when a backtrack happens, is decided
dynamically by harness.select_next_parent() from the scores as they come in (same
selection/plateau/backtrack rule as run_s3e9.py, harness.py unchanged in its logic --
only a `next_id()` public wrapper was added so a solo node's OOF cache file can be named
before harness.add_node() assigns its id).
"""
import copy
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness  # noqa: E402
import eval_s3e14 as ev  # noqa: E402

COMP = "playground-series-s3e14"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 20          # aim >=18 evaluated nodes (mix of solo + blend)
MAX_WALL_S = 40 * 60       # hard stop, matches the ~40 min training budget
EVAL_TIMEOUT_S = 180
LINEAR_BEST = 340.59891    # linear-iteration's best (5-way blend, STATUS.md exp #7)

NOISE_FEATS = ev.NOISE_FEATS  # 6 already-proven-dead features (STATUS.md exp #2->#3) -- default
                              # drop set for every node, NOT re-tested as a mutation


def dc(cfg):
    return copy.deepcopy(cfg)


# ---------------------------------------------------------------------------
# Root: exact reproduction of the current best single LGB (scripts/train_v2..v6.py)
# ---------------------------------------------------------------------------
ROOT_CONFIG = {
    "kind": "solo",
    "model": "lgb",
    "params": dict(objective="regression_l1", metric="mae", n_estimators=3000,
                   learning_rate=0.02, num_leaves=63, min_child_samples=30,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                   reg_alpha=1.0, reg_lambda=2.0, random_state=42, early_stopping_rounds=150),
    "features": {"drop": list(NOISE_FEATS)},
    "postprocess": {"snap": False},
}
ROOT_MUTATION = ("root: s3e14 current-best single LGB (regression_l1, num_leaves=63, 21 "
                  "pruned features) from scripts/train_v2.py-train_v6.py -- solo OOF "
                  "~342.02154 per STATUS.md; the fair single-model reference point for "
                  "this tree-search run")


# ---------------------------------------------------------------------------
# Six first-generation SOLO lineage seeds -- variety of model type/hyperparam/features so
# the BLEND lineage (seeded right after) has a diverse pool of cached OOF matrices to draw
# on. Each mirrors a STATUS.md-proven config where possible (verifiable sanity check) or
# probes an explicitly UNTESTED direction (FEAT, REG) per experience.md's "don't re-test
# dead ends" rule.
# ---------------------------------------------------------------------------
def seed_cat():
    cfg = {
        "kind": "solo", "model": "cat",
        "params": dict(loss_function="MAE", eval_metric="MAE", iterations=4000,
                       learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=42,
                       native_cat=True),
        "features": {"drop": list(NOISE_FEATS)},
        "postprocess": {"snap": False},
    }
    desc = ("model-type: CatBoost solo w/ native categorical handling of the 7 low-"
            "cardinality env cols -- STATUS.md exp #2/#3 unchanged config, solo "
            "~343.60196; weakest of the 3 original solo learners but a useful blend-"
            "diversity source (ordered boosting differs structurally from LGB/XGB)")
    return cfg, desc


def seed_xgb():
    cfg = {
        "kind": "solo", "model": "xgb",
        "params": dict(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.02,
                       max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=1.0, reg_lambda=2.0, random_state=42, eval_metric="mae",
                       early_stopping_rounds=150),
        "features": {"drop": list(NOISE_FEATS)},
        "postprocess": {"snap": False},
    }
    desc = "model-type: XGB solo, STATUS.md exp #2/#3 unchanged config -- solo ~342.21778"
    return cfg, desc


def seed_lgbtuned():
    cfg = {
        "kind": "solo", "model": "lgb",
        "params": dict(objective="regression_l1", metric="mae", n_estimators=3000,
                       subsample_freq=1, random_state=42, learning_rate=0.015389816656608301,
                       num_leaves=63, max_depth=5, min_child_samples=30,
                       subsample=0.7899031923134097, colsample_bytree=0.6842166782718478,
                       reg_alpha=0.012648426946439915, reg_lambda=0.00346212601571565,
                       early_stopping_rounds=150),
        "features": {"drop": list(NOISE_FEATS)},
        "postprocess": {"snap": False},
    }
    desc = ("hyperparam: reuse the Optuna fold-0-proxy tuned LGB params from STATUS.md "
            "exp #4 (50-trial recipe already proven -- per experience.md do not re-run "
            "Optuna, add tuned solo to the pool) -- solo ~341.68775, currently the "
            "strongest single learner on this data")
    return cfg, desc


def seed_seedbag():
    p = dc(ROOT_CONFIG["params"])
    p["random_state"] = 2024
    cfg = {
        "kind": "solo", "model": "lgb", "params": p,
        "features": {"drop": list(NOISE_FEATS)},
        "postprocess": {"snap": False},
    }
    desc = ("seed variation: identical root LGB params, random_state 42->2024 -- STATUS.md "
            "exp #7's 5th blend member (solo ~342.04208); cheapest diversity source per "
            "experience.md's seed-bagging recipe, seeded upfront so BLEND can reference it "
            "without a mid-run detour to train a new solo model")
    return cfg, desc


def seed_feat():
    drop = list(NOISE_FEATS) + ["MaxOfUpperTRange", "MinOfUpperTRange", "MaxOfLowerTRange"]
    cfg = {
        "kind": "solo", "model": "lgb", "params": dc(ROOT_CONFIG["params"]),
        "features": {"drop": drop},
        "postprocess": {"snap": False},
    }
    desc = ("feature-subset SUBTRACTION beyond the already-pruned 21: drop the 3 remaining "
            "raw temp-range columns (MaxOfUpperTRange/MinOfUpperTRange/MaxOfLowerTRange) "
            "now that temp_spread already condenses them (computed before the drop, so "
            "still valid) -- 18-feature test, UNTESTED in the linear-iteration run (which "
            "stopped pruning at 21); doubles as a differently-featured blend-diversity "
            "member if it holds up")
    return cfg, desc


def seed_reg():
    p = dc(ROOT_CONFIG["params"])
    p.update(num_leaves=31, min_child_samples=50)
    cfg = {
        "kind": "solo", "model": "lgb", "params": p,
        "features": {"drop": list(NOISE_FEATS)},
        "postprocess": {"snap": False},
    }
    desc = ("hyperparam nudge: regularize further along the proven 'regularize over "
            "capacity' axis -- num_leaves 63->31, min_child_samples 30->50 (root's "
            "leaves=63 is high for ~15k rows; untested whether it already undershot the "
            "sweet spot)")
    return cfg, desc


SOLO_SEEDS = [("CAT", seed_cat), ("XGB", seed_xgb), ("LGBTUNED", seed_lgbtuned),
              ("SEEDBAG", seed_seedbag), ("FEAT", seed_feat), ("REG", seed_reg)]


# ---------------------------------------------------------------------------
# BLEND lineage: seeded once the 6 solo lineages exist. This is the Phase C-2b node kind
# -- weight-searches over CACHED member OOF matrices, no retraining.
# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    """SOLO_SEEDS name -> its first-gen node id (by mutation-prefix match)."""
    out = {}
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name, _ in SOLO_SEEDS:
                if name not in out and n["mutation"].startswith(f"[{name}]"):
                    out[name] = n["id"]
    return out


def _best_node_in_lineage(tree, lineage_id):
    best = None
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["id"] == tree["root_id"]:
            continue
        if harness.lineage_of(tree, n["id"]) == lineage_id:
            if best is None or n["score"] < best["score"]:
                best = n
    return best


def _id_for(tree, name):
    """Best-scoring evaluated node id in the named lineage ('ROOT' = the root node)."""
    if name == "ROOT":
        return tree["root_id"]
    lineage_id = _lineage_seed_ids(tree)[name]
    best = _best_node_in_lineage(tree, lineage_id)
    return best["id"] if best else lineage_id


def solo_pool(tree):
    """All evaluated kind=='solo' node ids in the tree, best (lowest score) first."""
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["XGB"], ids["CAT"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet",
           "postprocess": {"snap": True}}
    desc = (f"ensemble seed: 3-way blend of root LGB(#{tree['root_id']}) + XGB(#{ids['XGB']}) "
            f"+ CAT(#{ids['CAT']}) CACHED OOF, dirichlet weight search + snap -- mirrors "
            f"STATUS.md exp #3's starting point (340.75961 raw / 340.71180 snapped) as this "
            f"tree-search's entry point into ensemble space; no retraining, evaluates in <1s")
    return cfg, desc


def _mk_add_named(name):
    def fn(tree, parent_cfg):
        add_id = _id_for(tree, name)
        if add_id in parent_cfg["members"]:
            return None
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        desc = (f"BLEND: add {name} solo member (#{add_id}) -> {len(child['members'])}-way, "
                f"dirichlet weight search + snap")
        return child, desc
    return fn


def _blend_method_swap(tree, parent_cfg):
    child = dc(parent_cfg)
    cur = child.get("weight_search", "dirichlet")
    new_method = "grid_simplex" if cur == "dirichlet" else "dirichlet"
    if new_method == "grid_simplex" and len(child["members"]) > 5:
        return None  # combinatorial cap -- skip to fallback
    child["weight_search"] = new_method
    child.pop("result", None)
    desc = (f"BLEND: weight-search method comparison {cur}->{new_method} on the same "
            f"{len(child['members'])} members (same pool, different search strategy)")
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
            f"or just being tolerated by the weight search")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("LGBTUNED"),   # -> 4-way, mirrors STATUS.md exp #5 (340.62702)
    _mk_add_named("SEEDBAG"),    # -> 5-way, mirrors STATUS.md exp #7 (340.59891) -- the KEY node
    _blend_method_swap,          # dirichlet <-> grid_simplex on the same 5-way pool
    _mk_add_named("FEAT"),       # -> 6-way, UNTESTED in linear iteration (different feature set)
    _blend_remove_weakest,       # leaner blend probe
]


def blend_fallback(tree, parent_cfg, attempt):
    pool = solo_pool(tree)
    unused = [nid for nid in pool if nid not in parent_cfg["members"]]
    if unused:
        add_id = unused[0]
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        desc = f"BLEND fallback: add next-best unused pool member #{add_id} -> {len(child['members'])}-way"
        return child, desc
    child = dc(parent_cfg)
    child.pop("result", None)
    desc = f"BLEND fallback: queue exhausted & pool fully included, re-run weight search (attempt {attempt})"
    return child, desc


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    seed_key = "random_state" if c["model"] in ("lgb", "xgb") else "random_seed"
    c["params"][seed_key] = 3000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): same "
            f"config, alt {seed_key}={3000 + attempt}")
    return c, desc


# ---------------------------------------------------------------------------
# Per-lineage authored mutation queues (SOLO): fn(parent_config) -> (child_config, desc)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"].update(params_update)
    return c


CAT_QUEUE = [
    lambda c: (_bump(c, bagging_temperature=1.0),
               "CAT: add bagging_temperature=1.0 (Bayesian-bootstrap row weighting) -- "
               "untested regularization axis for CatBoost on this data"),
    lambda c: (_bump(c, depth=6),
               "CAT: depth 7->6, probe the over/underfit capacity boundary"),
]

XGB_QUEUE = [
    lambda c: (_bump(c, max_depth=4, min_child_weight=8),
               "XGB: max_depth 6->4, min_child_weight 5->8 -- lean harder into "
               "regularization"),
    lambda c: (_bump(c, colsample_bytree=0.5, subsample=0.6),
               "XGB: heavier row/feature subsampling for variance reduction"),
]

LGBTUNED_QUEUE = [
    lambda c: (_bump(c, random_state=7),
               "LGBTUNED: seed variation on the Optuna-tuned params (random_state 42->7) "
               "-- second, independent seed-bag source distinct from root's seedbag"),
    lambda c: (_bump(c, learning_rate=0.01, n_estimators=4000),
               "LGBTUNED: lr 0.0154->0.01 with more estimators, finer optimization pace"),
]

SEEDBAG_QUEUE = [
    lambda c: (_bump(c, random_state=777),
               "SEEDBAG: third seed variant (2024->777) -- STATUS.md 'next ideas' flagged "
               "3-5 seed bags as untried"),
]

def _feat(c, drop_list):
    child = dc(c)
    child["features"] = {"drop": sorted(set(drop_list))}
    return child


FEAT_QUEUE = [
    lambda c: (_feat(c, c["features"]["drop"] + ["seeds_per_fruitset"]),
               "FEAT: further trim seeds_per_fruitset (ratio feature, likely redundant "
               "with fruitset_x_seeds product term) -- 17 feats"),
    lambda c: (_feat(c, [f for f in c["features"]["drop"] if f != "MaxOfUpperTRange"]),
               "FEAT: restore MaxOfUpperTRange only, isolate its marginal effect vs the "
               "aggressive 18-feature trim"),
]

REG_QUEUE = [
    lambda c: (_bump(c, reg_lambda=4.0),
               "REG: reg_lambda 2.0->4.0, push L2 further along the known-good direction"),
    lambda c: (_bump(c, num_leaves=90, subsample=0.7),
               "REG: opposite-direction probe -- num_leaves 63->90 (more capacity) + "
               "subsample 0.8->0.7 (more row bagging), bracket the sweet spot from both "
               "sides"),
]

SOLO_QUEUES = {"CAT": CAT_QUEUE, "XGB": XGB_QUEUE, "LGBTUNED": LGBTUNED_QUEUE,
               "SEEDBAG": SEEDBAG_QUEUE, "FEAT": FEAT_QUEUE, "REG": REG_QUEUE}
LINEAGE_NAMES = {}  # lineage_id (node id of the first-gen node) -> short name


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES[lineage_id]
    idx = harness.lineage_size(tree, lineage_id) - 1
    if name == "BLEND":
        result = None
        if idx < len(BLEND_QUEUE):
            result = BLEND_QUEUE[idx](tree, parent_node["config"])
        if result is None:
            attempt = max(idx - len(BLEND_QUEUE), 0)
            result = blend_fallback(tree, parent_node["config"], attempt)
        child_cfg, desc = result
    else:
        queue = SOLO_QUEUES[name]
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, child_cfg, is_root=False):
    nid = harness.next_id(tree)
    r = ev.evaluate(child_cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
    stored_cfg = child_cfg
    if r["status"] == "evaluated" and r.get("result") is not None:
        stored_cfg = {**child_cfg, "result": r["result"]}
    if is_root:
        real_nid = harness.add_root(tree, mutation, stored_cfg, r["score"], r["status"], r["wall_s"])
    else:
        real_nid = harness.add_node(tree, parent_id, mutation, stored_cfg, r["score"], r["status"], r["wall_s"])
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    harness.save(tree, TREE_PATH)
    return real_nid, r


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = harness.load(TREE_PATH)
        print(f"Resuming existing tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = harness.new_tree(COMP)

    def n_evaluated():
        return sum(1 for n in tree["nodes"] if n["status"] == "evaluated")

    # --- root ---
    if not tree["nodes"]:
        nid, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} score={r['score']} wall_s={r['wall_s']}")

    # --- seed the 6 first-generation SOLO lineages ---
    for name, seed_fn in SOLO_SEEDS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} score={r['score']} status={r['status']} wall_s={r['wall_s']}")

    # --- seed the BLEND lineage (needs XGB + CAT solo ids) ---
    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} score={r['score']} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = [n for n, _ in SOLO_SEEDS] + ["BLEND"]
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
        parent_id, lineage_id = harness.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted (no more eligible expansions). Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        nid, r = eval_and_add(tree, parent_id, mutation, child_cfg)
        gb = harness.global_best(tree)
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"score={r['score']} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best={gb['score'] if gb else None} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = harness.global_best(tree)
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} score={gb['score']} mutation={gb['mutation']}")
    print(f"Linear-iteration best: {LINEAR_BEST}  (tree {'BEAT' if gb['score'] < LINEAR_BEST else 'did not beat'} it)")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for e in tree["search_state"]["backtrack_log"]:
        print(" ", e)


if __name__ == "__main__":
    main()
