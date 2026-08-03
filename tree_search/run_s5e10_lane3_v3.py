"""tree_search/run_s5e10_lane3_v3.py — harness_v3 tree-search driver for
playground-series-s5e10 (RMSE, minimize), Stage 4.

Follows references/07_tree_search.md §4's five driver requirements:
  1. digit-for-digit root verification against the linear-iteration best LightGBM solo
  2. OOF cache reuse — not applicable here (this run's cache dir is fresh; the linear
     round-2 caches live in the competition dir under a different naming scheme and are
     re-evaluated as real nodes so nothing is imported unverified)
  3. resume-state contract — all driver bookkeeping lives in search_state["driver_state"]
  4. subprocess-level evaluation timeout for every solo node
  5. burst-seed sanity gate applied right after each explore-burst seed

Node space (general tabular / GBDT-dominant => the v2/v3 default solo+blend dual track):
  solo   model in {lgb, xgb, cat} x params x feature variant in {v1, v2, v3, v23}
  blend  weighted mix of already-evaluated solo nodes, scored leave-fold-out

Mutation directions, in the order the proposer tries them:
  capacity   num_leaves / max_depth / depth up and down
  shrinkage  learning_rate down (with the iteration cap already effectively infinite via
             early stopping)
  regularise min_child_samples / lambda_l2 / l2_leaf_reg
  sampling   feature_fraction / bagging_fraction / subsample / colsample
  features   switch feature variant (the v2 engineered set races the library's prior that
             explicit products hurt GBDTs)
  seed       seed bagging (experience.md: the last reliable lever near a noise ceiling)
  boundary   harness_v3.boundary_candidates on any param pinned at its declared edge
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
COMP = "playground-series-s5e10"
EVAL_MODULE = os.path.join(_HERE, "eval_s5e10_lane3.py")
TREE_PATH = os.path.join(_REPO, "competitions", COMP, "experiments_tree_lane3.json")
CACHE_DIR = os.path.join(_HERE, "cache_s5e10_lane3")

TOTAL_BUDGET = int(os.environ.get("S5E10_BUDGET", "60"))
SOLO_TIMEOUT_S = 900
BLEND_TIMEOUT_S = 600

# The linear-iteration reference numbers this search must beat (competitions/<comp>/
# experiments.json, round 2): best solo = xgb_a 0.056019, best blend (honest
# leave-fold-out) = 0.055985.
LINEAR_BEST_SOLO = 0.056019
LINEAR_BEST_BLEND = 0.055985
ROOT_EXPECTED = 0.056029  # lgb_a from 03_models.py, must reproduce to 6 dp

ROOT_CFG = {
    "kind": "solo", "model": "lgb",
    "params": {"learning_rate": 0.03, "num_leaves": 96, "min_child_samples": 40,
               "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l2": 1.0},
    "features": {"variant": "v1"},
}

SEARCH_SPACE = {
    "lgb": {"learning_rate": (0.008, 0.10), "num_leaves": (16, 1024),
            "min_child_samples": (5, 400), "feature_fraction": (0.5, 1.0),
            "bagging_fraction": (0.5, 1.0), "lambda_l2": (0.0, 40.0),
            "lambda_l1": (0.0, 10.0)},
    "xgb": {"learning_rate": (0.008, 0.10), "max_depth": (4, 14),
            "min_child_weight": (1, 300), "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0), "reg_lambda": (0.0, 40.0)},
    # CatBoost's depth floor is 6, not 4: a depth-4 CatBoost node took 568s (vs ~20s for a
    # LightGBM node) because the shallower the tree, the more boosting rounds early
    # stopping lets it run. Shallow CatBoost also scored worst of any node so far
    # (0.056188), so the cheap and the good directions agree here.
    "cat": {"learning_rate": (0.02, 0.15), "depth": (6, 10),
            "l2_leaf_reg": (0.5, 40.0)},
}
# Per-model wall-clock ceilings for the subprocess kill (harness_v3 feature 8). CatBoost
# is ~20x a LightGBM node on this dataset, so it gets a larger budget but still a bounded
# one; anything past it is a runaway and is worth more as a freed slot than as a result.
MODEL_TIMEOUT_S = {"lgb": 300, "xgb": 420, "cat": 480}
INT_PARAMS = {"num_leaves", "min_child_samples", "max_depth", "min_child_weight", "depth"}
VARIANTS = ["v1", "v2", "v3", "v23"]


def core(cfg: dict) -> dict:
    """Canonical hashable stored form (07_tree_search.md §2): sort blend members, drop
    output-only fields."""
    c = json.loads(json.dumps(cfg, sort_keys=True))
    if c.get("kind") == "blend":
        c["members"] = sorted(c["members"])
    return c


def dstate(tree: dict) -> dict:
    return tree["search_state"].setdefault("driver_state", {
        "node_results": {}, "burst_injected": False, "lineage_names": {},
        "evals_to_match_linear_best": None, "rng_calls": 0,
    })


def _rng(tree: dict) -> random.Random:
    """Deterministic RNG whose stream position is persisted, so a resume does not replay
    the same proposals (module-level RNG state is exactly what H-1 feature 7 forbids)."""
    ds = dstate(tree)
    r = random.Random(20260731)
    for _ in range(ds["rng_calls"]):
        r.random()
    return r


def _bump_rng(tree: dict, n: int) -> None:
    dstate(tree)["rng_calls"] += n


def eval_and_add(tree: dict, parent_id, mutation: str, cfg: dict, *, is_root=False):
    cfg = core(cfg)
    if not is_root and hv2.find_duplicate_config(tree, cfg) is not None:
        nid, dup = hv3.add_node(tree, parent_id, mutation, cfg, None, "failed", 0.0)
        print(f"    dedup-rejected (dup of #{dup})")
        return None, None
    nid_hint = max([n["id"] for n in tree["nodes"]], default=-1) + 1
    t0 = time.time()
    if cfg.get("kind") == "blend":
        # blend evaluation is pure numpy over cached OOFs -- no native fit() to hang, so
        # it runs in-process (the subprocess boundary exists for native trainers)
        sys.path.insert(0, _HERE)
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ev_s5e10", EVAL_MODULE)
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        r = ev.evaluate(cfg, node_id=nid_hint, timeout_s=BLEND_TIMEOUT_S)
    else:
        tmo = MODEL_TIMEOUT_S.get(cfg.get("model", "lgb"), SOLO_TIMEOUT_S)
        r = hv3.eval_solo_subprocess(EVAL_MODULE, cfg, tmo, node_id=nid_hint)
    wall = time.time() - t0
    score, status = r.get("score"), r.get("status", "failed")
    if is_root:
        nid = hv2.add_root(tree, mutation, cfg, score, status, round(wall, 1))
        hv3.update_phase(tree)
    else:
        nid, dup = hv3.add_node(tree, parent_id, mutation, cfg, score, status, round(wall, 1))
        if nid is None:
            print(f"    dedup-rejected on add (dup of #{dup})")
            return None, None
    ds = dstate(tree)
    ds["node_results"][str(nid)] = r.get("result")
    gb = hv3.global_best(tree)
    flag = ""
    if gb is not None and gb["id"] == nid:
        flag = "  <-- new global best"
    if (ds["evals_to_match_linear_best"] is None and score is not None
            and score <= LINEAR_BEST_BLEND):
        ds["evals_to_match_linear_best"] = hv3.n_evaluated(tree)
    print(f"  #{nid:<3} {mutation[:58]:<58} score {score}  ({wall:.0f}s){flag}")
    if status == "failed":
        print(f"       FAILED: {r.get('error')}")
    return nid, score


# ---------------------------------------------------------------------------
# mutation proposer
# ---------------------------------------------------------------------------
def _jitter(rng, val, lo, hi, name):
    factor = rng.choice([0.5, 0.65, 0.8, 1.25, 1.5, 2.0])
    new = val * factor
    if name in ("feature_fraction", "bagging_fraction", "subsample", "colsample_bytree"):
        new = val + rng.choice([-0.15, -0.08, 0.08, 0.15])
    if name in ("lambda_l2", "lambda_l1", "reg_lambda", "l2_leaf_reg") and val == 0:
        new = rng.choice([0.5, 2.0, 8.0])
    new = min(max(new, lo), hi)
    if name in INT_PARAMS:
        new = int(round(new))
    else:
        new = round(float(new), 5)
    return new


def propose_child(tree: dict, parent: dict, rng: random.Random) -> tuple[dict, str]:
    cfg = json.loads(json.dumps(parent["config"]))
    if cfg.get("kind") == "blend":
        # Blend lineage: move around the subset lattice of the current solo pool.
        # The first version only ever added the best outside member or dropped the
        # weakest one, which made add and drop exact inverses — the lineage bounced
        # between two configs and burned 10 dedup rejections in a row. Randomising WHICH
        # member moves (and occasionally resampling the whole subset) makes the walk
        # actually explore. Blend nodes cost ~1s, so breadth here is nearly free.
        solos = [n["id"] for n in tree["nodes"]
                 if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
        cur = set(cfg["members"]) & set(solos)
        outside = [s for s in solos if s not in cur]
        roll = rng.random()
        if roll < 0.20 and len(solos) >= 3:
            size = rng.randint(2, len(solos))
            cfg["members"] = sorted(rng.sample(solos, size))
            return cfg, f"blend: resample random subset of {size}/{len(solos)} solos"
        if outside and (roll < 0.65 or len(cur) <= 2):
            add = rng.choice(outside)
            cfg["members"] = sorted(cur | {add})
            return cfg, f"blend: add member #{add} ({len(cfg['members'])} members)"
        if len(cur) > 2:
            drop = rng.choice(sorted(cur))
            cfg["members"] = sorted(cur - {drop})
            return cfg, f"blend: drop member #{drop} ({len(cfg['members'])} members)"
        if len(solos) >= 2:
            cfg["members"] = sorted(rng.sample(solos, min(len(solos), rng.randint(2, 4))))
            return cfg, f"blend: fallback resample ({len(cfg['members'])} members)"
        return cfg, "blend: no move available"

    model = cfg["model"]
    space = SEARCH_SPACE[model]
    params = cfg.setdefault("params", {})

    # boundary-push first (07_tree_search.md: often the single largest lever)
    cands = hv3.boundary_candidates(cfg, space, params_key="params", edge_frac=0.05)
    if cands and rng.random() < 0.45:
        c = rng.choice(cands)
        name, val, edge = c["param"], c["new_value"], c["edge"]
        if name in INT_PARAMS:
            val = max(1, int(round(val)))
        else:
            val = max(0.0, round(float(val), 5))
        params[name] = val
        return cfg, f"boundary-push ({edge} edge) {name} {c['old_value']} -> {val}"

    groups_by_move = {
        "capacity": [p for p in ("num_leaves", "max_depth", "depth") if p in space],
        "shrinkage": [p for p in ("learning_rate",) if p in space],
        "regularise": [p for p in ("min_child_samples", "min_child_weight", "lambda_l2",
                                   "lambda_l1", "reg_lambda", "l2_leaf_reg") if p in space],
        "sampling": [p for p in ("feature_fraction", "bagging_fraction", "subsample",
                                 "colsample_bytree") if p in space],
    }
    # CatBoost declares no sampling knobs here, so its "sampling" bucket is empty; picking
    # it used to crash the whole driver on rng.choice([]). Only offer moves that exist.
    choices = ["features", "seed"] + [m for m, g in groups_by_move.items() if g]
    weights = {"capacity": 3, "shrinkage": 2, "regularise": 3, "sampling": 2,
               "features": 2, "seed": 2}
    move = rng.choices(choices, weights=[weights[m] for m in choices])[0]
    if move == "features":
        cur = cfg.get("features", {}).get("variant", "v1")
        new = rng.choice([v for v in VARIANTS if v != cur])
        cfg["features"] = {"variant": new}
        return cfg, f"features: variant {cur} -> {new}"
    if move == "seed":
        off = int(params.get("seed_offset", 0))
        params["seed_offset"] = off + rng.choice([1, 2, 3, 7])
        return cfg, f"seed bagging: seed_offset -> {params['seed_offset']}"
    name = rng.choice(groups_by_move[move])
    lo, hi = space[name]
    cur = params.get(name, _default_of(model, name))
    params[name] = _jitter(rng, cur, lo, hi, name)
    return cfg, f"{move}: {name} {cur} -> {params[name]}"


_DEFAULTS = {
    "lgb": {"learning_rate": 0.03, "num_leaves": 96, "min_child_samples": 40,
            "feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l2": 1.0,
            "lambda_l1": 0.0},
    "xgb": {"learning_rate": 0.03, "max_depth": 8, "min_child_weight": 20,
            "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
    "cat": {"learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0},
}


def _default_of(model, name):
    return _DEFAULTS[model][name]


def _score_of(tree, nid):
    n = next(x for x in tree["nodes"] if x["id"] == nid)
    return n["score"] if n["score"] is not None else 9e9


def evaluated_solos(tree):
    return [n["id"] for n in tree["nodes"]
            if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        print(f"resumed tree with {hv3.n_evaluated(tree)} evaluated nodes")
    else:
        tree = hv2.new_tree(COMP)
    hv3.init_budget(tree, total_budget=TOTAL_BUDGET)
    dstate(tree)

    # ---- root, digit-for-digit verified -------------------------------------
    if tree["root_id"] is None:
        print("root: lgb_a (linear-iteration LightGBM champion), verifying digit-for-digit")
        _nid, score = eval_and_add(tree, None, "root = linear-iteration lgb_a", ROOT_CFG,
                                   is_root=True)
        assert score is not None and round(score, 6) == ROOT_EXPECTED, (
            f"ROOT VERIFICATION FAILED: got {score}, expected {ROOT_EXPECTED} -- the "
            f"tree-search evaluator does not reproduce the linear-iteration pipeline, "
            f"so every node in this tree would be incomparable. Aborting.")
        print(f"  root verified: {score} == {ROOT_EXPECTED}")
        hv3.save_search_state(tree, TREE_PATH)

    # ---- first-generation lineages ------------------------------------------
    seeds = [
        ({"kind": "solo", "model": "xgb", "params": {}, "features": {"variant": "v1"}},
         "seed lineage: XGBoost (linear-iteration best solo)"),
        ({"kind": "solo", "model": "cat", "params": {}, "features": {"variant": "v1"}},
         "seed lineage: CatBoost (different categorical handling => diversity)"),
        ({"kind": "solo", "model": "lgb",
          "params": {"num_leaves": 255, "min_child_samples": 120, "learning_rate": 0.02,
                     "lambda_l2": 5.0}, "features": {"variant": "v1"}},
         "seed lineage: high-capacity LightGBM"),
        ({"kind": "solo", "model": "lgb", "params": {},
          "features": {"variant": "v2"}},
         "seed lineage: LightGBM on engineered feature set v2"),
    ]
    root_id = tree["root_id"]
    have = {hv2.config_hash(n["config"]) for n in tree["nodes"]}
    for cfg, label in seeds:
        if hv2.config_hash(core(cfg)) in have:
            continue
        eval_and_add(tree, root_id, label, cfg)
        hv3.save_search_state(tree, TREE_PATH)

    # first blend seed (needs >= 2 solos)
    solos = evaluated_solos(tree)
    if len(solos) >= 2 and not any(n["config"].get("kind") == "blend" for n in tree["nodes"]):
        eval_and_add(tree, root_id, "seed lineage: blend of all solos so far",
                     {"kind": "blend", "members": sorted(solos)})
        hv3.save_search_state(tree, TREE_PATH)

    # ---- main loop -----------------------------------------------------------
    while not hv3.should_stop(tree):
        phase = tree["search_state"]["budget"]["phase"]
        ds = dstate(tree)
        if phase == "explore_burst" and not ds["burst_injected"]:
            print("\n--- MANDATORY EXPLORE BURST ---")
            inject_burst(tree)
            ds["burst_injected"] = True
            hv3.save_search_state(tree, TREE_PATH)
            continue
        parent_id, _lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            print("no expandable parent left")
            break
        parent = next(n for n in tree["nodes"] if n["id"] == parent_id)
        rng = _rng(tree)
        cfg, mutation = propose_child(tree, parent, rng)
        _bump_rng(tree, 24)
        eval_and_add(tree, parent_id, mutation, cfg)
        hv3.save_search_state(tree, TREE_PATH)

    report(tree)


def inject_burst(tree: dict) -> None:
    """Explore burst: the MANDATORY kitchen-sink blend of the entire current solo pool
    (07_tree_search.md §3 -- every post-exploit gain in the historical runs came from
    this, never from a long-shot solo alone), plus long-shot solos in contrasting
    directions. Each long-shot seed goes through the burst-seed sanity gate."""
    root_id = tree["root_id"]
    solos = evaluated_solos(tree)
    if len(solos) >= 2:
        eval_and_add(tree, root_id, "BURST: kitchen-sink blend of the entire solo pool",
                     {"kind": "blend", "members": sorted(solos)})
        hv3.save_search_state(tree, TREE_PATH)
    longshots = [
        ({"kind": "solo", "model": "lgb",
          "params": {"learning_rate": 0.008, "num_leaves": 512, "min_child_samples": 200,
                     "lambda_l2": 20.0, "feature_fraction": 0.6, "bagging_fraction": 0.7},
          "features": {"variant": "v23"}},
         "BURST long-shot: very slow + very large + heavily regularised, all features"),
        ({"kind": "solo", "model": "lgb",
          "params": {"learning_rate": 0.05, "num_leaves": 24, "min_child_samples": 10,
                     "lambda_l2": 0.0, "feature_fraction": 1.0, "bagging_fraction": 1.0},
          "features": {"variant": "v1"}},
         "BURST long-shot: opposite corner -- small, fast, unregularised"),
        ({"kind": "solo", "model": "xgb",
          "params": {"max_depth": 13, "learning_rate": 0.02, "min_child_weight": 100,
                     "reg_lambda": 20.0, "subsample": 0.7, "colsample_bytree": 0.6},
          "features": {"variant": "v2"}},
         "BURST long-shot: very deep XGBoost on the engineered set"),
        ({"kind": "solo", "model": "cat",
          "params": {"depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 12.0},
          "features": {"variant": "v3"}},
         "BURST long-shot: deep CatBoost + tuple-frequency feature"),
        ({"kind": "solo", "model": "lgb",
          "params": {"seed_offset": 101, "num_leaves": 160, "min_child_samples": 60,
                     "learning_rate": 0.025},
          "features": {"variant": "v2"}},
         "BURST long-shot: mid-capacity LightGBM, fresh seed, engineered set"),
    ]
    for cfg, label in longshots:
        nid, _ = eval_and_add(tree, root_id, label, cfg)
        if nid is not None:
            passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
            if not passed:
                print(f"       burst-seed sanity gate FAILED (bound {bound:.6f}) -- "
                      f"lineage #{nid} marked plateaued")
        hv3.save_search_state(tree, TREE_PATH)


def report(tree: dict) -> None:
    st = tree["search_state"]
    ds = dstate(tree)
    gb = hv3.global_best(tree)
    print("\n" + "=" * 74)
    print("TREE SEARCH REPORT — playground-series-s5e10 (RMSE, minimize)")
    print("=" * 74)
    print(f"evaluated nodes      : {hv3.n_evaluated(tree)} / {st['budget']['total_budget']}")
    print(f"stop reason          : {st['budget'].get('stop_reason')}")
    print(f"phase                : {st['budget']['phase']}")
    print(f"global best          : node #{gb['id']} score {gb['score']}")
    print(f"  config             : {json.dumps(gb['config'])[:300]}")
    print(f"linear-iteration best: solo {LINEAR_BEST_SOLO} / blend {LINEAR_BEST_BLEND}")
    print(f"evals to match linear: {ds['evals_to_match_linear_best']}")
    print(f"dedup rejections     : {sum(1 for n in tree['nodes'] if n['config'].get('kind') == 'placeholder')}")
    print(f"plateaued lineages   : {st.get('plateaued')}")
    print("backtrack log:")
    for e in st.get("backtrack_log", []):
        print(f"  - {e}")
    cg = st.get("cost_guard_log", [])
    print(f"cost-guard triggers  : {len(cg)}" + ("" if cg else "  (none fired this run)"))
    for e in cg:
        print(f"  - {e['warning']}")
    print("\ntop 12 nodes:")
    ranked = sorted([n for n in tree["nodes"] if n["status"] == "evaluated"],
                    key=lambda n: n["score"])[:12]
    for n in ranked:
        print(f"  #{n['id']:<3} {n['score']:.6f}  {n['mutation'][:64]}")
    hv3.save_search_state(tree, TREE_PATH)


if __name__ == "__main__":
    main()
