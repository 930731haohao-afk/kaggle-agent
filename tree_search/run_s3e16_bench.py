"""Stage-4 tree-search driver for playground-series-s3e16 (crab Age, MAE).

Follows the 07_tree_search.md §4 driver checklist:
  1. digit-for-digit root verification against the linear-iteration best solo
  2. (OOF cache reuse: NOT used -- this is a fresh run with no prior compatible cache;
     stated explicitly rather than silently skipped)
  3. resume-state contract -- ALL driver bookkeeping lives in
     tree["search_state"]["driver_state"], persisted via save_search_state/load_search_state
  4. subprocess-level eval timeout for every solo node (eval_solo_subprocess)
  5. burst-seed sanity gate applied to every explore-burst seed

Node space: the v2/v3 default solo+blend dual track.
Score: post-processed (clip to [1,29] + round) MAE -- see eval_s3e16_bench.score_pp.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness as hv1  # noqa: E402
import harness_v3 as hv3  # noqa: E402

COMP = "playground-series-s3e16"
COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
TREE_PATH = os.path.join(COMP_DIR, "experiments_tree.json")
EVAL_MODULE = os.path.join(_HERE, "eval_s3e16_bench.py")
CACHE_DIR = os.path.join(_HERE, "cache_s3e16_bench")
SOLO_TIMEOUT_S = 420.0

# Linear-iteration (rounds 1+2) best score, on the same post-processed metric.
LINEAR_BEST = 1.339847          # exp14 CatBoost-MAE depth 6 lr 0.06, groups=['cnt']
ROOT_EXPECTED = 1.339847        # the root IS that config -> digit-for-digit assert

SEARCH_SPACE = {
    "lgb": {"learning_rate": {"low": 0.01, "high": 0.15, "log": True},
            "num_leaves": (15, 255), "min_data_in_leaf": (5, 200),
            "feature_fraction": (0.4, 1.0), "bagging_fraction": (0.5, 1.0),
            "lambda_l1": (0.0, 10.0), "lambda_l2": (0.0, 10.0)},
    "cat": {"learning_rate": {"low": 0.02, "high": 0.15, "log": True},
            "depth": (4, 10), "l2_leaf_reg": (1.0, 20.0)},
    "xgb": {"learning_rate": {"low": 0.02, "high": 0.15, "log": True},
            "max_depth": (4, 12), "min_child_weight": (1, 50),
            "subsample": (0.5, 1.0), "colsample_bytree": (0.4, 1.0),
            "reg_lambda": (0.0, 10.0)},
    "lgbmc": {"learning_rate": {"low": 0.01, "high": 0.15, "log": True},
              "num_leaves": (15, 255), "min_data_in_leaf": (5, 200)},
}
FEATURE_GROUPS = ["resid", "frac", "shape", "meat", "cnt"]


# ---------------------------------------------------------------------------
# driver state (checklist item 3: never module-level globals)
# ---------------------------------------------------------------------------
def dstate(tree: dict) -> dict:
    return tree["search_state"].setdefault("driver_state", {
        "lineage_names": {}, "burst_injected": False, "results": {},
        "mut_counter": {}, "evals_to_match_linear_best": None, "log": [],
    })


def note(tree, msg):
    print(msg, flush=True)
    dstate(tree)["log"].append(msg)


def core(cfg: dict) -> dict:
    out = {k: v for k, v in cfg.items() if k not in ("result", "want_importance", "note")}
    if out.get("kind") == "blend":
        out["members"] = sorted(out["members"])
    return out


def rng_for(tree, salt: int) -> np.random.Generator:
    """Deterministic per-proposal RNG: seeded by node count + salt, so a resumed run
    reproduces the same proposal stream."""
    return np.random.default_rng(1000 * len(tree["nodes"]) + salt)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, cfg, *, is_root=False):
    cfg = core(cfg)
    if not is_root:
        dup = hv3.v2.find_duplicate_config(tree, cfg)
        if dup is not None:
            return None, dup, None
    nid = hv1.next_id(tree)
    kind = cfg.get("kind", "solo")

    t0 = time.time()
    if kind == "blend":
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ev_s3e16", EVAL_MODULE)
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        r = ev.evaluate(cfg, node_id=nid, timeout_s=SOLO_TIMEOUT_S)
    else:
        r = hv3.eval_solo_subprocess(EVAL_MODULE, cfg, SOLO_TIMEOUT_S, node_id=nid)
    wall = r.get("wall_s", round(time.time() - t0, 2))

    if is_root:
        got = hv3.add_root(tree, mutation, cfg, r["score"], r["status"], wall, kind=kind)
    else:
        got, dup = hv3.add_node(tree, parent_id, mutation, cfg, r["score"], r["status"],
                                wall, kind=kind)
        if got is None:
            dstate(tree)["dedup_rejections"] = dstate(tree).get("dedup_rejections", []) + [
                {"mutation": mutation, "dup_id": dup}]
            return None, dup, r
    ds = dstate(tree)
    ds["results"][str(got)] = {"result": r.get("result"), "error": r.get("error"),
                              "wall_s": wall}
    if r["status"] == "evaluated" and ds["evals_to_match_linear_best"] is None \
            and r["score"] is not None and r["score"] <= LINEAR_BEST:
        ds["evals_to_match_linear_best"] = hv3.n_evaluated(tree)
    note(tree, f"  node #{got:<3d} [{kind:<5s}] {mutation:<34s} score={r['score']} "
               f"({wall}s){'  ERR: ' + str(r['error']) if r.get('error') else ''}")
    return got, None, r


# ---------------------------------------------------------------------------
# mutation proposers
# ---------------------------------------------------------------------------
def mutate_solo(tree, cfg, salt):
    """Returns (new_cfg, mutation_label) or (None, None)."""
    rng = rng_for(tree, salt)
    cfg = json.loads(json.dumps(core(cfg)))
    model = cfg["model"]
    space = SEARCH_SPACE[model]
    kinds = ["boundary", "jitter", "features", "seed"]
    pick = kinds[salt % len(kinds)]

    if pick == "boundary":
        cands = hv3.boundary_candidates(cfg, space)
        if cands:
            c = cands[int(rng.integers(len(cands)))]
            cfg["params"][c["param"]] = c["new_value"]
            return cfg, f"boundary_push:{c['param']}:{c['edge']}->{c['new_value']}"
        pick = "jitter"

    if pick == "jitter":
        names = [n for n in space if n in cfg.get("params", {})] or list(space)
        name = names[int(rng.integers(len(names)))]
        spec = space[name]
        if isinstance(spec, dict):
            lo, hi = spec["low"], spec["high"]
            cur = cfg.setdefault("params", {}).get(name, float(np.sqrt(lo * hi)))
            new = float(np.clip(cur * float(np.exp(rng.normal(0, 0.45))), lo, hi))
            new = round(new, 5)
        else:
            lo, hi = spec
            cur = cfg.setdefault("params", {}).get(name, (lo + hi) / 2)
            step = (hi - lo) * 0.30
            new = float(np.clip(cur + rng.normal(0, step), lo, hi))
            new = int(round(new)) if isinstance(lo, int) and isinstance(hi, int) else round(new, 4)
        cfg["params"][name] = new
        return cfg, f"jitter:{name}={new}"

    if pick == "features":
        groups = list((cfg.get("features") or {}).get("groups", []))
        avail = [g for g in FEATURE_GROUPS if g not in groups]
        if avail and (not groups or rng.random() < 0.6):
            g = avail[int(rng.integers(len(avail)))]
            groups = sorted(groups + [g])
            label = f"feat_add:{g}"
        elif groups:
            g = groups[int(rng.integers(len(groups)))]
            groups = sorted([x for x in groups if x != g])
            label = f"feat_drop:{g}"
        else:
            return None, None
        cfg["features"] = {"groups": groups}
        return cfg, f"{label} -> {groups or 'base-only'}"

    cfg["seed"] = int(rng.integers(1, 10_000))
    return cfg, f"seed_change:{cfg['seed']}"


def solo_pool(tree, limit=None):
    """Evaluated solo node ids, best-first."""
    solos = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n.get("kind", "solo") == "solo"
             and n["score"] is not None]
    solos.sort(key=lambda n: n["score"])
    ids = [n["id"] for n in solos]
    return ids[:limit] if limit else ids


def mutate_blend(tree, cfg, salt):
    rng = rng_for(tree, salt)
    members = sorted(cfg.get("members", []))
    pool = solo_pool(tree)
    unused = [m for m in pool if m not in members]
    r = rng.random()
    if unused and (len(members) < 3 or r < 0.55):
        m = unused[0] if r < 0.3 else unused[int(rng.integers(len(unused)))]
        return ({"kind": "blend", "members": sorted(members + [m]),
                 "weight_search": cfg.get("weight_search", "dirichlet")},
                f"blend_add_member:#{m}")
    if len(members) > 2 and r < 0.8:
        worst = max(members, key=lambda m: next(n["score"] for n in tree["nodes"] if n["id"] == m))
        return ({"kind": "blend", "members": sorted([m for m in members if m != worst]),
                 "weight_search": cfg.get("weight_search", "dirichlet")},
                f"blend_drop_member:#{worst}")
    other = "grid_simplex" if cfg.get("weight_search", "dirichlet") == "dirichlet" else "dirichlet"
    if other == "grid_simplex" and len(members) > 5:
        return None, None
    return ({"kind": "blend", "members": members, "weight_search": other},
            f"blend_weight_search:{other}")


def propose_child(tree, parent_id, salt):
    parent = next(n for n in tree["nodes"] if n["id"] == parent_id)
    cfg = parent["config"]
    if cfg.get("kind") == "blend":
        return mutate_blend(tree, cfg, salt)
    return mutate_solo(tree, cfg, salt)


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------
ROOT_CFG = {"kind": "solo", "model": "cat",
            "params": {"depth": 6, "learning_rate": 0.06},
            "features": {"groups": ["cnt"]}, "seed": 42}

SEEDS = [
    ("seed:lgb-l1-cnt", {"kind": "solo", "model": "lgb",
                         "params": {"learning_rate": 0.05, "num_leaves": 31},
                         "features": {"groups": ["cnt"]}, "seed": 42}),
    ("seed:cat-deep", {"kind": "solo", "model": "cat",
                       "params": {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
                       "features": {"groups": ["cnt"]}, "seed": 42}),
    ("seed:xgb-cnt", {"kind": "solo", "model": "xgb",
                      "params": {"max_depth": 6, "learning_rate": 0.05},
                      "features": {"groups": ["cnt"]}, "seed": 42}),
    ("seed:lgb-base-only", {"kind": "solo", "model": "lgb",
                            "params": {"learning_rate": 0.03, "num_leaves": 63,
                                       "min_data_in_leaf": 40},
                            "features": {"groups": []}, "seed": 7}),
]

BURST_SEEDS = [
    ("burst:lgb-huber-lowlr", {"kind": "solo", "model": "lgb",
                               "params": {"objective": "huber", "alpha": 1.5,
                                          "learning_rate": 0.02, "num_leaves": 127,
                                          "min_data_in_leaf": 60},
                               "features": {"groups": ["cnt"]}, "seed": 11}),
    ("burst:lgb-quantile50", {"kind": "solo", "model": "lgb",
                              "params": {"objective": "quantile", "alpha": 0.5,
                                         "learning_rate": 0.04, "num_leaves": 63},
                              "features": {"groups": ["cnt"]}, "seed": 13}),
    ("burst:cat-shallow-slow", {"kind": "solo", "model": "cat",
                                "params": {"depth": 4, "learning_rate": 0.03,
                                           "l2_leaf_reg": 10.0, "iterations": 6000},
                                "features": {"groups": ["cnt", "resid"]}, "seed": 17}),
    ("burst:xgb-deep-regularized", {"kind": "solo", "model": "xgb",
                                    "params": {"max_depth": 10, "learning_rate": 0.03,
                                               "min_child_weight": 30, "subsample": 0.7,
                                               "colsample_bytree": 0.6, "reg_lambda": 5.0},
                                    "features": {"groups": ["cnt"]}, "seed": 19}),
    ("burst:lgb-allfeats-strongreg", {"kind": "solo", "model": "lgb",
                                      "params": {"learning_rate": 0.03, "num_leaves": 200,
                                                 "min_data_in_leaf": 100,
                                                 "feature_fraction": 0.5,
                                                 "lambda_l2": 5.0},
                                      "features": {"groups": FEATURE_GROUPS}, "seed": 23}),
]


def inject_explore_burst(tree):
    ds = dstate(tree)
    note(tree, "\n>>> EXPLORE BURST (mandatory; 07_tree_search.md §3)")
    root_id = tree["root_id"]

    # 1. mandatory kitchen-sink blend of the ENTIRE current solo pool
    pool = solo_pool(tree)
    if len(pool) >= 2:
        nid, dup, _ = eval_and_add(tree, root_id, "burst:kitchen_sink_blend",
                                   {"kind": "blend", "members": sorted(pool),
                                    "weight_search": "dirichlet"})
        if nid is None:
            note(tree, f"  kitchen-sink blend deduped against #{dup}")

    # 2. long-shot solo lineages, each sanity-gated
    for label, cfg in BURST_SEEDS:
        nid, dup, _ = eval_and_add(tree, root_id, label, cfg)
        if nid is None:
            note(tree, f"  {label} deduped against #{dup}")
            continue
        passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
        if not passed:
            note(tree, f"  !! burst-seed sanity gate FAILED for #{nid} "
                       f"(score {tree['nodes'][-1]['score']} > bound {bound:.6g}) "
                       f"-- lineage marked plateaued")
    ds["burst_injected"] = True
    hv3.save_search_state(tree, TREE_PATH)


# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        print(f"resumed tree with {len(tree['nodes'])} nodes")
    else:
        tree = hv3.v2.new_tree(COMP)
        hv3.init_budget(tree)
        dstate(tree)

    # ---- root: digit-for-digit verification (checklist item 1) ----
    if not tree["nodes"]:
        note(tree, "== ROOT ==")
        nid, _, r = eval_and_add(tree, None, "root:linear-best-solo (CatBoost d6 lr.06 +cnt)",
                                 ROOT_CFG, is_root=True)
        assert r["status"] == "evaluated", f"root eval failed: {r.get('error')}"
        assert round(r["score"], 6) == ROOT_EXPECTED, (
            f"ROOT VERIFICATION FAILED: got {r['score']!r}, expected {ROOT_EXPECTED} -- "
            f"the root's own data/features drifted from the linear-iteration run; "
            f"aborting before the drift contaminates the whole tree.")
        note(tree, f"  root verified digit-for-digit == {ROOT_EXPECTED}")
        hv3.save_search_state(tree, TREE_PATH)

    # ---- first-generation lineage seeds ----
    if len([n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]) == 0:
        note(tree, "\n== FIRST-GENERATION SEEDS ==")
        for label, cfg in SEEDS:
            eval_and_add(tree, tree["root_id"], label, cfg)
        pool = solo_pool(tree, limit=3)
        eval_and_add(tree, tree["root_id"], "seed:blend-top3",
                     {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet"})
        hv3.save_search_state(tree, TREE_PATH)

    # ---- main loop ----
    note(tree, "\n== MAIN LOOP ==")
    dup_streak = 0
    while not hv3.should_stop(tree):
        # NOTE: do NOT call update_phase() here -- hv3.add_node already calls it after
        # every node, and during "explore_burst" each call increments
        # evals_since_burst_improve, so an extra manual call would double-count the
        # patience counter and stop the burst early.
        phase = tree["search_state"]["budget"]["phase"]
        ds = dstate(tree)
        if phase == "explore_burst" and not ds["burst_injected"]:
            inject_explore_burst(tree)
            continue
        parent_id, lineage_id = hv3.v2.select_next_parent(tree)
        if parent_id is None:
            note(tree, "select_next_parent returned None -- nothing left to expand")
            break

        key = str(lineage_id)
        salt = ds["mut_counter"].get(key, 0)
        ds["mut_counter"][key] = salt + 1

        placed = False
        for attempt in range(4):
            cfg, mutation = propose_child(tree, parent_id, salt + attempt)
            if cfg is None:
                continue
            nid, dup, _ = eval_and_add(tree, parent_id, mutation, cfg)
            if nid is not None:
                placed = True
                dup_streak = 0
                break
            note(tree, f"  (dedup vs #{dup}: {mutation})")
        if not placed:
            dup_streak += 1
            # let the harness burn an expansion slot so the search cannot spin forever
            hv3.add_node(tree, parent_id, "dedup_exhausted", {"kind": "solo",
                         "dedup_placeholder": True, "parent": parent_id,
                         "n": len(tree["nodes"])}, None, "failed", 0.0)
            if dup_streak >= 6:
                note(tree, "6 consecutive dedup-exhausted proposals -- stopping")
                break
        hv3.save_search_state(tree, TREE_PATH)
        if time.time() - t_start > 5400:
            note(tree, "driver wall-clock guard (90 min) hit -- stopping")
            break

    # ---- report (07_tree_search.md §6 honest reporting rules) ----
    hv3.save_search_state(tree, TREE_PATH)
    best = hv3.v2.global_best(tree)
    ds = dstate(tree)
    n_eval = hv3.n_evaluated(tree)
    report = {
        "comp": COMP,
        "metric": "MAE after clip[1,29]+round (the submitted form)",
        "linear_iteration_best": LINEAR_BEST,
        "root_score": tree["nodes"][0]["score"],
        "tree_best_score": best["score"],
        "tree_best_node": best["id"],
        "tree_best_config": best["config"],
        "tree_best_result": ds["results"].get(str(best["id"])),
        "improvement_vs_linear": round(LINEAR_BEST - best["score"], 6),
        "n_nodes": len(tree["nodes"]),
        "n_evaluated": n_eval,
        "evals_to_match_linear_best": ds["evals_to_match_linear_best"],
        "dedup_rejections": ds.get("dedup_rejections", []),
        "plateaued_lineages": tree["search_state"].get("plateaued"),
        "backtrack_log": tree["search_state"].get("backtrack_log"),
        "phase": tree["search_state"]["budget"]["phase"],
        "wall_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(COMP_DIR, "tree_search_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n" + "=" * 74)
    print(json.dumps({k: v for k, v in report.items() if k != "backtrack_log"},
                     indent=2, default=str))
    print("=" * 74)
    print("\nTop 12 evaluated nodes:")
    ev = sorted([n for n in tree["nodes"] if n["status"] == "evaluated"],
                key=lambda n: n["score"])
    for n in ev[:12]:
        print(f"  #{n['id']:<3d} {n['score']:.6f}  [{n.get('kind')}] {n['mutation']}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "10")
    main()
