"""tree_search/run_s5e10_july_v3.py — harness_v3 driver for playground-series-s5e10
(road accident risk, RMSE, minimize).

Follows references/07_tree_search.md §4's five-item driver checklist:
  1. digit-for-digit root verification against the linear stage-3 baseline (LGB_base,
     OOF RMSE 0.056048)
  2. OOF cache reuse -- NOT used. The only pre-existing cache for this competition is
     tree_search/cache_s5e10 (an earlier run of this same competition) and this run's
     lane isolation forbids reading it, so every node is trained for real into the fresh
     cache_s5e10_july_tree. Cost is affordable: an LGB node is ~11s of training.
  3. resume-state contract -- all driver bookkeeping lives in
     tree["search_state"]["driver_state"], never in module globals
  4. subprocess-level evaluation timeout -- hv3.eval_solo_subprocess for every node
  5. burst-seed sanity gate after every explore-burst long-shot

Node space (07_tree_search.md §2, "general tabular GBDT-dominant" -> solo+blend dual
track), with one comp-local addition that is the whole point of this search: the EDA
(competitions/.../eda_structure.json) shows the target is near-purely ADDITIVE, so the
mutation set includes LightGBM/XGBoost `interaction_constraints` lanes that force the
GBDT toward additivity -- a direction a GBDT never takes on its own.
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import harness_v3 as hv3  # noqa: E402
import harness_v2 as hv2  # noqa: E402

COMP = "playground-series-s5e10"
COMP_DIR = os.path.join(_REPO, "competitions", COMP)
EVAL_MODULE = os.path.join(_HERE, "eval_s5e10_july.py")
TREE_PATH = os.path.join(COMP_DIR, "experiments_tree_v3_july_r2.json")
CACHE_DIR = os.path.join(_HERE, "cache_s5e10_july_tree2")

LINEAR_BEST = 0.056004        # stage-3 BLEND_5way
ROOT_EXPECTED = 0.056048      # stage-3 LGB_base, must reproduce digit-for-digit
TOTAL_BUDGET = 60
SOLO_TIMEOUT_S = 900

ENG = ["speed_hi", "is_night", "acc_hi", "curv_x_acchi", "curv_sq", "acc_clip"]
NOISE_FEATS = ["num_lanes", "time_of_day", "road_signs_present", "school_season"]

ROOT_CFG = {"kind": "solo", "model": "lgb", "params": {}, "features": {"drop": list(ENG)}}

SEARCH_SPACE = {
    "lgb": {"learning_rate": {"low": 0.01, "high": 0.15, "log": True},
            "num_leaves": (7, 255), "min_child_samples": (10, 400),
            "lambda_l2": (0.0, 20.0), "feature_fraction": (0.5, 1.0),
            "bagging_fraction": (0.5, 1.0)},
    "xgb": {"learning_rate": {"low": 0.01, "high": 0.15, "log": True},
            "max_depth": (3, 12), "min_child_weight": (1, 200), "reg_lambda": (0.0, 20.0)},
    "cat": {"learning_rate": {"low": 0.02, "high": 0.2, "log": True},
            "depth": (4, 10), "l2_leaf_reg": (1.0, 30.0)},
}

# per-model mutation queues; each entry is (name, params-delta, features-delta or None)
MUT_LGB = [
    ("ic_additive", {"interaction_constraints": "additive"}, None),
    ("ic_additive_curvacc", {"interaction_constraints": "additive+curv_acc"}, None),
    ("leaves31", {"num_leaves": 31}, None),
    ("leaves127", {"num_leaves": 127}, None),
    ("mcs150", {"min_child_samples": 150}, None),
    ("lr003_more", {"learning_rate": 0.03}, None),
    ("l2_10", {"lambda_l2": 10.0}, None),
    ("ff07", {"feature_fraction": 0.7}, None),
    ("bag06", {"bagging_fraction": 0.6}, None),
    ("linear_tree", {"linear_tree": True, "lambda_l2": 5.0}, None),
    ("drop_noise", {}, {"drop_add": NOISE_FEATS}),
    ("add_eng", {}, {"drop_remove": ENG}),
    ("seed2024", {"seed": 2024, "bagging_seed": 2024, "feature_fraction_seed": 2024}, None),
    ("seed7", {"seed": 7, "bagging_seed": 7, "feature_fraction_seed": 7}, None),
    ("mcs400", {"min_child_samples": 400}, None),
    ("leaves15", {"num_leaves": 15}, None),
]
MUT_XGB = [
    ("ic_additive", {"interaction_constraints": "additive"}, None),
    ("ic_additive_curvacc", {"interaction_constraints": "additive+curv_acc"}, None),
    ("depth5", {"max_depth": 5}, None),
    ("depth9", {"max_depth": 9}, None),
    ("mcw80", {"min_child_weight": 80}, None),
    ("lr003", {"learning_rate": 0.03}, None),
    ("lam10", {"reg_lambda": 10.0}, None),
    ("add_eng", {}, {"drop_remove": ENG}),
    ("seed2024", {"random_state": 2024}, None),
]
MUT_CAT = [
    ("depth6_fast", {"depth": 6, "learning_rate": 0.12}, None),
    ("depth10", {"depth": 10, "learning_rate": 0.12}, None),
    ("l2_15", {"l2_leaf_reg": 15.0, "learning_rate": 0.12}, None),
    ("add_eng", {"learning_rate": 0.12}, {"drop_remove": ENG}),
    ("seed2024", {"random_seed": 2024, "learning_rate": 0.12}, None),
]
MUT_RIDGE = [
    ("knots24", {"n_knots": 24}, None),
    ("alpha10", {"alpha": 10.0}, None),
    ("knots40_a01", {"n_knots": 40, "alpha": 0.1}, None),
]
MUT_BY_MODEL = {"lgb": MUT_LGB, "xgb": MUT_XGB, "cat": MUT_CAT, "ridge": MUT_RIDGE}


def core(cfg):
    """Canonical hashable stored form (07_tree_search.md §2): drop output-only fields,
    sort blend members."""
    c = {k: v for k, v in cfg.items() if k not in ("result", "want_importance")}
    if c.get("kind") == "blend":
        c["members"] = sorted(c["members"])
    if c.get("kind") == "solo":
        c["features"] = {"drop": sorted((c.get("features") or {}).get("drop", []))}
        c["params"] = dict(sorted((c.get("params") or {}).items()))
    return c


def ds(tree):
    return tree["search_state"].setdefault("driver_state", {})


def evaluate_node(cfg, node_id):
    return hv3.eval_solo_subprocess(EVAL_MODULE, core(cfg), SOLO_TIMEOUT_S, node_id=node_id)


def eval_and_add(tree, parent_id, mutation, cfg, *, root=False):
    """Reserve the id the harness will assign, evaluate under it (so the OOF cache key
    matches the node id), then add. Returns (nid, result) or (None, None) on dedup."""
    cfg = core(cfg)
    if not root and hv2.find_duplicate_config(tree, cfg) is not None:
        return None, None
    nid = max((n["id"] for n in tree["nodes"]), default=-1) + 1
    r = evaluate_node(cfg, nid)
    if root:
        got = hv3.add_root(tree, mutation, cfg, r["score"], r["status"], r["wall_s"])
        assert got == nid, f"root id mismatch {got} != {nid}"
        return nid, r
    got, _dup = hv3.add_node(tree, parent_id, mutation, cfg, r["score"], r["status"], r["wall_s"])
    if got is None:
        return None, None
    assert got == nid, f"node id mismatch {got} != {nid}"
    return got, r


def evaluated_solos(tree):
    return [n["id"] for n in tree["nodes"]
            if n["status"] == "evaluated" and n.get("kind") == "solo" and n["score"] is not None]


def apply_mut(base_cfg, params_delta, feat_delta):
    cfg = json.loads(json.dumps(core(base_cfg)))
    cfg["params"] = {**cfg.get("params", {}), **(params_delta or {})}
    drop = set((cfg.get("features") or {}).get("drop", []))
    if feat_delta:
        drop |= set(feat_delta.get("drop_add", []))
        drop -= set(feat_delta.get("drop_remove", []))
    cfg["features"] = {"drop": sorted(drop)}
    return cfg


def propose_child(tree, parent_id):
    """Return (config, mutation_name) or (None, None) when this parent is exhausted."""
    parent = next(n for n in tree["nodes"] if n["id"] == parent_id)
    pcfg = core(parent["config"])
    st = ds(tree)
    used = st.setdefault("used_mutations", {})
    key = str(parent_id)
    tried = set(used.get(key, []))

    if pcfg.get("kind") == "blend":
        # grow the blend: absorb the best solo not yet in it
        members = set(pcfg["members"])
        pool = sorted(evaluated_solos(tree),
                      key=lambda i: next(n["score"] for n in tree["nodes"] if n["id"] == i))
        for cand in pool:
            if cand not in members:
                cfg = {"kind": "blend", "members": sorted(members | {cand}),
                       "weight_search": "dirichlet"}
                if hv2.find_duplicate_config(tree, core(cfg)) is None:
                    used.setdefault(key, []).append(f"absorb{cand}")
                    return cfg, f"blend_absorb_solo{cand}"
        return None, None

    model = pcfg["model"]
    # (a) boundary pushes first -- 07_tree_search.md §3 calls this a first-class mutation
    for b in hv3.boundary_candidates(pcfg, SEARCH_SPACE.get(model, {})):
        name = f"bpush_{b['param']}_{b['edge']}"
        if name in tried:
            continue
        val = b["new_value"]
        if b["param"] in ("num_leaves", "min_child_samples", "max_depth", "min_child_weight"):
            val = max(2, int(round(val)))
        cfg = apply_mut(pcfg, {b["param"]: val}, None)
        if hv2.find_duplicate_config(tree, core(cfg)) is None:
            used.setdefault(key, []).append(name)
            return cfg, name
    # (b) the model's mutation queue
    for name, pd_, fd in MUT_BY_MODEL.get(model, []):
        if name in tried:
            continue
        cfg = apply_mut(pcfg, pd_, fd)
        if core(cfg) == pcfg or hv2.find_duplicate_config(tree, core(cfg)) is not None:
            used.setdefault(key, []).append(name)
            continue
        used.setdefault(key, []).append(name)
        return cfg, name
    return None, None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        log(f"resumed tree with {len(tree['nodes'])} nodes")
    else:
        tree = hv3.new_tree(COMP)
        hv3.init_budget(tree, total_budget=TOTAL_BUDGET)
        t0 = time.time()
        nid, r = eval_and_add(tree, None, "root:LGB_base(stage3 baseline)", ROOT_CFG, root=True)
        log(f"root #{nid} RMSE {r['score']} ({r['wall_s']}s)")
        assert r["status"] == "evaluated", f"root failed: {r['error']}"
        assert round(float(r["score"]), 6) == ROOT_EXPECTED, (
            f"ROOT VERIFICATION FAILED: got {r['score']}, expected {ROOT_EXPECTED}. "
            "The tree-search feature/fold pipeline has drifted from stage 3 -- abort.")
        ds(tree)["t_start"] = t0
        ds(tree)["seeded"] = False
        hv3.save_search_state(tree, TREE_PATH)

    # ---- first-generation lineages (direct children of root) ----
    if not ds(tree).get("seeded"):
        seeds = [
            ("seed:lgb_ic_additive", apply_mut(ROOT_CFG, {"interaction_constraints": "additive"}, None)),
            ("seed:lgb_ic_additive_curvacc",
             apply_mut(ROOT_CFG, {"interaction_constraints": "additive+curv_acc"}, None)),
            ("seed:lgb_eng18", apply_mut(ROOT_CFG, {}, {"drop_remove": ENG})),
            ("seed:lgb_reg", apply_mut(ROOT_CFG, {"num_leaves": 31, "min_child_samples": 150,
                                                  "learning_rate": 0.03}, None)),
            ("seed:xgb", {"kind": "solo", "model": "xgb", "params": {},
                          "features": {"drop": list(ENG)}}),
            ("seed:cat", {"kind": "solo", "model": "cat", "params": {"learning_rate": 0.12},
                          "features": {"drop": list(ENG)}}),
            ("seed:ridge_additive", {"kind": "solo", "model": "ridge", "params": {},
                                     "features": {"drop": list(ENG)}}),
        ]
        for name, cfg in seeds:
            nid, r = eval_and_add(tree, tree["root_id"], name, cfg)
            if nid is None:
                log(f"{name}: dedup-rejected")
                continue
            log(f"#{nid} {name} -> {r['score']} ({r['wall_s']}s) {r.get('error') or ''}")
            hv3.save_search_state(tree, TREE_PATH)
        # first blend seed over everything evaluated so far
        solos = evaluated_solos(tree)
        nid, r = eval_and_add(tree, tree["root_id"], "seed:blend_all",
                              {"kind": "blend", "members": sorted(solos),
                               "weight_search": "dirichlet"})
        if nid is not None:
            log(f"#{nid} seed:blend_all -> {r['score']} weights={r['result']['weights']}")
        ds(tree)["seeded"] = True
        hv3.save_search_state(tree, TREE_PATH)

    # ---- main loop ----
    while not hv3.should_stop(tree):
        phase = hv3.update_phase(tree)
        if phase == "explore_burst" and not ds(tree).get("burst_injected"):
            log("=== EXPLORE BURST ===")
            solos = evaluated_solos(tree)
            burst = [
                ("burst:kitchen_sink_blend",
                 {"kind": "blend", "members": sorted(solos), "weight_search": "dirichlet"}),
                ("burst:lgb_tiny_lr", apply_mut(ROOT_CFG, {"learning_rate": 0.008,
                                                           "num_leaves": 255}, None)),
                ("burst:lgb_ic_add_eng_deep",
                 apply_mut(ROOT_CFG, {"interaction_constraints": "additive+curv_acc",
                                      "num_leaves": 255, "learning_rate": 0.02},
                           {"drop_remove": ENG})),
                ("burst:lgb_linear_ic_additive",
                 apply_mut(ROOT_CFG, {"linear_tree": True, "interaction_constraints": "additive",
                                      "lambda_l2": 5.0}, None)),
                ("burst:xgb_ic_add_deep",
                 {"kind": "solo", "model": "xgb",
                  "params": {"interaction_constraints": "additive+curv_acc", "max_depth": 10,
                             "learning_rate": 0.02},
                  "features": {"drop": []}}),
                ("burst:cat_deep_eng",
                 {"kind": "solo", "model": "cat",
                  "params": {"depth": 10, "learning_rate": 0.12, "l2_leaf_reg": 10.0},
                  "features": {"drop": []}}),
            ]
            for name, cfg in burst:
                nid, r = eval_and_add(tree, tree["root_id"], name, cfg)
                if nid is None:
                    log(f"{name}: dedup-rejected")
                    continue
                log(f"#{nid} {name} -> {r['score']} ({r['wall_s']}s) {r.get('error') or ''}")
                if cfg.get("kind") == "solo":
                    ok, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
                    if not ok:
                        log(f"   sanity gate FAILED (bound {bound}) -> lineage closed")
                hv3.save_search_state(tree, TREE_PATH)
            ds(tree)["burst_injected"] = True
            hv3.save_search_state(tree, TREE_PATH)
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            log("no expandable parent -> stop")
            break
        cfg, mutation = propose_child(tree, parent_id)
        if cfg is None:
            # parent exhausted: mark its lineage plateaued so selection moves on
            st = tree["search_state"]
            if lineage_id not in st["plateaued"]:
                st["plateaued"].append(lineage_id)
                st["backtrack_log"].append(dict(
                    at_node_id=parent_id, plateaued_lineage=lineage_id,
                    reason="driver: mutation queue exhausted for this parent"))
            hv3.save_search_state(tree, TREE_PATH)
            continue
        nid, r = eval_and_add(tree, parent_id, mutation, cfg)
        if nid is None:
            log(f"dedup-rejected child of #{parent_id}: {mutation}")
            hv3.save_search_state(tree, TREE_PATH)
            continue
        best = hv2.global_best(tree)
        flag = " <-- NEW BEST" if best and best["id"] == nid else ""
        log(f"#{nid} (parent #{parent_id}, L{lineage_id}) {mutation} -> {r['score']} "
            f"({r['wall_s']}s){flag} {r.get('error') or ''}")
        hv3.save_search_state(tree, TREE_PATH)

    # ---- report ----
    best = hv2.global_best(tree)
    n_eval = sum(1 for n in tree["nodes"] if n["status"] == "evaluated")
    order = [n for n in tree["nodes"] if n["status"] == "evaluated"]
    evals_to_match = None
    for i, n in enumerate(order, 1):
        if n["score"] is not None and n["score"] <= LINEAR_BEST:
            evals_to_match = i
            break
    report = {
        "comp": COMP,
        "best_node_id": best["id"], "best_score": best["score"],
        "best_config": best["config"],
        "linear_iteration_best": LINEAR_BEST,
        "root_score": next(n["score"] for n in tree["nodes"] if n["id"] == tree["root_id"]),
        "n_nodes": len(tree["nodes"]), "n_evaluated": n_eval,
        "evals_to_match_linear_best": evals_to_match,
        "dedup_rejections": tree["search_state"].get("dedup_rejections",
                                                     getattr(hv3, "DEDUP_REJECTIONS", None)),
        "plateaued_lineages": tree["search_state"].get("plateaued", []),
        "backtrack_log": tree["search_state"].get("backtrack_log", []),
        "cost_guard_log": tree["search_state"].get("cost_guard_log", []),
        "phase": tree["search_state"]["budget"]["phase"],
        "wall_s": round(time.time() - ds(tree).get("t_start", time.time()), 1),
        "failed_nodes": [{"id": n["id"], "mutation": n["mutation"]}
                         for n in tree["nodes"] if n["status"] == "failed"],
    }
    with open(os.path.join(COMP_DIR, "tree_search_report_july.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(json.dumps({k: v for k, v in report.items()
                    if k not in ("backtrack_log", "best_config")}, indent=2, default=str))
    log(f"BEST #{best['id']} {best['score']} :: {json.dumps(best['config'], default=str)}")


if __name__ == "__main__":
    main()
