"""tree_search/run_s3e16_run5_phase3.py -- phase-3 resume of the s3e16 Stage-4 search.

Two reasons this phase exists, both recorded honestly rather than quietly fixed:

  1. Phase 2 raised the node budget by writing `budget["total"]`, but the harness reads
     `budget["total_budget"]`. The raise silently did nothing and the search hard-stopped
     at 60/60 -- ON THE VERY EVALUATION THAT SET A NEW GLOBAL BEST (node #81, 1.332892).
     A search that stops while still improving has not converged, so the budget is now
     raised correctly and the search continues.
  2. Round 3 produced one more member the earlier node space could not reach: the ordinal
     (multiclass -> weighted median) head, tuned by Optuna against the ROUNDED MAE. Solo
     1.334202 -- on par with the tuned regressor while being a different estimator.
"""

import json
import os
import sys
import time

import numpy as np  # noqa: F401  (kept so the module mirrors phase-2's import surface)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v3 as hv3  # noqa: E402
import run_s3e16_run5 as p1  # noqa: E402

EVAL_PATH = os.path.join(_HERE, "eval_s3e16_run5b.py")
NEW_TOTAL_BUDGET = 115
WALL_GUARD_S = float(os.environ.get("WALL_GUARD_S", 420))

NEW_SEEDS = [
    ("mc_optuna_median",
     {"kind": "solo", "model": "lgbmc", "features": {"set": "raw"},
      "params": {"decision": "median", "learning_rate": 0.0348753049056347,
                 "num_leaves": 21, "min_child_samples": 275,
                 "feature_fraction": 0.990465027646603,
                 "bagging_fraction": 0.8784546829334369,
                 "reg_alpha": 1.525842735657131,
                 "reg_lambda": 0.03465243828255803, "bagging_freq": 1}}, 1.334202),
]


NON_NEGATIVE = ("reg_alpha", "reg_lambda", "min_child_samples", "min_child_weight",
                "num_leaves", "max_bin", "max_depth", "depth", "l2_leaf_reg", "alpha")


def clamped_propose(tree, parent_id, lineage_id):
    """p1.propose_child, with the proposed params clamped back into legal territory.

    hv3.boundary_candidates pushes a param `edge_frac * push_factor` past the edge of its
    declared range with no knowledge of the LEARNER's own domain, so pushing
    reg_lambda=0.0015 or min_child_samples=12 off their low edges proposed NEGATIVE
    values, which LightGBM rejects outright ("Check failed: (lambda_l2) >= (0.0)"). Those
    nodes came back status="failed" and burned budget. Clamping here keeps the boundary
    push -- which is a genuinely productive mutation type -- while making its low-edge
    variant legal. Phase-1/2 proposals are untouched; this only affects new proposals.
    """
    cfg, mutation = p1.propose_child(tree, parent_id, lineage_id)
    params = cfg.get("params")
    if isinstance(params, dict):
        fixed = dict(params)
        for k in NON_NEGATIVE:
            if k in fixed and isinstance(fixed[k], (int, float)):
                floor = 1 if k in ("num_leaves", "max_bin", "max_depth", "depth") else 0
                if fixed[k] < floor:
                    fixed[k] = floor
                    mutation += f" [clamped {k} to {floor}]"
        cfg = {**cfg, "params": fixed}
    return cfg, mutation


def main():
    t_start = time.time()
    tree = hv3.load_search_state(p1.TREE_PATH)
    p1.EVAL_PATH = EVAL_PATH
    ds, log = p1.ds, p1.log
    # The ordinal head is a phase-3 model type, so the mutation proposer needs its
    # search space too -- without it, boundary_candidates KeyErrors on an lgbmc parent.
    p1.SEARCH_SPACE["lgbmc"] = {
        "learning_rate": {"low": 0.01, "high": 0.20, "log": True},
        "num_leaves": (7, 127), "min_child_samples": (10, 400),
        "feature_fraction": (0.4, 1.0), "bagging_fraction": (0.4, 1.0),
        "reg_alpha": (0.0, 10.0), "reg_lambda": (0.0, 10.0)}
    log(f"resumed: {len(tree['nodes'])} nodes, best={hv3.global_best(tree)['score']}")

    if not ds(tree).get("phase3_seeded"):
        b = tree["search_state"]["budget"]
        b["total_budget"] = NEW_TOTAL_BUDGET     # the key the harness actually reads
        b["phase"] = "exploit"
        b["stop_reason"] = None
        b["burst_start_eval"] = None
        b["best_at_burst_start"] = None
        b["evals_since_burst_improve"] = 0
        tree["search_state"]["plateaued"] = []
        tree["search_state"]["streak"] = {}
        ds(tree)["burst_injected"] = False
        ds(tree)["phase3_seeded"] = True
        ds(tree)["phase2_best"] = 1.332892

        for name, cfg, expected in NEW_SEEDS:
            nid = hv3.v2.v1.next_id(tree)
            got = p1.import_round1_oof(tree, name, nid, expected)
            added, dup = hv3.add_node(tree, 0, f"phase-3 seed: {name}", p1.core(cfg),
                                      got, "evaluated", 0.0)
            if added is None:
                log(f"  seed {name} dedup-rejected against #{dup}")
                continue
            ds(tree)["node_names"][str(added)] = name
            ds(tree)["node_results"][str(added)] = {"mae": got, "reused_from": name}
            log(f"  seeded #{added} {name} mae={got}")

        # re-blend around the new member, and around the phase-2 champion's membership
        gb_members = sorted(hv3.global_best(tree)["config"].get("members") or [])
        newest = max(n["id"] for n in tree["nodes"] if n["status"] == "evaluated")
        p1.add_and_eval(tree, 0, "phase-3 blend: phase-2 champion + the tuned ordinal head",
                        {"kind": "blend", "members": sorted(set(gb_members) | {newest}),
                         "k": 1500}, name="blend_phase3_champ_plus_ordinal")
        p1.add_and_eval(tree, 0, "phase-3 blend: top-12 solos",
                        {"kind": "blend", "members": sorted(p1.top_solos(tree, n=12)),
                         "k": 1500}, name="blend_phase3_top12")
        hv3.save_search_state(tree, p1.TREE_PATH)

    while not hv3.should_stop(tree):
        if tree["search_state"]["budget"]["phase"] == "explore_burst" \
                and not ds(tree).get("burst_injected"):
            log("=== PHASE-3 EXPLORE BURST ===")
            ds(tree)["burst_injected"] = True
            pool = p1.top_solos(tree, n=99)
            burst = [
                ("kitchen-sink blend of the entire solo pool (k=4000)",
                 {"kind": "blend", "members": sorted(pool), "k": 4000}),
                ("kitchen-sink blend of the top-15 solos (k=4000)",
                 {"kind": "blend", "members": sorted(p1.top_solos(tree, n=15)),
                  "k": 4000}),
                ("kitchen-sink blend of the top-6 solos (k=4000)",
                 {"kind": "blend", "members": sorted(p1.top_solos(tree, n=6)), "k": 4000}),
                ("long-shot: ordinal head on the 'full' feature set",
                 {"kind": "solo", "model": "lgbmc", "features": {"set": "full"},
                  "params": {"decision": "median", "learning_rate": 0.0349,
                             "num_leaves": 21, "min_child_samples": 275,
                             "feature_fraction": 0.99, "bagging_fraction": 0.878}}),
                ("long-shot: tuned-regressor params with a quantile(0.5) head",
                 {"kind": "solo", "model": "lgb", "features": {"set": "raw"},
                  "params": {"objective": "quantile", "alpha": 0.5,
                             "learning_rate": 0.0132, "num_leaves": 27,
                             "min_child_samples": 12, "feature_fraction": 0.723,
                             "bagging_fraction": 0.5035, "max_bin": 183,
                             "bagging_freq": 1}}),
                ("long-shot: CatBoost MAE with a deeper, lower-lr configuration",
                 {"kind": "solo", "model": "cat", "features": {"set": "raw"},
                  "params": {"depth": 8, "learning_rate": 0.025, "l2_leaf_reg": 8.0}}),
            ]
            for mut, cfg in burst:
                if hv3.should_stop(tree):
                    break
                nid, _res = p1.add_and_eval(tree, 0, "[burst3] " + mut, cfg)
                if nid is not None and cfg.get("kind") == "solo":
                    passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
                    if not passed:
                        log(f"  burst sanity gate FIRED on #{nid} (bound {bound})")
                        ds(tree).setdefault("sanity_gate_fired", []).append(
                            {"node": nid, "bound": bound})
                hv3.save_search_state(tree, p1.TREE_PATH)
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            log("no expandable parent left")
            break
        cfg, mutation = clamped_propose(tree, parent_id, lineage_id)
        p1.add_and_eval(tree, parent_id, mutation, cfg)
        hv3.save_search_state(tree, p1.TREE_PATH)
        if time.time() - t_start > WALL_GUARD_S:
            log(f"driver wall-clock guard ({WALL_GUARD_S:.0f}s) hit -- pausing; rerun to resume")
            tree["search_state"]["budget"]["stop_reason"] = "driver wall-clock guard"
            break

    gb = hv3.global_best(tree)
    st = tree["search_state"]
    evaluated = [n for n in tree["nodes"] if n.get("status") == "evaluated"]
    lin_blend = ds(tree)["linear_best_blend"]
    order = sorted(evaluated, key=lambda n: n["id"])
    evals_to_match = next((i for i, n in enumerate(order, start=1)
                           if n["score"] is not None and n["score"] <= lin_blend), None)
    report = {
        "phase": "phase3", "n_nodes": len(tree["nodes"]), "n_evaluated": len(evaluated),
        "global_best": {"id": gb["id"], "score": gb["score"], "config": gb["config"],
                        "result": ds(tree).get("node_results", {}).get(str(gb["id"]))},
        "linear_best_solo": ds(tree)["linear_best_solo"], "linear_best_blend": lin_blend,
        "phase1_best": 1.333729, "phase2_best": 1.332892,
        "evals_to_match_linear_best_blend": evals_to_match,
        "budget": st["budget"],
        "plateaued_lineages": st.get("plateaued", []),
        "backtrack_log": st.get("backtrack_log", []),
        "dedup_streak": st.get("dedup_streak", {}),
        "cost_guard_log": st.get("cost_guard_log", []),
        "sanity_gate_fired": ds(tree).get("sanity_gate_fired", []),
        "wall_s": round(time.time() - t_start, 1),
        "top15": [{"id": n["id"], "score": n["score"], "kind": n.get("kind"),
                   "mutation": n["mutation"][:90]}
                  for n in sorted([e for e in evaluated if e["score"] is not None],
                                  key=lambda x: x["score"])[:15]],
        "best_solos": [{"id": n["id"], "score": n["score"],
                        "name": ds(tree).get("node_names", {}).get(str(n["id"]))}
                       for n in sorted([e for e in evaluated
                                        if e["score"] is not None and e.get("kind") == "solo"],
                                       key=lambda x: x["score"])[:10]],
    }
    with open(os.path.join(p1.COMP_DIR, "tree_search_report_phase3.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=hv3._json_default)
    hv3.save_search_state(tree, p1.TREE_PATH)
    log(f"n_evaluated={len(evaluated)} stop_reason={st['budget'].get('stop_reason')} "
        f"wall={report['wall_s']}s")
    log(f"GLOBAL BEST #{gb['id']} score={gb['score']} kind={gb.get('kind')}")
    log("top15: " + json.dumps(report["top15"]))


if __name__ == "__main__":
    main()
