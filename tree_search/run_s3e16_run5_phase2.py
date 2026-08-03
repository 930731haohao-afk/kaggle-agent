"""tree_search/run_s3e16_run5_phase2.py -- phase-2 resume of the s3e16 Stage-4 search.

Phase 1 (run_s3e16_run5.py) stopped on the post-burst patience rule at 45 evaluated nodes
with global best 1.333729 (blend #48). Round 2 then produced two members the phase-1 node
space structurally could not reach:

  * lgb_optuna_raw -- LGB tuned by Optuna whose OBJECTIVE was the rounded OOF MAE itself.
    Solo 1.334256, i.e. one solo nearly matching the whole phase-1 blend.
  * lgb_multiclass_median -- P(Age=a|x) via a multiclass head, weighted median read off it
    (the Bayes rule for MAE on a discrete target). Solo 1.340563, structurally unlike
    every regressor in the pool.

This driver resumes the SAME tree (ids continue, dedup still sees every phase-1 config),
imports those members as new root lineages with a digit-verified OOF reuse, raises the
budget, reopens the phase machine and lets the search absorb them.
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v3 as hv3  # noqa: E402
import run_s3e16_run5 as p1  # noqa: E402

EVAL_PATH = os.path.join(_HERE, "eval_s3e16_run5b.py")
NEW_BUDGET = 110

# name -> (node config, expected rounded MAE from round2_result.json / its stdout)
NEW_SEEDS = [
    ("lgb_optuna_raw", {"kind": "solo", "model": "lgb", "features": {"set": "raw"},
                        "params": {"learning_rate": 0.013220877172448705,
                                   "num_leaves": 27, "min_child_samples": 12,
                                   "feature_fraction": 0.723265929656533,
                                   "bagging_fraction": 0.5035092372450791,
                                   "reg_alpha": 2.5289053692819397,
                                   "reg_lambda": 0.0014824522730226845,
                                   "max_bin": 183, "bagging_freq": 1}}, 1.334256),
    ("lgb_optuna_seed2024", {"kind": "solo", "model": "lgb", "features": {"set": "raw"},
                             "params": {"learning_rate": 0.013220877172448705,
                                        "num_leaves": 27, "min_child_samples": 12,
                                        "feature_fraction": 0.723265929656533,
                                        "bagging_fraction": 0.5035092372450791,
                                        "reg_alpha": 2.5289053692819397,
                                        "reg_lambda": 0.0014824522730226845,
                                        "max_bin": 183, "bagging_freq": 1,
                                        "seed": 2024, "bagging_seed": 2024,
                                        "feature_fraction_seed": 2024}}, 1.335782),
    ("lgb_optuna_seed7", {"kind": "solo", "model": "lgb", "features": {"set": "raw"},
                          "params": {"learning_rate": 0.013220877172448705,
                                     "num_leaves": 27, "min_child_samples": 12,
                                     "feature_fraction": 0.723265929656533,
                                     "bagging_fraction": 0.5035092372450791,
                                     "reg_alpha": 2.5289053692819397,
                                     "reg_lambda": 0.0014824522730226845,
                                     "max_bin": 183, "bagging_freq": 1,
                                     "seed": 7, "bagging_seed": 7,
                                     "feature_fraction_seed": 7}}, 1.336376),
    ("mc_raw_median", {"kind": "solo", "model": "lgbmc", "features": {"set": "raw"},
                       "params": {"decision": "median"}}, 1.340563),
    ("mc_eng_median", {"kind": "solo", "model": "lgbmc", "features": {"set": "eng"},
                       "params": {"decision": "median"}}, 1.344533),
    ("mc_raw_mean", {"kind": "solo", "model": "lgbmc", "features": {"set": "raw"},
                     "params": {"decision": "mean"}}, 1.377780),
]

OOF_FILES = {
    "lgb_optuna_raw": "lgb_optuna_raw",
    "lgb_optuna_seed2024": "lgb_optuna_seed2024",
    "lgb_optuna_seed7": "lgb_optuna_seed7",
    "mc_raw_median": "mc_raw_median",
    "mc_eng_median": "mc_eng_median",
    "mc_raw_mean": "mc_raw_mean",
}


def main():
    t_start = time.time()
    tree = hv3.load_search_state(p1.TREE_PATH)
    p1.EVAL_PATH = EVAL_PATH          # phase-2 evaluator for every new solo eval
    ds = p1.ds
    log = p1.log
    log(f"resumed: {len(tree['nodes'])} nodes, phase="
        f"{tree['search_state']['budget']['phase']}, best="
        f"{hv3.global_best(tree)['score']}")

    if not ds(tree).get("phase2_seeded"):
        b = tree["search_state"]["budget"]
        b["total"] = NEW_BUDGET
        b["phase"] = "exploit"
        b["burst_started_at"] = None
        tree["search_state"]["plateaued"] = []
        tree["search_state"]["streak"] = {}
        ds(tree)["burst_injected"] = False
        ds(tree)["phase2_seeded"] = True

        for name, cfg, expected in NEW_SEEDS:
            nid = hv3.v2.v1.next_id(tree)
            got = p1.import_round1_oof(tree, OOF_FILES[name], nid, expected)
            added, dup = hv3.add_node(tree, 0, f"phase-2 seed: {name}", p1.core(cfg),
                                      got, "evaluated", 0.0)
            if added is None:
                log(f"  seed {name} dedup-rejected against #{dup}")
                continue
            ds(tree)["node_names"][str(added)] = name
            ds(tree)["node_results"][str(added)] = {"mae": got, "reused_from": name}
            log(f"  seeded #{added} {name} mae={got}")

        p1.add_and_eval(tree, 0, "phase-2 blend: top-10 solos incl. the new members",
                        {"kind": "blend", "members": sorted(p1.top_solos(tree, n=10)),
                         "k": 1500}, name="blend_phase2_top10")
        p1.add_and_eval(tree, 0, "phase-2 blend: entire solo pool",
                        {"kind": "blend", "members": sorted(p1.top_solos(tree, n=99)),
                         "k": 1500}, name="blend_phase2_all")
        hv3.save_search_state(tree, p1.TREE_PATH)

    while not hv3.should_stop(tree):
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not ds(tree).get("burst_injected"):
            log("=== PHASE-2 EXPLORE BURST ===")
            ds(tree)["burst_injected"] = True
            burst = [
                ("kitchen-sink blend of the entire solo pool (k=3000)",
                 {"kind": "blend", "members": sorted(p1.top_solos(tree, n=99)),
                  "k": 3000}),
                ("kitchen-sink blend of the top-12 solos (k=3000)",
                 {"kind": "blend", "members": sorted(p1.top_solos(tree, n=12)),
                  "k": 3000}),
                ("long-shot: multiclass head on the tuned-LGB hyperparameters",
                 {"kind": "solo", "model": "lgbmc", "features": {"set": "raw"},
                  "params": {"decision": "median", "learning_rate": 0.03,
                             "num_leaves": 27, "min_child_samples": 12,
                             "feature_fraction": 0.72, "bagging_fraction": 0.5}}),
                ("long-shot: multiclass median on the 'core' feature set",
                 {"kind": "solo", "model": "lgbmc", "features": {"set": "core"},
                  "params": {"decision": "median"}}),
                ("long-shot: tuned params, much lower lr + many rounds",
                 {"kind": "solo", "model": "lgb", "features": {"set": "raw"},
                  "params": {"learning_rate": 0.005, "num_leaves": 27,
                             "min_child_samples": 12, "feature_fraction": 0.723,
                             "bagging_fraction": 0.5, "reg_alpha": 2.53,
                             "reg_lambda": 0.0015, "max_bin": 183, "bagging_freq": 1,
                             "num_boost_round": 12000,
                             "early_stopping_rounds": 400}}),
                ("long-shot: tuned params on the 'core' feature set",
                 {"kind": "solo", "model": "lgb", "features": {"set": "core"},
                  "params": {"learning_rate": 0.0132, "num_leaves": 27,
                             "min_child_samples": 12, "feature_fraction": 0.723,
                             "bagging_fraction": 0.5035, "reg_alpha": 2.529,
                             "reg_lambda": 0.0015, "max_bin": 183,
                             "bagging_freq": 1}}),
            ]
            for mut, cfg in burst:
                if hv3.should_stop(tree):
                    break
                nid, _res = p1.add_and_eval(tree, 0, "[burst2] " + mut, cfg)
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
        cfg, mutation = p1.propose_child(tree, parent_id, lineage_id)
        p1.add_and_eval(tree, parent_id, mutation, cfg)
        hv3.save_search_state(tree, p1.TREE_PATH)
        if time.time() - t_start > 3000:
            log("driver wall-clock guard (50 min) hit -- stopping early")
            break

    gb = hv3.global_best(tree)
    st = tree["search_state"]
    evaluated = [n for n in tree["nodes"] if n.get("status") == "evaluated"]
    lin_blend = ds(tree)["linear_best_blend"]
    order = sorted(evaluated, key=lambda n: n["id"])
    evals_to_match = next((i for i, n in enumerate(order, start=1)
                           if n["score"] is not None and n["score"] <= lin_blend), None)
    report = {
        "phase": "phase2", "n_nodes": len(tree["nodes"]),
        "n_evaluated": len(evaluated),
        "global_best": {"id": gb["id"], "score": gb["score"], "config": gb["config"],
                        "result": ds(tree).get("node_results", {}).get(str(gb["id"]))},
        "linear_best_solo": ds(tree)["linear_best_solo"],
        "linear_best_blend": lin_blend,
        "phase1_best": 1.333729,
        "evals_to_match_linear_best_blend": evals_to_match,
        "budget_phase": st["budget"]["phase"],
        "plateaued_lineages": st.get("plateaued", []),
        "backtrack_log": st.get("backtrack_log", []),
        "dedup_streak": st.get("dedup_streak", {}),
        "cost_guard_log": st.get("cost_guard_log", []),
        "sanity_gate_fired": ds(tree).get("sanity_gate_fired", []),
        "wall_s": round(time.time() - t_start, 1),
        "top12": [{"id": n["id"], "score": n["score"], "kind": n.get("kind"),
                   "mutation": n["mutation"][:90]}
                  for n in sorted([e for e in evaluated if e["score"] is not None],
                                  key=lambda x: x["score"])[:12]],
    }
    with open(os.path.join(p1.COMP_DIR, "tree_search_report_phase2.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=hv3._json_default)
    hv3.save_search_state(tree, p1.TREE_PATH)
    log(json.dumps({k: report[k] for k in ("n_evaluated", "phase1_best", "budget_phase",
                                           "wall_s")}, indent=2))
    log(f"GLOBAL BEST #{gb['id']} score={gb['score']}")
    log("top12: " + json.dumps(report["top12"], indent=1))


if __name__ == "__main__":
    main()
