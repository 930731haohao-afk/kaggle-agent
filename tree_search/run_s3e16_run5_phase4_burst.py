"""tree_search/run_s3e16_run5_phase4_burst.py -- the mandatory explore burst, forced.

Phase 3 ended on the hard budget cap (115/115) WITHOUT the phase machine ever flipping to
"explore_burst": plateau-saturation never triggered because the blend lineage kept finding
small improvements, so `update_phase` stayed in "exploit" until the numeric cap fired.
07_tree_search.md §3 is explicit that the burst must not be skipped -- three independent
lines of evidence say every post-exploit gain came from a mandatorily-injected kitchen-sink
mega-blend, never from a hand-written long-shot solo alone. So it is injected here by hand
rather than quietly omitted, with the budget raised to pay for it.
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v3 as hv3  # noqa: E402
import run_s3e16_run5 as p1  # noqa: E402

EVAL_PATH = os.path.join(_HERE, "eval_s3e16_run5b.py")
BUDGET = 132


def main():
    t0 = time.time()
    tree = hv3.load_search_state(p1.TREE_PATH)
    p1.EVAL_PATH = EVAL_PATH
    ds, log = p1.ds, p1.log
    before = hv3.global_best(tree)["score"]
    log(f"resumed: {len(tree['nodes'])} nodes, best={before}")

    b = tree["search_state"]["budget"]
    b["total_budget"] = BUDGET
    b["phase"] = "explore_burst"
    b["stop_reason"] = None
    b["burst_start_eval"] = hv3.n_evaluated(tree)
    b["best_at_burst_start"] = before
    b["evals_since_burst_improve"] = 0

    pool_all = p1.top_solos(tree, n=99)
    blends = [n["id"] for n in tree["nodes"]
              if n.get("kind") == "blend" and n.get("status") == "evaluated"
              and n["score"] is not None]
    blends.sort(key=lambda i: next(n["score"] for n in tree["nodes"] if n["id"] == i))

    burst = [
        (f"kitchen-sink blend of the ENTIRE solo pool ({len(pool_all)} members, k=6000)",
         {"kind": "blend", "members": sorted(pool_all), "k": 6000}),
        ("kitchen-sink blend of the top-20 solos (k=6000)",
         {"kind": "blend", "members": sorted(p1.top_solos(tree, n=20)), "k": 6000}),
        ("kitchen-sink blend of the top-15 solos (k=6000)",
         {"kind": "blend", "members": sorted(p1.top_solos(tree, n=15)), "k": 6000}),
        ("kitchen-sink blend of the top-10 solos (k=6000)",
         {"kind": "blend", "members": sorted(p1.top_solos(tree, n=10)), "k": 6000}),
        ("kitchen-sink blend of the top-8 solos (k=6000)",
         {"kind": "blend", "members": sorted(p1.top_solos(tree, n=8)), "k": 6000}),
        ("kitchen-sink blend of the top-5 solos (k=6000)",
         {"kind": "blend", "members": sorted(p1.top_solos(tree, n=5)), "k": 6000}),
        ("champion membership re-searched at k=6000",
         {"kind": "blend",
          "members": sorted(hv3.global_best(tree)["config"].get("members") or []),
          "k": 6000}),
        ("long-shot: blend of the champion's members plus every ordinal-head solo",
         {"kind": "blend",
          "members": sorted(set(hv3.global_best(tree)["config"].get("members") or [])
                            | {n["id"] for n in tree["nodes"]
                               if n.get("status") == "evaluated"
                               and n["config"].get("model") == "lgbmc"}),
          "k": 6000}),
    ]

    for mut, cfg in burst:
        nid, _res = p1.add_and_eval(tree, 0, "[burst4-forced] " + mut, cfg)
        if nid is not None and cfg.get("kind") == "solo":
            passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
            if not passed:
                log(f"  burst sanity gate FIRED on #{nid} (bound {bound})")
        hv3.save_search_state(tree, p1.TREE_PATH)

    gb = hv3.global_best(tree)
    log(f"forced burst done: {before} -> {gb['score']} (node #{gb['id']})")
    report = {"phase": "phase4_forced_burst", "best_before": before,
              "best_after": gb["score"], "best_node": gb["id"],
              "best_config": gb["config"],
              "best_result": ds(tree).get("node_results", {}).get(str(gb["id"])),
              "n_evaluated": hv3.n_evaluated(tree),
              "wall_s": round(time.time() - t0, 1),
              "burst_nodes": [{"id": n["id"], "score": n["score"],
                               "mutation": n["mutation"][:100]}
                              for n in tree["nodes"]
                              if n["mutation"].startswith("[burst4-forced]")]}
    with open(os.path.join(p1.COMP_DIR, "tree_search_report_phase4.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=hv3._json_default)
    hv3.save_search_state(tree, p1.TREE_PATH)
    log(json.dumps(report["burst_nodes"], indent=1))


if __name__ == "__main__":
    main()
