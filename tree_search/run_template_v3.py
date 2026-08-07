"""COMPETITION-AGNOSTIC v3 tree-search driver template.

This file is the one the skill instructions cite. It deliberately contains NO competition
constants: no recorded champion configs, no digit-verify targets, no linear-best scores. The
per-competition drivers (run_<comp>_v3.py) embed exactly those things — a recorded root
config, `LINEAR_BEST`, byte-level verification numbers — because their job is to REPRODUCE a
recorded tree. Citing one of them as "the template" handed a re-run of that competition its
own recorded champion as the starting point (2026-08-10 round-6 finding). A fresh benchmark
run copies THIS file, fills the placeholders from its own Stage 1–3 artifacts, and records
the copy's path into docs/rerun_manifest.json `driver_actual`.

What a driver must do (the Phase H-1 contract, references/07_tree_search.md):
  1. build the root from THIS run's own baseline (Stage 3's best solo config — never from a
     recorded tree or a previous run's champion);
  2. evaluate solos via the competition's pinned eval module (docs/rerun_manifest.json),
     blends via harness_v3.eval_blend_with_cost_guard;
  3. write every node through harness_v3.add_node so the budget/phase machine and dedup run;
  4. persist the tree after every node; honest-reporting fields per §6.
"""
import importlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402,F401  (cache_oof/load_oof live here)
import harness_v3 as hv3  # noqa: E402

# ---------------------------------------------------------------------------
# PLACEHOLDERS — fill from THIS run's own artifacts, then delete this comment.
# ---------------------------------------------------------------------------
COMP = "<competition-slug>"                    # e.g. from docs/rerun_manifest.json
EVAL_MODULE = "<eval_module_name>"             # the manifest's pinned eval module (no .py)
TOTAL_BUDGET = 60                              # nodes; per 07_tree_search.md §4
ROOT_CONFIG: dict = {}                         # Stage 3's best solo config — THIS run's own
TREE_PATH = os.path.join(_HERE, "..", "competitions", COMP, "experiments_tree_v3.json")


def main() -> None:
    if not ROOT_CONFIG or COMP.startswith("<"):
        raise SystemExit("fill the placeholders from this run's own artifacts first; a "
                         "driver seeded from a recorded tree is a warm start from the "
                         "competition's own answer")
    ev = importlib.import_module(EVAL_MODULE)
    tree = hv3.new_tree(COMP)
    tree["search_state"]["budget"] = {
        "total_budget": TOTAL_BUDGET, "explore_burst_size": 6, "post_burst_patience": 10,
        "phase": "exploit", "burst_start_eval": 0, "best_at_burst_start": None,
        "evals_since_burst_improve": 0}

    def persist():
        json.dump(tree, open(TREE_PATH, "w"), ensure_ascii=False, indent=2)

    # root
    nid = hv3.next_id(tree)
    r = ev.evaluate(ROOT_CONFIG, node_id=nid, timeout_s=600)
    hv3.add_node(tree, None, "root: this run's Stage-3 best solo", ROOT_CONFIG,
                 r["score"], r["status"], r.get("wall_s", 0.0))
    persist()

    # search loop: propose (mutations per §3), evaluate, add, persist — see
    # references/07_tree_search.md for the mutation vocabulary and stopping rules.
    while not hv3.should_stop(tree):
        proposal, parent_id, mutation = propose_next(tree)          # implement per §3
        nid = hv3.next_id(tree)
        t0 = time.time()
        try:
            if proposal.get("kind") == "blend":
                _w, s, _oofs, _warn = hv3.eval_blend_with_cost_guard(
                    ev.CACHE_DIR, proposal["members"], ev.metric, tree=tree)
                res = {"score": s, "status": "evaluated", "wall_s": time.time() - t0}
            else:
                res = ev.evaluate(proposal, node_id=nid, timeout_s=600)
        except Exception as e:  # noqa: BLE001 — a failed node is recorded, never hidden
            res = {"score": None, "status": "failed", "wall_s": time.time() - t0,
                   "error": str(e)}
        hv3.add_node(tree, parent_id, mutation, proposal, res["score"], res["status"],
                     res["wall_s"])
        persist()


def propose_next(tree):
    """Implement the §3 mutation vocabulary (param nudge, boundary push, seed bag,
    feature drop, blend of evaluated solos). No recorded configs — mutations derive from
    THIS tree's own nodes."""
    raise NotImplementedError("fill in per references/07_tree_search.md §3")


if __name__ == "__main__":
    main()
