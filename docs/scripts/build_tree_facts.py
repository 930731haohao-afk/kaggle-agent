"""docs/scripts/build_tree_facts.py — deterministic fact extractor for
docs/tree_search_prototype.md (Phase C-3 tree-search feasibility report).

Reads the three tree-search runs' persisted state
(competitions/playground-series-{s3e9,s3e14,s3e5}/experiments_tree.json) and
computes, purely mechanically (no judgment calls), the numbers the report cites:
node counts, global-best scores/ids, per-node wall-clock ranges, backtrack
counts, first-evaluation-to-beat-linear-best, tie groups, and the winning
blend nodes' weights/cutpoints.

The linear-iteration reference numbers (each comp's pre-tree-search best score,
which lives in STATUS.md prose, not in experiments_tree.json) are NOT
re-derived here — they are transcribed verbatim from each comp's STATUS.md
"Appendix — Tree search" section (see the docstring note next to each constant
below citing the exact source line) and recorded in the `linear_reference`
block so every number in the report has a traceable origin in this file.

Usage:
    uv run python3 docs/scripts/build_tree_facts.py
Writes docs/tree_facts.json.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))


def load_tree(comp: str) -> dict:
    path = os.path.join(_REPO_ROOT, "competitions", f"playground-series-{comp}",
                         "experiments_tree.json")
    with open(path) as f:
        return json.load(f)


def lineage_of(tree, node_id):
    nodes = {n["id"]: n for n in tree["nodes"]}
    root_id = tree["root_id"]
    if node_id == root_id:
        return root_id
    cur = node_id
    while nodes[cur]["parent_id"] != root_id:
        cur = nodes[cur]["parent_id"]
    return cur


def summarize(comp: str, kind_key=None) -> dict:
    tree = load_tree(comp)
    nodes = tree["nodes"]
    evaluated = [n for n in nodes if n["status"] == "evaluated"]
    failed = [n for n in nodes if n["status"] != "evaluated"]
    root = next(n for n in nodes if n["id"] == tree["root_id"])
    best = min(evaluated, key=lambda n: n["score"])
    wall_all = [n["wall_s"] for n in evaluated]
    total_wall = round(sum(wall_all), 1)

    # first-generation lineage count (children of root)
    n_lineages = sum(1 for n in nodes if n["parent_id"] == tree["root_id"])

    # node-kind split (solo vs blend), only meaningful for s3e14/s3e5
    kind_counts = {}
    if kind_key:
        for n in evaluated:
            k = n["config"].get(kind_key, "solo")
            kind_counts[k] = kind_counts.get(k, 0) + 1

    # wall_s range split by kind (to support "blend nodes cost <1s" claims)
    wall_by_kind = {}
    if kind_key:
        for n in evaluated:
            k = n["config"].get(kind_key, "solo")
            wall_by_kind.setdefault(k, []).append(n["wall_s"])
        wall_by_kind = {k: {"min": min(v), "max": max(v), "n": len(v)}
                        for k, v in wall_by_kind.items()}
    else:
        wall_by_kind = {"solo": {"min": min(wall_all), "max": max(wall_all), "n": len(wall_all)}}

    backtrack_log = tree["search_state"]["backtrack_log"]
    n_backtracks = len(backtrack_log)
    n_reopens = sum(1 for e in backtrack_log if e.get("plateaued_lineage") is None)

    # score-tie groups among evaluated non-root nodes (5-decimal match)
    from collections import defaultdict
    by_score = defaultdict(list)
    for n in evaluated:
        if n["id"] == tree["root_id"]:
            continue
        by_score[round(n["score"], 5)].append(n["id"])
    tie_groups = {str(s): ids for s, ids in by_score.items() if len(ids) >= 2}

    return dict(
        comp=comp,
        n_nodes_total=len(nodes),
        n_evaluated=len(evaluated),
        n_failed=len(failed),
        n_first_gen_lineages=n_lineages,
        root_id=root["id"],
        root_score=root["score"],
        root_wall_s=root["wall_s"],
        global_best_id=best["id"],
        global_best_score=best["score"],
        global_best_mutation=best["mutation"],
        total_wall_s=total_wall,
        wall_s_min_all=min(wall_all),
        wall_s_max_all=max(wall_all),
        wall_by_kind=wall_by_kind,
        kind_counts=kind_counts,
        n_backtracks=n_backtracks,
        n_reopen_events=n_reopens,
        backtrack_log=backtrack_log,
        tie_groups_5dp=tie_groups,
    )


def first_eval_to_beat(comp: str, threshold: float, better="lower") -> dict:
    """Node id (== evaluation index, 0-based; eval number = id+1 since root is
    eval 1) of the first evaluated node whose score reaches-or-beats `threshold`."""
    tree = load_tree(comp)
    for n in tree["nodes"]:
        if n["status"] != "evaluated":
            continue
        ok = (n["score"] <= threshold) if better == "lower" else (n["score"] >= threshold)
        if ok:
            return dict(node_id=n["id"], evaluation_number=n["id"] + 1, score=n["score"])
    return None


def node_detail(comp: str, node_id: int) -> dict:
    tree = load_tree(comp)
    n = next(x for x in tree["nodes"] if x["id"] == node_id)
    return dict(id=n["id"], score=n["score"], wall_s=n["wall_s"],
                config=n["config"], mutation=n["mutation"])


def main():
    facts = {
        "generated_by": "docs/scripts/build_tree_facts.py",
        "sources": [
            "tree_search/harness.py",
            "competitions/playground-series-s3e9/experiments_tree.json",
            "competitions/playground-series-s3e14/experiments_tree.json",
            "competitions/playground-series-s3e5/experiments_tree.json",
            "competitions/playground-series-s3e9/STATUS.md (Phase C-2a appendix)",
            "competitions/playground-series-s3e14/STATUS.md (Phase C-2b appendix)",
            "competitions/playground-series-s3e5/STATUS.md (Phase C-2c appendix)",
        ],
        "harness_constants": {
            "MAX_CHILDREN_PER_NODE": 3,
            "PLATEAU_STREAK": 3,
        },
        "s3e9": summarize("s3e9"),
        "s3e14": summarize("s3e14", kind_key="kind"),
        "s3e5": summarize("s3e5", kind_key="kind"),
        "s3e14_beat_linear_first_eval": first_eval_to_beat("s3e14", 340.59891, "lower"),
        "s3e14_node11": node_detail("s3e14", 11),
        "s3e14_node10_sanity_anchor": node_detail("s3e14", 10),
        "s3e14_node8": node_detail("s3e14", 8),
        "s3e14_node5": node_detail("s3e14", 5),
        "s3e14_node9": node_detail("s3e14", 9),
        "s3e5_node11": node_detail("s3e5", 11),
        "s3e9_node1": node_detail("s3e9", 1),
        # Linear-iteration reference scores — transcribed verbatim from each
        # comp's STATUS.md "Tree search" appendix (see `sources` above); these
        # numbers do NOT live in experiments_tree.json because linear iteration
        # predates and is independent of the tree-search harness.
        "linear_reference": {
            "s3e9": {
                "best_score": 12.07003,
                "label": "7-way seed-bagged blend, exp #8",
                "source": "competitions/playground-series-s3e9/STATUS.md line ~225",
            },
            "s3e14": {
                "best_score": 340.59891,
                "label": "5-way blend, exp #7",
                "source": "competitions/playground-series-s3e14/STATUS.md line ~136",
            },
            "s3e5": {
                "best_score": 0.56769,
                "label": "6-way blend, exp #8",
                "source": "competitions/playground-series-s3e5/STATUS.md line ~177",
            },
        },
        "derived_diffs": {
            "s3e9_tree_minus_linear": round(12.07459 - 12.07003, 5),
            "s3e14_linear_minus_tree": round(340.59891 - 340.52635, 5),
            "s3e5_linear_minus_tree": round(0.56769 - 0.56766, 5),
        },
        # s3e5's score sign convention is -QWK internally (harness.py assumes
        # lower-is-better); these are the positive, human-readable QWK mirrors
        # of the same numbers already present above (eval_s3e5.py's own
        # convention: result["qwk"] is always reported positive).
        "s3e5_qwk_positive": {
            "root_qwk": round(-(-0.56244), 5),
            "global_best_qwk": round(-(-0.56766), 5),
            "tie_group_qwks": sorted(round(-float(k), 5)
                                      for k in summarize("s3e5")["tie_groups_5dp"]),
        },
    }
    out_path = os.path.join(_REPO_ROOT, "docs", "tree_facts.json")
    with open(out_path, "w") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
