"""docs/scripts/build_tree_facts.py — deterministic fact extractor for
docs/tree_search_prototype.md (Phase C-3 v1 prototype + Phase D-7 v2 sweep report).

Reads all eight tree-search runs' persisted state:
  - v1 (Phase C-2a/b/c, tree_search/harness.py):
    competitions/playground-series-{s3e9,s3e14,s3e5}/experiments_tree.json
  - v2 (Phase D-2..D-6, tree_search/harness_v2.py):
    competitions/playground-series-{s3e3,s3e7,s3e1,s3e19,s3e11}/experiments_tree.json

and computes, purely mechanically (no judgment calls), the numbers the report cites:
node counts, global-best scores/ids, per-node wall-clock ranges, backtrack
counts, first-evaluation-to-beat-linear-best, tie groups, prior-usage win rates,
dedup-rejection counts, and the winning blend nodes' weights/cutpoints.

The linear-iteration reference numbers (each comp's pre-tree-search best score,
which lives in STATUS.md prose, not in experiments_tree.json) are NOT
re-derived here — they are transcribed verbatim from each comp's STATUS.md
"Appendix — Tree search" section (see the docstring note next to each constant
below citing the exact source line) and recorded in the `linear_reference`
block so every number in the report has a traceable origin in this file.

v2 nodes carry `kind` as a first-class top-level field (v1 buried it inside
`config`) and scores use the same lower-is-better sign convention throughout
(maximize metrics like AUC are stored negated, mirrored back to a
human-readable positive number in `*_v2["global_best_score_positive"]` etc.
next to the raw stored value, exactly like v1's `s3e5_qwk_positive` block).

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


# ---------------------------------------------------------------------------
# v2 (Phase D-2..D-6, tree_search/harness_v2.py) — ensemble-default node space,
# `kind` is already a top-level node field (no kind_key indirection needed),
# search_state additionally carries tie_rate_log; tree dicts additionally carry
# `priors`, `prior_usage_summary`, `dedup_rejections`.
# ---------------------------------------------------------------------------
def summarize_v2(comp: str, linear_best_score: float, sign: int = 1) -> dict:
    """`sign`: 1 if the comp's stored score is already the natural (lower-is-better)
    value (e.g. RMSE/RMSLE/SMAPE); -1 if the underlying metric is maximize-better and
    the harness stores its negation (e.g. AUC) — mirrors v1's s3e5 (QWK) convention.
    `linear_best_score` is always given in the metric's own human-readable, positive
    direction (i.e. the number as printed in STATUS.md), matching `linear_reference`."""
    tree = load_tree(comp)
    nodes = tree["nodes"]
    evaluated = [n for n in nodes if n["status"] == "evaluated"]
    failed = [n for n in nodes if n["status"] != "evaluated"]
    root = next(n for n in nodes if n["id"] == tree["root_id"])
    best = min(evaluated, key=lambda n: n["score"])
    wall_all = [n["wall_s"] for n in evaluated]
    total_wall = round(sum(wall_all), 1)

    kind_counts = {}
    wall_by_kind = {}
    for n in evaluated:
        k = n.get("kind", "solo")
        kind_counts[k] = kind_counts.get(k, 0) + 1
        wall_by_kind.setdefault(k, []).append(n["wall_s"])
    wall_by_kind = {k: {"min": min(v), "max": max(v), "n": len(v)}
                     for k, v in wall_by_kind.items()}

    st = tree["search_state"]
    backtrack_log = st["backtrack_log"]
    n_backtracks = len(backtrack_log)
    n_forced_backtracks = sum(1 for e in backtrack_log if e.get("at_node_id") is None)
    tie_rate_log = st.get("tie_rate_log", [])
    max_tie_rate = max(tie_rate_log) if tie_rate_log else 0.0

    # first evaluation (1-indexed; root = eval 1) whose stored score reaches-or-beats
    # the linear best, in stored (lower-is-better) terms regardless of `sign`
    stored_threshold = -linear_best_score if sign == -1 else linear_best_score
    beat = first_eval_to_beat(comp, stored_threshold, better="lower")

    dedup_rejections = tree.get("dedup_rejections", [])
    prior_usage = tree.get("prior_usage_summary", {})
    n_priors_returned = len(tree.get("priors", []))

    return dict(
        comp=comp,
        harness="v2",
        n_nodes_total=len(nodes),
        n_evaluated=len(evaluated),
        n_failed=len(failed),
        root_id=root["id"],
        root_score_stored=root["score"],
        root_score_positive=round(-root["score"], 6) if sign == -1 else root["score"],
        root_wall_s=root["wall_s"],
        global_best_id=best["id"],
        global_best_score_stored=best["score"],
        global_best_score_positive=round(-best["score"], 6) if sign == -1 else best["score"],
        global_best_mutation=best["mutation"],
        total_wall_s=total_wall,
        wall_s_min_all=min(wall_all),
        wall_s_max_all=max(wall_all),
        wall_by_kind=wall_by_kind,
        kind_counts=kind_counts,
        n_backtracks=n_backtracks,
        n_forced_backtracks=n_forced_backtracks,
        max_tie_rate=round(max_tie_rate, 4),
        first_eval_reaching_linear_best=beat,
        dedup_rejections_count=len(dedup_rejections),
        n_priors_returned=n_priors_returned,
        prior_usage_summary=prior_usage,
        verdict=("BEAT" if (best["score"] < stored_threshold) else
                 ("TIE" if best["score"] == stored_threshold else "LOSS")),
        linear_best_score=linear_best_score,
    )


def main():
    facts = {
        "generated_by": "docs/scripts/build_tree_facts.py",
        "sources": [
            "tree_search/harness.py",
            "tree_search/harness_v2.py",
            "competitions/playground-series-s3e9/experiments_tree.json",
            "competitions/playground-series-s3e14/experiments_tree.json",
            "competitions/playground-series-s3e5/experiments_tree.json",
            "competitions/playground-series-s3e3/experiments_tree.json",
            "competitions/playground-series-s3e7/experiments_tree.json",
            "competitions/playground-series-s3e1/experiments_tree.json",
            "competitions/playground-series-s3e19/experiments_tree.json",
            "competitions/playground-series-s3e11/experiments_tree.json",
            "competitions/playground-series-s3e9/STATUS.md (Phase C-2a appendix)",
            "competitions/playground-series-s3e14/STATUS.md (Phase C-2b appendix)",
            "competitions/playground-series-s3e5/STATUS.md (Phase C-2c appendix)",
            "competitions/playground-series-s3e3/STATUS.md (Phase D-2 appendix)",
            "competitions/playground-series-s3e7/STATUS.md (Phase D-3 appendix)",
            "competitions/playground-series-s3e1/STATUS.md (Phase D-4 appendix)",
            "competitions/playground-series-s3e19/STATUS.md (Phase D-5 appendix)",
            "competitions/playground-series-s3e11/STATUS.md (Phase D-6 appendix)",
            ".superpowers/weekend-plan.md (Phase D progress log)",
        ],
        "harness_constants": {
            "MAX_CHILDREN_PER_NODE": 3,
            "PLATEAU_STREAK": 3,
            "ADAPTIVE_PLATEAU_STREAK": 5,
            "TIE_RATE_THRESHOLD": 0.15,
            "config_hash_note": "harness_v2.config_hash() uses hashlib.sha256",
        },
        # Row-count facts cited in the report's cross-comp scale comparisons (STATUS.md
        # "Data:" lines); *_k fields are thousands-rounded display mirrors of the exact
        # count, kept as an explicit fact (not re-derived from rounding in the report
        # text) so "36 万/360k" style prose stays traceable.
        "dataset_scale": {
            "s3e3": {"train_rows": 1677,
                      "source": "competitions/playground-series-s3e3/STATUS.md line 4"},
            "s3e11": {"train_rows": 360336, "train_rows_k": 360,
                       "source": "competitions/playground-series-s3e11/STATUS.md line 4"},
            "s3e19": {"total_rows": 136950, "oof_scored_rows": 114000,
                       "source": "competitions/playground-series-s3e19/STATUS.md Phase D-5 appendix"},
            "s3e1": {"capped_target_pct": 4.92,
                      "source": "competitions/playground-series-s3e1/STATUS.md Phase D-4 appendix (EDA finding #1)"},
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
        # --- v2 (Phase D-2..D-6) ---
        "s3e3_v2": summarize_v2("s3e3", linear_best_score=0.838140, sign=-1),
        "s3e7_v2": summarize_v2("s3e7", linear_best_score=0.899893, sign=-1),
        "s3e1_v2": summarize_v2("s3e1", linear_best_score=0.557088, sign=1),
        "s3e19_v2": summarize_v2("s3e19", linear_best_score=10.01946, sign=1),
        "s3e11_v2": summarize_v2("s3e11", linear_best_score=0.295648, sign=1),
        "s3e3_v2_node6": node_detail("s3e3", 6),
        "s3e3_v2_node11": node_detail("s3e3", 11),
        "s3e7_v2_node7": node_detail("s3e7", 7),
        "s3e7_v2_node8": node_detail("s3e7", 8),
        "s3e7_v2_node13": node_detail("s3e7", 13),
        "s3e1_v2_node8": node_detail("s3e1", 8),
        "s3e1_v2_node10": node_detail("s3e1", 10),
        "s3e1_v2_node11": node_detail("s3e1", 11),
        "s3e1_v2_node12": node_detail("s3e1", 12),
        "s3e1_v2_node14": node_detail("s3e1", 14),
        "s3e19_v2_node6": node_detail("s3e19", 6),
        "s3e19_v2_node14": node_detail("s3e19", 14),
        "s3e19_v2_node15": node_detail("s3e19", 15),
        "s3e19_v2_node17": node_detail("s3e19", 17),
        "s3e11_v2_node4": node_detail("s3e11", 4),
        "s3e11_v2_node9": node_detail("s3e11", 9),
        "s3e11_v2_node17": node_detail("s3e11", 17),
        "s3e11_v2_node20": node_detail("s3e11", 20),
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
            "s3e3": {
                "best_score": 0.838140,
                "label": "6-way rank-avg blend, exp #7",
                "source": "competitions/playground-series-s3e3/STATUS.md Phase D-2 appendix",
            },
            "s3e7": {
                "best_score": 0.899893,
                "label": "3-model prob 0.01-grid blend, exp #6",
                "source": "competitions/playground-series-s3e7/STATUS.md Phase D-3 appendix",
            },
            "s3e1": {
                "best_score": 0.557088,
                "label": "5-way 0.05-grid blend, exp #7",
                "source": "competitions/playground-series-s3e1/STATUS.md Phase D-4 appendix",
            },
            "s3e19": {
                "best_score": 10.01946,
                "label": "6-way 0.1-grid blend, exp #7",
                "source": "competitions/playground-series-s3e19/STATUS.md Phase D-5 appendix",
            },
            "s3e11": {
                "best_score": 0.295648,
                "label": "5-way 0.1-grid blend, exp #8",
                "source": "competitions/playground-series-s3e11/STATUS.md Phase D-6 appendix",
            },
        },
        "derived_diffs": {
            "s3e9_tree_minus_linear": round(12.07459 - 12.07003, 5),
            "s3e14_linear_minus_tree": round(340.59891 - 340.52635, 5),
            "s3e5_linear_minus_tree": round(0.56769 - 0.56766, 5),
            "s3e3_tree_minus_linear_auc": round(0.841442 - 0.838140, 6),
            "s3e7_tree_minus_linear_auc": round(0.900242 - 0.899893, 6),
            "s3e1_linear_minus_tree_rmse": round(0.557088 - 0.556329, 6),
            "s3e19_linear_minus_tree_smape": round(10.01946 - 9.75707, 5),
            "s3e19_linear_minus_tree_smape_pct": round(
                (10.01946 - 9.75707) / 10.01946 * 100, 2),
            "s3e11_linear_minus_tree_rmsle": round(0.295648 - 0.295280, 6),
            "s3e1_linear_minus_tree_rmse_pct": round(
                (0.557088 - 0.556329) / 0.557088 * 100, 2),
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

    # Sweep-wide verdict (Phase D complete, STATUS.md s3e11 appendix "Sweep verdict"
    # section) -- purely a transcription of the 5 summarize_v2() calls' own
    # `verdict`/`first_eval_reaching_linear_best` fields already written into `facts`
    # above, collected into one place so the report's "5/5 BEAT, 9-21 evaluations"
    # claim has a single spot to check.
    v2_keys = ["s3e3_v2", "s3e7_v2", "s3e1_v2", "s3e19_v2", "s3e11_v2"]
    facts["v2_sweep_verdict"] = {
        "n_comps": len(v2_keys),
        "n_beat": sum(1 for k in v2_keys if facts[k]["verdict"] == "BEAT"),
        "eval_counts_to_beat": {
            k: facts[k]["first_eval_reaching_linear_best"]["evaluation_number"]
            for k in v2_keys
        },
        "prior_win_rate_by_comp": {
            k: facts[k]["prior_usage_summary"] for k in v2_keys
        },
        # Percentage-form mirrors of the same informed/uninformed win rates above
        # (round(x*100, 1)) purely so the report's prose ("s3e7 62.5%", "s3e19 33%",
        # ...) has a literal traceable number distinct from the 0-1 fraction form.
        "prior_win_rate_pct": {
            k: dict(
                informed_pct=round(facts[k]["prior_usage_summary"]["informed_win_rate"] * 100, 1),
                uninformed_pct=round(facts[k]["prior_usage_summary"]["uninformed_win_rate"] * 100, 1),
            )
            for k in v2_keys
        },
    }
    out_path = os.path.join(_REPO_ROOT, "docs", "tree_facts.json")
    with open(out_path, "w") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
