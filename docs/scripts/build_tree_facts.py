"""docs/scripts/build_tree_facts.py — deterministic fact extractor for
docs/tree_search_prototype.md (Phase C-3 v1 -> Phase D-7 v2 -> Phase F-3 v1+v2+v3+scale
FINAL report).

Reads every tree-search run's persisted state that exists on disk as of Phase F-3.
Counted precisely by `find competitions -iname "experiments_tree*.json"` (see the
Phase F-3 task brief, which estimated 12 and was corrected here by direct disk count):
**15 tree files across 10 competitions** —

  - v1 (Phase C-2a/b/c, tree_search/harness.py), 3 files:
    competitions/playground-series-{s3e9,s3e14,s3e5}/experiments_tree.json
  - v2 (Phase D-2..D-6 + E-1..E-4, tree_search/harness_v2.py), 9 files:
    competitions/playground-series-{s3e3,s3e7,s3e1,s3e19,s3e11}/experiments_tree.json (D-2..D-6)
    competitions/playground-series-s3e9/experiments_tree_v2.json (E-1, revenge match vs its own v1 loss)
    competitions/playground-series-s3e5/experiments_tree_v2.json (E-2, revenge match vs its own v1 tie)
    competitions/playground-series-s3e16/experiments_tree.json (E-3, rounded-MAE acid test)
    competitions/playground-series-s3e20/experiments_tree.json (E-4, structure-dominated regime)
  - scale (Phase E-5, tree_search/harness_v2.py, 80-node budget extension of D-2's s3e3 run), 1 file:
    competitions/playground-series-s3e3/experiments_tree_scale.json
  - v3 (Phase F-2, tree_search/harness_v3.py), 2 files:
    competitions/playground-series-{s3e7,s3e14}/experiments_tree_v3.json

and computes, purely mechanically (no judgment calls), the numbers the report cites:
node counts, global-best scores/ids, per-node wall-clock ranges, backtrack
counts, first-evaluation-to-beat-reference, tie groups, prior-usage win rates,
dedup-rejection counts, curve-based improvement/idle-tail statistics (scale run),
and budget-efficiency ratios (best-found-at-eval / n_evaluated) recomputed
identically across all 15 runs for the cross-run calibration table Phase E-5's
docs/scaling_experiment.md first introduced for 10 runs — here regenerated
mechanically for all 15 so every number in that comparison also has a
from-this-script origin, not just a copy-paste from scaling_experiment.md.

The linear-iteration (or, for s3e9/s3e5's v2 revenge matches and s3e3's scale
extension, the PRIOR tree's own best) reference numbers are NOT re-derived here —
they are transcribed verbatim from each comp's STATUS.md "Appendix — Tree search"
section (see the docstring note next to each constant below citing the exact
source line) and recorded in the `linear_reference` block so every number in the
report has a traceable origin in this file.

v2/v3 nodes carry `kind` as a first-class top-level field (v1 buried it inside
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


def load_tree(comp: str, filename: str = "experiments_tree.json") -> dict:
    """`filename` lets Phase F-3 point at the non-default tree files that coexist
    with a comp's original `experiments_tree.json` (`experiments_tree_v2.json`,
    `experiments_tree_v3.json`, `experiments_tree_scale.json`) without disturbing
    any existing call site (default unchanged)."""
    path = os.path.join(_REPO_ROOT, "competitions", f"playground-series-{comp}",
                         filename)
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


def summarize(comp: str, kind_key=None, filename: str = "experiments_tree.json") -> dict:
    tree = load_tree(comp, filename)
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


def first_eval_to_beat(comp: str, threshold: float, better="lower",
                        filename: str = "experiments_tree.json") -> dict:
    """Node id (== evaluation index, 0-based; eval number = id+1 since root is
    eval 1) of the first evaluated node whose score reaches-or-beats `threshold`."""
    tree = load_tree(comp, filename)
    for n in tree["nodes"]:
        if n["status"] != "evaluated":
            continue
        ok = (n["score"] <= threshold) if better == "lower" else (n["score"] >= threshold)
        if ok:
            return dict(node_id=n["id"], evaluation_number=n["id"] + 1, score=n["score"])
    return None


def node_detail(comp: str, node_id: int, filename: str = "experiments_tree.json") -> dict:
    tree = load_tree(comp, filename)
    n = next(x for x in tree["nodes"] if x["id"] == node_id)
    return dict(id=n["id"], score=n["score"], wall_s=n["wall_s"],
                config=n["config"], mutation=n["mutation"])


def eval_rank(tree: dict, node_id: int) -> int:
    """1-indexed rank of `node_id` among all `status == "evaluated"` nodes with
    id <= node_id — i.e. "this was the Nth evaluation actually run", correctly
    skipping over any `status == "failed"` node (crashed solo fit, dedup-burned
    placeholder, ValueError-caught config, ...) that consumed an id but not an
    evaluation slot. Equivalent, for runs with zero failures, to the older
    id+1 convention used by `first_eval_to_beat`/the v1-v2 report sections; for
    runs with interleaved failures (s3e1, s3e5_v2, s3e16, s3e3_scale, s3e7_v3,
    s3e14_v3) this is the mechanically correct generalization (matches, by
    construction, `experiments_tree_scale.json`'s own `curve[*]["eval_index"]`,
    which is appended once per evaluated node in id order)."""
    return sum(1 for n in tree["nodes"] if n["id"] <= node_id and n["status"] == "evaluated")


def budget_efficiency(comp: str, filename: str = "experiments_tree.json") -> dict:
    """Purely structural (sign/metric-agnostic) cross-run comparability numbers:
    where in the run (by evaluation rank, `eval_rank`) was the global best found,
    relative to the total number of evaluations spent. Recomputes, per tree, the
    same `best/total ratio` and `idle tail` columns docs/scaling_experiment.md's
    "Cross-run calibration" table introduced for 10 runs — done here for all 15
    so the comparison has a from-this-script origin instead of being copied from
    that document's own (independently, but identically, computed) table."""
    tree = load_tree(comp, filename)
    evaluated = [n for n in tree["nodes"] if n["status"] == "evaluated"]
    n_evaluated = len(evaluated)
    best = min(evaluated, key=lambda n: n["score"])
    best_rank = eval_rank(tree, best["id"])
    return dict(
        comp=comp, filename=filename, n_evaluated=n_evaluated,
        best_node_id=best["id"], best_found_at_eval=best_rank,
        best_total_ratio=round(best_rank / n_evaluated, 2),
        idle_tail_evals=n_evaluated - best_rank,
    )


def curve_summary(comp: str, filename: str = "experiments_tree_scale.json") -> dict:
    """Reads `tree["curve"]` (Phase E-5's per-evaluation log: eval_index, node_id,
    lineage, phase, auc, cum_best_auc, is_new_global_best, wall_s, cum_wall_s) and
    extracts, mechanically, every number docs/scaling_experiment.md's curve-based
    prose cites: the full improvement sequence, the exploit/explore phase-boundary
    eval, per-phase improvement counts, the post-explore-burst gain over the
    pre-burst ceiling, and the idle tail after the final improvement."""
    tree = load_tree(comp, filename)
    curve = tree.get("curve", [])
    total_evals = len(curve)
    improvements = [c for c in curve if c.get("is_new_global_best")]
    explore_start = next((c["eval_index"] for c in curve if c["phase"] == "explore"), None)
    pre = [c for c in improvements if explore_start is None or c["eval_index"] < explore_start]
    post = [c for c in improvements if explore_start is not None and c["eval_index"] >= explore_start]
    final_best = improvements[-1]["auc"] if improvements else None
    final_best_eval = improvements[-1]["eval_index"] if improvements else None
    return dict(
        comp=comp,
        total_evals=total_evals,
        n_improvements=len(improvements),
        improvements=[dict(eval_index=c["eval_index"], node_id=c["node_id"],
                            lineage=c["lineage"], phase=c["phase"], score=c["auc"])
                      for c in improvements],
        explore_burst_start_eval=explore_start,
        n_improvements_pre_explore=len(pre),
        n_improvements_post_explore=len(post),
        pre_explore_ceiling=pre[-1]["auc"] if pre else None,
        post_explore_gain=(round(final_best - pre[-1]["auc"], 6)
                            if pre and post else None),
        final_best=final_best,
        final_best_eval=final_best_eval,
        idle_tail_evals=(total_evals - final_best_eval) if final_best_eval else None,
        total_wall_s=round(curve[-1]["cum_wall_s"], 1) if curve else None,
    )


# ---------------------------------------------------------------------------
# v2 (Phase D-2..D-6, tree_search/harness_v2.py) — ensemble-default node space,
# `kind` is already a top-level node field (no kind_key indirection needed),
# search_state additionally carries tie_rate_log; tree dicts additionally carry
# `priors`, `prior_usage_summary`, `dedup_rejections`.
# ---------------------------------------------------------------------------
def summarize_v2(comp: str, linear_best_score: float, sign: int = 1,
                  filename: str = "experiments_tree.json") -> dict:
    """`sign`: 1 if the comp's stored score is already the natural (lower-is-better)
    value (e.g. RMSE/RMSLE/SMAPE); -1 if the underlying metric is maximize-better and
    the harness stores its negation (e.g. AUC) — mirrors v1's s3e5 (QWK) convention.
    `linear_best_score` is always given in the metric's own human-readable, positive
    direction (i.e. the number as printed in STATUS.md), matching `linear_reference`.
    `filename` (Phase F-3 addition): which tree file to read — lets this same
    function summarize `experiments_tree_v2.json`/`experiments_tree_v3.json`/
    `experiments_tree_scale.json` runs that coexist with a comp's original
    `experiments_tree.json`, and `linear_best_score` in those calls is whichever
    PRIOR result (linear iteration, or an earlier tree version) the run's own
    STATUS.md appendix names as its comparison target."""
    tree = load_tree(comp, filename)
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
    beat = first_eval_to_beat(comp, stored_threshold, better="lower", filename=filename)

    dedup_rejections = tree.get("dedup_rejections", [])
    prior_usage = tree.get("prior_usage_summary", {})
    n_priors_returned = len(tree.get("priors", []))

    return dict(
        comp=comp,
        harness="v2",
        source_file=filename,
        n_nodes_total=len(nodes),
        n_evaluated=len(evaluated),
        n_failed=len(failed),
        root_id=root["id"],
        root_score_stored=root["score"],
        root_score_positive=round(-root["score"], 6) if sign == -1 else root["score"],
        root_wall_s=root["wall_s"],
        global_best_id=best["id"],
        global_best_eval_rank=eval_rank(tree, best["id"]),
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
            # --- Phase F-3 additions: E-1..E-5 + F-1/F-2 sources ---
            "tree_search/harness_v3.py",
            "competitions/playground-series-s3e9/experiments_tree_v2.json",
            "competitions/playground-series-s3e5/experiments_tree_v2.json",
            "competitions/playground-series-s3e16/experiments_tree.json",
            "competitions/playground-series-s3e20/experiments_tree.json",
            "competitions/playground-series-s3e3/experiments_tree_scale.json",
            "competitions/playground-series-s3e7/experiments_tree_v3.json",
            "competitions/playground-series-s3e14/experiments_tree_v3.json",
            "competitions/playground-series-s3e9/STATUS.md (Phase E-1 appendix)",
            "competitions/playground-series-s3e5/STATUS.md (Phase E-2 appendix)",
            "competitions/playground-series-s3e16/STATUS.md (Phase E-3 appendix)",
            "competitions/playground-series-s3e20/STATUS.md (Phase E-4 appendix)",
            "competitions/playground-series-s3e3/STATUS.md (Phase D-2 + E-5 appendices)",
            "competitions/playground-series-s3e7/STATUS.md (Phase F-2 appendix)",
            "competitions/playground-series-s3e14/STATUS.md (Phase F-2 appendix)",
            "docs/scaling_experiment.md (Phase E-5)",
            ".superpowers/weekend-plan.md (Phase E/F progress log)",
        ],
        "harness_constants": {
            "MAX_CHILDREN_PER_NODE": 3,
            "PLATEAU_STREAK": 3,
            "ADAPTIVE_PLATEAU_STREAK": 5,
            "TIE_RATE_THRESHOLD": 0.15,
            "config_hash_note": "harness_v2.config_hash() uses hashlib.sha256",
        },
        # v3 (tree_search/harness_v3.py) module-level defaults — transcribed directly
        # from the constants defined at that file's top level (not re-derived), so the
        # report's "budget 60 / burst 5-8 / patience 20 / cost-guard 45s / k=800" prose
        # has a single traceable origin distinct from any per-run tree JSON.
        "v3_constants": {
            "DEFAULT_TOTAL_BUDGET": 60,
            "EXPLORE_BURST_MIN": 5,
            "EXPLORE_BURST_MAX": 8,
            "DEFAULT_EXPLORE_BURST_SIZE": 6,
            "DEFAULT_POST_BURST_PATIENCE": 20,
            "DEFAULT_BLEND_K": 800,
            "DEFAULT_ASCENT_ROUNDS": 6,
            "DEFAULT_COST_GUARD_THRESHOLD_S": 45.0,
            "DEFAULT_COST_GUARD_COARSEN_K": 200,
            "boundary_push_edge_frac": 0.05,
            "boundary_push_push_factor": 1.5,
            "source": "tree_search/harness_v3.py module-level constants",
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
        # --- Phase E: v2 revenge matches (s3e9/s3e5) + acid tests (s3e16/s3e20) ---
        # s3e9 E-1: linear_best_score given at FULL precision (12.070034, not the
        # 5dp-rounded 12.07003 used for the v1 comparison above) because STATUS.md's
        # own "full precision" row is what v2 is checked against for an exact-tie
        # verdict -- see the appendix's "identical to 6 decimal places" claim.
        "s3e9_v2": summarize_v2("s3e9", linear_best_score=12.070034, sign=1,
                                 filename="experiments_tree_v2.json"),
        "s3e5_v2": summarize_v2("s3e5", linear_best_score=0.56769, sign=-1,
                                 filename="experiments_tree_v2.json"),
        "s3e16_v2": summarize_v2("s3e16", linear_best_score=1.33812, sign=1),
        "s3e20_v2": summarize_v2("s3e20", linear_best_score=21.1487, sign=1),
        "s3e9_v2_node13": node_detail("s3e9", 13, filename="experiments_tree_v2.json"),
        "s3e5_v2_node7_lgbbound": node_detail("s3e5", 7, filename="experiments_tree_v2.json"),
        "s3e5_v2_node17": node_detail("s3e5", 17, filename="experiments_tree_v2.json"),
        "s3e16_v2_node9_weightsearch_only": node_detail("s3e16", 9),
        "s3e16_v2_node15": node_detail("s3e16", 15),
        "s3e20_v2_node0_root": node_detail("s3e20", 0),
        "s3e20_v2_node28_structural_best": node_detail("s3e20", 28),
        "s3e20_v2_node37_blend_best": node_detail("s3e20", 37),
        # s3e16's Public/Private LB numbers -- the ONE tree-search comp with a real
        # Kaggle anchor (all others are OOF-only, see report's honest-caveats section).
        "s3e16_lb_anchor": {
            "public_lb": 1.34356, "private_lb": 1.34075, "oof_rounded_mae": 1.33812,
            "cv_lb_gap": round(1.34356 - 1.33812, 5),
            "source": "competitions/playground-series-s3e16/STATUS.md 'Leaderboard result (submission #1)'",
        },
        # The raw-vs-rounded MAE inversion (E-3's headline finding): the linear
        # champion's raw OOF MAE vs the v2 tree's winning node's raw OOF MAE --
        # transcribed verbatim (these numbers don't live in experiments_tree.json,
        # only the ROUNDED score used for every node's stored `score` does).
        "s3e16_raw_vs_rounded_inversion": {
            "linear_champion_raw_mae": 1.35589, "linear_champion_rounded_mae": 1.33812,
            "tree_v2_best_raw_mae": 1.35712, "tree_v2_best_rounded_mae": 1.33563,
            "source": "competitions/playground-series-s3e16/STATUS.md Phase E-3 appendix",
        },
        # s3e20's GBDT-blend weight (the low-confidence marginal finding, honest
        # caveat #3 in the report) -- transcribed, not derivable from the tree JSON's
        # own weight vector without loading the blend node's config in full.
        "s3e20_gbdt_blend_weight_pct": {
            "value": 2.17,
            "source": "competitions/playground-series-s3e20/STATUS.md Phase E-4 appendix "
                      "(BLEND row: 'GBDT 獲得極小非零權重 2.17%')",
        },
        # --- Phase E-5: scale extension (s3e3, 80-evaluated-node budget) ---
        "s3e3_scale": summarize_v2("s3e3", linear_best_score=0.838140, sign=-1,
                                    filename="experiments_tree_scale.json"),
        "s3e3_scale_curve": curve_summary("s3e3", filename="experiments_tree_scale.json"),
        "s3e3_scale_node38_pre_explore_ceiling": node_detail(
            "s3e3", 38, filename="experiments_tree_scale.json"),
        "s3e3_scale_node55_global_best": node_detail(
            "s3e3", 55, filename="experiments_tree_scale.json"),
        # --- Phase F-2: v3 validation runs (s3e7, s3e14) ---
        "s3e7_v3": summarize_v2("s3e7", linear_best_score=0.899893, sign=-1,
                                 filename="experiments_tree_v3.json"),
        "s3e14_v3": summarize_v2("s3e14", linear_best_score=340.59891, sign=1,
                                  filename="experiments_tree_v3.json"),
        "s3e7_v3_node10_preburst_ceiling": node_detail(
            "s3e7", 10, filename="experiments_tree_v3.json"),
        "s3e7_v3_node48_global_best": node_detail(
            "s3e7", 48, filename="experiments_tree_v3.json"),
        "s3e14_v3_node15_preburst_ceiling": node_detail(
            "s3e14", 15, filename="experiments_tree_v3.json"),
        "s3e14_v3_node44_global_best": node_detail(
            "s3e14", 44, filename="experiments_tree_v3.json"),
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
            "s3e16": {
                "best_score": 1.33812,
                "label": "Phase-B blend 0.6/0.3/0.1 LGB/XGB/CAT, rounded MAE, real Kaggle LB anchor",
                "source": "competitions/playground-series-s3e16/STATUS.md Phase E-3 appendix",
            },
            "s3e20": {
                "best_score": 21.1487,
                "label": "Phase-B v5 tuned structural baseline (EB shrinkage + W2020 + WNB + window)",
                "source": "competitions/playground-series-s3e20/STATUS.md Phase E-4 appendix",
            },
        },
        # Prior-TREE reference scores (distinct from `linear_reference` above): for the
        # 5 runs that are a second-or-third pass at a comp ALREADY tree-searched once,
        # this is the earlier tree's own best -- the number that run's own STATUS.md
        # appendix names as its comparison target, transcribed the same way as
        # `linear_reference`.
        "prior_tree_reference": {
            "s3e9_v2_vs_v1_tree": {
                "prior_best_score": 12.07459,
                "label": "v1 (harness.py, Phase C-2a) single-model-only tree best",
                "source": "competitions/playground-series-s3e9/STATUS.md Phase E-1 appendix",
            },
            "s3e5_v2_vs_v1_tree": {
                "prior_best_score": 0.56766,
                "label": "v1 (harness.py, Phase C-2c) node #11 4-way blend",
                "source": "competitions/playground-series-s3e5/STATUS.md Phase E-2 appendix",
            },
            "s3e3_scale_vs_v2_tree": {
                "prior_best_score": 0.841442,
                "label": "v2 (harness_v2.py, Phase D-2) 22-node sweep best",
                "source": "competitions/playground-series-s3e3/STATUS.md Phase D-2 appendix / docs/scaling_experiment.md",
            },
            "s3e7_v3_vs_v2_tree": {
                "prior_best_score": 0.900242,
                "label": "v2 (harness_v2.py, Phase D-3) 22-node sweep best",
                "source": "competitions/playground-series-s3e7/STATUS.md Phase F-2 appendix",
            },
            "s3e14_v3_vs_v1_tree": {
                "prior_best_score": 340.52635,
                "label": "v1-proto (harness.py, Phase C-2b) 20-node tree best",
                "source": "competitions/playground-series-s3e14/STATUS.md Phase F-2 appendix",
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
            # --- Phase F-3 additions ---
            "s3e9_v2_minus_v1_tree": round(12.07459 - 12.070034, 6),
            "s3e9_v2_minus_linear_full_precision": round(12.070034 - 12.070034, 6),
            "s3e5_v2_minus_v1_tree": round(0.57066 - 0.56766, 5),
            "s3e5_v2_minus_linear": round(0.57066 - 0.56769, 5),
            "s3e16_linear_minus_tree_mae": round(1.33812 - 1.33563, 5),
            "s3e16_linear_minus_tree_mae_pct": round((1.33812 - 1.33563) / 1.33812 * 100, 3),
            "s3e20_baseline_minus_structural": round(21.1487 - 21.0589, 4),
            "s3e20_baseline_minus_structural_pct": round((21.1487 - 21.0589) / 21.1487 * 100, 3),
            "s3e20_baseline_minus_blend": round(21.1487 - 21.0332, 4),
            "s3e20_baseline_minus_blend_pct": round((21.1487 - 21.0332) / 21.1487 * 100, 3),
            "s3e3_scale_minus_v2_tree": round(0.845051 - 0.841442, 6),
            "s3e3_scale_minus_linear": round(0.845051 - 0.838140, 6),
            "s3e7_v3_minus_v2_tree": round(0.900455 - 0.900242, 6),
            "s3e7_v3_minus_linear": round(0.900455 - 0.899893, 6),
            "s3e14_v3_minus_v1_tree": round(340.52635 - 340.35572, 5),
            "s3e14_v3_minus_linear": round(340.59891 - 340.35572, 5),
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
        # Phase F-2's two honest operational-ledger items (driver-level bugs/overruns
        # hit while validating harness_v3 in the wild) -- transcribed verbatim from
        # each comp's Phase F-2 STATUS.md appendix "Honest operational ledger" section,
        # since these are prose facts (wall-clock overshoots, bug counts) that don't
        # live in the tree JSON's own numeric fields.
        "f2_operational_ledger": {
            "s3e7_v3": {
                "n_driver_restarts": 3,
                "hung_eval_wall_min": 28,
                "hung_eval_retry_wall_s": 54.8,
                "sum_eval_wall_s": 1593,
                "sum_eval_wall_min": 26.6,
                "note": "3 driver-level bugs (blend-vs-solo dispatch on literal name, "
                        "resume state not surviving restart, SIGALRM cannot interrupt "
                        "a native fit()); harness_v3.py itself needed zero changes.",
                "source": "competitions/playground-series-s3e7/STATUS.md Phase F-2 appendix",
            },
            "s3e14_v3": {
                "sum_eval_wall_s": 2338,
                "sum_eval_wall_min": 39,
                "wall_guard_trip_at_eval": 59,
                "dart_eval_count": 6,
                "dart_worst_oof_mae": 6544,
                "note": "process hit the driver's own 35-min wall guard at 59/60; "
                        "DART evals (107-140s each, 6144-6544 OOF MAE) burned ~12 min "
                        "for zero value.",
                "source": "competitions/playground-series-s3e14/STATUS.md Phase F-2 appendix",
            },
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
    # -----------------------------------------------------------------------
    # Phase F-3: all-15-runs master table, budget-efficiency, and the two
    # cross-run mechanism findings (explore-burst mega-blend, boundary-push).
    # Every field below is either pulled directly from an already-computed
    # `facts[...]` entry above, or (for v1's 3 runs, which predate `summarize_v2`
    # and its `verdict` field) the SAME verdict already published in
    # docs/tree_search_prototype.md section 2/6 (s3e9 LOSS, s3e14 WIN, s3e5 TIE
    # -- the s3e5 "TIE" is the one documented judgment call in this whole file:
    # 0.56766 is arithmetically 0.00003 better than 0.56769, but section 3's own
    # prose calls it a tie because the gap is inside single-sample-cutpoint noise;
    # carried forward here unchanged, not re-derived).
    # -----------------------------------------------------------------------
    facts["all_15_runs"] = [
        dict(comp="s3e9", harness="v1", phase="C-2a", reference_label="linear",
             reference_score=facts["linear_reference"]["s3e9"]["best_score"],
             tree_best=facts["s3e9"]["global_best_score"], verdict="LOSS",
             n_nodes_total=facts["s3e9"]["n_nodes_total"],
             n_evaluated=facts["s3e9"]["n_evaluated"]),
        dict(comp="s3e14", harness="v1", phase="C-2b", reference_label="linear",
             reference_score=facts["linear_reference"]["s3e14"]["best_score"],
             tree_best=facts["s3e14"]["global_best_score"], verdict="WIN",
             n_nodes_total=facts["s3e14"]["n_nodes_total"],
             n_evaluated=facts["s3e14"]["n_evaluated"]),
        dict(comp="s3e5", harness="v1", phase="C-2c", reference_label="linear",
             reference_score=facts["linear_reference"]["s3e5"]["best_score"],
             tree_best=facts["s3e5_qwk_positive"]["global_best_qwk"], verdict="TIE",
             n_nodes_total=facts["s3e5"]["n_nodes_total"],
             n_evaluated=facts["s3e5"]["n_evaluated"]),
        dict(comp="s3e3", harness="v2", phase="D-2", reference_label="linear",
             reference_score=facts["linear_reference"]["s3e3"]["best_score"],
             tree_best=facts["s3e3_v2"]["global_best_score_positive"], verdict=facts["s3e3_v2"]["verdict"],
             n_nodes_total=facts["s3e3_v2"]["n_nodes_total"], n_evaluated=facts["s3e3_v2"]["n_evaluated"]),
        dict(comp="s3e7", harness="v2", phase="D-3", reference_label="linear",
             reference_score=facts["linear_reference"]["s3e7"]["best_score"],
             tree_best=facts["s3e7_v2"]["global_best_score_positive"], verdict=facts["s3e7_v2"]["verdict"],
             n_nodes_total=facts["s3e7_v2"]["n_nodes_total"], n_evaluated=facts["s3e7_v2"]["n_evaluated"]),
        dict(comp="s3e1", harness="v2", phase="D-4", reference_label="linear",
             reference_score=facts["linear_reference"]["s3e1"]["best_score"],
             tree_best=facts["s3e1_v2"]["global_best_score_positive"], verdict=facts["s3e1_v2"]["verdict"],
             n_nodes_total=facts["s3e1_v2"]["n_nodes_total"], n_evaluated=facts["s3e1_v2"]["n_evaluated"]),
        dict(comp="s3e19", harness="v2", phase="D-5", reference_label="linear",
             reference_score=facts["linear_reference"]["s3e19"]["best_score"],
             tree_best=facts["s3e19_v2"]["global_best_score_positive"], verdict=facts["s3e19_v2"]["verdict"],
             n_nodes_total=facts["s3e19_v2"]["n_nodes_total"], n_evaluated=facts["s3e19_v2"]["n_evaluated"]),
        dict(comp="s3e11", harness="v2", phase="D-6", reference_label="linear",
             reference_score=facts["linear_reference"]["s3e11"]["best_score"],
             tree_best=facts["s3e11_v2"]["global_best_score_positive"], verdict=facts["s3e11_v2"]["verdict"],
             n_nodes_total=facts["s3e11_v2"]["n_nodes_total"], n_evaluated=facts["s3e11_v2"]["n_evaluated"]),
        dict(comp="s3e9", harness="v2", phase="E-1 (revenge vs v1 tree)", reference_label="v1 tree + linear (full precision)",
             reference_score=facts["prior_tree_reference"]["s3e9_v2_vs_v1_tree"]["prior_best_score"],
             tree_best=facts["s3e9_v2"]["global_best_score_positive"], verdict="BEAT v1 tree / TIE linear",
             n_nodes_total=facts["s3e9_v2"]["n_nodes_total"], n_evaluated=facts["s3e9_v2"]["n_evaluated"]),
        dict(comp="s3e5", harness="v2", phase="E-2 (revenge vs v1 tie)", reference_label="v1 tree + linear",
             reference_score=facts["prior_tree_reference"]["s3e5_v2_vs_v1_tree"]["prior_best_score"],
             tree_best=facts["s3e5_v2"]["global_best_score_positive"], verdict=facts["s3e5_v2"]["verdict"],
             n_nodes_total=facts["s3e5_v2"]["n_nodes_total"], n_evaluated=facts["s3e5_v2"]["n_evaluated"]),
        dict(comp="s3e16", harness="v2", phase="E-3 (rounded-MAE acid test)", reference_label="linear (LB-anchored)",
             reference_score=facts["linear_reference"]["s3e16"]["best_score"],
             tree_best=facts["s3e16_v2"]["global_best_score_positive"], verdict=facts["s3e16_v2"]["verdict"],
             n_nodes_total=facts["s3e16_v2"]["n_nodes_total"], n_evaluated=facts["s3e16_v2"]["n_evaluated"]),
        dict(comp="s3e20", harness="v2", phase="E-4 (structure-dominated)", reference_label="Phase-B baseline",
             reference_score=facts["linear_reference"]["s3e20"]["best_score"],
             tree_best=facts["s3e20_v2"]["global_best_score_positive"], verdict=facts["s3e20_v2"]["verdict"],
             n_nodes_total=facts["s3e20_v2"]["n_nodes_total"], n_evaluated=facts["s3e20_v2"]["n_evaluated"]),
        dict(comp="s3e3", harness="v2-scale", phase="E-5 (80-node budget curve)", reference_label="v2 tree + linear",
             reference_score=facts["prior_tree_reference"]["s3e3_scale_vs_v2_tree"]["prior_best_score"],
             tree_best=facts["s3e3_scale"]["global_best_score_positive"], verdict=facts["s3e3_scale"]["verdict"],
             n_nodes_total=facts["s3e3_scale"]["n_nodes_total"], n_evaluated=facts["s3e3_scale"]["n_evaluated"]),
        dict(comp="s3e7", harness="v3", phase="F-2 (validation run)", reference_label="v2 tree + linear",
             reference_score=facts["prior_tree_reference"]["s3e7_v3_vs_v2_tree"]["prior_best_score"],
             tree_best=facts["s3e7_v3"]["global_best_score_positive"], verdict=facts["s3e7_v3"]["verdict"],
             n_nodes_total=facts["s3e7_v3"]["n_nodes_total"], n_evaluated=facts["s3e7_v3"]["n_evaluated"]),
        dict(comp="s3e14", harness="v3", phase="F-2 (validation run)", reference_label="v1 tree + linear",
             reference_score=facts["prior_tree_reference"]["s3e14_v3_vs_v1_tree"]["prior_best_score"],
             tree_best=facts["s3e14_v3"]["global_best_score_positive"], verdict=facts["s3e14_v3"]["verdict"],
             n_nodes_total=facts["s3e14_v3"]["n_nodes_total"], n_evaluated=facts["s3e14_v3"]["n_evaluated"]),
    ]
    facts["all_15_runs_count"] = len(facts["all_15_runs"])
    facts["n_distinct_comps_covered"] = len({r["comp"] for r in facts["all_15_runs"]})

    # Prior-usage win-rate percentage mirrors for the 4 Phase-E scan comps (same
    # rounding convention as `v2_sweep_verdict.prior_win_rate_pct` for the 5 D-scan
    # comps above), computed here (post-dict, since it reads `facts[...]` entries
    # built above) so the report's E-scan prose percentages are traceable.
    _e_scan_keys = ["s3e9_v2", "s3e5_v2", "s3e16_v2", "s3e20_v2"]
    facts["e_scan_prior_win_rate_pct"] = {
        k: dict(
            informed_pct=round(facts[k]["prior_usage_summary"]["informed_win_rate"] * 100, 1),
            uninformed_pct=round(facts[k]["prior_usage_summary"]["uninformed_win_rate"] * 100, 1),
        )
        for k in _e_scan_keys
    }

    # Budget-efficiency, recomputed mechanically (via `budget_efficiency`/`eval_rank`)
    # for every one of the 15 runs -- generalizes docs/scaling_experiment.md's
    # 10-run "cross-run calibration" table to all 15, from this script alone.
    _be_specs = [
        ("s3e9", "experiments_tree.json"), ("s3e14", "experiments_tree.json"),
        ("s3e5", "experiments_tree.json"), ("s3e3", "experiments_tree.json"),
        ("s3e7", "experiments_tree.json"), ("s3e1", "experiments_tree.json"),
        ("s3e19", "experiments_tree.json"), ("s3e11", "experiments_tree.json"),
        ("s3e9", "experiments_tree_v2.json"), ("s3e5", "experiments_tree_v2.json"),
        ("s3e16", "experiments_tree.json"), ("s3e20", "experiments_tree.json"),
        ("s3e3", "experiments_tree_scale.json"),
        ("s3e7", "experiments_tree_v3.json"), ("s3e14", "experiments_tree_v3.json"),
    ]
    budget_eff = [budget_efficiency(c, f) for c, f in _be_specs]
    ratios = [r["best_total_ratio"] for r in budget_eff]
    idle_tails = [r["idle_tail_evals"] for r in budget_eff]
    facts["budget_efficiency"] = dict(
        per_run=budget_eff,
        mean_best_total_ratio=round(sum(ratios) / len(ratios), 3),
        min_best_total_ratio=min(ratios),
        max_best_total_ratio=max(ratios),
        mean_idle_tail=round(sum(idle_tails) / len(idle_tails), 2),
        min_idle_tail=min(idle_tails),
        max_idle_tail=max(idle_tails),
    )

    # Ten-competition full-coverage verdict: for each of the 10 distinct comps,
    # the BEST tree-search result across whichever version(s) touched it, vs its
    # linear/baseline reference -- purely a lookup into `all_15_runs` (min score
    # per comp, mechanically, respecting each comp's own lower-is-better stored
    # convention since `tree_best`/`reference_score` above are already the
    # human-readable positive numbers).
    _maximize_comps = {"s3e3", "s3e7", "s3e5"}  # AUC (s3e3/s3e7) and QWK (s3e5):
    # higher tree_best = better, despite the table storing the positive
    # (human-readable) metric number the same way as every minimize-comp -- handled
    # explicitly here since `all_15_runs`/`linear_reference` don't carry a
    # maximize/minimize flag themselves.
    # Classification is done on a NUMERIC diff against linear (not by parsing the
    # descriptive `verdict` strings, which can be compound e.g. s3e9's "BEAT v1
    # tree / TIE linear" -- a substring match on those would double-count). s3e9 is
    # the one comp where the transcribed 5dp-rounded `linear_reference` figure
    # (12.07003) differs from the FULL-precision number the exact tie was checked
    # against (12.070034, see `derived_diffs.s3e9_v2_minus_linear_full_precision`
    # == 0.0 exactly) -- overridden here for classification only.
    _classification_linear_override = {"s3e9": 12.070034}
    by_comp = {}
    for r in facts["all_15_runs"]:
        by_comp.setdefault(r["comp"], []).append(r)
    ten_comp = {}
    for comp, runs in by_comp.items():
        if comp in _maximize_comps:
            best_run = max(runs, key=lambda r: r["tree_best"])
        else:
            best_run = min(runs, key=lambda r: r["tree_best"])
        linear_score = facts["linear_reference"].get(comp, {}).get("best_score")
        classify_against = _classification_linear_override.get(comp, linear_score)
        if comp in _maximize_comps:
            better_amount = round(best_run["tree_best"] - classify_against, 6)
        else:
            better_amount = round(classify_against - best_run["tree_best"], 6)
        verdict_vs_linear = "TIE" if abs(better_amount) < 1e-9 else (
            "WIN" if better_amount > 0 else "LOSS")
        ten_comp[comp] = dict(
            best_phase=best_run["phase"], best_harness=best_run["harness"],
            best_tree_score=best_run["tree_best"], linear_score=linear_score,
            descriptive_verdict=best_run["verdict"],
            verdict_vs_linear=verdict_vs_linear,
        )
    n_win = sum(1 for v in ten_comp.values() if v["verdict_vs_linear"] == "WIN")
    n_tie = sum(1 for v in ten_comp.values() if v["verdict_vs_linear"] == "TIE")
    n_loss = sum(1 for v in ten_comp.values() if v["verdict_vs_linear"] == "LOSS")
    facts["ten_comp_coverage"] = dict(
        n_comps=len(ten_comp), by_comp=ten_comp,
        n_win=n_win, n_tie=n_tie, n_loss=n_loss,
    )

    # Mechanism finding #1: the explore-burst mega-blend (v3 feature 1 + E-5's
    # discovery) -- 3/3 runs where a mandatory late-stage kitchen-sink blend fired
    # supplied 100% of the post-exploit-phase gain.
    facts["burst_mechanism_finding"] = dict(
        n_runs=3,
        runs=dict(
            s3e3_scale=dict(pre_burst_ceiling=facts["s3e3_scale_curve"]["pre_explore_ceiling"],
                             post_burst_best=facts["s3e3_scale_curve"]["final_best"],
                             gain=facts["s3e3_scale_curve"]["post_explore_gain"]),
            s3e7_v3=dict(pre_burst_ceiling=round(-facts["s3e7_v3_node10_preburst_ceiling"]["score"], 6),
                         post_burst_best=facts["s3e7_v3"]["global_best_score_positive"],
                         gain=round(facts["s3e7_v3"]["global_best_score_positive"]
                                     - round(-facts["s3e7_v3_node10_preburst_ceiling"]["score"], 6), 6)),
            s3e14_v3=dict(pre_burst_ceiling=facts["s3e14_v3_node15_preburst_ceiling"]["score"],
                          post_burst_best=facts["s3e14_v3"]["global_best_score_positive"],
                          gain=round(facts["s3e14_v3"]["global_best_score_positive"]
                                      - facts["s3e14_v3_node15_preburst_ceiling"]["score"], 5),
                          gain_abs=round(abs(facts["s3e14_v3"]["global_best_score_positive"]
                                              - facts["s3e14_v3_node15_preburst_ceiling"]["score"]), 5)),
        ),
    )

    # Mechanism finding #2: boundary-push as the single largest lever -- 3
    # comps (D-6/E-2/E-3) where an Optuna-tuned hyperparameter sitting on its own
    # search-box edge was the run's headline lever, per harness_v3.py's own
    # docstring (feature 4 evidence).
    facts["boundary_push_mechanism_finding"] = dict(
        n_comps=3,
        comps=dict(
            s3e11_v2=dict(param="CatBoost max_depth", pushed="10 -> 12",
                          solo_before=0.295779, solo_after=0.295461,
                          source="competitions/playground-series-s3e11/STATUS.md Phase D-6 appendix"),
            s3e5_v2=dict(param="tuned-LGB max_depth/min_child_samples/reg_lambda box edges",
                         pushed="max_depth 3 -> 2 (LGBBOUND)",
                         solo_score=facts["s3e5_v2_node7_lgbbound"]["score"],
                         blend_weight=0.297,
                         source="competitions/playground-series-s3e5/STATUS.md Phase E-2 appendix"),
            s3e16_v2=dict(param="tuned-LGB learning_rate (Optuna box [0.01,0.06] log)",
                          pushed="lr 0.0102 -> 0.005 (LGBBOUND)",
                          solo_before=1.33979, solo_after=1.33950,
                          source="competitions/playground-series-s3e16/STATUS.md Phase E-3 appendix"),
        ),
        bonus_confirmation_s3e7_v3=dict(
            param="root max_depth (Optuna box [3,12], sat at 3)",
            note="boundary_candidates() auto-flagged it; the pushed depth-2 variant "
                 "(EXPL_BOUND2) earned 0.114 weight in the winning mega-blend despite "
                 "a worse solo score -- 4th confirming data point, found automatically "
                 "by v3 rather than by a human re-reading the Optuna trial table.",
            source="competitions/playground-series-s3e7/STATUS.md Phase F-2 appendix",
        ),
    )

    out_path = os.path.join(_REPO_ROOT, "docs", "tree_facts.json")
    with open(out_path, "w") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
