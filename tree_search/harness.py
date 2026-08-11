"""tree_search/harness.py — minimal ERA-inspired (Aygün et al. 2026) candidate-tree
search harness. Reusable across competitions; the only per-comp piece is an
`evaluate(config) -> score` function (see tree_search/a per-competition evaluator for the one competition one).

Node = one complete, evaluated (or failed) solution::

    {id, parent_id, mutation, config, score, status, wall_s}

    id          int, unique, monotonically assigned (0 = root)
    parent_id   int or None (root)
    mutation    one-line human description of what changed vs the parent
    config      JSON-serializable dict: {"model", "params", "features": {"drop": [...]},
                "postprocess": {...}} — whatever the per-comp evaluator expects
    score       CV score (lower-is-better metrics assumed throughout this harness;
                flip sign before calling in if the comp metric is maximize-better),
                or None if status == "failed"
    status      "evaluated" | "failed"
    wall_s      wall-clock seconds the eval took (or time spent before failing)

Tree = {"comp": str, "root_id": 0, "nodes": [Node, ...], "search_state": {...}}.
Call `save(tree, path)` after EVERY add_root/add_node call — this is the crash-safe
resume mechanism the design calls for (write is atomic via a temp-file + os.replace).
`load(path)` reconstructs the same dict from disk.

--- Selection rule (deliberately simple; this docstring IS the spec) ---

"Lineage" of a node = the direct child-of-root it descends from. Root's first
generation of children each found their own lineage (a "candidate subtree"); the
root itself has no lineage. The FIRST generation is seeded directly by the caller
(the mutation proposer) via repeated `add_node(tree, parent_id=root_id, ...)` calls
— `select_next_parent` only governs which lineage/node to continue expanding once at
least one lineage already exists, matching the design's framing of an LLM proposing
several initial directions before the tree-search loop takes over deciding *which one
to keep pulling on*.

At any time exactly one lineage is "active" (`search_state["active_lineage"]`). The
node returned by `select_next_parent()` is: within the active lineage, the evaluated
node with the best (lowest) score that still has expansion budget
(`< MAX_CHILDREN_PER_NODE` children of its own).

PLATEAU / BACKTRACK: every time a new child is added to the *active* lineage via
`add_node`, compare its score to the global-best score that existed immediately
before it was added:
  - if it improves on that prior global best -> the lineage's non-improvement streak
    resets to 0 (it's still "winning", keep pulling on it).
  - if it does not -> the streak +=1.
When the streak reaches `PLATEAU_STREAK` (3) — i.e. 3 consecutive children of the
current best lineage failed to move the global best — the lineage is marked
"plateaued" and excluded from future selection. The next `select_next_parent()` call
then re-picks the active lineage as the best-scoring *non-plateaued* lineage (the
2nd-best subtree root); if that one plateaus too, the 3rd-best, and so on.

Two documented extensions beyond the literal 3-non-improving-children rule, needed so
the search never stalls on a finite, hand-authored mutation budget:
  1. A lineage that runs out of expansion budget entirely (every one of its nodes is
     already at MAX_CHILDREN_PER_NODE children) is also marked "plateaued" (a forced
     backtrack) even if it never accumulated 3 non-improving children.
  2. If literally every lineage is plateaued/exhausted, all plateau flags are cleared
     once (a "reopen", logged in `search_state["backtrack_log"]`) so the search can
     keep going toward the node budget instead of dead-ending.
"""
import json
import os
from collections import defaultdict

MAX_CHILDREN_PER_NODE = 3
PLATEAU_STREAK = 3


def new_tree(comp: str) -> dict:
    return {
        "comp": comp,
        "root_id": None,
        "nodes": [],
        "search_state": {
            "active_lineage": None,
            "streak": {},
            "plateaued": [],
            "backtrack_log": [],
        },
    }


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save(tree: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tree, f, indent=2, default=str)
    os.replace(tmp, path)


def _next_id(tree: dict) -> int:
    return (max((n["id"] for n in tree["nodes"]), default=-1)) + 1


def next_id(tree: dict) -> int:
    """Public wrapper around _next_id — the id that the next add_node(...) call will
    assign. Added for Phase C-2b (one competition ensemble-node space): a solo node's evaluator
    caches its OOF/test prediction matrix to disk keyed by node id so later blend nodes
    can reference it without retraining, and the eval call happens BEFORE add_node
    assigns the real id. Safe only under the same strictly-sequential, single-writer
    usage every run_*.py driver in this repo already assumes (no concurrent add_node
    calls between next_id() and the matching add_node() for that node)."""
    return _next_id(tree)


# The whole search machine reads this vocabulary and nothing ever validated it. n_evaluated,
# _lineage_best (so select_next_parent), global_best and the plateau bookkeeping all test
# `status == "evaluated"`, so one evaluator returning a different word made that competition's
# entire search invisible to every one of them while the tree still read as complete.
STATUSES = ("evaluated", "failed")


def _check_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(
            f"unknown node status {status!r}: the harness counts, ranks and stops on "
            f"{STATUSES[0]!r} and skips {STATUSES[1]!r}; any other word is silently invisible "
            f"to n_evaluated, select_next_parent, global_best and the plateau machine")
    return status


def add_root(tree: dict, mutation: str, config: dict, score, status: str, wall_s: float) -> int:
    assert not tree["nodes"], "root already exists"
    _check_status(status)
    nid = 0
    tree["root_id"] = nid
    tree["nodes"].append(dict(id=nid, parent_id=None, mutation=mutation, config=config,
                               score=score, status=status, wall_s=wall_s))
    return nid


def _nodes_by_id(tree: dict) -> dict:
    return {n["id"]: n for n in tree["nodes"]}


def lineage_of(tree: dict, node_id: int):
    """Id of the root's direct child that node_id descends from (root_id itself if
    node_id IS the root)."""
    nodes = _nodes_by_id(tree)
    root_id = tree["root_id"]
    if node_id == root_id:
        return root_id
    cur = node_id
    while nodes[cur]["parent_id"] != root_id:
        cur = nodes[cur]["parent_id"]
    return cur


def _children_count(tree: dict) -> dict:
    c = defaultdict(int)
    for n in tree["nodes"]:
        if n["parent_id"] is not None:
            c[n["parent_id"]] += 1
    return c


def _lineage_best(tree: dict, exclude=()) -> dict:
    """lineage_id -> best (lowest-score) evaluated node in that lineage, excluding
    lineages whose id is in `exclude`."""
    root_id = tree["root_id"]
    best = {}
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["id"] == root_id:
            continue
        lid = lineage_of(tree, n["id"])
        if lid in exclude:
            continue
        if lid not in best or n["score"] < best[lid]["score"]:
            best[lid] = n
    return best


def global_best(tree: dict):
    ev = [n for n in tree["nodes"] if n["status"] == "evaluated"]
    return min(ev, key=lambda n: n["score"]) if ev else None


def lineage_size(tree: dict, lineage_id) -> int:
    """Number of evaluated nodes belonging to this lineage (including its root)."""
    return sum(1 for n in tree["nodes"]
               if n["status"] == "evaluated" and lineage_of(tree, n["id"]) == lineage_id)


def add_node(tree: dict, parent_id: int, mutation: str, config: dict, score,
             status: str, wall_s: float, *, error: str = None) -> int:
    """Append a child of parent_id and update plateau/streak bookkeeping. Returns the
    new node id. Caller is responsible for calling save(tree, path) afterwards.

    `error` is the evaluator's own message, persisted verbatim. Without it a wiring bug
    (AttributeError, KeyError, a tuple unpacked at the wrong arity) and a learner
    legitimately refusing a bad config are byte-identical nodes, so a search that never
    happened is indistinguishable from one that happened and found nothing."""
    _check_status(status)
    nid = _next_id(tree)
    prior_best = global_best(tree)
    prior_best_score = prior_best["score"] if prior_best else float("inf")
    node = dict(id=nid, parent_id=parent_id, mutation=mutation, config=config,
                score=score, status=status, wall_s=wall_s)
    if error:
        node["error"] = str(error)
    tree["nodes"].append(node)

    if status == "evaluated" and parent_id != tree["root_id"]:
        lineage = lineage_of(tree, nid)
        st = tree["search_state"]
        if lineage == st.get("active_lineage"):
            key = str(lineage)
            if score < prior_best_score:
                st["streak"][key] = 0
            else:
                st["streak"][key] = st["streak"].get(key, 0) + 1
                if st["streak"][key] >= PLATEAU_STREAK and lineage not in st["plateaued"]:
                    st["plateaued"].append(lineage)
                    st["backtrack_log"].append(dict(
                        at_node_id=nid, plateaued_lineage=lineage,
                        reason=(f"{PLATEAU_STREAK} consecutive children of the active "
                                f"lineage failed to beat prior global best "
                                f"{prior_best_score}")))
    return nid


def select_next_parent(tree: dict):
    """Return (parent_id, lineage_id) to expand next, or (None, None) if no lineage
    exists yet (caller should seed first-generation lineages directly off the root)."""
    root_id = tree["root_id"]
    st = tree["search_state"]
    children_count = _children_count(tree)

    active = st.get("active_lineage")
    plateaued = set(st.get("plateaued", []))
    reopened_once = False

    while True:
        candidates = _lineage_best(tree, exclude=plateaued)
        if not candidates:
            gated = set(st.get("gated_lineages", []))
            if not reopened_once and (set(st.get("plateaued", [])) - gated):
                # Reopen only lineages that plateaued on their own merits. Two classes must
                # survive the fallback: lineages the burst-seed sanity gate rejected (their
                # plateau mark is a verdict, not exhaustion), and -- when the v3 phase machine
                # has already entered explore_burst -- the whole set, because reopening then
                # sends the search back into lineages the phase machine just declared spent
                # while the phase can never move backwards (2026-08-03 audit).
                # That phase is only ever correct here because harness_v3 wraps this
                # function (harness_v3.select_next_parent) to refresh its phase machine
                # around the call, and stops the search itself if the burst it is waiting
                # for can no longer arrive -- v1 has no phase machine of its own and this
                # read is a pure courtesy to v3's (2026-08-03 audit).
                phase = (tree.get("search_state", {}).get("budget", {}) or {}).get("phase")
                if phase in ("explore_burst", "stopped"):
                    return None, None
                st["plateaued"] = sorted(gated)
                st["backtrack_log"].append(dict(
                    at_node_id=None, plateaued_lineage=None,
                    reason="all lineages plateaued -> reopened once (fallback so the "
                           "search can keep going toward the node budget); "
                           f"kept {len(gated)} gate-rejected lineage(s) closed"))
                plateaued = set(gated)
                reopened_once = True
                continue
            return None, None  # nothing evaluated yet at all

        if active is None or active in plateaued or active not in candidates:
            active = min(candidates, key=lambda lid: candidates[lid]["score"])
            st["active_lineage"] = active
            st["streak"][str(active)] = 0

        lineage_node_ids = [n["id"] for n in tree["nodes"]
                             if n["status"] == "evaluated" and n["id"] != root_id
                             and lineage_of(tree, n["id"]) == active]
        avail = [nid for nid in lineage_node_ids
                 if children_count.get(nid, 0) < MAX_CHILDREN_PER_NODE]

        if not avail:
            if active not in plateaued:
                plateaued.add(active)
                st["plateaued"] = sorted(plateaued)
                st["backtrack_log"].append(dict(
                    at_node_id=None, plateaued_lineage=active,
                    reason="expansion budget exhausted (all nodes in lineage reached "
                           "MAX_CHILDREN_PER_NODE)"))
            active = None
            continue

        by_id = _nodes_by_id(tree)
        parent_id = min(avail, key=lambda nid: by_id[nid]["score"])
        return parent_id, active
