"""tree_search/harness_v3.py — Phase F-1 tree-search harness, folding every validated
lesson from the 11 tree-search runs (v1's s3e9/s3e14/s3e5 + v2's s3e3/s3e7/s3e1/s3e19/
s3e11 + the s3e3-scale E-5 experiment + the s3e16 E-2/E-3/C-2c build) into the harness
AS DEFAULTS, instead of leaving them as driver-script-local workarounds
(run_s3e3_scale.py's dedup-budget-burn, eval_s3e16_v2.py's k=800+coordinate-ascent,
D-6's manual "phase 2" re-seed, ...). v2 (tree_search/harness_v2.py) is left completely
untouched for reproducibility; this module imports it (`import harness_v2 as v2`),
re-exports every piece whose behavior doesn't change, and adds/overrides only what the
six Phase F-1 recommendations call for. Node schema, `kind`, OOF-cache contract, child
dedup hashing, metric-aware plateau and `suggest_priors` are ALL inherited unchanged
from v2 — see harness_v2.py's own docstring for those.

--- The six v3 features (each also documented at its own definition below) ---

1. Budget & stopping policy (`init_budget`, `update_phase`, `should_stop`) — default
   total budget 60 evaluated nodes; phase machine exploit -> explore_burst -> stopped:
   stays "exploit" until every known lineage is plateaued (plateau-saturation), then
   MANDATORY "explore_burst" (the driver is expected to inject 5-8 fresh long-shot
   lineages once this phase is observed), then stops 15-20 evals into the burst if
   none of them improved the global best; hard-stops at the total budget regardless.
   Evidence: docs/scaling_experiment.md's Phase E-5 s3e3-scale curve — every one of the
   3 post-exploit-phase global-best improvements came from a mandatory explore burst
   (kitchen-sink blend), and the run wasted a 28-eval (35% of its 80-node budget) idle
   tail with no stop rule in place.

2. Dedup-consumes-budget (`add_node`'s dedup path) — a same-parent config proposal that
   gets dedup-rejected TWICE IN A ROW burns a real `status="failed"` placeholder child
   so the parent naturally counts toward `MAX_CHILDREN_PER_NODE` and the search moves
   on, instead of spinning. Evidence: E-5 (docs/scaling_experiment.md "Implementation
   note") — v2's dedup rejects a duplicate but never consumes an expansion slot, so
   once a kitchen-sink blend lineage's fallback exhausted its member pool it
   regenerated an identical config forever and stalled run_s3e3_scale.py at 50/80
   nodes; that run's hand-rolled fix (`_parent_dup_streak`) is formalized here as the
   harness's own default.

3. Post-plateau solo breakthrough auto-reopens blend lineages
   (`reopen_blend_lineage_on_solo_breakthrough`, called automatically from `add_node`)
   — whenever a solo node becomes the new global best, any already-`plateaued` blend
   lineage(s) are reopened (un-plateaued, streak reset) so they can absorb the new
   member on the very next `select_next_parent()` call. Evidence: D-6 (s3e11,
   docs/tree_search_prototype.md §"v3 候選規則" item 1) — the run's final global best
   was a depth-12 CatBoost solo node found AFTER the phase-1 BLEND lineage had already
   plateaued; harvesting that gain required a human-driven manual "phase 2" re-seed.
   v3 makes this the harness's own standard behavior.

4. Boundary-push as a first-class mutation type (`boundary_candidates`) — flags any
   hyperparameter in a config sitting within `edge_frac` of either edge of its declared
   search-space range and proposes pushing further past that edge. Evidence: D-6/E-2/
   E-3 (docs/tree_search_prototype.md §"v3 候選規則" item 2 + §7's four local-insight
   examples) — three separate comps (s3e11's CatBoost depth 10->12, s3e16's
   learning_rate below its Optuna box's own 0.01 floor, and the general "Optuna optimum
   sitting on the search box's own edge" pattern) each found their single largest lever
   this way; leaving it to be "discovered" ad hoc means it is discovered late or not at
   all.

5. Weight-search default k=800 + coordinate-ascent refinement, baked into v3's own
   `eval_blend` (v2's `eval_blend` is untouched; this wraps it). Evidence: E-2 (s3e16)
   — coarse Dirichlet grids silently tie (multiple candidate weight vectors round to
   the same discretized score, so the "winner" is arbitrary among near-duplicates)
   unless the search is both wide (k=800, up from ad hoc per-caller defaults) AND
   followed by a local coordinate-ascent hill-climb (ported from
   eval_s3e16_v2.py's `_coord_ascent_refine`, previously hand-copied per eval script);
   E-3 additionally showed that raw search budget alone (more draws, no new idea) beat
   a hand-tuned champion blend, i.e. the harness should never under-search by default.

6. Metric-aware blend-cost guard (`eval_blend_with_cost_guard`) — if a blend
   evaluation's wall-clock time exceeds a threshold (default 45s), the guard
   auto-coarsens the weight-search budget for that call ONLY with an explicit, logged
   warning (never silently). Evidence: C-2c (s3e5, docs/tree_search_prototype.md §3/§4)
   — QWK-after-rounder blend nodes cost ~45-46s each (a Nelder-Mead OptimizedRounder
   cutpoint-fit per candidate weight), on par with solo nodes, breaking the s3e14-
   derived "blend nodes are near-free" assumption; and E-2 showed that SILENTLY
   coarsening a search to save time just ties (loses real signal) without telling
   anyone, so any coarsening this harness ever does must be loud, not silent.

--- Phase H-1: three v3 production-readiness items (from Phase F-2's honest ledger) ---

7. Resume-state contract (`save_search_state`/`load_search_state`/`validate_state`) — a
   harness-official persist/restore pair for the FULL driver-visible search state (phase
   machine, plateau flags, lineage bookkeeping, budget counters, plus a free-form
   `search_state["driver_state"]` dict reserved for whatever per-driver bookkeeping needs
   to survive a restart), with a self-consistency check run before every save/after every
   load. Evidence: F-2's s3e7 run needed 3 restarts, two of which were resume bugs — a
   `KeyError` from dispatching a blend-vs-solo lineage on its literal name instead of its
   `kind`, and module-level driver state (id->name maps, burst-injected flags, per-node
   result dicts) that silently did NOT survive a process restart because nothing put it
   inside the persisted tree. `search_state["driver_state"]` gives every future driver one
   documented place to put that bookkeeping so this class of bug can't recur.

8. Subprocess eval timeout (`eval_solo_subprocess`) — runs a per-comp evaluator's
   `evaluate(config, node_id=..., timeout_s=...)` in a child process and hard-kills
   (SIGKILL, whole process group) it on timeout, returning a `status="failed"` result
   instead of hanging the search loop. Evidence: F-2's s3e7 run hit a CatBoost fit (depth
   9, bagging_temperature 2.0) that hung 28 minutes at 313% CPU — `signal.alarm`
   (SIGALRM), the mechanism every eval_*.py in this repo uses today, cannot interrupt a
   native (non-Python-bytecode) fit() call; only an OS-level process boundary can.

9. Burst-seed sanity gate (`burst_seed_sanity_gate`/`apply_burst_seed_sanity_gate`) —
   before a freshly-evaluated explore-burst long-shot seed is allowed to spawn a full
   mutation lineage, its score must fall within a configurable sanity band relative to
   the gap between the root's score and the current global best (default factor 3x that
   gap, with an absolute-magnitude floor so a root==global-best gap of 0 doesn't fail-
   gate everything). Failing the gate marks that seed's own lineage plateaued
   immediately (excluded from `select_next_parent`) — it burns exactly the one seed node
   that already cost real compute, not a whole lineage of children descending from it.
   Evidence: F-2's s3e14 run — a DART long-shot seed scored 6144-6544 MAE against a
   ~340 root/global-best (~18x the gap), and the driver spent ~12 minutes (6 evals)
   training further DART children off that single garbage seed before giving up.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # so `import harness_v2` resolves regardless of caller's cwd/sys.path
    sys.path.insert(0, _HERE)
import harness_v2 as v2  # noqa: E402 -- tree_search/harness_v2.py, unchanged (v2)

# --- re-exported unchanged v2 (and, transitively, v1) pieces ---------------------------
new_tree = v2.new_tree
load = v2.load
save = v2.save
next_id = v2.next_id
lineage_of = v2.lineage_of
global_best = v2.global_best
lineage_size = v2.lineage_size
select_next_parent = v2.select_next_parent
config_hash = v2.config_hash
find_duplicate_config = v2.find_duplicate_config
cache_oof = v2.cache_oof
load_oof = v2.load_oof
tie_rate = v2.tie_rate
suggest_priors = v2.suggest_priors
add_root = v2.add_root

MAX_CHILDREN_PER_NODE = v2.MAX_CHILDREN_PER_NODE
PLATEAU_STREAK = v2.PLATEAU_STREAK
ADAPTIVE_PLATEAU_STREAK = v2.ADAPTIVE_PLATEAU_STREAK
TIE_RATE_THRESHOLD = v2.TIE_RATE_THRESHOLD

# ---------------------------------------------------------------------------
# Feature 1: budget & stopping policy
# ---------------------------------------------------------------------------
DEFAULT_TOTAL_BUDGET = 60             # E-5 verdict: ~55-60 evaluated nodes (docs/scaling_experiment.md "Verdict for Stage-4 budgeting" #3)
EXPLORE_BURST_MIN = 5                 # E-5 verdict #1: "mandatory ... explore burst of 5-8 seeded long-shot lineages"
EXPLORE_BURST_MAX = 8
DEFAULT_EXPLORE_BURST_SIZE = 6        # evals-worth of burst window before the stop-patience counter is enforced
DEFAULT_POST_BURST_PATIENCE = 20      # E-5 verdict #2: "stop ~15-20 evals after the explore burst if none of them improved the global best"


def init_budget(tree: dict, total_budget: int = DEFAULT_TOTAL_BUDGET,
                 explore_burst_size: int = DEFAULT_EXPLORE_BURST_SIZE,
                 post_burst_patience: int = DEFAULT_POST_BURST_PATIENCE) -> dict:
    """Idempotently create/return `tree["search_state"]["budget"]`, the v3 phase-machine
    state: `{total_budget, explore_burst_size, post_burst_patience, phase, burst_start_eval,
    best_at_burst_start, evals_since_burst_improve, stop_reason}`. `phase` starts at
    "exploit" and only ever advances forward (exploit -> explore_burst -> stopped, see
    `update_phase`). Safe to call multiple times (e.g. once per driver resume) — a
    pre-existing budget dict is returned unmodified."""
    st = tree.setdefault("search_state", {})
    if "budget" not in st:
        st["budget"] = dict(
            total_budget=total_budget, explore_burst_size=explore_burst_size,
            post_burst_patience=post_burst_patience, phase="exploit",
            burst_start_eval=None, best_at_burst_start=None,
            evals_since_burst_improve=0, stop_reason=None,
        )
    return st["budget"]


def n_evaluated(tree: dict) -> int:
    """Count of `status == "evaluated"` nodes in the tree — the harness's one definition
    of "budget spent so far", shared by the phase machine and any driver loop."""
    return sum(1 for n in tree["nodes"] if n["status"] == "evaluated")


def _lineages_with_evals(tree: dict):
    root_id = tree["root_id"]
    return {v2.lineage_of(tree, n["id"]) for n in tree["nodes"]
            if n["status"] == "evaluated" and n["id"] != root_id}


def plateau_saturated(tree: dict) -> bool:
    """True iff at least one lineage has been evaluated AND every lineage the tree
    currently knows about is in `search_state["plateaued"]` — the harness's definition
    of "the exploit phase looks done" that triggers the mandatory explore burst
    (feature 1). Deliberately does NOT call `select_next_parent` (which has a
    reopen-once side effect) — this is a pure read of current plateau bookkeeping."""
    lineages = _lineages_with_evals(tree)
    plateaued = set(tree["search_state"].get("plateaued", []))
    return bool(lineages) and lineages <= plateaued


def update_phase(tree: dict) -> str:
    """Advance the v3 budget phase machine and return the (possibly just-updated)
    phase string. MUST be called after every node is added (`add_node`/`add_root`
    below call it automatically, so drivers using this module's `add_node` get the
    phase machine for free).

    Transitions (see module docstring, feature 1, for the evidence):
      exploit -> explore_burst   : as soon as `plateau_saturated(tree)` is True. The
                                    driver is expected to notice this phase and inject
                                    EXPLORE_BURST_MIN..EXPLORE_BURST_MAX fresh long-shot
                                    lineages off the root (the harness signals *when*,
                                    not *what* — mutation content is comp-specific).
      explore_burst -> stopped   : once at least `explore_burst_size` evals have
                                    elapsed since the burst started AND
                                    `post_burst_patience` consecutive evals (within the
                                    burst window) passed with no new global best.
      (any phase) -> stopped     : hard cap, `n_evaluated(tree) >= total_budget`,
                                    regardless of the above (E-5 verdict #3, the
                                    "numeric backstop").
    """
    _st = tree["search_state"]
    budget = init_budget(tree)
    n_eval = n_evaluated(tree)

    if budget["phase"] == "stopped":
        return "stopped"

    if n_eval >= budget["total_budget"]:
        budget["phase"] = "stopped"
        budget["stop_reason"] = (f"hard budget cap reached ({n_eval}/{budget['total_budget']} "
                                  f"evaluated nodes)")
        return "stopped"

    if budget["phase"] == "exploit":
        if plateau_saturated(tree):
            budget["phase"] = "explore_burst"
            budget["burst_start_eval"] = n_eval
            gb = v2.global_best(tree)
            budget["best_at_burst_start"] = gb["score"] if gb else None
            budget["evals_since_burst_improve"] = 0
        return budget["phase"]

    if budget["phase"] == "explore_burst":
        gb = v2.global_best(tree)
        cur_best = gb["score"] if gb else None
        prior_best = budget["best_at_burst_start"]
        if prior_best is None or (cur_best is not None and cur_best < prior_best):
            budget["best_at_burst_start"] = cur_best
            budget["evals_since_burst_improve"] = 0
            budget["patience_at_eval"] = n_eval
        else:
            # Count EVALUATED NODES, not update_phase invocations: failed adds, dedup
            # placeholder burns and defensive polls used to inflate this counter and stop the
            # burst early, while the sibling condition (evals_since_burst_start) counted only
            # evaluated nodes -- two conjuncts in different units (2026-08-03 audit).
            since = budget.get("patience_at_eval")
            if since is None:
                budget["patience_at_eval"] = n_eval
                since = n_eval
            budget["evals_since_burst_improve"] = max(0, n_eval - since)
        evals_since_burst_start = n_eval - budget["burst_start_eval"]
        if (evals_since_burst_start >= budget["explore_burst_size"]
                and budget["evals_since_burst_improve"] >= budget["post_burst_patience"]):
            budget["phase"] = "stopped"
            budget["stop_reason"] = (
                f"{budget['evals_since_burst_improve']} evals without improvement "
                f"post-burst (patience={budget['post_burst_patience']})")
        return budget["phase"]

    return budget["phase"]


def should_stop(tree: dict) -> bool:
    """True iff the v3 phase machine has reached "stopped" (see `update_phase`). Drivers
    should check this before every `select_next_parent()` call."""
    return tree.get("search_state", {}).get("budget", {}).get("phase") == "stopped"


# ---------------------------------------------------------------------------
# Feature 3: post-plateau solo breakthrough re-opens blend lineages
# ---------------------------------------------------------------------------
def _node_kind(tree, node):
    return node.get("kind", node["config"].get("kind", "solo"))


def reopen_blend_lineage_on_solo_breakthrough(tree: dict, node_id: int) -> list:
    """If `node_id` is a solo node that just became the tree's new global best,
    un-plateau (reopen) any blend-kind lineage(s) currently in `search_state["plateaued"]`
    and reset their streak counters to 0, so they become selectable again on the very
    next `select_next_parent()` call and can absorb the new solo member. Returns the list
    of reopened lineage ids (empty if nothing fired). See module docstring feature 3
    (D-6 lesson) for the evidence."""
    node = next((n for n in tree["nodes"] if n["id"] == node_id), None)
    if node is None or node["status"] != "evaluated" or _node_kind(tree, node) != "solo":
        return []
    gb = v2.global_best(tree)
    if gb is None or gb["id"] != node_id:
        return []  # not a breakthrough -- didn't become (or isn't) the global best

    st = tree["search_state"]
    plateaued = st.get("plateaued", [])
    reopened, still_plateaued = [], []
    for lid in plateaued:
        lineage_nodes = [n for n in tree["nodes"]
                          if n["id"] != tree["root_id"] and v2.lineage_of(tree, n["id"]) == lid]
        if any(_node_kind(tree, n) == "blend" for n in lineage_nodes):
            reopened.append(lid)
            st["streak"][str(lid)] = 0
        else:
            still_plateaued.append(lid)

    if reopened:
        st["plateaued"] = still_plateaued
        st.setdefault("backtrack_log", []).append(dict(
            at_node_id=node_id, plateaued_lineage=None, reopened_lineages=reopened,
            reason=(f"solo node #{node_id} became the new global best (score={node['score']}) "
                    f"post-plateau -- auto-reopening blend lineage(s) {reopened} so they can "
                    f"absorb it as a member (D-6 lesson, formalized as a default)")))
    return reopened


# ---------------------------------------------------------------------------
# Feature 2 + 3 + 1 wired into add_node: the single v3 entry point drivers should use.
# ---------------------------------------------------------------------------
def add_node(tree: dict, parent_id: int, mutation: str, config: dict, score, status: str,
             wall_s: float, *, kind: str = None):
    """v3's `add_node` — same contract as `harness_v2.add_node` (`(nid, dup_id)`,
    `dup_id is None` on success), plus every relevant Phase F-1 default wired in
    automatically so a driver gets them "for free" just by calling this instead of
    `harness_v2.add_node`:

      - Feature 2 (dedup consumes budget): if `config` dedup-rejects (config_hash matches
        an existing node) against the SAME `parent_id` for the SECOND time in a row, a
        `status="failed"` placeholder child is burned under that parent (see module
        docstring). The rejection itself still returns `(None, dup_id)` exactly like v2
        — the placeholder is a side effect on the tree, not a different return contract.
      - Feature 3: on a successful add, `reopen_blend_lineage_on_solo_breakthrough` runs
        automatically.
      - Feature 1: on a successful add (and also on a burned placeholder, since it counts
        toward MAX_CHILDREN_PER_NODE but NOT toward n_evaluated as it's status="failed"),
        `update_phase` runs automatically so the budget/phase state stays current without
        the driver having to remember to call it.

    Pass `kind` exactly as `harness_v2.add_node` accepts it (defaults to
    `config.get("kind", "solo")`).
    """
    dup_id = v2.find_duplicate_config(tree, config)
    if dup_id is not None:
        streak_map = tree["search_state"].setdefault("dedup_streak", {})
        key = str(parent_id)
        streak = streak_map.get(key, 0) + 1
        streak_map[key] = streak
        if streak >= 2:
            v2.add_node(
                tree, parent_id,
                f"[dead-end placeholder] parent #{parent_id}'s proposal dedup-rejected "
                f"{streak}x in a row (config identical to existing node #{dup_id}) -- "
                f"burning this expansion slot per E-5's dedup-budget-burn fix, formalized "
                f"as a v3 default",
                {"kind": "placeholder", "dead_end_parent": parent_id, "streak": streak},
                None, "failed", 0.0, allow_duplicate=True)
            streak_map[key] = 0
            update_phase(tree)
        return None, dup_id

    tree.setdefault("search_state", {}).setdefault("dedup_streak", {})[str(parent_id)] = 0
    nid, add_dup = v2.add_node(tree, parent_id, mutation, config, score, status, wall_s, kind=kind)
    if nid is not None:
        reopen_blend_lineage_on_solo_breakthrough(tree, nid)
        update_phase(tree)
    return nid, add_dup


# ---------------------------------------------------------------------------
# Feature 4: boundary-push as a first-class mutation type
# ---------------------------------------------------------------------------
def boundary_candidates(config: dict, search_space: dict, *, params_key: str = "params",
                         edge_frac: float = 0.05, push_factor: float = 1.5) -> list:
    """Flag every numeric hyperparameter in `config[params_key]` that sits within
    `edge_frac` (default 5%) of either edge of its declared range in `search_space`, and
    propose a mutation pushing further past that edge. `search_space[name]` may be
    either `(lo, hi)` or `{"low":.., "high":.., "log": bool}` (log-scale ranges, e.g.
    learning_rate, are compared/pushed in log-space).

    Returns a list of `{"param", "edge", "old_value", "new_value"}` dicts (empty if no
    param in `config[params_key]` sits on an edge, or `search_space` has no entry for
    it). A driver turns each dict into a mutation by copying `config` and setting
    `config[params_key][param] = new_value`.

    Evidence (module docstring feature 4): D-6 (s3e11 CatBoost depth 10->12, searched
    [4,10], won at 10 -- the box's own upper edge), E-2/E-3 (s3e16 learning_rate
    searched [0.01,0.06] log-scale, won at 0.0102 -- ~2% of the log-range from the box's
    own lower edge, "the strongest boundary signal this comp has"). Both were found by a
    human re-reading the Optuna trial table after the fact; this makes the check
    automatic and comp-agnostic.
    """
    params = config.get(params_key) or {}
    out = []
    for name, val in params.items():
        if name not in search_space or isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        spec = search_space[name]
        if isinstance(spec, dict):
            lo, hi, log = spec["low"], spec["high"], spec.get("log", False)
        else:
            lo, hi = spec
            log = False
        if hi <= lo:
            continue

        if log:
            lo_l, hi_l, val_l = np.log(lo), np.log(hi), np.log(val)
            frac_from_low = (val_l - lo_l) / (hi_l - lo_l)
        else:
            frac_from_low = (val - lo) / (hi - lo)

        if frac_from_low <= edge_frac:
            edge = "low"
            if log:
                new_val = val / push_factor
            else:
                step = (hi - lo) * edge_frac * push_factor
                if isinstance(val, int):
                    step = max(1, round(step))  # integer params must actually move
                new_val = val - step
        elif frac_from_low >= 1.0 - edge_frac:
            edge = "high"
            if log:
                new_val = val * push_factor
            else:
                step = (hi - lo) * edge_frac * push_factor
                if isinstance(val, int):
                    step = max(1, round(step))
                new_val = val + step
        else:
            continue

        if isinstance(val, int):
            new_val = int(round(new_val))
        out.append({"param": name, "edge": edge, "old_value": val, "new_value": new_val})
    return out


# ---------------------------------------------------------------------------
# Feature 5: weight-search default k=800 + coordinate-ascent refinement
# ---------------------------------------------------------------------------
DEFAULT_BLEND_K = 800
DEFAULT_ASCENT_ROUNDS = 6
_ASCENT_DELTAS = (0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005)


def _coordinate_ascent_refine(oofs, metric_fn, w0, s0, rounds=DEFAULT_ASCENT_ROUNDS,
                               deltas=_ASCENT_DELTAS):
    """Per-coordinate hill-climb starting from (w0, s0), ported verbatim (as a default,
    not a per-eval-script copy) from eval_s3e16_v2.py's `_coord_ascent_refine`. Stops
    early if a full sweep over every coordinate/delta finds no improvement."""
    best_w, best_s = np.array(w0, dtype=float), s0
    k = len(best_w)
    for _ in range(rounds):
        improved = False
        for i in range(k):
            for d in deltas:
                w_try = best_w.copy()
                w_try[i] = max(0.0, w_try[i] + d)
                if w_try.sum() <= 0:
                    continue
                w_try = w_try / w_try.sum()
                s = metric_fn(oofs @ w_try)
                if s < best_s - 1e-9:
                    best_s, best_w = s, w_try
                    improved = True
        if not improved:
            break
    return best_w, best_s


def eval_blend(cache_dir: str, members: list, metric_fn, weight_search: str = "dirichlet",
               k: int = DEFAULT_BLEND_K, seed: int = 42, grid_step: float = 0.05,
               coordinate_ascent: bool = True, ascent_rounds: int = DEFAULT_ASCENT_ROUNDS):
    """v3's default ensemble-node evaluator: thin wrapper over
    `harness_v2.eval_blend` (unchanged, still directly importable as `v2.eval_blend`
    for exact v2-comparability -- see tests/test_tree_harness_v3.py's smoke test) that
    changes only the DEFAULTS and adds an optional refinement pass:

      - `k` defaults to 800 (was left to each caller in v2, which meant "1500" in some
        drivers, "1500" implicitly via v2's own hardcoded default in others, and never
        audited) — see module docstring feature 5 for the E-2/E-3 evidence.
      - `coordinate_ascent=True` by default: after the Dirichlet/grid search, run
        `_coordinate_ascent_refine` starting from the coarse-search winner. Can never
        make the result worse (only accepts strictly improving moves) and is what
        eval_s3e16_v2.py hand-ported per-eval-script before this harness existed.

    Passing `coordinate_ascent=False` and the same `k`/`seed`/`grid_step` as a v2 call
    reproduces `harness_v2.eval_blend`'s output EXACTLY (digit-for-digit) -- this
    wrapper adds nothing on that path, it only changes what happens by default.

    Returns `(best_weights, best_score, oofs)`, same as `harness_v2.eval_blend`.
    """
    best_w, best_s, oofs = v2.eval_blend(cache_dir, members, metric_fn,
                                          weight_search=weight_search, k=k, seed=seed,
                                          grid_step=grid_step)
    if coordinate_ascent:
        best_w, best_s = _coordinate_ascent_refine(oofs, metric_fn, best_w, best_s,
                                                     rounds=ascent_rounds)
    return best_w, best_s, oofs


# ---------------------------------------------------------------------------
# Feature 6: metric-aware blend-cost guard
# ---------------------------------------------------------------------------
DEFAULT_COST_GUARD_THRESHOLD_S = 45.0   # C-2c: s3e5's QWK-after-rounder blend nodes cost ~45-46s
DEFAULT_COST_GUARD_COARSEN_K = 200


def eval_blend_with_cost_guard(cache_dir: str, members: list, metric_fn, *, tree: dict = None,
                                weight_search: str = "dirichlet", k: int = DEFAULT_BLEND_K,
                                wall_time_threshold_s: float = DEFAULT_COST_GUARD_THRESHOLD_S,
                                coarsen_k: int = DEFAULT_COST_GUARD_COARSEN_K, seed: int = 42,
                                grid_step: float = 0.05, coordinate_ascent: bool = True,
                                ascent_rounds: int = DEFAULT_ASCENT_ROUNDS):
    """`eval_blend` (feature 5, above) wrapped with a wall-time guard: if the search
    takes longer than `wall_time_threshold_s` (default 45s, C-2c's s3e5 QWK-blend cost),
    it is re-run at a coarser `coarsen_k` (default 200) -- but ONLY with an explicit
    warning returned (and, if `tree` is passed, logged to
    `tree["search_state"]["cost_guard_log"]`). Never coarsens silently: E-2 showed that
    a silent coarsening just ties (loses real signal) with no one the wiser (see module
    docstring feature 6).

    Returns `(best_weights, best_score, oofs, warning)` -- `warning` is `None` on the
    fast path (nothing coarsened) or the human-readable message describing what was
    coarsened and why.
    """
    # The guard must decide BEFORE spending, not after. Until 2026-08-03 it ran the full-k
    # search, noticed the wall time, then re-ran at coarsen_k and returned the COARSER result
    # -- paying more than the unguarded path and recording a strictly-no-better blend. The
    # cost signal now comes from what previous blend evals on this tree actually cost.
    prior = ((tree or {}).get("search_state", {}) or {}).get("blend_wall_log", [])
    predicted = max(prior[-3:]) if prior else None
    warning = None
    k_used = k
    if predicted is not None and predicted > wall_time_threshold_s and k > coarsen_k:
        k_used = coarsen_k
        warning = (f"previous blend evals took up to {predicted:.1f}s > "
                   f"{wall_time_threshold_s:.0f}s threshold (members={members}) -- "
                   f"COARSENING this eval to k={coarsen_k} BEFORE running it, "
                   f"logged explicitly per C-2c/E-2 (never silent)")

    t0 = time.time()
    best_w, best_s, oofs = eval_blend(cache_dir, members, metric_fn, weight_search=weight_search,
                                       k=k_used, seed=seed, grid_step=grid_step,
                                       coordinate_ascent=coordinate_ascent, ascent_rounds=ascent_rounds)
    wall = time.time() - t0

    if tree is not None:
        st = tree.setdefault("search_state", {})
        st.setdefault("blend_wall_log", []).append(round(wall, 2))
        if warning is not None:
            st.setdefault("cost_guard_log", []).append(dict(
                members=members, original_k=k, coarsened_k=coarsen_k,
                predicted_wall_s=round(predicted, 2), actual_wall_s=round(wall, 2),
                warning=warning))
    if warning is None and wall > wall_time_threshold_s:
        # over threshold but nothing was coarsened: either the first slow eval on this tree
        # (the next one will coarsen) or k is already at/below coarsen_k. Log either way --
        # the contract is that slowness is never silent.
        warning = (f"blend eval took {wall:.1f}s > {wall_time_threshold_s:.0f}s threshold "
                   f"(members={members}, k={k_used}); "
                   + ("k is already <= coarsen_k, no further coarsening available"
                      if k_used <= coarsen_k else
                      "recorded -- subsequent blend evals on this tree will be coarsened"))
        if tree is not None:
            tree["search_state"].setdefault("cost_guard_log", []).append(dict(
                members=members, original_k=k, coarsened_k=None,
                predicted_wall_s=None, actual_wall_s=round(wall, 2), warning=warning))
    return best_w, best_s, oofs, warning


# ---------------------------------------------------------------------------
# Phase H-1 feature 7: resume-state contract
# ---------------------------------------------------------------------------
# `search_state` keys this contract knows about and validates. Not exhaustive by
# construction -- any key not listed here is left alone (drivers may add their own
# top-level search_state keys, e.g. run_s3e7_v3.py's "dedup_streak"/"node_results"/
# "cost_guard_log"), but a driver's own PER-RUN bookkeeping that doesn't already have a
# blessed spot (id->name maps, "have I injected the burst yet" flags, dedup offsets, ...)
# should go under "driver_state" specifically so it round-trips through save/load with
# zero custom reconstruction code.
_KNOWN_BUDGET_PHASES = ("exploit", "explore_burst", "stopped")


def validate_state(tree: dict) -> bool:
    """Self-check that `tree["search_state"]` is internally consistent enough to safely
    resume a search from. Raises `AssertionError` (with a specific, actionable message)
    on the first inconsistency found, so a driver can do `assert hv3.validate_state(tree)`
    at its own resume point and get a real diagnostic instead of a mysterious KeyError
    three iterations later. Returns True on success (never False).

    Checks performed (deliberately about STRUCTURE, not driver policy -- e.g. it does
    NOT assert `n_evaluated(tree) < budget["total_budget"]` while phase != "stopped",
    since a driver's own extra safety caps, like a wall-clock guard, may legitimately
    stop a run before the phase machine itself would have):
      - every lineage id referenced by `plateaued`, `streak`'s keys, and
        `active_lineage` is a real node id present in the tree.
      - if `search_state["budget"]` exists: `phase` is one of the three known values;
        `phase == "explore_burst"` implies `burst_start_eval` has been recorded;
        `phase == "stopped"` implies `stop_reason` has been recorded.
      - `dedup_streak`'s keys parse as ints referring to real node ids (parent ids the
        dedup-consumes-budget bookkeeping, feature 2, tracks per-parent streaks for).
    """
    st = tree.get("search_state", {})
    node_ids = {n["id"] for n in tree.get("nodes", [])}

    for lid in st.get("plateaued", []):
        assert lid in node_ids, (
            f"validate_state: search_state['plateaued'] references lineage id {lid!r} "
            f"which is not a known node id in this tree")
    for lid_key in st.get("streak", {}):
        lid = int(lid_key)
        assert lid in node_ids, (
            f"validate_state: search_state['streak'] references lineage id {lid!r} "
            f"which is not a known node id in this tree")
    active = st.get("active_lineage")
    if active is not None:
        assert active in node_ids, (
            f"validate_state: search_state['active_lineage']={active!r} is not a known "
            f"node id in this tree")
    for pid_key in st.get("dedup_streak", {}):
        pid = int(pid_key)
        assert pid in node_ids, (
            f"validate_state: search_state['dedup_streak'] references parent id "
            f"{pid!r} which is not a known node id in this tree")

    budget = st.get("budget")
    if budget is not None:
        phase = budget.get("phase")
        assert phase in _KNOWN_BUDGET_PHASES, (
            f"validate_state: search_state['budget']['phase']={phase!r} is not one of "
            f"the known phases {_KNOWN_BUDGET_PHASES}")
        if phase == "explore_burst":
            assert budget.get("burst_start_eval") is not None, (
                "validate_state: budget phase is 'explore_burst' but 'burst_start_eval' "
                "was never recorded -- inconsistent phase-machine state")
        if phase == "stopped":
            assert budget.get("stop_reason") is not None, (
                "validate_state: budget phase is 'stopped' but 'stop_reason' was never "
                "recorded -- inconsistent phase-machine state")
    return True


def save_search_state(tree: dict, tree_path: str) -> None:
    """Persist the ENTIRE driver-visible search state to `tree_path` (the whole tree --
    nodes + search_state -- exactly like `save()`), after first validating it with
    `validate_state`. This is the harness's own named entry point for the resume
    contract (feature 7): a driver that always writes through `save_search_state`
    instead of a bespoke persistence path can never silently write a corrupt or
    incomplete state to disk, and never needs its own recovery logic on the read side
    (see `load_search_state`).

    Any driver-local bookkeeping that must survive a restart (an id->name lineage map,
    a "burst already injected" flag, per-lineage dedup offsets, ...) belongs in
    `tree["search_state"]["driver_state"]` -- a free-form dict this function persists
    and `validate_state` never inspects, reserved specifically so it stops living in
    module-level Python globals that don't survive a process restart (F-2's failure #1/
    #2 -- see module docstring feature 7)."""
    validate_state(tree)
    save(tree, tree_path)


def load_search_state(tree_path: str) -> dict:
    """Load a tree from `tree_path` (see `load`) and validate its `search_state`
    (`validate_state`) before returning it, so a driver resuming mid-run gets back the
    exact phase machine / plateau flags / lineage bookkeeping / budget counters /
    `driver_state` it left off with -- zero custom reconstruction code required. Raises
    `AssertionError` (via `validate_state`) rather than silently resuming from an
    inconsistent state."""
    tree = load(tree_path)
    tree.setdefault("search_state", {})
    validate_state(tree)
    return tree


# ---------------------------------------------------------------------------
# Phase H-1 feature 8: subprocess eval timeout
# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass  # already dead, or we're not its group leader for some reason -- best effort
    try:
        proc.kill()  # belt-and-suspenders in case killpg didn't reach the leader itself
    except OSError:
        pass


def eval_solo_subprocess(eval_module_path: str, config: dict, timeout_s: float, *,
                          node_id: int = None, cmd_prefix: list = None) -> dict:
    """Run a per-comp evaluator's `evaluate(config, node_id=..., timeout_s=...) -> dict`
    (the `{"score", "status", "wall_s", "result", "error"}` contract every eval_*.py in
    this repo already implements) in a CHILD PROCESS, and hard-kill (SIGKILL, whole
    process group) it if it exceeds `timeout_s` wall-clock -- instead of hanging the
    search loop, which is what happens today: every eval_*.py uses `signal.alarm`
    (SIGALRM) for its in-process timeout, and SIGALRM CANNOT interrupt a native-code
    fit() call (CatBoost/LightGBM/XGBoost) that never returns to the Python bytecode
    interpreter to notice the pending signal (F-2's s3e7 run: a CatBoost fit hung 28
    minutes at 313% CPU past its 200s in-process timeout). A subprocess boundary is
    immune to that -- the OS can always kill it.

    `eval_module_path` is a per-comp evaluator module path (e.g.
    "tree_search/eval_s3e7.py"), loaded in the child the same way tests/conftest.py's
    `load_module` fixture loads it (importlib, by file path -- these modules aren't a
    package). `config` is passed to the child via a temp JSON file (numpy scalars/arrays
    coerced to native JSON types), not argv/stdin, so arbitrary config nesting survives.

    `cmd_prefix` (default `[sys.executable]`) is the subprocess command to run the child
    interpreter with -- pass `["uv", "run", "python3"]` if the child needs its own fresh
    `uv run` environment resolution; the default reuses the CURRENT interpreter (already
    uv-managed when the parent itself was launched via `uv run ...`), which is cheaper
    and sufficient for the common case.

    On a clean, on-time finish: returns the evaluator's own result dict, verbatim (plus
    `timeout=False`, `error=None` defaults if the evaluator's own dict omitted them).
    On timeout, or if the child crashes/exits nonzero without producing a result:
    returns `{"score": None, "status": "failed", "wall_s": <measured>, "result": None,
    "timeout": <bool>, "error": <human-readable message>}` -- NEVER raises, NEVER hangs.
    This matches the existing `status="failed"` contract exactly, so a driver's
    `eval_and_add` needs zero special-casing beyond calling this instead of
    `ev.evaluate(...)` directly for a solo eval it wants subprocess-isolated.
    """
    cmd_prefix = list(cmd_prefix) if cmd_prefix else [sys.executable]
    eval_module_path = os.path.abspath(eval_module_path)
    module_dir = os.path.dirname(eval_module_path)

    with tempfile.TemporaryDirectory() as td:
        cfg_path = os.path.join(td, "config.json")
        out_path = os.path.join(td, "result.json")
        with open(cfg_path, "w") as f:
            json.dump(config, f, default=_json_default)

        runner = (
            "import importlib.util, json, sys\n"
            f"sys.path.insert(0, {module_dir!r})\n"
            f"spec = importlib.util.spec_from_file_location('_eval_solo_subprocess_mod', {eval_module_path!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"with open({cfg_path!r}) as f:\n"
            "    config = json.load(f)\n"
            f"result = mod.evaluate(config, node_id={node_id!r}, timeout_s={timeout_s!r})\n"
            f"with open({out_path!r}, 'w') as f:\n"
            # default=str turns a numpy scalar into a STRING, so a score of np.float32 came
            # back as '0.9013' and every downstream comparison silently compared text.
            # Coerce numerics first; fall back to str only for genuinely unserializable
            # objects (2026-08-03 audit).
            "    def _enc(o):\n"
            "        import numpy as _np\n"
            "        if isinstance(o, _np.generic):\n"
            "            return o.item()\n"
            "        if isinstance(o, _np.ndarray):\n"
            "            return o.tolist()\n"
            "        return str(o)\n"
            "    json.dump(result, f, default=_enc)\n"
        )

        t0 = time.time()
        proc = subprocess.Popen(cmd_prefix + ["-c", runner], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, start_new_session=True)
        timed_out = False
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc)
            _, stderr = proc.communicate()
        wall = time.time() - t0

        if timed_out:
            return dict(score=None, status="failed", wall_s=round(wall, 2), result=None,
                        timeout=True,
                        error=(f"eval_solo_subprocess: hard-killed (SIGKILL) after "
                               f"exceeding timeout_s={timeout_s}s -- a native-code fit() "
                               f"in the child ignored its own in-process SIGALRM timeout "
                               f"(see harness_v3 module docstring feature 8 / F-2 s3e7 "
                               f"CatBoost hang)"))
        if proc.returncode != 0 or not os.path.exists(out_path):
            stderr_txt = (stderr or b"").decode(errors="replace")[-2000:]
            return dict(score=None, status="failed", wall_s=round(wall, 2), result=None,
                        timeout=False,
                        error=(f"eval_solo_subprocess: child process exited "
                               f"{proc.returncode} without producing a result "
                               f"(stderr tail: {stderr_txt})"))
        with open(out_path) as f:
            result = json.load(f)
        result.setdefault("timeout", False)
        result.setdefault("error", None)
        return result


# ---------------------------------------------------------------------------
# Phase H-1 feature 9: burst-seed sanity gate
# ---------------------------------------------------------------------------
DEFAULT_SANITY_FACTOR = 3.0        # "no worse than 3x the root-to-global-best gap"
DEFAULT_SANITY_MIN_BAND_FRAC = 0.05  # absolute floor so a 0-width gap doesn't fail-gate everything


def _root_node(tree: dict) -> dict:
    root_id = tree["root_id"]
    return next(n for n in tree["nodes"] if n["id"] == root_id)


def burst_seed_sanity_bound(tree: dict, *, factor: float = DEFAULT_SANITY_FACTOR,
                             min_band_frac: float = DEFAULT_SANITY_MIN_BAND_FRAC) -> float:
    """The max allowable (lower-is-better) score for a fresh explore-burst long-shot seed
    node, computed from THIS tree's own state -- no comp-specific metric knowledge
    needed, since every node score in this harness already follows the lower-is-better
    sign convention (module docstring, harness_v2/v1). Band = `global_best_score +
    max(factor * |global_best_score - root_score|, min_band_frac * |global_best_score|)`
    -- the `max(...)` floor guarantees a non-degenerate band even when the root happens
    to equal the current global best (gap 0), which would otherwise fail-gate every
    single burst seed by construction.

    Evidence (module docstring feature 9): F-2's s3e14 DART burst seeds scored
    6144-6544 MAE against a ~340 root/global-best (~18x the gap) -- this bound is sized
    to catch exactly that order-of-magnitude blowup while still tolerating a genuinely
    worse-but-plausible long-shot (e.g. 2x the gap is a normal, allowed exploratory
    result, not a bug)."""
    gb = global_best(tree)
    root = _root_node(tree)
    gb_score = gb["score"] if gb is not None else root["score"]
    root_score = root["score"]
    gap = abs(gb_score - root_score)
    band = max(gap * factor, min_band_frac * abs(gb_score))
    return gb_score + band


def burst_seed_sanity_gate(tree: dict, seed_score, *, factor: float = DEFAULT_SANITY_FACTOR,
                            min_band_frac: float = DEFAULT_SANITY_MIN_BAND_FRAC):
    """Does `seed_score` (lower-is-better, a just-evaluated burst long-shot seed node's
    score) pass the sanity band (`burst_seed_sanity_bound`)? Pure predicate, no
    search_state side effects -- see `apply_burst_seed_sanity_gate` for the version that
    also acts on a failure. Returns `(passed: bool, bound: float)`."""
    bound = burst_seed_sanity_bound(tree, factor=factor, min_band_frac=min_band_frac)
    return seed_score <= bound, bound


def apply_burst_seed_sanity_gate(tree: dict, seed_node_id: int, *,
                                  factor: float = DEFAULT_SANITY_FACTOR,
                                  min_band_frac: float = DEFAULT_SANITY_MIN_BAND_FRAC):
    """Run the sanity gate against an already-evaluated burst seed node (`seed_node_id`,
    expected to be a direct child of root -- i.e. its own lineage). On failure,
    immediately marks that lineage `plateaued` (so `select_next_parent` never selects it
    for further expansion) and appends a `backtrack_log` entry explaining why -- burning
    exactly the one seed node's already-sunk evaluation cost instead of letting a full
    mutation queue train children off a garbage seed (the fix for F-2's s3e14 lesson:
    ~12 minutes / 6 evals spent chasing one 6144-6544-MAE DART seed against a ~340
    root/global-best).

    A no-op (returns `(True, bound)` without touching `search_state`) if the node is
    missing, not yet evaluated, or scoreless -- safe to call unconditionally right after
    every burst seed's `add_node`/`eval_and_add` call. Returns `(passed: bool, bound:
    float or None)`."""
    node = next((n for n in tree["nodes"] if n["id"] == seed_node_id), None)
    if node is None or node["status"] != "evaluated" or node["score"] is None:
        return True, None

    passed, bound = burst_seed_sanity_gate(tree, node["score"], factor=factor,
                                            min_band_frac=min_band_frac)
    if not passed:
        st = tree.setdefault("search_state", {})
        plateaued = st.setdefault("plateaued", [])
        # Also record it as GATE-rejected, so v1's reopen-once fallback (harness.py) cannot
        # resurrect a lineage this gate refused: that mark is a verdict, not exhaustion.
        gated = st.setdefault("gated_lineages", [])
        if seed_node_id not in gated:
            gated.append(seed_node_id)
        if seed_node_id not in plateaued:
            plateaued.append(seed_node_id)
            st.setdefault("backtrack_log", []).append(dict(
                at_node_id=seed_node_id, plateaued_lineage=seed_node_id,
                reason=(f"burst-seed sanity gate FAILED: node #{seed_node_id}'s score "
                        f"{node['score']!r} exceeds the sanity bound {bound:.6g} "
                        f"(factor={factor}x the root-to-global-best gap, floor="
                        f"{min_band_frac:.0%} of |global best|) -- burning this seed's "
                        f"own already-sunk evaluation cost only, NOT opening a mutation "
                        f"lineage on top of it (feature 9; F-2 s3e14 lesson)")))
    return passed, bound
