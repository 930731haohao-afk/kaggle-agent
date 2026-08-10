"""tree_search/harness_v2.py — Stage 4 / Phase D-1 tree-search harness, implementing the
four Stage-4 recommendations from docs/tree_search_prototype.md §5-6 (the several competitions
prototype's "known gaps" -> "recommendations" sections). v1 (tree_search/harness.py) is
left untouched for reproducibility; this module reuses its unchanged pieces (new_tree,
load, save, next_id, lineage_of, global_best, lineage_size, select_next_parent — the
selection/backtrack rule itself did not need to change) and only replaces/adds what the
four recommendations call for:

  1. Ensemble-default node space (§6.1): every node now carries a first-class `kind`
     ("solo"|"blend") field (previously buried inside `config`, v1-only). The
     OOF-caching contract each a per-competition evaluator/a per-competition evaluator hand-rolled is formalized here
     as `cache_oof(cache_dir, node_id, oof, **extra)` / `load_oof(cache_dir, node_id)`,
     and a comp-agnostic `eval_blend(cache_dir, members, metric_fn, weight_search=...)`
     does Dirichlet (or grid-simplex) weight search over cached member OOFs. Per-comp
     evaluators only need to supply `eval_solo(config) -> (score, oof)` for solo nodes
     and a `metric_fn(blended_oof) -> score` for blend nodes — any postprocessing
     (rounding, snap-to-grid, OptimizedRounder, ...) MUST happen *inside* metric_fn so
     it is applied to every candidate weight vector during the search itself, not just
     to the winner (see point 2 below for why that matters for discretized metrics).

  2. Metric-aware plateau (§6.2): `PLATEAU_STREAK` is no longer a single global
     constant. `tie_rate(tree)` measures the exact-duplicate fraction among all
     evaluated scores so far. If `tie_rate(tree) > TIE_RATE_THRESHOLD` (0.15) — i.e. the
     score surface looks discretized (one competition's QWK-after-rounder: 4 exact-duplicate
     5-decimal groups out of 40 nodes, tie_rate ~0.19; one competition/one competition's continuous
     RMSE/MAE never tied) — two things change for the active lineage's streak
     bookkeeping in `add_node`: (a) the streak-to-plateau threshold becomes
     `ADAPTIVE_PLATEAU_STREAK` (5) instead of `PLATEAU_STREAK` (3), and (b) a child
     whose score is an EXACT tie with the prior global best is treated as neutral —
     it neither resets the streak (it didn't improve) nor increments it (it didn't
     regress either; on a discretized surface a tie is expected/uninformative, not
     evidence the lineage is stuck). Below the threshold, ties count as non-improving
     exactly like v1 (continuous metrics essentially never tie, so this never fires).

  3. Child dedup (§6.4 / known-gap in §5): `find_duplicate_config(tree, config)` hashes
     every node's config (stable, order-independent JSON hash) and returns the id of any
     existing node with an identical config, or None. `add_node` calls this by default
     and REJECTS the child — returns `(None, dup_id)` instead of adding a node — so the
     mutation proposer is explicitly told the config it wanted to try already exists and
     must propose something else. This is the exact fix for a run in which three fallback
     children carried byte-identical `members` and therefore identical scores, wasting
     plateau-streak budget under v1.

  4. Experience-library mutation prior (§6.3): `suggest_priors(comp_meta)` does simple
     keyword-to-section-header matching against knowledge/experience.md (every `##`/`###`
     markdown section, no LLM call) and returns the evidence-tagged bullet lines
     ("... | 證據: ...") from any section whose header contains one of comp_meta's
     metric/tag keywords, verbatim — the runner surfaces these to the mutation proposer
     (the agent) at each expansion as the simplified ERA idea-injection hook.

Node schema (superset of v1's): {id, parent_id, mutation, config, score, status, wall_s,
kind}. `kind` defaults to `config.get("kind", "solo")` if not passed explicitly.
"""
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_EXPERIENCE_PATH = os.path.join(_REPO_ROOT, "knowledge", "experience.md")

if _HERE not in sys.path:  # so `import harness` resolves regardless of caller's cwd/sys.path
    sys.path.insert(0, _HERE)
import harness as v1  # noqa: E402 — tree_search/harness.py, unchanged (v1)

# --- re-exported unchanged v1 pieces (no behavior change needed for any of the four
# recommendations) -----------------------------------------------------------------
new_tree = v1.new_tree
load = v1.load
save = v1.save
next_id = v1.next_id
lineage_of = v1.lineage_of
global_best = v1.global_best
lineage_size = v1.lineage_size
select_next_parent = v1.select_next_parent

MAX_CHILDREN_PER_NODE = v1.MAX_CHILDREN_PER_NODE
PLATEAU_STREAK = v1.PLATEAU_STREAK          # continuous-metric threshold (unchanged, §6.2)
ADAPTIVE_PLATEAU_STREAK = 5                  # discretized-metric threshold (§6.2)
TIE_RATE_THRESHOLD = 0.15                    # exact-duplicate-score fraction that flips
                                              # a lineage's plateau rule to "adaptive"


# ---------------------------------------------------------------------------
# Recommendation 2: metric-aware plateau
# ---------------------------------------------------------------------------
def tie_rate(tree: dict) -> float:
    """Exact-duplicate-score fraction among all evaluated nodes in the tree:
    `1 - distinct_scores / n_evaluated`. 0.0 if fewer than 2 evaluated nodes exist.
    This is the harness's cheap proxy for "is this comp's metric surface discretized"
    (recommendation #2) — measured on already-rounded `score` values exactly as stored
    on each node, matching how a per-competition evaluator rounds every score to 5 decimals before
    handing it to add_node."""
    scores = [n["score"] for n in tree["nodes"] if n["status"] == "evaluated"]
    if len(scores) < 2:
        return 0.0
    distinct = len(set(scores))
    return 1.0 - (distinct / len(scores))


# ---------------------------------------------------------------------------
# Recommendation 3: child dedup
# ---------------------------------------------------------------------------
def _canonical_for_hash(v):
    """Value-level canonicalization applied before hashing a config, so the hash compares
    VALUES rather than their Python spelling: numpy scalars/arrays become their native
    equivalents (np.int64(31) hashed as the string "31" via `default=str`, while a plain
    31 hashed as the number 31 -- two different hashes for the same hyperparameter), and
    an integral float collapses onto the int (`max_depth: 6.0` == `max_depth: 6`).
    Deliberately conservative: nothing else is normalized, so materially different configs
    still hash differently (2026-08-03 audit)."""
    if isinstance(v, bool):
        return v                       # bool before int: True must not collapse onto 1
    if isinstance(v, np.generic):      # np.int64 / np.float32 / np.bool_ / ...
        return _canonical_for_hash(v.item())
    if isinstance(v, np.ndarray):
        return [_canonical_for_hash(x) for x in v.tolist()]
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {str(k): _canonical_for_hash(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canonical_for_hash(x) for x in v]
    return v


def config_hash(config: dict) -> str:
    """Stable (key-order-independent) hash of a JSON-serializable node config, used for
    child-dedup. `default=str` mirrors harness.save()'s tolerance of non-JSON-native
    values (e.g. numpy scalars in a params dict).

    What this hash CAN catch (2026-08-03 audit, after reading how the drivers actually
    build configs -- run_v3_generic.py, a per-competition driver, a per-competition driver, a per-competition driver):
    every driver stores the MERGED config a node was evaluated with (full `params` dict,
    not the one-line mutation), normalized by its own local `core(cfg)` helper, and passes
    that same dict to both `find_duplicate_config` and `add_node` -- so an identical
    re-proposal of an already-evaluated config IS caught, regardless of key order and
    (since the audit) regardless of whether a value is spelled `31`, `31.0` or
    `np.int64(31)`.

    What it CANNOT catch: two configs that differ in ANY stored value, even one the
    evaluator ignores or treats as equivalent -- a bumped `random_state`, a reordered
    `features.drop` list, a `members` list a driver forgot to sort, a flag the driver's
    `core()` didn't strip. Semantic equivalence is the DRIVER's job (that is exactly what
    `core()` exists for: drop `result`/`want_importance`, sort `members`); this harness
    only guarantees that whatever the driver calls "the same config" is detected as such."""
    canon = json.dumps(_canonical_for_hash(config), sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def find_duplicate_config(tree: dict, config: dict):
    """Return the id of an existing node (any status, including the root) whose config
    hash exactly equals `config`'s hash, or None if no duplicate exists. Callers
    proposing a child MUST check this before spending compute on an eval; `add_node`
    below also checks it as a safety net so a duplicate can never silently enter the
    tree."""
    h = config_hash(config)
    for n in tree["nodes"]:
        if config_hash(n["config"]) != h:
            continue
        # A node that FAILED is not evidence the config is exhausted: an eval timeout, an
        # OOM or a subprocess crash is a property of that attempt, and banning the config
        # forever means a transient failure permanently removes a candidate from the search
        # (2026-08-03 audit). Only evaluated nodes -- results we actually have -- dedup.
        if n.get("status") == "failed":
            continue
        return n["id"]
    return None


# ---------------------------------------------------------------------------
# add_root / add_node — v1 behavior + kind field + dedup + adaptive plateau
# ---------------------------------------------------------------------------
def _kind_of(config, kind):
    if kind is not None:
        return kind
    if isinstance(config, dict) and config.get("kind"):
        return config["kind"]
    return "solo"


def add_root(tree: dict, mutation: str, config: dict, score, status: str, wall_s: float,
             *, kind: str = None) -> int:
    nid = v1.add_root(tree, mutation, config, score, status, wall_s)
    tree["nodes"][0]["kind"] = _kind_of(config, kind)
    return nid


def add_node(tree: dict, parent_id: int, mutation: str, config: dict, score, status: str,
             wall_s: float, *, kind: str = None, allow_duplicate: bool = False):
    """Same contract as harness.add_node, plus:

      - child dedup (recommendation #3): if `config`'s hash matches any existing node's
        config hash, the child is REJECTED — returns `(None, dup_id)` and no node is
        added, so the caller (mutation proposer) knows to propose a different config.
        Pass `allow_duplicate=True` to bypass (e.g. a deliberate exact re-run).
      - metric-aware plateau (recommendation #2): see `tie_rate` / module docstring —
        once `tie_rate(tree) > TIE_RATE_THRESHOLD`, the active lineage's streak-to-
        plateau threshold becomes `ADAPTIVE_PLATEAU_STREAK` instead of `PLATEAU_STREAK`,
        and an exact tie against the prior global best neither resets nor increments
        the streak (neutral) instead of counting as non-improving.
      - `kind` is stored as a first-class node field (recommendation #1), defaulting to
        `config.get("kind", "solo")`.

    Returns `(nid, dup_id)`: on success `dup_id is None`; on rejection `nid is None` and
    `dup_id` is the id of the pre-existing node with the identical config.
    """
    if not allow_duplicate:
        dup_id = find_duplicate_config(tree, config)
        if dup_id is not None:
            return None, dup_id

    nid = v1._next_id(tree)
    prior_best = v1.global_best(tree)
    prior_best_score = prior_best["score"] if prior_best else float("inf")
    node = dict(id=nid, parent_id=parent_id, mutation=mutation, config=config,
                score=score, status=status, wall_s=wall_s, kind=_kind_of(config, kind))
    tree["nodes"].append(node)

    if status == "evaluated" and parent_id != tree["root_id"]:
        lineage = v1.lineage_of(tree, nid)
        st = tree["search_state"]
        if lineage == st.get("active_lineage"):
            key = str(lineage)
            rate = tie_rate(tree)
            discretized = rate > TIE_RATE_THRESHOLD
            streak_limit = ADAPTIVE_PLATEAU_STREAK if discretized else PLATEAU_STREAK
            st.setdefault("tie_rate_log", []).append(round(rate, 4))
            if score < prior_best_score:
                st["streak"][key] = 0
            elif score == prior_best_score and discretized:
                pass  # neutral tie: streak unchanged (recommendation #2)
            else:
                st["streak"][key] = st["streak"].get(key, 0) + 1
                if st["streak"][key] >= streak_limit and lineage not in st["plateaued"]:
                    st["plateaued"].append(lineage)
                    st["backtrack_log"].append(dict(
                        at_node_id=nid, plateaued_lineage=lineage,
                        reason=(f"{streak_limit} consecutive non-improving children of "
                                f"the active lineage failed to beat prior global best "
                                f"{prior_best_score} (tie_rate={rate:.3f}, "
                                f"discretized={discretized})")))
    return nid, None


# ---------------------------------------------------------------------------
# Recommendation 1: ensemble-default node space — OOF cache contract + generic blend eval
# ---------------------------------------------------------------------------
def cache_oof(cache_dir: str, node_id: int, oof, **extra) -> str:
    """Persist node_id's OOF prediction array (plus any extra named arrays the caller
    wants alongside it, e.g. pred=test_pred, y=target) to
    `<cache_dir>/solo_<node_id>.npz`, atomically (temp file + os.replace). Formalizes the
    OOF-caching contract a per-competition evaluator/a per-competition evaluator each hand-rolled per-comp. Returns
    the path written."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"solo_{node_id}.npz")
    tmp = path.replace(".npz", ".tmp.npz")  # np.savez auto-appends .npz otherwise
    # Identity stamp. The cache is keyed by node id ALONE, and node ids restart at 0 in every
    # new tree, so a fresh run in a directory another tree already used silently reads that
    # tree's vectors and scores a model it never trained. Recording the row count and (when
    # the caller supplies it) the config hash lets load_oof refuse instead of returning
    # someone else's predictions (2026-08-03 audit).
    stamp = {"_n_rows": np.asarray(len(np.asarray(oof)))}
    if "config" in extra:
        stamp["_config_hash"] = np.asarray(config_hash(extra.pop("config")))
    np.savez(tmp, oof=np.asarray(oof), **stamp, **extra)
    os.replace(tmp, path)
    return path


def load_oof(cache_dir: str, node_id: int, expect_rows: int = None,
             expect_config: dict = None):
    """Load node_id's cached OOF array (see cache_oof). Raises ValueError with a clear
    message if the node was never cached — callers (typically eval_blend) should let
    this propagate so the eval dispatch can catch it and mark the node "failed" instead
    of crashing the search loop, same contract the per-competition evaluators each
    hand-rolled as load_solo_cache.

    `expect_rows` / `expect_config`, when given, are checked against the stamp cache_oof
    wrote. A mismatch means this id belongs to a different tree or a different dataset --
    the cache is keyed by node id alone, so that is a real and silent failure mode
    (2026-08-03 audit)."""
    path = os.path.join(cache_dir, f"solo_{node_id}.npz")
    if not os.path.exists(path):
        raise ValueError(f"solo node #{node_id} has no cached OOF at {path} -- it must be "
                          f"evaluated (kind='solo') before being referenced as a blend "
                          f"member")
    d = np.load(path, allow_pickle=True)
    oof = d["oof"]
    if expect_rows is not None and len(oof) != expect_rows:
        raise ValueError(
            f"cached OOF for node #{node_id} has {len(oof)} rows, expected {expect_rows} "
            f"({path}) -- this cache entry belongs to a different dataset or tree")
    if expect_config is not None and "_config_hash" in d:
        want = config_hash(expect_config)
        got = str(d["_config_hash"])
        if got != want:
            raise ValueError(
                f"cached OOF for node #{node_id} was produced by a different config "
                f"({got[:12]} != {want[:12]}, {path}) -- stale cache from an earlier tree")
    return oof


def _grid_simplex_weights(n_members, step=0.05):
    if n_members > 5:
        raise ValueError(f"grid_simplex only supported up to 5 members ({n_members} "
                          f"given) -- combinatorial blowup, use weight_search='dirichlet'")
    grid_vals = np.arange(0, 1.0 + 1e-9, step)
    rows = []
    for combo in itertools.product(grid_vals, repeat=n_members - 1):
        rem = 1.0 - sum(combo)
        if rem < -1e-9:
            continue
        rows.append(combo + (max(rem, 0.0),))
    return np.array(rows)


def blend_optimism(cache_dir: str, members: list, y, metric, *, k: int = 1500,
                   seed: int = 42, n_folds: int = 5) -> dict:
    """Measure how much of a blend's score is weight-search optimism rather than skill.

    KNOWN LIMITATION this quantifies (2026-08-03 audit, recorded rather than silently
    fixed). `eval_blend` searches ~k weight vectors on the members' OOF predictions and
    returns the BEST score it found -- on the very vectors it searched. A solo node has no
    such freedom, so blends carry a structural advantage over solos in the same tree, and
    the two kinds are nonetheless compared with one min(). The experience library already
    measured the size of this on a spectral-regression competition, and it was large
    enough to flip a comparison.

    This does the honest version WITHOUT changing how anything is scored: search weights on
    each fold's training rows, score the held-out rows with those weights, and return the
    gap. `metric(y_true, y_pred) -> float` (lower better) is taken explicitly because a
    node's metric_fn closes over the full target and cannot score a subset.

    Report the gap for a competition; do NOT wire this into node scoring without re-running
    everything -- changing the scoring rule makes old and new trees incomparable.
    """
    import numpy as _np
    oofs = _np.stack([load_oof(cache_dir, m) for m in members], axis=1)
    y = _np.asarray(y, dtype=_np.float64)
    n, n_mem = oofs.shape
    rng = _np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = [(_np.setdiff1d(idx, idx[i::n_folds]), idx[i::n_folds]) for i in range(n_folds)]

    def _search(rows):
        W = _np.vstack([_np.random.default_rng(seed).dirichlet(_np.ones(n_mem), size=k),
                        _np.eye(n_mem)])
        scores = _np.array([metric(y[rows], oofs[rows] @ w) for w in W])
        i = int(scores.argmin())
        return W[i], float(scores[i])

    _, s_in = _search(_np.arange(n))
    held = [metric(y[va], oofs[va] @ _search(tr)[0]) for tr, va in folds]
    s_held = float(_np.mean(held))
    return {"in_sample": s_in, "held_out": s_held, "optimism": s_held - s_in,
            "n_folds": n_folds, "n_members": n_mem}


# BLEND SCORING WAS NOT ONE FUNCTION (fixed 2026-08-04).
#
# The search drivers score blend nodes through harness_v3.eval_blend_with_cost_guard, whose
# defaults are k=800 PLUS a coordinate-ascent refinement pass. The 29 per-competition eval
# modules score them through this function, whose defaults are k=1500 and NO refinement. So the
# same blend config received two different scores depending on which path evaluated it, and the
# search compared them with a single min() -- a node did not have a score, it had a score per
# route (2026-08-04 architecture gate).
#
# Unified here rather than by editing 29 call sites: this is the single place every one of them
# goes through, and the switch is one constant to flip. Coordinate ascent only accepts strictly
# improving moves, so unifying UPWARD cannot make any blend worse; it does mean blend scores are
# not digit-comparable with runs recorded before this date, which is why the constant exists and
# is stated rather than silently assumed.
UNIFIED_BLEND_SCORING = True
UNIFIED_BLEND_K = 1500
UNIFIED_ASCENT_ROUNDS = 6
# Deterministic coarsening, INSIDE the unified scorer so both routes coarsen identically.
# The first unification put the coarsening decision in harness_v3's cost guard only, so a
# blend big enough to coarsen got k=200 on the driver route and k=1500 on the eval-module
# route -- a score per route again, through the guard this time (2026-08-07 re-verification;
# one competition's champion blend, 8 members x 140,700 rows, was the live case). The boundary is
# preserved from the k=800 calibration: coarsen when members*rows > 1.875e6, i.e. budget =
# 1.875e6 * UNIFIED_BLEND_K. Raising k without raising the budget had silently LOWERED the
# boundary and started coarsening blends that never coarsened before.
UNIFIED_COARSEN_BOUNDARY = 1_875_000          # members * rows
UNIFIED_COARSEN_K = 200


def unified_k_for(members_count: int, n_rows: int, k: int = None) -> int:
    """The k every route must use for this problem size. Deterministic: no clock, no load."""
    k = UNIFIED_BLEND_K if k is None else k
    if n_rows and members_count * n_rows > UNIFIED_COARSEN_BOUNDARY and k > UNIFIED_COARSEN_K:
        return UNIFIED_COARSEN_K
    return k


def eval_blend(cache_dir: str, members: list, metric_fn, weight_search: str = "dirichlet",
               k: int = 1500, seed: int = 42, grid_step: float = 0.05, target=None,
               unified: bool = None, cost_budget_units: int = None):
    """Public entry point. Applies the unified scoring contract, then delegates.

    `unified=False` reproduces the pre-2026-08-04 v2 behaviour exactly (k as passed, no
    refinement) for anyone who needs digit-for-digit comparability with an old run.
    `cost_budget_units` overrides the coarsening boundary (units = k * members * rows),
    mainly for tests; the default reproduces the k=800 calibration.
    """
    use = UNIFIED_BLEND_SCORING if unified is None else unified
    if not use:
        return _eval_blend_core(cache_dir, members, metric_fn, weight_search=weight_search,
                                k=k, seed=seed, grid_step=grid_step, target=target)
    n_rows = _cached_rows(cache_dir, members)
    # The caller's explicit k feeds the SAME deterministic rule instead of being ignored:
    # ignoring it meant an old driver passing k=800 got 1500 here and 800 on the v3 route --
    # a divergence in the other direction (2026-08-07 round-3).
    k_req = k if k != 1500 else UNIFIED_BLEND_K
    if cost_budget_units is not None:
        k_used = (UNIFIED_COARSEN_K
                  if n_rows and k_req * len(members) * n_rows > cost_budget_units
                  and k_req > UNIFIED_COARSEN_K else k_req)
    else:
        k_used = unified_k_for(len(members), n_rows, k=k_req)
    best_w, best_s, oofs = _eval_blend_core(
        cache_dir, members, metric_fn, weight_search=weight_search,
        k=k_used, seed=seed, grid_step=grid_step, target=target)
    # Coordinate ascent accepts only strictly improving moves, so this cannot make a blend
    # worse -- it makes the two routes agree. Imported lazily: harness_v3 imports this module.
    if best_w is not None and weight_search != "nnls":
        try:
            from harness_v3 import _coordinate_ascent_refine
            best_w, best_s = _coordinate_ascent_refine(
                oofs, metric_fn, np.asarray(best_w), best_s, rounds=UNIFIED_ASCENT_ROUNDS)
        except Exception:  # noqa: BLE001 - refinement is an improvement, never a dependency
            pass
    return best_w, best_s, oofs


def _cached_rows(cache_dir: str, members: list) -> int:
    """Row count of the cached OOF vectors -- deterministic, and robust to a corrupt member.

    Tries EVERY member and skips unreadable files: returning 0 on the first corrupt file
    disabled coarsening for the whole blend, silently running an unguarded full-k search
    (2026-08-07 re-verification).
    """
    for nid in members:
        fp = os.path.join(cache_dir, f"solo_{nid}.npz")
        if not os.path.exists(fp):
            continue
        try:
            with np.load(fp) as z:
                if "_n_rows" in z:
                    return int(z["_n_rows"])
                return int(np.asarray(z["oof"]).shape[0])
        except Exception:  # noqa: BLE001 - try the next member
            continue
    return 0


def _eval_blend_core(cache_dir: str, members: list, metric_fn, weight_search: str = "dirichlet",
                     k: int = 1500, seed: int = 42, grid_step: float = 0.05, target=None):
    """Comp-agnostic ensemble node evaluator (recommendation #1): loads each member's
    cached OOF array via `load_oof`, searches blend weights, scores every candidate with
    `metric_fn(blended_oof) -> score` (lower-is-better, same sign convention as
    harness.py throughout — flip sign before calling in for maximize-better metrics).
    ANY postprocessing (rounding, snap-to-grid, OptimizedRounder cutpoint-fitting, ...)
    must happen INSIDE metric_fn so it is applied to every candidate weight vector, not
    just retroactively to the winner — this is what makes the search itself correct for
    discretized metrics (a weight vector that looks best on raw OOF is not necessarily
    best after rounding).

    weight_search="dirichlet": `k` Dirichlet(1,...,1) draws plus the n unit-vectors and
    the uniform blend, then a refinement round of `k//3` draws concentrated around the
    coarse best (mirrors one competition's evaluator two-round dirichlet search, generalized to an
    arbitrary metric_fn instead of a hardcoded MAE proxy).
    weight_search="grid_simplex": exhaustive `grid_step`-spaced simplex grid, members<=5.

    Member OOFs may be 1-D `(n_samples,)` (the common single-target case) or 2-D
    `(n_samples, n_outputs)` (multiclass probabilities, or a multi-target regression like
    one competition's 5 soil properties). Anything else raises, naming the offending shape.

    Returns (best_weights: np.ndarray, best_score: float, oofs: np.ndarray[n_samples,
    n_members] for 1-D members / np.ndarray[n_samples, n_outputs, n_members] for 2-D
    ones — in both cases `oofs @ w` is the blended prediction)."""
    if len(members) < 2:
        raise ValueError(f"blend needs >=2 members, got {members!r}")
    member_oofs = [np.asarray(load_oof(cache_dir, m)) for m in members]
    shapes = {m: o.shape for m, o in zip(members, member_oofs)}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"blend members must all have the same OOF shape, got {shapes} "
                          f"-- refusing to blend mismatched members")
    ndim = member_oofs[0].ndim
    if ndim == 1:
        oofs = np.stack(member_oofs, axis=1)      # (n_samples, n_members)
    elif ndim == 2:
        # 2-D member OOFs stack on the LAST axis so `oofs @ w` still contracts over
        # MEMBERS and yields (n_samples, n_outputs). Stacking on axis=1 like the 1-D case
        # gives (n_samples, n_members, n_outputs), and `@ w` then contracts the OUTPUT axis
        # against the weight vector: a shape error when n_outputs != n_members and, when
        # they happen to be equal (one competition: 5 targets, and a 5-member blend is the obvious
        # thing to try), a silently wrong number that still looks like a score
        # (2026-08-03 audit).
        oofs = np.stack(member_oofs, axis=-1)     # (n_samples, n_outputs, n_members)
    else:
        raise ValueError(f"blend members must have 1-D (n_samples,) or 2-D "
                          f"(n_samples, n_outputs) OOFs; member #{members[0]} has shape "
                          f"{member_oofs[0].shape} ({ndim}-D) -- unsupported")
    n = len(members)

    if weight_search == "dirichlet":
        rng = np.random.default_rng(seed)
        candidates = [np.eye(n)[i] for i in range(n)]
        candidates.append(np.full(n, 1.0 / n))
        candidates += list(rng.dirichlet(np.ones(n), size=k))
        best_w, best_s = None, None
        for w in candidates:
            s = metric_fn(oofs @ w)
            if best_s is None or s < best_s:
                best_s, best_w = s, w
        conc = np.clip(best_w, 1e-3, None) * 200.0
        for w in rng.dirichlet(conc, size=max(k // 3, 50)):
            s = metric_fn(oofs @ w)
            if s < best_s:
                best_s, best_w = s, w
        return best_w, best_s, oofs
    elif weight_search == "grid_simplex":
        grid = _grid_simplex_weights(n, step=grid_step)
        best_w, best_s = None, None
        for w in grid:
            s = metric_fn(oofs @ w)
            if best_s is None or s < best_s:
                best_s, best_w = s, w
        return best_w, best_s, oofs
    elif weight_search == "nnls":
        # Advertised as first-class by the firing-class evaluators (and by the one competition NNLS
        # diagnostic) but never implemented here: every such proposal raised
        # "unknown weight_search method" (2026-08-03 audit). NNLS is a closed form on the
        # TARGET vector, which metric_fn only closes over -- so the caller must pass it.
        if target is None:
            raise ValueError(
                "weight_search='nnls' needs the target vector: call "
                "eval_blend(..., weight_search='nnls', target=y_true). It is the closed-form "
                "least-squares solution, so it cannot be derived from metric_fn alone.")
        from scipy.optimize import nnls as _nnls
        # 2-D members: solve the least-squares problem over every (sample, output) row at
        # once -- (n_samples, n_outputs, n_members) -> (n_samples*n_outputs, n_members),
        # with the target flattened the same C-order way, so the two stay aligned. A no-op
        # reshape for 1-D members (2026-08-03 audit).
        A = np.asarray(oofs, dtype=np.float64).reshape(-1, n)
        b = np.asarray(target, dtype=np.float64).reshape(-1)
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"weight_search='nnls': target has {b.shape[0]} values but the "
                              f"member OOFs have {A.shape[0]} rows (oofs shape "
                              f"{np.shape(oofs)}) -- shapes must match")
        w, _ = _nnls(A, b)
        w = np.ones(len(members)) / len(members) if w.sum() <= 0 else w / w.sum()
        return w, metric_fn(oofs @ w), oofs
    else:
        raise ValueError(f"unknown weight_search method {weight_search!r}")


# ---------------------------------------------------------------------------
# Recommendation 4: experience-library mutation prior
# ---------------------------------------------------------------------------
_HEADER_RE = re.compile(r"^(#{2,3})\s+(.*)$")


def _load_experience_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _split_sections(text: str):
    """Split markdown text into a flat list of {"level", "title", "lines"} dicts, one
    per `##`/`###` header; `lines` holds every `- ...` bullet appearing directly under
    that header (before the next `##`/`###` header of either level)."""
    sections = []
    cur = None
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            cur = {"level": len(m.group(1)), "title": m.group(2).strip(), "lines": []}
            sections.append(cur)
        elif cur is not None:
            stripped = line.strip()
            if stripped.startswith("- "):
                cur["lines"].append(stripped[2:].strip())
            elif stripped and cur["lines"] and not stripped.startswith(("#", "```")):
                # A continuation line belongs to the bullet above it. Dropping it used to
                # separate a multi-line bullet from its own 證據 tag, so the self-evidence
                # filter had nothing to match and the bullet was returned unfiltered
                # (2026-08-04 architecture gate). "|"-prefixed lines are NOT excluded: the
                # library's own write-back convention puts "  | 證據: ..." on its own
                # continuation line (see knowledge/vision_experience.md), and excluding the
                # pipe detached exactly the evidence the filter needed to see
                # (2026-08-07 re-verification).
                cur["lines"][-1] += " " + stripped
    return sections


# ---------------------------------------------------------------------------
# self-evidence filter (2026-07-30)
#
# Every library bullet carries its own `證據: <comp>, exp #N, scoreA -> scoreB` citation, and
# by July the library held the recorded benchmark runs' OWN answers -- aug-2022's final feature
# set and ensemble choice sit under a header its own metric matches. Re-running a competition
# with the library unfiltered therefore hands the agent the answer that competition already
# produced, and the score improves for reasons that have nothing to do with the method under
# test. Nothing filtered on the source competition before this: matching was on metric and tag
# headers only, and every bullet under a matching header was returned.
#
# The citation makes the fix cheap. Drop any bullet whose evidence field names the competition
# being solved, and report how many were dropped so the exclusion is visible rather than silent.
# ---------------------------------------------------------------------------
_EVIDENCE_RE = re.compile(r"證據[:：]\s*(.*)$")

# competition slug -> the short forms the library actually cites it by (collected from the file)
COMP_CITATION_ALIASES = {
    "tabular-playground-series-jan-2022": ["tpsjan22", "jan-2022", "jan2022"],
    "tabular-playground-series-aug-2022": ["tpsaug22", "aug-2022", "aug2022"],
    "tabular-playground-series-sep-2022": ["tpssep22", "sep-2022", "sep2022"],
    "afsis-soil-properties": ["afsis"],
    "conway-s-reverse-game-of-life": ["conway"],
    "cat-in-the-dat": ["citd"],
}


# Suffixes the infrastructure appends to a base competition slug when it derives a workspace:
# the headless launcher's repeat runs (".repeat-r1-<date>"), quarantined leftovers
# (".leftover_pre_run"), and make_v5_arm's arm workspaces ("-v5-<arm>"). A DERIVED workspace
# is still solving the BASE competition, so its aliases must be the base slug's aliases --
# otherwise a derived workspace name ("<slug>.repeat-r1-<date>") matches nothing the library cites and
# the self-evidence filter silently fails open on every repeat run
# (2026-08-07, bucket-A finding #49).
_WORKSPACE_SUFFIX_RE = re.compile(r"(\.(repeat|leftover)[\w-]*|-v5-[\w-]+)$")


def base_competition(comp: str) -> str:
    """Strip workspace-derivation suffixes down to the competition actually being solved."""
    prev = None
    while comp and comp != prev:
        prev = comp
        comp = _WORKSPACE_SUFFIX_RE.sub("", comp)
    return comp


def comp_aliases(comp: str) -> list:
    """Every token the experience library might cite `comp` by, longest first."""
    if not comp:
        return []
    comp = base_competition(comp)
    al = {comp}
    al.update(COMP_CITATION_ALIASES.get(comp, []))
    short = comp.replace("playground-series-", "").replace("tabular-", "")
    al.add(short)
    return sorted(al, key=len, reverse=True)


def _mentions_own_competition(text: str, aliases: list) -> bool:
    """Does ANY part of `text` name the competition being solved?

    Scoped to the whole bullet, not just the citation tail. The citation-only version had two
    holes: a bullet that named the competition in its BODY while citing a different one passed,
    and `_split_sections` only collects lines starting with '- ', so a multi-line bullet whose
    證據 sits on an indented continuation line had no citation at all and was never filtered
    (2026-08-04 architecture gate). Scanning the body is strictly safer: the cost of a false
    positive is one lost prior, the cost of a miss is an uninterpretable benchmark number.
    """
    if not aliases or not text:
        return False
    low = text.lower()
    for a in aliases:
        if re.search(rf"(?<![0-9a-z]){re.escape(a.lower())}(?![0-9a-z])", low):
            return True
    return False


def _cites_own_competition(line: str, aliases: list) -> bool:
    # Kept as the citation-scoped test for callers that need it; the retrieval path now uses
    # _mentions_own_competition, which also covers the body.
    m = _EVIDENCE_RE.search(line)
    if not m or not aliases:
        return False
    ev = m.group(1).lower()
    # Token-boundary match, not naive substring: 'one competition' is a substring of several competitions/
    # one competition and 'one competition' of one competition, so the old test excluded every sibling competition's evidence
    # from those lanes -- over-exclusion, silently starving them of transferable priors
    # (2026-08-03 audit). Boundary = anything that is not alphanumeric.
    for a in aliases:
        if re.search(rf"(?<![0-9a-z]){re.escape(a.lower())}(?![0-9a-z])", ev):
            return True
    return False


def suggest_priors(comp_meta: dict, experience_path: str = None, max_items: int = 20,
                   exclude_self: bool = True) -> list:
    """Simplified ERA idea-injection hook (recommendation #4) — NO LLM calls inside the
    harness, just keyword-to-section-header matching against knowledge/experience.md.

    `comp_meta` describes the target competition, e.g.
    `{"metric": "mae", "tags": ["small_sample", "duplicate_rows"]}`. Every string value
    under the `metric`/`tags`/`data_type`/`keywords` keys is lowercased and matched as a
    substring against every `##`/`###` section header in the experience library (e.g.
    metric="mae" matches header "MAE/整數目標", tag="qwk" matches "QWK/序數目標"). Every
    bullet line under a matching header is returned verbatim, in file order, deduplicated
    — each line already carries its own `... | 證據: comp, exp #N, scoreA -> scoreB`
    evidence citation from the source file, so the mutation proposer (the agent) sees the
    same evidence a human reading the doc would, with no summarization or paraphrasing
    happening inside the harness.

    Returns [] if comp_meta has no usable keywords or nothing in the library matches.
    """
    experience_path = experience_path or DEFAULT_EXPERIENCE_PATH
    text = _load_experience_text(experience_path)
    sections = _split_sections(text)

    keywords = []
    metric = comp_meta.get("metric")
    if metric:
        keywords.append(str(metric))
    for k in ("tags", "data_type", "keywords"):
        v = comp_meta.get(k)
        if v:
            keywords.extend([str(v)] if isinstance(v, str) else [str(x) for x in v])
    keywords_l = [kw.lower() for kw in keywords if kw]
    if not keywords_l:
        return []

    # comp_meta must name the competition for the filter to work; if it does not, say so loudly
    # rather than quietly returning contaminated priors.
    comp = (comp_meta.get("comp") or comp_meta.get("competition")
            or comp_meta.get("name") or "")
    aliases = comp_aliases(comp) if exclude_self else []
    if exclude_self and not comp:
        raise ValueError("suggest_priors: comp_meta needs a 'comp' key so bullets whose evidence "
                         "comes from that same competition can be excluded; pass "
                         "exclude_self=False only for a deliberately unfiltered read")

    out, seen, dropped = [], set(), []
    for sec in sections:
        title_l = sec["title"].lower()
        if any(kw in title_l for kw in keywords_l):
            for line in sec["lines"]:
                if line in seen:
                    continue
                seen.add(line)
                if _mentions_own_competition(line, aliases):
                    dropped.append(line)
                    continue
                out.append(line)
                if len(out) >= max_items:
                    break
        if len(out) >= max_items:
            break
    # HOW MUCH was withheld is itself the signal: the count varies with how much the
    # experience library recorded about this competition, and zero withheld is a signal too.
    # Round 10 gated the identical banner in query_library.py and missed this one, which is
    # the function it was copied FROM and the one SKILL.md advertises (round 11).
    if dropped and os.environ.get("KAGGLE_KB_AUDIT") == "1":
        print(f"[suggest_priors] excluded {len(dropped)} bullet(s) whose evidence comes from "
              f"{comp!r} itself (self-evidence filter); {len(out)} returned")
    return out
