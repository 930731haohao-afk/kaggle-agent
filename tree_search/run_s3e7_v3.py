"""tree_search/run_s3e7_v3.py — Phase F-2 harness_v3 validation run for
playground-series-s3e7 (Hotel Reservation Cancellation, ROC-AUC, maximize), retrofitted
in Phase H-2 to be the canonical example driver for `.claude/skills/*/references/
07_tree_search.md`. Re-runs the comp under harness_v3's DEFAULT automatic policy
(budget/phase machine, dedup-consumes-budget, reopen-blend-on-solo-breakthrough,
boundary_candidates, k=800+coordinate-ascent blend search, blend-cost guard) to verify
(a) no regression vs the harness_v2 sweep (tree_search/run_s3e7.py, best 0.900242 @
eval 13/22, experiments_tree.json — UNTOUCHED by this run) and (b) the v3 policy
machinery does real work end to end.

Phase H-2 update: this driver now goes through harness_v3's three Phase H-1 production-
readiness entry points instead of the lower-level pieces it originally used —
`hv3.save_search_state`/`hv3.load_search_state` (validated resume, replacing bare
`hv3.save`/`hv3.load` for THIS driver's own tree; the read-only OLD v2 tree still uses
plain `hv2.load`), `hv3.eval_solo_subprocess` (child-process-isolated solo evals,
replacing a direct `ev.evaluate(...)` call, so a hung native `fit()` gets hard-killed
instead of wedging the loop), and `hv3.apply_burst_seed_sanity_gate` (called on every
explore-burst seed right after it's evaluated). `DEDUP_REJECTIONS`/`COST_GUARD_FIRED`
also moved from module-level globals into `tree["search_state"]["driver_state"]` so they
survive a resume (see `_driver_state`/`_dedup_rejections`/`_cost_guard_fired` below). Run
`uv run python3 tree_search/run_s3e7_v3.py --dry-run` to verify this wiring end-to-end
with zero training compute and zero writes to TREE_PATH.

eval_s3e7.py (the per-comp evaluator) is reused byte-for-byte, unmodified — only this
DRIVER changes. Two deliberate driver-level deviations from run_s3e7.py's own
conventions, both scoped to this file:

  1. Node config STORAGE convention: every node in this v3 tree stores `core(proposal)`
     (drop "result"/"want_importance", sort blend member lists) as its `config` — never
     the evaluator's OUTPUT merged in. This is what makes harness_v3's own
     `find_duplicate_config`/`add_node` dedup (feature 2, "dedup consumes budget")
     meaningful out of the box, with no bespoke per-driver stripping wrapper needed on
     the harness side. Rich per-eval detail (blend weights, feature importances, etc.)
     is kept in a side dict, `tree["search_state"]["node_results"][str(node_id)]`, so it
     is persisted across resumes exactly like every other piece of search_state.
  2. Reuse-cached-OOF-where-digit-verified: before spending compute retraining a solo
     node, the driver checks whether `core(proposal)` byte-matches a node in the OLD
     (v2) tree; if so, it reloads that node's cached OOF from tree_search/cache_s3e7/
     (shared cache dir — eval_s3e7.py's CACHE_DIR is fixed, not driver-overridable),
     recomputes AUC from it fresh, and only accepts the reuse if that recomputed AUC
     matches the OLD tree's stored AUC to 6 decimals (the "digit-verified" bar the task
     brief calls for). This lets the v2-proven first-generation population (root + 6
     solo seeds + the 3-way BLEND seed, node ids 0-7 — confirmed identical ordering to
     experiments_tree.json) re-enter this v3 tree in ~seconds total instead of ~10
     minutes of retraining, leaving the real compute budget for genuinely NEW
     mutations (boundary-push, new seed-bags, a new middle-ground XGB, and the
     mandatory explore burst).

Blend nodes are NEVER reused from cache (they're already near-free — no retraining
happens for a blend either way) — they're evaluated fresh every time via this driver's
own `evaluate_blend_v3`, which calls harness_v3's `eval_blend_with_cost_guard`
(k=800 default + coordinate-ascent refinement, feature 5; wall-time cost guard,
feature 6) directly instead of going through eval_s3e7.evaluate_blend's v2-era
`hv2.eval_blend` call (k=1500, no ascent, no cost guard) — this is the one place a v3
run needs different plumbing than "just call eval_s3e7.evaluate()" to actually exercise
the two harness_v3 features that target ensemble search quality.

Root: BYTE-IDENTICAL to experiments_tree.json's root (Optuna fold-0-proxy-tuned LGB,
STATUS.md exp #4) — reused via the digit-verify path, and asserted equal to 0.899215
before anything else proceeds (the "verify root digit-for-digit" requirement).

boundary_candidates (feature 4): the root's own LGB params sit EXACTLY on
`optuna_lgb.py`'s own search-box lower edge for `max_depth` (searched [3,12], won at 3,
frac_from_low=0.0) — a real, unforced boundary-push opportunity, computed
programmatically (not hand-noticed) via `hv3.boundary_candidates(...)` against the box
verbatim-transcribed from tree_search/../scripts/optuna_lgb.py's `objective()`.
"""
import copy
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402 -- only for load/load_oof/cache_oof (OLD-tree reuse plumbing)
import harness_v3 as hv3  # noqa: E402
import eval_s3e7 as ev  # noqa: E402

COMP = "playground-series-s3e7"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
OLD_TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree.json")       # v2 tree -- READ-ONLY
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_v3.json")        # this run's output
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_s3e7.py")                 # Phase H-1 feature 8:
                                                                        # eval_solo_subprocess loads
                                                                        # the evaluator BY PATH, in a
                                                                        # child process
EVAL_TIMEOUT_S = 200
MAX_WALL_S = 35 * 60          # task's per-comp wall-time ceiling (soft safety net; should_stop
                               # is the intended stop mechanism, this just guards against a
                               # runaway loop)
ITER_SAFETY_CAP = 150         # generous vs. the 60-node hard budget cap

LINEAR_BEST = 0.899893
V2_TREE_BEST = 0.900242
V2_TREE_BEST_EVAL = 13
V2_TREE_N_EVAL = 22

# verbatim transcription of optuna_lgb.py's objective() search box (Phase B Round 1 --
# the box the root LGB_TUNED config was actually tuned inside)
LGB_SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.08, "log": True},
    "num_leaves": {"low": 15, "high": 255, "log": True},
    "max_depth": (3, 12),
    "min_child_samples": (5, 100),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": {"low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"low": 1e-3, "high": 10.0, "log": True},
}

# Phase H-1 feature 7 (resume-state contract): DEDUP_REJECTIONS and COST_GUARD_FIRED used
# to be plain module-level lists -- exactly the class of bug H-1 fixes (F-2's own s3e7 run
# lost driver state across a resume because nothing put it inside the persisted tree). Both
# now live in tree["search_state"]["driver_state"] (see _driver_state()/_dedup_rejections()/
# _cost_guard_fired() below) so they survive a process restart via save_search_state /
# load_search_state, with zero custom reconstruction code. LINEAGE_NAMES and BURST_INJECTED
# keep their existing resume-safety design (rebuilt FROM the tree's own node mutations on
# resume, see main()) -- that self-healing approach is at least as robust as persisting them
# and is left as-is. _dedup_offset (a pure indexing convenience into per-lineage mutation
# queues) is NOT migrated: losing it across a resume costs at most one extra dedup-rejected
# proposal attempt for the affected lineage (self-correcting, not a correctness bug), so the
# added complexity of persisting an int-keyed dict isn't justified here.
_dedup_offset = {}
BURST_INJECTED = [False]
BOUNDARY_LOG = []


def _driver_state(tree):
    """Phase H-1 feature 7's reserved free-form spot for driver-local bookkeeping that must
    survive a restart -- see harness_v3.save_search_state's own docstring."""
    return tree.setdefault("search_state", {}).setdefault("driver_state", {})


def _dedup_rejections(tree):
    return _driver_state(tree).setdefault("dedup_rejections", [])


def _cost_guard_fired(tree):
    return _driver_state(tree).setdefault("cost_guard_fired", [])


def dc(cfg):
    return copy.deepcopy(cfg)


def _node_results(tree):
    """Side-channel result store, persisted INSIDE the tree (search_state) so it
    survives resume/reload -- a plain module-level dict would silently lose data (e.g.
    root's feature importance) across a resumed run, since only the tree itself is
    saved to disk between process invocations."""
    return tree.setdefault("search_state", {}).setdefault("node_results", {})


def result_of(tree, nid):
    return _node_results(tree).get(str(nid))


def core(cfg):
    """Canonical, hashable, STORED form of a node config: drop eval-output/bookkeeping
    keys ("result", "want_importance"), sort blend member lists (order-insensitive same-
    member-SET dedup). Every node's `config` field in this tree IS `core(proposal)` --
    see module docstring point 1."""
    out = {k: v for k, v in cfg.items() if k not in ("result", "want_importance")}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


# ---------------------------------------------------------------------------
# OLD (v2) tree -- read-only, used only to look up reusable cached OOFs
# ---------------------------------------------------------------------------
OLD = hv2.load(OLD_TREE_PATH)
OLD_BY_HASH = {}
for _n in OLD["nodes"]:
    if _n["status"] == "evaluated":
        OLD_BY_HASH.setdefault(hv3.config_hash(core(_n["config"])), _n)


def try_reuse(stored_cfg):
    """If stored_cfg (already core()-normalized, kind=='solo') matches a node in the OLD
    v2 tree, reload its cached OOF and digit-verify (recompute AUC from the OOF, compare
    to the OLD tree's own stored AUC to 6 decimals). Returns (oof, auc, old_node) on a
    verified hit, else None -- caller falls back to a real eval."""
    if stored_cfg.get("kind") != "solo":
        return None
    old = OLD_BY_HASH.get(hv3.config_hash(stored_cfg))
    if old is None:
        return None
    try:
        oof = hv2.load_oof(ev.CACHE_DIR, old["id"])
    except ValueError:
        return None
    recomputed = ev.auc(ev._y, oof)
    old_auc = (old["config"].get("result") or {}).get("auc")
    if old_auc is None or round(recomputed, 6) != round(old_auc, 6):
        return None
    return oof, recomputed, old


# ---------------------------------------------------------------------------
# v3 blend evaluation: bypasses eval_s3e7.evaluate_blend's v2-era hv2.eval_blend call,
# uses harness_v3's k=800+coordinate-ascent + cost-guard machinery directly (module
# docstring, features 5+6).
# ---------------------------------------------------------------------------
def _neg_auc(vec):
    return -ev.auc(ev._y, vec)


def _weight_search_generic(oofs, metric_fn, k=hv3.DEFAULT_BLEND_K, seed=42):
    """Same coarse-Dirichlet + concentrated-refinement search as harness_v2.eval_blend,
    generalized to operate on an already-materialized oofs matrix (needed for rank-space
    blends, whose oofs are rank-TRANSFORMED, not the raw cached arrays load_oof returns)
    -- followed by harness_v3's coordinate-ascent refinement (feature 5)."""
    n = oofs.shape[1]
    rng = np.random.default_rng(seed)
    candidates = [np.eye(n)[i] for i in range(n)]
    candidates.append(np.full(n, 1.0 / n))
    candidates += list(rng.dirichlet(np.ones(n), size=k))
    best_w = best_s = None
    for w in candidates:
        s = metric_fn(oofs @ w)
        if best_s is None or s < best_s:
            best_s, best_w = s, w
    conc = np.clip(best_w, 1e-3, None) * 200.0
    for w in rng.dirichlet(conc, size=max(k // 3, 50)):
        s = metric_fn(oofs @ w)
        if s < best_s:
            best_s, best_w = s, w
    best_w, best_s = hv3._coordinate_ascent_refine(oofs, metric_fn, best_w, best_s)
    return best_w, best_s


def evaluate_blend_v3(tree, stored_cfg):
    members = stored_cfg["members"]  # already sorted by core()
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    space = stored_cfg.get("space", "prob")
    t0 = time.time()
    if space == "prob":
        best_w, best_neg, oofs, warning = hv3.eval_blend_with_cost_guard(
            ev.CACHE_DIR, members, _neg_auc, tree=tree)
        best_score = -best_neg
    elif space == "rank":
        from scipy.stats import rankdata
        raw = np.stack([hv2.load_oof(ev.CACHE_DIR, m) for m in members], axis=1)
        n = len(ev._y)
        oofs = np.stack([rankdata(raw[:, i]) / n for i in range(raw.shape[1])], axis=1)
        best_w, best_neg = _weight_search_generic(oofs, _neg_auc)
        best_score = -best_neg
        warning = None
    else:
        raise ValueError(f"unknown blend space {space!r}")
    wall = time.time() - t0
    if warning:
        _cost_guard_fired(tree).append(warning)
    result = dict(members=members, weights=[round(float(w), 4) for w in best_w],
                  method="dirichlet", space=space, auc=round(best_score, 6))
    return result, best_score, wall, warning


# ---------------------------------------------------------------------------
# eval_and_add: the single per-node entry point. Handles dedup pre-check, reuse-from-OLD
# for solo nodes, v3 blend evaluation for blend nodes, and always routes the final
# add through hv3.add_root/hv3.add_node (so update_phase / reopen-blend-on-breakthrough
# / dedup-consumes-budget all run automatically).
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, proposal_cfg, is_root=False, lineage_id=None):
    stored = core(proposal_cfg)
    if not is_root:
        dup_id = hv3.find_duplicate_config(tree, stored)
        if dup_id is not None:
            _dedup_rejections(tree).append(dict(mutation=mutation, dup_id=dup_id))
            if lineage_id is not None:
                _dedup_offset[lineage_id] = _dedup_offset.get(lineage_id, 0) + 1
            # route through hv3.add_node anyway so its own dedup-consumes-budget
            # bookkeeping (feature 2) sees every rejection, even ones this driver's own
            # find_dup already caught -- defense in depth, matches the harness contract.
            nid_null, dup_echo = hv3.add_node(tree, parent_id, mutation + " [dedup pre-check]",
                                               stored, None, "failed", 0.0)
            assert nid_null is None
            hv3.save_search_state(tree, TREE_PATH)  # Phase H-1 feature 7
            return None, dup_id, None

    nid = hv3.next_id(tree)
    kind = stored.get("kind", "solo")

    if kind == "solo":
        reused = try_reuse(stored)
        if reused is not None:
            oof, auc_val, old = reused
            hv2.cache_oof(ev.CACHE_DIR, nid, oof, auc=auc_val)
            score, status, wall_s = round(-auc_val, 6), "evaluated", 0.05
            result = dict(old["config"].get("result") or {})
            result["auc"] = round(auc_val, 6)
            full_mutation = (mutation + f" [REUSED cached OOF from prior v2 tree node "
                              f"#{old['id']}, digit-verified AUC {auc_val:.6f} == "
                              f"{result.get('auc')} to 6dp, no retraining]")
        else:
            # Phase H-1 feature 8: run the solo eval in a child process so a hung native
            # fit() (F-2's s3e7 CatBoost 28-minute hang) gets hard-killed instead of
            # wedging the whole search loop -- see EVAL_MODULE_PATH above.
            r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, proposal_cfg, EVAL_TIMEOUT_S, node_id=nid)
            score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
            full_mutation = mutation
    elif kind == "blend":
        try:
            result, best_score, wall_s, warning = evaluate_blend_v3(tree, stored)
            score, status = round(-best_score, 6), "evaluated"
            full_mutation = mutation if not warning else mutation + f" [{warning}]"
        except Exception as e:  # noqa: BLE001 -- keep the search loop alive on a bad blend
            score, status, wall_s, result = None, "failed", 0.0, None
            full_mutation = mutation + f" [ERROR: {type(e).__name__}: {e}]"
    else:
        raise ValueError(f"unknown node kind {kind!r}")

    if is_root:
        real_nid = hv3.add_root(tree, full_mutation, stored, score, status, wall_s)
    else:
        real_nid, dup = hv3.add_node(tree, parent_id, full_mutation, stored, score, status, wall_s)
        assert dup is None, f"unexpected post-eval dedup for a pre-checked-unique config (dup=#{dup})"
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    if result is not None:
        _node_results(tree)[str(real_nid)] = result
    hv3.save_search_state(tree, TREE_PATH)  # Phase H-1 feature 7
    return real_nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


def auc_of(r):
    if r is None:
        return None
    if r.get("result") and "auc" in r["result"]:
        return r["result"]["auc"]
    return -r["score"] if r.get("score") is not None else None


# ---------------------------------------------------------------------------
# Root + first-generation seeds -- BYTE-IDENTICAL to run_s3e7.py's (reused via
# try_reuse wherever the OLD tree has a matching, digit-verified cached OOF).
# ---------------------------------------------------------------------------
LGB_TUNED_PARAMS = dict(
    objective="binary", metric="auc", n_estimators=3000,
    learning_rate=0.06811817027632358, num_leaves=178, max_depth=3,
    min_child_samples=47, subsample=0.8888159684496226, subsample_freq=1,
    colsample_bytree=0.5316042232727975, reg_alpha=2.142043500176781,
    reg_lambda=0.011502513321845967, random_state=42, early_stopping_rounds=150,
)
ROOT_CONFIG = {"kind": "solo", "model": "lgb", "params": dc(LGB_TUNED_PARAMS),
               "features": {"drop": []}, "want_importance": True}
ROOT_MUTATION = ("root: s3e7 linear-winner's strongest solo config -- Optuna fold-0-"
                 "proxy-tuned LGB (50 trials, STATUS.md exp #4), byte-identical to "
                 "experiments_tree.json's root (v2 sweep) -- verified digit-for-digit")

XGB_HAND = dict(objective="binary:logistic", n_estimators=3000, learning_rate=0.03,
                max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42,
                eval_metric="auc", early_stopping_rounds=150)
CAT_HAND = dict(loss_function="Logloss", eval_metric="AUC", iterations=4000,
                learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                random_seed=42, thread_count=-1, early_stopping_rounds=200)


def seed_xgbhand():
    return ({"kind": "solo", "model": "xgb", "params": dc(XGB_HAND), "features": {"drop": []}},
            "model-type: hand-set XGB, byte-identical to v2's XGBHAND seed")


def seed_cathand():
    return ({"kind": "solo", "model": "cat", "params": dc(CAT_HAND), "features": {"drop": []}},
            "model-type: hand-set CatBoost, byte-identical to v2's CATHAND seed")


def seed_seedbag():
    p = dc(LGB_TUNED_PARAMS)
    p["random_state"] = 2024
    return ({"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}},
            "seed variation: LGB_tuned, random_state 42->2024, byte-identical to v2's SEEDBAG seed")


def seed_regnudge():
    p = dc(LGB_TUNED_PARAMS)
    p["reg_alpha"] = 4.0
    p["num_leaves"] = 90
    return ({"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}},
            "regularization nudge, byte-identical to v2's REGNUDGE seed")


def seed_xgbdiv():
    p = dc(XGB_HAND)
    p["max_depth"] = 8
    p["colsample_bytree"] = 0.6
    p["learning_rate"] = 0.05
    return ({"kind": "solo", "model": "xgb", "params": p, "features": {"drop": []}},
            "model-variant: XGB pushed deeper/higher-lr, byte-identical to v2's XGBDIV seed")


def seed_featprune(weak_feats):
    cfg = {"kind": "solo", "model": "lgb", "params": dc(LGB_TUNED_PARAMS),
           "features": {"drop": list(weak_feats)}}
    return cfg, (f"feature-PRUNING: drop the {len(weak_feats)} lowest-gain-importance "
                 f"engineered features ({', '.join(weak_feats)}), byte-identical to v2's "
                 f"FEATPRUNE seed if importances reproduce")


def seed_boundarypush():
    edges = hv3.boundary_candidates({"params": LGB_TUNED_PARAMS}, LGB_SEARCH_SPACE)
    if not edges:
        return None  # nothing on an edge -- skip this lineage (shouldn't happen here)
    p = dc(LGB_TUNED_PARAMS)
    descs = []
    for e in edges:
        p[e["param"]] = e["new_value"]
        descs.append(f"{e['param']} {e['old_value']}->{e['new_value']} (edge={e['edge']})")
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = (f"model-variant: harness_v3.boundary_candidates()-flagged push(es) past the "
            f"root's Optuna search-box edge(s) -- {', '.join(descs)} -- (feature 4; "
            f"D-6/E-2/E-3 pattern, this comp's own genuine edge-hit, not forced)")
    return cfg, desc


def seed_xgbtuned():
    p = dict(objective="binary:logistic", n_estimators=3000, learning_rate=0.04,
             max_depth=5, min_child_weight=10, subsample=0.75, colsample_bytree=0.75,
             reg_alpha=1.0, reg_lambda=3.0, random_state=42, eval_metric="auc",
             early_stopping_rounds=150)
    cfg = {"kind": "solo", "model": "xgb", "params": p, "features": {"drop": []}}
    desc = ("model-variant: a third, disciplined middle-ground XGB (depth5/lr0.04/"
            "mcw10), distinct from XGBHAND's shallow hand-set and XGBDIV's deliberate-"
            "deep diversity direction -- added to the pool [PRIOR P9 -- add-to-pool, "
            "never replace]")
    return cfg, desc


SOLO_SEED_SPECS = [("XGBHAND", seed_xgbhand), ("CATHAND", seed_cathand),
                    ("SEEDBAG", seed_seedbag), ("REGNUDGE", seed_regnudge),
                    ("XGBDIV", seed_xgbdiv)]
NEW_SEED_SPECS = [("BOUNDARYPUSH", seed_boundarypush), ("XGBTUNED", seed_xgbtuned)]

ALL_LINEAGE_NAMES = ([n for n, _ in SOLO_SEED_SPECS] + ["FEATPRUNE", "BLEND"] +
                      [n for n, _ in NEW_SEED_SPECS])
LINEAGE_NAMES = {}


def _lineage_seed_ids(tree):
    out = {}
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name in ALL_LINEAGE_NAMES:
                if name not in out and n["mutation"].startswith(f"[{name}]"):
                    out[name] = n["id"]
    return out


def _best_node_in_lineage(tree, lineage_id):
    best = None
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["id"] == tree["root_id"]:
            continue
        if hv3.lineage_of(tree, n["id"]) == lineage_id:
            if best is None or n["score"] < best["score"]:
                best = n
    return best


def _id_for(tree, name):
    if name == "ROOT":
        return tree["root_id"]
    lineage_id = _lineage_seed_ids(tree)[name]
    best = _best_node_in_lineage(tree, lineage_id)
    return best["id"] if best else lineage_id


def solo_pool(tree):
    nodes = [n for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


# ---------------------------------------------------------------------------
# NEW per-lineage mutation queues -- each ONE genuinely-untried-by-v2 item, then falls
# back to a seed-variation (never replays v2's own already-explored, already-losing
# queue items -- no new information in that, and the budget is better spent on new
# ground plus the mandatory explore burst).
# ---------------------------------------------------------------------------
def _bump(cfg, **kw):
    c = dc(cfg)
    c["params"].update(kw)
    return c


XGBHAND_QUEUE = [
    lambda c: (_bump(c, gamma=0.1, colsample_bylevel=0.8),
               "XGBHAND: NEW -- gamma=0.1 (min split-loss) + colsample_bylevel=0.8, an "
               "axis v2's queue never probed [PRIOR none -- fresh regularization axis]"),
]
CATHAND_QUEUE = [
    lambda c: (_bump(c, bagging_temperature=1.0),
               "CATHAND: NEW -- bagging_temperature=1.0 (Bayesian-bootstrap row "
               "weighting), the same untested CatBoost axis eval_s3e14's CAT_QUEUE "
               "probed on a different comp [PRIOR none -- cross-comp transfer test]"),
]
SEEDBAG_QUEUE = [
    lambda c: (_bump(c, random_state=555),
               "SEEDBAG: NEW -- third independent seed (555; v2's own queue only tried "
               "777/reg_lambda push) [PRIOR P2 -- add-to-pool + seed-bag]"),
]
REGNUDGE_QUEUE = [
    lambda c: (_bump(c, min_child_samples=120),
               "REGNUDGE: NEW -- min_child_samples 47->120, further leaf-size "
               "regularization than v2's queue reached (80) [PRIOR P6]"),
]
XGBDIV_QUEUE = [
    lambda c: (_bump(c, subsample=0.6, learning_rate=0.06),
               "XGBDIV: NEW -- subsample 0.8->0.6 + lr 0.05->0.06, a combo v2's queue "
               "didn't try [PRIOR none -- this run's own further-diversity direction]"),
]
BOUNDARYPUSH_QUEUE = []   # v2 never had this lineage; let solo_fallback seed-vary it
XGBTUNED_QUEUE = []       # ditto


def _featprune_queue_fn(weak_feats):
    target = weak_feats[1] if len(weak_feats) > 1 else weak_feats[0]

    def fn(c):
        child = dc(c)
        child["features"] = {"drop": [target]}
        return child, (f"FEATPRUNE: NEW -- isolate the OTHER weakest engineered feature "
                        f"({target}) alone (v2 only isolated {weak_feats[0]}) [PRIOR P3]")
    return fn


SOLO_QUEUES = {"XGBHAND": XGBHAND_QUEUE, "CATHAND": CATHAND_QUEUE, "SEEDBAG": SEEDBAG_QUEUE,
               "REGNUDGE": REGNUDGE_QUEUE, "XGBDIV": XGBDIV_QUEUE,
               "BOUNDARYPUSH": BOUNDARYPUSH_QUEUE, "XGBTUNED": XGBTUNED_QUEUE}
# FEATPRUNE filled in main() once weak_feats is known


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    seed_key = "random_state" if c["model"] in ("lgb", "xgb") else "random_seed"
    c["params"][seed_key] = 4000 + attempt
    return c, f"fallback seed-variation: alt {seed_key}={4000 + attempt}"


# ---------------------------------------------------------------------------
# BLEND lineage
# ---------------------------------------------------------------------------
def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = sorted([tree["root_id"], ids["XGBHAND"], ids["CATHAND"]])
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet", "space": "prob"}
    desc = (f"ensemble seed: 3-way blend of root(#{tree['root_id']}) + XGBHAND(#{ids['XGBHAND']}) "
            f"+ CATHAND(#{ids['CATHAND']}) -- byte-identical member set to v2's BLEND seed")
    return cfg, desc


def _mk_add_named(name, prior_tag):
    def fn(tree, parent_node):
        add_id = _id_for(tree, name)
        members = parent_node["config"]["members"]
        if add_id in members:
            return None
        child = {"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": "dirichlet", "space": parent_node["config"].get("space", "prob")}
        desc = f"BLEND: add {name} solo member (#{add_id}) -> {len(child['members'])}-way {prior_tag}"
        return child, desc
    return fn


def _blend_space_swap(tree, parent_node):
    cur = parent_node["config"].get("space", "prob")
    child = dc(parent_node["config"])
    child["space"] = "rank" if cur == "prob" else "prob"
    desc = f"BLEND: space swap {cur}->{child['space']} on the same members [PRIOR P18]"
    return child, desc


def _blend_remove_weakest(tree, parent_node):
    res = result_of(tree, parent_node["id"]) or {}
    weights = res.get("weights")
    members = parent_node["config"]["members"]
    if not weights or len(members) <= 2:
        return None
    idx = int(np.argmin(weights))
    weak_id = members[idx]
    child = dc(parent_node["config"])
    child["members"] = sorted([m for m in members if m != weak_id])
    desc = f"BLEND: remove lowest-weight member #{weak_id} (w={weights[idx]:.3f}) [PRIOR P16]"
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("SEEDBAG", "[PRIOR P2 -- add-to-pool + seed-bag]"),
    _mk_add_named("BOUNDARYPUSH", "[feature 4 -- test whether the boundary-pushed solo "
                                   "helps the blend regardless of its own solo score]"),
    _mk_add_named("XGBTUNED", "[PRIOR P9 -- add-to-pool, never replace]"),
    _blend_space_swap,
    _blend_remove_weakest,
]


def blend_fallback(tree, parent_node, attempt):
    pool = solo_pool(tree)
    members = parent_node["config"]["members"]
    for add_id in [nid for nid in pool if nid not in members]:
        child = {"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": "dirichlet", "space": parent_node["config"].get("space", "prob")}
        if hv3.find_duplicate_config(tree, child) is None:
            return child, f"BLEND fallback: add next-best unused pool member #{add_id}"
    cur = parent_node["config"].get("space", "prob")
    child = dc(parent_node["config"])
    child["space"] = "rank" if cur == "prob" else "prob"
    if hv3.find_duplicate_config(tree, child) is None:
        return child, f"BLEND fallback: space swap {cur}->{child['space']}"
    return None


def _lineage_kind(tree, lineage_id):
    """The kind ("solo"/"blend") of lineage_id's own first-gen node -- every descendant
    in a lineage shares its kind, so this is the reliable way to dispatch a lineage
    that has NO authored queue at all (e.g. EXPL_MEGABLEND, injected by the explore
    burst with no BLEND_QUEUE entry) -- checking `name == "BLEND"` literally would miss
    any other blend-kind lineage and KeyError on SOLO_QUEUES[name]."""
    node = next(n for n in tree["nodes"] if n["id"] == lineage_id)
    return node["config"].get("kind", "solo")


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES[lineage_id]
    idx = hv3.lineage_size(tree, lineage_id) - 1 + _dedup_offset.get(lineage_id, 0)
    if _lineage_kind(tree, lineage_id) == "blend":
        result = None
        if name == "BLEND" and idx < len(BLEND_QUEUE):
            result = BLEND_QUEUE[idx](tree, parent_node)
        if result is None:
            offset = idx - len(BLEND_QUEUE) if name == "BLEND" else idx
            result = blend_fallback(tree, parent_node, max(offset, 0))
        if result is None:
            return None, None
        child_cfg, desc = result
    else:
        queue = SOLO_QUEUES.get(name, [])
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


# ---------------------------------------------------------------------------
# Explore burst (feature 1): injected once the phase machine flips to "explore_burst".
# ---------------------------------------------------------------------------
def _burst_seeds():
    return [
        ("EXPL_DART", {"kind": "solo", "model": "lgb",
                        "params": dict(LGB_TUNED_PARAMS, boosting_type="dart",
                                       drop_rate=0.1, max_drop=50, n_estimators=1200,
                                       early_stopping_rounds=1200),  # dart doesn't improve
                                       # monotonically fold-to-fold, so early stopping is
                                       # set to never realistically trigger (esr==n_estimators)
                                       # rather than disabled outright (lgb.early_stopping()
                                       # requires an int, None raises TypeError)
                        "features": {"drop": []}},
         "long-shot: DART boosting mode (never tried in v2), alternative regularization "
         "via dropped trees -- n_estimators trimmed (no early stopping under dart)"),
        ("EXPL_XT", {"kind": "solo", "model": "lgb",
                      "params": dict(LGB_TUNED_PARAMS, extra_trees=True),
                      "features": {"drop": []}},
         "long-shot: extra_trees=True (extremely-randomized split thresholds), pure "
         "diversity via randomization instead of regularization"),
        ("EXPL_CATDEEP", {"kind": "solo", "model": "cat",
                           "params": dict(CAT_HAND, depth=9, bagging_temperature=2.0),
                           "features": {"drop": []}},
         "long-shot: CatBoost pushed deeper (7->9) + Bayesian-bootstrap bagging_temp=2.0 "
         "-- untested capacity+regularization combo"),
        ("EXPL_XGBLEAF", {"kind": "solo", "model": "xgb",
                           "params": dict(XGB_HAND, tree_method="hist",
                                          grow_policy="lossguide", max_leaves=64),
                           "features": {"drop": []}},
         "long-shot: leaf-wise-growth XGB (grow_policy=lossguide), structurally closer "
         "to LGB's own splitting strategy via a different implementation"),
        ("EXPL_BOUND2", {"kind": "solo", "model": "lgb",
                          "params": dict(LGB_TUNED_PARAMS, max_depth=2, learning_rate=0.1),
                          "features": {"drop": []}},
         "long-shot: paired boundary-push+compensation -- max_depth 3->2 (one step "
         "further past the box edge than EXPL/BOUNDARYPUSH) WITH learning_rate raised "
         "0.068->0.1 to compensate for the shallower trees' reduced per-tree capacity"),
    ]


BURST_NAMES = [name for name, _, _ in _burst_seeds()] + ["EXPL_MEGABLEND"]
ALL_LINEAGE_NAMES.extend(BURST_NAMES)  # so resume's LINEAGE_NAMES-rebuild loop and
                                        # propose_child's SOLO_QUEUES.get(name, []) both
                                        # recognize burst-injected lineages after a restart


def _sanity_check_burst_seed(tree, nid, name):
    """Phase H-1 feature 9: immediately after a fresh explore-burst long-shot seed is
    evaluated, check it against the sanity band before letting its lineage spawn any
    children. On failure the lineage is marked plateaued right here (one burned seed
    node, not a whole wasted mutation queue) -- the exact fix for F-2's s3e14 lesson
    (a DART seed at ~18x the root/global-best gap that ate ~12 minutes / 6 evals of
    further children before the run gave up on it)."""
    passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
    if not passed:
        print(f"[BURST {name}] #{nid} FAILED sanity gate (bound={bound}) -- lineage "
              f"plateaued immediately, no children will be spawned from it")
    return passed


def inject_explore_burst(tree, root_id):
    seeds = _burst_seeds()
    injected_ids = []
    for name, cfg, desc in seeds:
        nid, dup, r = eval_and_add(tree, root_id, f"[{name}] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            SOLO_QUEUES[name] = []
            injected_ids.append(nid)
            print(f"[BURST {name}] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")
            _sanity_check_burst_seed(tree, nid, name)
    # 6th long-shot: a mega-blend of the FULL solo pool at burst-start time (real test of
    # feature 5's k=800+ascent search + feature 6's cost guard at larger member count)
    pool = solo_pool(tree)
    if len(pool) >= 2:
        cfg = {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet", "space": "prob"}
        nid, dup, r = eval_and_add(tree, root_id, "[EXPL_MEGABLEND] long-shot: kitchen-sink "
                                    f"blend of the entire {len(pool)}-member solo pool via v3's "
                                    "k=800+coordinate-ascent search + cost guard "
                                    "[feature 5/6 -- direct test at scale]", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "EXPL_MEGABLEND"
            injected_ids.append(nid)
            _sanity_check_burst_seed(tree, nid, "EXPL_MEGABLEND")
            print(f"[BURST EXPL_MEGABLEND] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")
    return injected_ids


# ---------------------------------------------------------------------------
_PRIOR_TAG_RE = re.compile(r"\[PRIOR (P\d+|none)")


def prior_usage_summary(tree):
    by_id = {n["id"]: n for n in tree["nodes"]}
    informed, uninformed = [], []
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["parent_id"] is None or n["parent_id"] == tree["root_id"]:
            continue
        m = _PRIOR_TAG_RE.search(n["mutation"])
        if not m:
            continue
        tag = m.group(1)
        parent = by_id.get(n["parent_id"])
        if parent is None or parent["score"] is None or n["score"] is None:
            continue
        won = n["score"] < parent["score"]
        (informed if tag != "none" else uninformed).append(dict(node_id=n["id"], tag=tag, won=won))

    def rate(lst):
        return (sum(1 for x in lst if x["won"]) / len(lst)) if lst else None
    return dict(informed=informed, uninformed=uninformed,
                informed_win_rate=rate(informed), uninformed_win_rate=rate(uninformed))


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        # Phase H-1 feature 7: load_search_state validates search_state consistency
        # before handing the tree back, so a corrupt resume fails loudly here instead of
        # a mysterious KeyError several iterations into the loop below.
        tree = hv3.load_search_state(TREE_PATH)
        print(f"Resuming existing tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv3.new_tree(COMP)
        hv3.init_budget(tree)   # explicit -- v3's own caveat: budget state only auto-inits
                                # on the first hv3.add_node() call, which for a from-scratch
                                # tree happens after root+several seeds; init upfront for clarity.
        print(f"init_budget -> {tree['search_state']['budget']}")

    def n_evaluated():
        return hv3.n_evaluated(tree)

    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} AUC={auc_of(r)} wall_s={r['wall_s']} status={r['status']}")
        assert r["status"] == "evaluated"
        assert round(auc_of(r), 6) == 0.899215, (
            f"ROOT DIGIT-VERIFY FAILED: expected 0.899215 (experiments_tree.json node #0), "
            f"got {auc_of(r)}")
        print("ROOT DIGIT-VERIFIED against experiments_tree.json node #0 (0.899215). OK.")

    # boundary_candidates is deterministic on (root params, search box) -- recompute it
    # every run (fresh AND resumed) so the final tree save never records an empty log
    # just because the seeding step ran in an earlier process (module state is not
    # persisted across resumes; the first full run hit exactly that reporting bug).
    if not BOUNDARY_LOG:
        BOUNDARY_LOG.extend(hv3.boundary_candidates({"params": LGB_TUNED_PARAMS}, LGB_SEARCH_SPACE))
    print(f"boundary_candidates(root LGB_TUNED vs its Optuna box) -> {BOUNDARY_LOG}")

    root_node = tree["nodes"][tree["root_id"]]
    importance = (result_of(tree, tree["root_id"]) or {}).get("importance") or {}
    eng_importance = {k: v for k, v in importance.items() if k in ev.ENGINEERED_FEATURES}
    ranked_weak = sorted(eng_importance, key=lambda k: eng_importance[k]) if eng_importance else []
    print(f"Engineered-feature importance (ascending): {[(k, eng_importance[k]) for k in ranked_weak]}")
    weak2 = ranked_weak[:2] if len(ranked_weak) >= 2 else ranked_weak

    SOLO_QUEUES["FEATPRUNE"] = [_featprune_queue_fn(weak2)] if weak2 else []

    def seed_featprune_fn():
        return seed_featprune(weak2)

    all_named_seeds = list(SOLO_SEED_SPECS) + [("FEATPRUNE", seed_featprune_fn)] + list(NEW_SEED_SPECS)
    for name, seed_fn in all_named_seeds:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        seeded = seed_fn()
        if seeded is None:
            continue
        cfg, desc = seeded
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")

    already_blend = any(n["mutation"].startswith("[BLEND]") for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")

    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in ALL_LINEAGE_NAMES:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name
                    if name in BURST_NAMES:
                        SOLO_QUEUES.setdefault(name, [])

    # resume-safety: BURST_INJECTED is module-level (not persisted) -- infer it from the
    # tree itself so a resumed process never re-injects the burst (which would just
    # dedup-reject every seed, harmless but wasteful, and is the wrong thing to rely on).
    if any(n["mutation"].startswith("[EXPL_") for n in tree["nodes"] if n["parent_id"] == tree["root_id"]):
        BURST_INJECTED[0] = True

    # --- v3 phase-machine-driven adaptive tree-search loop ---
    iterations = 0
    while (not hv3.should_stop(tree)) and n_evaluated() < hv3.init_budget(tree)["total_budget"] \
            and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > ITER_SAFETY_CAP:
            print("Safety cap on iterations reached, stopping.")
            break

        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not BURST_INJECTED[0]:
            BURST_INJECTED[0] = True
            print(f"\n>>> PHASE -> explore_burst (n_eval={n_evaluated()}); injecting "
                  f"{len(_burst_seeds())+1} fresh long-shot lineages per feature 1 <<<\n")
            inject_explore_burst(tree, tree["root_id"])
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted. Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        if child_cfg is None:
            st = tree["search_state"]
            if lineage_id not in st["plateaued"]:
                st["plateaued"].append(lineage_id)
                st["backtrack_log"].append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=f"lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)}'s mutation "
                           f"space exhausted -> forced backtrack"))
            hv3.save_search_state(tree, TREE_PATH)  # Phase H-1 feature 7
            print(f"FORCED BACKTRACK: lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} exhausted; "
                  f"plateaued={tree['search_state']['plateaued']}")
            continue
        nid, dup, r = eval_and_add(tree, parent_id, mutation, child_cfg, lineage_id=lineage_id)
        if nid is None:
            print(f"DEDUP REJECTED <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"(dup of #{dup})")
            continue
        gb = hv3.global_best(tree)
        gb_auc = -gb["score"] if gb else None
        budget = tree["search_state"]["budget"]
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_AUC={gb_auc} (#{gb['id'] if gb else '-'}) "
              f"| phase={budget['phase']} n_eval={n_evaluated()} "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = hv3.global_best(tree)
    gb_auc = -gb["score"] if gb else None
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    prior_stats = prior_usage_summary(tree)
    tree["prior_usage_summary"] = {
        "n_informed": len(prior_stats["informed"]), "n_uninformed": len(prior_stats["uninformed"]),
        "informed_win_rate": prior_stats["informed_win_rate"],
        "uninformed_win_rate": prior_stats["uninformed_win_rate"],
    }
    # top-level mirrors for docs/scripts/build_tree_facts.py, sourced from the
    # resume-durable tree["search_state"]["driver_state"] lists (Phase H-1 feature 7)
    # rather than module globals that would be empty after a mid-run process restart.
    dedup_rejections = _dedup_rejections(tree)
    cost_guard_fired = _cost_guard_fired(tree)
    tree["dedup_rejections"] = dedup_rejections
    tree["boundary_candidates_log"] = BOUNDARY_LOG
    tree["cost_guard_fired"] = cost_guard_fired

    evals_to_match = None
    running_best = None
    for i, n in enumerate([n for n in tree["nodes"] if n["status"] == "evaluated"], start=1):
        s = -n["score"]
        running_best = s if running_best is None else max(running_best, s)
        if running_best >= LINEAR_BEST and evals_to_match is None:
            evals_to_match = i
    tree["evals_to_match_linear_best"] = evals_to_match
    hv3.save_search_state(tree, TREE_PATH)  # Phase H-1 feature 7

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} AUC={gb_auc} mutation={gb['mutation'][:160]}")
    print(f"v2-tree best: {V2_TREE_BEST} @ eval {V2_TREE_BEST_EVAL}/{V2_TREE_N_EVAL} -- v3 "
          f"{'BEAT' if gb_auc > V2_TREE_BEST else ('MATCHED' if gb_auc == V2_TREE_BEST else 'REGRESSED vs')} it")
    print(f"Linear-iteration best: {LINEAR_BEST} (v3 {'BEAT' if gb_auc > LINEAR_BEST else 'did not beat'} it)")
    print(f"Evals to match/beat linear best: {evals_to_match}")
    print(f"Budget/phase state: {tree['search_state']['budget']}")
    print(f"Boundary candidates found: {BOUNDARY_LOG}")
    print(f"Cost-guard fired: {len(cost_guard_fired)} time(s): {cost_guard_fired}")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for e in tree["search_state"]["backtrack_log"]:
        print(" ", e)
    print(f"Dedup rejections ({len(dedup_rejections)}):")
    for e in dedup_rejections:
        print(" ", e)
    print(f"Prior usage: informed={len(prior_stats['informed'])} (win rate={prior_stats['informed_win_rate']}), "
          f"uninformed={len(prior_stats['uninformed'])} (win rate={prior_stats['uninformed_win_rate']})")


def dry_run():
    """Phase H-2 wiring check: verifies this driver's Phase H-1 entry-point plumbing
    (save_search_state/load_search_state round-trip, eval_solo_subprocess's module path,
    the sanity-gate functions) actually works, WITHOUT spending any real training compute
    and WITHOUT touching TREE_PATH -- everything happens on a scratch tree in a temp
    dir. Run with `uv run python3 tree_search/run_s3e7_v3.py --dry-run`."""
    import tempfile
    print("[dry-run] checking Phase H-1 entry points wired by this driver...")
    assert os.path.exists(EVAL_MODULE_PATH), f"eval module not found: {EVAL_MODULE_PATH}"

    scratch = hv3.new_tree(COMP)
    hv3.init_budget(scratch)
    root_cfg = {"kind": "solo", "model": "lgb", "params": {}, "features": {"drop": []}}
    rid = hv3.add_root(scratch, "dry-run root", root_cfg, -0.5, "evaluated", 0.01)

    with tempfile.TemporaryDirectory() as td:
        scratch_path = os.path.join(td, "scratch_tree.json")
        hv3.save_search_state(scratch, scratch_path)          # feature 7
        reloaded = hv3.load_search_state(scratch_path)        # feature 7
        assert reloaded["root_id"] == rid, "save/load_search_state round-trip mismatch"
    print("[dry-run] save_search_state/load_search_state round-trip OK")

    passed, bound = hv3.burst_seed_sanity_gate(scratch, -0.4)  # feature 9
    print(f"[dry-run] burst_seed_sanity_gate reachable (passed={passed}, bound={bound})")

    print(f"[dry-run] eval_solo_subprocess module path OK: {EVAL_MODULE_PATH}")
    print("[dry-run] OK -- no model was trained, no cache written, TREE_PATH untouched.")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run()
    else:
        main()
