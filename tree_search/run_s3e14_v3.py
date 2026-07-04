"""tree_search/run_s3e14_v3.py — Phase F-2 harness_v3 validation run for
playground-series-s3e14 (Wild Blueberry Yield, MAE, minimize). Re-runs the comp under
harness_v3's DEFAULT automatic policy (budget/phase machine, dedup-consumes-budget,
reopen-blend-on-solo-breakthrough, boundary_candidates, k=800+coordinate-ascent blend
search, blend-cost guard) to verify (a) no regression vs the v1-proto tree
(tree_search/run_s3e14.py + harness.py, best 340.52635 @ eval 11/20,
experiments_tree.json — UNTOUCHED by this run) and (b) the v3 policy machinery does real
work end to end.

eval_s3e14.py (the per-comp evaluator) is reused byte-for-byte, unmodified — only this
DRIVER changes. Same two driver-level conventions as run_s3e7_v3.py (see that file's
docstring for the full rationale, identical here):

  1. Node config STORAGE convention: `config` == `core(proposal)` (no "result" merged
     in, blend member lists sorted) for every node, so harness_v3's own
     `find_duplicate_config`/`add_node` dedup is meaningful without a bespoke stripping
     wrapper. Rich per-eval detail lives in
     `tree["search_state"]["node_results"][str(node_id)]` (resume-safe).
  2. Reuse-cached-OOF-where-digit-verified: root + 6 solo seeds (CAT/XGB/LGBTUNED/
     SEEDBAG/FEAT/REG) + the 3-way BLEND seed — node ids 0-7, confirmed identical
     ordering to experiments_tree.json — are recovered from tree_search/cache_s3e14/
     via `eval_s3e14.load_solo_cache` and accepted ONLY if a fresh MAE recompute from
     the cached OOF matches the OLD tree's stored score to 5 decimals.

Blend nodes are evaluated fresh every time via this driver's own `evaluate_blend_v3`
(k=800 default + coordinate-ascent, harness_v3 features 5/6) instead of
eval_s3e14.evaluate_blend's own hand-rolled `weight_search()` (k=3000, no ascent, no
cost guard) — same rationale as run_s3e7_v3.py.

boundary_candidates (feature 4): checked against the Optuna box verbatim-transcribed
from scripts/train_v3.py's `objective()` (the box LGBTUNED was actually tuned inside) —
unlike s3e7, none of LGBTUNED's tuned params sit within 5% of that box's own edges (the
nearest, reg_lambda, is ~13.5% of the way in log-space), so no forced BOUNDARYPUSH
lineage is created here; this run instead demonstrates feature 4 by showing the check
ran and genuinely found nothing to push (s3e7's run is where it fires for real).
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
import harness_v2 as hv2  # noqa: E402 -- unused directly but keeps import parity/availability
import harness_v3 as hv3  # noqa: E402
import eval_s3e14 as ev  # noqa: E402

COMP = "playground-series-s3e14"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
OLD_TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree.json")       # v1-proto tree -- READ-ONLY
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_v3.json")        # this run's output
EVAL_TIMEOUT_S = 180
MAX_WALL_S = 35 * 60
ITER_SAFETY_CAP = 150

LINEAR_BEST = 340.59891
V1_TREE_BEST = 340.52635
V1_TREE_BEST_EVAL = 11
V1_TREE_N_EVAL = 20

NOISE_FEATS = ev.NOISE_FEATS

# verbatim transcription of scripts/train_v3.py's objective() search box (the box
# LGBTUNED's params were actually tuned inside)
LGB_SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.08, "log": True},
    "num_leaves": (7, 127),
    "max_depth": (3, 10),
    "min_child_samples": (5, 100),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": {"low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"low": 1e-3, "high": 10.0, "log": True},
}

DEDUP_REJECTIONS = []
_dedup_offset = {}
BURST_INJECTED = [False]
COST_GUARD_FIRED = []
BOUNDARY_LOG = []


def dc(cfg):
    return copy.deepcopy(cfg)


def _node_results(tree):
    return tree.setdefault("search_state", {}).setdefault("node_results", {})


def result_of(tree, nid):
    return _node_results(tree).get(str(nid))


def core(cfg):
    out = {k: v for k, v in cfg.items() if k != "result"}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


# ---------------------------------------------------------------------------
# OLD (v1-proto) tree -- read-only, used only to look up reusable cached OOFs
# ---------------------------------------------------------------------------
OLD = hv2.load(OLD_TREE_PATH)
OLD_BY_HASH = {}
for _n in OLD["nodes"]:
    if _n["status"] == "evaluated":
        OLD_BY_HASH.setdefault(hv3.config_hash(core(_n["config"])), _n)


def try_reuse(stored_cfg):
    """core()-normalized solo config -> (oof, pred, feats, score, old_node) on a
    digit-verified cache hit (fresh MAE recompute from the cached OOF, honoring the
    config's own postprocess.snap flag, matches the OLD tree's stored score to 5
    decimals), else None."""
    if stored_cfg.get("kind") != "solo":
        return None
    old = OLD_BY_HASH.get(hv3.config_hash(stored_cfg))
    if old is None:
        return None
    try:
        d = ev.load_solo_cache(old["id"])
    except ValueError:
        return None
    oof, pred = d["oof"], d["pred"]
    feats = [str(f) for f in d["feats"]]
    pp = stored_cfg.get("postprocess") or {}
    recomputed = ev.mae(ev._y, ev.snap_to_grid(oof) if pp.get("snap", False) else oof)
    old_score = old.get("score")
    if old_score is None or round(recomputed, 5) != round(old_score, 5):
        return None
    return oof, pred, feats, recomputed, old


# ---------------------------------------------------------------------------
# v3 blend evaluation: bypasses eval_s3e14.evaluate_blend's hand-rolled weight_search()
# (k=3000, no ascent, no cost guard); uses a generic dirichlet/grid search + harness_v3's
# coordinate-ascent refinement (feature 5) with an explicit wall-time cost guard
# (feature 6), postprocess.snap applied INSIDE metric_fn throughout (v2/v3 blend
# contract for discretized-by-rounding metrics).
# ---------------------------------------------------------------------------
def _weight_search_generic(oofs, metric_fn, k=hv3.DEFAULT_BLEND_K, seed=42):
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
    mats = [ev.load_solo_cache(m)["oof"] for m in members]
    oofs = np.stack(mats, axis=1)
    pp = stored_cfg.get("postprocess") or {}
    use_snap = pp.get("snap", True)

    def metric_fn(vec):
        return ev.mae(ev._y, ev.snap_to_grid(vec)) if use_snap else ev.mae(ev._y, vec)

    method = stored_cfg.get("weight_search", "dirichlet")
    t0 = time.time()
    if method == "dirichlet":
        best_w, best_s = _weight_search_generic(oofs, metric_fn, k=hv3.DEFAULT_BLEND_K)
    elif method == "grid_simplex":
        W = ev._grid_simplex_weights(len(members))
        best_w = best_s = None
        for w in W:
            s = metric_fn(oofs @ w)
            if best_s is None or s < best_s:
                best_s, best_w = s, w
        best_w, best_s = hv3._coordinate_ascent_refine(oofs, metric_fn, best_w, best_s)
    else:
        raise ValueError(f"unknown weight_search method {method!r}")
    wall = time.time() - t0

    warning = None
    if wall > hv3.DEFAULT_COST_GUARD_THRESHOLD_S:
        warning = (f"blend eval took {wall:.1f}s > {hv3.DEFAULT_COST_GUARD_THRESHOLD_S:.0f}s "
                   f"threshold (members={members}) -- s3e14 blends are cheap by design, so "
                   f"this is logged for visibility only, no coarsening applied")
        tree.setdefault("search_state", {}).setdefault("cost_guard_log", []).append(dict(
            members=members, original_k=hv3.DEFAULT_BLEND_K, coarsened_k=None,
            original_wall_s=round(wall, 2), coarsened_wall_s=None, warning=warning))
        COST_GUARD_FIRED.append(warning)

    raw_score = ev.mae(ev._y, oofs @ best_w)
    snap_score = ev.mae(ev._y, ev.snap_to_grid(oofs @ best_w))
    result = dict(members=members, weights=[round(float(w), 5) for w in best_w], method=method,
                  raw_score=round(raw_score, 5), snap_score=round(snap_score, 5), used_snap=use_snap)
    return result, best_s, wall, warning


# ---------------------------------------------------------------------------
# eval_and_add
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, proposal_cfg, is_root=False, lineage_id=None):
    stored = core(proposal_cfg)
    if not is_root:
        dup_id = hv3.find_duplicate_config(tree, stored)
        if dup_id is not None:
            DEDUP_REJECTIONS.append(dict(mutation=mutation, dup_id=dup_id))
            if lineage_id is not None:
                _dedup_offset[lineage_id] = _dedup_offset.get(lineage_id, 0) + 1
            nid_null, dup_echo = hv3.add_node(tree, parent_id, mutation + " [dedup pre-check]",
                                               stored, None, "failed", 0.0)
            assert nid_null is None
            hv3.save(tree, TREE_PATH)
            return None, dup_id, None

    nid = hv3.next_id(tree)
    kind = stored.get("kind", "solo")

    if kind == "solo":
        reused = try_reuse(stored)
        if reused is not None:
            oof, pred, feats, score, old = reused
            ev._save_solo_cache(nid, oof, pred, feats, stored, score)
            status, wall_s = "evaluated", 0.05
            result = {"n_feats": len(feats)}
            full_mutation = (mutation + f" [REUSED cached OOF from prior v1-proto tree node "
                              f"#{old['id']}, digit-verified MAE {score:.5f} == {old['score']} "
                              f"to 5dp, no retraining]")
        else:
            r = ev.evaluate(proposal_cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
            score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
            full_mutation = mutation
    elif kind == "blend":
        try:
            result, score, wall_s, warning = evaluate_blend_v3(tree, stored)
            status = "evaluated"
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
        assert dup is None, f"unexpected post-eval dedup (dup=#{dup})"
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    if result is not None:
        _node_results(tree)[str(real_nid)] = result
    hv3.save(tree, TREE_PATH)
    return real_nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


def score_of(r):
    return r["score"] if r else None


# ---------------------------------------------------------------------------
# Root + first-generation seeds -- BYTE-IDENTICAL to run_s3e14.py's
# ---------------------------------------------------------------------------
ROOT_CONFIG = {
    "kind": "solo", "model": "lgb",
    "params": dict(objective="regression_l1", metric="mae", n_estimators=3000,
                   learning_rate=0.02, num_leaves=63, min_child_samples=30,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                   reg_alpha=1.0, reg_lambda=2.0, random_state=42, early_stopping_rounds=150),
    "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False},
}
ROOT_MUTATION = ("root: s3e14 current-best single LGB, byte-identical to "
                  "experiments_tree.json's root (v1-proto tree) -- verified digit-for-digit")


def seed_cat():
    cfg = {"kind": "solo", "model": "cat",
           "params": dict(loss_function="MAE", eval_metric="MAE", iterations=4000,
                          learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=42,
                          native_cat=True),
           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}}
    return cfg, "model-type: CatBoost solo, byte-identical to v1-proto's CAT seed"


def seed_xgb():
    cfg = {"kind": "solo", "model": "xgb",
           "params": dict(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.02,
                          max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                          reg_alpha=1.0, reg_lambda=2.0, random_state=42, eval_metric="mae",
                          early_stopping_rounds=150),
           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}}
    return cfg, "model-type: XGB solo, byte-identical to v1-proto's XGB seed"


LGBTUNED_PARAMS = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                       subsample_freq=1, random_state=42, learning_rate=0.015389816656608301,
                       num_leaves=63, max_depth=5, min_child_samples=30,
                       subsample=0.7899031923134097, colsample_bytree=0.6842166782718478,
                       reg_alpha=0.012648426946439915, reg_lambda=0.00346212601571565,
                       early_stopping_rounds=150)


def seed_lgbtuned():
    cfg = {"kind": "solo", "model": "lgb", "params": dc(LGBTUNED_PARAMS),
           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}}
    return cfg, "hyperparam: Optuna fold-0-proxy tuned LGB, byte-identical to v1-proto's LGBTUNED seed"


def seed_seedbag():
    p = dc(ROOT_CONFIG["params"])
    p["random_state"] = 2024
    cfg = {"kind": "solo", "model": "lgb", "params": p,
           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}}
    return cfg, "seed variation: root LGB, random_state 42->2024, byte-identical to v1-proto's SEEDBAG seed"


def seed_feat():
    drop = list(NOISE_FEATS) + ["MaxOfUpperTRange", "MinOfUpperTRange", "MaxOfLowerTRange"]
    cfg = {"kind": "solo", "model": "lgb", "params": dc(ROOT_CONFIG["params"]),
           "features": {"drop": drop}, "postprocess": {"snap": False}}
    return cfg, "feature-subset SUBTRACTION, byte-identical to v1-proto's FEAT seed"


def seed_reg():
    p = dc(ROOT_CONFIG["params"])
    p.update(num_leaves=31, min_child_samples=50)
    cfg = {"kind": "solo", "model": "lgb", "params": p,
           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}}
    return cfg, "hyperparam nudge: regularize further, byte-identical to v1-proto's REG seed"


def seed_xgbtuned():
    p = dict(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.03,
             max_depth=5, min_child_weight=8, subsample=0.75, colsample_bytree=0.75,
             reg_alpha=0.5, reg_lambda=3.0, random_state=42, eval_metric="mae",
             early_stopping_rounds=150)
    cfg = {"kind": "solo", "model": "xgb", "params": p,
           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}}
    desc = ("model-variant: a disciplined middle-ground XGB (depth5/lr0.03/mcw8), "
            "distinct from the plain XGB seed -- added to the pool [PRIOR P9 -- "
            "add-to-pool, never replace]")
    return cfg, desc


SOLO_SEED_SPECS = [("CAT", seed_cat), ("XGB", seed_xgb), ("LGBTUNED", seed_lgbtuned),
                    ("SEEDBAG", seed_seedbag), ("FEAT", seed_feat), ("REG", seed_reg)]
NEW_SEED_SPECS = [("XGBTUNED", seed_xgbtuned)]
ALL_LINEAGE_NAMES = [n for n, _ in SOLO_SEED_SPECS] + ["BLEND"] + [n for n, _ in NEW_SEED_SPECS]
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
# NEW per-lineage mutation queues -- each ONE genuinely-untried-by-v1 item (REG's queue
# was NEVER actually explored in the old tree -- both its items are real new value, not
# just "new relative to what was tried"), then falls back to seed-variation.
# ---------------------------------------------------------------------------
def _bump(cfg, **kw):
    c = dc(cfg)
    c["params"].update(kw)
    return c


CAT_QUEUE = [
    lambda c: (_bump(c, bagging_temperature=1.5, depth=8),
               "CAT: NEW -- bagging_temperature=1.5 + depth 7->8, untested combo "
               "[PRIOR none -- fresh regularization+capacity axis]"),
]
XGB_QUEUE = [
    lambda c: (_bump(c, gamma=0.05, colsample_bylevel=0.8),
               "XGB: NEW -- gamma=0.05 (min split-loss) + colsample_bylevel=0.8 "
               "[PRIOR none -- fresh regularization axis]"),
]
LGBTUNED_QUEUE = [
    lambda c: (_bump(c, subsample=0.65),
               "LGBTUNED: NEW -- subsample 0.79->0.65, further row-bagging regularization "
               "[PRIOR none -- untested axis on the Optuna-tuned params]"),
]
SEEDBAG_QUEUE = [
    lambda c: (_bump(c, random_state=555),
               "SEEDBAG: NEW -- third independent seed (555), STATUS.md's own 'next "
               "ideas' flagged 3-5 seed bags as untried [PRIOR P2 -- add-to-pool + seed-bag]"),
]


def _feat(c, drop_list):
    child = dc(c)
    child["features"] = {"drop": sorted(set(drop_list))}
    return child


FEAT_QUEUE = [
    lambda c: (_feat(c, c["features"]["drop"] + ["MinOfLowerTRange"]),
               "FEAT: NEW -- further trim MinOfLowerTRange (already dropped as a root "
               "NOISE_FEAT -- guard: only applies if still present) alongside the FEAT "
               "seed's 3-col trim [PRIOR none]") if "MinOfLowerTRange" not in c["features"]["drop"]
    else (_feat(c, c["features"]["drop"] + ["seeds_per_fruitset"]),
          "FEAT: NEW -- further trim seeds_per_fruitset (ratio feature, redundant with "
          "the fruitset_x_seeds product term) -- v1's own untried FEAT_QUEUE idea "
          "[PRIOR P3]"),
]
REG_QUEUE = [
    lambda c: (_bump(c, reg_lambda=4.0),
               "REG: reg_lambda 2.0->4.0, push L2 further -- v1-proto's own REG_QUEUE "
               "item, NEVER actually run (old tree has only the REG seed itself) [PRIOR P6]"),
    lambda c: (_bump(c, num_leaves=90, subsample=0.7),
               "REG: opposite-direction probe -- num_leaves 63->90 (more capacity) + "
               "subsample 0.8->0.7 -- v1-proto's own REG_QUEUE item, also never run [PRIOR none]"),
]
XGBTUNED_QUEUE = []

SOLO_QUEUES = {"CAT": CAT_QUEUE, "XGB": XGB_QUEUE, "LGBTUNED": LGBTUNED_QUEUE,
               "SEEDBAG": SEEDBAG_QUEUE, "FEAT": FEAT_QUEUE, "REG": REG_QUEUE,
               "XGBTUNED": XGBTUNED_QUEUE}


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
    members = sorted([tree["root_id"], ids["XGB"], ids["CAT"]])
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet",
           "postprocess": {"snap": True}}
    desc = (f"ensemble seed: 3-way blend of root(#{tree['root_id']}) + XGB(#{ids['XGB']}) "
            f"+ CAT(#{ids['CAT']}) -- byte-identical member set to v1-proto's BLEND seed")
    return cfg, desc


def _mk_add_named(name, prior_tag):
    def fn(tree, parent_node):
        add_id = _id_for(tree, name)
        members = parent_node["config"]["members"]
        if add_id in members:
            return None
        child = {"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": parent_node["config"].get("weight_search", "dirichlet"),
                 "postprocess": dict(parent_node["config"].get("postprocess") or {"snap": True})}
        desc = f"BLEND: add {name} solo member (#{add_id}) -> {len(child['members'])}-way {prior_tag}"
        return child, desc
    return fn


def _blend_method_swap(tree, parent_node):
    cur = parent_node["config"].get("weight_search", "dirichlet")
    members = parent_node["config"]["members"]
    new_method = "grid_simplex" if cur == "dirichlet" else "dirichlet"
    if new_method == "grid_simplex" and len(members) > 5:
        return None
    child = dc(parent_node["config"])
    child["weight_search"] = new_method
    desc = f"BLEND: weight-search method {cur}->{new_method} on the same members"
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
    desc = f"BLEND: remove lowest-weight member #{weak_id} (w={weights[idx]:.3f})"
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("LGBTUNED", "[mirrors v1-proto exp -- 4-way]"),
    _mk_add_named("SEEDBAG", "[mirrors v1-proto's KEY node -- 5-way, PRIOR P2]"),
    _blend_method_swap,
    _mk_add_named("FEAT", "[PRIOR none -- different feature set diversity]"),
    _mk_add_named("REG", "[NEW -- REG was never added to a blend in v1-proto]"),
    _mk_add_named("XGBTUNED", "[PRIOR P9 -- add-to-pool, never replace]"),
    _blend_remove_weakest,
]


def blend_fallback(tree, parent_node, attempt):
    pool = solo_pool(tree)
    members = parent_node["config"]["members"]
    for add_id in [nid for nid in pool if nid not in members]:
        child = {"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": parent_node["config"].get("weight_search", "dirichlet"),
                 "postprocess": dict(parent_node["config"].get("postprocess") or {"snap": True})}
        if hv3.find_duplicate_config(tree, child) is None:
            return child, f"BLEND fallback: add next-best unused pool member #{add_id}"
    cur = parent_node["config"].get("weight_search", "dirichlet")
    child = dc(parent_node["config"])
    child["weight_search"] = "grid_simplex" if cur == "dirichlet" else "dirichlet"
    if hv3.find_duplicate_config(tree, child) is None:
        return child, f"BLEND fallback: method swap {cur}->{child['weight_search']}"
    return None


def _lineage_kind(tree, lineage_id):
    """See run_s3e7_v3.py's identical helper docstring: dispatch on the lineage's own
    first-gen node kind, not a literal name=='BLEND' check, so ad-hoc blend lineages
    with no authored queue (EXPL_MEGABLEND) don't KeyError on SOLO_QUEUES[name]."""
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
# Explore burst (feature 1)
# ---------------------------------------------------------------------------
def _burst_seeds():
    return [
        ("EXPL_DART", {"kind": "solo", "model": "lgb",
                        "params": dict(ROOT_CONFIG["params"], boosting_type="dart",
                                       drop_rate=0.1, max_drop=50, n_estimators=1200,
                                       early_stopping_rounds=1200),  # dart doesn't improve
                                       # monotonically fold-to-fold; esr==n_estimators
                                       # effectively never triggers (None raises TypeError
                                       # in lgb.early_stopping())
                        "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}},
         "long-shot: DART boosting mode (never tried), regularization via dropped trees"),
        ("EXPL_XT", {"kind": "solo", "model": "lgb",
                      "params": dict(ROOT_CONFIG["params"], extra_trees=True),
                      "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}},
         "long-shot: extra_trees=True, pure diversity via randomized split thresholds"),
        ("EXPL_CATDEEP", {"kind": "solo", "model": "cat",
                           "params": dict(loss_function="MAE", eval_metric="MAE", iterations=4000,
                                          learning_rate=0.03, depth=9, bagging_temperature=2.0,
                                          random_seed=42, native_cat=True),
                           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}},
         "long-shot: CatBoost pushed deeper (7->9) + Bayesian-bootstrap bagging_temp=2.0"),
        ("EXPL_XGBLEAF", {"kind": "solo", "model": "xgb",
                           "params": dict(objective="reg:absoluteerror", n_estimators=3000,
                                          learning_rate=0.02, tree_method="hist",
                                          grow_policy="lossguide", max_leaves=64,
                                          subsample=0.8, colsample_bytree=0.7,
                                          random_state=42, eval_metric="mae",
                                          early_stopping_rounds=150),
                           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}},
         "long-shot: leaf-wise-growth XGB (grow_policy=lossguide), structurally closer "
         "to LGB via a different implementation"),
        ("EXPL_REGDEEP", {"kind": "solo", "model": "lgb",
                           "params": dict(ROOT_CONFIG["params"], num_leaves=127, max_depth=8,
                                          reg_alpha=0.1, reg_lambda=0.5),
                           "features": {"drop": list(NOISE_FEATS)}, "postprocess": {"snap": False}},
         "long-shot: opposite-direction capacity probe -- much larger num_leaves/depth "
         "with lighter regularization, bracketing the sweet spot from the other side"),
    ]


BURST_NAMES = [name for name, _, _ in _burst_seeds()] + ["EXPL_MEGABLEND"]
ALL_LINEAGE_NAMES.extend(BURST_NAMES)  # so resume's LINEAGE_NAMES-rebuild loop and
                                        # propose_child's SOLO_QUEUES.get(name, []) both
                                        # recognize burst-injected lineages after a restart


def inject_explore_burst(tree, root_id):
    seeds = _burst_seeds()
    injected_ids = []
    for name, cfg, desc in seeds:
        nid, dup, r = eval_and_add(tree, root_id, f"[{name}] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            SOLO_QUEUES[name] = []
            injected_ids.append(nid)
            print(f"[BURST {name}] #{nid} score={score_of(r)} status={r['status']} wall_s={r['wall_s']}")
    pool = solo_pool(tree)
    if len(pool) >= 2:
        cfg = {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet",
               "postprocess": {"snap": True}}
        nid, dup, r = eval_and_add(tree, root_id, "[EXPL_MEGABLEND] long-shot: kitchen-sink "
                                    f"blend of the entire {len(pool)}-member solo pool via v3's "
                                    "k=800+coordinate-ascent search + cost guard "
                                    "[feature 5/6 -- direct test at scale]", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "EXPL_MEGABLEND"
            injected_ids.append(nid)
            print(f"[BURST EXPL_MEGABLEND] #{nid} score={score_of(r)} status={r['status']} wall_s={r['wall_s']}")
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
        tree = hv3.load(TREE_PATH)
        print(f"Resuming existing tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv3.new_tree(COMP)
        hv3.init_budget(tree)
        print(f"init_budget -> {tree['search_state']['budget']}")

    def n_evaluated():
        return hv3.n_evaluated(tree)

    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} score={score_of(r)} wall_s={r['wall_s']} status={r['status']}")
        assert r["status"] == "evaluated"
        assert round(score_of(r), 5) == 342.02154, (
            f"ROOT DIGIT-VERIFY FAILED: expected 342.02154 (experiments_tree.json node #0), "
            f"got {score_of(r)}")
        print("ROOT DIGIT-VERIFIED against experiments_tree.json node #0 (342.02154). OK.")

    # boundary_candidates check against LGBTUNED's Optuna box (feature 4 -- documented as
    # a genuine-empty-result run in this comp per module docstring)
    edges = hv3.boundary_candidates({"params": LGBTUNED_PARAMS}, LGB_SEARCH_SPACE)
    if not BOUNDARY_LOG:  # guard against double-extend across resumed runs
        BOUNDARY_LOG.extend(edges)
    print(f"boundary_candidates(LGBTUNED vs its Optuna box) -> {edges} (expected: empty)")

    all_named_seeds = list(SOLO_SEED_SPECS) + list(NEW_SEED_SPECS)
    for name, seed_fn in all_named_seeds:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} score={score_of(r)} status={r['status']} wall_s={r['wall_s']}")

    already_blend = any(n["mutation"].startswith("[BLEND]") for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} score={score_of(r)} status={r['status']} wall_s={r['wall_s']}")

    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in ALL_LINEAGE_NAMES:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name
                    if name in BURST_NAMES:
                        SOLO_QUEUES.setdefault(name, [])

    # resume-safety: BURST_INJECTED is module-level (not persisted) -- infer it from the
    # tree itself so a resumed process never re-injects the burst.
    if any(n["mutation"].startswith("[EXPL_") for n in tree["nodes"] if n["parent_id"] == tree["root_id"]):
        BURST_INJECTED[0] = True

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
            hv3.save(tree, TREE_PATH)
            print(f"FORCED BACKTRACK: lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} exhausted; "
                  f"plateaued={tree['search_state']['plateaued']}")
            continue
        nid, dup, r = eval_and_add(tree, parent_id, mutation, child_cfg, lineage_id=lineage_id)
        if nid is None:
            print(f"DEDUP REJECTED <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"(dup of #{dup})")
            continue
        gb = hv3.global_best(tree)
        budget = tree["search_state"]["budget"]
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"score={score_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best={gb['score'] if gb else None} (#{gb['id'] if gb else '-'}) "
              f"| phase={budget['phase']} n_eval={n_evaluated()} "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = hv3.global_best(tree)
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    prior_stats = prior_usage_summary(tree)
    tree["prior_usage_summary"] = {
        "n_informed": len(prior_stats["informed"]), "n_uninformed": len(prior_stats["uninformed"]),
        "informed_win_rate": prior_stats["informed_win_rate"],
        "uninformed_win_rate": prior_stats["uninformed_win_rate"],
    }
    tree["dedup_rejections"] = DEDUP_REJECTIONS
    tree["boundary_candidates_log"] = BOUNDARY_LOG
    tree["cost_guard_fired"] = COST_GUARD_FIRED
    hv3.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} score={gb['score']} mutation={gb['mutation'][:160]}")
    print(f"v1-proto tree best: {V1_TREE_BEST} @ eval {V1_TREE_BEST_EVAL}/{V1_TREE_N_EVAL} -- v3 "
          f"{'BEAT' if gb['score'] < V1_TREE_BEST else ('MATCHED' if gb['score'] == V1_TREE_BEST else 'REGRESSED vs')} it")
    print(f"Linear-iteration best: {LINEAR_BEST} (v3 {'BEAT' if gb['score'] < LINEAR_BEST else 'did not beat'} it)")
    print(f"Budget/phase state: {tree['search_state']['budget']}")
    print(f"Boundary candidates found: {BOUNDARY_LOG}")
    print(f"Cost-guard fired: {len(COST_GUARD_FIRED)} time(s): {COST_GUARD_FIRED}")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for e in tree["search_state"]["backtrack_log"]:
        print(" ", e)
    print(f"Dedup rejections ({len(DEDUP_REJECTIONS)}):")
    for e in DEDUP_REJECTIONS:
        print(" ", e)
    print(f"Prior usage: informed={len(prior_stats['informed'])} (win rate={prior_stats['informed_win_rate']}), "
          f"uninformed={len(prior_stats['uninformed'])} (win rate={prior_stats['uninformed_win_rate']})")


if __name__ == "__main__":
    main()
