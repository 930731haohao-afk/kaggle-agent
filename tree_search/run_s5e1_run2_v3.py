"""tree_search/run_s5e1_run2_v3.py — v3 tree-search driver for playground-series-s5e1
(run2 benchmark; MAPE, minimize; structure-dominant: level/shape decoupled models).

Checklist per references/07_tree_search.md:
  1. Root digit-verify: ridge_level_conv50 (linear champion solo) must reproduce pooled
     rounded MAPE 6.928121 to 6dp or abort.
  2. No OOF reuse — cache_s5e1_run2 is fresh (archived arm caches are off-limits).
  3. Resume-state via hv3.save_search_state/load_search_state.
  4. Solo evals through hv3.eval_solo_subprocess.
  5. Burst seeds gated by hv3.apply_burst_seed_sanity_gate.

Node space: structural knobs on the ridge shape model + level-anchor rule (convNN/trend),
holiday window, Fourier orders, recency weight; LGB lineages for diversity (level-anchor
form and the dossier's join_feature arm, kept alive per TASK-TS-FUTURE race rule); blend
nodes mix cached solo OOFs (k=800 + coordinate ascent).
"""
import copy
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402
import eval_s5e1_run2 as ev  # noqa: E402

COMP = "playground-series-s5e1"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_run2.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_s5e1_run2.py")
EVAL_TIMEOUT_S = 240
MAX_WALL_S = 3600 * 2
ITER_SAFETY_CAP = 400
ROOT_MAPE = 6.928121           # linear champion ridge_level_conv50, digit-verified
LINEAR_BEST = 6.928121         # linear best solo == root (fitted blend failed honest gate)

LINEAGE_NAMES = {}
BURST_INJECTED = [False]
BOUNDARY_LOG = []

RIDGE_SEARCH_SPACE = {
    "alpha": (0.03, 3.0), "doy_k": (2, 12), "c104_k": (1, 4), "c52_k": (1, 4),
    "hol_before": (0, 10), "hol_after": (0, 15),
}
LGB_SEARCH_SPACE = {
    "num_leaves": (7, 255), "learning_rate": (0.01, 0.12),
    "n_estimators": (200, 2500), "min_child_samples": (5, 100),
}


def dc(x):
    return copy.deepcopy(x)


def core(cfg):
    c = dc(cfg)
    c.pop("result", None)
    if c.get("kind") == "blend":
        c["members"] = sorted(c["members"])
    return c


def _driver_state(tree):
    return tree["search_state"].setdefault("driver_state", {})


def _dedup_rejections(tree):
    return _driver_state(tree).setdefault("dedup_rejections", [])


def _node_results(tree):
    return _driver_state(tree).setdefault("node_results", {})


def _dedup_offset(tree):
    return _driver_state(tree).setdefault("dedup_offset", {})


def metric_fn(vec):
    return ev.metric(vec)


def eval_and_add(tree, parent_id, mutation, proposal_cfg, is_root=False, lineage_id=None):
    stored = core(proposal_cfg)
    if not is_root:
        dup_id = hv3.find_duplicate_config(tree, stored)
        if dup_id is not None:
            _dedup_rejections(tree).append(dict(mutation=mutation, dup_id=dup_id))
            if lineage_id is not None:
                off = _dedup_offset(tree)
                off[str(lineage_id)] = off.get(str(lineage_id), 0) + 1
            nid_null, _dup_echo = hv3.add_node(tree, parent_id, mutation + " [dedup pre-check]",
                                              stored, None, "failed", 0.0)
            assert nid_null is None
            hv3.save_search_state(tree, TREE_PATH)
            return None, dup_id, None

    nid = hv3.next_id(tree)
    kind = stored.get("kind", "solo")
    if kind == "solo":
        r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, proposal_cfg, EVAL_TIMEOUT_S, node_id=nid)
        score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
        full_mutation = mutation if not r.get("error") else mutation + f" [ERR: {str(r['error'])[:150]}]"
    elif kind == "blend":
        try:
            t0 = time.time()
            best_w, best_s, _oofs, warning = hv3.eval_blend_with_cost_guard(
                ev.CACHE_DIR, stored["members"], metric_fn, tree=tree)
            wall_s = round(time.time() - t0, 1)
            score, status = round(best_s, 6), "evaluated"
            result = {"mape": round(best_s, 6),
                      "weights": [round(float(w), 4) for w in best_w]}
            full_mutation = mutation if not warning else mutation + f" [{warning}]"
        except Exception as e:  # noqa: BLE001
            score, status, wall_s, result = None, "failed", 0.0, None
            full_mutation = mutation + f" [ERROR: {type(e).__name__}: {e}]"
    else:
        raise ValueError(f"unknown node kind {kind!r}")

    if is_root:
        real_nid = hv3.add_root(tree, full_mutation, stored, score, status, wall_s)
    else:
        real_nid, dup = hv3.add_node(tree, parent_id, full_mutation, stored, score, status, wall_s)
        assert dup is None, f"unexpected post-eval dedup (dup=#{dup})"
    assert real_nid == nid, f"id mismatch: predicted {nid}, got {real_nid}"
    if result is not None:
        _node_results(tree)[str(real_nid)] = result
    hv3.save_search_state(tree, TREE_PATH)
    return real_nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


# ---------------------------------------------------------------------------
# Root + first-generation lineage seeds
# ---------------------------------------------------------------------------
ROOT_CONFIG = {"kind": "solo", "model": "ridge", "target": "level_anchor",
               "level_rule": "conv50", "alpha": 1.0, "log_gdp": False, "yearc": False}
ROOT_MUTATION = ("root: linear champion ridge_level_conv50 — structural Ridge shape model "
                 "on log(y/level(country,year)) [country/store/product + cxp + sxp + cxdow "
                 "+ Fourier(doy 8, prodx104w 2, prodx52w 2) + per-holiday-name dummies "
                 "-5..+10], level anchor gdp_pc x conv50 ratio; pooled rounded MAPE over "
                 "2014/2015/2016 expanding-year folds")


def _r(over):
    c = dc(ROOT_CONFIG)
    c.update(over)
    return c


LGBDIV_CONFIG = {"kind": "solo", "model": "lgb", "target": "level_anchor",
                 "level_rule": "conv50", "params": {}}
LGBJOIN_CONFIG = {"kind": "solo", "model": "lgb", "target": "log",
                  "features": ev.M.LGB_FEATURES + ["log_gdp_pc"], "params": {}}

SOLO_SEED_SPECS = [
    ("ALPHA", lambda: (_r({"alpha": 0.3}),
     "ridge alpha 1.0->0.3 — weaker shrinkage on structural design [comp-local]")),
    ("CONVW", lambda: (_r({"level_rule": "conv35"}),
     "level anchor conv weight 50->35 (less pull to cross-country mean) [comp-local: "
     "conv sweep was U-shaped 25/50/75 -> 7.26/7.10/7.54 raw]")),
    ("LEVTREND", lambda: (_r({"level_rule": "trend"}),
     "level anchor rule -> per-country log-linear trend over last 3 fit years "
     "[comp-local: trend was 2nd-best rule for LGB]")),
    ("HOLWIN", lambda: (_r({"hol_before": 7, "hol_after": 12}),
     "holiday window -5..+10 -> -7..+12 [PRIOR tpsjan22: window width biggest lever]")),
    ("CYCK", lambda: (_r({"c104_k": 3, "c52_k": 3}),
     "product-cycle Fourier order 2->3 (104w and 52w) [comp-local: share sinusoids]")),
    ("RECENT", lambda: (_r({"recency_halflife_years": 2.0}),
     "recency sample weight halflife 2y — favor post-2014 regime [comp-local: regime "
     "shift in 2015]")),
    ("LGBDIV", lambda: (dc(LGBDIV_CONFIG),
     "LGB shape model on same conv50 level anchor — GBDT diversity member "
     "[comp-local: solo 7.069 rounded]")),
    ("LGBJOIN", lambda: (dc(LGBJOIN_CONFIG),
     "LGB log-target with log_gdp_pc join_feature — dossier TASK-TS-FUTURE race arm, "
     "kept alive although CV prefers level_anchor [prior: LB picked join twice]")),
]

SOLO_QUEUES = {
    "ALPHA": [lambda c: (_bump(c, alpha=0.1), "ALPHA: 0.3->0.1 [comp-local]")],
    "CONVW": [lambda c: (_bump(c, level_rule="conv65"), "CONVW: try 65 (other side) [comp-local]")],
    "LEVTREND": [lambda c: (_bump(c, level_rule="conv100"),
                            "LEVTREND branch: conv100 = pure GDP-proportional level [comp-local]")],
    "HOLWIN": [lambda c: (_bump(c, hol_before=10, hol_after=15),
               "HOLWIN: push to box edge -10..+15 [PRIOR tpsjan22 hol]")],
    "CYCK": [lambda c: (_bump(c, doy_k=10), "CYCK: doy Fourier 8->10 [comp-local]")],
    "RECENT": [lambda c: (_bump(c, recency_halflife_years=1.0),
               "RECENT: halflife 2y->1y stronger [comp-local]")],
    "LGBDIV": [lambda c: (_bump_lgb(c, num_leaves=255, learning_rate=0.04, n_estimators=2000),
               "LGBDIV: more capacity, slower lr [comp-local]")],
    "LGBJOIN": [lambda c: (_bump_lgb(c, num_leaves=63, min_child_samples=40),
                "LGBJOIN: shallower + regularized [PRIOR: extrapolation rewards reg]")],
}


def _bump(cfg, **kw):
    c = dc(cfg)
    c.update(kw)
    return c


def _bump_lgb(cfg, **kw):
    c = dc(cfg)
    c["params"] = dict(c.get("params", {}))
    c["params"].update(kw)
    return c


RIDGE_FALLBACK_NUDGES = [
    lambda c: {"alpha": round(c.get("alpha", 1.0) * 0.5, 5)},
    lambda c: {"alpha": round(c.get("alpha", 1.0) * 2.0, 5)},
    lambda c: {"doy_k": c.get("doy_k", 8) + 1},
    lambda c: {"hol_after": c.get("hol_after", 10) + 2},
    lambda c: {"hol_before": c.get("hol_before", 5) + 2},
    lambda c: {"c104_k": c.get("c104_k", 2) + 1},
    lambda c: {"recency_halflife_years": 3.0 if c.get("recency_halflife_years") is None else None},
    lambda c: {"cxdow": not c.get("cxdow", True)},
]
LGB_FALLBACK_NUDGES = [
    lambda p: {"num_leaves": max(7, p.get("num_leaves", 127) // 2)},
    lambda p: {"min_child_samples": p.get("min_child_samples", 20) * 2},
    lambda p: {"learning_rate": round(p.get("learning_rate", 0.06) * 0.6, 4),
               "n_estimators": 2000},
    lambda p: {"seed": p.get("seed", 42) + 1000},
]


def solo_fallback(parent_cfg, attempt):
    if parent_cfg.get("model") == "lgb":
        n = LGB_FALLBACK_NUDGES[attempt % len(LGB_FALLBACK_NUDGES)](parent_cfg.get("params", {}))
        n = {k: v for k, v in n.items() if v is not None}
        if not n:
            n = {"seed": 42 + attempt}
        return _bump_lgb(parent_cfg, **n), f"fallback nudge {n}"
    n = RIDGE_FALLBACK_NUDGES[attempt % len(RIDGE_FALLBACK_NUDGES)](parent_cfg)
    n = {k: v for k, v in n.items() if v is not None}
    if not n:
        n = {"alpha": round(parent_cfg.get("alpha", 1.0) * (3 + attempt), 5)}
    return _bump(parent_cfg, **n), f"fallback nudge {n}"


def solo_pool(tree):
    nodes = [n for n in tree["nodes"] if n["status"] == "evaluated"
             and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def result_of(tree, nid):
    return _node_results(tree).get(str(nid))


# ---------------------------------------------------------------------------
# BLEND lineage
# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    out = {}
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name in ALL_LINEAGE_NAMES:
                if name not in out and n["mutation"].startswith(f"[{name}]"):
                    out[name] = n["id"]
    return out


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    pool = solo_pool(tree)
    div = [ids[k] for k in ("LGBDIV", "LGBJOIN") if k in ids]
    best_other = [i for i in pool if i != tree["root_id"] and i not in div][:2]
    members = sorted(set([tree["root_id"]] + div + best_other))
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    desc = (f"ensemble seed: root + LGB diversity members {div} + top-2 structural "
            f"variants {best_other} [PRIOR: blends beat solos in 9/10 comps]")
    return cfg, desc


def blend_fallback(tree, parent_node, attempt):
    pool = solo_pool(tree)
    members = parent_node["config"]["members"]
    for add_id in [nid for nid in pool if nid not in members]:
        child = {"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": "dirichlet"}
        if hv3.find_duplicate_config(tree, child) is None:
            return child, f"BLEND fallback: add next-best unused pool member #{add_id}"
    res = result_of(tree, parent_node["id"]) or {}
    weights = res.get("weights")
    if weights and len(members) > 2:
        idx = int(np.argmin(weights))
        child = {"kind": "blend", "members": sorted(m for m in members if m != members[idx]),
                 "weight_search": "dirichlet"}
        if hv3.find_duplicate_config(tree, child) is None:
            return child, f"BLEND fallback: drop lowest-weight member #{members[idx]}"
    return None


def _blend_add_best_unused(tree, parent_node):
    return blend_fallback(tree, parent_node, 0)


BLEND_QUEUE = [_blend_add_best_unused]


def _lineage_kind(tree, lineage_id):
    node = next(n for n in tree["nodes"] if n["id"] == lineage_id)
    return node["config"].get("kind", "solo")


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES.get(lineage_id, f"L{lineage_id}")
    idx = hv3.lineage_size(tree, lineage_id) - 1 + _dedup_offset(tree).get(str(lineage_id), 0)
    if _lineage_kind(tree, lineage_id) == "blend":
        result = None
        if idx < len(BLEND_QUEUE):
            result = BLEND_QUEUE[idx](tree, parent_node)
        if result is None:
            result = blend_fallback(tree, parent_node, max(idx - len(BLEND_QUEUE), 0))
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
# Explore burst
# ---------------------------------------------------------------------------
def _burst_seeds():
    return [
        ("EXPL_OLS", _r({"alpha": 0.03}),
         "long-shot: near-OLS ridge — structural design may need no shrinkage"),
        ("EXPL_COMBO", _r({"level_rule": "conv35", "hol_before": 7, "hol_after": 12,
                           "c104_k": 3, "alpha": 0.3}),
         "long-shot: stack individually-promising levers in one config"),
        ("EXPL_CTRYTREND", _r({"country_yearc": True}),
         "long-shot: country x year_c residual trends inside the shape model "
         "(extrapolates per-country drift; risky by design)"),
        ("EXPL_LGBBIG", {"kind": "solo", "model": "lgb", "target": "level_anchor",
                         "level_rule": "conv50",
                         "params": {"num_leaves": 255, "learning_rate": 0.02,
                                    "n_estimators": 2500, "min_child_samples": 10}},
         "long-shot: high-capacity LGB shape model — blend diversity"),
        ("EXPL_MAPEW", {"kind": "solo", "model": "lgb", "target": "mape_w",
                        "features": ev.M.LGB_FEATURES + ["log_gdp_pc"], "params": {}},
         "long-shot: direct MAPE objective (L1 + 1/y weights) — decorrelated errors"),
    ]


ALL_LINEAGE_NAMES = [name for name, _ in SOLO_SEED_SPECS] + ["BLEND"]
BURST_NAMES = [n for n, _, _ in _burst_seeds()] + ["EXPL_MEGABLEND"]
ALL_LINEAGE_NAMES.extend(BURST_NAMES)


def inject_explore_burst(tree, root_id):
    for name, cfg, desc in _burst_seeds():
        nid, _dup, r = eval_and_add(tree, root_id, f"[{name}] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            SOLO_QUEUES.setdefault(name, [])
            print(f"[BURST {name}] #{nid} MAPE={r['score']} status={r['status']}")
            passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
            if not passed:
                print(f"[BURST {name}] #{nid} FAILED sanity gate (bound={bound}) — plateaued")
    pool = solo_pool(tree)
    if len(pool) >= 2:
        cfg = {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet"}
        nid, _dup, r = eval_and_add(tree, root_id, "[EXPL_MEGABLEND] long-shot: "
                                   f"kitchen-sink blend of the entire {len(pool)}-member "
                                   "solo pool (k=800+coord-ascent)", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "EXPL_MEGABLEND"
            hv3.apply_burst_seed_sanity_gate(tree, nid)
            print(f"[BURST EXPL_MEGABLEND] #{nid} MAPE={r['score']} status={r['status']}")


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        print(f"Resuming tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv3.new_tree(COMP)
        hv3.init_budget(tree)
        print(f"init_budget -> {tree['search_state']['budget']}")

    priors = hv2.suggest_priors({"comp": COMP, "metric": "MAPE",
                                 "tags": ["timeseries", "panel", "structural", "smape"],
                                 "data_type": "tabular"})
    print(f"suggest_priors: {len(priors)} hits")
    for p in priors[:8]:
        print("  PRIOR:", str(p)[:160])

    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} MAPE={r['score']} status={r['status']}")
        assert r["status"] == "evaluated"
        assert round(r["score"], 6) == ROOT_MAPE, (
            f"ROOT DIGIT-VERIFY FAILED: expected {ROOT_MAPE}, got {r['score']}")
        print(f"ROOT DIGIT-VERIFIED: {ROOT_MAPE}. OK.")

    if not BOUNDARY_LOG:
        ridge_knobs = {k: ROOT_CONFIG[k] for k in ("alpha",) if k in ROOT_CONFIG}
        BOUNDARY_LOG.extend(hv3.boundary_candidates({"params": ridge_knobs}, RIDGE_SEARCH_SPACE))
    print(f"boundary_candidates(root vs ridge box) -> {BOUNDARY_LOG}")

    for name, seed_fn in SOLO_SEED_SPECS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} MAPE={r['score'] if r else 'dup'}")

    already_blend = any(n["mutation"].startswith("[BLEND]") for n in tree["nodes"]
                        if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} MAPE={r['score'] if r else 'dup'}")

    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in ALL_LINEAGE_NAMES:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name
                    if name in BURST_NAMES:
                        SOLO_QUEUES.setdefault(name, [])

    if any(n["mutation"].startswith("[EXPL_") for n in tree["nodes"]
           if n["parent_id"] == tree["root_id"]):
        BURST_INJECTED[0] = True

    iterations = 0
    while (not hv3.should_stop(tree)) and hv3.n_evaluated(tree) < tree["search_state"]["budget"]["total_budget"] \
            and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > ITER_SAFETY_CAP:
            print("Iteration safety cap reached, stopping.")
            break
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not BURST_INJECTED[0]:
            BURST_INJECTED[0] = True
            print(f"\n>>> PHASE -> explore_burst (n_eval={hv3.n_evaluated(tree)}) <<<\n")
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
                    reason=f"lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} mutation "
                           f"space exhausted -> forced backtrack"))
            hv3.save_search_state(tree, TREE_PATH)
            continue
        nid, dup, r = eval_and_add(tree, parent_id, mutation, child_cfg, lineage_id=lineage_id)
        if nid is None:
            print(f"DEDUP <- parent#{parent_id} {LINEAGE_NAMES.get(lineage_id, lineage_id)} (dup #{dup})")
            continue
        gb = hv3.global_best(tree)
        budget = tree["search_state"]["budget"]
        print(f"#{nid} <- #{parent_id} {LINEAGE_NAMES.get(lineage_id, lineage_id):14s} "
              f"MAPE={r['score']} {r['status']} | best={gb['score']} (#{gb['id']}) "
              f"| phase={budget['phase']} n={hv3.n_evaluated(tree)}")

    gb = hv3.global_best(tree)
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated"
                 and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated"
                  and n["config"].get("kind") == "blend")
    tree["dedup_rejections"] = _dedup_rejections(tree)
    tree["boundary_candidates_log"] = BOUNDARY_LOG
    tree["cost_guard_fired"] = tree["search_state"].get("cost_guard_log", [])

    evals_to_match, running = None, None
    for i, n in enumerate([n for n in tree["nodes"] if n["status"] == "evaluated"], start=1):
        running = n["score"] if running is None else min(running, n["score"])
        if running <= LINEAR_BEST and evals_to_match is None:
            evals_to_match = i
    tree["evals_to_match_linear_best"] = evals_to_match
    hv3.save_search_state(tree, TREE_PATH)

    print(f"\nDone. {hv3.n_evaluated(tree)} evaluated ({n_solo} solo / {n_blend} blend), "
          f"wall={time.time() - t_start:.1f}s")
    print(f"Global best: #{gb['id']} MAPE={gb['score']}")
    print(f"  mutation: {gb['mutation'][:200]}")
    print(f"  config: {gb['config']}")
    print(f"Linear best: {LINEAR_BEST} — evals to match: {evals_to_match}")
    print(f"Budget/phase: {tree['search_state']['budget']}")
    print(f"Dedup rejections: {len(tree['dedup_rejections'])}")
    print(f"Cost-guard fired: {len(tree['cost_guard_fired'])} time(s)")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])}):")
    for e in tree["search_state"]["backtrack_log"]:
        print("  ", e)


if __name__ == "__main__":
    main()
