"""tree_search/run_tpsjan22_v3.py — v3 tree-search driver for
tabular-playground-series-jan-2022 (SMAPE, minimize; structural-Ridge-dominant comp).

Follows the run_s3e7_v3.py template + references/07_tree_search.md checklist:
  1. Root digit-verify: ridge_hol (linear champion) must reproduce pooled SMAPE
     4.415985 to 6dp or the run aborts.
  2. No prior tree exists for this comp — no OOF reuse path.
  3. Resume-state via hv3.save_search_state/load_search_state; driver bookkeeping in
     tree["search_state"]["driver_state"].
  4. Solo evals through hv3.eval_solo_subprocess (OS-level timeout).
  5. Burst seeds gated by hv3.apply_burst_seed_sanity_gate.

Node space: structure-dominant comp (07_tree_search.md §2 row 2) — solo params are
structural knobs (Fourier orders, holiday window, per-product split, GDP offset,
recency weighting, ridge alpha) plus one GBDT (GDP-normalized LGB) diversity lineage;
blend nodes mix cached solo OOFs via v3's k=800+coordinate-ascent weight search.
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
import eval_tpsjan22 as ev  # noqa: E402

COMP = "tabular-playground-series-jan-2022"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_v3.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_tpsjan22.py")
EVAL_TIMEOUT_S = 300
MAX_WALL_S = 3600 * 2
ITER_SAFETY_CAP = 400
ROOT_SMAPE = 4.415985          # linear champion (ridge_hol), digit-verified
LINEAR_BEST = 4.415985         # linear iteration best == root (blend collapsed to ridge)

LINEAGE_NAMES = {}             # node_id -> lineage name (rebuilt on resume)
BURST_INJECTED = [False]
BOUNDARY_LOG = []

RIDGE_SEARCH_SPACE = {
    "alpha": (0.01, 1.0), "fourier_k": (2, 12), "hol_lo": (-10, 0), "hol_hi": (0, 15),
}
LGB_SEARCH_SPACE = {
    "num_leaves": (7, 127), "lr": (0.01, 0.1), "n_estimators": (200, 2000),
    "min_child_samples": (5, 100),
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


def evaluate_blend_v3(tree, stored_cfg):
    members = stored_cfg["members"]
    best_w, best_s, oofs = hv3.eval_blend_with_cost_guard(
        ev.CACHE_DIR, members, metric_fn, tree=tree)[:3]
    # (warning captured below via the 4-tuple; re-call cheaply avoided by unpacking)
    return best_w, best_s


def eval_and_add(tree, parent_id, mutation, proposal_cfg, is_root=False, lineage_id=None):
    stored = core(proposal_cfg)
    if not is_root:
        dup_id = hv3.find_duplicate_config(tree, stored)
        if dup_id is not None:
            _dedup_rejections(tree).append(dict(mutation=mutation, dup_id=dup_id))
            if lineage_id is not None:
                off = _dedup_offset(tree)
                off[str(lineage_id)] = off.get(str(lineage_id), 0) + 1
            nid_null, dup_echo = hv3.add_node(tree, parent_id, mutation + " [dedup pre-check]",
                                              stored, None, "failed", 0.0)
            assert nid_null is None
            hv3.save_search_state(tree, TREE_PATH)
            return None, dup_id, None

    nid = hv3.next_id(tree)
    kind = stored.get("kind", "solo")
    if kind == "solo":
        r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, proposal_cfg, EVAL_TIMEOUT_S, node_id=nid)
        score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
        full_mutation = mutation
    elif kind == "blend":
        try:
            t0 = time.time()
            best_w, best_s, oofs, warning = hv3.eval_blend_with_cost_guard(
                ev.CACHE_DIR, stored["members"], metric_fn, tree=tree)
            wall_s = round(time.time() - t0, 1)
            score, status = round(best_s, 6), "evaluated"
            result = {"smape": round(best_s, 6),
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
ROOT_PARAMS = {"alpha": 0.1, "fourier_k": 8, "fourier_product": True,
               "holidays": True, "hol_lo": -5, "hol_hi": 10}
ROOT_CONFIG = {"kind": "solo", "model": "ridge", "params": dc(ROOT_PARAMS)}
ROOT_MUTATION = ("root: linear-iteration champion ridge_hol — structural Ridge on "
                 "log(num_sold): log_gdp + year_c + store + product + dow + Fourier(8)"
                 "xproduct + per-holiday-name dummies shifted -5..+10, pooled SMAPE "
                 "over 2017+2018 expanding-year folds, round-to-int in metric")


def _p(params_over):
    c = dc(ROOT_CONFIG)
    c["params"].update(params_over)
    return c


SOLO_SEED_SPECS = [
    ("ALPHA", lambda: (_p({"alpha": 0.03}),
     "alpha 0.1->0.03 — weaker shrinkage; structural design is small-p, regularization "
     "mostly hurts level coefficients [PRIOR none — comp-local]")),
    ("FOURK", lambda: (_p({"fourier_k": 10}),
     "fourier_k 8->10 — finer within-year seasonality [PRIOR none — comp-local]")),
    ("HOLWIN", lambda: (_p({"hol_lo": -7, "hol_hi": 12}),
     "holiday window -5..+10 -> -7..+12 — experience.md tpsjan22: post-holiday lift "
     "persists +3..+7d, window width was the single biggest lever [PRIOR tpsjan22 hol]")),
    ("PERPROD", lambda: (_p({"per_product": True}),
     "per-product separate Ridge fits — full product interaction on every term, not "
     "just Fourier [PRIOR none — comp-local structural]")),
    ("GDPOFF", lambda: (_p({"gdp_offset": True}),
     "fix GDP elasticity at 1 (offset) instead of learning the log_gdp coefficient "
     "under L2 shrinkage [PRIOR tpsjan22 gdp recipe]")),
    ("DOWPROD", lambda: (_p({"dow_product": True}),
     "dow x product interaction — EDA showed dow effect identical across "
     "stores/countries but products untested [PRIOR none — comp-local]")),
    ("RECENT", lambda: (_p({"recent_weight": 0.85}),
     "recency sample-weighting 0.85^years-back — extrapolation favors recent structure "
     "[PRIOR none — comp-local]")),
    ("LGBDIV", lambda: ({"kind": "solo", "model": "lgb",
                          "params": {"num_leaves": 31, "lr": 0.05, "n_estimators": 800,
                                     "min_child_samples": 20, "seed": 42}},
     "GDP-normalized LGB (target log(num_sold/gdp_pc)) — diversity member for blends "
     "[PRIOR tpsjan22 gdp recipe; experience.md: GBDT solo 6.33 but blend diversity]")),
]

SOLO_QUEUES = {
    "ALPHA": [lambda c: (_bump(c, alpha=0.01),
                          "ALPHA: 0.03->0.01 toward OLS [PRIOR none]")],
    "FOURK": [lambda c: (_bump(c, fourier_k=12),
                          "FOURK: 10->12 (box edge) [PRIOR none]")],
    "HOLWIN": [lambda c: (_bump(c, hol_lo=-10, hol_hi=15),
                           "HOLWIN: push to box edge -10..+15 [PRIOR tpsjan22 hol]")],
    "PERPROD": [lambda c: (_bump(c, dow_product=False, per_product=True, alpha=0.03),
                            "PERPROD: per_product + alpha 0.03 (smaller per-fit n) [PRIOR none]")],
    "GDPOFF": [],
    "DOWPROD": [],
    "RECENT": [lambda c: (_bump(c, recent_weight=0.7),
                           "RECENT: 0.85->0.7 stronger recency [PRIOR none]")],
    "LGBDIV": [lambda c: (_bump(c, num_leaves=63, lr=0.03, n_estimators=1500),
                           "LGBDIV: more capacity, slower lr [PRIOR none]")],
}


def _bump(cfg, **kw):
    c = dc(cfg)
    c["params"].update(kw)
    return c


RIDGE_FALLBACK_NUDGES = [
    lambda p: {"alpha": round(p.get("alpha", 0.1) * 0.5, 5)},
    lambda p: {"alpha": round(p.get("alpha", 0.1) * 2.0, 5)},
    lambda p: {"fourier_k": p.get("fourier_k", 8) + 1},
    lambda p: {"hol_hi": p.get("hol_hi", 10) + 2},
    lambda p: {"hol_lo": p.get("hol_lo", -5) - 2},
    lambda p: {"year_c2": not p.get("year_c2", False)},
    lambda p: {"fourier_country": not p.get("fourier_country", False)},
    lambda p: {"recent_weight": 0.9 if p.get("recent_weight") is None else None},
]
LGB_FALLBACK_NUDGES = [
    lambda p: {"num_leaves": max(7, p.get("num_leaves", 31) // 2)},
    lambda p: {"min_child_samples": p.get("min_child_samples", 20) * 2},
    lambda p: {"lr": round(p.get("lr", 0.05) * 0.6, 4), "n_estimators": 2000},
    lambda p: {"seed": p.get("seed", 42) + 1000},
]


def solo_fallback(parent_cfg, attempt):
    nudges = LGB_FALLBACK_NUDGES if parent_cfg.get("model") == "lgb" else RIDGE_FALLBACK_NUDGES
    n = nudges[attempt % len(nudges)](parent_cfg.get("params", {}))
    n = {k: v for k, v in n.items() if v is not None}
    if not n:
        n = {"alpha": round(parent_cfg["params"].get("alpha", 0.1) * (3 + attempt), 5)}
    c = _bump(parent_cfg, **n)
    return c, f"fallback nudge {n}"


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
    best_non_root = [i for i in pool if i != tree["root_id"] and i != ids.get("LGBDIV")][:2]
    members = sorted(set([tree["root_id"], ids["LGBDIV"]] + best_non_root))
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    desc = (f"ensemble seed: root + LGBDIV (diversity) + top-2 structural variants "
            f"{best_non_root} [PRIOR: blends beat solos in 9/10 comps]")
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
        ("EXPL_HUGEK", _p({"fourier_k": 16, "alpha": 0.3}),
         "long-shot: Fourier order 16 (past box edge) with alpha raised to 0.3 to "
         "compensate — test whether seasonality detail was capped"),
        ("EXPL_OLS", _p({"alpha": 0.001}),
         "long-shot: near-OLS (alpha 0.001) — structural design may need no shrinkage"),
        ("EXPL_COMBO", _p({"per_product": True, "hol_lo": -7, "hol_hi": 12,
                            "gdp_offset": True}),
         "long-shot: stack the individually-promising structural levers "
         "(per_product + wide holidays + gdp offset) in one config"),
        ("EXPL_LGBBIG", {"kind": "solo", "model": "lgb",
                          "params": {"num_leaves": 127, "lr": 0.02, "n_estimators": 2500,
                                     "min_child_samples": 10, "seed": 42}},
         "long-shot: high-capacity LGB — blend diversity even if solo is weak"),
    ]


ALL_LINEAGE_NAMES = [name for name, _ in SOLO_SEED_SPECS] + ["BLEND"]
BURST_NAMES = [n for n, _, _ in _burst_seeds()] + ["EXPL_MEGABLEND"]
ALL_LINEAGE_NAMES.extend(BURST_NAMES)


def inject_explore_burst(tree, root_id):
    for name, cfg, desc in _burst_seeds():
        nid, dup, r = eval_and_add(tree, root_id, f"[{name}] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            SOLO_QUEUES.setdefault(name, [])
            print(f"[BURST {name}] #{nid} SMAPE={r['score']} status={r['status']}")
            passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
            if not passed:
                print(f"[BURST {name}] #{nid} FAILED sanity gate (bound={bound}) — plateaued")
    pool = solo_pool(tree)
    if len(pool) >= 2:
        cfg = {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet"}
        nid, dup, r = eval_and_add(tree, root_id, "[EXPL_MEGABLEND] long-shot: "
                                   f"kitchen-sink blend of the entire {len(pool)}-member "
                                   "solo pool (k=800+coord-ascent)", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "EXPL_MEGABLEND"
            hv3.apply_burst_seed_sanity_gate(tree, nid)
            print(f"[BURST EXPL_MEGABLEND] #{nid} SMAPE={r['score']} status={r['status']}")


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        print(f"Resuming tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv3.new_tree(COMP)
        hv3.init_budget(tree)
        print(f"init_budget -> {tree['search_state']['budget']}")

    priors = hv2.suggest_priors({"metric": "SMAPE", "tags": ["timeseries", "panel", "structural"],
                                 "data_type": "tabular"})
    print(f"suggest_priors: {len(priors)} hits")
    for p in priors[:8]:
        print("  PRIOR:", str(p)[:160])

    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} SMAPE={r['score']} status={r['status']}")
        assert r["status"] == "evaluated"
        assert round(r["score"], 6) == ROOT_SMAPE, (
            f"ROOT DIGIT-VERIFY FAILED: expected {ROOT_SMAPE}, got {r['score']}")
        print(f"ROOT DIGIT-VERIFIED: {ROOT_SMAPE}. OK.")

    if not BOUNDARY_LOG:
        BOUNDARY_LOG.extend(hv3.boundary_candidates({"params": ROOT_PARAMS}, RIDGE_SEARCH_SPACE))
    print(f"boundary_candidates(root vs ridge box) -> {BOUNDARY_LOG}")

    for name, seed_fn in SOLO_SEED_SPECS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} SMAPE={r['score'] if r else 'dup'} ")

    already_blend = any(n["mutation"].startswith("[BLEND]") for n in tree["nodes"]
                        if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} SMAPE={r['score'] if r else 'dup'}")

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
        print(f"#{nid} <- #{parent_id} {LINEAGE_NAMES.get(lineage_id, lineage_id):12s} "
              f"SMAPE={r['score']} {r['status']} | best={gb['score']} (#{gb['id']}) "
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
    print(f"Global best: #{gb['id']} SMAPE={gb['score']}")
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
