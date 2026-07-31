"""tree_search/run_citd_v3.py -- Stage-4 harness_v3 tree-search driver for
cat-in-the-dat (all-categorical binary classification, ROC-AUC, MAXIMIZE).

Follows the 07_tree_search.md driver checklist, modeled on run_s6e2_v3.py:
  - Root: linear-stage best solo (lr C=0.1 on full OHE, cached OOF AUC 0.803103),
    RETRAINED via eval_citd in a subprocess and asserted digit-for-digit.
  - Full linear pool reseeded as first-generation cached-OOF nodes (zero compute)
    per the afsis run-1 lesson: blend ceiling is structurally capped if the pool
    isn't fully reseeded (knowledge/experience.md, CV 設計 section).
  - Lineages: LRC (C exploration + boundary push), LRPAIRS (comp-local NEW idea:
    pairwise interaction crosses appended to the OHE), LGBTE (LGB-on-TE params),
    SEEDBAG (lgb_te reseed), BLEND (root + pool). Explore burst: kitchen-sink
    mega-blend + 2 long-shot solos (sanity-gated).

SIGN CONVENTION: harness score = -AUC (lower better). result["auc"] human-readable.
Run: `uv run python3 tree_search/run_citd_v3.py` (resumable; state at
competitions/cat-in-the-dat/experiments_tree_v3.json). `--dry-run` = wiring only.
"""
import copy
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402
import eval_citd as ev  # noqa: E402

COMP = "cat-in-the-dat"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_v3.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_citd.py")
MEMBERS_DIR = os.path.join(_COMP_DIR, "data_proc", "members")
EVAL_TIMEOUT_S = 900
MAX_WALL_S = 3 * 60 * 60
ITER_SAFETY_CAP = 200

ROOT_KNOWN_AUC = 0.803103  # scripts/members.py lr_c010 cached OOF (experiments.json #1)

LINEAR_POOL = {  # name -> (cached npz stem, canonical config)
    "POOL_LR005": ("lr_c005", {"kind": "solo", "model": "lr",
                               "params": {"C": 0.05, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-5},
                               "features": {"variant": "base"}}),
    "POOL_LR020": ("lr_c020", {"kind": "solo", "model": "lr",
                               "params": {"C": 0.2, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-5},
                               "features": {"variant": "base"}}),
    "POOL_LGBCAT": ("lgb_base", {"kind": "solo", "model": "lgb_cat", "params": {},
                                 "features": {"variant": "base"}}),
    "POOL_LGBTE": ("lgb_te", {"kind": "solo", "model": "lgb_te", "params": {},
                              "features": {"variant": "base"}}),
}

LR_SEARCH_SPACE = {"C": {"low": 0.05, "high": 0.3, "log": True}}


def dc(x):
    return copy.deepcopy(x)


def core(cfg):
    out = {k: v for k, v in cfg.items() if k != "result"}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def _driver_state(tree):
    return tree.setdefault("search_state", {}).setdefault("driver_state", {})


def _dedup_rejections(tree):
    return _driver_state(tree).setdefault("dedup_rejections", [])


def _node_results(tree):
    return tree.setdefault("search_state", {}).setdefault("node_results", {})


def result_of(tree, nid):
    return _node_results(tree).get(str(nid))


def auc_of(info):
    if info is None:
        return None
    if info.get("result") and "auc" in info["result"]:
        return info["result"]["auc"]
    return -info["score"] if info.get("score") is not None else None


def lr_cfg(C, variant="base", tol=1e-5):
    return {"kind": "solo", "model": "lr",
            "params": {"C": C, "solver": "lbfgs", "max_iter": 2000, "tol": tol},
            "features": {"variant": variant}}


ROOT_CFG = lr_cfg(0.1)


# ---------------------------------------------------------------------------
# eval_and_add
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, proposal_cfg, is_root=False):
    stored = core(proposal_cfg)
    if not is_root:
        dup_id = hv3.find_duplicate_config(tree, stored)
        if dup_id is not None:
            _dedup_rejections(tree).append(dict(mutation=mutation, dup_id=dup_id))
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
        if status == "failed":
            mutation = mutation + f" [ERROR: {r.get('error')}]"
    elif kind == "blend":
        try:
            t0 = time.time()
            best_w, best_neg, _oofs, warning = hv3.eval_blend_with_cost_guard(
                ev.CACHE_DIR, stored["members"], lambda vec: -ev.auc(ev._y, vec), tree=tree)
            best_score = -best_neg
            wall_s = time.time() - t0
            result = dict(members=stored["members"],
                          weights=[round(float(w), 4) for w in best_w],
                          auc=round(best_score, 6))
            if warning:
                result["cost_guard_warning"] = warning
            score, status = round(-best_score, 6), "evaluated"
        except Exception as e:  # noqa: BLE001
            score, status, wall_s, result = None, "failed", 0.0, None
            mutation = mutation + f" [ERROR: {type(e).__name__}: {e}]"
    else:
        raise ValueError(f"unknown node kind {kind!r}")

    if is_root:
        real_nid = hv3.add_root(tree, mutation, stored, score, status, wall_s)
    else:
        real_nid, dup = hv3.add_node(tree, parent_id, mutation, stored, score, status, wall_s)
        assert dup is None, f"unexpected post-eval dedup (dup=#{dup})"
    assert real_nid == nid
    if result is not None:
        _node_results(tree)[str(real_nid)] = result
    hv3.save_search_state(tree, TREE_PATH)
    return real_nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


def add_pool_node(tree, root_id, name, npz_stem, cfg):
    """Reseed a linear-pool member as a cached-OOF first-generation node (zero compute)."""
    stored = core(cfg)
    if hv3.find_duplicate_config(tree, stored) is not None:
        return None
    d = np.load(os.path.join(MEMBERS_DIR, f"{npz_stem}.npz"))
    oof, pred = d["oof"], d["test"]
    score = ev.auc(ev._y, oof)  # recompute = digit-verified vs experiments.json
    nid = hv3.next_id(tree)
    hv2.cache_oof(ev.CACHE_DIR, nid, oof, pred=pred, auc=score)
    real_nid, dup = hv3.add_node(
        tree, root_id,
        f"[{name}] linear-pool reseed (cached OOF reuse, scripts/members.py {npz_stem}; "
        f"zero retraining) [PRIOR: full-pool reseed lesson, afsis run-1 vs run-2]",
        stored, round(-score, 6), "evaluated", 0.0)
    assert dup is None and real_nid == nid
    _node_results(tree)[str(real_nid)] = {"auc": round(score, 6)}
    hv3.save_search_state(tree, TREE_PATH)
    return real_nid


# ---------------------------------------------------------------------------
# lineage seeds + mutation queues
# ---------------------------------------------------------------------------
def seed_lrc():
    return (lr_cfg(0.15),
            "C exploration between the linear-stage grid points (0.1 best of "
            "0.05/0.1/0.2, curvature suggests optimum in (0.1, 0.2))")


def seed_lrpairs():
    return (lr_cfg(0.1, variant="pairs"),
            "comp-local NEW idea: append all pairwise interaction crosses of the 17 "
            "low-card columns to the OHE (~5-6k extra dummies) -- LR cannot learn "
            "interactions on its own; this is the standard lever OHE+LR lacks")


def seed_lgbte():
    return ({"kind": "solo", "model": "lgb_te",
             "params": {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 100},
             "features": {"variant": "base"}},
            "shallower/slower LGB-on-TE (leaves 127->63, lr 0.05->0.03, mcs 50->100) -- "
            "regularization direction per experience.md AUC section")


def seed_seedbag():
    return ({"kind": "solo", "model": "lgb_te", "params": {"seed": 2024},
             "features": {"variant": "base"}},
            "seed variation of pool lgb_te (seed 42->2024) [PRIOR: seed bagging is the "
            "cheapest residual gain, knowledge/experience.md]")


def _pool_ids(tree):
    out = {}
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["status"] == "evaluated":
            for name in list(LINEAR_POOL) + ["LRC", "LRPAIRS", "LGBTE", "SEEDBAG", "BLEND",
                                             "EXPL_MEGABLEND", "EXPL_DEEPLGB", "EXPL_LRLOWREG"]:
                if n["mutation"].startswith(f"[{name}]"):
                    out.setdefault(name, n["id"])
    return out


def seed_blend(tree):
    ids = _pool_ids(tree)
    members = sorted({tree["root_id"], ids.get("POOL_LGBTE"), ids.get("POOL_LR020")} - {None})
    return ({"kind": "blend", "members": members, "weight_search": "dirichlet"},
            f"ensemble seed: root + pool lgb_te + pool lr_c020 ({members})")


def solo_pool(tree):
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def _best_in_lineage(tree, lineage_id):
    best = None
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["id"] == tree["root_id"]:
            continue
        if hv3.lineage_of(tree, n["id"]) == lineage_id:
            if best is None or n["score"] < best["score"]:
                best = n
    return best


def _bump_params(cfg, **kw):
    c = dc(cfg)
    c.setdefault("params", {}).update(kw)
    return c


def _lrc_boundary(tree, parent_node):
    cfg = parent_node["config"]
    if cfg.get("model") != "lr":
        return None
    edges = hv3.boundary_candidates(cfg, LR_SEARCH_SPACE)
    if not edges:
        return None
    p = dc(cfg)
    descs = []
    for e in edges:
        p["params"][e["param"]] = round(e["new_value"], 4)
        descs.append(f"{e['param']} {e['old_value']}->{round(e['new_value'], 4)}")
    return p, f"boundary_candidates push: {', '.join(descs)} [feature 4]"


LRC_QUEUE = [
    lambda t, n: (lr_cfg(0.12), "LRC: C=0.12 (finer grid inside (0.1, 0.15))"),
    lambda t, n: _lrc_boundary(t, n),
    lambda t, n: (lr_cfg(0.08), "LRC: C=0.08 (probe below 0.1)"),
]
LRPAIRS_QUEUE = [
    lambda t, n: (lr_cfg(0.15, variant="pairs"), "LRPAIRS: pairs variant, C=0.15"),
    lambda t, n: (lr_cfg(0.1, variant="pairs_ord5"),
                  "LRPAIRS: add ord_5 x 12-partner crosses on top of pairs"),
    lambda t, n: (lr_cfg(0.05, variant="pairs"),
                  "LRPAIRS: pairs variant, stronger reg C=0.05 (more dummies may need it)"),
]
LGBTE_QUEUE = [
    lambda t, n: (_bump_params(n["config"], smoothing=5),
                  "LGBTE: weaker TE smoothing 20->5 (high-card noms have median count 25+)"),
    lambda t, n: (_bump_params(n["config"], num_leaves=255, learning_rate=0.02),
                  "LGBTE: capacity direction (leaves->255, lr->0.02) -- 300k rows may "
                  "reward capacity per s3e11 lesson"),
]
SEEDBAG_QUEUE = [
    lambda t, n: ({"kind": "solo", "model": "lgb_te", "params": {"seed": 777},
                   "features": {"variant": "base"}}, "SEEDBAG: third seed 777"),
]


def _lr_c_ladder_fallback(tree, parent_cfg, attempt):
    ladder = [0.13, 0.11, 0.14, 0.09, 0.18, 0.25, 0.07, 0.35, 0.5]
    variant = (parent_cfg.get("features") or {}).get("variant", "base")
    for c in ladder:
        cand = lr_cfg(c, variant=variant)
        if hv3.find_duplicate_config(tree, core(cand)) is None:
            return cand, f"fallback: next untried C={c} on variant '{variant}'"
    return None


def solo_fallback(tree, parent_node, attempt):
    cfg = parent_node["config"]
    if cfg.get("model") == "lr":
        return _lr_c_ladder_fallback(tree, cfg, attempt)
    c = _bump_params(cfg, seed=4000 + attempt)
    return c, f"fallback seed-variation: seed={4000 + attempt}"


def _mk_blend_add_best(name):
    def fn(tree, parent_node):
        ids = _pool_ids(tree)
        if name not in ids:
            return None
        best = _best_in_lineage(tree, ids[name])
        add_id = best["id"] if best else ids[name]
        members = parent_node["config"]["members"]
        if add_id in members:
            return None
        return ({"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": "dirichlet"}, f"BLEND: add best-of-{name} (#{add_id})")
    return fn


def _blend_remove_weakest(tree, parent_node):
    res = result_of(tree, parent_node["id"]) or {}
    weights = res.get("weights")
    members = parent_node["config"]["members"]
    if not weights or len(members) <= 2:
        return None
    idx = int(np.argmin(weights))
    child = dc(parent_node["config"])
    child["members"] = sorted([m for m in members if m != members[idx]])
    return child, f"BLEND: remove lowest-weight member #{members[idx]} (w={weights[idx]:.3f})"


def blend_fallback(tree, parent_node, attempt):
    pool = solo_pool(tree)
    members = parent_node["config"]["members"]
    for add_id in [nid for nid in pool if nid not in members]:
        child = {"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": "dirichlet"}
        if hv3.find_duplicate_config(tree, child) is None:
            return child, f"BLEND fallback: add next-best unused pool member #{add_id}"
    return None


BLEND_QUEUE = [_mk_blend_add_best("LRPAIRS"), _mk_blend_add_best("LRC"),
               _mk_blend_add_best("LGBTE"), _mk_blend_add_best("SEEDBAG"),
               _blend_remove_weakest]

SOLO_QUEUES = {"LRC": LRC_QUEUE, "LRPAIRS": LRPAIRS_QUEUE, "LGBTE": LGBTE_QUEUE,
               "SEEDBAG": SEEDBAG_QUEUE}
LINEAGE_NAMES = {}


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES.get(lineage_id, f"L{lineage_id}")
    idx = hv3.lineage_size(tree, lineage_id) - 1
    kind = next(n for n in tree["nodes"] if n["id"] == lineage_id)["config"].get("kind", "solo")
    if kind == "blend":
        result = None
        if idx < len(BLEND_QUEUE):
            result = BLEND_QUEUE[idx](tree, parent_node)
        if result is None:
            result = blend_fallback(tree, parent_node, idx)
        if result is None:
            return None, None
        child_cfg, desc = result
    else:
        queue = SOLO_QUEUES.get(name, [])
        result = queue[idx](tree, parent_node) if idx < len(queue) else None
        if result is None:
            result = solo_fallback(tree, parent_node, idx)
        if result is None:
            return None, None
        child_cfg, desc = result
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


def inject_explore_burst(tree):
    pool = solo_pool(tree)
    seeds = []
    if len(pool) >= 2:
        seeds.append(({"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet"},
                      f"[EXPL_MEGABLEND] explore-burst: kitchen-sink blend of the entire "
                      f"solo pool ({len(pool)} members) [PRIOR: mandatory mega-blend, "
                      f"07_tree_search.md Section 3]"))
    seeds.append(({"kind": "solo", "model": "lgb_te",
                   "params": {"num_leaves": 511, "learning_rate": 0.02,
                              "min_child_samples": 20, "feature_fraction": 0.6},
                   "features": {"variant": "base"}},
                  "[EXPL_DEEPLGB] explore-burst long-shot: much deeper LGB-on-TE "
                  "(leaves->511, lr->0.02, mcs->20, ff->0.6)"))
    seeds.append((lr_cfg(1.0, variant="pairs_ord5"),
                  "[EXPL_LRLOWREG] explore-burst long-shot: near-unregularized LR "
                  "(C=1.0) on the richest feature set (pairs_ord5)"))
    return seeds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def dry_run():
    print("[dry-run] root cfg:", ROOT_CFG, "known AUC", ROOT_KNOWN_AUC)
    for name, fn in [("LRC", seed_lrc), ("LRPAIRS", seed_lrpairs),
                     ("LGBTE", seed_lgbte), ("SEEDBAG", seed_seedbag)]:
        _cfg, desc = fn()
        print(f"  {name}: {desc}")
    print("[dry-run] pool:", {k: v[0] for k, v in LINEAR_POOL.items()})
    priors = hv2.suggest_priors({"metric": "auc", "tags": ["categorical", "binary", "encoding"],
                                 "data_type": "tabular"})
    print(f"[dry-run] suggest_priors hits: {len(priors)}")
    for p in priors[:6]:
        print("   -", p[:140])
    print("[dry-run] OK")


def main():
    global MAX_WALL_S
    if "--wall" in sys.argv:
        MAX_WALL_S = int(sys.argv[sys.argv.index("--wall") + 1])
    t_start = time.time()
    resumed = os.path.exists(TREE_PATH)
    tree = hv3.load_search_state(TREE_PATH) if resumed else hv3.new_tree(COMP)
    hv3.init_budget(tree)

    priors = hv2.suggest_priors({"metric": "auc", "tags": ["categorical", "binary", "encoding"],
                                 "data_type": "tabular"})
    print(f"[priors] {len(priors)} experience.md hits (queried once, per 07_tree_search.md §5)")

    if not resumed:
        mutation = (f"root: linear-stage best solo (lr C=0.1 full OHE), retrained via "
                    f"eval_citd for digit-for-digit verification vs cached {ROOT_KNOWN_AUC}")
        nid, _, info = eval_and_add(tree, None, mutation, ROOT_CFG, is_root=True)
        got = auc_of(info)
        assert got is not None and round(got, 6) == round(ROOT_KNOWN_AUC, 6), (
            f"ROOT VERIFICATION FAILED: retrained AUC {got} != cached {ROOT_KNOWN_AUC}")
        print(f"[root] verified digit-for-digit: {got:.6f} == {ROOT_KNOWN_AUC:.6f}")
    else:
        print(f"[resume] {len(tree['nodes'])} nodes ({hv3.n_evaluated(tree)} evaluated)")

    root_id = tree["root_id"]
    existing = _pool_ids(tree)
    for name, nid in existing.items():
        LINEAGE_NAMES[nid] = name

    # linear pool reseed (cached OOF, zero compute)
    for name, (stem, cfg) in LINEAR_POOL.items():
        if name in existing:
            continue
        nid = add_pool_node(tree, root_id, name, stem, cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            print(f"[pool {name}] node #{nid} AUC={-tree['nodes'][-1]['score']:.6f} (cached reuse)")

    # authored lineage seeds
    for name, fn in [("LRC", seed_lrc), ("LRPAIRS", seed_lrpairs),
                     ("LGBTE", seed_lgbte), ("SEEDBAG", seed_seedbag)]:
        if name in existing:
            continue
        cfg, desc = fn()
        nid, _dup, info = eval_and_add(tree, root_id, f"[{name}] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            print(f"[seed {name}] node #{nid} AUC={auc_of(info)}")
    if "BLEND" not in existing:
        cfg, desc = seed_blend(tree)
        nid, _dup, info = eval_and_add(tree, root_id, f"[BLEND] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "BLEND"
            print(f"[seed BLEND] node #{nid} AUC={auc_of(info)}")

    burst_injected = bool(_driver_state(tree).get("burst_injected"))
    it = 0
    while not hv3.should_stop(tree) and it < ITER_SAFETY_CAP and (time.time() - t_start) < MAX_WALL_S:
        it += 1
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not burst_injected:
            print(f"\n>>> PHASE -> explore_burst (n_eval={hv3.n_evaluated(tree)})")
            for cfg, desc in inject_explore_burst(tree):
                name = desc.split("]")[0][1:]
                nid, _dup, info = eval_and_add(tree, root_id, desc, cfg)
                if nid is not None:
                    LINEAGE_NAMES[nid] = name
                    print(f"[burst {name}] node #{nid} AUC={auc_of(info)}")
                    if cfg.get("kind") == "solo" and auc_of(info) is not None:
                        ok, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
                        print(f"    sanity gate: {'PASS' if ok else 'FAIL (plateaued)'} (bound={bound})")
            _driver_state(tree)["burst_injected"] = True
            burst_injected = True
            hv3.save_search_state(tree, TREE_PATH)
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            print("[stop] select_next_parent returned None")
            break
        if lineage_id not in LINEAGE_NAMES:
            for n in tree["nodes"]:
                if n["id"] == lineage_id and n["parent_id"] == root_id:
                    LINEAGE_NAMES[lineage_id] = n["mutation"].split("]")[0].lstrip("[")
        child_cfg, desc = propose_child(tree, parent_id, lineage_id)
        if child_cfg is None:
            st = tree["search_state"]
            plat = set(st.get("plateaued", []))
            if lineage_id not in plat:
                plat.add(lineage_id)
                st["plateaued"] = sorted(plat)
                st.setdefault("backtrack_log", []).append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=f"propose_child exhausted for lineage "
                           f"{LINEAGE_NAMES.get(lineage_id, lineage_id)} (driver-side plateau)"))
                print(f"[plateau] lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} exhausted")
            if st.get("active_lineage") == lineage_id:
                st["active_lineage"] = None
            hv3.save_search_state(tree, TREE_PATH)
            continue
        nid, _dup, info = eval_and_add(tree, parent_id, desc, child_cfg)
        if nid is None:
            continue
        gb = hv3.global_best(tree)
        print(f"[eval {it}] node #{nid} (parent #{parent_id}, "
              f"{LINEAGE_NAMES.get(lineage_id, lineage_id)}) AUC={auc_of(info)}  "
              f"n_eval={hv3.n_evaluated(tree)}  best={-gb['score'] if gb else None:.6f}")

    gb = hv3.global_best(tree)
    print(f"\n=== stopped: phase={tree['search_state']['budget']['phase']} "
          f"reason={tree['search_state']['budget'].get('stop_reason')} ===")
    print(f"n_evaluated={hv3.n_evaluated(tree)}  wall={time.time()-t_start:.1f}s")
    print(f"global best: node #{gb['id']}  AUC={-gb['score']:.6f}  kind={gb['config'].get('kind')}")
    print(f"dedup_rejections={len(_dedup_rejections(tree))}")
    print(f"backtrack_log={json.dumps(tree['search_state'].get('backtrack_log', []), indent=1)}")
    print(f"cost_guard_log={tree['search_state'].get('cost_guard_log', [])}")
    hv3.save_search_state(tree, TREE_PATH)
    print(f"\nRESULT {json.dumps(dict(n_evaluated=hv3.n_evaluated(tree), best_node=gb['id'], best_auc=-gb['score']))}")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run()
    else:
        main()
