"""Tree-search driver for tabular-playground-series-sep-2022 (harness_v3).

Root = ridge_struct baseline (digit-verified vs members_v1: fold2019 SMAPE 4.79072).
Lineages: RIDGE (regularization/Fourier/EOY knobs), HOL (holiday windows),
LEVEL (k estimator), LGBR (lgb_ratio params), LGBS (lgb_shape params),
DIV (diversity one-offs), BLEND (weighted mixes, k=800 dirichlet + coord ascent).

Node score = fold2019 SMAPE (postprocessed). Honest audits happen post-search.
"""
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness as hv1
import harness_v2 as hv2
import harness_v3 as hv3

COMP = "tabular-playground-series-sep-2022"
COMP_DIR = f"/home/tjyen/ai_agents/kaggle/competitions/{COMP}"
TREE_PATH = f"{COMP_DIR}/artifacts/tree_v3.json"
CACHE_DIR = os.path.join(_HERE, "cache_tssep22_main")
EVAL_PATH = os.path.join(_HERE, "eval_tssep22.py")
TIMEOUT_S = 300

_spec = importlib.util.spec_from_file_location("ev_tssep22", EVAL_PATH)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

Y = ev.fold_targets()
Y19 = Y["fold2019"]


def metric_fn(v):
    return ev.smape(Y19, v)


ROOT_CFG = {"kind": "solo", "model": "ridge_struct",
            "params": {"alpha": 3.0, "fourier_k_global": 4, "fourier_k_product": 4,
                       "kids_parity_k": 2, "eoy_start": 350, "eoy_end": 15,
                       "hol_before": 5, "hol_after": 10, "per_name": True},
            "level": {"k_est": "mean"}}
ROOT_EXPECT = 4.79072

RIDGE_SPACE = {"alpha": {"low": 0.03, "high": 100.0, "log": True},
               "fourier_k_global": (1, 6), "fourier_k_product": (1, 6),
               "kids_parity_k": (0, 6), "eoy_start": (335, 360), "eoy_end": (5, 25),
               "hol_before": (0, 12), "hol_after": (0, 16)}
LGB_SPACE = {"learning_rate": {"low": 0.01, "high": 0.2, "log": True},
             "num_leaves": (15, 255), "min_child_samples": (5, 80),
             "n_estimators": (300, 3000), "colsample_bytree": (0.5, 1.0),
             "subsample": (0.6, 1.0)}

SEEDS = [
    ("RIDGE", {"kind": "solo", "model": "ridge_struct",
               "params": {**ROOT_CFG["params"], "alpha": 1.0}, "level": {"k_est": "mean"}}),
    ("HOL", {"kind": "solo", "model": "ridge_struct",
             "params": {**ROOT_CFG["params"], "hol_before": 7, "hol_after": 12},
             "level": {"k_est": "mean"}}),
    ("LEVEL", {"kind": "solo", "model": "ridge_struct",
               "params": dict(ROOT_CFG["params"]), "level": {"k_est": "percountry"}}),
    ("LGBR", {"kind": "solo", "model": "lgb_ratio", "params": {}, "level": {"k_est": "mean"}}),
    ("LGBS", {"kind": "solo", "model": "lgb_shape", "params": {}, "level": {"k_est": "mean"}}),
    ("DIV", {"kind": "solo", "model": "lgb_join", "params": {}, "level": {"k_est": "mean"}}),
]


def core(cfg: dict) -> dict:
    c = json.loads(json.dumps(cfg))
    if c.get("kind") == "blend":
        c["members"] = sorted(c["members"])
    return c


def solo_pool(tree):
    return [n for n in tree["nodes"] if n["status"] == "evaluated"
            and n.get("kind", n["config"].get("kind")) == "solo" and n["score"] is not None]


def eval_and_add(tree, parent_id, mutation, cfg):
    cfg = core(cfg)
    dup = hv2.find_duplicate_config(tree, cfg)
    if dup is not None:
        nid, dup_id = hv3.add_node(tree, parent_id, mutation, cfg, None, "failed", 0.0)
        print(f"  dedup vs #{dup_id}")
        tree["search_state"].setdefault("driver_state", {})
        tree["search_state"]["driver_state"]["dedup_rejections"] = \
            tree["search_state"]["driver_state"].get("dedup_rejections", 0) + 1
        return None
    nid_guess = len(tree["nodes"]) + 1
    if cfg["kind"] == "blend":
        import time as _t
        t0 = _t.time()
        try:
            w, s, _oofs = hv3.eval_blend_with_cost_guard(CACHE_DIR, cfg["members"], metric_fn, tree=tree)
            nid, _ = hv3.add_node(tree, parent_id, mutation, cfg, round(float(s), 5),
                                  "evaluated", round(_t.time() - t0, 2), kind="blend")
            if nid is not None:
                node = next(n for n in tree["nodes"] if n["id"] == nid)
                node["blend_weights"] = [round(float(x), 5) for x in w]
                print(f"  #{nid} blend({len(cfg['members'])}) = {s:.5f}")
            return nid
        except ValueError as e:
            nid, _ = hv3.add_node(tree, parent_id, mutation, cfg, None, "failed",
                                  round(_t.time() - t0, 2), kind="blend")
            print(f"  blend failed: {e}")
            return None
    r = hv3.eval_solo_subprocess(EVAL_PATH, cfg, TIMEOUT_S, node_id=nid_guess)
    nid, _ = hv3.add_node(tree, parent_id, mutation, cfg, r["score"], r["status"],
                          r["wall_s"], kind="solo")
    if nid is not None and nid != nid_guess and r["status"] == "evaluated":
        os.rename(os.path.join(CACHE_DIR, f"solo_{nid_guess}.npz"),
                  os.path.join(CACHE_DIR, f"solo_{nid}.npz"))
    fs = (r.get("result") or {}).get("fold_scores")
    print(f"  #{nid} {mutation[:70]} -> {r['score']} {fs if fs else r.get('error','')}")
    return nid if r["status"] == "evaluated" else None


# ---------------- mutation proposers ----------------
RNG = np.random.default_rng(20260731)


def propose(tree, parent, lname):
    cfg = json.loads(json.dumps(parent["config"]))
    p = cfg.get("params", {})
    cands = []
    if lname == "BLEND" or cfg.get("kind") == "blend":
        pool = sorted(solo_pool(tree), key=lambda n: n["score"])
        prev = [n for n in tree["nodes"] if n.get("kind", n["config"].get("kind")) == "blend"
                and n["status"] == "evaluated"]
        used = {tuple(sorted(n["config"]["members"])) for n in prev}
        for k in (3, 4, 5, 6, 8, 10, len(pool)):
            mem = tuple(sorted(n["id"] for n in pool[:k]))
            if len(mem) >= 2 and mem not in used:
                return {"kind": "blend", "members": list(mem)}, f"blend top-{k} solos"
        return {"kind": "blend", "members": [n["id"] for n in pool]}, "blend kitchen-sink (dup-burn)"
    if cfg["model"] == "ridge_struct":
        if lname == "HOL":
            grid = [("hol_before", [3, 7, 10, 12]), ("hol_after", [7, 12, 14, 16]),
                    ("per_name", [False])]
        elif lname == "LEVEL":
            grid = [("__level__", ["percountry", "last", "mean"])]
        else:
            grid = [("alpha", [round(p["alpha"] / 3, 4), round(p["alpha"] * 3, 4)]),
                    ("fourier_k_global", [2, 6]), ("fourier_k_product", [2, 6]),
                    ("kids_parity_k", [0, 4]), ("eoy_start", [340, 345, 355]),
                    ("eoy_end", [10, 20]), ("dow_store", [True])]
        for name, vals in grid:
            for v in vals:
                c2 = json.loads(json.dumps(cfg))
                if name == "__level__":
                    c2["level"]["k_est"] = v
                    mut = f"level k_est={v}"
                else:
                    c2["params"][name] = v
                    mut = f"{name}={v}"
                cands.append((c2, mut))
        for bc in hv3.boundary_candidates(cfg, RIDGE_SPACE):
            c2 = json.loads(json.dumps(cfg))
            c2["params"][bc["param"]] = bc["new_value"]
            cands.append((c2, f"boundary-push {bc['param']}={bc['new_value']}"))
    else:  # lgb arms
        defaults = dict(learning_rate=0.05, num_leaves=64, min_child_samples=20,
                        n_estimators=900, colsample_bytree=0.8, subsample=0.9,
                        objective="regression_l1", fourier_k=4)
        cur = {**defaults, **p}
        grid = [("learning_rate", [round(cur["learning_rate"] * f, 4) for f in (0.5, 1.6)]),
                ("num_leaves", [max(15, cur["num_leaves"] // 2), min(255, cur["num_leaves"] * 2)]),
                ("min_child_samples", [10, 40]),
                ("n_estimators", [600, 1500]),
                ("objective", ["regression" if cur["objective"] == "regression_l1" else "regression_l1"]),
                ("fourier_k", [2, 6]),
                ("colsample_bytree", [0.6, 1.0])]
        for name, vals in grid:
            for v in vals:
                if cur.get(name) == v:
                    continue
                c2 = json.loads(json.dumps(cfg))
                c2["params"] = {**p, name: v}
                cands.append((c2, f"{name}={v}"))
        full = json.loads(json.dumps(cfg))
        full["params"] = cur
        for bc in hv3.boundary_candidates(full, LGB_SPACE):
            c2 = json.loads(json.dumps(cfg))
            c2["params"] = {**p, bc["param"]: bc["new_value"]}
            cands.append((c2, f"boundary-push {bc['param']}={bc['new_value']}"))
    RNG.shuffle(cands)
    for c2, mut in cands:
        if hv2.find_duplicate_config(tree, core(c2)) is None:
            return core(c2), mut
    return core(cands[0][0]) if cands else cfg, "exhausted (dup-burn)"


def inject_burst(tree):
    root_id = tree["root_id"]
    pool = solo_pool(tree)
    print("== explore burst ==")
    nid = eval_and_add(tree, root_id, "BURST kitchen-sink blend",
                       {"kind": "blend", "members": [n["id"] for n in pool]})
    longshots = [
        ("BURST ridge low-reg hi-K", {"kind": "solo", "model": "ridge_struct",
         "params": {**ROOT_CFG["params"], "alpha": 0.1, "fourier_k_global": 6,
                    "fourier_k_product": 6, "dow_store": True}, "level": {"k_est": "mean"}}),
        ("BURST lgb_ratio big", {"kind": "solo", "model": "lgb_ratio",
         "params": {"learning_rate": 0.02, "num_leaves": 255, "n_estimators": 3000},
         "level": {"k_est": "mean"}}),
        ("BURST lgb_shape l2 deep", {"kind": "solo", "model": "lgb_shape",
         "params": {"objective": "regression", "num_leaves": 127, "learning_rate": 0.03,
                    "n_estimators": 2000}, "level": {"k_est": "mean"}}),
        ("BURST naive", {"kind": "solo", "model": "naive", "params": {}, "level": {"k_est": "mean"}}),
    ]
    for mut, cfg in longshots:
        nid = eval_and_add(tree, root_id, mut, cfg)
        if nid is not None:
            hv3.apply_burst_seed_sanity_gate(tree, nid)
    hv3.save_search_state(tree, TREE_PATH)


def main():
    priors = hv2.suggest_priors({"metric": "smape", "tags": ["time-series", "panel", "retail", "smape"],
                                 "data_type": "tabular", "comp": COMP})
    print(f"=== suggest_priors: {len(priors)} bullets ===")
    for b in priors[:12]:
        print(" -", b[:140].replace("\n", " "))

    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        print("resumed tree with", len(tree["nodes"]), "nodes")
    else:
        tree = hv1.new_tree(COMP)
        hv3.init_budget(tree)
        r = hv3.eval_solo_subprocess(EVAL_PATH, core(ROOT_CFG), TIMEOUT_S, node_id=1)
        assert r["status"] == "evaluated", r
        assert round(r["score"], 5) == ROOT_EXPECT, f"root drift: {r['score']} != {ROOT_EXPECT}"
        hv2.add_root(tree, "root: ridge_struct members_v1 baseline (digit-verified)",
                     core(ROOT_CFG), r["score"], "evaluated", r["wall_s"], kind="solo")
        print(f"root #1 verified: {r['score']}")
        ds = tree["search_state"].setdefault("driver_state", {})
        ds["lineage_names"] = {}
        for lname, cfg in SEEDS:
            nid = eval_and_add(tree, tree["root_id"], f"seed {lname}", cfg)
            if nid is not None:
                ds["lineage_names"][str(nid)] = lname
        pool = sorted(solo_pool(tree), key=lambda n: n["score"])
        bid = eval_and_add(tree, tree["root_id"], "seed BLEND top-4",
                           {"kind": "blend", "members": [n["id"] for n in pool[:4]]})
        if bid is not None:
            ds["lineage_names"][str(bid)] = "BLEND"
        hv3.save_search_state(tree, TREE_PATH)

    ds = tree["search_state"]["driver_state"]
    while not hv3.should_stop(tree):
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not ds.get("burst_injected"):
            inject_burst(tree)
            ds["burst_injected"] = True
            hv3.save_search_state(tree, TREE_PATH)
            continue
        parent_id, lineage_id = hv1.select_next_parent(tree)
        if parent_id is None:
            print("no parent available; stopping")
            break
        parent = next(n for n in tree["nodes"] if n["id"] == parent_id)
        lname = ds["lineage_names"].get(str(lineage_id), "RIDGE")
        cfg, mut = propose(tree, parent, lname)
        eval_and_add(tree, parent_id, f"[{lname}] {mut}", cfg)
        hv3.save_search_state(tree, TREE_PATH)

    best = min((n for n in tree["nodes"] if n["score"] is not None), key=lambda n: n["score"])
    n_eval = hv3.n_evaluated(tree)
    print("\n=== search done ===")
    print("stop_reason:", tree["search_state"]["budget"].get("stop_reason"))
    print(f"evaluated nodes: {n_eval}; global best #{best['id']} = {best['score']} ({best['mutation'][:80]})")
    print("dedup_rejections:", ds.get("dedup_rejections", 0))
    print("backtrack_log entries:", len(tree["search_state"]["backtrack_log"]))
    hv3.save_search_state(tree, TREE_PATH)


if __name__ == "__main__":
    main()
