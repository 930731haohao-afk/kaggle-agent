"""tree_search/run_s3e16_v3.py — faithful harness_v3 re-run for playground-series-s3e16
(Crab Age, MAE with integer-rounding decision metric, minimize). Bespoke driver #1 of the
"faithful full v3 re-run" set, mirroring run_s3e14_v3.py's structure: the EXACT committed v1
seeds (experiments_tree.json nodes #0-#8, byte-transcribed below), digit-verified cache reuse,
a hand-scripted per-lineage mutation queue, and the real harness_v3 phase/blend/dedup policy.
Decision metric = round_clip_mae (rounding INSIDE the metric). eval_s3e16_v2 trains LGB fresh;
XGB/CAT are read-only legacy reuses (module contract), so their seeds reuse the cached OOF.

Usage: uv run python3 tree_search/run_s3e16_v3.py
Output: competitions/playground-series-s3e16/experiments_tree_v3.json (overwrites the adapter stub).
"""
import copy
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as hv1        # noqa: E402  select_next_parent, lineage helpers
import harness_v2 as hv2     # noqa: E402
import harness_v3 as hv3     # noqa: E402
import eval_s3e16_v2 as ev   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s3e16"
COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
OLD_TREE_PATH = os.path.join(COMP_DIR, "experiments_tree.json")   # committed v1 tree, READ-ONLY
TREE_PATH = os.path.join(COMP_DIR, "experiments_tree_v3.json")    # this run's output
MAX_WALL_S = 25 * 60
ITER_CAP = 120
V1_TREE_BEST = 1.33563          # experiments_tree.json committed champion (rounded MAE)
ROOT_SCORE = 1.33885

DEDUP = []


def dc(x): return copy.deepcopy(x)


def core(cfg):
    out = {k: v for k, v in cfg.items() if k != "result"}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def _node_results(tree):
    return tree.setdefault("search_state", {}).setdefault("node_results", {})


# ---- EXACT committed v1 seed configs (byte-transcribed from experiments_tree.json) ----
ROOT_CFG = {"kind": "solo", "model": "lgb", "params": {
    "objective": "regression_l1", "metric": "mae", "n_estimators": 3000, "learning_rate": 0.02,
    "num_leaves": 63, "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 2.0, "random_state": 42},
    "features": {"drop": []}}
SEEDS = [
    ("XGB", {"kind": "solo", "model": "xgb", "params": {
        "objective": "reg:absoluteerror", "n_estimators": 3000, "learning_rate": 0.02, "max_depth": 6,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 1.0,
        "reg_lambda": 2.0, "random_state": 42}, "features": {"drop": []}}, 1.3416, "vanilla XGB (reg:absoluteerror)"),
    ("CAT", {"kind": "solo", "model": "cat", "params": {
        "loss_function": "MAE", "eval_metric": "MAE", "iterations": 4000, "learning_rate": 0.03,
        "depth": 7, "l2_leaf_reg": 5.0, "random_seed": 42}, "features": {"drop": []}}, 1.33846, "vanilla CatBoost (MAE)"),
    ("LGB_tuned", {"kind": "solo", "model": "lgb", "params": {
        "learning_rate": 0.010194593774108831, "num_leaves": 61, "min_child_samples": 30,
        "subsample": 0.8094065093571273, "colsample_bytree": 0.7216637100997246, "reg_alpha": 0.1131044878505013,
        "reg_lambda": 2.563832295704767, "n_estimators": 3000, "random_state": 42, "objective": "regression_l1",
        "metric": "mae", "subsample_freq": 1}, "features": {"drop": []}}, 1.33979, "Optuna fold0-proxy tuned LGB"),
    ("LGB_tuned_seed2024", {"kind": "solo", "model": "lgb", "params": {
        "learning_rate": 0.010194593774108831, "num_leaves": 61, "min_child_samples": 30,
        "subsample": 0.8094065093571273, "colsample_bytree": 0.7216637100997246, "reg_alpha": 0.1131044878505013,
        "reg_lambda": 2.563832295704767, "n_estimators": 3000, "random_state": 2024, "objective": "regression_l1",
        "metric": "mae", "subsample_freq": 1}, "features": {"drop": []}}, 1.33914, "seed-bagged LGB_tuned (seed 2024)"),
    ("LGBBOUND", {"kind": "solo", "model": "lgb", "params": {
        "learning_rate": 0.005, "num_leaves": 61, "min_child_samples": 30, "subsample": 0.8094065093571273,
        "colsample_bytree": 0.7216637100997246, "reg_alpha": 0.1131044878505013, "reg_lambda": 2.563832295704767,
        "n_estimators": 3000, "random_state": 42, "objective": "regression_l1", "metric": "mae",
        "subsample_freq": 1}, "features": {"drop": []}}, 1.3395, "boundary-push: lr 0.0102->0.005"),
    ("TWEEDIE", {"kind": "solo", "model": "lgb", "params": {
        "num_leaves": 61, "min_child_samples": 30, "subsample": 0.8094065093571273,
        "colsample_bytree": 0.7216637100997246, "reg_alpha": 0.1131044878505013, "reg_lambda": 2.563832295704767,
        "n_estimators": 3000, "random_state": 42, "objective": "tweedie", "tweedie_variance_power": 1.5,
        "metric": "mae", "learning_rate": 0.02, "subsample_freq": 1}, "features": {"drop": []}}, 1.37467,
     "diverse-objective probe: tweedie"),
    ("POISSON", {"kind": "solo", "model": "lgb", "params": {
        "objective": "poisson", "metric": "mae", "n_estimators": 3000, "learning_rate": 0.02, "num_leaves": 63,
        "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_alpha": 1.0,
        "reg_lambda": 2.0, "random_state": 42}, "features": {"drop": []}}, 1.37501, "diverse-objective probe: poisson"),
    ("FEATPRUNE", {"kind": "solo", "model": "lgb", "params": {
        "objective": "regression_l1", "metric": "mae", "n_estimators": 3000, "learning_rate": 0.02, "num_leaves": 63,
        "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_alpha": 1.0,
        "reg_lambda": 2.0, "random_state": 42}, "features": {"drop": ["Height"]}}, 1.34025, "collinear feature-pruned probe"),
]

# hand-scripted per-lineage mutation queues (genuinely-untried-by-v1 nudges)
SOLO_QUEUES = {
    "ROOT": [lambda c: (dict(dc(c), params=dict(c["params"], num_leaves=31, min_child_samples=80)),
                        "regularize deeper: num_leaves 63->31, mcs 40->80")],
    "CAT": [lambda c: (dict(dc(c), params=dict(c["params"], depth=5, l2_leaf_reg=10.0)),
                       "CAT shallow+reg: depth 7->5, l2 5->10")],
    "XGB": [lambda c: (dict(dc(c), params=dict(c["params"], max_depth=3, reg_alpha=3.0, reg_lambda=8.0)),
                       "XGB shallow-reg: depth 6->3, strong reg")],
    "LGB_tuned": [lambda c: (dict(dc(c), params=dict(c["params"], num_leaves=15, min_child_samples=80)),
                             "LGB_tuned regularize: leaves->15, mcs->80")],
}


def solo_fallback(base, i):
    p = dict(base["params"]); seed_key = "random_seed" if base["model"] == "cat" else "random_state"
    p[seed_key] = 3000 + i
    return {"kind": "solo", "model": base["model"], "params": p, "features": dc(base.get("features", {"drop": []}))}, \
        f"seed variation #{3000 + i}"


LINEAGE_NAMES = {}


def blend_score(members):
    oofs = np.stack([ev.load_solo_cache(m)["oof"] if hasattr(ev, "load_solo_cache") else hv2.load_oof(ev.CACHE_DIR, m)
                     for m in members], axis=1)

    def metric_fn(vec): return ev.round_clip_mae(vec)
    w, s = hv2.eval_blend(ev.CACHE_DIR, members, metric_fn)[:2] if False else _weight_search(oofs, metric_fn)
    return w, s, oofs


def _weight_search(oofs, metric_fn, k=800, seed=42):
    n = oofs.shape[1]; rng = np.random.default_rng(seed)
    cands = [np.eye(n)[i] for i in range(n)] + [np.full(n, 1.0 / n)] + list(rng.dirichlet(np.ones(n), size=k))
    bw = bs = None
    for w in cands:
        s = metric_fn(oofs @ w)
        if bs is None or s < bs: bs, bw = s, w
    bw, bs = hv3._coordinate_ascent_refine(oofs, metric_fn, bw, bs)
    return bw, bs


LEGACY = {"xgb": ("XGB", 1.35763, 1.3416), "cat": ("CAT", 1.36162, 1.33846)}  # module trains LGB only


def eval_and_add(tree, parent, mutation, cfg, lineage_id=None):
    stored = core(cfg)
    if parent != tree["root_id"]:
        dup = hv3.find_duplicate_config(tree, stored)
        if dup is not None:
            DEDUP.append(dict(mutation=mutation, dup=dup))
            hv3.add_node(tree, parent, mutation + " [dedup]", stored, None, "failed", 0.0)
            return None, dup
    nid = hv3.next_id(tree)
    if stored.get("kind") == "blend":
        try:
            w, s, _ = blend_score(stored["members"])
            _node_results(tree)[str(nid)] = {"members": stored["members"], "weights": [round(float(x), 5) for x in w]}
            r = {"score": round(s, 5), "status": "evaluated", "wall_s": 0.1}
        except Exception as e:  # noqa: BLE001
            r = {"score": None, "status": "failed", "wall_s": 0.0}; mutation += f" [ERR {e}]"
    elif stored.get("model") in LEGACY:                      # XGB/CAT: read-only legacy reuse
        try:
            name, xr, xrd = LEGACY[stored["model"]]
            oof, pred, graw, grnd = ev.load_legacy_solo(name, xr, xrd, tol=5e-3)
            hv2.cache_oof(ev.CACHE_DIR, nid, oof, pred=pred, y=ev._y)
            r = {"score": round(grnd, 5), "status": "evaluated", "wall_s": 0.05}
            _node_results(tree)[str(nid)] = {"mae_raw": round(graw, 5), "mae_rounded": round(grnd, 5), "reused": name}
        except Exception as e:  # noqa: BLE001
            r = {"score": None, "status": "failed", "wall_s": 0.0}; mutation += f" [ERR {e}]"
    else:                                                    # LGB: fresh deterministic train
        try:
            rr = ev.evaluate(stored, node_id=nid, timeout_s=300)
            r = {"score": rr["score"], "status": rr["status"], "wall_s": rr["wall_s"]}
            _node_results(tree)[str(nid)] = rr.get("result") or {}
        except Exception as e:  # noqa: BLE001
            r = {"score": None, "status": "failed", "wall_s": 0.0}; mutation += f" [ERR {e}]"
    rid, _dup = hv3.add_node(tree, parent, mutation, stored, r["score"], r["status"], r["wall_s"])
    return rid, None


def main():
    t0 = time.time()
    tree = hv3.new_tree(COMP)
    hv3.init_budget(tree, total_budget=28)
    # root
    rid = hv3.add_root(tree, "[ROOT] committed v1 root LGB L1", core(ROOT_CFG), None, "pending", 0.0)
    rr = ev.evaluate(core(ROOT_CFG), node_id=rid, timeout_s=300)
    tree["nodes"][0].update(score=rr["score"], status=rr["status"], wall_s=rr["wall_s"])
    _node_results(tree)[str(rid)] = rr.get("result") or {}
    assert round(rr["score"], 4) == round(ROOT_SCORE, 4), f"ROOT digit-verify FAILED: {rr['score']} vs {ROOT_SCORE}"
    print(f"ROOT digit-verified: {rr['score']} == {ROOT_SCORE}")
    LINEAGE_NAMES[rid] = "ROOT"
    # seeds
    for name, cfg, exp, desc in SEEDS:
        nid, _ = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        LINEAGE_NAMES[nid] = name
        sc = tree["nodes"][-1]["score"]
        vok = sc is not None and abs(sc - exp) < 3e-3
        print(f"[{name}] #{nid} score={sc} (committed {exp}) verify={'OK' if vok else 'DRIFT'}")
    # blend seed over the 3 best committed members (root, CAT, LGB_tuned_seed) -> mirror committed [BLEND]
    pool = [n["id"] for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    bid, _ = eval_and_add(tree, tree["root_id"], "[BLEND] full-pool ensemble seed", {"kind": "blend", "members": sorted(pool)})
    LINEAGE_NAMES[bid] = "BLEND"
    print(f"[BLEND] #{bid} score={tree['nodes'][-1]['score']}")
    # forward policy loop
    it = 0
    while (not hv3.should_stop(tree)) and hv3.n_evaluated(tree) < hv3.init_budget(tree)["total_budget"] \
            and (time.time() - t0) < MAX_WALL_S and it < ITER_CAP:
        it += 1
        try:
            parent_id, lineage_id = hv1.select_next_parent(tree)
        except Exception:
            parent_id = None
        if parent_id is None:
            # fallback: reblend the growing pool
            pool = [n["id"] for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
            _eid, _ = eval_and_add(tree, tree["root_id"], "[BLEND] pool reblend", {"kind": "blend", "members": sorted(pool)})
            if hv3.n_evaluated(tree) >= hv3.init_budget(tree)["total_budget"]: break
            continue
        name = LINEAGE_NAMES.get(parent_id, "ROOT")
        parent = next(n for n in tree["nodes"] if n["id"] == parent_id)
        pool = [n["id"] for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
        if parent["config"].get("kind") != "solo" or "params" not in parent["config"]:
            eval_and_add(tree, tree["root_id"], "[BLEND] pool reblend", {"kind": "blend", "members": sorted(pool)})
            json.dump(tree, open(TREE_PATH, "w"), ensure_ascii=False, indent=2)
            continue
        q = SOLO_QUEUES.get(name, [])
        idx = sum(1 for n in tree["nodes"] if n["parent_id"] == parent_id)  # per-parent child count
        if 0 <= idx < len(q):
            cfg, desc = q[idx](parent["config"])
        else:
            cfg, desc = solo_fallback(parent["config"], it)
        eval_and_add(tree, parent_id, f"[{name}] {desc}", cfg, lineage_id)
        json.dump(tree, open(TREE_PATH, "w"), ensure_ascii=False, indent=2)
    gb = hv3.global_best(tree)
    json.dump(tree, open(TREE_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"\nDONE s3e16: {hv3.n_evaluated(tree)} evaluated nodes, wall={time.time()-t0:.0f}s")
    print(f"v3 champion #{gb['id']} score={gb['score']} | committed v1 champion {V1_TREE_BEST} -> "
          f"v3 {'BEAT' if gb['score'] < V1_TREE_BEST else ('MATCHED' if abs(gb['score']-V1_TREE_BEST)<1e-9 else 'above')} it")


if __name__ == "__main__":
    main()
