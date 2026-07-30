"""Comp- and arm-agnostic v5 search driver.

Generalizes run_s3e19_v5.py: takes --comp and --arm, imports that arm's generated
evaluator (or the competition's baseline evaluator for --arm baseline), runs the same
harness_v3 forward search used everywhere else, and writes a submission from the
champion node. Blend champions are finalized by blending the members' CACHED test
predictions with the evaluator's recovered weights (no retraining).

Usage:
  VIRTUAL_ENV= uv run python3 tree_search/run_arm_search.py \
      --comp playground-series-s3e19 --arm ratio --nodes 45
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402
import seed_from_ledger as sfl  # noqa: E402
from make_v5_arm import BASE, _slug  # noqa: E402
from run_v3_generic import core, solo_pool, variant as mk_variant  # noqa: E402

# per-competition seed config source and param whitelist
SEED_TREE = {
    "playground-series-s3e19": "competitions/playground-series-s3e19/experiments_tree_v3.json",
}
WHITELIST = {
    "playground-series-s3e19": "llm_proposer_input_s3e19.json",
    "tabular-playground-series-sep-2022": "llm_proposer_input_s3e19.json",  # same model families
}
DEFAULT_SEED_CFG = {"kind": "solo", "model": "lgb",
                    "params": {"num_leaves": 31, "learning_rate": 0.05,
                               "n_estimators": 400, "random_state": 42},
                    "features": {"drop": []}}


def load_evaluator(comp: str, arm: str):
    if arm == "baseline":
        mod = BASE[comp][0]
    else:
        mod = f"eval_{arm}_{_slug(comp)}"
    return importlib.import_module(mod)


def workspace(comp: str, arm: str) -> str:
    sub = comp if arm == "baseline" else f"{comp}-v5-{arm}"
    return os.path.join(_ROOT, "competitions", sub)


def seed_config(comp: str) -> dict:
    p = SEED_TREE.get(comp)
    if p and os.path.exists(os.path.join(_ROOT, p)):
        t = json.load(open(os.path.join(_ROOT, p)))
        n0 = next(n for n in t["nodes"] if n["config"].get("kind") == "solo"
                  and isinstance(n.get("score"), (int, float)))
        return json.loads(json.dumps(n0["config"]))
    return json.loads(json.dumps(DEFAULT_SEED_CFG))


def champ(t):
    ev_n = [n for n in t["nodes"] if n["status"] == "evaluated"
            and isinstance(n["score"], (int, float))]
    return min(ev_n, key=lambda n: n["score"]) if ev_n else None


def search(comp: str, arm: str, ev, n_nodes: int, wl: dict):
    ws = workspace(comp, arm)
    os.makedirs(ws, exist_ok=True)
    tpath = os.path.join(ws, f"experiments_tree_v5_{arm}.json")
    if os.path.exists(tpath):
        tree = json.load(open(tpath))
    else:
        tree = {"comp": f"{comp}:{arm}", "nodes": [], "root_id": 0, "search_state": {}}
        cfg = core(seed_config(comp))
        t0 = time.time()
        r = ev.evaluate(cfg, node_id=0, timeout_s=600)
        hv2.add_root(tree, f"[V5:{arm}] seed config", cfg, r["score"], r["status"],
                     time.time() - t0)
        json.dump(tree, open(tpath, "w"), indent=2)
        print(f"[{arm}] root: {r['status']} score={r['score']}")

        # Seed the ledger's config-only operators as real nodes. Before 07-30 these were
        # written to plan["node_configs"] and read by nobody, so the judgment layer's
        # model-level decisions changed nothing while the ledger reported them realized.
        specs = sfl.load_emitted(ws)
        if specs:
            seed_nodes, unconsumable = sfl.materialize(specs, cfg, wl)
            seeded = []
            for sn in seed_nodes:
                nid = hv3.next_id(tree)
                label = f"[V5:{arm}] injected {sn.get('seed_role')} " \
                        f"({sn.get('metric_family') or sn.get('archetype')})"
                try:
                    r = ev.evaluate(core(sn), node_id=nid, timeout_s=600)
                except Exception as e:  # noqa: BLE001
                    r = {"score": None, "status": "failed", "wall_s": 0.0, "error": str(e)}
                hv3.add_node(tree, tree["root_id"], label, core(sn), r["score"],
                             r["status"], r.get("wall_s", 0.0))
                seeded.append({"node_id": nid, "seed_role": sn.get("seed_role"),
                               "model": sn.get("model"), "status": r["status"],
                               "score": r["score"]})
                print(f"[{arm}] injected node {nid} ({sn.get('seed_role')}): "
                      f"{r['status']} score={r['score']}")
            json.dump(tree, open(tpath, "w"), indent=2)
            p = sfl.write_consumption_report(ws, seeded, unconsumable)
            print(f"[{arm}] consumption report: {p} "
                  f"({len(seeded)} seeded, {len(unconsumable)} unconsumable)")

    c0 = champ(tree)["score"]
    n_start = len([n for n in tree["nodes"] if n["status"] == "evaluated"])
    tree.setdefault("search_state", {})["budget"] = {
        "total_budget": n_start + n_nodes, "explore_burst_size": 6,
        "post_burst_patience": 10, "phase": "exploit", "burst_start_eval": 0,
        "best_at_burst_start": c0, "evals_since_burst_improve": 0}
    fams = [m for m in ("xgb", "cat", "lgb") if m in wl]
    for i in range(n_nodes):
        pool = solo_pool(tree)
        ids = [n["id"] for n in pool]
        if i % 2 == 1 and len(ids) >= 2:
            cfg = {"kind": "blend", "members": sorted(ids), "weight_search": "dirichlet"}
            mut, parent = f"[V5:{arm}] reblend over {len(ids)}", pool[0]["id"]
        else:
            fam = fams[(i // 2) % len(fams)]
            base = next((n["config"] for n in pool if n["config"].get("model") == fam),
                        pool[0]["config"])
            cfg = mk_variant(base, wl, i)
            mut, parent = f"[V5:{arm}] shallow-reg {cfg['model']}", tree["root_id"]
        nid = hv3.next_id(tree)
        try:
            r = ev.evaluate(core(cfg), node_id=nid, timeout_s=600)
        except Exception as e:  # noqa: BLE001
            r = {"score": None, "status": "failed", "wall_s": 0.0, "error": str(e)}
        hv3.add_node(tree, parent, mut, core(cfg), r["score"], r["status"], r.get("wall_s", 0.0))
        json.dump(tree, open(tpath, "w"), indent=2)
        print(f"[{arm}] node {nid} {r['status']}; champion={champ(tree)['score']:.6f}")
        if hv3.should_stop(tree):
            print(f"[{arm}] budget machine stop")
            break
    return tree, tpath


def finalize(comp: str, arm: str, ev, tree) -> str:
    import pandas as pd
    c = champ(tree)
    cfg = c["config"]
    cache = ev.CACHE_DIR
    if cfg["kind"] == "solo":
        pred = np.asarray(np.load(os.path.join(cache, f"solo_{c['id']}.npz"),
                                  allow_pickle=True)["pred"])
    else:
        r = ev.evaluate(cfg, node_id=c["id"], timeout_s=1800)
        w = np.asarray((r.get("result") or {})["weights"], dtype=float)
        preds = np.stack([np.asarray(np.load(os.path.join(cache, f"solo_{m}.npz"),
                                             allow_pickle=True)["pred"])
                          for m in cfg["members"]])
        pred = (w[:, None] * preds).sum(axis=0) / w.sum()
        sc = (r.get("result") or {}).get("scale_used")
        if sc:
            pred = pred * float(sc)
    base_data = os.path.join(_ROOT, "competitions", comp, "data", "sample_submission.csv")
    sub = pd.read_csv(base_data)
    sub[sub.columns[-1]] = pred
    out = os.path.join(workspace(comp, arm), "submission.csv")
    sub.to_csv(out, index=False)
    print(f"[{arm}] submission: {out} (champion #{c['id']} score={c['score']:.6f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--nodes", type=int, default=45)
    a = ap.parse_args()
    wl = json.load(open(os.path.join(_HERE, WHITELIST[a.comp])))["param_whitelist"]
    ev = load_evaluator(a.comp, a.arm)
    tree, tpath = search(a.comp, a.arm, ev, a.nodes, wl)
    out = finalize(a.comp, a.arm, ev, tree)
    summary = {"comp": a.comp, "arm": a.arm, "champion_cv": champ(tree)["score"],
               "nodes_evaluated": len([n for n in tree["nodes"] if n["status"] == "evaluated"]),
               "tree": tpath, "submission": out}
    json.dump(summary, open(os.path.join(workspace(a.comp, a.arm), f"arm_result_{a.arm}.json"), "w"),
              indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
