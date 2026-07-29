"""v5 upstream-injection experiment driver for s3e19 (race -> extend -> submission).

Races external-data variants (each = its own digit-exact workspace + generated
evaluator, see make_s3e19_v5_variant.py) with a small harness_v3 budget, then
extends the winning variant's tree with the full budget and writes a submission
from the champion node. Mutation/blend scheme and budget-machine wiring copied
from run_v3_generic.py so search behavior matches every other v3 run; the only
new element is the per-variant fresh tree seeded from the committed baseline
node-0 config (external columns active via the variant's FEATURE_COLS).

Race structure per references/00_problem_dossier.md: data-level ideas get their
own lane (variant workspaces) raced at small budget; winner gets the full run.

Usage:
  VIRTUAL_ENV= uv run python3 tree_search/run_s3e19_v5.py --race
  VIRTUAL_ENV= uv run python3 tree_search/run_s3e19_v5.py --full
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
from run_v3_generic import core, solo_pool, variant as mk_variant  # noqa: E402

COMP = "playground-series-s3e19"
VARIANTS = ["gdp", "gdp_hol"]
BASE_TREE = os.path.join(_ROOT, "competitions", COMP, "experiments_tree_v3.json")
WL = json.load(open(os.path.join(_HERE, "llm_proposer_input_s3e19.json")))["param_whitelist"]


def ensure_variant(v: str):
    if not os.path.exists(os.path.join(_HERE, f"eval_s3e19_v5_{v}.py")):
        import make_s3e19_v5_variant as mk
        mk.build(v)
    return importlib.import_module(f"eval_s3e19_v5_{v}")


def tree_path(v: str) -> str:
    return os.path.join(_ROOT, "competitions", f"{COMP}-v5-{v}", "experiments_tree_v5.json")


def save_tree(tree, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(tree, open(tmp, "w"), ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def champ(t):
    ev_n = [n for n in t["nodes"] if n["status"] == "evaluated"
            and isinstance(n["score"], (int, float))]
    return min(ev_n, key=lambda n: n["score"]) if ev_n else None


def seed_config():
    base = json.load(open(BASE_TREE))
    n0 = next(n for n in base["nodes"] if n["config"].get("kind") == "solo"
              and isinstance(n.get("score"), (int, float)))
    return json.loads(json.dumps(n0["config"]))


def init_tree(v: str, ev):
    p = tree_path(v)
    if os.path.exists(p):
        return json.load(open(p)), p
    tree = {"comp": f"{COMP}-v5-{v}", "nodes": [], "root_id": 0, "search_state": {}}
    cfg = core(seed_config())
    t0 = time.time()
    r = ev.evaluate(cfg, node_id=0, timeout_s=300)
    hv2.add_root(tree, "[V5] committed node-0 config + external columns", cfg,
                 r["score"], r["status"], time.time() - t0)
    save_tree(tree, p)
    print(f"[{v}] root: score={r['score']:.6f}")
    return tree, p


def forward_search(v: str, ev, n_nodes: int):
    tree, p = init_tree(v, ev)
    c0 = champ(tree)["score"]
    n_start = len([n for n in tree["nodes"] if n["status"] == "evaluated"])
    tree.setdefault("search_state", {})
    tree["search_state"]["budget"] = {
        "total_budget": n_start + n_nodes, "explore_burst_size": 6,
        "post_burst_patience": 10, "phase": "exploit", "burst_start_eval": 0,
        "best_at_burst_start": c0, "evals_since_burst_improve": 0}
    models_avail = [m for m in ("xgb", "cat", "lgb") if m in WL]
    for i in range(n_nodes):
        pool = solo_pool(tree)
        pool_ids = [n["id"] for n in pool]
        if i % 2 == 1 and len(pool_ids) >= 2:            # full-pool reblend
            cfg = {"kind": "blend", "members": sorted(pool_ids), "weight_search": "dirichlet"}
            mut = f"[V5] pool reblend over {len(pool_ids)} members (dirichlet)"
            parent = pool[0]["id"]
        else:                                            # fresh bias-dominated solo variant
            fam = models_avail[(i // 2) % len(models_avail)]
            base = next((n["config"] for n in pool if n["config"].get("model") == fam),
                        pool[0]["config"])
            cfg = mk_variant(base, WL, i)
            mut = f"[V5] shallow-reg {cfg['model']} variant"
            parent = tree["root_id"]
        nid = hv3.next_id(tree)
        try:
            r = ev.evaluate(core(cfg), node_id=nid, timeout_s=300)
        except Exception as e:  # noqa: BLE001 — a failed node must not kill the race
            r = {"score": None, "status": "failed", "wall_s": 0.0, "error": str(e)}
        hv3.add_node(tree, parent, mut, core(cfg), r["score"], r["status"],
                     r.get("wall_s", 0.0))
        save_tree(tree, p)
        print(f"[{v}] node {nid} {r['status']}; champion={champ(tree)['score']:.6f}")
        if hv3.should_stop(tree):
            print(f"[{v}] budget machine stop")
            break
    return tree


def finalize(v: str, ev, tree) -> str:
    """Write submission.csv from the champion node's cached test predictions."""
    import pandas as pd
    c = champ(tree)
    cfg = c["config"]
    if cfg["kind"] == "solo":
        rec = hv2.load_oof(ev.CACHE_DIR, c["id"])
        pred = np.asarray(rec["pred"])
    else:
        # re-run the champion blend evaluation to recover weights + blended test pred
        r = ev.evaluate(cfg, node_id=c["id"], timeout_s=600)
        pred = np.asarray(r.get("pred") if r.get("pred") is not None else r["result"]["pred"])
    sub = pd.read_csv(os.path.join(_ROOT, "competitions", COMP, "data", "sample_submission.csv"))
    sub[sub.columns[-1]] = pred
    out = os.path.join(_ROOT, "competitions", f"{COMP}-v5-{v}", "submission.csv")
    sub.to_csv(out, index=False)
    print(f"[{v}] submission: {out} (champion #{c['id']} score={c['score']:.6f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--race-nodes", type=int, default=12)
    ap.add_argument("--extend-nodes", type=int, default=40)
    args = ap.parse_args()

    results = {}
    for v in VARIANTS:
        ev = ensure_variant(v)
        tree = forward_search(v, ev, args.race_nodes)
        results[v] = champ(tree)["score"]
    winner = min(results, key=results.get)
    print(json.dumps({"race_results": results, "winner": winner}, indent=2))

    if args.full:
        ev = ensure_variant(winner)
        tree = forward_search(winner, ev, args.extend_nodes)
        finalize(winner, ev, tree)
        base_champ = champ(json.load(open(BASE_TREE)))["score"]
        summary = {"winner": winner, "race": results,
                   "v5_champion_cv": champ(tree)["score"],
                   "v3_committed_champion_cv": base_champ}
        out = os.path.join(_ROOT, "competitions", f"{COMP}-v5-{winner}", "v5_result.json")
        json.dump(summary, open(out, "w"), indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
