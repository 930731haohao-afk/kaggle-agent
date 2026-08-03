"""run_v3_generic.py — generic v3 forward-search extension for an adapted comp.

Instead of hand-authoring a bespoke ~800-line driver per comp, this drives harness_v3's
node/phase machinery generically: starting from the adapted tree (which already holds the
real cached solo pool), it proposes fresh candidates via GENERIC mutation operators (shallow
& strongly-regularized variants of pool members, across model families) plus full-pool
reblends, evaluates each with the comp's own eval module (ev.evaluate dispatches solo AND
blend), and adds them through hv3.add_node (so the budget/phase machine runs). The champion
that emerges is a genuinely v3-searched node (not a v1 reformat) -> removes the adapter confound.

The committed experiments_tree_v3.json is read-only here: this run's nodes go to a separate
tree file (--output, default experiments_tree_v3_gen.json beside it).

Usage: uv run python3 tree_search/run_v3_generic.py --comp s3e16 --nodes 24
"""
import argparse
import copy
import importlib
import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness as hv1        # noqa: E402  (select_next_parent lives here)
import harness_v2 as hv2     # noqa: E402
import harness_v3 as hv3     # noqa: E402

REPO = os.path.dirname(_HERE)
EVAL_MOD = {"s3e16": "eval_s3e16_v2", "s3e1": "eval_s3e1", "s3e3": "eval_s3e3",
            "s3e9": "eval_s3e9_v2", "s3e11": "eval_s3e11", "s3e19": "eval_s3e19"}
DEPTH_KEY = {"lgb": "max_depth", "xgb": "max_depth", "cat": "depth"}
SEED_KEY = {"lgb": "random_state", "xgb": "random_state", "cat": "random_seed"}


def core(cfg):
    out = {k: v for k, v in cfg.items() if k != "result"}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def solo_pool(tree):
    ns = [n for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    ns.sort(key=lambda n: n["score"])
    return ns


def variant(base_cfg, wl, i):
    """Generic bias-dominated mutation of a solo config: shallower + stronger reg + reseed,
    varied by i so successive calls differ. Constrained to the whitelist."""
    m = base_cfg["model"]; keys = set(wl.get(m, []))
    p = {k: v for k, v in base_cfg.get("params", {}).items() if k in keys}
    dk = DEPTH_KEY.get(m)
    if dk in keys: p[dk] = [2, 3, 4][i % 3]
    for rk, rv in (("reg_alpha", [1.0, 3.0, 8.0][i % 3]), ("reg_lambda", [4.0, 10.0, 20.0][i % 3]),
                   ("l2_leaf_reg", [8.0, 12.0, 20.0][i % 3]), ("min_child_weight", [20, 40, 80][i % 3]),
                   ("min_child_samples", [60, 120, 200][i % 3]), ("num_leaves", [4, 7, 15][i % 3])):
        if rk in keys: p[rk] = rv
    sk = SEED_KEY.get(m)
    if sk in keys: p[sk] = 1000 + i
    return {"kind": "solo", "model": m, "params": p,
            "features": copy.deepcopy(base_cfg.get("features", {"drop": []}))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True, choices=EVAL_MOD)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--output", default=None,
                    help="tree file this run writes (default: experiments_tree_v3_gen.json "
                         "beside the source tree; the source tree is never modified)")
    args = ap.parse_args()
    comp = args.comp
    ev = importlib.import_module(EVAL_MOD[comp])
    wl = json.load(open(os.path.join(_HERE, f"llm_proposer_input_{comp}.json")))["param_whitelist"]
    comp_dir = os.path.join(REPO, "competitions", f"playground-series-{comp}")
    src_path = os.path.join(comp_dir, "experiments_tree_v3.json")
    # This driver used to append its [V3GEN] nodes to the COMMITTED experiments_tree_v3.json
    # and overwrite its search_state.budget in place, so a generic exploratory run corrupted
    # the record of the lane whose numbers are already reported. It now seeds from that tree
    # (read-only) and writes everywhere else (2026-08-03 audit).
    tree_path = args.output or os.path.join(comp_dir, "experiments_tree_v3_gen.json")
    if os.path.abspath(tree_path) == os.path.abspath(src_path):
        raise SystemExit(f"--output must not be the committed source tree ({src_path})")
    # resume this run's own tree if it exists, otherwise seed from the committed one
    seed_path = tree_path if os.path.exists(tree_path) else src_path
    tree = json.load(open(seed_path))
    print(f"{comp}: seeded from {seed_path} -> writing {tree_path}")

    def champ(t):
        ev_n = [n for n in t["nodes"] if n["status"] == "evaluated" and isinstance(n["score"], (int, float))]
        return min(ev_n, key=lambda n: n["score"])
    v1_champ = champ(tree)["score"]
    n_start = len([n for n in tree["nodes"] if n["status"] == "evaluated"])
    # extend budget for a forward search
    tree.setdefault("search_state", {}).setdefault("budget", {})
    tree["search_state"]["budget"] = {"total_budget": n_start + args.nodes, "explore_burst_size": 6,
                                       "post_burst_patience": 10, "phase": "exploit",
                                       "burst_start_eval": 0, "best_at_burst_start": v1_champ,
                                       "evals_since_burst_improve": 0}
    models_avail = [m for m in ("xgb", "cat", "lgb") if m in wl]
    added = 0
    for i in range(args.nodes):
        pool = solo_pool(tree)
        pool_ids = [n["id"] for n in pool]
        if i % 2 == 1 and len(pool_ids) >= 2:            # full-pool reblend
            cfg = {"kind": "blend", "members": sorted(pool_ids), "weight_search": "dirichlet"}
            mut = f"[V3GEN] pool reblend over {len(pool_ids)} members (dirichlet)"
            parent = pool[0]["id"]
        else:                                            # fresh bias-dominated solo variant
            fam = models_avail[(i // 2) % len(models_avail)]
            base = next((n["config"] for n in pool if n["config"].get("model") == fam), pool[0]["config"])
            cfg = variant(base, wl, i)
            mut = f"[V3GEN] shallow-reg {cfg['model']} variant (decorrelating candidate)"
            parent = tree["root_id"]
        nid = hv3.next_id(tree)
        try:
            r = ev.evaluate(core(cfg), node_id=nid, timeout_s=300)
        except Exception as e:  # noqa: BLE001
            r = {"score": None, "status": "failed", "wall_s": 0.0, "result": None, "error": str(e)}
        hv3.add_node(tree, parent, mut, core(cfg), r["score"], r["status"], r["wall_s"])
        if r["status"] == "evaluated":
            added += 1
        json.dump(tree, open(tree_path, "w"), ensure_ascii=False, indent=2)
        if hv3.should_stop(tree):
            break
    v3_champ = champ(tree)["score"]
    print(f"{comp}: v1-adapter champion={v1_champ:.6f} -> v3-searched champion={v3_champ:.6f} "
          f"(Δ={v3_champ - v1_champ:+.6f}; added {added} nodes; champion #{champ(tree)['id']} "
          f"kind={champ(tree)['config'].get('kind')}; tree={tree_path})")


if __name__ == "__main__":
    main()
