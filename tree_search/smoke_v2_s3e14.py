"""tree_search/smoke_v2_s3e14.py — Phase D-1 smoke test / comparability proof: re-run
root + 2 SOLO seed nodes (CAT, XGB) + 1 BLEND node for s3e14 under harness_v2's node
bookkeeping (add_root/add_node with dedup + kind field + adaptive-plateau logic wired
in), reusing already-cached OOFs from tree_search/cache_s3e14/ wherever possible (only
re-trains a solo node if its cache is missing or its stored config doesn't match).

This deliberately does NOT re-derive eval_s3e14.py's own math (that file is untouched,
v1) — the point is to prove the harness SWAP (v1 harness.py -> v2 harness_v2.py) does
not change a single digit of the score an unchanged per-comp evaluator computes. Output:
tree_search/smoke_v2_s3e14.json (small; not the source of truth — competitions/.../
experiments_tree.json remains that, and is NOT overwritten by this script).

Run: uv run python3 tree_search/smoke_v2_s3e14.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import eval_s3e14 as ev  # noqa: E402

COMP = "playground-series-s3e14"
V1_TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
OUT_PATH = os.path.join(_HERE, "smoke_v2_s3e14.json")


def strip_result(cfg):
    """v1's run_s3e14.eval_and_add stores {**child_cfg, "result": r["result"]} on
    evaluated nodes -- "result" is evaluate()'s OUTPUT echoed back for record-keeping,
    not part of the INPUT config eval_s3e14.evaluate()/evaluate_solo() expects."""
    return {k: v for k, v in cfg.items() if k != "result"}


def _cached_score(node_id, expected_config):
    """Return the cached score for node_id if a cache exists AND its stored config
    matches expected_config exactly (else None -> caller should re-evaluate/retrain)."""
    path = os.path.join(ev.CACHE_DIR, f"solo_{node_id}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    try:
        stored_cfg = json.loads(str(d["config_json"]))
    except Exception:
        return None
    if stored_cfg != expected_config:
        return None
    return float(d["score"])


def eval_solo_reusing_cache(node_id, config):
    cached = _cached_score(node_id, config)
    if cached is not None:
        return round(cached, 5), "cache-hit (no retrain)"
    r = ev.evaluate(config, node_id=node_id, timeout_s=180)
    return r["score"], "retrained (no matching cache found)"


def main():
    v1_tree = json.load(open(V1_TREE_PATH))
    v1_by_id = {n["id"]: n for n in v1_tree["nodes"]}

    tree = hv2.new_tree(COMP)
    results = []

    # --- root (#0) ---
    root_cfg_in = strip_result(v1_by_id[0]["config"])
    root_score, root_note = eval_solo_reusing_cache(0, root_cfg_in)
    hv2.add_root(tree, v1_by_id[0]["mutation"], {**root_cfg_in, "kind": "solo"},
                 root_score, "evaluated", 0.0)
    results.append(dict(node="root(#0)", v1_score=v1_by_id[0]["score"], v2_score=root_score,
                        match=(v1_by_id[0]["score"] == root_score), note=root_note))

    # --- 2 solo seed nodes: CAT(#1), XGB(#2) ---
    for name, v1_id in (("CAT", 1), ("XGB", 2)):
        cfg_in = strip_result(v1_by_id[v1_id]["config"])
        score, note = eval_solo_reusing_cache(v1_id, cfg_in)
        nid, dup = hv2.add_node(tree, tree["root_id"], v1_by_id[v1_id]["mutation"],
                                 {**cfg_in, "kind": "solo"}, score, "evaluated", 0.0)
        assert dup is None, f"unexpected dedup rejection for {name}: dup={dup}"
        assert nid == v1_id, f"node id drift for {name}: expected {v1_id}, got {nid}"
        results.append(dict(node=f"{name}(#{v1_id})", v1_score=v1_by_id[v1_id]["score"],
                            v2_score=score, match=(v1_by_id[v1_id]["score"] == score), note=note))

    # --- 1 blend node: #7, 3-way root+XGB+CAT, dirichlet + snap. Uses eval_s3e14's OWN,
    # unchanged evaluate_blend (reads CACHED member OOFs 0/1/2, no retraining) so this
    # step proves harness_v2 bookkeeping parity, not a reimplementation of the search. ---
    blend_cfg_in = strip_result(v1_by_id[7]["config"])
    r = ev.evaluate(blend_cfg_in, node_id=None, timeout_s=180)
    nid, dup = hv2.add_node(tree, tree["root_id"], v1_by_id[7]["mutation"],
                             {**blend_cfg_in, "kind": "blend"}, r["score"], r["status"], r["wall_s"])
    # NOTE: this smoke tree only seeds 3 solo nodes (root/CAT/XGB) before the blend, vs
    # v1's full 6-solo-lineage tree, so the blend node's id here (#3) legitimately
    # differs from v1's #7 -- only the SCORE is the comparability claim, not the id.
    assert dup is None, f"unexpected dedup rejection for blend node: dup={dup}"
    results.append(dict(node=f"BLEND(v1 #7, v2 #{nid})", v1_score=v1_by_id[7]["score"], v2_score=r["score"],
                        match=(v1_by_id[7]["score"] == r["score"]),
                        note="recomputed via unchanged eval_s3e14.evaluate_blend, cached members 0/1/2"))

    all_match = all(res["match"] for res in results)
    out = {"comp": COMP, "all_match": all_match, "results": results, "tree_v2": tree}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"Comparability check: {'PASS' if all_match else 'FAIL'}")
    for res in results:
        print(f"  {res['node']}: v1={res['v1_score']} v2={res['v2_score']} "
              f"match={res['match']} ({res['note']})")
    print(f"Written to {OUT_PATH}")
    if not all_match:
        sys.exit(1)


if __name__ == "__main__":
    main()
