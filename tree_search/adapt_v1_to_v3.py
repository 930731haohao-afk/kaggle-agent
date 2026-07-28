"""adapt_v1_to_v3.py — reformat a committed v1/v2 experiments_tree into the v3 layout the
wiring runner (run_wire_v3.py) reads. The ONLY structural difference is that v3 stores each
node's `result` under search_state.node_results[str(id)] and keeps config result-free
(core() convention). This does NOT re-run the search — it faithfully reformats the committed
tree so the ±1e-5 injection wiring can operate on the real committed pool + champion.

Usage: uv run python3 tree_search/adapt_v1_to_v3.py <comp> <old_tree_filename>
  e.g. ... s3e16 experiments_tree.json
Writes competitions/playground-series-<comp>/experiments_tree_v3.json (only if absent).
"""
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)


def core(cfg):
    out = {k: v for k, v in cfg.items() if k != "result"}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def main():
    comp, old_name = sys.argv[1], sys.argv[2]
    slug = f"playground-series-{comp}"
    cdir = os.path.join(REPO, "competitions", slug)
    old_path = os.path.join(cdir, old_name)
    out_path = os.path.join(cdir, "experiments_tree_v3.json")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists"); return
    old = json.load(open(old_path))

    nodes, node_results = [], {}
    for n in old["nodes"]:
        n = copy.deepcopy(n)
        result = n["config"].get("result")
        n["config"] = core(n["config"])
        n.setdefault("kind", n["config"].get("kind", "solo"))
        nodes.append(n)
        if n["status"] == "evaluated" and result is not None:
            node_results[str(n["id"])] = result

    ss = dict(old.get("search_state", {}))
    ss["node_results"] = node_results
    ev_nodes = [n for n in nodes if n["status"] == "evaluated" and isinstance(n["score"], (int, float))]
    champ = min(ev_nodes, key=lambda n: n["score"]) if ev_nodes else None
    ss["budget"] = {"total_budget": len(ev_nodes), "phase": "stopped",
                    "stop_reason": "adapted from committed v1/v2 tree (no re-search)",
                    "best_at_burst_start": champ["score"] if champ else None}

    v3 = dict(comp=old.get("comp", slug), root_id=old.get("root_id", 0),
              nodes=nodes, search_state=ss,
              prior_usage_summary=old.get("prior_usage_summary", {}),
              dedup_rejections=old.get("dedup_rejections", []),
              boundary_candidates_log=[], cost_guard_fired=[])
    json.dump(v3, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[ok] {comp}: {len(nodes)} nodes ({len(ev_nodes)} evaluated), champion #{champ['id']} "
          f"score={champ['score']:.6f} -> {out_path}")


if __name__ == "__main__":
    main()
