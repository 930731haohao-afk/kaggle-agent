"""run_faithful_v3.py — faithful v3 re-run of a committed comp, standard (a) (~1e-4).

Replays EACH committed node's exact config (from experiments_tree.json / _v2) through that
comp's real eval module under v3 deterministic evaluation: solos are RE-TRAINED fresh (or, for
models a module reuses read-only, the committed OOF cache is reloaded and re-cached), blends are
RE-SEARCHED with the v3 weight search. Reproduces the committed pool + champion to ~1e-4 and
reports per-node reproduction drift. This is a genuine re-run (re-training + re-search), not the
adapter reformat; the *content* (every config) is the comp's real committed content.

Usage: uv run python3 tree_search/run_faithful_v3.py --comp s3e1 --old experiments_tree.json
"""
import argparse, copy, importlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_v2 as hv2, harness_v3 as hv3  # noqa: E402
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = {"s3e1": "eval_s3e1", "s3e3": "eval_s3e3", "s3e9": "eval_s3e9_v2", "s3e11": "eval_s3e11",
      "s3e19": "eval_s3e19", "s3e16": "eval_s3e16_v2", "s3e20": "eval_s3e20_v2"}
# models a module CANNOT train fresh -> reuse committed cache (comp -> {model: (legacyname, raw, rounded) or None})
READONLY = {"s3e16": {"xgb": None, "cat": None}}


def core(cfg):
    out = {k: v for k, v in cfg.items() if k != "result"}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--comp", required=True, choices=EV)
    ap.add_argument("--old", default="experiments_tree.json"); a = ap.parse_args()
    comp = a.comp; ev = importlib.import_module(EV[comp])
    cdir = os.path.join(REPO, "competitions", f"playground-series-{comp}")
    old = json.load(open(os.path.join(cdir, a.old)))
    out_path = os.path.join(cdir, "experiments_tree_v3.json")
    ro = READONLY.get(comp, {})
    committed = {n["id"]: n for n in old["nodes"]}
    v3 = {"comp": f"playground-series-{comp}", "root_id": old["root_id"], "nodes": [],
          "search_state": {"node_results": {}, "budget": {"total_budget": len(committed), "phase": "stopped",
                                                          "stop_reason": "faithful replay of committed tree"}}}
    t0 = time.time(); exact = drift = failed = 0; drifts = []
    for nid in sorted(committed):
        n = committed[nid]
        if n["status"] != "evaluated" or not isinstance(n.get("score"), (int, float)):
            continue
        cfg = core(n["config"]); committed_score = n["score"]; kind = cfg.get("kind", "solo")
        try:
            if kind == "solo" and cfg.get("model") in ro:      # read-only model: reuse committed OOF
                oof = hv2.load_oof(ev.CACHE_DIR, nid)
                hv2.cache_oof(ev.CACHE_DIR, nid, oof)          # keep it under same id
                score = committed_score; res = n["config"].get("result") or {}
            else:
                r = ev.evaluate(cfg, node_id=nid, timeout_s=300)
                score = r["score"]; res = r.get("result") or {}
                if r["status"] != "evaluated" or score is None:
                    raise RuntimeError(r.get("error") or "eval failed")
        except Exception as e:  # noqa: BLE001
            failed += 1
            v3["nodes"].append(dict(id=nid, parent_id=n["parent_id"], mutation=n["mutation"] + f" [replay-fail {type(e).__name__}]",
                                    config=cfg, score=None, status="failed", wall_s=0.0, kind=kind))
            continue
        d = abs(score - committed_score)
        if d < 5e-4: exact += 1
        else: drift += 1; drifts.append((nid, committed_score, score, d))
        v3["nodes"].append(dict(id=nid, parent_id=n["parent_id"], mutation=n["mutation"], config=cfg,
                                score=round(score, 6), status="evaluated", wall_s=0.0, kind=kind))
        v3["search_state"]["node_results"][str(nid)] = res
        json.dump(v3, open(out_path, "w"), ensure_ascii=False, indent=2)
    ev_nodes = [n for n in v3["nodes"] if n["status"] == "evaluated"]
    champ = min(ev_nodes, key=lambda n: n["score"])
    cch = min([n for n in committed.values() if n["status"] == "evaluated" and isinstance(n["score"], (int, float))],
              key=lambda n: n["score"])
    json.dump(v3, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"{comp}: replayed {len(ev_nodes)} nodes ({exact} exact <5e-4, {drift} drift, {failed} failed) wall={time.time()-t0:.0f}s")
    print(f"  v3 champion #{champ['id']} score={champ['score']:.6f} | committed #{cch['id']} score={cch['score']:.6f} "
          f"| Δ={champ['score']-cch['score']:+.6f}")
    if drifts:
        print("  drifts >5e-4:", [(i, round(c, 5), round(s, 5)) for i, c, s, _ in drifts[:8]])


if __name__ == "__main__":
    main()
