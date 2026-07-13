"""tree_search/run_s6e1_wire.py — three-arm prior-wiring attribution, competition #2
(s6e1, exam-score regression, R², 630k rows, 21-feature GBDT pool — cross-metric check
of the s3e5 pilot: does "arm A confirms the theorem / arm B expands the pool" replicate?)

Arms (paired; OFF tree never mutated):
  OFF : committed experiments_tree_v3.json (48 nodes, champion #27 blend R2 0.787183 —
        the recombine-null comp). Historically the v3 driver never even CALLED
        suggest_priors, so this arm is a purer no-delivery baseline than s3e5's.
  A   : --arm A — mechanical prior_wiring.wire_priors() on 27 matched priors
        (20 INT + 7 EXT; the generic Optuna/ensemble/feature-engineering sections
        apply to any mid-size GBDT-blend regression like this one).
  B   : --arm B — frozen LLM proposals (llm_proposals_s6e1.json, propose-once/freeze/
        replay) validated against a DYNAMIC whitelist derived from the tree itself
        (param keys per model = union of keys already exercised in tree configs;
        droppable feature names = union of names seen in configs) then evaluated,
        plus the declared full-pool reblend.

Usage: uv run python3 tree_search/run_s6e1_wire.py --arm A|B
Artifacts: wire_s6e1_{A|B}.json, wire_s6e1_ledger_{a|b}.json (new files only).
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness_v2 as hv2            # noqa: E402
import harness_v4 as hv4            # noqa: E402
import prior_wiring as pw           # noqa: E402
import eval_s6e1 as ev              # noqa: E402

REPO = os.path.dirname(_HERE)
OFF_TREE_PATH = os.path.join(REPO, "competitions", "playground-series-s6e1",
                             "experiments_tree_v3.json")
PROPOSALS_PATH = os.path.join(_HERE, "llm_proposals_s6e1.json")
EVAL_TIMEOUT_S = 900

COMP_META = {"metric": "r2", "tags": ["regression", "tabular"], "data_type": "tabular",
             "keywords": ["optuna", "ensemble", "特徵工程", "共線"]}
CTX_BASE = dict(has_categorical=True, prob_metric=False)


def _recipe_hash(cfg):
    return hv2.config_hash({k: v for k, v in cfg.items() if k != "result"})


def champion(tree):
    evn = [n for n in tree["nodes"] if n["status"] == "evaluated"
           and isinstance(n["score"], (int, float))]
    return min(evn, key=lambda n: n["score"])


def best_solo(tree):
    solos = [n for n in tree["nodes"] if n["status"] == "evaluated"
             and n["config"].get("kind") == "solo"]
    return min(solos, key=lambda n: n["score"])


def cached_solo_ids(tree):
    have = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(ev.CACHE_DIR)
            if f.startswith("solo_")}
    return [n["id"] for n in tree["nodes"] if n["status"] == "evaluated"
            and n["config"].get("kind") == "solo" and n["id"] in have]


def dynamic_whitelist(tree):
    """Param keys per model = union of keys already exercised by evaluated nodes; the
    LLM may only turn knobs this evaluator has demonstrably accepted before."""
    wl = {}
    for n in tree["nodes"]:
        cfg = n["config"]
        if cfg.get("kind") == "solo":
            wl.setdefault(cfg.get("model"), set()).update(cfg.get("params", {}))
    return wl


def known_drop_names(tree):
    names = set()
    for n in tree["nodes"]:
        names.update(n["config"].get("features", {}).get("drop", []) or [])
    return names


def eval_and_attach(tree, cfg, mutation):
    nid = hv2.next_id(tree)
    r = ev.evaluate(cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
    stored = cfg if r.get("result") is None else {**cfg, "result": r["result"]}
    real_nid, dup = hv2.add_node(tree, tree["root_id"], mutation, stored, r["score"],
                                 r["status"], r["wall_s"], allow_duplicate=True)
    assert dup is None and real_nid == nid
    r2 = r["result"]["r2"] if r.get("result") and "r2" in r["result"] else None
    return nid, r, r2


def summarize(off, tree, ledger, out_ledger_path, tag_prefix="[PRIOR-"):
    off_champ, new_champ = champion(off), champion(tree)
    ranked = sorted([n for n in tree["nodes"] if n["status"] == "evaluated"
                     and isinstance(n["score"], (int, float))], key=lambda n: n["score"])
    ranks = {n["id"]: i + 1 for i, n in enumerate(ranked)}
    print("\n================ summary ================")
    print(f"OFF champion #{off_champ['id']} R2={-off_champ['score']:.6f}")
    print(f"ARM champion #{new_champ['id']} R2={-new_champ['score']:.6f} "
          f"(changed: {new_champ['id'] != off_champ['id']})")
    for n in tree["nodes"]:
        if n["mutation"].startswith(tag_prefix) and n["id"] not in {m["id"] for m in off["nodes"]}:
            print(f"  #{n['id']} rank {ranks.get(n['id'])}/{len(ranked)} "
                  f"R2={-n['score']:.6f}  {n['mutation'][:90]}")
    json.dump(dict(comp="playground-series-s6e1",
                   off=dict(champion_id=off_champ["id"], champion_r2=-off_champ["score"]),
                   arm=dict(nodes=len(tree["nodes"]), champion_id=new_champ["id"],
                            champion_r2=-new_champ["score"]),
                   ledger=ledger),
              open(out_ledger_path, "w"), ensure_ascii=False, indent=2)
    print(f"ledger -> {out_ledger_path}")


def run_arm_a():
    off = json.load(open(OFF_TREE_PATH))
    print(f"[OFF] {len(off['nodes'])} nodes, champion #{champion(off)['id']} "
          f"R2={-champion(off)['score']:.6f}")
    tree = deepcopy(off)
    priors = hv4.suggest_priors_v4(COMP_META, mode="ext")
    ctx = dict(CTX_BASE, blend_members=cached_solo_ids(tree))
    base = {k: v for k, v in best_solo(tree)["config"].items() if k != "result"}
    wired = pw.wire_priors(priors, base, tree, ctx)
    print(f"[A] {pw.ledger_summary(wired['ledger'])}")
    results = []
    for cand in wired["candidates"]:
        nid, r, r2 = eval_and_attach(tree, cand["config"], cand["mutation"])
        results.append(dict(node_id=nid, tag=cand["tag"], rule=cand["rule"],
                            status=r["status"], r2=r2, wall_s=round(r["wall_s"], 1)))
        print(f"[A] wired #{nid} {cand['tag']}/{cand['rule']}: {r['status']} "
              f"R2={r2} wall_s={r['wall_s']:.0f}")
    hv2.save(tree, os.path.join(_HERE, "wire_s6e1_A.json"))
    summarize(off, tree, dict(disposition=wired["ledger"], results=results),
              os.path.join(_HERE, "wire_s6e1_ledger_a.json"))


def run_arm_b():
    off = json.load(open(OFF_TREE_PATH))
    print(f"[OFF] {len(off['nodes'])} nodes, champion #{champion(off)['id']} "
          f"R2={-champion(off)['score']:.6f}")
    frozen = json.load(open(PROPOSALS_PATH))
    tree = deepcopy(off)
    wl = dynamic_whitelist(tree)
    names = known_drop_names(tree)
    existing = {_recipe_hash(n["config"]) for n in tree["nodes"]}
    ledger, new_ids = [], []
    for i, prop in enumerate(frozen["proposals"]):
        tag, cfg = f"PRIOR-LLM-{i:02d}", prop["config"]
        ok, reason = True, ""
        if cfg.get("kind") != "solo":
            ok, reason = False, "frozen proposals must be solo"
        elif cfg.get("model") not in wl:
            ok, reason = False, f"model {cfg.get('model')!r} never exercised in tree"
        else:
            bad = set(cfg.get("params", {})) - wl[cfg["model"]]
            if bad:
                ok, reason = False, f"params never exercised for {cfg['model']}: {sorted(bad)}"
            unknown = set(cfg.get("features", {}).get("drop", [])) - names
            if ok and unknown:
                ok, reason = False, f"unknown feature names: {sorted(unknown)}"
        if ok and _recipe_hash(cfg) in existing:
            ok, reason = False, "deduped vs existing node"
        if not ok:
            ledger.append(dict(tag=tag, status="rejected", reason=reason))
            print(f"[B] {tag} REJECTED: {reason}")
            continue
        existing.add(_recipe_hash(cfg))
        mut = f"[{tag}] {prop['rationale'][:120]} <= priors {','.join(prop['prior_tags'])}"
        nid, r, r2 = eval_and_attach(tree, cfg, mut)
        ledger.append(dict(tag=tag, status=r["status"], node_id=nid, r2=r2,
                           wall_s=round(r["wall_s"], 1), prior_tags=prop["prior_tags"],
                           expected=prop.get("expected", "")))
        new_ids.append(nid)
        print(f"[B] wired #{nid} {tag}: {r['status']} R2={r2} wall_s={r['wall_s']:.0f}")
    if new_ids:
        pool = sorted(cached_solo_ids(tree))
        cfg = {"kind": "blend", "members": pool, "weight_search": "dirichlet"}
        if _recipe_hash(cfg) not in existing:
            mut = f"[PRIOR-LLM-REBLEND] full-pool reblend incl LLM members {new_ids}"
            nid, r, r2 = eval_and_attach(tree, cfg, mut)
            ledger.append(dict(tag="PRIOR-LLM-REBLEND", status=r["status"], node_id=nid,
                               r2=r2, wall_s=round(r["wall_s"], 1)))
            print(f"[B] reblend #{nid}: R2={r2} wall_s={r['wall_s']:.0f}")
    hv2.save(tree, os.path.join(_HERE, "wire_s6e1_B.json"))
    summarize(off, tree, ledger, os.path.join(_HERE, "wire_s6e1_ledger_b.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    args = ap.parse_args()
    t0 = time.time()
    (run_arm_a if args.arm == "A" else run_arm_b)()
    print(f"total wall {time.time() - t0:.0f}s")
