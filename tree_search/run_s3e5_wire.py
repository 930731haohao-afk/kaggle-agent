"""tree_search/run_s3e5_wire.py — arm-A pilot: mechanically wire priors into the s3e5 search.

Paired attribution design (zero re-baseline risk):
  arm OFF = the committed historical v2 tree AS-IS
            (competitions/playground-series-s3e5/experiments_tree_v2.json, 23 nodes,
            champion #17 blend QWK 0.57066) — it IS the "priors not wired" arm, because
            that run stored tree["priors"] and nothing ever read it.
  arm A   = a COPY of that tree + prior_wiring.wire_priors() candidates evaluated on the
            same fold split and the same member OOF cache (cache_s3e5/v2). Shared nodes
            are therefore IDENTICAL by construction; the only delta is the wired nodes.

Artifacts (all NEW files; the committed tree and existing caches are never mutated):
  tree_search/wire_s3e5_A.json       — arm-A tree (historical 23 nodes + wired nodes)
  tree_search/wire_s3e5_ledger.json  — full disposition ledger + per-candidate results

Honest expectations, stated before running: the NNLS convex-optimality analysis
(docs/recombine_findings.md) predicts pool-blend candidates cannot beat an already
convex-optimal champion, and a regularization push on an Optuna-tuned config is
flat-to-worse more often than not. The pilot's claim is NOT "wiring improves scores";
it is "priors now demonstrably reach the queue, get evaluated, and are traceable"
— which the stage-5 no-op finding showed was previously false.
"""

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
import eval_s3e5_v2 as ev           # noqa: E402

REPO = os.path.dirname(_HERE)
OFF_TREE_PATH = os.path.join(REPO, "competitions", "playground-series-s3e5",
                             "experiments_tree_v2.json")
A_TREE_PATH = os.path.join(_HERE, "wire_s3e5_A.json")
LEDGER_PATH = os.path.join(_HERE, "wire_s3e5_ledger.json")

COMP_META = {"metric": "qwk", "tags": ["ordinal", "small-data", "wine"],
             "data_type": "tabular", "keywords": ["qwk", "序數", "小樣本"]}
CTX = dict(has_categorical=False, prob_metric=False)  # blend_members filled at runtime
EVAL_TIMEOUT_S = 120


def champion(tree):
    ev_nodes = [n for n in tree["nodes"] if n["status"] == "evaluated"
                and isinstance(n["score"], (int, float))]
    return min(ev_nodes, key=lambda n: n["score"])


def main():
    t0 = time.time()

    # ---- arm OFF: the committed historical tree, read-only ----
    off = json.load(open(OFF_TREE_PATH))
    off_champ = champion(off)
    print(f"[OFF] historical tree: {len(off['nodes'])} nodes, "
          f"champion #{off_champ['id']} QWK={-off_champ['score']:.5f}")

    # ---- arm A: copy + wire + evaluate ----
    tree = deepcopy(off)
    priors = hv4.suggest_priors_v4(COMP_META, mode="ext")
    cached_solos = [n["id"] for n in tree["nodes"]
                    if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    ctx = dict(CTX, blend_members=cached_solos)
    wired = pw.wire_priors(priors, tree["nodes"][0]["config"], tree, ctx)
    print(f"[A]  {pw.ledger_summary(wired['ledger'])}")

    results = []
    for cand in wired["candidates"]:
        nid = hv2.next_id(tree)
        r = ev.evaluate(cand["config"], node_id=nid, timeout_s=EVAL_TIMEOUT_S)
        stored = cand["config"] if r.get("result") is None else {**cand["config"], "result": r["result"]}
        real_nid, dup = hv2.add_node(tree, tree["root_id"], cand["mutation"], stored,
                                     r["score"], r["status"], r["wall_s"], allow_duplicate=True)
        assert dup is None and real_nid == nid
        qwk = r["result"]["qwk"] if r.get("result") and "qwk" in r["result"] else None
        results.append(dict(node_id=nid, tag=cand["tag"], rule=cand["rule"],
                            status=r["status"], qwk=qwk, wall_s=round(r["wall_s"], 2)))
        print(f"[A]  wired node #{nid} {cand['tag']}/{cand['rule']}: "
              f"status={r['status']} QWK={qwk} wall_s={r['wall_s']:.2f}")

    hv2.save(tree, A_TREE_PATH)

    # ---- paired comparison ----
    a_champ = champion(tree)
    prior_nodes = [n for n in tree["nodes"] if n["mutation"].startswith("[PRIOR-")]
    ranked = sorted([n for n in tree["nodes"] if n["status"] == "evaluated"
                     and isinstance(n["score"], (int, float))], key=lambda n: n["score"])
    ranks = {n["id"]: i + 1 for i, n in enumerate(ranked)}

    print("\n================ paired attribution summary ================")
    print(f"OFF: {len(off['nodes'])} nodes, champion #{off_champ['id']} "
          f"QWK={-off_champ['score']:.5f}, [PRIOR-*] nodes: "
          f"{sum(1 for n in off['nodes'] if n['mutation'].startswith('[PRIOR-'))}")
    print(f"A  : {len(tree['nodes'])} nodes, champion #{a_champ['id']} "
          f"QWK={-a_champ['score']:.5f}, [PRIOR-*] nodes: {len(prior_nodes)}")
    for n in prior_nodes:
        print(f"     #{n['id']} rank {ranks[n['id']]}/{len(ranked)} "
              f"QWK={-n['score']:.5f}  {n['mutation'][:80]}")
    changed = a_champ["id"] != off_champ["id"]
    print(f"champion changed: {changed}"
          + (f" (delta QWK {-a_champ['score'] - -off_champ['score']:+.5f})" if changed else ""))

    json.dump(dict(comp="playground-series-s3e5",
                   off=dict(nodes=len(off["nodes"]), champion_id=off_champ["id"],
                            champion_qwk=-off_champ["score"]),
                   arm_a=dict(nodes=len(tree["nodes"]), champion_id=a_champ["id"],
                              champion_qwk=-a_champ["score"], wired_results=results),
                   ledger=wired["ledger"]),
              open(LEDGER_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"\nledger -> {LEDGER_PATH}\ntree   -> {A_TREE_PATH}")
    print(f"total wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
