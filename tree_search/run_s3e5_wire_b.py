"""tree_search/run_s3e5_wire_b.py — arm-B pilot: LLM-proposed prior wiring on s3e5.

Same paired design as run_s3e5_wire.py (arm OFF = committed v2 tree untouched), but the
candidates come from a FROZEN LLM proposal file instead of mechanical templates:

  proposer  : Claude subagent, given llm_proposer_input_s3e5.json (priors + full search
              state + evaluator contract) — the project's propose-once/freeze/replay
              pattern (docs/recombine_findings.md), so reruns replay the frozen file and
              never re-call the LLM: bit-replayable arm-B trees.
  proposals : tree_search/llm_proposals_s3e5.json (4 solos, each citing prior tags,
              all aimed at DECORRELATING the pool — the one mechanism the NNLS analysis
              leaves open).
  protocol  : evaluate the 4 solos tagged [PRIOR-LLM-nn]; then ONE full-pool reblend
              including the new members (declared up front; symmetric with arm A's pool
              blend so the arms stay comparable); finally report the OOF error
              correlation of each new member vs the historical pool, which is the
              proposals' own testable claim.

Validation before evaluation (LLM output is untrusted): schema whitelist per model,
feature-drop names must be a subset of names already seen in the tree's configs,
recipe-hash dedupe vs tree + batch. Invalid or duplicate proposals are ledgered, not
silently dropped.

Artifacts: tree_search/wire_s3e5_B.json, tree_search/wire_s3e5_ledger_b.json.
"""

import json
import os
import sys
import time
from copy import deepcopy

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness_v2 as hv2            # noqa: E402
import eval_s3e5_v2 as ev           # noqa: E402

REPO = os.path.dirname(_HERE)
OFF_TREE_PATH = os.path.join(REPO, "competitions", "playground-series-s3e5",
                             "experiments_tree_v2.json")
PROPOSALS_PATH = os.path.join(_HERE, "llm_proposals_s3e5.json")
B_TREE_PATH = os.path.join(_HERE, "wire_s3e5_B.json")
LEDGER_PATH = os.path.join(_HERE, "wire_s3e5_ledger_b.json")
EVAL_TIMEOUT_S = 120

_PARAM_WHITELIST = {
    "lgb": {"num_leaves", "max_depth", "learning_rate", "min_child_samples", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "n_estimators", "random_state"},
    "xgb": {"max_depth", "learning_rate", "min_child_weight", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "n_estimators", "random_state"},
    "cat": {"depth", "learning_rate", "l2_leaf_reg", "iterations", "random_strength",
            "min_data_in_leaf", "random_seed"},
}


def _recipe_hash(cfg):
    return hv2.config_hash({k: v for k, v in cfg.items() if k != "result"})


def _known_feature_names(tree):
    names = set()
    for n in tree["nodes"]:
        names.update(n["config"].get("features", {}).get("drop", []) or [])
    return names


def validate(cfg, tree, known_drop_names):
    """Return (ok, reason). LLM output is untrusted input."""
    if cfg.get("kind") != "solo":
        return False, f"arm-B frozen proposals must be solo, got kind={cfg.get('kind')!r}"
    model = cfg.get("model")
    if model not in _PARAM_WHITELIST:
        return False, f"unknown model {model!r}"
    bad = set(cfg.get("params", {})) - _PARAM_WHITELIST[model]
    if bad:
        return False, f"non-whitelisted params for {model}: {sorted(bad)}"
    nest = cfg.get("params", {}).get("n_estimators") or cfg.get("params", {}).get("iterations") or 0
    if nest > 1500:
        return False, f"n_estimators/iterations {nest} > 1500 (timeout budget)"
    drop = cfg.get("features", {}).get("drop", [])
    unknown = set(drop) - known_drop_names
    if unknown:
        return False, f"feature names never seen in tree configs: {sorted(unknown)}"
    if cfg.get("rounder", "full_oof") not in ("full_oof", "fold_avg"):
        return False, f"bad rounder {cfg.get('rounder')!r}"
    return True, ""


def champion(tree):
    ev_nodes = [n for n in tree["nodes"] if n["status"] == "evaluated"
                and isinstance(n["score"], (int, float))]
    return min(ev_nodes, key=lambda n: n["score"])


def oof_corr_vs_pool(new_ids, pool_ids):
    """Error-correlation (residual Pearson) of each new member vs its closest pool member."""
    y = ev._y.astype(float)
    out = {}
    pool = {}
    for pid in pool_ids:
        try:
            pool[pid] = hv2.load_oof(ev.CACHE_DIR, pid) - y
        except Exception:
            pass
    for nid in new_ids:
        res = hv2.load_oof(ev.CACHE_DIR, nid) - y
        cors = {pid: float(np.corrcoef(res, pres)[0, 1]) for pid, pres in pool.items()}
        top = max(cors, key=cors.get)
        out[nid] = dict(max_corr=round(cors[top], 4), vs=top,
                        mean_corr=round(float(np.mean(list(cors.values()))), 4))
    return out


def main():
    t0 = time.time()
    off = json.load(open(OFF_TREE_PATH))
    off_champ = champion(off)
    print(f"[OFF] {len(off['nodes'])} nodes, champion #{off_champ['id']} QWK={-off_champ['score']:.5f}")

    frozen = json.load(open(PROPOSALS_PATH))
    tree = deepcopy(off)
    known_names = _known_feature_names(tree)
    existing = {_recipe_hash(n["config"]) for n in tree["nodes"]}

    ledger, new_solo_ids = [], []
    for i, prop in enumerate(frozen["proposals"]):
        tag = f"PRIOR-LLM-{i:02d}"
        cfg = prop["config"]
        ok, reason = validate(cfg, tree, known_names)
        if ok and _recipe_hash(cfg) in existing:
            ok, reason = False, "deduped vs existing node (recipe hash)"
        if not ok:
            ledger.append(dict(tag=tag, status="rejected", reason=reason,
                               prior_tags=prop["prior_tags"]))
            print(f"[B]  {tag} REJECTED: {reason}")
            continue
        existing.add(_recipe_hash(cfg))
        nid = hv2.next_id(tree)
        r = ev.evaluate(cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
        stored = cfg if r.get("result") is None else {**cfg, "result": r["result"]}
        mut = (f"[{tag}] {prop['rationale'][:120]} <= priors {','.join(prop['prior_tags'])}")
        real_nid, dup = hv2.add_node(tree, tree["root_id"], mut, stored, r["score"],
                                     r["status"], r["wall_s"], allow_duplicate=True)
        assert dup is None and real_nid == nid
        qwk = r["result"]["qwk"] if r.get("result") and "qwk" in r["result"] else None
        ledger.append(dict(tag=tag, status=r["status"], node_id=nid, qwk=qwk,
                           wall_s=round(r["wall_s"], 2), prior_tags=prop["prior_tags"],
                           expected=prop["expected"]))
        new_solo_ids.append(nid)
        print(f"[B]  wired node #{nid} {tag}: status={r['status']} QWK={qwk} wall_s={r['wall_s']:.1f}")

    # declared follow-up: one full-pool reblend including the new members
    if new_solo_ids:
        pool = [n["id"] for n in tree["nodes"] if n["status"] == "evaluated"
                and n["config"].get("kind") == "solo"]
        cfg = {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet",
               "rounder": "full_oof"}
        if _recipe_hash(cfg) not in existing:
            nid = hv2.next_id(tree)
            r = ev.evaluate(cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
            stored = cfg if r.get("result") is None else {**cfg, "result": r["result"]}
            mut = (f"[PRIOR-LLM-REBLEND] full-pool reblend including LLM members "
                   f"{new_solo_ids} (declared arm-B protocol step)")
            hv2.add_node(tree, tree["root_id"], mut, stored, r["score"], r["status"],
                         r["wall_s"], allow_duplicate=True)
            qwk = r["result"]["qwk"] if r.get("result") and "qwk" in r["result"] else None
            ledger.append(dict(tag="PRIOR-LLM-REBLEND", status=r["status"], node_id=nid,
                               qwk=qwk, wall_s=round(r["wall_s"], 2)))
            print(f"[B]  reblend node #{nid}: QWK={qwk} wall_s={r['wall_s']:.1f}")

    hv2.save(tree, B_TREE_PATH)

    # the proposals' own testable claim: decorrelation vs the historical pool
    hist_pool = [n["id"] for n in off["nodes"] if n["status"] == "evaluated"
                 and n["config"].get("kind") == "solo"]
    corr = oof_corr_vs_pool(new_solo_ids, hist_pool) if new_solo_ids else {}

    b_champ = champion(tree)
    ranked = sorted([n for n in tree["nodes"] if n["status"] == "evaluated"
                     and isinstance(n["score"], (int, float))], key=lambda n: n["score"])
    ranks = {n["id"]: i + 1 for i, n in enumerate(ranked)}

    print("\n================ arm-B summary ================")
    print(f"OFF champion #{off_champ['id']} QWK={-off_champ['score']:.5f}")
    print(f"B   champion #{b_champ['id']} QWK={-b_champ['score']:.5f} "
          f"(changed: {b_champ['id'] != off_champ['id']})")
    for row in ledger:
        if row.get("node_id") is not None:
            c = corr.get(row["node_id"])
            corr_s = f" err-corr max={c['max_corr']} (vs #{c['vs']}), mean={c['mean_corr']}" if c else ""
            print(f"  #{row['node_id']} rank {ranks[row['node_id']]}/{len(ranked)} "
                  f"QWK={row['qwk']}  {row['tag']}{corr_s}")

    json.dump(dict(comp="playground-series-s3e5", arm="B",
                   off=dict(champion_id=off_champ["id"], champion_qwk=-off_champ["score"]),
                   arm_b=dict(nodes=len(tree["nodes"]), champion_id=b_champ["id"],
                              champion_qwk=-b_champ["score"]),
                   ledger=ledger,
                   decorrelation={str(k): v for k, v in corr.items()}),
              open(LEDGER_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"\nledger -> {LEDGER_PATH}\ntree   -> {B_TREE_PATH}")
    print(f"total wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
