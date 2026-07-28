"""gen_proposals.py — build robust, schema-valid llm_proposals_<comp>.json for the adapted
S3 comps by mirroring an EXISTING pool solo config (guaranteed-valid keys) and making a
bias-dominated decorrelating variant: shallow depth + strong regularization + a fresh seed.
One or two variants per comp (from distinct model families where available). Skips comps
whose pool has no GBDT family (e.g. s3e20 structural pool).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH_KEY = {"lgb": "max_depth", "xgb": "max_depth", "cat": "depth"}
SEED_KEY = {"lgb": "random_state", "xgb": "random_state", "cat": "random_seed"}


def shallow_variant(cfg, wl_keys, seed):
    m = cfg["model"]
    p = {k: v for k, v in cfg.get("params", {}).items() if k in wl_keys}
    dk = DEPTH_KEY.get(m)
    if dk and dk in wl_keys:
        p[dk] = 2 if m != "cat" else 3
    if m == "cat":
        if "l2_leaf_reg" in wl_keys: p["l2_leaf_reg"] = 12.0
    else:
        if "reg_alpha" in wl_keys: p["reg_alpha"] = 3.0
        if "reg_lambda" in wl_keys: p["reg_lambda"] = 10.0
        if "min_child_weight" in wl_keys: p["min_child_weight"] = 40
        if "min_child_samples" in wl_keys: p["min_child_samples"] = 120
        if "num_leaves" in wl_keys: p["num_leaves"] = 4
    sk = SEED_KEY.get(m)
    if sk and sk in wl_keys: p[sk] = seed
    return {"kind": "solo", "model": m, "params": p, "features": {"drop": []}}


def main():
    comp = sys.argv[1]
    d = json.load(open(os.path.join(_HERE, f"llm_proposer_input_{comp}.json")))
    wl = d["param_whitelist"]
    gbdt = [m for m in ("xgb", "lgb", "cat") if m in wl]
    if not gbdt:
        print(f"[skip] {comp}: no GBDT family in pool (structural) -> injection N/A"); return
    # one template solo config per available family
    tmpl = {}
    for n in d["evaluated_nodes"]:
        c = n.get("config", {})
        if c.get("kind") == "solo" and c.get("model") in gbdt and c["model"] not in tmpl:
            tmpl[c["model"]] = c
    proposals = []
    for i, m in enumerate([x for x in ("xgb", "cat", "lgb") if x in tmpl][:2]):
        v = shallow_variant(tmpl[m], set(wl[m]), seed=7 + i)
        proposals.append({
            "config": v,
            "expected": f"Solo (shallow {m}) below the pool's best member but decorrelated; tests whether a bias-dominated external member lifts the reblend.",
            "prior_tags": ["INT-02", "INT-04"],
            "rationale": f"Shallow, heavily-regularized {m} = bias-dominated member to decorrelate the variance-leaning pool (the s6e1/s4e1 winning archetype). Mirrors an existing pool config's schema; only depth/reg/seed changed.",
        })
    out = {"meta": {"comp": comp, "proposer": "claude (mirror-existing-schema variant)",
                    "strategy": "bias-dominated shallow variants of existing pool members for decorrelation"},
           "proposals": proposals}
    json.dump(out, open(os.path.join(_HERE, f"llm_proposals_{comp}.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[ok] {comp}: {len(proposals)} proposals ({[p['config']['model'] for p in proposals]})")


if __name__ == "__main__":
    main()
