"""tree_search/run_wire_v3.py — generic prior-wiring runner for v3-standard comps
(s4e1 AUC / s4e11 accuracy / s5e10 RMSE — extends the s3e5+s6e1 three-arm experiment,
see docs/prior_wiring_findings.md for design & the s6e1 significant result).

All three comps share the v3 infrastructure: experiments_tree_v3.json with TRUE
champion weights in search_state.node_results, deterministic evaluators, full solo OOF
caches. That is exactly what makes ±1e-5-scale paired comparisons defensible here and
NOT on the v1/v2 S3 batch.

Usage:
  uv run python3 tree_search/run_wire_v3.py --comp s4e1 --arm A
  uv run python3 tree_search/run_wire_v3.py --comp s4e1 --make-proposer-input
  uv run python3 tree_search/run_wire_v3.py --comp s4e1 --arm B   # needs frozen proposals

Artifacts per comp: wire_<comp>_{A|B}.json, wire_<comp>_ledger_{a|b}.json,
llm_proposer_input_<comp>.json, llm_proposals_<comp>.json (frozen by the operator).
"""

import argparse
import importlib
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

REPO = os.path.dirname(_HERE)
EVAL_TIMEOUT_S = 900

COMPS = {
    "s4e1": dict(
        slug="playground-series-s4e1", eval_mod="eval_s4e1", metric_key="auc",
        display=lambda s: -s,   # score = -AUC
        comp_meta={"metric": "auc", "tags": ["binary", "tabular", "churn"],
                   "data_type": "tabular", "keywords": ["auc", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=True, prob_metric=False),  # AUC is monotone-invariant
    ),
    "s4e11": dict(
        slug="playground-series-s4e11", eval_mod="eval_s4e11", metric_key="accuracy",
        display=lambda s: -s,   # score = -accuracy
        comp_meta={"metric": "accuracy", "tags": ["binary", "tabular", "threshold"],
                   "data_type": "tabular", "keywords": ["optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=True, prob_metric=False),
    ),
    "s5e10": dict(
        slug="playground-series-s5e10", eval_mod="eval_s5e10", metric_key="rmse",
        display=lambda s: s,    # score = +RMSE (already lower-better)
        comp_meta={"metric": "rmse", "tags": ["regression", "tabular"],
                   "data_type": "tabular", "keywords": ["rmse", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=True, prob_metric=False),
    ),
    "s6e2": dict(
        slug="playground-series-s6e2", eval_mod="eval_s6e2", metric_key="auc",
        display=lambda s: -s,   # score = -AUC
        comp_meta={"metric": "auc", "tags": ["binary", "tabular", "heart"],
                   "data_type": "tabular", "keywords": ["auc", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=False, prob_metric=False),  # all-numeric UCI features
    ),
    "s3e7": dict(
        slug="playground-series-s3e7", eval_mod="eval_s3e7", metric_key="auc",
        display=lambda s: -s,   # score = -AUC
        comp_meta={"metric": "auc", "tags": ["binary", "tabular", "cancellation"],
                   "data_type": "tabular", "keywords": ["auc", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=True, prob_metric=False),
    ),
    "s3e14": dict(
        slug="playground-series-s3e14", eval_mod="eval_s3e14", metric_key="mae",
        display=lambda s: s,    # score = +MAE (lower-better)
        comp_meta={"metric": "mae", "tags": ["regression", "tabular"],
                   "data_type": "tabular", "keywords": ["mae", "optuna", "ensemble", "特徵工程", "共線"]},
        ctx=dict(has_categorical=False, prob_metric=False),
    ),
    "s3e16": dict(
        slug="playground-series-s3e16", eval_mod="eval_s3e16_v2", metric_key="mae",
        display=lambda s: s,    # score = rounded MAE (lower-better)
        comp_meta={"metric": "mae", "tags": ["regression", "tabular", "integer-target"],
                   "data_type": "tabular", "keywords": ["mae", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=False, prob_metric=False),
    ),
    "s3e1": dict(
        slug="playground-series-s3e1", eval_mod="eval_s3e1", metric_key="rmse", display=lambda s: s,
        comp_meta={"metric": "rmse", "tags": ["regression", "tabular", "geo"],
                   "data_type": "tabular", "keywords": ["rmse", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=False, prob_metric=False),
    ),
    "s3e3": dict(
        slug="playground-series-s3e3", eval_mod="eval_s3e3", metric_key="auc", display=lambda s: -s,
        comp_meta={"metric": "auc", "tags": ["binary", "tabular"],
                   "data_type": "tabular", "keywords": ["auc", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=True, prob_metric=False),
    ),
    "s3e9": dict(
        slug="playground-series-s3e9", eval_mod="eval_s3e9_v2", metric_key="rmse", display=lambda s: s,
        comp_meta={"metric": "rmse", "tags": ["regression", "tabular"],
                   "data_type": "tabular", "keywords": ["rmse", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=False, prob_metric=False),
    ),
    "s3e11": dict(
        slug="playground-series-s3e11", eval_mod="eval_s3e11", metric_key="rmsle", display=lambda s: s,
        comp_meta={"metric": "rmsle", "tags": ["regression", "tabular"],
                   "data_type": "tabular", "keywords": ["rmsle", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=False, prob_metric=False),
    ),
    "s3e19": dict(
        slug="playground-series-s3e19", eval_mod="eval_s3e19", metric_key="smape", display=lambda s: s,
        comp_meta={"metric": "smape", "tags": ["regression", "tabular", "timeseries"],
                   "data_type": "tabular", "keywords": ["smape", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=True, prob_metric=False),
    ),
    "s3e20": dict(
        slug="playground-series-s3e20", eval_mod="eval_s3e20_v2", metric_key="rmse", display=lambda s: s,
        comp_meta={"metric": "rmse", "tags": ["regression", "tabular"],
                   "data_type": "tabular", "keywords": ["rmse", "optuna", "ensemble", "特徵工程"]},
        ctx=dict(has_categorical=False, prob_metric=False),
    ),
}


def _recipe_hash(cfg):
    return hv2.config_hash({k: v for k, v in cfg.items() if k != "result"})


def load_comp(comp):
    spec = COMPS[comp]
    ev = importlib.import_module(spec["eval_mod"])
    tree_path = os.path.join(REPO, "competitions", spec["slug"], "experiments_tree_v3.json")
    off = json.load(open(tree_path))
    return spec, ev, off


def champion(tree):
    evn = [n for n in tree["nodes"] if n["status"] == "evaluated"
           and isinstance(n["score"], (int, float))]
    return min(evn, key=lambda n: n["score"])


def best_solo(tree):
    solos = [n for n in tree["nodes"] if n["status"] == "evaluated"
             and n["config"].get("kind") == "solo"]
    return min(solos, key=lambda n: n["score"])


def cached_solo_ids(tree, ev):
    have = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(ev.CACHE_DIR)
            if f.startswith("solo_") and f.endswith(".npz")}
    return [n["id"] for n in tree["nodes"] if n["status"] == "evaluated"
            and n["config"].get("kind") == "solo" and n["id"] in have]


def eval_and_attach(tree, ev, cfg, mutation, metric_key):
    nid = hv2.next_id(tree)
    r = ev.evaluate(cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
    stored = cfg if r.get("result") is None else {**cfg, "result": r["result"]}
    real_nid, dup = hv2.add_node(tree, tree["root_id"], mutation, stored, r["score"],
                                 r["status"], r["wall_s"], allow_duplicate=True)
    assert dup is None and real_nid == nid
    met = r["result"].get(metric_key) if r.get("result") else None
    return nid, r, met


def summarize(off, tree, spec, ledger, out_path):
    disp = spec["display"]
    off_c, new_c = champion(off), champion(tree)
    ranked = sorted([n for n in tree["nodes"] if n["status"] == "evaluated"
                     and isinstance(n["score"], (int, float))], key=lambda n: n["score"])
    ranks = {n["id"]: i + 1 for i, n in enumerate(ranked)}
    print("\n================ summary ================")
    print(f"OFF champion #{off_c['id']} {spec['metric_key']}={disp(off_c['score']):.6f}")
    print(f"ARM champion #{new_c['id']} {spec['metric_key']}={disp(new_c['score']):.6f} "
          f"(changed: {new_c['id'] != off_c['id']})")
    off_ids = {n["id"] for n in off["nodes"]}
    for n in tree["nodes"]:
        if n["id"] not in off_ids and n["mutation"].startswith("[PRIOR-"):
            if isinstance(n["score"], (int, float)):
                print(f"  #{n['id']} rank {ranks.get(n['id'])}/{len(ranked)} "
                      f"{spec['metric_key']}={disp(n['score']):.6f}  {n['mutation'][:85]}")
            else:
                print(f"  #{n['id']} [{n['status']}] {n['mutation'][:85]}")
    json.dump(dict(comp=spec["slug"],
                   off=dict(champion_id=off_c["id"], champion_metric=disp(off_c["score"])),
                   arm=dict(nodes=len(tree["nodes"]), champion_id=new_c["id"],
                            champion_metric=disp(new_c["score"])),
                   ledger=ledger),
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"ledger -> {out_path}")


def run_arm_a(comp):
    spec, ev, off = load_comp(comp)
    disp = spec["display"]
    print(f"[OFF] {len(off['nodes'])} nodes, champion #{champion(off)['id']} "
          f"{spec['metric_key']}={disp(champion(off)['score']):.6f}")
    tree = deepcopy(off)
    priors = hv4.suggest_priors_v4(spec["comp_meta"], mode="ext")
    ctx = dict(spec["ctx"], blend_members=cached_solo_ids(tree, ev))
    base = {k: v for k, v in best_solo(tree)["config"].items() if k != "result"}
    wired = pw.wire_priors(priors, base, tree, ctx)
    print(f"[A] {pw.ledger_summary(wired['ledger'])}")
    results = []
    for cand in wired["candidates"]:
        nid, r, met = eval_and_attach(tree, ev, cand["config"], cand["mutation"],
                                      spec["metric_key"])
        results.append(dict(node_id=nid, tag=cand["tag"], rule=cand["rule"],
                            status=r["status"], metric=met, wall_s=round(r["wall_s"], 1)))
        print(f"[A] wired #{nid} {cand['tag']}/{cand['rule']}: {r['status']} "
              f"{spec['metric_key']}={met} wall_s={r['wall_s']:.0f}")
    hv2.save(tree, os.path.join(_HERE, f"wire_{comp}_A.json"))
    summarize(off, tree, spec, dict(disposition=wired["ledger"], results=results),
              os.path.join(_HERE, f"wire_{comp}_ledger_a.json"))


def make_proposer_input(comp):
    spec, ev, off = load_comp(comp)
    disp = spec["display"]
    a_path = os.path.join(_HERE, f"wire_{comp}_A.json")
    tree = json.load(open(a_path)) if os.path.exists(a_path) else off
    priors = hv4.suggest_priors_v4(spec["comp_meta"], mode="ext")
    nodes, wl = [], {}
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or not isinstance(n["score"], (int, float)):
            continue
        cfg = {k: v for k, v in n["config"].items() if k != "result"}
        nodes.append(dict(id=n["id"], metric=disp(n["score"]),
                          mutation=n["mutation"][:110], config=cfg))
        if cfg.get("kind") == "solo":
            wl.setdefault(cfg.get("model"), set()).update(cfg.get("params", {}))
    reverse = spec["metric_key"] != "rmse"   # rmse: lower is better
    nodes.sort(key=lambda d: d["metric"], reverse=reverse)
    pkg = dict(competition=spec["slug"] + f" (metric {spec['metric_key']}, "
                          f"{'maximize' if reverse else 'minimize'})",
               evaluator_contract=dict(
                   solo={"kind": "solo", "model": "lgb|xgb|cat",
                         "params": "ONLY keys in param_whitelist for that model",
                         "features": {"drop": "subset of names seen in tree configs; [] = keep all"}},
                   blend={"kind": "blend", "members": "existing evaluated solo node ids",
                          "weight_search": "dirichlet"},
                   notes="shared folds; blends use cached OOF (seconds); solo retrain minutes"),
               param_whitelist={m: sorted(k) for m, k in wl.items()},
               priors=[dict(tag=f"{p['provenance']}-{i:02d}", source=p.get("source"),
                            text=p["text"]) for i, p in enumerate(priors)],
               evaluated_nodes=nodes, champion=nodes[0])
    out = os.path.join(_HERE, f"llm_proposer_input_{comp}.json")
    json.dump(pkg, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"proposer input -> {out} ({len(nodes)} nodes, {len(priors)} priors, "
          f"whitelist={{m: len(k) for m, k in wl.items()}})")


def run_arm_b(comp):
    spec, ev, off = load_comp(comp)
    disp = spec["display"]
    print(f"[OFF] {len(off['nodes'])} nodes, champion #{champion(off)['id']} "
          f"{spec['metric_key']}={disp(champion(off)['score']):.6f}")
    frozen = json.load(open(os.path.join(_HERE, f"llm_proposals_{comp}.json")))
    tree = deepcopy(off)
    wl, names = {}, set()
    for n in tree["nodes"]:
        cfg = n["config"]
        if cfg.get("kind") == "solo":
            wl.setdefault(cfg.get("model"), set()).update(cfg.get("params", {}))
        names.update(cfg.get("features", {}).get("drop", []) or [])
    existing = {_recipe_hash(n["config"]) for n in tree["nodes"]}
    ledger, new_ids = [], []
    for i, prop in enumerate(frozen["proposals"]):
        tag, cfg = f"PRIOR-LLM-{i:02d}", prop["config"]
        ok, reason = True, ""
        if cfg.get("kind") != "solo":
            ok, reason = False, "must be solo"
        elif cfg.get("model") not in wl:
            ok, reason = False, f"model {cfg.get('model')!r} never exercised"
        else:
            bad = set(cfg.get("params", {})) - wl[cfg["model"]]
            if bad:
                ok, reason = False, f"params never exercised: {sorted(bad)}"
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
        nid, r, met = eval_and_attach(tree, ev, cfg, mut, spec["metric_key"])
        ledger.append(dict(tag=tag, status=r["status"], node_id=nid, metric=met,
                           wall_s=round(r["wall_s"], 1), prior_tags=prop["prior_tags"],
                           expected=prop.get("expected", "")))
        new_ids.append(nid)
        print(f"[B] wired #{nid} {tag}: {r['status']} {spec['metric_key']}={met} "
              f"wall_s={r['wall_s']:.0f}")
    if new_ids:
        pool = sorted(cached_solo_ids(tree, ev))
        cfg = {"kind": "blend", "members": pool, "weight_search": "dirichlet"}
        if _recipe_hash(cfg) not in existing:
            mut = f"[PRIOR-LLM-REBLEND] full-pool reblend incl LLM members {new_ids}"
            nid, r, met = eval_and_attach(tree, ev, cfg, mut, spec["metric_key"])
            ledger.append(dict(tag="PRIOR-LLM-REBLEND", status=r["status"], node_id=nid,
                               metric=met, wall_s=round(r["wall_s"], 1)))
            print(f"[B] reblend #{nid}: {spec['metric_key']}={met} wall_s={r['wall_s']:.0f}")
    hv2.save(tree, os.path.join(_HERE, f"wire_{comp}_B.json"))
    summarize(off, tree, spec, ledger, os.path.join(_HERE, f"wire_{comp}_ledger_b.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", choices=sorted(COMPS), required=True)
    ap.add_argument("--arm", choices=["A", "B"])
    ap.add_argument("--make-proposer-input", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.make_proposer_input:
        make_proposer_input(args.comp)
    elif args.arm == "A":
        run_arm_a(args.comp)
    elif args.arm == "B":
        run_arm_b(args.comp)
    else:
        ap.error("need --arm A|B or --make-proposer-input")
    print(f"total wall {time.time() - t0:.0f}s")
