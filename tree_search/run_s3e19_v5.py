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
ID_COL = "id"
VARIANTS = ["gdp", "gdp_hol"]
BASE_TREE = os.path.join(_ROOT, "competitions", COMP, "experiments_tree_v3.json")
WL = json.load(open(os.path.join(_HERE, "llm_proposer_input_s3e19.json")))["param_whitelist"]
# finalize's score-agreement bounds -- same reasoning and same measured deltas as
# run_arm_search.py's (2026-08-03 audit).
SCORE_TOL_SOLO = 1e-6
SCORE_TOL_BLEND = 1e-3


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


def add_node(tree, parent, label, cfg, r):
    """hv3.add_node + record the evaluator's OWN result dict on the node, so finalize can
    submit the blend weights the champion actually won with instead of re-searching them
    (2026-08-03 audit; twin of run_arm_search.add_node)."""
    nid, dup = hv3.add_node(tree, parent, label, cfg, r["score"], r["status"],
                            r.get("wall_s", 0.0))
    if nid is not None:
        node = next((n for n in tree["nodes"] if n["id"] == nid), None)
        if node is not None:
            node["result"] = r.get("result")
    return nid, dup


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
    tree["nodes"][0]["result"] = r.get("result")   # see add_node (2026-08-03 audit)
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
        add_node(tree, parent, mut, core(cfg), r)
        save_tree(tree, p)
        print(f"[{v}] node {nid} {r['status']}; champion={champ(tree)['score']:.6f}")
        if hv3.should_stop(tree):
            print(f"[{v}] budget machine stop")
            break
    return tree


def finalize(v: str, ev, tree) -> str:
    """Write submission.csv from the champion node's cached test predictions.

    Twin of run_arm_search.finalize/write_submission; keep the two in step. The three
    corrections there apply verbatim here (2026-08-03 audit): the submission is joined on
    the id column instead of assigned positionally (the cached predictions are in the
    evaluator's test-frame order, which matches sample_submission only by accident), the
    blend weights come from the champion node instead of a fresh re-search, and the
    evaluator's scoring-path post-processing is applied on BOTH branches, not just blends.
    """
    import pandas as pd
    c = champ(tree)
    cfg = c["config"]

    def load(nid):
        # load_oof returns only the OOF vector; the test predictions live in the same npz
        # under "pred", so read the cache files directly.
        d = np.load(os.path.join(ev.CACHE_DIR, f"solo_{nid}.npz"), allow_pickle=True)
        return np.asarray(d["oof"], dtype=float), np.asarray(d["pred"], dtype=float)

    if cfg["kind"] == "solo":
        oof, pred = load(c["id"])
        tol = SCORE_TOL_SOLO
    else:
        w = (c.get("result") or {}).get("weights")
        if w is None:
            # Legacy node (forward_search records the result since 2026-08-03): re-search
            # the weights, and treat the score check below as the thing that decides
            # whether the re-searched blend really is the champion's blend.
            r = ev.evaluate(cfg, node_id=c["id"], timeout_s=900)
            res = r.get("result") or {}
            if r.get("status") != "evaluated" or "weights" not in res:
                raise RuntimeError(
                    f"[{v}] champion #{c['id']} stores no blend weights and re-searching "
                    f"them failed ({r.get('status')}: {r.get('error')!r})")
            w = res["weights"]
            print(f"[{v}] blend weights re-searched (tree recorded none)")
        w = np.asarray(w, dtype=float)
        w = w / w.sum()
        oofs, preds = zip(*[load(m) for m in cfg["members"]])
        oof = (w[:, None] * np.stack(oofs)).sum(axis=0)
        pred = (w[:, None] * np.stack(preds)).sum(axis=0)
        tol = SCORE_TOL_BLEND
        print(f"[{v}] blend champion members={cfg['members']} "
              f"weights={[round(float(x), 4) for x in w]}")

    # Re-derive the champion's score from the same vectors the evaluator scored: recovers
    # the post-processing the submission needs AND proves the cached vectors (and the blend
    # weights) are the ones that produced the recorded score.
    pp = cfg.get("postprocess") or {}
    score, scale = ev.maybe_postprocess(oof, pp)
    if abs(score - c["score"]) > tol:
        raise RuntimeError(
            f"[{v}] champion #{c['id']} re-derives to {score:.6f} but the tree records "
            f"{c['score']:.6f} (|Δ|={abs(score - c['score']):.2e} > {tol:g}) -- refusing to "
            f"submit predictions that do not match the reported CV (2026-08-03 audit)")
    pred = np.asarray(pred, dtype=float) * float(scale)
    if pp.get("clip_min") is not None or pp.get("clip_max") is not None:
        pred = np.clip(pred, pp.get("clip_min"), pp.get("clip_max"))
    if pp.get("round_to_int"):
        pred = np.round(pred)

    sub = pd.read_csv(os.path.join(_ROOT, "competitions", COMP, "data", "sample_submission.csv"))
    tgt = [col for col in sub.columns if col != ID_COL]
    if ID_COL not in sub.columns or len(tgt) != 1:
        raise RuntimeError(
            f"[{v}] sample_submission has columns {list(sub.columns)}; this finalizer writes "
            f"exactly ONE target column joined on {ID_COL!r} and will not guess the layout "
            f"of a multi-target submission")
    ids = np.asarray(ev._test[ID_COL]).tolist()
    if len(ids) != len(pred):
        raise RuntimeError(f"[{v}] {len(pred)} predictions for {len(ids)} test rows")
    bad = int((~np.isfinite(pred)).sum())
    if bad:
        raise RuntimeError(f"[{v}] {bad} non-finite prediction(s) (NaN/inf)")
    sub_ids = sub[ID_COL].tolist()
    if len(set(ids)) != len(ids) or len(set(sub_ids)) != len(sub_ids):
        raise RuntimeError(f"[{v}] duplicate {ID_COL!r} values -- an id join is not defined")
    if set(ids) != set(sub_ids):
        raise RuntimeError(
            f"[{v}] id sets differ: {len(set(sub_ids) - set(ids))} sample_submission id(s) "
            f"have no prediction, {len(set(ids) - set(sub_ids))} predicted id(s) are unknown")
    merged = sub[[ID_COL]].merge(pd.DataFrame({ID_COL: ids, tgt[0]: pred}),
                                 on=ID_COL, how="left")
    assert merged[tgt[0]].notna().all(), "id join left NaN despite matching id sets"
    out = os.path.join(_ROOT, "competitions", f"{COMP}-v5-{v}", "submission.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"[{v}] submission: {out} (champion #{c['id']} score={c['score']:.6f}, "
          f"re-derived {score:.6f}, scale={scale})")
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
