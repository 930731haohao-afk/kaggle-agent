"""Comp- and arm-agnostic v5 search driver.

Generalizes run_s3e19_v5.py: takes --comp and --arm, imports that arm's generated
evaluator (or the competition's baseline evaluator for --arm baseline), runs the same
harness_v3 forward search used everywhere else, and writes a submission from the
champion node. Blend champions are finalized by blending the members' CACHED test
predictions with the weights the champion node RECORDED (no retraining, no re-search);
either way finalize re-derives the champion's score from the cached vectors and refuses
to write a submission that does not reproduce the CV the tree reports.

Usage:
  VIRTUAL_ENV= uv run python3 tree_search/run_arm_search.py \
      --comp playground-series-s3e19 --arm ratio --nodes 45
"""
import argparse
import hashlib
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
import seed_from_ledger as sfl  # noqa: E402
from make_v5_arm import BASE, _slug  # noqa: E402
from run_v3_generic import core, solo_pool, variant as mk_variant  # noqa: E402

# per-competition seed config source and param whitelist
SEED_TREE = {
    "playground-series-s3e19": "competitions/playground-series-s3e19/experiments_tree_v3.json",
}
WHITELIST = {
    "playground-series-s3e19": "llm_proposer_input_s3e19.json",
    "tabular-playground-series-sep-2022": "llm_proposer_input_s3e19.json",  # same model families
    "playground-series-s5e1": "llm_proposer_input_s3e19.json",              # same model families
}
DEFAULT_SEED_CFG = {"kind": "solo", "model": "lgb",
                    "params": {"num_leaves": 31, "learning_rate": 0.05,
                               "n_estimators": 400, "random_state": 42},
                    "features": {"drop": []}}
# finalize re-derives the champion's score from the cached vectors before writing a
# submission; these are the bounds it accepts. A solo score is recomputed from its own
# cached OOF and must match the 6-dp value the tree stores (measured |Δ|=2.9e-7 on the
# s3e19 ratio arm). A blend is recomputed from weights the evaluators record rounded to
# 4 dp, which costs about an order of magnitude more (measured |Δ|=3.3e-6, same arm), so
# that branch gets a looser -- but still far tighter than any real blend difference --
# bound (2026-08-03 audit).
SCORE_TOL_SOLO = 1e-6
SCORE_TOL_BLEND = 1e-3


def load_evaluator(comp: str, arm: str):
    if arm == "baseline":
        mod = BASE[comp][0]
    else:
        mod = f"eval_{arm}_{_slug(comp)}"
    ev = importlib.import_module(mod)
    if arm == "baseline":
        # The baseline arm runs the competition's OWN evaluator, whose CACHE_DIR is keyed by
        # node_id alone and is shared with that competition's pre-existing v1/v3 trees. Node
        # ids restart at 0 here, so without a private cache dir this arm reads another tree's
        # cached OOF vectors and scores a model it never trained. Redirect the cache instead
        # of trusting id disjointness (2026-08-03 audit).
        priv = os.path.join(os.path.dirname(os.path.abspath(ev.__file__)),
                            f"cache_v5baseline_{_slug(comp)}")
        os.makedirs(priv, exist_ok=True)
        ev.CACHE_DIR = priv
        print(f"[baseline] private OOF cache: {priv}")
    return ev


def workspace(comp: str, arm: str) -> str:
    """Where THIS arm WRITES: its tree, consumption report, submission and arm_result.

    The baseline arm used to return the competition's own directory, so a v5 baseline run
    overwrote the committed submission.csv and dropped v5 tree/result files into tracked
    space (competitions/playground-series-s5e1/arm_result_baseline.json is one such file,
    and the sep-2022/s5e1 lanes had to be snapshotted to `.v5arms-*` to survive it). Every
    arm now gets its own directory, same reasoning as load_evaluator's private OOF cache
    (2026-08-03 audit)."""
    return os.path.join(_ROOT, "competitions", f"{comp}-v5-{arm}")


def spec_source(comp: str, arm: str) -> str:
    """Where the arm's emitted injection specs are READ from (writes go to workspace()):
    a built arm carries its own manifest/ledger, while the baseline arm has none of its
    own and reads the competition's."""
    return os.path.join(_ROOT, "competitions", comp) if arm == "baseline" \
        else workspace(comp, arm)


def data_fingerprint(ev) -> str:
    """Identity of the feature table this arm's evaluator just loaded.

    make_v5_arm rebuilds an arm's processed CSVs in place, and neither the persisted tree
    nor the OOF/pred cache is keyed by the data, so a re-run after a rebuild resumed with
    scores and cached vectors computed on the PREVIOUS feature table (2026-08-03 audit).
    Same intent as the config-hash stamp harness_v2.cache_oof now writes, one level up:
    content hash of the processed CSVs, plus the loaded frames' shape/columns and
    FEATURE_COLS so the recompute-features fallback path (which writes no CSVs) is still
    covered."""
    h = hashlib.sha256()
    for name in ("train_processed.csv", "test_processed.csv"):
        p = os.path.join(getattr(ev, "DATA", ""), name)
        if not os.path.exists(p):
            continue
        h.update(name.encode())
        with open(p, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    for df in (ev._train, ev._test):
        h.update(repr((tuple(df.shape), list(df.columns))).encode())
    h.update(repr(list(getattr(ev, "FEATURE_COLS", []))).encode())
    return h.hexdigest()


def seed_config(comp: str) -> dict:
    p = SEED_TREE.get(comp)
    if p and os.path.exists(os.path.join(_ROOT, p)):
        t = json.load(open(os.path.join(_ROOT, p)))
        n0 = next(n for n in t["nodes"] if n["config"].get("kind") == "solo"
                  and isinstance(n.get("score"), (int, float)))
        return json.loads(json.dumps(n0["config"]))
    return json.loads(json.dumps(DEFAULT_SEED_CFG))


def champ(t):
    ev_n = [n for n in t["nodes"] if n["status"] == "evaluated"
            and isinstance(n["score"], (int, float))]
    return min(ev_n, key=lambda n: n["score"]) if ev_n else None


def add_node(tree, parent, label, cfg, r):
    """hv3.add_node + record the evaluator's OWN result dict on the node.

    finalize needs the blend weights that dict carries; with nothing recorded it had to
    RE-SEARCH the weights at submission time, so the submitted blend could differ from the
    one that won and nothing compared the two (2026-08-03 audit)."""
    nid, dup = hv3.add_node(tree, parent, label, cfg, r["score"], r["status"],
                            r.get("wall_s", 0.0))
    if nid is not None:
        node = next((n for n in tree["nodes"] if n["id"] == nid), None)
        if node is not None:
            node["result"] = r.get("result")
    return nid, dup


def search(comp: str, arm: str, ev, n_nodes: int, wl: dict):
    ws = workspace(comp, arm)
    os.makedirs(ws, exist_ok=True)
    tpath = os.path.join(ws, f"experiments_tree_v5_{arm}.json")
    fp = data_fingerprint(ev)
    if os.path.exists(tpath):
        tree = json.load(open(tpath))
        prev = tree.get("data_fingerprint")
        if prev is None:
            # tree written before the stamp existed: adopt the current fingerprint so the
            # NEXT rebuild is caught, and say out loud that THIS resume is unverified.
            tree["data_fingerprint"] = fp
            print(f"[{arm}] WARNING: tree predates the data fingerprint -- resuming "
                  f"unverified (stamping {fp[:12]} now)")
        elif prev != fp:
            raise RuntimeError(
                f"[{arm}] the arm's data changed since this tree was created "
                f"({prev[:12]} -> {fp[:12]}). Every persisted score and every cached "
                f"OOF/pred vector was computed on the PREVIOUS feature table, so resuming "
                f"would silently mix two datasets. Delete {tpath} and {ev.CACHE_DIR} to "
                f"re-run this arm from scratch (2026-08-03 audit).")
    else:
        tree = {"comp": f"{comp}:{arm}", "nodes": [], "root_id": 0, "search_state": {},
                "data_fingerprint": fp}
        cfg = core(seed_config(comp))
        t0 = time.time()
        r = ev.evaluate(cfg, node_id=0, timeout_s=600)
        hv2.add_root(tree, f"[V5:{arm}] seed config", cfg, r["score"], r["status"],
                     time.time() - t0)
        tree["nodes"][0]["result"] = r.get("result")   # see add_node (2026-08-03 audit)
        json.dump(tree, open(tpath, "w"), indent=2)
        print(f"[{arm}] root: {r['status']} score={r['score']}")
        if r["status"] != "evaluated" or r["score"] is None:
            # evaluate() never raises, so a failed root used to persist and then crash the
            # budget machine with TypeError on None. Fail loudly at the real cause instead.
            raise RuntimeError(
                f"[{arm}] root evaluation failed ({r['status']}): {r.get('error')!r}. "
                f"Nothing downstream is meaningful; fix the evaluator or the seed config.")

        # Seed the ledger's config-only operators as real nodes. Before 07-30 these were
        # written to plan["node_configs"] and read by nobody, so the judgment layer's
        # model-level decisions changed nothing while the ledger reported them realized.
        specs = sfl.load_emitted(spec_source(comp, arm))
        if specs:
            seed_nodes, unconsumable = sfl.materialize(
                specs, cfg, wl, capabilities=getattr(ev, "SUPPORTS", {}))
            seeded = []
            for sn in seed_nodes:
                nid = hv3.next_id(tree)
                label = f"[V5:{arm}] injected {sn.get('seed_role')} " \
                        f"({sn.get('metric_family') or sn.get('archetype')})"
                try:
                    r = ev.evaluate(core(sn), node_id=nid, timeout_s=600)
                except Exception as e:  # noqa: BLE001
                    r = {"score": None, "status": "failed", "wall_s": 0.0, "error": str(e)}
                add_node(tree, tree["root_id"], label, core(sn), r)
                seeded.append({"node_id": nid, "seed_role": sn.get("seed_role"),
                               "model": sn.get("model"), "status": r["status"],
                               "score": r["score"]})
                print(f"[{arm}] injected node {nid} ({sn.get('seed_role')}): "
                      f"{r['status']} score={r['score']}")
            json.dump(tree, open(tpath, "w"), indent=2)
            p = sfl.write_consumption_report(ws, seeded, unconsumable)
            print(f"[{arm}] consumption report: {p} "
                  f"({len(seeded)} seeded, {len(unconsumable)} unconsumable)")

    c0 = champ(tree)["score"]
    n_start = len([n for n in tree["nodes"] if n["status"] == "evaluated"])
    tree.setdefault("search_state", {})["budget"] = {
        "total_budget": n_start + n_nodes, "explore_burst_size": 6,
        "post_burst_patience": 10, "phase": "exploit", "burst_start_eval": 0,
        "best_at_burst_start": c0, "evals_since_burst_improve": 0}
    fams = [m for m in ("xgb", "cat", "lgb") if m in wl]
    if not fams:
        raise RuntimeError(
            f"[{arm}] param whitelist has no supported model family "
            f"(saw {sorted(wl)}); expected at least one of xgb/cat/lgb")
    for i in range(n_nodes):
        pool = solo_pool(tree)
        ids = [n["id"] for n in pool]
        if i % 2 == 1 and len(ids) >= 2:
            cfg = {"kind": "blend", "members": sorted(ids), "weight_search": "dirichlet"}
            mut, parent = f"[V5:{arm}] reblend over {len(ids)}", pool[0]["id"]
        else:
            fam = fams[(i // 2) % len(fams)]
            base = next((n["config"] for n in pool if n["config"].get("model") == fam),
                        pool[0]["config"])
            cfg = mk_variant(base, wl, i)
            mut, parent = f"[V5:{arm}] shallow-reg {cfg['model']}", tree["root_id"]
        nid = hv3.next_id(tree)
        try:
            r = ev.evaluate(core(cfg), node_id=nid, timeout_s=600)
        except Exception as e:  # noqa: BLE001
            r = {"score": None, "status": "failed", "wall_s": 0.0, "error": str(e)}
        add_node(tree, parent, mut, core(cfg), r)
        json.dump(tree, open(tpath, "w"), indent=2)
        print(f"[{arm}] node {nid} {r['status']}; champion={champ(tree)['score']:.6f}")
        if hv3.should_stop(tree):
            print(f"[{arm}] budget machine stop")
            break
    return tree, tpath


def apply_postprocess(pred, pp: dict, scale: float):
    """Reproduce the evaluator's scoring-path post-processing (scale -> clip -> round, the
    order maybe_postprocess uses) on a TEST prediction vector.

    maybe_postprocess only ever runs on the OOF vector, so the submission has to redo it
    here -- and doing it on the blend branch alone meant a solo champion of a
    scale/clip/round competition was submitted un-post-processed while the tree reported the
    post-processed CV (2026-08-03 audit)."""
    out = np.asarray(pred, dtype=float) * float(scale)
    if pp.get("clip_min") is not None or pp.get("clip_max") is not None:
        out = np.clip(out, pp.get("clip_min"), pp.get("clip_max"))
    if pp.get("round_to_int"):
        out = np.round(out)
    return out


def write_submission(comp: str, arm: str, ev, pred, out_dir: str) -> str:
    """Join `pred` onto sample_submission BY ID and write it.

    The cached test predictions are in the evaluator's test-frame row order, which matches
    sample_submission's order only by accident; the previous positional
    `sub[sub.columns[-1]] = pred` silently produced a scrambled submission whenever the two
    differed, never checked the values were finite, and quietly wrote into the last column
    of a multi-target sample_submission (2026-08-03 audit)."""
    import pandas as pd
    id_col = getattr(ev, "ID", None) or BASE[comp][1]
    sub = pd.read_csv(os.path.join(_ROOT, "competitions", comp, "data",
                                   "sample_submission.csv"))
    tgt = [c for c in sub.columns if c != id_col]
    if id_col not in sub.columns or len(tgt) != 1:
        raise RuntimeError(
            f"[{arm}] sample_submission has columns {list(sub.columns)}; this driver writes "
            f"exactly ONE target column joined on {id_col!r} and cannot infer the layout of "
            f"a multi-target submission -- write a comp-specific finalizer instead of "
            f"letting it guess (2026-08-03 audit)")
    ids = np.asarray(ev._test[id_col]).tolist()
    pred = np.asarray(pred, dtype=float)
    if len(ids) != len(pred):
        raise RuntimeError(f"[{arm}] {len(pred)} predictions for {len(ids)} test rows -- the "
                           f"cached vector belongs to a different dataset")
    bad = int((~np.isfinite(pred)).sum())
    if bad:
        raise RuntimeError(f"[{arm}] {bad} non-finite prediction(s) (NaN/inf) -- refusing to "
                           f"write a submission")
    sub_ids = sub[id_col].tolist()
    if len(set(ids)) != len(ids) or len(set(sub_ids)) != len(sub_ids):
        raise RuntimeError(f"[{arm}] duplicate {id_col!r} values in the test frame or in "
                           f"sample_submission -- an id join is not well defined")
    if set(ids) != set(sub_ids):
        miss, extra = set(sub_ids) - set(ids), set(ids) - set(sub_ids)
        raise RuntimeError(
            f"[{arm}] id sets differ: {len(miss)} sample_submission id(s) have no "
            f"prediction, {len(extra)} predicted id(s) are not in sample_submission "
            f"(e.g. missing={sorted(miss)[:3]}, extra={sorted(extra)[:3]})")
    merged = sub[[id_col]].merge(pd.DataFrame({id_col: ids, tgt[0]: pred}),
                                 on=id_col, how="left")
    assert merged[tgt[0]].notna().all(), "id join left NaN despite matching id sets"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "submission.csv")
    merged.to_csv(out, index=False)
    return out


def finalize(comp: str, arm: str, ev, tree) -> str:
    c = champ(tree)
    cfg = c["config"]
    cache = ev.CACHE_DIR

    def load(nid):
        d = np.load(os.path.join(cache, f"solo_{nid}.npz"), allow_pickle=True)
        return np.asarray(d["oof"], dtype=float), np.asarray(d["pred"], dtype=float)

    if cfg["kind"] == "solo":
        oof, pred = load(c["id"])
        tol = SCORE_TOL_SOLO
    else:
        w = (c.get("result") or {}).get("weights")
        if w is None:
            # Legacy node (search() records the evaluator's result since 2026-08-03): the
            # weights have to be re-searched, and a re-search is only trustworthy if it
            # lands back on the score the tree reports -- the check below is what makes
            # that a claim rather than a hope.
            r = ev.evaluate(cfg, node_id=c["id"], timeout_s=1800)
            res = r.get("result") or {}
            if r.get("status") != "evaluated" or "weights" not in res:
                raise RuntimeError(
                    f"[{arm}] champion #{c['id']} stores no blend weights and re-searching "
                    f"them failed ({r.get('status')}: {r.get('error')!r})")
            w = res["weights"]
            print(f"[{arm}] blend weights re-searched (tree recorded none)")
        w = np.asarray(w, dtype=float)
        w = w / w.sum()
        oofs, preds = zip(*[load(m) for m in cfg["members"]])
        oof = (w[:, None] * np.stack(oofs)).sum(axis=0)
        pred = (w[:, None] * np.stack(preds)).sum(axis=0)
        tol = SCORE_TOL_BLEND
        print(f"[{arm}] blend champion members={cfg['members']} "
              f"weights={[round(float(x), 4) for x in w]}")

    # Re-derive the champion's score on the SAME vectors the evaluator scored. This does
    # two jobs: it recovers the post-processing the submission needs (previously computed
    # on the blend branch only), and it proves the cached vectors -- and, for a blend, the
    # weights being submitted -- are the ones that produced the recorded score.
    pp = cfg.get("postprocess") or {}
    mp = getattr(ev, "maybe_postprocess", None)
    if mp is None:
        raise RuntimeError(
            f"[{arm}] evaluator {ev.__name__} has no maybe_postprocess(), so finalize "
            f"cannot reproduce the scoring path or verify the champion's score "
            f"(2026-08-03 audit)")
    score, scale = mp(oof, pp)
    if abs(score - c["score"]) > tol:
        raise RuntimeError(
            f"[{arm}] champion #{c['id']} re-derives to {score:.6f} but the tree records "
            f"{c['score']:.6f} (|Δ|={abs(score - c['score']):.2e} > {tol:g}) -- the cached "
            f"vectors (or the blend weights) are not the ones that produced the recorded "
            f"score; refusing to submit predictions that do not match the reported CV "
            f"(2026-08-03 audit)")
    pred = apply_postprocess(pred, pp, scale)
    out = write_submission(comp, arm, ev, pred, workspace(comp, arm))
    print(f"[{arm}] submission: {out} (champion #{c['id']} score={c['score']:.6f}, "
          f"re-derived {score:.6f}, scale={scale})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--nodes", type=int, default=45)
    a = ap.parse_args()
    wl = json.load(open(os.path.join(_HERE, WHITELIST[a.comp])))["param_whitelist"]
    ev = load_evaluator(a.comp, a.arm)
    tree, tpath = search(a.comp, a.arm, ev, a.nodes, wl)
    out = finalize(a.comp, a.arm, ev, tree)
    summary = {"comp": a.comp, "arm": a.arm, "champion_cv": champ(tree)["score"],
               "nodes_evaluated": len([n for n in tree["nodes"] if n["status"] == "evaluated"]),
               "tree": tpath, "submission": out}
    json.dump(summary, open(os.path.join(workspace(a.comp, a.arm), f"arm_result_{a.arm}.json"), "w"),
              indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
