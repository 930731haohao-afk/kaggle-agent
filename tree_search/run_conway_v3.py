"""tree_search/run_conway_v3.py — harness_v3 tree-search driver for
conway-s-reverse-game-of-life (per-cell binary MAE, minimize).

Fold-0 proxy evaluation via eval_conway.py (see its docstring for the node space:
solo cnn / solo lgb / reuse / blend / refine). Root = cached-OOF reuse of the
linear-protocol champion cnn_v1 (digit-verified). Budget deviates from the 60-node
default: TOTAL=40, burst 5, patience 12 — CNN solo evals cost ~3 min each (search
grade), so 60 nodes would blow the session budget; documented in STATUS.md.
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
os.chdir(_REPO_ROOT)

import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402

COMP = "conway-s-reverse-game-of-life"
COMP_DIR = os.path.join(_REPO_ROOT, "competitions", COMP)
TREE_PATH = os.path.join(COMP_DIR, "experiments_tree_v3.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_conway.py")
CACHE_DIR = os.path.join(_HERE, "cache_conway")
TOTAL_BUDGET, BURST_SIZE, PATIENCE = 40, 5, 12
SOLO_TIMEOUT_S = 1500

_ev = None
def ev():
    global _ev
    if _ev is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("eval_conway", EVAL_MODULE_PATH)
        _ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ev)
    return _ev


# search-grade CNN base
CNN_BASE = dict(channels=64, depth=8, residual=True, epochs=10, batch=512, lr=2e-3,
                wd=1e-4, synth_n=50000, d4_aug=True, tta=True, seed=42, per_delta=False)

SEARCH_SPACE = dict(
    channels=(32, 128), depth=(4, 14), lr=(5e-4, 5e-3), wd=(1e-5, 1e-3),
    epochs=(6, 16), batch=(256, 1024), synth_n=(20000, 120000),
    window=(5, 9), n_estimators=(50, 300), num_leaves=(31, 255),
    lam=(0.05, 1.0), n_steps=(400, 3000), focus_k=(40, 400),
)


def core(cfg):
    c = json.loads(json.dumps(cfg))
    if c.get("kind") == "blend":
        c["members"] = sorted(c["members"])
    return c


def _ds(tree):
    return tree["search_state"].setdefault("driver_state", {})


def eval_and_add(tree, parent_id, mutation, cfg, lineage_note=""):
    nid_guess = len(tree["nodes"])
    kind = cfg.get("kind", "solo")
    dup = hv2.find_duplicate_config(tree, core(cfg))
    if dup is not None:
        print(f"  [dedup] proposal identical to node {dup}, skipping", flush=True)
        _ds(tree)["dedup_rejections"] = _ds(tree).get("dedup_rejections", 0) + 1
        return None
    if kind == "solo" and cfg.get("model") != "reuse":
        r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, cfg, SOLO_TIMEOUT_S, node_id=nid_guess)
    else:
        r = ev().evaluate(cfg, node_id=nid_guess)
    status = "evaluated" if r["status"] == "ok" else "failed"
    score = r["score"] if status == "evaluated" else None
    nid, dup2 = hv3.add_node(tree, parent_id, mutation, core(cfg), score, status,
                             r.get("wall_s", 0.0), kind=kind)
    if nid is None:
        print(f"  [dedup-late] duplicate of {dup2}", flush=True)
        return None
    assert nid == nid_guess, f"node id drift {nid} != {nid_guess}"
    _ds(tree).setdefault("results", {})[str(nid)] = r.get("result")
    best = hv3.global_best(tree)
    print(f"node {nid} [{mutation}] {lineage_note} -> score={score} status={status} "
          f"wall={r.get('wall_s')}s  (best={best['score'] if best else None} @node{best['id'] if best else '-'})",
          flush=True)
    if status == "failed":
        print(f"  error: {str(r.get('error'))[:300]}", flush=True)
    hv3.save_search_state(tree, TREE_PATH)
    return nid


# ---------------------------------------------------------------------------
# lineage seeds
# ---------------------------------------------------------------------------
def seed_all(tree):
    ds = _ds(tree)
    ids = ds.setdefault("seed_ids", {})
    root_id = tree["root_id"]

    if "lgbreuse" not in ids:
        ids["lgbreuse"] = eval_and_add(tree, root_id, "seed:LGBREUSE",
            dict(kind="solo", model="reuse", params=dict(name="lgb_v1")), "LGBREUSE")
    if "arch" not in ids:
        ids["arch"] = eval_and_add(tree, root_id, "seed:ARCH searchgrade base",
            dict(kind="solo", model="cnn", params=dict(CNN_BASE)), "ARCH")
    if "perdelta" not in ids:
        ids["perdelta"] = eval_and_add(tree, root_id, "seed:PERDELTA",
            dict(kind="solo", model="cnn", params={**CNN_BASE, "per_delta": True}), "PERDELTA")
    # REFINE lineage dropped: pre-search probe showed forward-consistency hill-climb
    # improves forward-mismatch (0.099->0.068) but WORSENS MAE at every delta
    # (0.1127->0.1165 on 3k fold-0 boards) — per-cell marginals beat any single
    # consistent preimage for MAE. Negative result logged in experiments.json.
    if "seedbag" not in ids:
        ids["seedbag"] = eval_and_add(tree, root_id, "seed:SEEDBAG cnn seed777",
            dict(kind="solo", model="cnn", params={**CNN_BASE, "seed": 777}), "SEEDBAG")
    if "blend" not in ids and ids.get("lgbreuse") is not None:
        ids["blend"] = eval_and_add(tree, root_id, "seed:BLEND root+lgb",
            dict(kind="blend", members=[root_id, ids["lgbreuse"]],
                 per_delta_weights=False, thresh_sweep=False), "BLEND")
    hv3.save_search_state(tree, TREE_PATH)


# ---------------------------------------------------------------------------
# mutation proposals
# ---------------------------------------------------------------------------
def solo_members(tree):
    """Evaluated solo+refine node ids, best-first."""
    ns = [n for n in tree["nodes"] if n["status"] == "evaluated"
          and n["kind"] in ("solo", "refine")]
    return [n["id"] for n in sorted(ns, key=lambda n: n["score"])]


def refine_sources(tree):
    """Candidate sources: best blend or solo nodes not yet refined."""
    done = {n["config"].get("params", {}).get("source") for n in tree["nodes"]
            if n["kind"] == "refine"}
    ns = [n for n in tree["nodes"] if n["status"] == "evaluated"
          and n["kind"] in ("solo", "blend") and n["id"] not in done]
    return [n["id"] for n in sorted(ns, key=lambda n: n["score"])]


def propose(tree, parent, lineage_id):
    """Return (mutation, cfg) or None. parent is a node dict."""
    ds = _ds(tree)
    pk = parent["kind"]
    pcfg = parent["config"]
    if pk == "solo" and pcfg.get("model") == "reuse" and pcfg["params"]["name"] == "lgb_v1":
        q = [("lgb window9", dict(kind="solo", model="lgb",
                                  params=dict(window=9, n_estimators=100))),
             ("lgb window7 200est leaves127", dict(kind="solo", model="lgb",
                                  params=dict(window=7, n_estimators=200, num_leaves=127)))]
    elif pk == "solo" and pcfg.get("model") == "lgb":
        p = dict(pcfg["params"])
        q = [("lgb more trees", dict(kind="solo", model="lgb",
                                     params={**p, "n_estimators": int(p.get("n_estimators", 100) * 2)})),
             ("lgb leaves255", dict(kind="solo", model="lgb",
                                    params={**p, "num_leaves": 255}))]
    elif pk == "solo" and pcfg.get("model") == "cnn":
        p = dict(pcfg["params"])
        cand = [
            ("cnn ch96", {**p, "channels": 96}),
            ("cnn depth12", {**p, "depth": 12}),
            ("cnn ch96 depth12", {**p, "channels": 96, "depth": 12}),
            ("cnn lr3e-3", {**p, "lr": 3e-3}),
            ("cnn lr1e-3", {**p, "lr": 1e-3}),
            ("cnn wd3e-4", {**p, "wd": 3e-4}),
            ("cnn synth100k", {**p, "synth_n": 100000}),
            ("cnn epochs14", {**p, "epochs": 14}),
            ("cnn batch256", {**p, "batch": 256}),
            ("cnn depth10 ch80", {**p, "depth": 10, "channels": 80}),
        ]
        # boundary push on params sitting at declared-space edges
        for bp in hv3.boundary_candidates({"params": p}, SEARCH_SPACE):
            cand.append((f"boundary-push {bp['param']}={bp['new_value']}",
                         {**p, bp["param"]: bp["new_value"]}))
        q = [(m, dict(kind="solo", model="cnn", params=c)) for m, c in cand]
    elif pk == "refine":
        p = dict(pcfg["params"])
        cand = [
            ("refine lam0.15", {**p, "lam": 0.15}),
            ("refine lam0.6", {**p, "lam": 0.6}),
            ("refine steps2400", {**p, "n_steps": 2400}),
            ("refine focus_k240", {**p, "focus_k": 240}),
            ("refine steps3000 lam0.2", {**p, "n_steps": 3000, "lam": 0.2}),
        ]
        for src in refine_sources(tree)[:1]:
            cand.append((f"refine new source node{src}", {**p, "source": src}))
        q = [(m, dict(kind="refine", params=c)) for m, c in cand]
    elif pk == "blend":
        members = list(pcfg["members"])
        pool = solo_members(tree)
        cand = []
        if not pcfg.get("per_delta_weights"):
            cand.append(("blend per-delta weights",
                         dict(kind="blend", members=members, per_delta_weights=True,
                              thresh_sweep=pcfg.get("thresh_sweep", False))))
        if not pcfg.get("thresh_sweep"):
            cand.append(("blend thresh sweep",
                         dict(kind="blend", members=members,
                              per_delta_weights=pcfg.get("per_delta_weights", False),
                              thresh_sweep=True)))
        for m in pool:
            if m not in members:
                cand.append((f"blend add node{m}",
                             dict(kind="blend", members=members + [m],
                                  per_delta_weights=pcfg.get("per_delta_weights", False),
                                  thresh_sweep=pcfg.get("thresh_sweep", False))))
                break
        q = cand
    else:
        return None
    tried = ds.setdefault("tried_mutations", {})
    key = str(parent["id"])
    used = set(tried.get(key, []))
    for m, cfg in q:
        if m not in used:
            tried.setdefault(key, []).append(m)
            return m, cfg
    return None


def inject_burst(tree):
    ds = _ds(tree)
    root_id = tree["root_id"]
    pool = solo_members(tree)
    print(f"== explore burst: pool={pool} ==", flush=True)
    # 1. kitchen-sink blend of every solo/refine member, per-delta weights + thresh sweep
    nid = eval_and_add(tree, root_id, "burst:kitchen-sink blend",
        dict(kind="blend", members=pool, per_delta_weights=True, thresh_sweep=True),
        "BURST")
    if nid is not None:
        hv3.apply_burst_seed_sanity_gate(tree, nid)
    # 2. second kitchen-sink variant: per-delta weights WITHOUT thresh sweep
    nid = eval_and_add(tree, root_id, "burst:kitchen-sink blend no-thresh",
        dict(kind="blend", members=pool, per_delta_weights=True, thresh_sweep=False),
        "BURST")
    if nid is not None:
        hv3.apply_burst_seed_sanity_gate(tree, nid)
    # 3. long-shot: combine the two winning exploit levers (lr3e-3 + ch96)
    nid = eval_and_add(tree, root_id, "burst:longshot cnn ch96 lr3e-3",
        dict(kind="solo", model="cnn",
             params={**CNN_BASE, "channels": 96, "lr": 3e-3}), "BURST")
    if nid is not None:
        hv3.apply_burst_seed_sanity_gate(tree, nid)
    # 4. long-shot scale push at the winning lr
    nid = eval_and_add(tree, root_id, "burst:longshot lr3e-3 synth100k epochs14",
        dict(kind="solo", model="cnn",
             params={**CNN_BASE, "lr": 3e-3, "synth_n": 100000, "epochs": 14}), "BURST")
    if nid is not None:
        hv3.apply_burst_seed_sanity_gate(tree, nid)
    # 5. final kitchen-sink re-blend absorbing the burst solos
    pool2 = solo_members(tree)
    if pool2 != pool:
        nid = eval_and_add(tree, root_id, "burst:kitchen-sink blend v2",
            dict(kind="blend", members=pool2, per_delta_weights=True, thresh_sweep=True),
            "BURST")
    ds["burst_injected"] = True
    hv3.save_search_state(tree, TREE_PATH)


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        print(f"resumed tree with {len(tree['nodes'])} nodes", flush=True)
    else:
        tree = hv3.new_tree(COMP)
        hv3.init_budget(tree, total_budget=TOTAL_BUDGET,
                        explore_burst_size=BURST_SIZE, post_burst_patience=PATIENCE)
        # root: cached-OOF reuse of linear champion cnn_v1, digit-verified
        r = ev().evaluate(dict(kind="solo", model="reuse", params=dict(name="cnn_v1")),
                          node_id=0)
        assert r["status"] == "ok", f"root eval failed: {r['error']}"
        with open(f"{COMP_DIR}/scripts/cache/score_cnn_v1.json") as fh:
            stored = round(float(json.load(fh)["fold_scores"][0]), 6)
        assert round(r["score"], 6) == stored, f"root drift {r['score']} != {stored}"
        hv3.add_root(tree, "root: cnn_v1 fold-0 cached reuse (digit-verified)",
                     core(dict(kind="solo", model="reuse", params=dict(name="cnn_v1"))),
                     r["score"], "evaluated", r["wall_s"], kind="solo")
        print(f"root score {r['score']:.6f} (verified == {stored})", flush=True)
        hv3.save_search_state(tree, TREE_PATH)

    seed_all(tree)

    while not hv3.should_stop(tree):
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not _ds(tree).get("burst_injected"):
            inject_burst(tree)
            continue
        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            print("no parent available — stopping", flush=True)
            break
        parent = next(n for n in tree["nodes"] if n["id"] == parent_id)
        prop = propose(tree, parent, lineage_id)
        if prop is None:
            # exhausted queue for this parent: burn a failed placeholder so the
            # lineage moves on (dedup-consumes-budget analogue)
            ds = _ds(tree)
            ds["noop_counter"] = ds.get("noop_counter", 0) + 1
            hv3.add_node(tree, parent_id, "no-proposal",
                         {"kind": "noop", "parent": parent_id, "n": ds["noop_counter"]},
                         None, "failed", 0.0, kind=parent["kind"])
            hv3.save_search_state(tree, TREE_PATH)
            continue
        mutation, cfg = prop
        eval_and_add(tree, parent_id, mutation, cfg, f"lineage{lineage_id}")

    best = hv3.global_best(tree)
    n_ev = hv3.n_evaluated(tree)
    print(f"\n== DONE: {n_ev} evaluated nodes, best score {best['score']:.6f} @node{best['id']} "
          f"({(time.time() - t_start) / 60:.1f} min) ==", flush=True)
    print("best config:", json.dumps(best["config"])[:500], flush=True)
    print("backtrack_log:", json.dumps(tree["search_state"]["backtrack_log"], indent=1)[:2000],
          flush=True)
    print("plateaued:", tree["search_state"]["plateaued"], flush=True)
    print("dedup_rejections:", _ds(tree).get("dedup_rejections", 0), flush=True)


if __name__ == "__main__":
    main()
