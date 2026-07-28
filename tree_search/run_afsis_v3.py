"""tree_search/run_afsis_v3.py — harness_v3 tree-search driver for afsis-soil-properties
(5-target soil property regression, MCRMSE, minimize-better).

Stage 4 of the kaggle-agent skill (references/07_tree_search.md). The linear iteration
protocol has already run (competitions/afsis-soil-properties/scripts/baseline.py: a
16-config solo pool + one full-pool per-target blend), so the tree-search entry
condition — "a baseline solo AND at least one blend exist" — is satisfied.

Driver contract (07_tree_search.md §4), all five items wired:
 1. Root is digit-verified: the root config is byte-identical to the linear stage's best
    solo (SVR-RBF on SG-d1+SNV spectra + spatial, node #9011), its OOF is reloaded from
    the shared cache, MCRMSE recomputed from that OOF, and asserted equal to the linear
    stage's stored 0.471939 to 6 decimals before anything else runs.
 2. OOF cache reuse: every linear-pool config that re-enters the tree as a first-
    generation seed is reused the same way (recompute-from-cached-OOF, accept only on a
    6-dp match) instead of retraining.
 3. Resume state lives in tree["search_state"]["driver_state"] only — no module globals.
 4. Solo evals go through hv3.eval_solo_subprocess (OS-level timeout).
 5. Every explore-burst seed goes through hv3.apply_burst_seed_sanity_gate.

MCRMSE is already lower-is-better, so scores are stored unflipped (unlike the AUC comps).
"""
import copy
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import harness as hv1     # noqa: E402
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402
import eval_afsis as ev   # noqa: E402

COMP = "afsis-soil-properties"
COMP_DIR = os.path.join(_REPO, "competitions", COMP)
TREE_PATH = os.path.join(COMP_DIR, "experiments_tree_v3.json")
LINEAR_STATE = os.path.join(COMP_DIR, "scripts", "state", "linear_pool.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_afsis.py")
EVAL_TIMEOUT_S = 900
MAX_WALL_S = 3.0 * 3600
ITER_SAFETY_CAP = 400

ROOT_NAME = "svr_rbf_sg1"
LINEAR_BEST_SOLO = 0.471939      # linear stage's best solo, asserted on the root
LINEAR_BEST_BLEND = 0.433219     # linear stage's blend — the score tree search must beat

SEARCH_SPACE = {                 # declared ranges for hv3.boundary_candidates
    "gamma_scale": {"low": 0.1, "high": 4.0, "log": True},
    "C": {"low": 0.5, "high": 100.0, "log": True},
    "epsilon": {"low": 0.01, "high": 0.5, "log": True},
    "n_components": {"low": 5, "high": 60},
    "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
}
SPATIAL_WEIGHT_SPACE = {"spatial_weight": {"low": 0.1, "high": 10.0, "log": True}}


# ---------------------------------------------------------------------------
# resume-durable driver state (H-1 feature 7) — never module globals
# ---------------------------------------------------------------------------
def _ds(tree):
    return tree.setdefault("search_state", {}).setdefault("driver_state", {})


def _lineage_names(tree):
    return _ds(tree).setdefault("lineage_names", {})


def _node_results(tree):
    return _ds(tree).setdefault("node_results", {})


def _queues(tree):
    return _ds(tree).setdefault("queues", {})


def _dedup_rejections(tree):
    return _ds(tree).setdefault("dedup_rejections", [])


def _boundary_log(tree):
    return _ds(tree).setdefault("boundary_log", [])


def _priors_log(tree):
    return _ds(tree).setdefault("priors_log", [])


def core(cfg):
    """Canonical hashable stored form (07_tree_search.md §2). Solo configs are
    normalized to their full default-filled form so that two spellings of the same
    model (one omitting `spatial_weight`, one stating the default) hash identically —
    without this the dedup silently lets duplicates through."""
    out = {k: v for k, v in cfg.items() if k != "result"}
    if out.get("kind") == "blend" and "members" in out:
        return dict(out, members=sorted(out["members"]))
    if out.get("kind", "solo") == "solo":
        out.setdefault("use_spatial", True)
        out["spatial_weight"] = float(out.get("spatial_weight", 1.0))
        p = dict(out.get("params") or {})
        if out.get("model") == "krr":
            p.setdefault("kernel", "rbf")
            p.setdefault("gamma_scale", 1.0)
            p.setdefault("alphas", list(ev.DEFAULT_ALPHAS))
            if p["kernel"] == "poly":
                p.setdefault("degree", 2)
                p.setdefault("coef0", 1.0)
        elif out.get("model") == "svr":
            p.setdefault("kernel", "rbf")
            p.setdefault("gamma_scale", 1.0)
            p.setdefault("C", 10.0)
            p.setdefault("epsilon", 0.1)
        out["params"] = {k: p[k] for k in sorted(p)}
    return out


def n_evaluated(tree):
    return hv3.n_evaluated(tree)


def solo_pool(tree):
    ns = [n for n in tree["nodes"] if n["status"] == "evaluated"
          and n["config"].get("kind") == "solo" and n["score"] is not None]
    ns.sort(key=lambda n: n["score"])
    return ns


# ---------------------------------------------------------------------------
# reuse of linear-stage cached OOF, digit-verified
# ---------------------------------------------------------------------------
def _linear_pool():
    with open(LINEAR_STATE) as f:
        return json.load(f)


def try_reuse(stored, linear):
    """If `stored` byte-matches a linear-stage config, reload that node's cached OOF,
    recompute MCRMSE from it, and accept only on a 6-dp match with the stored score."""
    for name, rec in linear.items():
        if name == "_blend" or rec.get("status") != "evaluated":
            continue
        if core(rec["config"]) != stored:
            continue
        path = os.path.join(ev.CACHE_DIR, f"solo_{rec['node_id']}.npz")
        if not os.path.exists(path):
            return None
        d = np.load(path, allow_pickle=True)
        oof, pred = d["oof"], d["pred"]
        recomputed = ev.mcrmse(ev._Y, oof)
        if round(recomputed, 6) != round(rec["score"], 6):
            print(f"[reuse REJECTED] {name}: recomputed {recomputed:.6f} != stored "
                  f"{rec['score']:.6f} -> retraining instead")
            return None
        return name, oof, pred, recomputed
    return None


# ---------------------------------------------------------------------------
# per-node eval + add
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, proposal, linear, *, is_root=False,
                 lineage_id=None):
    stored = core(proposal)
    if not is_root:
        dup_id = hv2.find_duplicate_config(tree, stored)
        if dup_id is not None:
            _dedup_rejections(tree).append(dict(mutation=mutation, dup_id=dup_id))
            nid_null, _ = hv3.add_node(tree, parent_id, mutation + " [dedup pre-check]",
                                       stored, None, "failed", 0.0)
            assert nid_null is None
            hv3.save_search_state(tree, TREE_PATH)
            return None, dup_id, None

    nid = hv1.next_id(tree)
    kind = stored.get("kind", "solo")
    if kind == "solo":
        reused = try_reuse(stored, linear)
        if reused is not None:
            name, oof, pred, score = reused
            hv2.cache_oof(ev.CACHE_DIR, nid, oof, pred=pred, mcrmse=score)
            per = {ev.TARGETS[t]: round(ev.rmse(ev._Y[:, t], oof[:, t]), 6)
                   for t in range(ev.N_TARGETS)}
            result = {"mcrmse": round(score, 6), "per_target": per, "reused_from": name}
            score, status, wall_s = round(score, 6), "evaluated", 0.05
            full_mut = (mutation + f" [REUSED cached OOF from linear-stage '{name}', "
                        f"digit-verified MCRMSE {score:.6f}, no retraining]")
        else:
            r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, proposal, EVAL_TIMEOUT_S,
                                         node_id=nid)
            score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
            full_mut = mutation if status == "evaluated" else mutation + f" [{r.get('error')}]"
    else:
        r = ev.evaluate(proposal, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
        score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
        full_mut = mutation if status == "evaluated" else mutation + f" [{r.get('error')}]"

    if is_root:
        real_nid = hv3.add_root(tree, full_mut, stored, score, status, wall_s)
    else:
        real_nid, dup = hv3.add_node(tree, parent_id, full_mut, stored, score, status,
                                     wall_s)
        assert dup is None, f"unexpected post-eval dedup (dup=#{dup})"
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    if result is not None:
        _node_results(tree)[str(real_nid)] = result
    hv3.save_search_state(tree, TREE_PATH)
    return real_nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


# ---------------------------------------------------------------------------
# mutation queues per lineage
# ---------------------------------------------------------------------------
def _krr(variant, kernel="rbf", gamma_scale=1.0, use_spatial=True, spatial_weight=1.0,
         alphas=None, **kw):
    p = dict(kernel=kernel, gamma_scale=gamma_scale)
    if alphas is not None:
        p["alphas"] = alphas
    p.update(kw)
    return dict(kind="solo", model="krr", variant=variant, use_spatial=use_spatial,
                spatial_weight=spatial_weight, params=p)


def _svr(variant, C=10.0, epsilon=0.1, gamma_scale=1.0, use_spatial=True,
         spatial_weight=1.0, kernel="rbf"):
    return dict(kind="solo", model="svr", variant=variant, use_spatial=use_spatial,
                spatial_weight=spatial_weight,
                params=dict(kernel=kernel, gamma_scale=gamma_scale, C=C, epsilon=epsilon))


FINE_ALPHAS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

QUEUE_DEFS = {
    "KRR_KERNEL": [
        ("gamma-scale sweep down (bias-dominated kernel width)", _krr("sg1", gamma_scale=0.5)),
        ("gamma-scale sweep up", _krr("sg1", gamma_scale=2.0)),
        ("gamma 0.25 + finer alpha grid", _krr("sg1", gamma_scale=0.25, alphas=FINE_ALPHAS)),
        ("gamma 3.0", _krr("sg1", gamma_scale=3.0)),
        ("laplacian kernel, tuned gamma", _krr("sg1", kernel="laplacian", gamma_scale=0.3)),
        ("poly-3 kernel on SNV", _krr("snv", kernel="poly", degree=3, gamma_scale=1.0, coef0=1.0)),
        ("poly-2 on SG-d1", _krr("sg1", kernel="poly", degree=2, gamma_scale=1.0, coef0=1.0)),
        ("finer alpha grid at gamma 1.0", _krr("sg1", alphas=FINE_ALPHAS)),
    ],
    "VARIANT": [
        ("best-variant x weak spatial", _krr("sg1", spatial_weight=0.3)),
        ("best-variant x strong spatial", _krr("sg1", spatial_weight=3.0)),
        ("sg1w11 + gamma 0.5", _krr("sg1w11", gamma_scale=0.5)),
        ("sg1w11 no spatial", _krr("sg1w11", use_spatial=False)),
        ("snv + gamma 0.5", _krr("snv", gamma_scale=0.5)),
        ("sg2 + gamma 0.5 (2nd-derivative branch)", _krr("sg2", gamma_scale=0.5)),
        ("raw spectra + weak spatial", _krr("raw", spatial_weight=0.3, gamma_scale=0.5)),
        ("sg1 no spatial + finer alphas", _krr("sg1", use_spatial=False, alphas=FINE_ALPHAS)),
    ],
    "SVR": [
        ("SVR C down + tighter eps", _svr("sg1", C=3.0, epsilon=0.05)),
        ("SVR on sg1w11", _svr("sg1w11", C=10.0)),
        ("SVR gamma 0.5", _svr("sg1", gamma_scale=0.5)),
        ("SVR gamma 2.0 + eps 0.2", _svr("sg1", gamma_scale=2.0, epsilon=0.2)),
        ("SVR on snv", _svr("snv", C=10.0)),
        ("SVR no spatial", _svr("sg1", use_spatial=False)),
        ("SVR C 100 (upper edge)", _svr("sg1", C=100.0)),
    ],
    "DIVERSE": [
        ("PLS fewer components", dict(kind="solo", model="pls", variant="sg1",
                                      use_spatial=True, params=dict(n_components=12))),
        ("PLS more components", dict(kind="solo", model="pls", variant="sg1",
                                     use_spatial=True, params=dict(n_components=35))),
        ("LGB on 60 PCA components", dict(kind="solo", model="lgbpca", variant="sg1",
                                          use_spatial=True,
                                          params=dict(n_components=60, learning_rate=0.05))),
        ("LGB on 20 PCA components, slower lr", dict(kind="solo", model="lgbpca",
                                                      variant="sg1", use_spatial=True,
                                                      params=dict(n_components=20,
                                                                  learning_rate=0.03,
                                                                  n_estimators=700))),
        ("PLS on snv, 15 comps", dict(kind="solo", model="pls", variant="snv",
                                       use_spatial=True, params=dict(n_components=15))),
        ("LGB PCA on snv", dict(kind="solo", model="lgbpca", variant="snv",
                                 use_spatial=True, params=dict(n_components=40))),
    ],
}
# linear-pool member -> designed mutation queue; pool members not listed here become
# their own generic lineage (boundary-push + comp-local spatial_weight nudges only).
QUEUE_FOR = {"krr_rbf_sg1": "KRR_KERNEL", "krr_rbf_sg1_nospatial": "VARIANT",
             "pls_sg1": "DIVERSE"}
BLEND_LINEAGE = "BLEND"
BURST_LINEAGES = ["EXPL_KITCHENSINK", "EXPL_LOWGAMMA", "EXPL_HIGHREG", "EXPL_SPATIALFREE"]


def propose_blend(tree, topk=None, per_target=True, method="greedy"):
    pool = solo_pool(tree)
    if len(pool) < 2:
        return None
    ids = [n["id"] for n in pool]
    if topk is not None:
        ids = ids[:topk]
    if len(ids) < 2:
        return None
    return dict(kind="blend", members=sorted(ids), per_target=per_target,
                weight_search=method)


# (top-k members by solo score, per-target weights?, weight-search method).
# Greedy forward selection is listed first and most often because experience.md records
# it as this comp's working weight search; Dirichlet is kept in the plan so the search
# itself decides rather than the prior.
BLEND_PLAN = [
    (None, True, "greedy"), (None, True, "bagged_greedy"), (None, True, "dirichlet"),
    (12, True, "greedy"), (8, True, "greedy"), (None, True, "greedy_ascent"),
    (None, False, "greedy"), (20, True, "greedy"), (5, True, "dirichlet"),
    (16, True, "bagged_greedy"), (6, True, "greedy_ascent"), (10, True, "greedy"),
    (24, True, "greedy"), (None, True, "bagged_greedy"), (14, True, "greedy_ascent"),
]


def propose_child(tree, parent_id, lineage_id):
    """Return (config, mutation_str) or (None, None) when the lineage is exhausted."""
    names = _lineage_names(tree)
    name = names.get(str(lineage_id), names.get(lineage_id))
    queues = _queues(tree)

    if name == BLEND_LINEAGE or (name or "").startswith("EXPL_KITCHENSINK"):
        idx = queues.get(f"{name}_idx", 0)
        while idx < len(BLEND_PLAN):
            topk, per_t, method = BLEND_PLAN[idx]
            idx += 1
            cfg = propose_blend(tree, topk=topk, per_target=per_t, method=method)
            if cfg is None:
                continue
            queues[f"{name}_idx"] = idx
            label = f"top{topk}" if topk else "kitchen-sink (all solos)"
            return cfg, (f"[{name}] reblend {label} via {method}, "
                         f"{'per-target' if per_t else 'shared'} weights over "
                         f"{len(cfg['members'])} members")
        queues[f"{name}_idx"] = idx
        return None, None

    # solo lineages: pop the next queued mutation, then boundary-pushes off the
    # lineage's current best node
    q = QUEUE_DEFS.get(name)
    if q is not None:
        idx = queues.get(f"{name}_idx", 0)
        if idx < len(q):
            queues[f"{name}_idx"] = idx + 1
            mut, cfg = q[idx]
            return copy.deepcopy(cfg), f"[{name}] {mut}"

    # queue exhausted -> boundary-push on the lineage's best node (feature 4)
    by_lin = [n for n in tree["nodes"] if n["status"] == "evaluated" and n["score"] is not None
              and hv1.lineage_of(tree, n["id"]) == lineage_id
              and n["config"].get("kind") == "solo"]
    if not by_lin:
        return None, None
    best = min(by_lin, key=lambda n: n["score"])
    cands = hv3.boundary_candidates(best["config"], dict(SEARCH_SPACE))
    used = queues.setdefault(f"{name}_bpush", [])
    for c in cands:
        key = f"{best['id']}:{c['param']}:{c['new_value']}"
        if key in used:
            continue
        used.append(key)
        cfg = copy.deepcopy(best["config"])
        cfg.setdefault("params", {})[c["param"]] = c["new_value"]
        _boundary_log(tree).append(dict(node_id=best["id"], **c))
        return cfg, (f"[{name}] boundary-push {c['param']} {c['old_value']} -> "
                     f"{c['new_value']} (sat on the {c['edge']} edge of its declared range)")

    # last resort: spatial_weight nudge, a comp-local knob outside `params`
    sw = float(best["config"].get("spatial_weight", 1.0))
    for new_sw in (sw * 2.0, sw * 0.5, sw * 4.0, sw * 0.25):
        key = f"{best['id']}:spatial_weight:{new_sw}"
        if key in used:
            continue
        used.append(key)
        cfg = copy.deepcopy(best["config"])
        cfg["spatial_weight"] = round(new_sw, 4)
        return cfg, f"[{name}] spatial_weight {sw} -> {round(new_sw, 4)} (comp-local knob)"
    return None, None


# ---------------------------------------------------------------------------
# explore burst (mandatory, 07_tree_search.md §3)
# ---------------------------------------------------------------------------
BURST_SEEDS = [
    ("EXPL_LOWGAMMA", "long-shot: very wide RBF kernel (gamma 0.1, lower edge) + finest "
     "alpha grid — the opposite regularization regime from the exploit-phase champion",
     _krr("sg1", gamma_scale=0.1, alphas=FINE_ALPHAS)),
    ("EXPL_HIGHREG", "long-shot: linear kernel on raw spectra with heavy ridge only — "
     "the pre-kernel regime, kept as a contrast lineage",
     _krr("raw", kernel="linear", alphas=[0.3, 1.0, 3.0, 10.0, 30.0, 100.0])),
    ("EXPL_SPATIALFREE", "long-shot: spectra-only SVR at high C, no spatial covariates "
     "at all (train/test sites do not overlap, so spatial may be pure covariate shift)",
     _svr("sg1w11", C=30.0, use_spatial=False)),
]


def inject_explore_burst(tree, root_id, linear):
    names = _lineage_names(tree)
    for lname, mut, cfg in BURST_SEEDS:
        nid, dup, r = eval_and_add(tree, root_id, f"[{lname}] {mut}", cfg, linear)
        if nid is None:
            print(f"  burst seed {lname} dedup-rejected (dup of #{dup})")
            continue
        names[str(nid)] = lname
        passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
        flag = "" if passed else f"  <-- SANITY GATE FAILED (bound {bound:.6g}), lineage plateaued"
        print(f"  burst seed #{nid} {lname} MCRMSE={r['score']} {flag}")
    # the mandatory kitchen-sink blend of the entire current solo pool
    cfg = propose_blend(tree, topk=None, per_target=True)
    if cfg is not None:
        nid, dup, r = eval_and_add(tree, root_id,
                                   f"[EXPL_KITCHENSINK] mandatory explore-burst kitchen-sink "
                                   f"blend of all {len(cfg['members'])} solo nodes, per-target "
                                   f"weights", cfg, linear)
        if nid is not None:
            names[str(nid)] = "EXPL_KITCHENSINK"
            print(f"  burst seed #{nid} EXPL_KITCHENSINK MCRMSE={r['score']}")
        else:
            print(f"  kitchen-sink blend dedup-rejected (dup of #{dup})")
    hv3.save_search_state(tree, TREE_PATH)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def seed_first_generation(tree, linear):
    """Root + first-generation lineage seeds, reusing linear-stage cached OOF."""
    names = _lineage_names(tree)
    root_cfg = core(linear[ROOT_NAME]["config"])
    nid, _, r = eval_and_add(tree, None, f"[ROOT] linear-stage best solo ({ROOT_NAME})",
                             root_cfg, linear, is_root=True)
    assert r["status"] == "evaluated", f"root failed: {r}"
    assert round(r["score"], 6) == round(LINEAR_BEST_SOLO, 6), (
        f"ROOT DIGIT VERIFICATION FAILED: got {r['score']} expected {LINEAR_BEST_SOLO}")
    print(f"ROOT #{nid} verified: MCRMSE {r['score']:.6f} == linear best solo "
          f"{LINEAR_BEST_SOLO:.6f}")

    # Every linear-stage solo re-enters as a first-generation lineage. These cost ZERO
    # compute (cached-OOF reuse, digit-verified) and exist so the blend lineage can see
    # the same member diversity the linear blend had — the first run of this driver
    # seeded only 4 of them and its kitchen-sink blend lost to the linear blend purely
    # for lack of members, the same structural node-space gap 07_tree_search.md §1
    # records for s3e9-v1.
    seeds = [(QUEUE_FOR.get(name, name), f"seed: linear-stage pool member '{name}'",
              rec["config"])
             for name, rec in linear.items()
             if name not in ("_blend", "mean", ROOT_NAME) and rec.get("status") == "evaluated"]
    seeds.append(("SVR", "seed: SVR-RBF on SG-d1 at C=30 (root family, own lineage)",
                  _svr("sg1", C=30.0)))
    for lname, mut, cfg in seeds:
        cid, dup, rr = eval_and_add(tree, nid, f"[{lname}] {mut}", cfg, linear)
        if cid is None:
            print(f"  seed {lname} dedup-rejected (dup of #{dup})")
            continue
        names[str(cid)] = lname
        print(f"  seed #{cid} {lname} MCRMSE={rr['score']} ({rr['wall_s']}s)")

    bcfg = propose_blend(tree, topk=None, per_target=True)
    cid, dup, rr = eval_and_add(tree, nid,
                                f"[{BLEND_LINEAGE}] seed: per-target weighted blend of the "
                                f"{len(bcfg['members'])} solo nodes evaluated so far", bcfg,
                                linear)
    if cid is not None:
        names[str(cid)] = BLEND_LINEAGE
        print(f"  seed #{cid} {BLEND_LINEAGE} MCRMSE={rr['score']}")
    hv3.save_search_state(tree, TREE_PATH)
    return nid


def main():
    t0 = time.time()
    linear = _linear_pool()

    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        print(f"resumed tree with {n_evaluated(tree)} evaluated nodes")
    else:
        tree = hv1.new_tree(COMP)
        tree["search_state"].setdefault("plateaued", [])
        tree["search_state"].setdefault("backtrack_log", [])
        tree["search_state"].setdefault("streak", {})
    # 80 rather than the documented default 60: 15 of the first 16 nodes are zero-compute
    # cached-OOF re-imports of the linear pool (they occupy a budget slot but search
    # nothing), so the effective search budget stays at ~60 evaluated-by-training nodes.
    hv3.init_budget(tree, total_budget=80)

    # priors, queried once and logged verbatim (07_tree_search.md §5)
    if not _priors_log(tree):
        priors = hv2.suggest_priors({"metric": "rmse", "data_type": "spectral",
                                     "tags": ["p>>n", "spectral", "multi-target",
                                              "small_sample", "ensemble"]},
                                    experience_path=os.path.join(_REPO, "knowledge",
                                                                 "experience.md"))
        _priors_log(tree).extend(priors if isinstance(priors, list) else [str(priors)])
        print(f"priors: {len(_priors_log(tree))} experience-library bullets matched")

    if not tree["nodes"]:
        root_id = seed_first_generation(tree, linear)
    else:
        root_id = tree["root_id"]

    burst_done = any(n["mutation"].startswith("[EXPL_") for n in tree["nodes"])
    iterations = 0
    while (not hv3.should_stop(tree)) \
            and n_evaluated(tree) < hv3.init_budget(tree)["total_budget"] \
            and (time.time() - t0) < MAX_WALL_S:
        iterations += 1
        if iterations > ITER_SAFETY_CAP:
            print("iteration safety cap reached")
            break
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not burst_done:
            burst_done = True
            print(f"\n>>> PHASE -> explore_burst (n_eval={n_evaluated(tree)}); injecting "
                  f"{len(BURST_SEEDS)} long-shot lineages + the mandatory kitchen-sink "
                  f"blend <<<")
            inject_explore_burst(tree, root_id, linear)
            continue

        parent_id, lineage_id = hv1.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted")
            break
        cfg, mutation = propose_child(tree, parent_id, lineage_id)
        if cfg is None:
            st = tree["search_state"]
            if lineage_id not in st["plateaued"]:
                st["plateaued"].append(lineage_id)
                st["backtrack_log"].append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=f"lineage {_lineage_names(tree).get(str(lineage_id), lineage_id)} "
                           f"mutation space exhausted -> forced backtrack"))
            hv3.save_search_state(tree, TREE_PATH)
            print(f"FORCED BACKTRACK: lineage "
                  f"{_lineage_names(tree).get(str(lineage_id), lineage_id)} exhausted; "
                  f"plateaued={st['plateaued']}")
            continue

        nid, dup, r = eval_and_add(tree, parent_id, mutation, cfg, linear,
                                   lineage_id=lineage_id)
        # inherit the parent's lineage name for display
        if nid is not None:
            lname = _lineage_names(tree).get(str(lineage_id))
            if lname:
                _lineage_names(tree)[str(nid)] = lname
        if nid is None:
            print(f"DEDUP REJECTED <- parent#{parent_id} (dup of #{dup})")
            continue
        gb = hv1.global_best(tree)
        b = tree["search_state"]["budget"]
        print(f"#{nid} <- parent#{parent_id} "
              f"lineage={_lineage_names(tree).get(str(lineage_id), lineage_id)} "
              f"MCRMSE={r['score']} status={r['status']} wall={r['wall_s']}s "
              f"| best={gb['score']} (#{gb['id']}) | phase={b['phase']} "
              f"n_eval={n_evaluated(tree)} | plateaued={tree['search_state']['plateaued']}",
              flush=True)

    gb = hv1.global_best(tree)
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated"
                 and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated"
                  and n["config"].get("kind") == "blend")
    evals_to_match, running = None, None
    for i, n in enumerate([n for n in tree["nodes"] if n["status"] == "evaluated"], start=1):
        s = n["score"]
        running = s if running is None else min(running, s)
        if running <= LINEAR_BEST_BLEND and evals_to_match is None:
            evals_to_match = i
    tree["evals_to_match_linear_best"] = evals_to_match
    tree["dedup_rejections"] = _dedup_rejections(tree)
    tree["boundary_candidates_log"] = _boundary_log(tree)
    tree["cost_guard_fired"] = []
    hv3.save_search_state(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated(tree)} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={time.time() - t0:.1f}s")
    print(f"Global best: #{gb['id']} MCRMSE={gb['score']} | {gb['mutation'][:200]}")
    print(f"evals_to_match_linear_best({LINEAR_BEST_BLEND}): {evals_to_match}")
    print(f"dedup rejections: {len(_dedup_rejections(tree))}")
    print(f"plateaued lineages: {tree['search_state']['plateaued']}")
    print(f"stop_reason: {tree['search_state']['budget'].get('stop_reason')}")


if __name__ == "__main__":
    main()
