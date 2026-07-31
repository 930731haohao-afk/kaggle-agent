"""tree_search/run_s4e1_v3.py — Tier4 harness_v3 tree-search driver for
playground-series-s4e1 (Bank Churn, ROC-AUC, maximize). First tree-search run for this
comp (no prior v2 tree to reuse from), following the driver checklist in
.claude/skills/kaggle-agent/references/07_tree_search.md and modeled on
tree_search/run_s3e7_v3.py.

Root: whichever tier2/tier3 solo model has the highest OOF AUC among the pool saved to
competitions/playground-series-s4e1/scripts/cache/solo_*.npz (LGB no-imbalance, XGB,
CAT, LGB_tuned if tier3 r2 ran) — picked PROGRAMMATICALLY (max cached "auc"), then
RETRAINED via eval_s4e1.evaluate_solo (byte-identical features.py/folds) and asserted to
match the cached score to 6 decimals before anything else proceeds (root digit-for-digit
verification requirement). Retraining (not reusing the competitions/ cache directly) is
deliberate: this comp's solo evals are cheap (~10-40s each on 165k rows / 20 cores), so
paying the retrain cost buys an independent verification for free instead of needing a
cross-cache-dir OOF-copy shortcut.

Node-space: solo + blend (default schema, `07_tree_search.md` §2 table). Six first-
generation solo lineages off the root (LGBHAND/XGBHAND/CATHAND/SEEDBAG/BOUNDARYPUSH/
FEATPRUNE) + one BLEND lineage. Each lineage's queue has 1-2 authored "new idea" mutations
before falling back to a generic seed-variation. Explore burst (mandatory once the
phase machine flips) injects a kitchen-sink blend of the entire solo pool + 2 long-shot
solo seeds, each gated by `apply_burst_seed_sanity_gate`.

Run: `uv run python3 tree_search/run_s4e1_v3.py` (resumable; state persisted to
competitions/playground-series-s4e1/experiments_tree_v3.json after every node).
`--dry-run` verifies wiring with zero training compute.
"""
import copy
import glob
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v3 as hv3  # noqa: E402
import eval_s4e1 as ev  # noqa: E402

COMP = "playground-series-s4e1"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_v3.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_s4e1.py")
SCRIPTS_CACHE = os.path.join(_COMP_DIR, "scripts", "cache")
EVAL_TIMEOUT_S = 240
MAX_WALL_S = 60 * 60  # soft safety net; should_stop is the intended stop mechanism
ITER_SAFETY_CAP = 200

# search box for boundary_candidates (feature 4) -- matches scripts/05_iterate.py's
# Optuna objective() search space verbatim
LGB_SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.1, "log": True},
    "num_leaves": {"low": 7, "high": 255, "log": True},
    "max_depth": (3, 12),
    "min_child_samples": (5, 100),
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
    "reg_alpha": {"low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"low": 1e-3, "high": 10.0, "log": True},
}


def dc(cfg):
    return copy.deepcopy(cfg)


def core(cfg):
    out = {k: v for k, v in cfg.items() if k not in ("result", "want_importance")}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def _driver_state(tree):
    return tree.setdefault("search_state", {}).setdefault("driver_state", {})


def _dedup_rejections(tree):
    return _driver_state(tree).setdefault("dedup_rejections", [])


def _node_results(tree):
    return tree.setdefault("search_state", {}).setdefault("node_results", {})


def result_of(tree, nid):
    return _node_results(tree).get(str(nid))


# ---------------------------------------------------------------------------
# root selection: pick the tier2/tier3 pool member with the highest cached OOF AUC
# ---------------------------------------------------------------------------
LGB_BASE = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                max_depth=-1, min_child_samples=30, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, seed=42)
XGB_BASE = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=42, tree_method="hist")
CAT_BASE = dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05, depth=7,
                l2_leaf_reg=3.0, random_seed=42)


def _candidate_root_configs():
    cands = {}
    for name in ("LGB", "XGB", "CAT", "LGB_tuned"):
        path = os.path.join(SCRIPTS_CACHE, f"solo_{name}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        cands[name] = float(d["auc"])
    tuned_params_path = os.path.join(SCRIPTS_CACHE, "lgb_tuned_params.json")
    model_map = {"LGB": ("lgb", LGB_BASE), "XGB": ("xgb", XGB_BASE), "CAT": ("cat", CAT_BASE)}
    if "LGB_tuned" in cands and os.path.exists(tuned_params_path):
        tuned = json.load(open(tuned_params_path))
        p = dict(LGB_BASE); p.update(tuned)
        model_map["LGB_tuned"] = ("lgb", p)
    return cands, model_map


def pick_root():
    cands, model_map = _candidate_root_configs()
    if not cands:
        raise RuntimeError("no cached tier2/tier3 solo scores found under "
                            f"{SCRIPTS_CACHE} -- run scripts/04_train_blend.py first")
    winner = max(cands, key=cands.get)
    model, params = model_map[winner]
    cfg = {"kind": "solo", "model": model, "params": dc(params), "features": {"drop": []},
           "want_importance": True}
    return winner, cfg, cands[winner], cands


# ---------------------------------------------------------------------------
# eval_and_add: single per-node entry point (dedup pre-check + subprocess eval + add)
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, proposal_cfg, is_root=False):
    stored = core(proposal_cfg)
    if not is_root:
        dup_id = hv3.find_duplicate_config(tree, stored)
        if dup_id is not None:
            _dedup_rejections(tree).append(dict(mutation=mutation, dup_id=dup_id))
            nid_null, _dup_echo = hv3.add_node(tree, parent_id, mutation + " [dedup pre-check]",
                                               stored, None, "failed", 0.0)
            assert nid_null is None
            hv3.save_search_state(tree, TREE_PATH)
            return None, dup_id, None

    nid = hv3.next_id(tree)
    kind = stored.get("kind", "solo")

    if kind == "solo":
        r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, proposal_cfg, EVAL_TIMEOUT_S, node_id=nid)
        score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
    elif kind == "blend":
        try:
            t0 = time.time()
            best_w, best_neg, _oofs = hv3.eval_blend(ev.CACHE_DIR, stored["members"],
                                                     lambda vec: -ev.auc(ev._y, vec))
            best_score = -best_neg
            wall_s = time.time() - t0
            result = dict(members=stored["members"], weights=[round(float(w), 4) for w in best_w],
                          auc=round(best_score, 6))
            score, status = round(-best_score, 6), "evaluated"
        except Exception as e:  # noqa: BLE001
            score, status, wall_s, result = None, "failed", 0.0, None
            mutation = mutation + f" [ERROR: {type(e).__name__}: {e}]"
    else:
        raise ValueError(f"unknown node kind {kind!r}")

    if is_root:
        real_nid = hv3.add_root(tree, mutation, stored, score, status, wall_s)
    else:
        real_nid, dup = hv3.add_node(tree, parent_id, mutation, stored, score, status, wall_s)
        assert dup is None, f"unexpected post-eval dedup (dup=#{dup})"
    assert real_nid == nid
    if result is not None:
        _node_results(tree)[str(real_nid)] = result
    hv3.save_search_state(tree, TREE_PATH)
    return real_nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


def auc_of(r):
    if r is None:
        return None
    if r.get("result") and "auc" in r["result"]:
        return r["result"]["auc"]
    return -r["score"] if r.get("score") is not None else None


# ---------------------------------------------------------------------------
# first-generation seeds
# ---------------------------------------------------------------------------
def seed_lgbhand(root_cfg):
    p = dc(LGB_BASE); p["num_leaves"] = 31; p["min_child_samples"] = 15
    return ({"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}},
            "model-variant: shallower/less-regularized hand LGB (num_leaves 63->31, "
            "min_child_samples 30->15) -- diversity direction distinct from root")


def seed_xgbhand():
    p = dc(XGB_BASE); p["max_depth"] = 8; p["learning_rate"] = 0.03
    return ({"kind": "solo", "model": "xgb", "params": p, "features": {"drop": []}},
            "model-variant: deeper/lower-lr hand XGB (depth 6->8, lr 0.05->0.03)")


def seed_cathand():
    p = dc(CAT_BASE); p["depth"] = 9; p["l2_leaf_reg"] = 5.0
    return ({"kind": "solo", "model": "cat", "params": p, "features": {"drop": []}},
            "model-variant: deeper hand CatBoost (depth 7->9, l2_leaf_reg 3->5)")


def seed_seedbag(root_cfg):
    p = dc(root_cfg["params"])
    seed_key = "seed" if root_cfg["model"] == "lgb" else ("random_state" if root_cfg["model"] == "xgb" else "random_seed")
    p[seed_key] = 2024
    return ({"kind": "solo", "model": root_cfg["model"], "params": p, "features": {"drop": []}},
            f"seed variation: root's exact params, {seed_key} -> 2024 [PRIOR: seed-bagging "
            f"is the cheapest residual gain after tuning, knowledge/experience.md]")


def seed_boundarypush(root_cfg):
    if root_cfg["model"] != "lgb":
        return None
    edges = hv3.boundary_candidates(root_cfg, LGB_SEARCH_SPACE)
    if not edges:
        return None
    p = dc(root_cfg["params"])
    descs = []
    for e in edges:
        p[e["param"]] = e["new_value"]
        descs.append(f"{e['param']} {e['old_value']}->{e['new_value']} (edge={e['edge']})")
    return ({"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}},
            f"model-variant: boundary_candidates()-flagged push(es) -- {', '.join(descs)} "
            f"[feature 4]")


def seed_featprune():
    weak = ["age_decade", "products_eq1", "has_balance"]  # low-signal engineered flags
    return ({"kind": "solo", "model": "lgb", "params": dc(LGB_BASE),
             "features": {"drop": weak}},
            f"feature-pruning: drop {len(weak)} low-signal engineered flags ({', '.join(weak)}) "
            f"[knowledge/experience.md: feature pruning is a recurring win on small "
            f"engineered feature sets]")


ALL_LINEAGE_NAMES = ["LGBHAND", "XGBHAND", "CATHAND", "SEEDBAG", "BOUNDARYPUSH", "FEATPRUNE", "BLEND"]
LINEAGE_NAMES = {}


def _lineage_seed_ids(tree):
    out = {}
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name in ALL_LINEAGE_NAMES:
                if name not in out and n["mutation"].startswith(f"[{name}]"):
                    out[name] = n["id"]
    return out


def _best_node_in_lineage(tree, lineage_id):
    best = None
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["id"] == tree["root_id"]:
            continue
        if hv3.lineage_of(tree, n["id"]) == lineage_id:
            if best is None or n["score"] < best["score"]:
                best = n
    return best


def solo_pool(tree):
    nodes = [n for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def _bump(cfg, **kw):
    c = dc(cfg)
    c["params"].update(kw)
    return c


def _lineage_kind(tree, lineage_id):
    node = next(n for n in tree["nodes"] if n["id"] == lineage_id)
    return node["config"].get("kind", "solo")


SOLO_QUEUES_BUILDERS = {
    "LGBHAND": [lambda c: (_bump(c, min_child_samples=60, reg_lambda=2.0),
                           "LGBHAND: NEW -- stronger regularization (min_child_samples "
                           "15->60, +reg_lambda=2.0)")],
    "XGBHAND": [lambda c: (_bump(c, subsample=0.6, colsample_bytree=0.6),
                           "XGBHAND: NEW -- more aggressive row/col subsampling (0.8->0.6)")],
    "CATHAND": [lambda c: (_bump(c, bagging_temperature=1.0),
                           "CATHAND: NEW -- bagging_temperature=1.0 (Bayesian-bootstrap "
                           "row weighting)")],
    "SEEDBAG": [lambda c: (_bump(c, **{("seed" if c["model"] == "lgb" else
                                        ("random_state" if c["model"] == "xgb" else "random_seed")): 777}),
                           "SEEDBAG: NEW -- third independent seed (777)")],
    "BOUNDARYPUSH": [],
    "FEATPRUNE": [lambda c: ({**c, "features": {"drop": ["age_decade"]}},
                             "FEATPRUNE: NEW -- isolate dropping just age_decade alone")],
}


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    seed_key = "seed" if c["model"] == "lgb" else ("random_state" if c["model"] == "xgb" else "random_seed")
    c["params"][seed_key] = 4000 + attempt
    return c, f"fallback seed-variation: alt {seed_key}={4000 + attempt}"


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = sorted({tree["root_id"], ids.get("XGBHAND"), ids.get("CATHAND")} - {None})
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    return cfg, f"ensemble seed: blend of root + XGBHAND + CATHAND ({members})"


def blend_fallback(tree, parent_node, attempt):
    pool = solo_pool(tree)
    members = parent_node["config"]["members"]
    for add_id in [nid for nid in pool if nid not in members]:
        child = {"kind": "blend", "members": sorted(members + [add_id]), "weight_search": "dirichlet"}
        if hv3.find_duplicate_config(tree, child) is None:
            return child, f"BLEND fallback: add next-best unused pool member #{add_id}"
    return None


def _blend_remove_weakest(tree, parent_node):
    res = result_of(tree, parent_node["id"]) or {}
    weights = res.get("weights")
    members = parent_node["config"]["members"]
    if not weights or len(members) <= 2:
        return None
    idx = int(np.argmin(weights))
    weak_id = members[idx]
    child = dc(parent_node["config"])
    child["members"] = sorted([m for m in members if m != weak_id])
    return child, f"BLEND: remove lowest-weight member #{weak_id} (w={weights[idx]:.3f})"


def _mk_add_named(name):
    def fn(tree, parent_node):
        ids = _lineage_seed_ids(tree)
        if name not in ids:
            return None
        add_id = _best_node_in_lineage(tree, ids[name])
        add_id = add_id["id"] if add_id else ids[name]
        members = parent_node["config"]["members"]
        if add_id in members:
            return None
        child = {"kind": "blend", "members": sorted(members + [add_id]), "weight_search": "dirichlet"}
        return child, f"BLEND: add {name} solo member (#{add_id})"
    return fn


BLEND_QUEUE = [_mk_add_named("SEEDBAG"), _mk_add_named("BOUNDARYPUSH"),
               _mk_add_named("FEATPRUNE"), _mk_add_named("LGBHAND"), _blend_remove_weakest]


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES[lineage_id]
    idx = hv3.lineage_size(tree, lineage_id) - 1
    if _lineage_kind(tree, lineage_id) == "blend":
        result = None
        if idx < len(BLEND_QUEUE):
            result = BLEND_QUEUE[idx](tree, parent_node)
        if result is None:
            result = blend_fallback(tree, parent_node, idx)
        if result is None:
            return None, None
        child_cfg, desc = result
    else:
        queue = SOLO_QUEUES_BUILDERS.get(name, [])
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


def inject_explore_burst(tree, root_id):
    pool = solo_pool(tree)
    seeds = []
    if len(pool) >= 2:
        cfg = {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet"}
        seeds.append((cfg, f"[EXPL_MEGABLEND] explore-burst: kitchen-sink blend of the "
                            f"entire solo pool ({len(pool)} members) [PRIOR: post-plateau "
                            f"gains historically come from mandatory mega-blends, not "
                            f"hand-written long-shot solos, 07_tree_search.md §3]"))
    root_node = next(n for n in tree["nodes"] if n["id"] == root_id)
    if root_node["config"]["model"] == "lgb":
        p = dc(root_node["config"]["params"]); p["num_leaves"] = 255; p["learning_rate"] = 0.01
        seeds.append(({"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}},
                      "[EXPL_DEEPLGB] explore-burst long-shot: much deeper/slower LGB "
                      "(num_leaves->255, lr->0.01)"))
    p_cat = dc(CAT_BASE); p_cat["depth"] = 10; p_cat["learning_rate"] = 0.02
    seeds.append(({"kind": "solo", "model": "cat", "params": p_cat, "features": {"drop": []}},
                  "[EXPL_DEEPCAT] explore-burst long-shot: much deeper/slower CatBoost "
                  "(depth->10, lr->0.02)"))
    return seeds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def dry_run():
    print("[dry-run] picking root...")
    winner, root_cfg, root_auc, cands = pick_root()
    print(f"[dry-run] root candidate pool: {cands}")
    print(f"[dry-run] chosen root = {winner} (cached AUC {root_auc:.6f})")
    print(f"[dry-run] root_cfg = {root_cfg}")
    print("[dry-run] seeds:")
    for name, fn in [("XGBHAND", seed_xgbhand), ("CATHAND", seed_cathand),
                     ("FEATPRUNE", seed_featprune)]:
        _cfg, desc = fn()
        print(f"  {name}: {desc}")
    print("[dry-run] OK -- no training performed.")


def main():
    t_start = time.time()
    resumed = os.path.exists(TREE_PATH)
    # Resume-state contract (harness_v3 feature 7): load_search_state() re-validates
    # search_state (phase/plateau/dedup_streak/driver_state self-consistency) on every
    # load instead of trusting whatever a killed process last managed to flush, unlike
    # the raw hv3.load() a plain resume would use.
    tree = hv3.load_search_state(TREE_PATH) if resumed else hv3.new_tree(COMP)
    hv3.init_budget(tree)  # 60 total / burst 6 / patience 20 -- harness defaults

    if not resumed:
        winner, root_cfg, root_auc, cands = pick_root()
        print(f"[root] candidate pool (cached OOF AUC): {cands}")
        print(f"[root] chosen: {winner} (cached AUC {root_auc:.6f}) -- retraining to verify digit-for-digit")
        mutation = (f"root: tier2/tier3 strongest cached solo ({winner}), retrained via "
                    f"eval_s4e1.evaluate_solo for independent digit-for-digit verification "
                    f"against cached AUC {root_auc:.6f}")
        nid, _, info = eval_and_add(tree, None, mutation, root_cfg, is_root=True)
        got_auc = auc_of(info)
        assert got_auc is not None and round(got_auc, 6) == round(root_auc, 6), (
            f"ROOT VERIFICATION FAILED: retrained AUC {got_auc} != cached {root_auc}")
        print(f"[root] verified digit-for-digit: {got_auc:.6f} == {root_auc:.6f}")
    else:
        print(f"[resume] loaded existing tree via load_search_state with {len(tree['nodes'])} "
              f"nodes ({hv3.n_evaluated(tree)} evaluated)")

    # First-generation seeds, keyed by lineage name and checked against what already
    # exists as a root child -- idempotent, so a run interrupted mid-seeding (killed
    # after N of 7 first-gen seeds, as this one was: root+LGBHAND+XGBHAND+CATHAND+
    # SEEDBAG done, BOUNDARYPUSH/FEATPRUNE/BLEND never reached) resumes by finishing
    # only the missing ones instead of silently entering the search loop with an
    # incomplete lineage set -- select_next_parent() only ever expands a lineage that
    # ALREADY exists as a root child (harness_v2.select_next_parent docstring: "caller
    # should seed first-generation lineages directly off the root"); it can never
    # invent a missing one on its own.
    root_id = tree["root_id"]
    root_cfg = next(n for n in tree["nodes"] if n["id"] == root_id)["config"]
    existing = _lineage_seed_ids(tree)
    for name, nid in existing.items():
        LINEAGE_NAMES[nid] = name
    if existing and len(existing) < len(ALL_LINEAGE_NAMES):
        missing = [n for n in ALL_LINEAGE_NAMES if n not in existing]
        print(f"[seed] {len(existing)}/{len(ALL_LINEAGE_NAMES)} first-gen lineages already "
              f"present {sorted(existing)} -- completing missing: {missing}")

    if "LGBHAND" not in existing:
        cfg, desc = seed_lgbhand(root_cfg)
        nid, _dup, info = eval_and_add(tree, root_id, f"[LGBHAND] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "LGBHAND"
            print(f"[seed LGBHAND] node #{nid} AUC={auc_of(info)}")
    if "XGBHAND" not in existing:
        cfg, desc = seed_xgbhand()
        nid, _dup, info = eval_and_add(tree, root_id, f"[XGBHAND] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "XGBHAND"
            print(f"[seed XGBHAND] node #{nid} AUC={auc_of(info)}")
    if "CATHAND" not in existing:
        cfg, desc = seed_cathand()
        nid, _dup, info = eval_and_add(tree, root_id, f"[CATHAND] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "CATHAND"
            print(f"[seed CATHAND] node #{nid} AUC={auc_of(info)}")
    if "SEEDBAG" not in existing:
        cfg, desc = seed_seedbag(root_cfg)
        nid, _dup, info = eval_and_add(tree, root_id, f"[SEEDBAG] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "SEEDBAG"
            print(f"[seed SEEDBAG] node #{nid} AUC={auc_of(info)}")
    if "BOUNDARYPUSH" not in existing:
        bp = seed_boundarypush(root_cfg)
        if bp:
            cfg, desc = bp
            nid, _dup, info = eval_and_add(tree, root_id, f"[BOUNDARYPUSH] {desc}", cfg)
            if nid is not None:
                LINEAGE_NAMES[nid] = "BOUNDARYPUSH"
                print(f"[seed BOUNDARYPUSH] node #{nid} AUC={auc_of(info)}")
    if "FEATPRUNE" not in existing:
        cfg, desc = seed_featprune()
        nid, _dup, info = eval_and_add(tree, root_id, f"[FEATPRUNE] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "FEATPRUNE"
            print(f"[seed FEATPRUNE] node #{nid} AUC={auc_of(info)}")
    if "BLEND" not in existing:
        cfg, desc = seed_blend(tree)
        nid, _dup, info = eval_and_add(tree, root_id, f"[BLEND] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "BLEND"
            print(f"[seed BLEND] node #{nid} AUC={auc_of(info)}")

    burst_injected = bool(tree["search_state"].get("driver_state", {}).get("burst_injected"))
    it = 0
    while not hv3.should_stop(tree) and it < ITER_SAFETY_CAP and (time.time() - t_start) < MAX_WALL_S:
        it += 1
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not burst_injected:
            print(f"\n>>> PHASE -> explore_burst (n_eval={hv3.n_evaluated(tree)}); "
                  f"injecting mega-blend + long-shot seeds")
            root_id = tree["root_id"]
            for cfg, desc in inject_explore_burst(tree, root_id):
                name = desc.split("]")[0][1:]
                nid, _dup, info = eval_and_add(tree, root_id, desc, cfg)
                if nid is not None:
                    LINEAGE_NAMES[nid] = name
                    score_auc = auc_of(info)
                    print(f"[burst {name}] node #{nid} AUC={score_auc}")
                    if cfg.get("kind") == "solo" and score_auc is not None:
                        ok, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
                        print(f"    sanity gate: {'PASS' if ok else 'FAIL (plateaued)'} (bound={bound})")
            _driver_state(tree)["burst_injected"] = True
            burst_injected = True
            hv3.save_search_state(tree, TREE_PATH)
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            print("[stop] select_next_parent returned None -- no lineage left to expand")
            break
        if lineage_id not in LINEAGE_NAMES:
            # burst-injected lineage with no authored name mapping yet
            root_children = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
            for n in root_children:
                if n["id"] == lineage_id:
                    LINEAGE_NAMES[lineage_id] = n["mutation"].split("]")[0].lstrip("[")
        child_cfg, desc = propose_child(tree, parent_id, lineage_id)
        if child_cfg is None:
            # propose_child exhausted (e.g. BLEND's fallback ran out of distinct unused
            # pool-member subsets to try -- every remaining candidate duplicates an
            # already-evaluated node). select_next_parent has NO other way to learn this
            # (it only marks a lineage plateaued when every node in it hits
            # MAX_CHILDREN_PER_NODE, which doesn't apply here), so left alone it keeps
            # re-selecting this same still-"best" lineage and getting None back forever
            # -- a silent spin that burns the entire ITER_SAFETY_CAP with zero further
            # evals and never reaches the other lineages' own untried mutations, nor the
            # mandatory explore-burst phase (observed live in this resumed run: 14/60
            # nodes, BLEND's dead end, phase stuck "exploit", loop only ended when
            # it hit ITER_SAFETY_CAP with an empty backtrack_log). Mark it plateaued
            # ourselves -- same fields/shape select_next_parent's own MAX_CHILDREN_PER_NODE
            # exhaustion path already uses -- so the search moves on instead of stalling.
            st = tree["search_state"]
            plat = set(st.get("plateaued", []))
            if lineage_id not in plat:
                plat.add(lineage_id)
                st["plateaued"] = sorted(plat)
                st.setdefault("backtrack_log", []).append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=f"propose_child exhausted for lineage "
                           f"{LINEAGE_NAMES.get(lineage_id, lineage_id)} (#{lineage_id}) -- no "
                           f"more distinct mutations available (driver-side plateau, not "
                           f"MAX_CHILDREN_PER_NODE) -- marking plateaued so select_next_parent "
                           f"moves to the next-best lineage instead of silently re-proposing "
                           f"the same dead end until ITER_SAFETY_CAP"))
                print(f"[plateau] lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} (#{lineage_id}) "
                      f"exhausted -- marked plateaued, moving on")
            if st.get("active_lineage") == lineage_id:
                st["active_lineage"] = None
            hv3.save_search_state(tree, TREE_PATH)
            continue
        nid, _dup, info = eval_and_add(tree, parent_id, desc, child_cfg)
        if nid is None:
            continue
        gb = hv3.global_best(tree)
        print(f"[eval {it}] node #{nid} (parent #{parent_id}, lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)}) "
              f"AUC={auc_of(info)}  n_eval={hv3.n_evaluated(tree)}  global_best_auc={-gb['score'] if gb else None}")

    gb = hv3.global_best(tree)
    print(f"\n=== stopped: phase={tree['search_state']['budget']['phase']} "
          f"reason={tree['search_state']['budget'].get('stop_reason')} ===")
    print(f"n_evaluated={hv3.n_evaluated(tree)}  wall={time.time()-t_start:.1f}s")
    print(f"global best: node #{gb['id']}  AUC={-gb['score']:.6f}  kind={gb['config'].get('kind')}")
    print(f"dedup_rejections={len(_dedup_rejections(tree))}")
    print(f"backtrack_log={tree['search_state'].get('backtrack_log')}")
    hv3.save_search_state(tree, TREE_PATH)
    print(f"\nRESULT {json.dumps(dict(n_evaluated=hv3.n_evaluated(tree), best_node=gb['id'], best_auc=-gb['score']))}")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run()
    else:
        main()
