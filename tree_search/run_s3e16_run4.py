"""tree_search/run_s3e16_run4.py — Stage-4 tree-search driver for playground-series-s3e16
(Crab Age, MAE on an integer target, minimize).

Follows references/07_tree_search.md §4's five driver requirements:
  1. root digit-verified — the root IS the Optuna winner from
     competitions/playground-series-s3e16/scripts/lgb_optuna_best.json, and its freshly
     retrained score is asserted equal to that file's recorded value to 6 decimals.
  2. OOF cache reuse — deliberately NOT implemented (see NOTE_REUSE below): a solo eval
     here costs 10-60s, so the reuse machinery would cost more correctness risk than the
     compute it saves. Stated rather than silently skipped.
  3. resume-state contract — all driver bookkeeping lives in
     tree["search_state"]["driver_state"]; save/load go through
     hv3.save_search_state/hv3.load_search_state.
  4. subprocess eval timeout — every solo eval goes through hv3.eval_solo_subprocess.
  5. burst-seed sanity gate — hv3.apply_burst_seed_sanity_gate runs on every burst seed.

Isolation: this run never reads tree_search/eval_s3e16_v2.py, tree_search/run_s3e16_v3.py
or tree_search/cache_s3e16/ (earlier runs of the same competition). Its evaluator is
eval_s3e16_run4.py and its cache is cache_s3e16_run4/.
"""
import copy
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import harness_v2 as hv2   # noqa: E402
import harness_v3 as hv3   # noqa: E402
import eval_s3e16_run4 as ev  # noqa: E402

COMP = "playground-series-s3e16"
COMP_DIR = os.path.join(_REPO_ROOT, "competitions", COMP)
TREE_PATH = os.path.join(COMP_DIR, "experiments_tree.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_s3e16_run4.py")
OPTUNA_PATH = os.path.join(COMP_DIR, "scripts", "lgb_optuna_best.json")

EVAL_TIMEOUT_S = 900
MAX_WALL_S = 5400
ITER_SAFETY_CAP = 300

# Stage-3 linear-iteration reference points (competitions/.../experiments.json):
LINEAR_BEST = 1.332379          # blend_equal_all, 8-member equal-weight, pp=round-0.2
LINEAR_BEST_SOLO = 1.336970     # lgb_huber_base

NOTE_REUSE = ("OOF cache reuse from a prior tree is not implemented in this driver: solo "
              "evals cost 10-60s here, and this run has no prior tree it is permitted to "
              "read (lane isolation), so there is nothing to reuse from.")

_OPT = json.load(open(OPTUNA_PATH))
ROOT_PARAMS = dict(_OPT["fixed"], **_OPT["best_params"])
SEARCH_BOX = _OPT["search_box"]
ROOT_SCORE_EXPECTED = round(float(_OPT["best_value"]), 6)

ROOT_CONFIG = {"kind": "solo", "model": "lgb", "params": dict(ROOT_PARAMS),
               "blocks": ["base"], "seeds": [42]}
ROOT_MUTATION = ("root: Optuna(40 TPE trials, full 5-fold, post-processed-MAE objective) "
                 "winner over the declared box in lgb_optuna_best.json — LightGBM(huber)")

# boundary_candidates edge_frac: 0.10 rather than the 0.05 default. Stated because it is a
# real choice, not a default: at 0.05 the Optuna winner flags nothing (its most extreme
# coordinate, reg_alpha=5.02, sits at 0.925 of the log-box), so the boundary-push mutation
# type would have been vacuous for this run. 0.10 makes it fire on genuinely near-edge
# coordinates instead of on manufactured ones.
EDGE_FRAC = 0.10

BURST_NAMES = ["EXPL_DART", "EXPL_XT", "EXPL_DEEPCAT", "EXPL_LINEARTREE", "EXPL_TINYLR",
               "EXPL_MEGABLEND"]
ALL_LINEAGE_NAMES = ["BOUND", "CAPACITY", "OBJL1", "FEATBLK", "XGB", "CAT", "SEEDBAG",
                     "BLEND"] + BURST_NAMES

LINEAGE_NAMES = {}
BURST_INJECTED = [False]
BOUNDARY_LOG = []
_dedup_offset = {}


# --------------------------------------------------------------------------- state
def _driver_state(tree):
    return tree.setdefault("search_state", {}).setdefault("driver_state", {})


def _dedup_rejections(tree):
    return _driver_state(tree).setdefault("dedup_rejections", [])


def _cost_guard_fired(tree):
    return _driver_state(tree).setdefault("cost_guard_fired", [])


def _node_results(tree):
    return tree.setdefault("search_state", {}).setdefault("node_results", {})


def result_of(tree, nid):
    return _node_results(tree).get(str(nid))


def dc(cfg):
    return copy.deepcopy(cfg)


def core(cfg):
    """Canonical hashable stored form: drop eval-output keys, sort blend members."""
    out = {k: v for k, v in cfg.items() if k not in ("result",)}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def mae_of(r):
    if r is None:
        return None
    if r.get("result") and "mae" in r["result"]:
        return r["result"]["mae"]
    return r.get("score")


# --------------------------------------------------------------------------- blend
def evaluate_blend_v3(tree, stored_cfg):
    members = stored_cfg["members"]
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    t0 = time.time()
    best_w, best_s, oofs, warning = hv3.eval_blend_with_cost_guard(
        ev.CACHE_DIR, members, ev.mae_pp, tree=tree)
    wall = time.time() - t0
    if warning:
        _cost_guard_fired(tree).append(warning)
    blended = oofs @ best_w
    result = dict(members=members, weights=[round(float(w), 4) for w in best_w],
                  method="dirichlet k=800 + coordinate-ascent", mae=round(float(best_s), 6),
                  raw_mae=round(float(np.mean(np.abs(ev.y() - blended))), 6),
                  pp=ev.best_pp_name(blended))
    return result, float(best_s), wall, warning


# --------------------------------------------------------------------------- eval+add
def eval_and_add(tree, parent_id, mutation, proposal_cfg, is_root=False, lineage_id=None):
    stored = core(proposal_cfg)
    if not is_root:
        dup_id = hv3.find_duplicate_config(tree, stored)
        if dup_id is not None:
            _dedup_rejections(tree).append(dict(mutation=mutation, dup_id=dup_id))
            if lineage_id is not None:
                _dedup_offset[lineage_id] = _dedup_offset.get(lineage_id, 0) + 1
            nid_null, _echo = hv3.add_node(tree, parent_id, mutation + " [dedup pre-check]",
                                           stored, None, "failed", 0.0)
            assert nid_null is None
            hv3.save_search_state(tree, TREE_PATH)
            return None, dup_id, None

    nid = hv3.next_id(tree)
    kind = stored.get("kind", "solo")
    if kind == "solo":
        r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, stored, EVAL_TIMEOUT_S, node_id=nid)
        score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
        full_mutation = mutation if status == "evaluated" else mutation + f" [FAILED: {r.get('error')}]"
    else:
        try:
            result, best_score, wall_s, warning = evaluate_blend_v3(tree, stored)
            score, status = round(float(best_score), 6), "evaluated"
            full_mutation = mutation if not warning else mutation + f" [{warning}]"
        except Exception as e:  # noqa: BLE001 — keep the loop alive on a bad blend
            score, status, wall_s, result = None, "failed", 0.0, None
            full_mutation = mutation + f" [ERROR: {type(e).__name__}: {e}]"

    if is_root:
        real_nid = hv3.add_root(tree, full_mutation, stored, score, status, wall_s)
    else:
        real_nid, dup = hv3.add_node(tree, parent_id, full_mutation, stored, score, status, wall_s)
        assert dup is None, f"unexpected post-eval dedup (dup=#{dup})"
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    if result is not None:
        _node_results(tree)[str(real_nid)] = result
    hv3.save_search_state(tree, TREE_PATH)
    return real_nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


# --------------------------------------------------------------------------- seeds
def _bump(cfg, **kw):
    c = dc(cfg)
    c["params"] = dict(c["params"], **kw)
    return c


def seed_bound():
    """Boundary-push (harness feature 4): push every near-edge tuned coordinate outward."""
    edges = hv3.boundary_candidates({"params": ROOT_PARAMS}, SEARCH_BOX, edge_frac=EDGE_FRAC)
    BOUNDARY_LOG.extend(edges)
    if not edges:
        return None
    c = dc(ROOT_CONFIG)
    pushed = {}
    for e in edges:
        pushed[e["param"]] = e["new_value"]
    c["params"] = dict(c["params"], **pushed)
    return c, ("boundary-push: " + ", ".join(f"{k} {ROOT_PARAMS[k]:.4g}->{v:.4g}"
                                             for k, v in pushed.items())
               + f" (edge_frac={EDGE_FRAC} against the declared Optuna box)")


def seed_capacity():
    return (_bump(ROOT_CONFIG, num_leaves=127, min_child_samples=80),
            "capacity: num_leaves 33->127 with min_child_samples 41->80 to compensate")


def seed_objl1():
    c = dc(ROOT_CONFIG)
    c["params"] = {k: v for k, v in c["params"].items() if k != "alpha"}
    c["params"]["objective"] = "l1"
    return c, "objective swap huber->l1 at the tuned hyperparameters (exact-metric match)"


def seed_featblk():
    c = dc(ROOT_CONFIG)
    c["blocks"] = ["base", "infant", "resid", "frac", "shape", "density"]
    return c, ("feature blocks: raw -> raw+engineered (infant/resid/frac/shape/density). "
               "Round-1 said engineered HURT at default params (1.34153 vs 1.33864); "
               "re-tested at tuned params because the two interact")


def seed_xgb():
    return ({"kind": "solo", "model": "xgb", "blocks": ["base"], "seeds": [42],
             "params": dict(n_estimators=5000, learning_rate=0.02, max_depth=6,
                            min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=1.0, reg_lambda=2.0)},
            "cross-library: XGBoost reg:absoluteerror, shallower/slower than the Stage-3 member")


def seed_cat():
    return ({"kind": "solo", "model": "cat", "blocks": ["base"], "seeds": [42],
             "params": dict(iterations=6000, learning_rate=0.03, depth=7, l2_leaf_reg=6.0)},
            "cross-library: CatBoost MAE, depth 7 + stronger l2 than the Stage-3 member")


def seed_seedbag():
    c = dc(ROOT_CONFIG)
    c["seeds"] = [42, 202, 777]
    return c, "seed-bag: root config averaged over 3 seeds (variance reduction, no new idea)"


SOLO_SEED_SPECS = [("BOUND", seed_bound), ("CAPACITY", seed_capacity), ("OBJL1", seed_objl1),
                   ("FEATBLK", seed_featblk), ("XGB", seed_xgb), ("CAT", seed_cat),
                   ("SEEDBAG", seed_seedbag)]


def solo_pool(tree):
    return sorted(n["id"] for n in tree["nodes"]
                  if n["status"] == "evaluated" and n["config"].get("kind", "solo") == "solo")


def seed_blend(tree):
    pool = solo_pool(tree)
    return ({"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet"},
            f"blend seed: weight search over the full {len(pool)}-member solo pool")


# --------------------------------------------------------------------------- mutations
def _q(fn, desc):
    return lambda parent_cfg: (fn(parent_cfg), desc)


SOLO_QUEUES = {
    "BOUND": [
        _q(lambda c: _bump(c, learning_rate=c["params"]["learning_rate"] * 0.6,
                           n_estimators=9000),
           "push further: learning_rate x0.6 with the tree budget raised to 9000"),
        _q(lambda c: _bump(c, reg_alpha=min(c["params"].get("reg_alpha", 1.0) * 3, 60.0)),
           "push further: reg_alpha x3 (L1 leaf penalty was the near-edge coordinate)"),
        _q(lambda c: _bump(c, colsample_bytree=min(c["params"]["colsample_bytree"] * 1.08, 1.0),
                           subsample=max(c["params"]["subsample"] * 0.85, 0.4)),
           "push further: colsample up / subsample down past the box corner"),
    ],
    "CAPACITY": [
        _q(lambda c: _bump(c, num_leaves=255, min_child_samples=120),
           "capacity: num_leaves 127->255, min_child_samples 80->120"),
        _q(lambda c: _bump(c, num_leaves=15, min_child_samples=20, learning_rate=0.03),
           "opposite direction: tiny trees (num_leaves 15) with a faster learning rate"),
        _q(lambda c: _bump(c, max_depth=6),
           "add an explicit depth cap on top of the leaf count"),
    ],
    "OBJL1": [
        _q(lambda c: _bump(c, learning_rate=c["params"]["learning_rate"] * 0.6, n_estimators=9000),
           "l1 + slower learning rate (l1's sign-gradient likes small steps)"),
        _q(lambda c: dict(dc(c), params=dict(c["params"], objective="quantile", alpha=0.5)),
           "objective l1 -> quantile@0.5 (same optimum, different LightGBM implementation)"),
        _q(lambda c: _bump(c, min_child_samples=100),
           "l1 + heavier leaf-size regularisation"),
    ],
    "FEATBLK": [
        _q(lambda c: dict(dc(c), blocks=["base", "infant", "resid"]),
           "trim engineered blocks to the two with real EDA support (infant flag, weight residual)"),
        _q(lambda c: dict(dc(c), blocks=["base", "density", "shape"]),
           "allometry-only engineered set (density + shape ratios)"),
        _q(lambda c: dict(dc(c), blocks=["base", "infant", "resid", "frac", "shape",
                                         "density", "log", "inter"]),
           "all engineered blocks including log/interaction"),
    ],
    "XGB": [
        _q(lambda c: _bump(c, max_depth=8, min_child_weight=60),
           "deeper XGB with a heavier child-weight floor"),
        _q(lambda c: _bump(c, learning_rate=0.01, n_estimators=9000),
           "slower XGB, larger tree budget"),
        _q(lambda c: _bump(c, grow_policy="lossguide", max_leaves=64),
           "leaf-wise XGB (grow_policy=lossguide) — structurally closer to LightGBM"),
    ],
    "CAT": [
        _q(lambda c: _bump(c, depth=9, l2_leaf_reg=10.0),
           "deeper CatBoost with stronger l2"),
        _q(lambda c: _bump(c, learning_rate=0.015, iterations=9000),
           "slower CatBoost, larger iteration budget"),
        _q(lambda c: _bump(c, depth=6, bagging_temperature=1.0),
           "shallower CatBoost + Bayesian-bootstrap bagging"),
    ],
    "SEEDBAG": [
        _q(lambda c: dict(dc(c), seeds=[42, 202, 777, 1234, 31337]),
           "seed-bag widened 3 -> 5 seeds"),
        _q(lambda c: dict(dc(c), seeds=[42, 202, 777],
                          params=dict(c["params"], subsample=0.5, colsample_bytree=0.7)),
           "seed-bag with more per-model randomness (lower subsample/colsample) for diversity"),
        _q(lambda c: dict(dc(c), seeds=[42, 202, 777],
                          params=dict(c["params"], objective="l1")),
           "seed-bag of the l1 objective instead of huber"),
    ],
}


def solo_fallback(parent_cfg, attempt):
    """Generic parameter jitter once a lineage's authored queue is exhausted."""
    jitters = [
        ({"learning_rate": parent_cfg["params"].get("learning_rate", 0.03) * 0.75},
         "fallback jitter: learning_rate x0.75"),
        ({"reg_lambda": parent_cfg["params"].get("reg_lambda", 1.0) * 5 + 0.1},
         "fallback jitter: reg_lambda x5"),
        ({"subsample": max(parent_cfg["params"].get("subsample", 0.9) * 0.8, 0.35)},
         "fallback jitter: subsample x0.8"),
        ({"colsample_bytree": max(parent_cfg["params"].get("colsample_bytree", 0.9) * 0.8, 0.35)},
         "fallback jitter: colsample_bytree x0.8"),
    ]
    if attempt >= len(jitters):
        return None
    kw, desc = jitters[attempt]
    return _bump(parent_cfg, **kw), desc


def blend_fallback(tree, parent_node, attempt):
    """Blend mutations: add the current best non-member, drop the lowest-weight member,
    or rebuild from the whole pool."""
    members = list(parent_node["config"]["members"])
    pool = solo_pool(tree)
    res = result_of(tree, parent_node["id"]) or {}
    weights = res.get("weights") or []
    outside = [m for m in pool if m not in members]
    by_score = {n["id"]: n["score"] for n in tree["nodes"] if n["score"] is not None}

    cands = []
    if outside:
        best_out = min(outside, key=lambda m: by_score.get(m, 9e9))
        cands.append((sorted(members + [best_out]),
                      f"blend: absorb the best non-member solo #{best_out}"))
    if weights and len(members) > 2:
        weakest = members[int(np.argmin(weights))]
        cands.append((sorted([m for m in members if m != weakest]),
                      f"blend: drop the lowest-weight member #{weakest} (w={min(weights)})"))
    if len(pool) > len(members):
        cands.append((sorted(pool), "blend: rebuild over the entire current solo pool"))
    if len(members) > 3:
        top3 = sorted(sorted(members, key=lambda m: by_score.get(m, 9e9))[:3])
        cands.append((top3, "blend: keep only the 3 best-scoring members"))

    if attempt >= len(cands):
        return None
    mem, desc = cands[attempt]
    if len(mem) < 2:
        return None
    return {"kind": "blend", "members": mem, "weight_search": "dirichlet"}, desc


def _lineage_kind(tree, lineage_id):
    node = next(n for n in tree["nodes"] if n["id"] == lineage_id)
    return node["config"].get("kind", "solo")


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES.get(lineage_id, "?")
    idx = hv3.lineage_size(tree, lineage_id) - 1 + _dedup_offset.get(lineage_id, 0)
    if _lineage_kind(tree, lineage_id) == "blend":
        out = blend_fallback(tree, parent_node, max(idx, 0))
        if out is None:
            return None, None
        child_cfg, desc = out
    else:
        queue = SOLO_QUEUES.get(name, [])
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            out = solo_fallback(parent_node["config"], idx - len(queue))
            if out is None:
                return None, None
            child_cfg, desc = out
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


# --------------------------------------------------------------------------- burst
def _burst_seeds():
    return [
        ("EXPL_DART", {"kind": "solo", "model": "lgb", "blocks": ["base"], "seeds": [42],
                       "params": dict(ROOT_PARAMS, boosting_type="dart", drop_rate=0.1,
                                      max_drop=50, n_estimators=1500,
                                      early_stopping_rounds=1500)},
         "long-shot: DART boosting (dropout on trees); early stopping set to never "
         "realistically fire because DART's validation curve is non-monotone"),
        ("EXPL_XT", {"kind": "solo", "model": "lgb", "blocks": ["base"], "seeds": [42],
                     "params": dict(ROOT_PARAMS, extra_trees=True)},
         "long-shot: extra_trees=True — randomised split thresholds, diversity via "
         "randomisation instead of regularisation"),
        ("EXPL_DEEPCAT", {"kind": "solo", "model": "cat", "blocks": ["base"], "seeds": [42],
                          "params": dict(iterations=6000, learning_rate=0.03, depth=10,
                                         l2_leaf_reg=3.0, bagging_temperature=1.5)},
         "long-shot: CatBoost pushed to depth 10 with Bayesian-bootstrap bagging"),
        ("EXPL_LINEARTREE", {"kind": "solo", "model": "lgb", "blocks": ["base"], "seeds": [42],
                             "params": dict(ROOT_PARAMS, linear_tree=True, num_leaves=15,
                                            reg_lambda=5.0)},
         "long-shot: linear_tree=True — piecewise-LINEAR leaves instead of piecewise-"
         "constant. The EDA showed every feature is smooth and monotone in Age, which is "
         "exactly the regime where constant leaves waste capacity on staircases"),
        ("EXPL_TINYLR", {"kind": "solo", "model": "lgb", "blocks": ["base"], "seeds": [42],
                         "params": dict(ROOT_PARAMS, learning_rate=0.004, n_estimators=20000,
                                        num_leaves=63)},
         "long-shot: learning_rate 0.004 (well below the tuned box floor) with a 20000-tree "
         "budget — the classic 'the Optuna box floor was the real optimum' probe"),
    ]


def _sanity_check_burst_seed(tree, nid, name):
    passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
    if not passed:
        print(f"[BURST {name}] #{nid} FAILED sanity gate (bound={bound}) — lineage plateaued, "
              f"no children will be spawned from it")
    return passed


def inject_explore_burst(tree, root_id):
    injected = []
    for name, cfg, desc in _burst_seeds():
        nid, _dup, r = eval_and_add(tree, root_id, f"[{name}] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            SOLO_QUEUES.setdefault(name, [])
            injected.append(nid)
            print(f"[BURST {name}] #{nid} MAE={mae_of(r)} status={r['status']} wall_s={r['wall_s']}")
            _sanity_check_burst_seed(tree, nid, name)
    pool = solo_pool(tree)
    if len(pool) >= 2:
        cfg = {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet"}
        nid, _dup, r = eval_and_add(
            tree, root_id,
            f"[EXPL_MEGABLEND] long-shot: kitchen-sink blend of the entire {len(pool)}-member "
            "solo pool (mandatory burst injection per 07_tree_search.md §3)", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "EXPL_MEGABLEND"
            injected.append(nid)
            print(f"[BURST EXPL_MEGABLEND] #{nid} MAE={mae_of(r)} status={r['status']} "
                  f"wall_s={r['wall_s']}")
            _sanity_check_burst_seed(tree, nid, "EXPL_MEGABLEND")
    return injected


# --------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    os.makedirs(ev.CACHE_DIR, exist_ok=True)
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        print(f"Resuming tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv3.new_tree(COMP)
        hv3.init_budget(tree)
        print(f"init_budget -> {tree['search_state']['budget']}")

    if not tree["nodes"]:
        nid, _dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} MAE={mae_of(r)} wall_s={r['wall_s']} status={r['status']}")
        assert r["status"] == "evaluated", f"root eval failed: {r}"
        assert round(mae_of(r), 6) == ROOT_SCORE_EXPECTED, (
            f"ROOT DIGIT-VERIFY FAILED: expected {ROOT_SCORE_EXPECTED} "
            f"(lgb_optuna_best.json best_value), got {mae_of(r)}")
        print(f"ROOT DIGIT-VERIFIED against lgb_optuna_best.json ({ROOT_SCORE_EXPECTED}). OK.")

    if not BOUNDARY_LOG:
        BOUNDARY_LOG.extend(hv3.boundary_candidates({"params": ROOT_PARAMS}, SEARCH_BOX,
                                                    edge_frac=EDGE_FRAC))
    print(f"boundary_candidates(root vs its Optuna box, edge_frac={EDGE_FRAC}) -> {BOUNDARY_LOG}")

    for name, seed_fn in SOLO_SEED_SPECS:
        if any(n["mutation"].startswith(f"[{name}]") for n in tree["nodes"]
               if n["parent_id"] == tree["root_id"]):
            continue
        seeded = seed_fn()
        if seeded is None:
            print(f"[{name} seed] SKIPPED (nothing to propose — see boundary log)")
            continue
        cfg, desc = seeded
        nid, _dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} MAE={mae_of(r)} status={r['status']} wall_s={r['wall_s']}")

    if not any(n["mutation"].startswith("[BLEND]") for n in tree["nodes"]
               if n["parent_id"] == tree["root_id"]):
        cfg, desc = seed_blend(tree)
        nid, _dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} MAE={mae_of(r)} status={r['status']} wall_s={r['wall_s']}")

    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in ALL_LINEAGE_NAMES:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name
                    if name in BURST_NAMES:
                        SOLO_QUEUES.setdefault(name, [])
    if any(n["mutation"].startswith("[EXPL_") for n in tree["nodes"]
           if n["parent_id"] == tree["root_id"]):
        BURST_INJECTED[0] = True

    iterations = 0
    while (not hv3.should_stop(tree)) and hv3.n_evaluated(tree) < hv3.init_budget(tree)["total_budget"] \
            and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > ITER_SAFETY_CAP:
            print("Iteration safety cap reached, stopping.")
            break

        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not BURST_INJECTED[0]:
            BURST_INJECTED[0] = True
            print(f"\n>>> PHASE -> explore_burst (n_eval={hv3.n_evaluated(tree)}); injecting "
                  f"{len(_burst_seeds())+1} long-shot lineages <<<\n")
            inject_explore_burst(tree, tree["root_id"])
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted. Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        if child_cfg is None:
            st = tree["search_state"]
            if lineage_id not in st["plateaued"]:
                st["plateaued"].append(lineage_id)
                st["backtrack_log"].append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=f"lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)}'s mutation "
                           f"space exhausted -> forced backtrack"))
            hv3.save_search_state(tree, TREE_PATH)
            print(f"FORCED BACKTRACK: lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"exhausted; plateaued={st['plateaued']}")
            continue
        nid, dup, r = eval_and_add(tree, parent_id, mutation, child_cfg, lineage_id=lineage_id)
        if nid is None:
            print(f"DEDUP REJECTED <- parent#{parent_id} "
                  f"lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} (dup of #{dup})")
            continue
        gb = hv3.global_best(tree)
        budget = tree["search_state"]["budget"]
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"MAE={mae_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best={gb['score'] if gb else None} (#{gb['id'] if gb else '-'}) "
              f"| phase={budget['phase']} n_eval={hv3.n_evaluated(tree)} "
              f"| plateaued={tree['search_state']['plateaued']}")

    # ---- honest reporting block (07_tree_search.md §6) ----
    total_wall = time.time() - t_start
    gb = hv3.global_best(tree)
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated"
                 and n["config"].get("kind", "solo") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated"
                  and n["config"].get("kind") == "blend")
    evals_to_match, running = None, None
    for i, n in enumerate([n for n in tree["nodes"] if n["status"] == "evaluated"], start=1):
        s = n["score"]
        running = s if running is None else min(running, s)
        if running <= LINEAR_BEST and evals_to_match is None:
            evals_to_match = i
    tree["evals_to_match_linear_best"] = evals_to_match
    tree["dedup_rejections"] = _dedup_rejections(tree)
    tree["cost_guard_fired"] = _cost_guard_fired(tree)
    tree["boundary_candidates_log"] = BOUNDARY_LOG
    tree["linear_reference"] = {"linear_best": LINEAR_BEST, "linear_best_solo": LINEAR_BEST_SOLO,
                                "note_reuse": NOTE_REUSE}
    hv3.save_search_state(tree, TREE_PATH)

    print(f"\nDone. {hv3.n_evaluated(tree)} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} MAE={gb['score']} mutation={gb['mutation'][:200]}")
    print(f"Linear-iteration best: {LINEAR_BEST} -> tree "
          f"{'BEAT' if gb['score'] < LINEAR_BEST else ('TIED' if gb['score'] == LINEAR_BEST else 'REGRESSED vs')} it")
    print(f"Evals to match/beat the linear best: {evals_to_match}")
    print(f"Budget/phase: {tree['search_state']['budget']}")
    print(f"Boundary candidates: {BOUNDARY_LOG}")
    print(f"Cost guard fired: {len(_cost_guard_fired(tree))} time(s): {_cost_guard_fired(tree)}")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for e in tree["search_state"]["backtrack_log"]:
        print("  ", e)
    print(f"Dedup rejections ({len(_dedup_rejections(tree))}):")
    for e in _dedup_rejections(tree):
        print("  ", e)
    print(f"NOTE on reuse: {NOTE_REUSE}")
    print(f"hv2 module in use: {hv2.__name__}")


if __name__ == "__main__":
    main()
