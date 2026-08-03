"""tree_search/run_s3e16_run5.py -- Stage 4 tree-search driver for playground-series-s3e16.

Entered per SKILL.md once the linear Iteration Protocol produced a baseline solo
(lgb_raw_l1, rounded OOF MAE 1.337524) and a first blend (honest 1.334121). Follows the
run_s3e7_v3.py driver checklist in references/07_tree_search.md:

  1. root digit-for-digit verified against the linear-iteration best solo;
  2. OOF cache reuse -- round 1's cached OOFs are re-imported and re-scored, accepted
     only on a 6-decimal match;
  3. resume state lives in tree["search_state"]["driver_state"], never in globals;
  4. every solo eval goes through hv3.eval_solo_subprocess (OS-level timeout);
  5. every explore-burst seed passes through hv3.apply_burst_seed_sanity_gate.

Decision metric throughout: MAE on the ROUNDED prediction (dossier metric contract).
Blend scores are leave-fold-out honest (weights refitted per fold).
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import harness_v3 as hv3  # noqa: E402

COMP = "playground-series-s3e16"
COMP_DIR = os.path.join(_REPO_ROOT, "competitions", COMP)
TREE_PATH = os.path.join(COMP_DIR, "experiments_tree.json")
EVAL_PATH = os.path.join(_HERE, "eval_s3e16_run5.py")
CACHE_DIR = os.path.join(_HERE, "cache_s3e16_run5")
OOF_DIR = os.path.join(COMP_DIR, "scripts", "oof")

ROOT_CONFIG = {"kind": "solo", "model": "lgb", "params": {}, "features": {"set": "raw"}}
ROOT_EXPECTED = 1.337524          # round1_result.json lgb_raw_l1, rounded-MAE
TOTAL_BUDGET = 60
SOLO_TIMEOUT_S = 420
RNG = np.random.default_rng(20260731)

SEARCH_SPACE = {
    "lgb": {"learning_rate": {"low": 0.005, "high": 0.10, "log": True},
            "num_leaves": (15, 255),
            "min_child_samples": (5, 200), "feature_fraction": (0.4, 1.0),
            "bagging_fraction": (0.4, 1.0), "reg_alpha": (0.0, 10.0),
            "reg_lambda": (0.0, 10.0), "max_bin": (63, 511)},
    "xgb": {"learning_rate": {"low": 0.005, "high": 0.10, "log": True},
            "max_depth": (3, 12),
            "min_child_weight": (1, 100), "subsample": (0.4, 1.0),
            "colsample_bytree": (0.4, 1.0), "reg_alpha": (0.0, 10.0),
            "reg_lambda": (0.0, 10.0)},
    "cat": {"learning_rate": {"low": 0.01, "high": 0.15, "log": True},
            "depth": (4, 10),
            "l2_leaf_reg": (0.5, 20.0)},
    "huber": {"alpha": (0.0001, 10.0), "epsilon": (1.05, 3.0)},
}
INT_PARAMS = {"num_leaves", "min_child_samples", "max_depth", "min_child_weight",
              "depth", "max_bin"}
FEATURE_SETS = ["raw", "core", "eng", "full"]

# The round-1 pool, imported as generation-1 seeds (config -> cached oof/pred file stem).
ROUND1_SEEDS = [
    ("lgb_eng_l1", {"kind": "solo", "model": "lgb", "params": {},
                    "features": {"set": "eng"}}, 1.341265),
    ("lgb_eng_huberloss", {"kind": "solo", "model": "lgb",
                           "params": {"objective": "huber", "alpha": 1.0, "metric": "mae"},
                           "features": {"set": "eng"}}, 1.338848),
    ("xgb_eng_l1", {"kind": "solo", "model": "xgb", "params": {},
                    "features": {"set": "eng"}}, 1.342332),
    ("cat_eng_mae", {"kind": "solo", "model": "cat", "params": {},
                     "features": {"set": "eng"}}, 1.342021),
    ("lgb_eng_l2", {"kind": "solo", "model": "lgb",
                    "params": {"objective": "regression", "metric": "l2"},
                    "features": {"set": "eng"}}, 1.376092),
    ("huber_full", {"kind": "solo", "model": "huber", "params": {},
                    "features": {"set": "full"}}, 1.396794),
]


def core(cfg):
    """Canonical hashable stored form of a node config (07_tree_search.md §2)."""
    c = {k: v for k, v in cfg.items() if k not in ("result", "want_importance")}
    if c.get("kind") == "blend":
        c["members"] = sorted(c.get("members") or [])
    return c


def ds(tree):
    return tree["search_state"].setdefault("driver_state", {})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# evaluation wrappers
# ---------------------------------------------------------------------------
def eval_node(tree, config, node_id):
    if config.get("kind") == "blend":
        sys.path.insert(0, _HERE)
        import eval_s3e16_run5 as ev
        return ev.evaluate(config, node_id=node_id, timeout_s=SOLO_TIMEOUT_S)
    return hv3.eval_solo_subprocess(EVAL_PATH, config, SOLO_TIMEOUT_S, node_id=node_id)


def add_and_eval(tree, parent_id, mutation, config, *, name=None):
    """Reserve the next node id, evaluate, then add. Returns (nid, res) or (None, dup)."""
    cfg = core(config)
    dup = hv3.v2.find_duplicate_config(tree, cfg)
    if dup is not None:
        # let the harness book the dedup-streak/placeholder bookkeeping
        return hv3.add_node(tree, parent_id, mutation, cfg, None, "failed", 0.0)
    nid = hv3.v2.v1.next_id(tree)
    res = eval_node(tree, cfg, nid)
    got, _ = hv3.add_node(tree, parent_id, mutation, cfg, res.get("score"),
                          res.get("status", "failed"), res.get("wall_s", 0.0))
    if got is not None:
        ds(tree).setdefault("node_results", {})[str(got)] = res.get("result")
        ds(tree).setdefault("node_names", {})[str(got)] = name or mutation[:40]
        if got != nid:
            log(f"  !! node id drift: reserved {nid}, got {got}")
    sc = res.get("score")
    log(f"  node #{got} [{mutation[:52]}] score={sc if sc is None else round(sc, 6)} "
        f"({res.get('wall_s')}s){' ERR=' + str(res.get('error')) if res.get('error') else ''}")
    return got, res


# ---------------------------------------------------------------------------
# mutation proposers
# ---------------------------------------------------------------------------
def jitter_params(model, params, n=2):
    space = SEARCH_SPACE[model]
    keys = list(space)
    out = dict(params)
    for k in RNG.choice(keys, size=min(n, len(keys)), replace=False):
        spec = space[k]
        lo, hi = (spec["low"], spec["high"]) if isinstance(spec, dict) else spec
        cur = out.get(k)
        if cur is None:
            v = RNG.uniform(lo, hi)
        else:
            span = (hi - lo) * 0.35
            v = float(np.clip(RNG.normal(float(cur), span), lo, hi))
        out[k] = int(round(v)) if k in INT_PARAMS else round(float(v), 4)
    return out


def propose_child(tree, parent_id, lineage_id):
    """Rule-based mutation proposer. Rotates through the mutation types that the
    tree-search evidence says matter: boundary push (four comps' largest lever),
    hyperparameter jitter, feature-set switch, feature pruning, seed bagging."""
    parent = next(n for n in tree["nodes"] if n["id"] == parent_id)
    cfg = parent["config"]
    if cfg.get("kind") == "blend":
        return propose_blend_child(tree, cfg)

    model = cfg["model"]
    params = dict(cfg.get("params") or {})
    feats = dict(cfg.get("features") or {"set": "eng"})
    rot = ds(tree).setdefault("rot", {})
    i = rot.get(str(parent_id), 0)
    rot[str(parent_id)] = i + 1

    # 1. boundary push first -- the harness flags params sitting at a search-space edge
    if i % 5 == 0:
        cands = hv3.boundary_candidates(cfg, SEARCH_SPACE[model])
        if cands:
            c = cands[int(RNG.integers(len(cands)))]
            newp = dict(params)
            v = c["new_value"]
            newp[c["param"]] = int(round(v)) if c["param"] in INT_PARAMS else round(float(v), 5)
            return ({**cfg, "params": newp},
                    f"boundary-push {model} {c['param']} {c['old_value']}->{newp[c['param']]} "
                    f"({c['edge']} edge)")

    if i % 5 == 1:
        return ({**cfg, "params": jitter_params(model, params, n=2)},
                f"jitter {model} params")
    if i % 5 == 2:
        alt = [f for f in FEATURE_SETS if f != feats.get("set")]
        return ({**cfg, "features": {"set": str(RNG.choice(alt))}},
                f"feature-set switch from {feats.get('set')}")
    if i % 5 == 3:
        p = jitter_params(model, params, n=3)
        p["seed"] = int(RNG.integers(1, 9999)) if model == "lgb" else None
        if model == "xgb":
            p = {k: v for k, v in p.items() if k != "seed"}
            p["random_state"] = int(RNG.integers(1, 9999))
        elif model == "cat":
            p = {k: v for k, v in p.items() if k != "seed"}
            p["random_seed"] = int(RNG.integers(1, 9999))
        p = {k: v for k, v in p.items() if v is not None}
        return {**cfg, "params": p}, f"seed-bag + jitter {model}"
    return ({**cfg, "params": jitter_params(model, params, n=4)},
            f"wide jitter {model}")


def top_solos(tree, n=6, exclude_models=None):
    rows = [x for x in tree["nodes"]
            if x.get("kind") == "solo" and x.get("status") == "evaluated"
            and x.get("score") is not None]
    if exclude_models:
        rows = [x for x in rows if x["config"].get("model") not in exclude_models]
    rows.sort(key=lambda x: x["score"])
    return [x["id"] for x in rows[:n]]


def propose_blend_child(tree, cfg):
    members = set(cfg.get("members") or [])
    pool = top_solos(tree, n=12)
    add = [m for m in pool if m not in members]
    if add and (len(members) < 8 or RNG.random() < 0.6):
        members.add(add[0])
        return ({"kind": "blend", "members": sorted(members), "k": 800},
                f"blend + member #{add[0]}")
    if len(members) > 2:
        worst = max(members, key=lambda m: next(
            n["score"] for n in tree["nodes"] if n["id"] == m))
        members.discard(worst)
        return ({"kind": "blend", "members": sorted(members), "k": 800},
                f"blend - member #{worst}")
    return ({"kind": "blend", "members": sorted(pool[:4]), "k": 800}, "blend reseed")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def import_round1_oof(tree, name, node_id, expected):
    """OOF cache reuse (checklist item 2): re-import round 1's cached OOF for an
    identical configuration and accept it only on a 6-decimal metric match."""
    sys.path.insert(0, os.path.join(COMP_DIR, "scripts"))
    import features as F
    oof = np.load(os.path.join(OOF_DIR, f"{name}_oof.npy"))
    pred = np.load(os.path.join(OOF_DIR, f"{name}_pred.npy"))
    _X, y, _Xte, _i = F.load("raw")
    got = round(F.mae_pp(oof, y.to_numpy(dtype=np.float64), "round"), 6)
    if got != round(expected, 6):
        raise AssertionError(f"OOF reuse mismatch for {name}: {got} != {expected}")
    hv3.v2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, mae=got)
    return got


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        log(f"resumed tree with {len(tree['nodes'])} nodes")
    else:
        tree = hv3.new_tree(COMP)
        hv3.init_budget(tree, total_budget=TOTAL_BUDGET)

        # ---- root, digit-for-digit verified ----
        res = eval_node(tree, core(ROOT_CONFIG), 0)
        assert res["status"] == "evaluated", res
        assert round(res["score"], 6) == ROOT_EXPECTED, (
            f"root verification FAILED: {res['score']} != {ROOT_EXPECTED}")
        hv3.add_root(tree, "root = linear-iteration best solo (lgb, raw features, L1)",
                     core(ROOT_CONFIG), res["score"], "evaluated", res["wall_s"])
        ds(tree)["node_results"] = {"0": res["result"]}
        ds(tree)["node_names"] = {"0": "lgb_raw_l1"}
        ds(tree)["linear_best_solo"] = ROOT_EXPECTED
        ds(tree)["linear_best_blend"] = 1.334121
        ds(tree)["burst_injected"] = False
        log(f"root verified: {res['score']} == {ROOT_EXPECTED}")

        # ---- generation-1 lineages: round-1 pool re-imported (no retraining) ----
        for name, cfg, expected in ROUND1_SEEDS:
            nid = hv3.v2.v1.next_id(tree)
            got = import_round1_oof(tree, name, nid, expected)
            added, _ = hv3.add_node(tree, 0, f"round-1 pool import: {name}", core(cfg),
                                    got, "evaluated", 0.0)
            ds(tree)["node_names"][str(added)] = name
            ds(tree)["node_results"][str(added)] = {"mae": got, "reused_from": name}
            log(f"  imported #{added} {name} mae={got}")

        # ---- first blend lineage ----
        add_and_eval(tree, 0, "blend seed: whole round-1 pool",
                     {"kind": "blend", "members": sorted(top_solos(tree, n=99)), "k": 800},
                     name="blend_all")
        hv3.save_search_state(tree, TREE_PATH)

    # ---- main loop ----
    while not hv3.should_stop(tree):
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not ds(tree).get("burst_injected"):
            log("=== EXPLORE BURST ===")
            ds(tree)["burst_injected"] = True
            gb = hv3.global_best(tree)
            burst = [
                ("kitchen-sink blend of the entire solo pool",
                 {"kind": "blend", "members": sorted(top_solos(tree, n=99)), "k": 1500}),
                ("kitchen-sink blend of the top-8 solos",
                 {"kind": "blend", "members": sorted(top_solos(tree, n=8)), "k": 1500}),
                ("long-shot: very deep LGB, tiny lr, many rounds",
                 {"kind": "solo", "model": "lgb",
                  "params": {"learning_rate": 0.008, "num_leaves": 255,
                             "min_child_samples": 10, "feature_fraction": 0.6,
                             "num_boost_round": 12000, "early_stopping_rounds": 300},
                  "features": {"set": "core"}}),
                ("long-shot: extremely shallow, heavily regularized LGB (decorrelator)",
                 {"kind": "solo", "model": "lgb",
                  "params": {"learning_rate": 0.05, "num_leaves": 7,
                             "min_child_samples": 200, "reg_alpha": 5.0,
                             "reg_lambda": 10.0, "feature_fraction": 0.5},
                  "features": {"set": "full"}}),
                ("long-shot: XGB deep + low min_child_weight on raw features",
                 {"kind": "solo", "model": "xgb",
                  "params": {"max_depth": 11, "min_child_weight": 2,
                             "learning_rate": 0.02, "colsample_bytree": 0.6},
                  "features": {"set": "raw"}}),
                ("long-shot: CatBoost deep, low lr, full features",
                 {"kind": "solo", "model": "cat",
                  "params": {"depth": 9, "learning_rate": 0.03, "l2_leaf_reg": 6.0},
                  "features": {"set": "full"}}),
            ]
            for mut, cfg in burst:
                if hv3.should_stop(tree):
                    break
                nid, res = add_and_eval(tree, 0, "[burst] " + mut, cfg)
                if nid is not None and cfg.get("kind") == "solo":
                    passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
                    if not passed:
                        log(f"  burst sanity gate FIRED on #{nid} (bound {bound})")
                        ds(tree).setdefault("sanity_gate_fired", []).append(
                            {"node": nid, "bound": bound})
                hv3.save_search_state(tree, TREE_PATH)
            log(f"burst done; global best {gb['score'] if gb else None} -> "
                f"{hv3.global_best(tree)['score']}")
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            log("no expandable parent left")
            break
        cfg, mutation = propose_child(tree, parent_id, lineage_id)
        add_and_eval(tree, parent_id, mutation, cfg)
        hv3.save_search_state(tree, TREE_PATH)
        if time.time() - t_start > 3600:
            log("driver wall-clock guard (60 min) hit -- stopping early")
            break

    # ---- honest report ----
    gb = hv3.global_best(tree)
    st = tree["search_state"]
    evaluated = [n for n in tree["nodes"] if n.get("status") == "evaluated"]
    lin_solo = ds(tree)["linear_best_solo"]
    lin_blend = ds(tree)["linear_best_blend"]
    order = sorted(evaluated, key=lambda n: n["id"])
    evals_to_match = None
    for i, n in enumerate(order, start=1):
        if n["score"] is not None and n["score"] <= lin_blend:
            evals_to_match = i
            break
    report = {
        "n_nodes": len(tree["nodes"]), "n_evaluated": len(evaluated),
        "global_best": {"id": gb["id"], "score": gb["score"],
                        "config": gb["config"],
                        "result": ds(tree).get("node_results", {}).get(str(gb["id"]))},
        "linear_best_solo": lin_solo, "linear_best_blend": lin_blend,
        "evals_to_match_linear_best_blend": evals_to_match,
        "phase": st["budget"]["phase"],
        "plateaued_lineages": st.get("plateaued", []),
        "backtrack_log": st.get("backtrack_log", []),
        "dedup_rejections": st.get("dedup_streak", {}),
        "cost_guard_log": st.get("cost_guard_log", []),
        "sanity_gate_fired": ds(tree).get("sanity_gate_fired", []),
        "wall_s": round(time.time() - t_start, 1),
        "top10": [{"id": n["id"], "score": n["score"], "kind": n.get("kind"),
                   "mutation": n["mutation"][:80]}
                  for n in sorted([e for e in evaluated if e["score"] is not None],
                                  key=lambda x: x["score"])[:10]],
    }
    with open(os.path.join(COMP_DIR, "tree_search_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=hv3._json_default)
    hv3.save_search_state(tree, TREE_PATH)
    log(json.dumps({k: report[k] for k in
                    ("n_evaluated", "linear_best_solo", "linear_best_blend",
                     "evals_to_match_linear_best_blend", "phase", "wall_s")}, indent=2))
    log(f"GLOBAL BEST #{gb['id']} score={gb['score']} config={gb['config']}")


if __name__ == "__main__":
    main()
