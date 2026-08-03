"""tree_search/run_s6e2_lane3.py — Stage 4 tree-search driver for playground-series-s6e2.

Follows references/07_tree_search.md §4 (driver checklist):
  1. root digit-for-digit verification against the linear iteration's best solo (cat_d4,
     OOF AUC 0.955540 from experiments.json #15) -- aborts on mismatch;
  2. OOF cache reuse of the 15 linear-iteration arms, accepted only on a 6-dp recompute
     match (implemented in eval_s6e2_lane3._reuse);
  3. all driver bookkeeping lives in tree["search_state"]["driver_state"], never in
     module globals, and is persisted via hv3.save_search_state;
  4. every solo evaluation goes through hv3.eval_solo_subprocess (OS-level timeout --
     signal.alarm cannot interrupt a native fit());
  5. hv3.apply_burst_seed_sanity_gate is called on every explore-burst seed.

BUDGET OVERRIDE: total_budget=40 rather than the harness default 60, with reason. A
CatBoost solo on this 630k x 13 dataset costs ~495s per 5-fold set (measured,
experiments.json #14/#15) against ~25-80s for LightGBM/XGBoost, and CatBoost is the
family holding the current best solo -- so the node space cannot simply exclude it. 40
nodes keeps the worst realistic mix inside this run's wall-clock safety net; the
reduction is recorded here and in STATUS.md rather than applied silently.

Earlier s6e2 tree-search artifacts exist in this directory (run_s6e2_v3.py,
run_s6e2_run2.py, cache_s6e2*, wire_s6e2_*.json). They are previous runs of the same
competition and were deliberately not opened; this driver, its evaluator and its cache
carry the `_lane3` suffix to stay disjoint.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_s6e2_lane3", os.path.join(_HERE, "eval_s6e2_lane3.py"))
EV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EV)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COMP = "playground-series-s6e2"
TREE_PATH = os.path.join(_HERE, "tree_s6e2_lane3.json")
EVAL_PATH = os.path.join(_HERE, "eval_s6e2_lane3.py")
RANK_CACHE = os.path.join(_HERE, "cache_s6e2_lane3_rank")

TOTAL_BUDGET = 26  # lowered from 40 mid-run (see STATUS.md): CatBoost + blend node costs
TIMEOUT = {"lgb": 400.0, "xgb": 400.0, "cat": 1800.0}  # lgb/xgb lowered from 900s: the DART long-shot does not converge inside any sane window
# Driver-level wall-clock cap (validate_state explicitly permits a driver stopping before
# the phase machine would): this run's overall safety net is 4h and the linear iteration
# already consumed ~45min, so the search gets 100 minutes. If it fires, that is reported
# as the stop reason -- never silently.
WALL_CLOCK_CAP_S = 6000.0
# CatBoost costs ~490s per node here vs ~25-80s for LGB/XGB (measured: node #1 took 489s).
# Without a cap one family would consume the whole budget, so CatBoost solo evaluations are
# capped; once the cap is hit, cat lineages stop proposing children and plateau. Reported,
# not silent (07_tree_search.md §6 "no silent caps").
CAT_EVAL_CAP = 5  # lowered from 6 mid-run: 5 CatBoost solos already cost ~48 min of wall clock
_T0 = time.time()

# Linear-iteration reference points (competitions/playground-series-s6e2/experiments.json)
ROOT_ARM_AUC = 0.955540          # cat_d4, exp #15 -- best solo
LINEAR_BEST_FITTED = 0.955554    # exp #16 blend, weights fit on the whole OOF
LINEAR_BEST_HONEST = 0.955553    # exp #16 blend, leave-one-fold-out weight refit

ROOT_CFG = {"kind": "solo", "model": "cat", "variant": "raw", "cat_declare": True,
            "params": {"depth": 4, "learning_rate": 0.06}}

SEEDS = [
    ("cat_lineage", {"kind": "solo", "model": "cat", "variant": "raw", "cat_declare": True,
                     "params": {"depth": 5, "learning_rate": 0.06}}),
    ("lgb_lineage", {"kind": "solo", "model": "lgb", "variant": "raw", "cat_declare": False,
                     "params": {"num_leaves": 8, "lambda_l2": 3.0}}),
    ("xgb_lineage", {"kind": "solo", "model": "xgb", "variant": "raw", "cat_declare": False,
                     "params": {"max_depth": 4, "min_child_weight": 50, "learning_rate": 0.04}}),
]

PRIOR_META = {"metric": "roc-auc", "data_type": "tabular",
              "tags": ["ensemble", "cv 設計", "特徵工程", "超參調校", "反面教訓"],
              "comp": COMP}


# ---------------------------------------------------------------------------
# driver state helpers (checklist item 3: everything lives inside the tree)
# ---------------------------------------------------------------------------
def dstate(tree: dict) -> dict:
    return tree["search_state"].setdefault("driver_state", {
        "lineage_names": {}, "burst_injected": False, "results": {},
        "cat_evals": 0, "attempts": {}, "eval_order": [],
        "dedup_rejections": 0, "failed_nodes": [], "burst_sanity_log": [],
    })


def record(tree: dict, nid: int, name: str, auc: float, extra: dict = None) -> None:
    ds = dstate(tree)
    ds["results"][str(nid)] = dict(name=name, auc=round(float(auc), 6), **(extra or {}))
    ds["eval_order"].append(nid)


def solo_nodes(tree: dict) -> list:
    return [n for n in tree["nodes"]
            if n["status"] == "evaluated" and n.get("kind") == "solo"]


def best_solos(tree: dict, k: int) -> list:
    return [n["id"] for n in sorted(solo_nodes(tree), key=lambda n: n["score"])[:k]]


# ---------------------------------------------------------------------------
# mutation proposal
# ---------------------------------------------------------------------------
def _param_variants(model: str, params: dict) -> list:
    """Deterministic ordered list of (mutation_label, new_params) for a solo config."""
    out = []
    p = dict(params)

    if model == "lgb":
        nl = p.get("num_leaves", 64)
        out += [("num_leaves x0.5", {**p, "num_leaves": max(4, int(nl // 2))}),
                ("num_leaves x2", {**p, "num_leaves": min(255, int(nl * 2))}),
                ("min_child_samples 300", {**p, "min_child_samples": 300}),
                ("min_child_samples 20", {**p, "min_child_samples": 20}),
                ("lambda_l2 10", {**p, "lambda_l2": 10.0}),
                ("lambda_l1 1.0", {**p, "lambda_l1": 1.0}),
                ("feature_fraction 1.0", {**p, "feature_fraction": 1.0}),
                ("feature_fraction 0.5", {**p, "feature_fraction": 0.5}),
                ("bagging_fraction 1.0", {**p, "bagging_fraction": 1.0}),
                ("lr 0.02", {**p, "learning_rate": 0.02}),
                ("max_bin 511", {**p, "max_bin": 511}),
                ("seed-bag 2024", {**p, "seed": 2024, "bagging_seed": 2024,
                                   "feature_fraction_seed": 2024}),
                ("seed-bag 7", {**p, "seed": 7, "bagging_seed": 7,
                                "feature_fraction_seed": 7}),
                ("extra_trees", {**p, "extra_trees": True}),
                ("path_smooth 10", {**p, "path_smooth": 10.0}),
                ("min_child_samples 600", {**p, "min_child_samples": 600}),
                ]
    elif model == "xgb":
        d = p.get("max_depth", 8)
        out += [("max_depth-1", {**p, "max_depth": max(3, d - 1)}),
                ("max_depth+1", {**p, "max_depth": min(10, d + 1)}),
                ("min_child_weight 150", {**p, "min_child_weight": 150}),
                ("reg_lambda 10", {**p, "reg_lambda": 10.0}),
                ("colsample 0.6", {**p, "colsample_bytree": 0.6}),
                ("subsample 1.0", {**p, "subsample": 1.0}),
                ("lr 0.02", {**p, "learning_rate": 0.02}),
                ("seed-bag 2024", {**p, "seed": 2024}),
                ("reg_alpha 1.0", {**p, "reg_alpha": 1.0}),
                ("max_bin 512", {**p, "max_bin": 512}),
                ]
    else:  # cat
        d = p.get("depth", 6)
        out += [("depth-1", {**p, "depth": max(3, d - 1)}),
                ("depth+1", {**p, "depth": min(8, d + 1)}),
                ("l2_leaf_reg 10", {**p, "l2_leaf_reg": 10.0}),
                ("lr 0.03", {**p, "learning_rate": 0.03}),
                ("one_hot_max_size 12", {**p, "one_hot_max_size": 12}),
                ("rsm 0.8", {**p, "rsm": 0.8}),
                ("seed-bag 2024", {**p, "random_seed": 2024}),
                ("l2_leaf_reg 1.0", {**p, "l2_leaf_reg": 1.0}),
                ]
    return out


def propose_solo_child(tree: dict, parent: dict, attempt: int):
    """Boundary pushes first (harness feature 4), then the deterministic variant list."""
    cfg = parent["config"]
    model = cfg["model"]
    if model == "cat" and dstate(tree)["cat_evals"] >= CAT_EVAL_CAP:
        return None, None
    cands = hv3.boundary_candidates(cfg, EV.SEARCH_SPACE[model])
    proposals = [(f"boundary-push {c['param']} {c['edge']} "
                  f"{c['old_value']}->{c['new_value']}",
                  {**cfg["params"], c["param"]: c["new_value"]}) for c in cands]
    proposals += _param_variants(model, cfg["params"])
    # feature-space mutations, tried after the hyperparameter ones
    proposals += [("cat_declare flip", None), ("variant raw_pruned", None)]

    if attempt >= len(proposals):
        return None, None
    label, new_params = proposals[attempt]
    child = copy.deepcopy(cfg)
    if new_params is None:
        if label == "cat_declare flip":
            child["cat_declare"] = not child.get("cat_declare", False)
        else:
            child["variant"] = "raw_pruned"
    else:
        child["params"] = new_params
    return EV.core(child), label


def propose_blend_child(tree: dict, parent: dict, attempt: int):
    """Blend lineage mutations: widen the member pool, then flip the averaging space."""
    pool = best_solos(tree, 12)
    sizes = [3, 4, 5, 6, 8, 10, 12]
    spaces = ["rank", "prob"]
    combos = [(s, sp) for s in sizes for sp in spaces]
    if attempt >= len(combos):
        return None, None
    size, space = combos[attempt]
    members = sorted(pool[:size])
    if len(members) < 2:
        return None, None
    cfg = {"kind": "blend", "members": members, "space": space,
           "weight_search": "dirichlet"}
    return EV.core(cfg), f"blend top-{len(members)} in {space} space"


# ---------------------------------------------------------------------------
# evaluation dispatch
# ---------------------------------------------------------------------------
def _rank_cache_members(members: list) -> str:
    """Materialise rank-transformed copies of each member's cached OOF/pred so that the
    harness's own eval_blend can be reused verbatim for rank-space blending (ROC-AUC is
    a pure rank metric, so this is the natural averaging space)."""
    os.makedirs(RANK_CACHE, exist_ok=True)
    for m in members:
        src = os.path.join(EV.CACHE_DIR, f"solo_{m}.npz")
        dst = os.path.join(RANK_CACHE, f"solo_{m}.npz")
        if os.path.exists(dst):
            continue
        d = np.load(src)
        oof, pred = d["oof"], d["pred"]
        hv2.cache_oof(RANK_CACHE, m,
                      np.argsort(np.argsort(oof)) / len(oof),
                      pred=np.argsort(np.argsort(pred)) / len(pred))
    return RANK_CACHE


def eval_node(tree: dict, cfg: dict, nid_hint: int):
    """Returns (score, status, wall_s, result_dict)."""
    if cfg["kind"] == "blend":
        cache = _rank_cache_members(cfg["members"]) if cfg["space"] == "rank" else EV.CACHE_DIR
        w, s, _oofs, warn = hv3.eval_blend_with_cost_guard(
            cache, cfg["members"], EV.auc_of, tree=tree)
        if warn:
            log.warning("cost guard: %s", warn)
        return s, "evaluated", 0.0, {"auc": -s, "weights": [round(float(x), 4) for x in w],
                                     "members": cfg["members"], "space": cfg["space"],
                                     "cost_guard_warning": warn}
    r = hv3.eval_solo_subprocess(EVAL_PATH, cfg, timeout_s=TIMEOUT[cfg["model"]],
                                 node_id=nid_hint)
    return r["score"], r["status"], r.get("wall_s", 0.0), (r.get("result") or
                                                           {"error": r.get("error")})


def eval_and_add(tree: dict, parent_id: int, label: str, cfg: dict, name: str):
    # Dedup BEFORE evaluating. The first version of this driver evaluated first and let
    # hv3.add_node reject afterwards, which on this competition means paying a full ~600s
    # CatBoost 5-fold training for a config the tree already holds (it burned 3 such
    # trainings, ~25 minutes, before the bug was caught). hv3.add_node is still called on
    # the duplicate so that feature 2's dedup-consumes-budget placeholder bookkeeping runs
    # exactly as it would have -- only the wasted training is skipped.
    if hv2.find_duplicate_config(tree, cfg) is not None:
        _nid, dup = hv3.add_node(tree, parent_id, label, cfg, None, "evaluated", 0.0)
        log.info("  dedup-rejected BEFORE evaluation (identical to node #%s): %s", dup, label)
        dstate(tree)["dedup_rejections"] += 1
        hv3.save_search_state(tree, TREE_PATH)
        return None
    if cfg.get("model") == "cat":
        # count the TRAINING, not the successful add, so the cap actually bounds compute
        dstate(tree)["cat_evals"] += 1
    nid_hint = hv2.v1._next_id(tree)
    score, status, wall, res = eval_node(tree, cfg, nid_hint)
    if status != "evaluated" or score is None:
        log.warning("  node %s FAILED (%s): %s", nid_hint, label, res)
        dstate(tree)["failed_nodes"].append({"label": label, "detail": str(res)[:400]})
        hv3.add_node(tree, parent_id, label, cfg, None, "failed", wall)
        hv3.save_search_state(tree, TREE_PATH)
        return None
    nid, dup = hv3.add_node(tree, parent_id, label, cfg, score, "evaluated", wall)
    if nid is None:
        log.info("  dedup-rejected (identical to node #%s): %s", dup, label)
        dstate(tree)["dedup_rejections"] += 1
        hv3.save_search_state(tree, TREE_PATH)
        return None
    record(tree, nid, name, -score, {"wall_s": wall, "detail": res})
    gb = hv2.v1.global_best(tree)
    log.info("  node #%-3d %-52s AUC %.6f  (%.0fs)%s", nid, label, -score, wall,
             "  <-- NEW GLOBAL BEST" if gb["id"] == nid else "")
    hv3.save_search_state(tree, TREE_PATH)
    return nid


# ---------------------------------------------------------------------------
# explore burst
# ---------------------------------------------------------------------------
def inject_burst(tree: dict) -> None:
    root = tree["root_id"]
    log.info("=== EXPLORE BURST (mandatory) ===")
    # 1. kitchen-sink blend of the ENTIRE current solo pool, both spaces.
    allsolo = sorted(n["id"] for n in solo_nodes(tree))
    for space in ("rank", "prob"):
        cfg = EV.core({"kind": "blend", "members": allsolo, "space": space})
        nid = eval_and_add(tree, root, f"[burst] kitchen-sink blend of all {len(allsolo)} "
                                       f"solos ({space} space)", cfg, f"burst_ks_{space}")
        if nid is not None:
            passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
            dstate(tree)["burst_sanity_log"].append(
                {"node": nid, "passed": bool(passed), "bound": bound})
            log.info("    burst sanity gate: passed=%s bound=%s", passed, bound)
    # 2. long-shot solo lineages in contrasting directions.
    longshots = [
        ("[burst] LGB dart boosting (contrarian regulariser)",
         {"kind": "solo", "model": "lgb", "variant": "raw", "cat_declare": False,
          "params": {"boosting": "dart", "num_leaves": 16, "learning_rate": 0.05,
                     "lambda_l2": 3.0, "drop_rate": 0.1}}),
        ("[burst] LGB extra_trees + very low lr, heavy smoothing",
         {"kind": "solo", "model": "lgb", "variant": "raw", "cat_declare": False,
          "params": {"extra_trees": True, "num_leaves": 32, "learning_rate": 0.015,
                     "lambda_l2": 5.0, "path_smooth": 20.0, "min_child_samples": 200}}),
        ("[burst] XGB deep + very strong shrinkage (opposite capacity corner)",
         {"kind": "solo", "model": "xgb", "variant": "raw", "cat_declare": False,
          "params": {"max_depth": 9, "min_child_weight": 250, "reg_lambda": 15.0,
                     "learning_rate": 0.02, "colsample_bytree": 0.5}}),
        # Originally a CatBoost depth-3 one-hot seed; replaced mid-run because a CatBoost
        # node costs ~600s here and the depth-3 corner was already covered by node #6.
        # Substituted with the same "declare the 8 nominal integer codes categorical"
        # idea at the winning LightGBM capacity, which is ~40x cheaper.
        ("[burst] LGB num_leaves 4 with the 8 nominal codes declared categorical",
         {"kind": "solo", "model": "lgb", "variant": "raw", "cat_declare": True,
          "params": {"num_leaves": 4, "lambda_l2": 3.0}}),
        ("[burst] LGB on the domain feature set at the winning capacity",
         {"kind": "solo", "model": "lgb", "variant": "domain", "cat_declare": False,
          "params": {"num_leaves": 8, "lambda_l2": 3.0}}),
    ]
    for label, cfg in longshots:
        nid = eval_and_add(tree, root, label, EV.core(cfg), label)
        if nid is not None:
            passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
            dstate(tree)["burst_sanity_log"].append(
                {"node": nid, "passed": bool(passed), "bound": bound})
            log.info("    burst sanity gate: passed=%s bound=%s", passed, bound)
        if hv3.should_stop(tree):
            return
    dstate(tree)["burst_injected"] = True
    hv3.save_search_state(tree, TREE_PATH)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    if os.path.exists(TREE_PATH):
        tree = hv3.load_search_state(TREE_PATH)
        log.info("resumed tree with %d evaluated nodes", hv3.n_evaluated(tree))
    else:
        tree = hv2.new_tree(COMP)
    hv3.init_budget(tree, total_budget=TOTAL_BUDGET)
    ds = dstate(tree)

    priors = hv2.suggest_priors(PRIOR_META)
    log.info("suggest_priors returned %d evidence-cited bullets (self-citing bullets "
             "dropped by the harness via comp_meta['comp'])", len(priors))
    tree.setdefault("priors_snapshot", priors[:20])

    # --- root, with digit-for-digit verification (checklist item 1) ---
    if tree["root_id"] is None:
        r = hv3.eval_solo_subprocess(EVAL_PATH, EV.core(ROOT_CFG), timeout_s=TIMEOUT["cat"],
                                     node_id=0)
        assert r["status"] == "evaluated", r
        auc = r["result"]["auc"]
        assert round(auc, 6) == round(ROOT_ARM_AUC, 6), (
            f"ROOT VERIFICATION FAILED: got {auc:.6f}, expected {ROOT_ARM_AUC:.6f} "
            f"(linear iteration experiments.json #15, cat_d4)")
        hv2.add_root(tree, "root = linear-iteration best solo (cat_d4)", EV.core(ROOT_CFG),
                     r["score"], "evaluated", r["wall_s"])
        record(tree, 0, "root_cat_d4", auc, {"reused": r["result"].get("reused")})
        log.info("root #0 verified digit-for-digit: AUC %.6f == %.6f", auc, ROOT_ARM_AUC)
        hv3.save_search_state(tree, TREE_PATH)

    # --- first-generation lineages ---
    if len(tree["nodes"]) == 1:
        for name, cfg in SEEDS:
            eval_and_add(tree, 0, f"seed lineage: {name}", EV.core(cfg), name)
        members = sorted(best_solos(tree, 4))
        if len(members) >= 2:
            eval_and_add(tree, 0, f"seed lineage: blend of top {len(members)} solos (rank)",
                         EV.core({"kind": "blend", "members": members, "space": "rank"}),
                         "blend_lineage")

    # --- main loop ---
    while not hv3.should_stop(tree):
        if time.time() - _T0 > WALL_CLOCK_CAP_S:
            b = tree["search_state"]["budget"]
            b["phase"] = "stopped"
            b["stop_reason"] = (f"driver wall-clock cap of {WALL_CLOCK_CAP_S:.0f}s reached "
                                f"after {hv3.n_evaluated(tree)} evaluated nodes (the phase "
                                f"machine had not yet stopped)")
            log.warning("WALL-CLOCK CAP: %s", b["stop_reason"])
            hv3.save_search_state(tree, TREE_PATH)
            break
        phase = hv3.update_phase(tree)
        if phase == "stopped":
            break
        if phase == "explore_burst" and not ds["burst_injected"]:
            inject_burst(tree)
            continue
        parent_id, lineage_id = hv2.select_next_parent(tree)
        if parent_id is None:
            log.info("select_next_parent exhausted the tree; stopping")
            break
        parent = tree["nodes"][parent_id]
        key = str(parent_id)
        attempt = ds["attempts"].get(key, 0)
        if parent.get("kind") == "blend":
            cfg, label = propose_blend_child(tree, parent, attempt)
        else:
            cfg, label = propose_solo_child(tree, parent, attempt)
        ds["attempts"][key] = attempt + 1
        if cfg is None:
            log.info("  parent #%s exhausted its proposal list; marking lineage plateaued",
                     parent_id)
            pl = tree["search_state"].setdefault("plateaued", [])
            if lineage_id is not None and lineage_id not in pl:
                pl.append(lineage_id)
            hv3.update_phase(tree)
            hv3.save_search_state(tree, TREE_PATH)
            continue
        eval_and_add(tree, parent_id, label, cfg, f"n{parent_id}+{attempt}")

    # --- honest report ---
    gb = hv2.v1.global_best(tree)
    n_eval = hv3.n_evaluated(tree)
    order = ds["eval_order"]
    evals_to_match = None
    running = None
    for i, nid in enumerate(order, start=1):
        a = ds["results"][str(nid)]["auc"]
        running = a if running is None else max(running, a)
        if evals_to_match is None and running > LINEAR_BEST_FITTED:
            evals_to_match = i
    report = {
        "n_evaluated": n_eval,
        "budget": tree["search_state"]["budget"],
        "global_best_node": gb["id"],
        "global_best_auc": round(-gb["score"], 6),
        "global_best_config": gb["config"],
        "linear_best_fitted": LINEAR_BEST_FITTED,
        "linear_best_honest": LINEAR_BEST_HONEST,
        "root_auc": ROOT_ARM_AUC,
        "evals_to_beat_linear_best_fitted": evals_to_match,
        "backtrack_log": tree["search_state"].get("backtrack_log", []),
        "plateaued": tree["search_state"].get("plateaued", []),
        "dedup_rejections": ds["dedup_rejections"],
        "failed_nodes": ds["failed_nodes"],
        "cost_guard_log": tree["search_state"].get("cost_guard_log", []),
        "burst_sanity_log": ds["burst_sanity_log"],
        "cat_evals": ds["cat_evals"],
        "all_results": ds["results"],
    }
    with open(os.path.join(_HERE, "report_s6e2_lane3.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    hv3.save_search_state(tree, TREE_PATH)

    log.info("\n=== TREE SEARCH DONE ===")
    log.info("evaluated %d nodes; global best = node #%d, AUC %.6f",
             n_eval, gb["id"], -gb["score"])
    log.info("linear-iteration best (fitted) %.6f -> delta %+.6f",
             LINEAR_BEST_FITTED, -gb["score"] - LINEAR_BEST_FITTED)
    log.info("evals to beat the linear best: %s", evals_to_match)
    log.info("plateaued lineages: %s", report["plateaued"])
    log.info("cost-guard triggers: %d; burst-sanity records: %d",
             len(report["cost_guard_log"]), len(report["burst_sanity_log"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
