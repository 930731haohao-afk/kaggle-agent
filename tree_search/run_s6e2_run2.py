"""tree_search/run_s6e2_run2.py — Stage 4 tree-search driver for playground-series-s6e2.

LANE ISOLATION: tree_search/ already holds eval_s6e2.py / run_s6e2_v3.py / cache_s6e2/
from a PREVIOUS run of this same competition. This run may not read or reuse them, so
everything here is suffixed `_run2` and caches to tree_search/cache_s6e2_run2/. No older
artifact is read.

Node space: the harness's general-tabular DEFAULT — solo + blend dual track
(references/07_tree_search.md §2 row 1).

The five driver-checklist items (§4) are all wired:
  1. digit-for-digit root verification — the root is bit-identical to the linear run's
     best solo (LGB_tuned, OOF AUC 0.955548) and the driver aborts unless the re-evaluated
     root reproduces that to 6 dp.
  2. OOF cache reuse — not used; the root is re-trained from scratch precisely so item 1
     is a real check rather than a tautology, and every other node is new.
  3. resume-state contract — all driver bookkeeping lives in
     tree["search_state"]["driver_state"], persisted via hv3.save_search_state.
  4. subprocess-level eval timeout — every solo eval goes through hv3.eval_solo_subprocess.
  5. burst-seed sanity gate — hv3.apply_burst_seed_sanity_gate after every burst seed.

Budget: hv3 defaults (60 nodes / burst 6 / patience 20) PLUS a wall-clock guard. The
wall guard exists because one CatBoost 5-fold on this 630k-row dataset costs ~374s, so a
CatBoost-heavy tail could otherwise overrun the session budget. Any wall-clock stop is
recorded explicitly in driver_state["wall_stop"] and reported — never silent.
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

COMP = "playground-series-s6e2"
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", COMP)
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_run2.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_s6e2_run2.py")
CACHE_DIR = os.path.join(_HERE, "cache_s6e2_run2")
os.makedirs(CACHE_DIR, exist_ok=True)

EVAL_TIMEOUT_S = 900
WALL_BUDGET_S = float(os.environ.get("S6E2_WALL_BUDGET_S", 6000))
TOTAL_BUDGET = int(os.environ.get("S6E2_TOTAL_BUDGET", 60))

# --- linear-iteration anchors (competitions/.../experiments.json) -------------
ROOT_AUC = 0.955548          # LGB_tuned, exp #9 — the linear best SOLO
LINEAR_BEST_FITTED = 0.955568   # linear best blend, weights fitted on the FULL OOF
LINEAR_BEST_HONEST = 0.955566   # linear best blend, leave-fold-out weights (the honest one)

LGB_TUNED_PARAMS = {
    "learning_rate": 0.03674659906666914,
    "num_leaves": 8,
    "max_depth": 5,
    "min_child_samples": 32,
    "feature_fraction": 0.5730250156949507,
    "bagging_fraction": 0.8066114470992117,
    "reg_alpha": 1.0471996992578287,
    "reg_lambda": 0.027660171185883174,
    "min_split_gain": 0.8017242933598588,
}
# the Optuna box the tuned params came from — boundary_candidates needs it to know which
# hyperparameters are sitting on an edge and deserve a push OUTSIDE the box
LGB_SEARCH_SPACE = {
    "learning_rate": (0.01, 0.15),
    "num_leaves": (8, 512),
    "max_depth": (3, 14),
    "min_child_samples": (10, 400),
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
    "reg_alpha": (1e-4, 20.0),
    "reg_lambda": (1e-4, 20.0),
    "min_split_gain": (1e-6, 1.0),
}
XGB_PARAMS = {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 8,
              "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5,
              "reg_lambda": 2.0}
CAT_PARAMS = {"learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 6.0,
              "bootstrap_type": "Bernoulli", "subsample": 0.8, "iterations": 1600}

T0 = time.time()


# ---------------------------------------------------------------------------
# driver state (checklist item 3 — never module-level globals)
# ---------------------------------------------------------------------------
def dstate(tree):
    return tree.setdefault("search_state", {}).setdefault("driver_state", {
        "names": {}, "results": {}, "burst_injected": False, "queue_offset": {},
        "wall_stop": None, "evals_to_match_linear_fitted": None,
        "evals_to_match_linear_honest": None, "log": [],
    })


def core(cfg):
    """Canonical stored form: drop output-only fields, sort blend members."""
    c = {k: v for k, v in cfg.items() if k not in ("result",)}
    if c.get("kind") == "blend":
        c["members"] = sorted(c["members"])
    return c


def auc_of(tree, nid):
    return dstate(tree)["results"].get(str(nid), {}).get("auc")


def log(tree, msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)
    dstate(tree)["log"].append(msg)


# ---------------------------------------------------------------------------
# eval + add
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, cfg, *, is_root=False, name=None):
    stored = core(cfg)
    if not is_root:
        dup = hv3.find_duplicate_config(tree, stored)
        if dup is not None:
            hv3.add_node(tree, parent_id, mutation + " [dedup pre-check]", stored,
                         None, "failed", 0.0)
            hv3.save_search_state(tree, TREE_PATH)
            log(tree, f"  dedup-rejected (identical to node #{dup}): {mutation}")
            return None

    nid = hv3.next_id(tree)
    if stored.get("kind") == "blend":
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ev_s6e2", EVAL_MODULE_PATH)
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        r = ev.evaluate(stored, node_id=None, timeout_s=EVAL_TIMEOUT_S)
    else:
        r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, stored, EVAL_TIMEOUT_S, node_id=nid)

    score, status, wall = r.get("score"), r.get("status"), r.get("wall_s", 0.0)
    if is_root:
        real = hv3.add_root(tree, mutation, stored, score, status, wall)
    else:
        real, _ = hv3.add_node(tree, parent_id, mutation, stored, score, status, wall)
    if real is None:
        hv3.save_search_state(tree, TREE_PATH)
        return None

    ds = dstate(tree)
    ds["results"][str(real)] = r.get("result") or {}
    if name:
        ds["names"][str(real)] = name
    a = (r.get("result") or {}).get("auc")
    log(tree, f"node #{real:>2} {status:<9} AUC={a if a is not None else 'FAIL':<10} "
              f"{wall:6.1f}s  {mutation}"
              + (f"  ERR={r.get('error')}" if r.get("error") else ""))

    # evals-to-beat bookkeeping (mechanical, per §6)
    n_ev = hv3.n_evaluated(tree)
    if a is not None:
        if ds["evals_to_match_linear_fitted"] is None and a >= LINEAR_BEST_FITTED:
            ds["evals_to_match_linear_fitted"] = n_ev
        if ds["evals_to_match_linear_honest"] is None and a >= LINEAR_BEST_HONEST:
            ds["evals_to_match_linear_honest"] = n_ev
    hv3.save_search_state(tree, TREE_PATH)
    return real


# ---------------------------------------------------------------------------
# solo mutation queue
# ---------------------------------------------------------------------------
def _bump(cfg, **kw):
    c = json.loads(json.dumps(cfg))
    p = c.setdefault("params", {})
    for k, v in kw.items():
        if k in ("features", "cat_native", "seed", "model"):
            c[k] = v
        else:
            p[k] = v
    return c


def solo_mutations(tree, parent_cfg, idx):
    """Ordered mutation queue for a solo lineage. idx cycles as the lineage grows."""
    p = parent_cfg.get("params", {})
    model = parent_cfg.get("model", "lgb")
    muts = []
    if model == "lgb":
        muts = [
            ("lr down 0.6x (more trees, finer steps)",
             _bump(parent_cfg, learning_rate=p.get("learning_rate", 0.05) * 0.6)),
            ("num_leaves up 2x (capacity probe)",
             _bump(parent_cfg, num_leaves=int(p.get("num_leaves", 31) * 2))),
            ("min_child_samples up 3x (regularise)",
             _bump(parent_cfg, min_child_samples=int(p.get("min_child_samples", 20) * 3))),
            ("feature_fraction down to 0.4 (decorrelate splits)",
             _bump(parent_cfg, feature_fraction=0.4)),
            ("min_split_gain -> 0 (remove the split penalty)",
             _bump(parent_cfg, min_split_gain=1e-6)),
            ("reg_lambda up to 5 (L2)",
             _bump(parent_cfg, reg_lambda=5.0)),
            ("native categoricals ON",
             _bump(parent_cfg, cat_native=not parent_cfg.get("cat_native", False))),
            ("max_depth up 2 (deeper, leaves unchanged)",
             _bump(parent_cfg, max_depth=int(p.get("max_depth", 5)) + 2)),
            ("seed bag 2024 (variance reduction)",
             _bump(parent_cfg, seed=2024)),
            ("feature version v1 (clinical ratios) retry under these params",
             _bump(parent_cfg, features={"version": "v1", "drop": []})),
            ("drop the two near-chance raw columns (BP, Cholesterol)",
             _bump(parent_cfg, features={"version": parent_cfg.get("features", {}).get("version", "raw"),
                                         "drop": ["BP", "Cholesterol"]})),
            ("bagging_fraction down to 0.6",
             _bump(parent_cfg, bagging_fraction=0.6)),
            ("seed bag 7 (second variance-reduction draw)",
             _bump(parent_cfg, seed=7)),
        ]
    elif model == "xgb":
        muts = [
            ("lr down 0.5x", _bump(parent_cfg, learning_rate=p.get("learning_rate", 0.05) * 0.5)),
            ("max_depth down to 4 (regularise)", _bump(parent_cfg, max_depth=4)),
            ("min_child_weight up 4x",
             _bump(parent_cfg, min_child_weight=p.get("min_child_weight", 8) * 4)),
            ("colsample down to 0.5", _bump(parent_cfg, colsample_bytree=0.5)),
            ("reg_lambda up to 8", _bump(parent_cfg, reg_lambda=8.0)),
            ("max_depth up to 8 (capacity probe)", _bump(parent_cfg, max_depth=8)),
            ("seed bag 2024", _bump(parent_cfg, seed=2024)),
        ]
    else:  # cat
        muts = [
            ("depth down to 5", _bump(parent_cfg, depth=5)),
            ("l2_leaf_reg up to 12", _bump(parent_cfg, l2_leaf_reg=12.0)),
            ("lr down to 0.035", _bump(parent_cfg, learning_rate=0.035)),
            ("depth up to 8 (capacity probe)", _bump(parent_cfg, depth=8)),
            ("native categoricals OFF (isolate the encoding's contribution)",
             _bump(parent_cfg, cat_native=False)),
            ("seed bag 2024", _bump(parent_cfg, seed=2024)),
        ]
    return muts[idx % len(muts)]


# ---------------------------------------------------------------------------
# blend proposals
# ---------------------------------------------------------------------------
def solo_pool(tree):
    return [n["id"] for n in tree["nodes"]
            if n["status"] == "evaluated"
            and n["config"].get("kind", "solo") == "solo"
            and os.path.exists(os.path.join(CACHE_DIR, f"solo_{n['id']}.npz"))]


def blend_proposal(tree, parent_node, idx):
    pool = solo_pool(tree)
    if len(pool) < 2:
        return None, None
    ranked = sorted(pool, key=lambda i: -(auc_of(tree, i) or 0))
    parent_cfg = parent_node["config"]
    cur = list(parent_cfg.get("members", [])) if parent_cfg.get("kind") == "blend" else []

    options = [
        ("kitchen-sink: every cached solo in the tree",
         dict(kind="blend", members=sorted(pool), weight_search="dirichlet", space="prob")),
        ("top-3 solos only (strength over breadth)",
         dict(kind="blend", members=sorted(ranked[:3]), weight_search="dirichlet", space="prob")),
        ("top-6 solos",
         dict(kind="blend", members=sorted(ranked[:6]), weight_search="dirichlet", space="prob")),
        ("add the newest solo to the parent blend",
         dict(kind="blend", members=sorted(set(cur) | {max(pool)}),
              weight_search="dirichlet", space="prob") if cur else None),
        ("rank-space instead of probability-space",
         dict(kind="blend", members=sorted(cur or ranked[:5]),
              weight_search="dirichlet", space="rank")),
        ("drop the weakest member of the parent blend",
         dict(kind="blend", members=sorted(set(cur) - {min(cur, key=lambda i: auc_of(tree, i) or 0)}),
              weight_search="dirichlet", space="prob") if len(cur) > 2 else None),
        ("top-10 solos",
         dict(kind="blend", members=sorted(ranked[:10]), weight_search="dirichlet", space="prob")),
    ]
    for j in range(len(options)):
        label, cfg = options[(idx + j) % len(options)]
        if cfg is None or len(cfg.get("members", [])) < 2:
            continue
        if hv3.find_duplicate_config(tree, core(cfg)) is None:
            return label, cfg
    return None, None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    tree = (hv3.load_search_state(TREE_PATH) if os.path.exists(TREE_PATH)
            else hv3.new_tree(COMP))
    hv3.init_budget(tree, total_budget=TOTAL_BUDGET)
    ds = dstate(tree)

    try:
        priors = hv3.suggest_priors({"metric": "roc_auc", "tags": ["binary", "tabular",
                                                                   "gbdt", "blend"],
                                     "data_type": "tabular", "comp": COMP})
    except Exception as e:  # noqa: BLE001 — priors are advisory, never fatal
        priors = []
        print(f"suggest_priors unavailable ({type(e).__name__}: {e}) — continuing without",
              flush=True)
    ds["priors"] = [p if isinstance(p, str) else str(p) for p in (priors or [])][:12]
    print(f"suggest_priors returned {len(ds['priors'])} bullets "
          f"(self-citing s6e2 bullets are dropped by the harness itself)", flush=True)

    # ---- root: digit-for-digit verification (checklist item 1) -------------
    if not tree["nodes"]:
        root_cfg = dict(kind="solo", model="lgb", params=LGB_TUNED_PARAMS,
                        features={"version": "raw", "drop": []}, cat_native=False, seed=42)
        rid = eval_and_add(tree, None, "ROOT = linear-iteration best solo (LGB_tuned)",
                           root_cfg, is_root=True, name="root")
        got = auc_of(tree, rid)
        assert got is not None and round(got, 6) == ROOT_AUC, (
            f"ROOT VERIFICATION FAILED: re-evaluated root AUC {got} != linear "
            f"experiments.json value {ROOT_AUC}. Aborting rather than letting data/feature "
            f"drift silently contaminate the whole tree.")
        log(tree, f"root verified digit-for-digit: {got:.6f} == {ROOT_AUC}")

        # ---- first-generation lineages -----------------------------------
        seeds = [
            ("boundary-push: params on the Optuna box edge, pushed OUTSIDE it", None),
            ("XGB hand-set (different split algorithm)",
             dict(kind="solo", model="xgb", params=XGB_PARAMS,
                  features={"version": "raw", "drop": []}, cat_native=False, seed=42)),
            ("CatBoost + native categoricals (ordered target statistics)",
             dict(kind="solo", model="cat", params=CAT_PARAMS,
                  features={"version": "raw", "drop": []}, cat_native=True, seed=42)),
            ("LGB high-capacity contrast (leaves 128, shallow-optimum stress test)",
             dict(kind="solo", model="lgb",
                  params={**LGB_TUNED_PARAMS, "num_leaves": 128, "max_depth": 9,
                          "min_child_samples": 100},
                  features={"version": "raw", "drop": []}, cat_native=False, seed=42)),
        ]
        edges = hv3.boundary_candidates({"params": LGB_TUNED_PARAMS}, LGB_SEARCH_SPACE)
        log(tree, f"boundary_candidates flagged {len(edges)} edge params: "
                  f"{[e.get('param') if isinstance(e, dict) else e for e in edges]}")
        # boundary_candidates returns {"param","edge","old_value","new_value"}; its
        # push can overshoot into an invalid region (num_leaves -30), so clamp to the
        # legal domain of each hyperparameter before using it.
        CLAMP = {"num_leaves": (2, 4096), "max_depth": (2, 20),
                 "min_child_samples": (2, 5000), "learning_rate": (0.001, 0.5),
                 "feature_fraction": (0.1, 1.0), "bagging_fraction": (0.1, 1.0),
                 "reg_alpha": (0.0, 200.0), "reg_lambda": (0.0, 200.0),
                 "min_split_gain": (0.0, 10.0)}
        bp = dict(LGB_TUNED_PARAMS)
        for e in (edges or []):
            if not (isinstance(e, dict) and "param" in e and "new_value" in e):
                continue
            k, v = e["param"], e["new_value"]
            lo, hi = CLAMP.get(k, (None, None))
            if lo is not None:
                v = min(max(v, lo), hi)
            if isinstance(LGB_TUNED_PARAMS.get(k), int):
                v = int(round(v))
            bp[k] = v
        if bp == LGB_TUNED_PARAMS:   # nothing on an edge -> push num_leaves below the box
            bp["num_leaves"] = 4
        seeds[0] = (seeds[0][0], dict(kind="solo", model="lgb", params=bp,
                                      features={"version": "raw", "drop": []},
                                      cat_native=False, seed=42))
        for label, cfg in seeds:
            eval_and_add(tree, rid, f"SEED {label}", cfg, name=label.split(":")[0])

        # blend lineage seed
        pool = solo_pool(tree)
        if len(pool) >= 2:
            eval_and_add(tree, rid, "SEED blend lineage: kitchen-sink of the seed pool",
                         dict(kind="blend", members=sorted(pool),
                              weight_search="dirichlet", space="prob"), name="blend")

    # ---- main loop --------------------------------------------------------
    by_id = {n["id"]: n for n in tree["nodes"]}
    while not hv3.should_stop(tree):
        if time.time() - T0 > WALL_BUDGET_S:
            ds["wall_stop"] = (f"WALL-CLOCK GUARD FIRED at {time.time() - T0:.0f}s "
                               f"(limit {WALL_BUDGET_S:.0f}s) after "
                               f"{hv3.n_evaluated(tree)}/{TOTAL_BUDGET} evaluated nodes. "
                               f"This is a budget stop, NOT a convergence stop.")
            log(tree, ds["wall_stop"])
            tree["search_state"]["budget"]["phase"] = "stopped"
            tree["search_state"]["budget"]["stop_reason"] = ds["wall_stop"]
            hv3.save_search_state(tree, TREE_PATH)
            break
        if hv3.n_evaluated(tree) >= TOTAL_BUDGET:
            log(tree, f"numeric node cap reached ({TOTAL_BUDGET})")
            tree["search_state"]["budget"]["phase"] = "stopped"
            tree["search_state"]["budget"]["stop_reason"] = "numeric node-budget cap"
            hv3.save_search_state(tree, TREE_PATH)
            break

        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not ds["burst_injected"]:
            ds["burst_injected"] = True
            log(tree, "=== MANDATORY EXPLORE BURST ===")
            rid = tree["root_id"]
            pool = solo_pool(tree)
            burst = [
                ("BURST kitchen-sink mega-blend of the ENTIRE solo pool",
                 dict(kind="blend", members=sorted(pool), weight_search="dirichlet",
                      space="prob")),
                ("BURST long-shot: LGB extra-tiny trees (leaves 4, lr 0.015)",
                 dict(kind="solo", model="lgb",
                      params={**LGB_TUNED_PARAMS, "num_leaves": 4, "max_depth": 3,
                              "learning_rate": 0.015},
                      features={"version": "raw", "drop": []}, cat_native=False, seed=42)),
                ("BURST long-shot: LGB DART boosting (contrasting regularisation route)",
                 dict(kind="solo", model="lgb",
                      params={**LGB_TUNED_PARAMS, "boosting_type": "dart",
                              "n_estimators": 800, "drop_rate": 0.1},
                      features={"version": "raw", "drop": []}, cat_native=False, seed=42)),
                ("BURST long-shot: XGB deep + heavy L2 (opposite corner from the LGB optimum)",
                 dict(kind="solo", model="xgb",
                      params={**XGB_PARAMS, "max_depth": 10, "reg_lambda": 20.0,
                              "min_child_weight": 40, "learning_rate": 0.03},
                      features={"version": "raw", "drop": []}, cat_native=False, seed=42)),
                ("BURST long-shot: CatBoost deep, no native cats",
                 dict(kind="solo", model="cat",
                      params={**CAT_PARAMS, "depth": 8, "l2_leaf_reg": 3.0},
                      features={"version": "raw", "drop": []}, cat_native=False, seed=42)),
                ("BURST rank-space mega-blend (different aggregation geometry)",
                 dict(kind="blend", members=sorted(pool), weight_search="dirichlet",
                      space="rank")),
            ]
            for label, cfg in burst:
                nid = eval_and_add(tree, rid, label, cfg, name=label.split(":")[0])
                if nid is not None and cfg.get("kind") == "solo":
                    passed, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
                    if not passed:
                        log(tree, f"  sanity gate FAILED for burst seed #{nid} "
                                  f"(bound {bound:.6f}) -> lineage plateaued")
                    hv3.save_search_state(tree, TREE_PATH)
            by_id = {n["id"]: n for n in tree["nodes"]}
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            log(tree, "select_next_parent returned None -> nothing left to expand")
            break
        by_id = {n["id"]: n for n in tree["nodes"]}
        parent = by_id[parent_id]
        idx = hv3.lineage_size(tree, lineage_id) - 1 + ds["queue_offset"].get(str(lineage_id), 0)

        if parent["config"].get("kind") == "blend":
            label, cfg = blend_proposal(tree, parent, idx)
            if cfg is None:
                ds["queue_offset"][str(lineage_id)] = ds["queue_offset"].get(str(lineage_id), 0) + 1
                hv3.add_node(tree, parent_id, "[blend queue exhausted]",
                             {"kind": "placeholder", "parent": parent_id, "n": idx},
                             None, "failed", 0.0)
                hv3.save_search_state(tree, TREE_PATH)
                continue
            eval_and_add(tree, parent_id, f"blend: {label}", cfg)
        else:
            label, cfg = solo_mutations(tree, parent["config"], idx)
            nid = eval_and_add(tree, parent_id, f"solo: {label}", cfg)
            if nid is None:
                ds["queue_offset"][str(lineage_id)] = ds["queue_offset"].get(str(lineage_id), 0) + 1

    # ---- report -----------------------------------------------------------
    hv3.save_search_state(tree, TREE_PATH)
    gb = hv3.global_best(tree)
    best_auc = -gb["score"] if gb and gb["score"] is not None else None
    evald = [n for n in tree["nodes"] if n["status"] == "evaluated"]
    solos = [n for n in evald if n["config"].get("kind", "solo") == "solo"]
    blends = [n for n in evald if n["config"].get("kind") == "blend"]
    best_solo = min(solos, key=lambda n: n["score"]) if solos else None

    report = {
        "comp": COMP,
        "n_evaluated": len(evald),
        "n_solo": len(solos), "n_blend": len(blends),
        "total_budget": TOTAL_BUDGET,
        "wall_s": round(time.time() - T0, 1),
        "global_best": {"node_id": gb["id"] if gb else None, "auc": best_auc,
                        "config": gb["config"] if gb else None},
        "best_solo": {"node_id": best_solo["id"] if best_solo else None,
                      "auc": -best_solo["score"] if best_solo else None,
                      "config": best_solo["config"] if best_solo else None},
        "linear_anchors": {"best_solo": ROOT_AUC,
                           "best_blend_full_oof_fitted": LINEAR_BEST_FITTED,
                           "best_blend_leave_fold_out_honest": LINEAR_BEST_HONEST},
        "evals_to_match_linear_fitted": ds["evals_to_match_linear_fitted"],
        "evals_to_match_linear_honest": ds["evals_to_match_linear_honest"],
        "dedup_rejections": tree["search_state"].get("dedup_rejections",
                                                     hv3.__dict__.get("DEDUP_REJECTIONS")),
        "plateaued": tree["search_state"].get("plateaued", []),
        "backtrack_log": tree["search_state"].get("backtrack_log", []),
        "stop_reason": tree["search_state"]["budget"].get("stop_reason"),
        "wall_stop": ds["wall_stop"],
        "phase": tree["search_state"]["budget"]["phase"],
        "burst_injected": ds["burst_injected"],
        "all_nodes": [{"id": n["id"], "kind": n["config"].get("kind", "solo"),
                       "status": n["status"], "mutation": n["mutation"],
                       "auc": (-n["score"] if n["score"] is not None else None),
                       "wall_s": n["wall_s"]} for n in tree["nodes"]],
        "caveats": [
            "OOF-ONLY: no Kaggle Public/Private LB anchor exists for this run.",
            "Every blend node's score is a FULL-OOF FITTED weight search and is therefore "
            "OPTIMISTIC. The winner is re-scored with leave-fold-out weight fitting in "
            "competitions/playground-series-s6e2/scripts/07_final.py before submission.",
            "The whole model family spans ~0.0008 AUC on this dataset, so any single "
            "improvement here is well under 0.1% relative and may be CV noise.",
        ],
    }
    out = os.path.join(_COMP_DIR, "tree_search_run2_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 78, flush=True)
    print(f"TREE SEARCH DONE  {len(evald)} evaluated nodes in {report['wall_s']:.0f}s", flush=True)
    print(f"global best  = node #{report['global_best']['node_id']}  AUC {best_auc}", flush=True)
    print(f"best solo    = node #{report['best_solo']['node_id']}  "
          f"AUC {report['best_solo']['auc']}", flush=True)
    print(f"linear best  = solo {ROOT_AUC} / blend-fitted {LINEAR_BEST_FITTED} / "
          f"blend-honest {LINEAR_BEST_HONEST}", flush=True)
    print(f"evals to match linear fitted={ds['evals_to_match_linear_fitted']} "
          f"honest={ds['evals_to_match_linear_honest']}", flush=True)
    print(f"stop_reason  = {report['stop_reason']}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
