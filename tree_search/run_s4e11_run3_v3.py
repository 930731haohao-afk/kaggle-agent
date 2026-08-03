"""tree_search/run_s4e11_run3_v3.py -- Stage-4 harness_v3 tree-search driver for
playground-series-s4e11 (Depression, ACCURACY, MAXIMIZE).

Written fresh for this run; it does not read or reuse any artifact of a previous run of
this competition (lane isolation). Follows the driver checklist in
references/07_tree_search.md, modeled structurally on run_s6e2_v3.py (a different
competition's driver, used only as an API template).

Root: the strongest solo from linear-iteration round 1 (scripts/pool_round1.json) chosen
PROGRAMMATICALLY by max honest accuracy, retrained through eval_s4e11_run3.evaluate_solo
and asserted to reproduce that score to 6 decimals before anything else runs.

Node-space: solo + blend (default schema). Eight first-generation lineages:
  LGBHAND / XGBHAND / CATHAND  -- model + hyperparameter diversity
  SEEDBAG                      -- cheapest residual gain after tuning
  BOUNDARYPUSH                 -- boundary_candidates() on the root's own search box
  FEATPRUNE                    -- drop the near-dead columns EDA flagged
  FEATVARIANT                  -- the comp-local lever: race the feature-set structure
                                  decisions (explicit interactions on/off, keep the raw
                                  block columns or only the coalesced ones, keep Name)
  BLEND                        -- weighted mixes of the solo pool

SIGN CONVENTION: the harness assumes lower-is-better; accuracy is MAXIMIZE, so every score
handed to the harness is `-accuracy` (eval_s4e11_run3.py negates). The human-readable
number is result["acc"] / -node["score"].

EVERY score in this tree -- solo and blend alike -- is the HONEST accuracy: thresholds are
always fitted leave-fold-out, and a blend's weights are too. See eval_s4e11_run3.py's
docstring for why that is not optional under this metric.

Run: `uv run python3 tree_search/run_s4e11_run3_v3.py` (resumable; state persisted to
competitions/playground-series-s4e11/experiments_tree_v3.json after every node).
`--dry-run` verifies wiring with zero training compute.
"""
import copy
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
import eval_s4e11_run3 as ev  # noqa: E402

COMP = "playground-series-s4e11"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_v3.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_s4e11_run3.py")
POOL_JSON = os.path.join(_COMP_DIR, "scripts", "pool_round1.json")
EVAL_TIMEOUT_S = 330
# Per-INVOCATION wall budget. The search is fully resumable, so it is run as a series of
# short foreground chunks (this session's tool calls are capped well below the search's
# total runtime, and a backgrounded job would be killed mid-training). Override with
# --max-wall <seconds>.
MAX_WALL_S = 240
ITER_SAFETY_CAP = 200
MIN_NODES_FLOOR = 25

LGB_SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.15, "log": True},
    "num_leaves": {"low": 15, "high": 511, "log": True},
    "min_child_samples": (5, 200),
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
    "reg_alpha": {"low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"low": 1e-3, "high": 10.0, "log": True},
}
XGB_SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.15, "log": True},
    "max_depth": (3, 12),
    "min_child_weight": (1, 100),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": {"low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"low": 1e-3, "high": 10.0, "log": True},
}

LGB_BASE = {}   # evaluator defaults: lr 0.03, leaves 63, mcs 50, ff/bf 0.8, a0.1 l1.0
XGB_BASE = {}   # evaluator defaults: lr 0.03, depth 6, sub/col 0.8, mcw 10
CAT_BASE = {}   # evaluator defaults: lr 0.05, depth 6, l2 3.0

# Near-dead columns per the EDA verdict: gender split 0.1782/0.1846 and family history
# 0.1754/0.1881 both sit within 0.7pp of the 18.17% base rate, and block_missing had
# exactly zero LGB gain (it fires on 21 of 140700 rows).
WEAK_COLS = ["gender_male", "gender", "fam_hist", "block_missing"]


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


def acc_of(r):
    if r is None:
        return None
    if r.get("result") and "acc" in r["result"]:
        return r["result"]["acc"]
    return -r["score"] if r.get("score") is not None else None


def _seed_key(model):
    return {"lgb": "seed", "xgb": "random_state", "cat": "random_seed"}[model]


# ---------------------------------------------------------------------------
# root selection
# ---------------------------------------------------------------------------
def pick_root():
    with open(POOL_JSON) as f:
        pool = json.load(f)["solo"]
    cands = {t: round(v["honest_acc"], 6) for t, v in pool.items()}
    winner = max(cands, key=lambda t: cands[t])
    model = {"LGB": "lgb", "XGB": "xgb", "CAT": "cat"}[winner]
    cfg = {"kind": "solo", "model": model, "params": {}, "features": {"drop": []}}
    return winner, cfg, cands[winner], cands


# ---------------------------------------------------------------------------
# node evaluation
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, config, *, is_root=False):
    stored = core(config)
    if not is_root:
        # Pre-check so a duplicate never costs a training run; hv3.add_node still owns the
        # dedup bookkeeping (including the E-5 dedup-budget-burn placeholder).
        if hv3.find_duplicate_config(tree, stored) is not None:
            _nid, dup = hv3.add_node(tree, parent_id, mutation, stored, None, "failed", 0.0)
            _dedup_rejections(tree).append(dict(parent_id=parent_id, dup_of=dup,
                                                mutation=mutation))
            hv3.save_search_state(tree, TREE_PATH)
            return None, dup, None

    if stored.get("kind") == "blend":
        r = ev.evaluate(stored, node_id=None, timeout_s=EVAL_TIMEOUT_S)
    else:
        # The OOF cache is keyed by node id, and the id is only assigned by add_node
        # AFTER the eval returns -- harness.next_id is the sanctioned way to reserve it
        # under this repo's strictly-sequential single-writer driver convention.
        r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, stored, EVAL_TIMEOUT_S,
                                     node_id=hv3.next_id(tree))
    score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r.get("result")
    if status != "evaluated":
        print(f"    !! node failed: {r.get('error')}")

    if is_root:
        nid = hv3.add_root(tree, mutation, stored, score, status, wall_s)
    else:
        nid, dup = hv3.add_node(tree, parent_id, mutation, stored, score, status, wall_s)
        if nid is None:
            _dedup_rejections(tree).append(dict(parent_id=parent_id, dup_of=dup,
                                                mutation=mutation))
            hv3.save_search_state(tree, TREE_PATH)
            return None, dup, None
    if result is not None:
        _node_results(tree)[str(nid)] = result
    hv3.save_search_state(tree, TREE_PATH)
    return nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


# ---------------------------------------------------------------------------
# first-generation seeds
# ---------------------------------------------------------------------------
def seed_lgbhand():
    return ({"kind": "solo", "model": "lgb",
             "params": {"num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.05},
             "features": {"drop": []}},
            "model-variant: shallower, more regularized LGB (num_leaves 63->31, "
            "min_child_samples 50->100, lr 0.03->0.05)")


def seed_xgbhand():
    return ({"kind": "solo", "model": "xgb",
             "params": {"max_depth": 8, "learning_rate": 0.02, "min_child_weight": 30},
             "features": {"drop": []}},
            "model-variant: deeper/slower XGB (depth 6->8, lr 0.03->0.02, "
            "min_child_weight 10->30)")


def seed_cathand():
    return ({"kind": "solo", "model": "cat",
             "params": {"depth": 8, "l2_leaf_reg": 6.0, "learning_rate": 0.08},
             "features": {"drop": []}},
            "model-variant: deeper CatBoost with a faster lr to keep it affordable "
            "(depth 6->8, l2_leaf_reg 3->6, lr 0.05->0.08)")


def seed_seedbag(root_cfg):
    p = dc(root_cfg["params"])
    p[_seed_key(root_cfg["model"])] = 2024
    return ({"kind": "solo", "model": root_cfg["model"], "params": p,
             "features": dc(root_cfg.get("features") or {"drop": []})},
            "seed variation: root's exact params, seed -> 2024 [PRIOR: seed-bagging is "
            "the cheapest residual gain after tuning, knowledge/experience.md s3e14/s3e9]")


def seed_boundarypush(root_cfg):
    space = {"lgb": LGB_SEARCH_SPACE, "xgb": XGB_SEARCH_SPACE}.get(root_cfg["model"])
    if space is None:
        return None
    # The root's params dict is empty (evaluator defaults), so materialize the defaults
    # boundary_candidates needs to see.
    defaults = {"lgb": dict(learning_rate=0.03, num_leaves=63, min_child_samples=50,
                            feature_fraction=0.8, bagging_fraction=0.8, reg_alpha=0.1,
                            reg_lambda=1.0),
                "xgb": dict(learning_rate=0.03, max_depth=6, min_child_weight=10,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                            reg_lambda=1.0)}[root_cfg["model"]]
    probe = {"kind": "solo", "model": root_cfg["model"], "params": defaults}
    edges = hv3.boundary_candidates(probe, space)
    if not edges:
        # nothing sits on an edge -- push the two knobs this metric is most sensitive to
        p = dict(defaults)
        if root_cfg["model"] == "xgb":
            p["max_depth"] = 11
            p["learning_rate"] = 0.015
            desc = "no flagged edges; hand-push depth 6->11 + lr 0.03->0.015"
        else:
            p["num_leaves"] = 255
            p["learning_rate"] = 0.015
            desc = "no flagged edges; hand-push num_leaves 63->255 + lr 0.03->0.015"
        return ({"kind": "solo", "model": root_cfg["model"], "params": p,
                 "features": dc(root_cfg.get("features") or {"drop": []})},
                f"model-variant: {desc} [feature 4 fallback]")
    p = dict(defaults)
    descs = []
    for e in edges:
        p[e["param"]] = e["new_value"]
        descs.append(f"{e['param']} {e['old_value']}->{e['new_value']} (edge={e['edge']})")
    return ({"kind": "solo", "model": root_cfg["model"], "params": p,
             "features": dc(root_cfg.get("features") or {"drop": []})},
            f"model-variant: boundary_candidates()-flagged push(es) -- "
            f"{', '.join(descs)} [feature 4]")


def seed_featprune(root_cfg):
    return ({"kind": "solo", "model": root_cfg["model"], "params": dc(root_cfg["params"]),
             "features": {"drop": sorted(WEAK_COLS)}},
            f"feature-pruning: root minus the {len(WEAK_COLS)} near-dead columns EDA "
            f"flagged ({', '.join(WEAK_COLS)}) [PRIOR: feature pruning is a recurring "
            f"win, knowledge/experience.md s3e7/s3e14/s3e9]")


def seed_featvariant(root_cfg):
    return ({"kind": "solo", "model": root_cfg["model"], "params": dc(root_cfg["params"]),
             "features": {"drop": [], "variant": {"add_interactions": True}}},
            "feature-structure: turn ON the 6 explicit interaction columns "
            "(pressure x financial, pressure - satisfaction, hours x pressure, "
            "age x student, suicidal x pressure, stress_load) [PRIOR: the 'trees learn "
            "interactions themselves' rule is NOT universal -- confirmed on s5e10/s6e2, "
            "REFUTED on s6e1 -- so it must be tested per competition with a same-fold "
            "on/off comparison, which is exactly this node]")


ALL_LINEAGE_NAMES = ["LGBHAND", "XGBHAND", "CATHAND", "SEEDBAG", "BOUNDARYPUSH",
                     "FEATPRUNE", "FEATVARIANT", "BLEND"]
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
        if n["status"] != "evaluated" or n["id"] == tree["root_id"] or n["score"] is None:
            continue
        if hv3.lineage_of(tree, n["id"]) == lineage_id:
            if best is None or n["score"] < best["score"]:
                best = n
    return best


def solo_pool(tree):
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["score"] is not None
             and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def _bump(cfg, **kw):
    c = dc(cfg)
    c.setdefault("params", {})
    c["params"].update(kw)
    return c


def _featset(cfg, **variant):
    c = dc(cfg)
    f = dc(c.get("features") or {})
    v = dc(f.get("variant") or {})
    v.update(variant)
    f["variant"] = v
    c["features"] = f
    return c


def _lineage_kind(tree, lineage_id):
    node = next(n for n in tree["nodes"] if n["id"] == lineage_id)
    return node["config"].get("kind", "solo")


SOLO_QUEUES_BUILDERS = {
    "LGBHAND": [
        lambda c: (_bump(c, num_leaves=127, min_child_samples=20, reg_lambda=5.0),
                   "LGBHAND: NEW -- opposite direction, more capacity with heavier L2 "
                   "(num_leaves 31->127, min_child_samples 100->20, reg_lambda 1->5)"),
        lambda c: (_bump(c, feature_fraction=0.6, bagging_fraction=0.6),
                   "LGBHAND: NEW -- harder row/column subsampling (0.8->0.6 both)"),
    ],
    "XGBHAND": [
        lambda c: (_bump(c, subsample=0.6, colsample_bytree=0.6),
                   "XGBHAND: NEW -- harder row/column subsampling (0.8->0.6)"),
        lambda c: (_bump(c, max_depth=4, min_child_weight=5),
                   "XGBHAND: NEW -- opposite direction, shallow tree (depth 8->4, "
                   "min_child_weight 30->5)"),
    ],
    "CATHAND": [
        lambda c: (_bump(c, one_hot_max_size=64),
                   "CATHAND: NEW -- one_hot_max_size=64 so the mid-cardinality "
                   "categoricals (city 98, profession 64, degree 115) go one-hot "
                   "instead of through ordered target statistics"),
        lambda c: (_bump(c, depth=10),
                   "CATHAND: NEW -- depth 8->10 [PRIOR: s3e11 found large tabular data "
                   "rewards depth, and depth 10->12 was that comp's single largest lever]"),
    ],
    "SEEDBAG": [
        lambda c: (_bump(c, **{_seed_key(c["model"]): 777}),
                   "SEEDBAG: NEW -- third independent seed (777)"),
    ],
    "BOUNDARYPUSH": [],
    "FEATPRUNE": [
        lambda c: ({**dc(c), "features": {"drop": sorted(set(WEAK_COLS) |
                                                          {"CGPA", "sleep_duration"})}},
                   "FEATPRUNE: NEW -- prune further: also drop CGPA (0.32% gain, and "
                   "80% structurally missing) and the raw sleep_duration category "
                   "(sleep_hours already carries its order)"),
        lambda c: ({**dc(c), "features": {"drop": ["name", "name_count"]}},
                   "FEATPRUNE: NEW -- ablate `Name` alone. EDA measured its target-rate "
                   "spread at 4.9x the binomial noise band and LGB gave it 8.1% of gain; "
                   "this node prices that signal instead of assuming it"),
    ],
    "FEATVARIANT": [
        lambda c: (_featset(c, keep_block_originals=False),
                   "FEATVARIANT: NEW -- drop the 4 raw block columns and keep ONLY the "
                   "coalesced pressure/satisfaction pair. EDA proved the blocks are "
                   "exactly disjoint, so this is lossless de-duplication of the split "
                   "budget -- unless the raw columns' own missingness is itself signal"),
        lambda c: (_featset(c, add_counts=False),
                   "FEATVARIANT: NEW -- drop the 4 frequency-encoding columns "
                   "(they took 0.05-0.08% of gain each)"),
        lambda c: (_featset(c, rare_min_count=50),
                   "FEATVARIANT: NEW -- collapse the junk tail harder (rare_min_count "
                   "10->50), which folds ~30 more low-count levels into __RARE__"),
    ],
}


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    c.setdefault("params", {})
    c["params"][_seed_key(c["model"])] = 4000 + attempt
    return c, f"fallback seed-variation: alt seed={4000 + attempt}"


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = sorted({tree["root_id"], ids.get("LGBHAND"), ids.get("CATHAND")} - {None})
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    return cfg, f"ensemble seed: blend of root + LGBHAND + CATHAND ({members})"


def blend_fallback(tree, parent_node, attempt):
    pool = solo_pool(tree)
    members = parent_node["config"]["members"]
    for add_id in [nid for nid in pool if nid not in members]:
        child = {"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": "dirichlet"}
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
        best = _best_node_in_lineage(tree, ids[name])
        add_id = best["id"] if best else ids[name]
        members = parent_node["config"]["members"]
        if add_id in members:
            return None
        child = {"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": "dirichlet"}
        return child, f"BLEND: add {name}'s best solo member (#{add_id})"
    return fn


BLEND_QUEUE = [_mk_add_named("XGBHAND"), _mk_add_named("FEATVARIANT"),
               _mk_add_named("SEEDBAG"), _mk_add_named("FEATPRUNE"),
               _mk_add_named("BOUNDARYPUSH"), _blend_remove_weakest]


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES.get(lineage_id, str(lineage_id))
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


def inject_explore_burst(tree):
    pool = solo_pool(tree)
    seeds = []
    if len(pool) >= 2:
        cfg = {"kind": "blend", "members": sorted(pool), "weight_search": "dirichlet"}
        seeds.append((cfg, f"[EXPL_MEGABLEND] explore-burst: mandatory kitchen-sink blend "
                           f"of the entire solo pool ({len(pool)} members) "
                           f"[07_tree_search.md Section 3: post-plateau gains have "
                           f"historically come from exactly this node]"))
    seeds.append(({"kind": "solo", "model": "lgb",
                   "params": {"num_leaves": 511, "learning_rate": 0.015,
                              "min_child_samples": 200, "reg_lambda": 10.0},
                   "features": {"drop": []}},
                  "[EXPL_BIGLGB] explore-burst long-shot: far more capacity with far "
                  "heavier regularization (num_leaves 63->511, lr 0.03->0.015, "
                  "min_child_samples 50->200, reg_lambda 1->10) -- 140k rows is the "
                  "regime where s3e11 found capacity, not shrinkage, was the answer"))
    seeds.append(({"kind": "solo", "model": "lgb",
                   "params": {"boosting": "dart", "num_leaves": 63,
                              "learning_rate": 0.05, "num_boost_round": 600,
                              "early_stopping_rounds": 10_000},
                   "features": {"drop": [], "variant": {"add_interactions": True}}},
                  "[EXPL_DART] explore-burst long-shot: DART boosting on the interaction "
                  "feature set -- a genuinely different bias/variance profile from every "
                  "GBDT already in the pool, which is what a blend pool is short of"))
    return seeds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def dry_run():
    winner, root_cfg, root_acc, cands = pick_root()
    print(f"[dry-run] root candidates (honest acc): {cands}")
    print(f"[dry-run] chosen root = {winner} ({root_acc:.6f}), cfg={root_cfg}")
    for name, fn in [("LGBHAND", seed_lgbhand), ("XGBHAND", seed_xgbhand),
                     ("CATHAND", seed_cathand)]:
        print(f"  {name}: {fn()[1]}")
    for name, fn in [("SEEDBAG", seed_seedbag), ("BOUNDARYPUSH", seed_boundarypush),
                     ("FEATPRUNE", seed_featprune), ("FEATVARIANT", seed_featvariant)]:
        r = fn(root_cfg)
        print(f"  {name}: {r[1] if r else '(skipped)'}")
    print("[dry-run] OK -- no training performed.")


def main(max_wall_s=MAX_WALL_S):
    t_start = time.time()
    resumed = os.path.exists(TREE_PATH)
    tree = hv3.load_search_state(TREE_PATH) if resumed else hv3.new_tree(COMP)
    hv3.init_budget(tree)

    if not resumed:
        winner, root_cfg, root_acc, cands = pick_root()
        print(f"[root] candidates (linear round-1 honest accuracy): {cands}")
        print(f"[root] chosen: {winner} ({root_acc:.6f}) -- retraining to verify")
        mutation = (f"root: linear-iteration round-1 strongest solo ({winner}), retrained "
                    f"through eval_s4e11_run3.evaluate_solo for digit-for-digit "
                    f"verification against honest accuracy {root_acc:.6f}")
        nid, _, info = eval_and_add(tree, None, mutation, root_cfg, is_root=True)
        got = acc_of(info)
        assert got is not None and round(got, 6) == round(root_acc, 6), (
            f"ROOT VERIFICATION FAILED: retrained acc {got} != round-1 {root_acc}")
        print(f"[root] verified digit-for-digit: {got:.6f} == {root_acc:.6f}")
    else:
        print(f"[resume] loaded tree with {len(tree['nodes'])} nodes "
              f"({hv3.n_evaluated(tree)} evaluated)")

    root_id = tree["root_id"]
    root_cfg = next(n for n in tree["nodes"] if n["id"] == root_id)["config"]
    existing = _lineage_seed_ids(tree)
    for name, nid in existing.items():
        LINEAGE_NAMES[nid] = name

    simple_seeds = [("LGBHAND", lambda: seed_lgbhand()),
                    ("XGBHAND", lambda: seed_xgbhand()),
                    ("CATHAND", lambda: seed_cathand()),
                    ("SEEDBAG", lambda: seed_seedbag(root_cfg)),
                    ("BOUNDARYPUSH", lambda: seed_boundarypush(root_cfg)),
                    ("FEATPRUNE", lambda: seed_featprune(root_cfg)),
                    ("FEATVARIANT", lambda: seed_featvariant(root_cfg))]
    for name, mk in simple_seeds:
        if name in existing:
            continue
        r = mk()
        if r is None:
            print(f"[seed {name}] skipped -- not applicable to this root")
            continue
        cfg, desc = r
        nid, _dup, info = eval_and_add(tree, root_id, f"[{name}] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            print(f"[seed {name}] node #{nid} acc={acc_of(info)}")
    if "BLEND" not in _lineage_seed_ids(tree):
        cfg, desc = seed_blend(tree)
        nid, _dup, info = eval_and_add(tree, root_id, f"[BLEND] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "BLEND"
            print(f"[seed BLEND] node #{nid} acc={acc_of(info)}")

    burst_injected = bool(_driver_state(tree).get("burst_injected"))
    it = 0
    while (not hv3.should_stop(tree) and it < ITER_SAFETY_CAP
           and (time.time() - t_start) < max_wall_s):
        it += 1
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not burst_injected:
            print(f"\n>>> PHASE -> explore_burst (n_eval={hv3.n_evaluated(tree)}); "
                  f"injecting mega-blend + long-shot seeds")
            for cfg, desc in inject_explore_burst(tree):
                name = desc.split("]")[0][1:]
                nid, _dup, info = eval_and_add(tree, tree["root_id"], desc, cfg)
                if nid is not None:
                    LINEAGE_NAMES[nid] = name
                    print(f"[burst {name}] node #{nid} acc={acc_of(info)}")
                    if cfg.get("kind") == "solo" and acc_of(info) is not None:
                        ok, bound = hv3.apply_burst_seed_sanity_gate(tree, nid)
                        print(f"    sanity gate: {'PASS' if ok else 'FAIL (plateaued)'} "
                              f"(bound={bound})")
            _driver_state(tree)["burst_injected"] = True
            burst_injected = True
            hv3.save_search_state(tree, TREE_PATH)
            continue

        parent_id, lineage_id = hv3.select_next_parent(tree)
        if parent_id is None:
            print("[stop] select_next_parent returned None")
            break
        if lineage_id not in LINEAGE_NAMES:
            for n in tree["nodes"]:
                if n["id"] == lineage_id:
                    LINEAGE_NAMES[lineage_id] = n["mutation"].split("]")[0].lstrip("[")
        child_cfg, desc = propose_child(tree, parent_id, lineage_id)
        if child_cfg is None:
            st = tree["search_state"]
            plat = set(st.get("plateaued", []))
            if lineage_id not in plat:
                plat.add(lineage_id)
                st["plateaued"] = sorted(plat)
                st.setdefault("backtrack_log", []).append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=f"propose_child exhausted for lineage "
                           f"{LINEAGE_NAMES.get(lineage_id, lineage_id)} -- no distinct "
                           f"mutations left (driver-side plateau)"))
                print(f"[plateau] lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                      f"exhausted -- marked plateaued")
            if st.get("active_lineage") == lineage_id:
                st["active_lineage"] = None
            hv3.save_search_state(tree, TREE_PATH)
            continue
        nid, _dup, info = eval_and_add(tree, parent_id, desc, child_cfg)
        if nid is None:
            continue
        gb = hv3.global_best(tree)
        print(f"[eval {it}] node #{nid} (parent #{parent_id}, "
              f"lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)}) acc={acc_of(info)}  "
              f"n_eval={hv3.n_evaluated(tree)}  "
              f"global_best={-gb['score'] if gb else None}  "
              f"[{time.time() - t_start:.0f}s]")

    gb = hv3.global_best(tree)
    print(f"\n=== stopped: phase={tree['search_state']['budget']['phase']} "
          f"reason={tree['search_state']['budget'].get('stop_reason')} ===")
    print(f"n_evaluated={hv3.n_evaluated(tree)}  wall={time.time() - t_start:.1f}s")
    print(f"global best: node #{gb['id']}  acc={-gb['score']:.6f}  "
          f"kind={gb['config'].get('kind')}")
    print(f"dedup_rejections={len(_dedup_rejections(tree))}")
    print(f"backtrack_log={json.dumps(tree['search_state'].get('backtrack_log'), default=str)}")
    print(f"plateaued={tree['search_state'].get('plateaued')}")
    print(f"MIN_NODES_FLOOR={MIN_NODES_FLOOR}  met={hv3.n_evaluated(tree) >= MIN_NODES_FLOOR}")
    hv3.save_search_state(tree, TREE_PATH)
    print(f"\nRESULT {json.dumps(dict(n_evaluated=hv3.n_evaluated(tree), best_node=gb['id'], best_acc=-gb['score']))}")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run()
    else:
        mw = MAX_WALL_S
        if "--max-wall" in sys.argv:
            mw = float(sys.argv[sys.argv.index("--max-wall") + 1])
        main(max_wall_s=mw)
