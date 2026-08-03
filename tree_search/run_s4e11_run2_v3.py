"""tree_search/run_s4e11_run2_v3.py -- Stage-4 harness_v3 tree-search driver for
playground-series-s4e11 (Depression, binary, ACCURACY, MAXIMIZE), run-2 lane.

Follows the driver checklist in
.claude/skills/kaggle-agent/references/07_tree_search.md, modeled on run_s6e2_v3.py.

Root: the strongest stage-3 solo (read PROGRAMMATICALLY from
competitions/playground-series-s4e11/blend_result.json -- CatBoost), retrained through
eval_s4e11_run2.evaluate_solo and asserted digit-for-digit against the stage-3 number
before anything else runs.

First-generation lineages -- note that ALL FIVE stage-3 pool members are seeded into the
tree (root = CAT, plus LGB / XGB / LGBSH / LR lineages). That is the afsis lesson: a
driver that reseeds only part of the linear pool structurally caps the blend branch
(0.4497 vs 0.4288 at identical budget). Beyond the pool: TEENC races the fold-aligned
target-encoding operator from the dossier, BOUNDARYPUSH is harness feature 4, FEATPRUNE
tests the recurring feature-pruning win, and BLEND is the ensemble lineage.

SIGN CONVENTION: the harness assumes lower-is-better; accuracy is MAXIMIZE, so every
score handed to it is `-accuracy`. result["acc"] carries the human-readable number.

HONESTY: blend nodes fit their weights AND their threshold on the same OOF vector they
are scored on. On this competition that optimism was measured at +0.000817 in stage 3,
and the honest (leave-fold-out) blend LOST to both the best solo and a plain equal-weight
average. So the tree's top-scoring node is a CANDIDATE, not the champion --
competitions/playground-series-s4e11/scripts/05_honest_select.py re-fits weights and
threshold leave-fold-out over the tree's top candidates and picks by that number.

Run: `uv run python3 tree_search/run_s4e11_run2_v3.py` (resumable; state persisted to
competitions/playground-series-s4e11/experiments_tree_v3_run2.json after every node).
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
import eval_s4e11_run2 as ev  # noqa: E402

COMP = "playground-series-s4e11"
_COMP_DIR = os.path.join(os.path.dirname(_HERE), "competitions", COMP)
TREE_PATH = os.path.join(_COMP_DIR, "experiments_tree_v3_run2.json")
EVAL_MODULE_PATH = os.path.join(_HERE, "eval_s4e11_run2.py")
BLEND_RESULT = os.path.join(_COMP_DIR, "blend_result.json")
EVAL_TIMEOUT_S = 1500
MAX_WALL_S = 150 * 60
ITER_SAFETY_CAP = 200

LGB_BASE = dict(ev.LGB_BASE)
XGB_BASE = dict(ev.XGB_BASE)
CAT_BASE = dict(ev.CAT_BASE)
LGB_SHALLOW = dict(LGB_BASE, num_leaves=7, max_depth=3, min_child_samples=200,
                   reg_alpha=2.0, reg_lambda=5.0, feature_fraction=0.6)

LGB_SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.15, "log": True},
    "num_leaves": {"low": 7, "high": 511, "log": True},
    "max_depth": (3, 14),
    "min_child_samples": (5, 300),
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
    "reg_alpha": {"low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"low": 1e-3, "high": 10.0, "log": True},
}
CAT_SEARCH_SPACE = {
    "learning_rate": {"low": 0.01, "high": 0.15, "log": True},
    "depth": (4, 10),
    "l2_leaf_reg": {"low": 1.0, "high": 20.0, "log": True},
}

# lowest-gain columns from the stage-3 baseline importance table (baseline_result.json):
# the two structural-missing flags are exact duplicates of is_student's information and
# scored gain 0.0; gender_male / degree_years / cnt_* are near-zero.
WEAK_FEATS = ["block_missing_academic", "block_missing_work", "gender_male",
              "degree_years", "cnt_Degree", "cnt_City"]


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


MODEL_OF = {"LGB": "lgb", "XGB": "xgb", "CAT": "cat", "LGBSH": "lgb", "LR": "lr"}
PARAMS_OF = {"LGB": LGB_BASE, "XGB": XGB_BASE, "CAT": CAT_BASE, "LGBSH": LGB_SHALLOW,
             "LR": {"C": 0.5}}


def pick_root():
    """Strongest stage-3 solo, read from blend_result.json (no hardcoded winner)."""
    info = json.load(open(BLEND_RESULT))
    members = info["members"]
    winner = max(members, key=lambda k: members[k]["acc"])
    cfg = {"kind": "solo", "model": MODEL_OF[winner], "params": dc(PARAMS_OF[winner]),
           "features": {"drop": []}}
    return winner, cfg, float(members[winner]["acc"]), {k: members[k]["acc"] for k in members}


def _seed_key(model):
    return {"lgb": "seed", "xgb": "random_state", "cat": "random_seed"}.get(model)


# ---------------------------------------------------------------------------
# eval_and_add
# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, proposal_cfg, is_root=False):
    stored = core(proposal_cfg)
    if not is_root:
        dup_id = hv3.find_duplicate_config(tree, stored)
        if dup_id is not None:
            _dedup_rejections(tree).append(dict(mutation=mutation, dup_id=dup_id))
            nid_null, _echo = hv3.add_node(tree, parent_id,
                                           mutation + " [dedup pre-check]", stored,
                                           None, "failed", 0.0)
            assert nid_null is None
            hv3.save_search_state(tree, TREE_PATH)
            return None, dup_id, None

    nid = hv3.next_id(tree)
    kind = stored.get("kind", "solo")
    if kind == "solo":
        r = hv3.eval_solo_subprocess(EVAL_MODULE_PATH, proposal_cfg, EVAL_TIMEOUT_S,
                                     node_id=nid)
        score, status, wall_s, result = r["score"], r["status"], r["wall_s"], r["result"]
    elif kind == "blend":
        try:
            t0 = time.time()
            best_w, best_neg, _oofs, warn = hv3.eval_blend_with_cost_guard(
                ev.CACHE_DIR, stored["members"],
                lambda vec: -ev.accuracy(ev._y, vec), tree=tree)
            best_score = -best_neg
            wall_s = time.time() - t0
            result = dict(members=stored["members"],
                          weights=[round(float(w), 4) for w in best_w],
                          acc=round(best_score, 6))
            if warn:
                result["cost_guard_warning"] = warn
                print(f"    [cost-guard] {warn}")
            score, status = round(-best_score, 6), "evaluated"
        except Exception as e:  # noqa: BLE001
            score, status, wall_s, result = None, "failed", 0.0, None
            mutation = mutation + f" [ERROR: {type(e).__name__}: {e}]"
    else:
        raise ValueError(f"unknown node kind {kind!r}")

    if is_root:
        real_nid = hv3.add_root(tree, mutation, stored, score, status, wall_s)
    else:
        real_nid, dup = hv3.add_node(tree, parent_id, mutation, stored, score, status,
                                     wall_s)
        assert dup is None, f"unexpected post-eval dedup (dup=#{dup})"
    assert real_nid == nid
    if result is not None:
        _node_results(tree)[str(real_nid)] = result
    hv3.save_search_state(tree, TREE_PATH)
    return real_nid, None, dict(score=score, status=status, wall_s=wall_s, result=result)


# ---------------------------------------------------------------------------
# first-generation seeds
# ---------------------------------------------------------------------------
def seed_pool_member(name):
    def fn(_root_cfg):
        return ({"kind": "solo", "model": MODEL_OF[name], "params": dc(PARAMS_OF[name]),
                 "features": {"drop": []}},
                f"stage-3 pool member {name} reseeded into the tree [PRIOR: afsis -- a "
                f"driver that reseeds only part of the linear pool structurally caps the "
                f"blend branch, 0.4497 vs 0.4288 at equal budget]")
    return fn


def seed_teenc(_root_cfg):
    return ({"kind": "solo", "model": "lgb", "params": dc(LGB_BASE),
             "features": {"drop": []}, "te": ["City", "Profession", "Degree", "Name"]},
            "encoding operator race: LGB + FOLD-ALIGNED target encoding of the 4 "
            "high-cardinality raw columns (computed inside each model fold from that "
            "fold's training rows only) vs the native-categorical default [dossier "
            "injection idea; s4e1 fold-alignment requirement]")


def seed_boundarypush(root_cfg):
    space = CAT_SEARCH_SPACE if root_cfg["model"] == "cat" else LGB_SEARCH_SPACE
    edges = hv3.boundary_candidates(root_cfg, space)
    edges = [e for e in edges
             if not (e["param"] == "max_depth" and isinstance(e["old_value"], (int, float))
                     and e["old_value"] < 0)]
    if not edges:
        # root sits in the interior: push depth outward manually as the seed idea
        p = dc(root_cfg["params"])
        key = "depth" if root_cfg["model"] == "cat" else "num_leaves"
        p[key] = 10 if key == "depth" else 127
        return ({"kind": "solo", "model": root_cfg["model"], "params": p,
                 "features": {"drop": []}},
                f"capacity push: {key} -> {p[key]} (boundary_candidates() found no edge "
                f"param on this root, so the first capacity step is authored) [feature 4]")
    p = dc(root_cfg["params"])
    descs = []
    for e in edges:
        p[e["param"]] = e["new_value"]
        descs.append(f"{e['param']} {e['old_value']}->{e['new_value']} (edge={e['edge']})")
    return ({"kind": "solo", "model": root_cfg["model"], "params": p,
             "features": {"drop": []}},
            f"boundary_candidates() push(es): {', '.join(descs)} [feature 4]")


def seed_featprune(root_cfg):
    return ({"kind": "solo", "model": root_cfg["model"], "params": dc(root_cfg["params"]),
             "features": {"drop": sorted(WEAK_FEATS)}},
            f"feature-pruning: root minus the {len(WEAK_FEATS)} lowest-gain columns "
            f"({', '.join(WEAK_FEATS)}); the two block_missing_* flags scored gain 0.0 "
            f"(redundant with is_student) [PRIOR: pruning is a recurring win, s3e7/s3e14]")


def seed_seedbag(root_cfg):
    p = dc(root_cfg["params"])
    key = _seed_key(root_cfg["model"])
    if key is None:
        return None
    p[key] = 2024
    return ({"kind": "solo", "model": root_cfg["model"], "params": p,
             "features": {"drop": []}},
            f"seed variation: root's exact params, {key} -> 2024 [PRIOR: seed bagging is "
            f"the cheapest residual gain after tuning]")


ALL_LINEAGE_NAMES = ["LGB", "XGB", "LGBSH", "LR", "TEENC", "BOUNDARYPUSH", "FEATPRUNE",
                     "SEEDBAG", "BLEND"]
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
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def _bump(cfg, **kw):
    c = dc(cfg)
    c["params"] = dict(c.get("params") or {})
    c["params"].update(kw)
    return c


def _lineage_kind(tree, lineage_id):
    node = next(n for n in tree["nodes"] if n["id"] == lineage_id)
    return node["config"].get("kind", "solo")


# ---------------------------------------------------------------------------
# per-lineage mutation queues (authored ideas first, then generic fallback)
# ---------------------------------------------------------------------------
SOLO_QUEUES = {
    "LGB": [
        lambda c: (_bump(c, num_leaves=31, min_child_samples=100, learning_rate=0.03),
                   "LGB: NEW -- shallower + slower (num_leaves 63->31, mcs 50->100, "
                   "lr 0.05->0.03); the shallow LGBSH member already beat base LGB by "
                   "0.0007, so walk base LGB toward it"),
        lambda c: (_bump(c, num_leaves=15, min_child_samples=150, reg_lambda=5.0),
                   "LGB: NEW -- further toward the regularized end (leaves 15, mcs 150, "
                   "reg_lambda 5)"),
        lambda c: ({**dc(c), "features": {"drop": ["cat_Name", "cnt_Name"]}},
                   "LGB: NEW -- drop the Name columns entirely; Name is 4th by gain but "
                   "is a first name, i.e. a level with no causal claim on depression -- "
                   "test whether that gain is real signal or fold-shared noise"),
    ],
    "XGB": [
        lambda c: (_bump(c, max_depth=8, min_child_weight=30, learning_rate=0.03),
                   "XGB: NEW -- deeper but heavier leaf constraint (depth 6->8, mcw "
                   "10->30, lr 0.03)"),
        lambda c: (_bump(c, max_depth=4, min_child_weight=50, reg_lambda=5.0),
                   "XGB: NEW -- shallow/strongly-regularized end (depth 4, mcw 50, "
                   "reg_lambda 5)"),
    ],
    "LGBSH": [
        lambda c: (_bump(c, num_leaves=15, max_depth=4),
                   "LGBSH: NEW -- one capacity step up from the shallow member "
                   "(leaves 7->15, depth 3->4)"),
        lambda c: (_bump(c, num_leaves=4, max_depth=2, min_child_samples=400),
                   "LGBSH: NEW -- one step further into bias-dominated territory "
                   "(leaves 4, depth 2, mcs 400) as a blend decorrelator "
                   "[TASK-BLEND-DECOR]"),
        lambda c: (_bump(c, learning_rate=0.02),
                   "LGBSH: NEW -- slower learning rate (0.05->0.02) at fixed shallow "
                   "capacity"),
    ],
    "LR": [
        lambda c: (_bump(c, C=0.05),
                   "LR: NEW -- 10x stronger L2 (C 0.5->0.05); cat-in-the-dat found the "
                   "sparse-OHE optimum well inside heavy shrinkage"),
        lambda c: (_bump(c, C=3.0),
                   "LR: NEW -- weaker L2 (C 0.5->3.0), the other side of the C grid"),
    ],
    "TEENC": [
        lambda c: ({**dc(c), "te": ["City", "Profession", "Degree"]},
                   "TEENC: NEW -- target-encode only the 3 semantically meaningful "
                   "high-cardinality columns, dropping Name from the TE set"),
        lambda c: ({**dc(c), "model": "cat", "params": dc(CAT_BASE)},
                   "TEENC: NEW -- same fold-aligned TE columns under CatBoost (whose "
                   "ordered target statistics may make explicit TE redundant -- worth "
                   "one node to find out)"),
    ],
    "BOUNDARYPUSH": [
        lambda c: (_bump(c, learning_rate=0.03),
                   "BOUNDARYPUSH: NEW -- pair the capacity push with a slower lr"),
        lambda c: (_bump(c, l2_leaf_reg=10.0) if c["model"] == "cat"
                   else _bump(c, reg_lambda=10.0),
                   "BOUNDARYPUSH: NEW -- pair the capacity push with stronger L2"),
    ],
    "FEATPRUNE": [
        lambda c: ({**dc(c), "features": {"drop": sorted(set(WEAK_FEATS) |
                                                         {"CGPA", "cnt_Name"})}},
                   "FEATPRUNE: NEW -- prune further (add CGPA and cnt_Name; CGPA is "
                   "80% missing and near-zero correlation)"),
        lambda c: ({**dc(c), "features": {"drop": ["block_missing_academic",
                                                   "block_missing_work"]}},
                   "FEATPRUNE: NEW -- isolate the two zero-gain flags alone, to tell a "
                   "real pruning gain from a lucky multi-column drop"),
    ],
    "SEEDBAG": [
        lambda c: (_bump(c, **{_seed_key(c["model"]): 777}),
                   "SEEDBAG: NEW -- third independent seed (777)"),
        lambda c: (_bump(c, **{_seed_key(c["model"]): 555}),
                   "SEEDBAG: NEW -- fourth independent seed (555)"),
    ],
}


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    key = _seed_key(c["model"])
    if key is None:
        c["params"] = dict(c.get("params") or {})
        c["params"]["C"] = [0.15, 1.0, 8.0, 0.02][attempt % 4]
        return c, f"fallback: LR C -> {c['params']['C']}"
    c["params"] = dict(c.get("params") or {})
    c["params"][key] = 4000 + attempt
    return c, f"fallback seed-variation: {key}={4000 + attempt}"


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = sorted({tree["root_id"], ids.get("LGBSH"), ids.get("LR")} - {None})
    return ({"kind": "blend", "members": members, "weight_search": "dirichlet"},
            f"ensemble seed: blend of root(CAT) + LGBSH + LR ({members}) -- the three "
            f"most decorrelated stage-3 members")


def blend_fallback(tree, parent_node, _attempt):
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
    weak = members[idx]
    child = dc(parent_node["config"])
    child["members"] = sorted([m for m in members if m != weak])
    return child, f"BLEND: drop lowest-weight member #{weak} (w={weights[idx]:.3f})"


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
        return ({"kind": "blend", "members": sorted(members + [add_id]),
                 "weight_search": "dirichlet"},
                f"BLEND: add best {name} member (#{add_id})")
    return fn


BLEND_QUEUE = [_mk_add_named("XGB"), _mk_add_named("LGB"), _mk_add_named("TEENC"),
               _mk_add_named("FEATPRUNE"), _blend_remove_weakest]


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES.get(lineage_id, "?")
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
        queue = SOLO_QUEUES.get(name, [])
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


def inject_explore_burst(tree, _root_id):
    pool = solo_pool(tree)
    seeds = []
    if len(pool) >= 2:
        seeds.append(({"kind": "blend", "members": sorted(pool),
                       "weight_search": "dirichlet"},
                      f"[EXPL_MEGABLEND] explore-burst: kitchen-sink blend of the entire "
                      f"solo pool ({len(pool)} members) -- MANDATORY per "
                      f"07_tree_search.md Section 3. Note this node's score is fitted "
                      f"in-sample and will be re-judged leave-fold-out before any "
                      f"submission (aug-2022: the mega-blend won the tree and LOST the "
                      f"honest gate)"))
    p = dc(LGB_BASE)
    p.update(num_leaves=255, learning_rate=0.02, min_child_samples=20)
    seeds.append(({"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}},
                  "[EXPL_DEEPLGB] explore-burst long-shot: high-capacity LGB "
                  "(leaves 255, lr 0.02, mcs 20) -- 140k rows may reward capacity "
                  "(s3e11's large-data inversion of the shallow-is-better prior)"))
    p2 = dc(CAT_BASE)
    p2.update(depth=8, learning_rate=0.03, l2_leaf_reg=8.0)
    seeds.append(({"kind": "solo", "model": "cat", "params": p2,
                   "features": {"drop": []}},
                  "[EXPL_DEEPCAT] explore-burst long-shot: deeper + slower + more "
                  "regularized CatBoost (depth 8, lr 0.03, l2 8)"))
    p3 = dc(LGB_BASE)
    p3.update(num_leaves=31, min_child_samples=100)
    seeds.append(({"kind": "solo", "model": "lgb", "params": p3,
                   "features": {"drop": []},
                   "te": ["City", "Profession", "Degree", "Name"]},
                  "[EXPL_TEMIX] explore-burst long-shot: fold-aligned TE *and* native "
                  "categoricals together at mid capacity -- the two encodings have only "
                  "been raced separately"))
    return seeds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def dry_run():
    winner, root_cfg, root_acc, cands = pick_root()
    print(f"[dry-run] stage-3 solo scores: {cands}")
    print(f"[dry-run] root = {winner} (acc {root_acc:.6f}), cfg={root_cfg}")
    for name in ["LGB", "XGB", "LGBSH", "LR"]:
        print(f"  {name}: {seed_pool_member(name)(root_cfg)[1][:90]}...")
    for name, fn in [("TEENC", seed_teenc), ("BOUNDARYPUSH", seed_boundarypush),
                     ("FEATPRUNE", seed_featprune), ("SEEDBAG", seed_seedbag)]:
        r = fn(root_cfg)
        print(f"  {name}: {(r[1][:90] + '...') if r else '(skipped)'}")
    print("[dry-run] OK -- no training performed.")


def main():
    t_start = time.time()
    resumed = os.path.exists(TREE_PATH)
    tree = hv3.load_search_state(TREE_PATH) if resumed else hv3.new_tree(COMP)
    hv3.init_budget(tree)

    if not resumed:
        winner, root_cfg, root_acc, cands = pick_root()
        print(f"[root] stage-3 solo accuracies: {cands}")
        print(f"[root] chosen: {winner} ({root_acc:.6f}) -- retraining for "
              f"digit-for-digit verification")
        mutation = (f"root: strongest stage-3 solo ({winner}), retrained via "
                    f"eval_s4e11_run2.evaluate_solo for independent digit-for-digit "
                    f"verification against the stage-3 accuracy {root_acc:.6f}")
        nid, _, info = eval_and_add(tree, None, mutation, root_cfg, is_root=True)
        got = acc_of(info)
        assert got is not None and round(got, 6) == round(root_acc, 6), (
            f"ROOT VERIFICATION FAILED: retrained acc {got} != stage-3 {root_acc}")
        print(f"[root] verified digit-for-digit: {got:.6f} == {root_acc:.6f}")
    else:
        print(f"[resume] loaded tree with {len(tree['nodes'])} nodes "
              f"({hv3.n_evaluated(tree)} evaluated)")

    root_id = tree["root_id"]
    root_cfg = next(n for n in tree["nodes"] if n["id"] == root_id)["config"]
    existing = _lineage_seed_ids(tree)
    for name, nid in existing.items():
        LINEAGE_NAMES[nid] = name

    builders = [(n, seed_pool_member(n)) for n in ("LGB", "XGB", "LGBSH", "LR")]
    builders += [("TEENC", seed_teenc), ("BOUNDARYPUSH", seed_boundarypush),
                 ("FEATPRUNE", seed_featprune), ("SEEDBAG", seed_seedbag)]
    for name, mk in builders:
        if name in existing:
            continue
        r = mk(root_cfg)
        if r is None:
            print(f"[seed {name}] skipped -- not applicable to this root")
            continue
        cfg, desc = r
        nid, _dup, info = eval_and_add(tree, root_id, f"[{name}] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = name
            print(f"[seed {name}] node #{nid} acc={acc_of(info)} "
                  f"({info['wall_s']}s)" if info else f"[seed {name}] node #{nid}")
    if "BLEND" not in existing:
        cfg, desc = seed_blend(tree)
        nid, _dup, info = eval_and_add(tree, root_id, f"[BLEND] {desc}", cfg)
        if nid is not None:
            LINEAGE_NAMES[nid] = "BLEND"
            print(f"[seed BLEND] node #{nid} acc={acc_of(info)}")

    burst_injected = bool(_driver_state(tree).get("burst_injected"))
    it = 0
    while (not hv3.should_stop(tree) and it < ITER_SAFETY_CAP
           and (time.time() - t_start) < MAX_WALL_S):
        it += 1
        phase = tree["search_state"]["budget"]["phase"]
        if phase == "explore_burst" and not burst_injected:
            print(f"\n>>> PHASE -> explore_burst (n_eval={hv3.n_evaluated(tree)}); "
                  f"injecting mega-blend + long-shot seeds")
            for cfg, desc in inject_explore_burst(tree, root_id):
                name = desc.split("]")[0][1:]
                nid, _dup, info = eval_and_add(tree, root_id, desc, cfg)
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
                           f"{LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                           f"(#{lineage_id}) -- driver-side plateau"))
                print(f"[plateau] lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                      f"(#{lineage_id}) exhausted -- marked plateaued")
            if st.get("active_lineage") == lineage_id:
                st["active_lineage"] = None
            hv3.save_search_state(tree, TREE_PATH)
            continue
        nid, _dup, info = eval_and_add(tree, parent_id, desc, child_cfg)
        if nid is None:
            continue
        gb = hv3.global_best(tree)
        print(f"[eval {it}] node #{nid} (parent #{parent_id}, "
              f"lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)}) "
              f"acc={acc_of(info)} wall={info['wall_s']}s "
              f"n_eval={hv3.n_evaluated(tree)} "
              f"global_best={-gb['score'] if gb else None}")

    gb = hv3.global_best(tree)
    print(f"\n=== stopped: phase={tree['search_state']['budget']['phase']} "
          f"reason={tree['search_state']['budget'].get('stop_reason')} ===")
    print(f"n_evaluated={hv3.n_evaluated(tree)}  wall={time.time() - t_start:.1f}s")
    print(f"global best: node #{gb['id']} acc={-gb['score']:.6f} "
          f"kind={gb['config'].get('kind')}")
    print(f"dedup_rejections={len(_dedup_rejections(tree))}")
    print(f"backtrack_log={tree['search_state'].get('backtrack_log')}")
    print(f"cost_guard_log={tree['search_state'].get('cost_guard_log')}")
    hv3.save_search_state(tree, TREE_PATH)
    print(f"\nRESULT {json.dumps(dict(n_evaluated=hv3.n_evaluated(tree), best_node=gb['id'], best_acc=-gb['score']))}")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run()
    else:
        main()
