"""tree_search/run_s3e5.py — drives the ERA-inspired candidate-tree search for
playground-series-s3e5 (Wine Quality, ordinal regression, QWK — Phase C-2c: the FIRST
tree-search generalization test on a DISCRETIZED metric).

Per experience.md's biggest s3e5 lesson, the search's root is not the plain baseline LGB
but the Optuna-DIRECT-post-rounder-QWK-tuned LGB from scripts/iterate.py (solo QWK
0.56244) -- reusing that recipe's OUTPUT (the tuned hyperparams), not re-running Optuna
inside the tree search. Six first-generation SOLO lineages are seeded for diversity:

  CAT       - Optuna-tuned CatBoost (scripts/iterate.py output, solo QWK ~0.56466)
  XGB       - vanilla/untuned XGB -- UNTESTED in the linear run's tuning effort (only
              LGB and CAT got an Optuna pass there), so this lineage's mutation queue is
              genuinely new ground for the tree search to explore
  LGBNUDGE  - a second, independently-mutating LGB-tuned lineage (distinct from the root
              lineage) so two different hyperparam-exploration paths exist off the same
              starting point
  FEAT      - root's tuned-LGB hyperparams + a feature-subset mutation (untested in the
              linear run, which never dropped features after the initial 11->21 build)
  CATORIG   - vanilla/untuned CatBoost (cheap diversity source, known blend contributor)
  LGBORIG   - vanilla/untuned LGB (cheap diversity source, known blend contributor)

...then a BLEND lineage (root LGBTUNED + CAT + XGB, dirichlet weight search) is seeded,
which is where deep composition happens -- exactly the pattern that let s3e14's tree
search reach an ensemble score no solo mutation alone could.

Per the task brief, explicitly NOT re-tested as nodes here (already proven dead ends in
STATUS.md): isotonic regression, multiclass+expected-value decode, and seed-bagging
(zero blend weight twice in the linear run, exp #6/#9).

SIGN CONVENTION: harness.py assumes lower-is-better scores; eval_s3e5.py hands it -QWK.
All prints below convert back to human-readable (positive, higher-is-better) QWK.
"""
import copy
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness  # noqa: E402
import eval_s3e5 as ev  # noqa: E402

COMP = "playground-series-s3e5"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 40         # pass 1: 26 nodes/~410s (best QWK 0.56766, 0.00003 shy of linear best);
                          # pass 2 (cap 34): spent its 8 nodes exhausting the weak solo lineages and
                          # hit the cap right before every lineage was plateaued, so the harness's
                          # reopen-once rule never fired; pass 3 (cap 40) lets it fire and hands the
                          # reopened BLEND lineage the fresh solo pool members (#15/#20) its
                          # fallback never got to consume. Total eval time stays well under budget.
MAX_WALL_S = 30 * 60      # hard stop, matches the ~30 min training budget
EVAL_TIMEOUT_S = 120
LINEAR_BEST = 0.56769     # linear-iteration's best (6-way blend, STATUS.md exp #8)


def dc(cfg):
    return copy.deepcopy(cfg)


def qwk_of(r):
    """Human-readable QWK from an eval() result dict (works for both solo & blend)."""
    if r.get("result") and "qwk" in r["result"]:
        return r["result"]["qwk"]
    return -r["score"] if r.get("score") is not None else None


# ---------------------------------------------------------------------------
# Root: Optuna-tuned LGB from scripts/iterate.py's tune_lgb step (cache/lgb_tuned_params.json),
# the linear-iteration's single strongest learner (solo post-rounder QWK 0.56244) and the
# biggest lever identified in STATUS.md/experience.md (direct-QWK Optuna objective).
# ---------------------------------------------------------------------------
LGB_TUNED_PARAMS = dict(
    num_leaves=57, max_depth=3, learning_rate=0.016850482169027743, min_child_samples=40,
    subsample=0.9520104938890754, colsample_bytree=0.7120571432335079,
    reg_alpha=0.7955110174517752, reg_lambda=0.001041891430753302, n_estimators=345,
)
CAT_TUNED_PARAMS = dict(
    depth=4, learning_rate=0.023711050140388417, l2_leaf_reg=6.458015909315451,
    iterations=333, random_strength=1.2907627214036907, min_data_in_leaf=27,
)

ROOT_CONFIG = {
    "kind": "solo", "model": "lgb",
    "params": dict(LGB_TUNED_PARAMS, random_state=42),
    "features": {"drop": []},
}
ROOT_MUTATION = ("root: s3e5 current-best single learner -- Optuna direct-post-rounder-QWK-"
                 "tuned LGB from scripts/iterate.py's tune_lgb step (STATUS.md exp #5 origin), "
                 "solo QWK ~0.56244; the fair single-model reference point for this tree-"
                 "search run and the biggest linear-iteration lever (Optuna objective = "
                 "post-rounder QWK directly, not RMSE)")


# ---------------------------------------------------------------------------
# Six first-generation SOLO lineage seeds.
# ---------------------------------------------------------------------------
def seed_cat():
    cfg = {"kind": "solo", "model": "cat", "params": dict(CAT_TUNED_PARAMS, random_seed=42),
           "features": {"drop": []}}
    desc = ("Optuna-tuned CatBoost (scripts/iterate.py's tune_cat step, STATUS.md exp #8 "
            "origin) -- solo QWK ~0.56466, second-strongest single learner; key blend-"
            "diversity source (ordered boosting differs structurally from LGB)")
    return cfg, desc


def seed_xgb():
    cfg = {"kind": "solo", "model": "xgb",
           "params": dict(objective="reg:squarederror", n_estimators=1500, learning_rate=0.03,
                          max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          min_child_weight=5, random_state=42),
           "features": {"drop": []}}
    desc = ("vanilla/untuned XGB (STATUS.md exp #3 unchanged config, solo QWK ~0.50847) -- "
            "UNTESTED tuning direction: the linear-iteration run only Optuna-tuned LGB and "
            "CAT, never XGB, so this lineage's hyperparam mutations are genuinely new "
            "ground for the tree search")
    return cfg, desc


def seed_lgbnudge():
    p = dc(ROOT_CONFIG["params"])
    p.update(n_estimators=500, learning_rate=0.012)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("second, independent LGB-tuned exploration lineage off the same Optuna-found "
            "hyperparams (n_estimators 345->500, lr 0.0169->0.012 -- slower/longer pace) "
            "-- distinct from the root lineage so two different local-search paths exist "
            "around the tuned optimum")
    return cfg, desc


def seed_feat():
    # Drop the weakest-EDA-correlation engineered ratios (fixed_to_volatile_acid,
    # citric_to_volatile_acid, sugar_to_alcohol, acid_ph_ratio) -- untested subtraction;
    # the linear run only ever ADDED features (11->21), never pruned.
    drop = ["fixed_to_volatile_acid", "citric_to_volatile_acid", "sugar_to_alcohol", "acid_ph_ratio"]
    cfg = {"kind": "solo", "model": "lgb", "params": dc(ROOT_CONFIG["params"]),
           "features": {"drop": drop}}
    desc = ("feature-subset SUBTRACTION on the root's tuned-LGB hyperparams: drop the 4 "
            "weakest-correlation engineered ratios (fixed/citric_to_volatile_acid, "
            "sugar_to_alcohol, acid_ph_ratio) -> 17-feature test, UNTESTED in the linear "
            "run (which only ever added features, 11->21, never pruned)")
    return cfg, desc


def seed_catorig():
    cfg = {"kind": "solo", "model": "cat", "params": dict(random_seed=42), "features": {"drop": []}}
    desc = ("vanilla/untuned CatBoost (STATUS.md exp #3 unchanged config, solo QWK ~0.52517) "
            "-- cheap diversity source, known (5%-weighted) blend contributor in the "
            "linear-iteration champion")
    return cfg, desc


def seed_lgborig():
    cfg = {"kind": "solo", "model": "lgb", "params": dict(random_state=42), "features": {"drop": []}}
    desc = ("vanilla/untuned LGB (STATUS.md exp #3 unchanged config, solo QWK ~0.50547) -- "
            "cheap diversity source, the ~5%-weighted member that survives in the linear-"
            "iteration champion blend")
    return cfg, desc


SOLO_SEEDS = [("CAT", seed_cat), ("XGB", seed_xgb), ("LGBNUDGE", seed_lgbnudge),
              ("FEAT", seed_feat), ("CATORIG", seed_catorig), ("LGBORIG", seed_lgborig)]


# ---------------------------------------------------------------------------
# BLEND lineage
# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    out = {}
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name, _ in SOLO_SEEDS:
                if name not in out and n["mutation"].startswith(f"[{name}]"):
                    out[name] = n["id"]
    return out


def _best_node_in_lineage(tree, lineage_id):
    best = None
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["id"] == tree["root_id"]:
            continue
        if harness.lineage_of(tree, n["id"]) == lineage_id:
            if best is None or n["score"] < best["score"]:
                best = n
    return best


def _id_for(tree, name):
    if name == "ROOT":
        return tree["root_id"]
    lineage_id = _lineage_seed_ids(tree)[name]
    best = _best_node_in_lineage(tree, lineage_id)
    return best["id"] if best else lineage_id


def solo_pool(tree):
    """All evaluated kind=='solo' node ids, best (highest QWK == lowest -QWK score) first."""
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["CAT"], ids["XGB"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    desc = (f"ensemble seed: 3-way blend of root LGBTUNED(#{tree['root_id']}) + "
            f"CAT_tuned(#{ids['CAT']}) + XGB(#{ids['XGB']}) CACHED OOF, dirichlet weight "
            f"search + OptimizedRounder -- this tree-search's entry point into ensemble "
            f"space; no retraining, evaluates in a fraction of a second")
    return cfg, desc


def _mk_add_named(name):
    def fn(tree, parent_cfg):
        add_id = _id_for(tree, name)
        if add_id in parent_cfg["members"]:
            return None
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        desc = (f"BLEND: add {name} solo member (#{add_id}) -> {len(child['members'])}-way, "
                f"dirichlet weight search + OptimizedRounder")
        return child, desc
    return fn


def _blend_remove_weakest(tree, parent_cfg):
    # parent_cfg IS the parent node's stored config (propose_child passes it directly,
    # not a copy) -- eval_and_add already merged the evaluator's `result` (incl. weights)
    # into that stored config, so it can be read straight off here.
    weights = (parent_cfg.get("result") or {}).get("weights")
    members = parent_cfg["members"]
    if not weights or len(members) <= 2:
        return None
    idx_min_member = min(members, key=lambda m: weights.get(str(m), weights.get(m, 1.0)))
    child = dc(parent_cfg)
    child["members"] = [m for m in members if m != idx_min_member]
    desc = (f"BLEND: remove lowest-weight member #{idx_min_member} -> leaner "
            f"{len(child['members'])}-way, test whether it was truly adding value or just "
            f"being tolerated by the weight search")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("CATORIG"),     # -> 4-way
    _mk_add_named("LGBORIG"),     # -> 5-way, mirrors STATUS.md's final 6-way pool composition
    _mk_add_named("FEAT"),        # -> 6-way, UNTESTED feature-subset member
    _mk_add_named("LGBNUDGE"),    # -> 7-way, second independent LGB-tuned path
    _blend_remove_weakest,        # leaner blend probe
]


def blend_fallback(tree, parent_cfg, attempt):
    pool = solo_pool(tree)
    unused = [nid for nid in pool if nid not in parent_cfg["members"]]
    if unused:
        add_id = unused[0]
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        desc = f"BLEND fallback: add next-best unused pool member #{add_id} -> {len(child['members'])}-way"
        return child, desc
    child = dc(parent_cfg)
    desc = f"BLEND fallback: queue exhausted & pool fully included, re-run weight search (attempt {attempt})"
    return child, desc


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    seed_key = "random_state" if c["model"] in ("lgb", "xgb") else "random_seed"
    c["params"][seed_key] = 3000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): same "
            f"config, alt {seed_key}={3000 + attempt} (NOTE: seed variants are a known "
            f"dead end per experience.md -- only used as a last-resort filler once every "
            f"authored mutation is exhausted, never counted on for a win)")
    return c, desc


# ---------------------------------------------------------------------------
# Per-lineage authored mutation queues (SOLO)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"].update(params_update)
    return c


CAT_QUEUE = [
    lambda c: (_bump(c, bagging_temperature=1.0),
               "CAT: add bagging_temperature=1.0 (Bayesian-bootstrap row weighting) -- "
               "untested regularization axis for the tuned CatBoost on this data"),
    lambda c: (_bump(c, depth=5),
               "CAT: depth 4->5, probe slightly more capacity around the Optuna optimum"),
    lambda c: (_bump(c, l2_leaf_reg=10.0),
               "CAT: l2_leaf_reg 6.46->10.0, push regularization further along the tuned "
               "direction"),
]

XGB_QUEUE = [
    lambda c: (_bump(c, max_depth=3, min_child_weight=10, reg_alpha=1.0, reg_lambda=3.0),
               "XGB: heavier regularization (max_depth 5->3, min_child_weight 5->10, "
               "reg_alpha/lambda added) -- untested direction, XGB was never Optuna-tuned "
               "in the linear run"),
    lambda c: (_bump(c, learning_rate=0.015, n_estimators=2500),
               "XGB: slower pace (lr 0.03->0.015, more estimators) -- finer optimization, "
               "untested direction"),
    lambda c: (_bump(c, colsample_bytree=0.5, subsample=0.6),
               "XGB: heavier row/feature subsampling for variance reduction on this small, "
               "imbalanced dataset"),
]

LGBNUDGE_QUEUE = [
    lambda c: (_bump(c, num_leaves=45, min_child_samples=30),
               "LGBNUDGE: num_leaves 57->45, min_child_samples 40->30 -- probe slightly "
               "less-regularized capacity around the Optuna optimum"),
    lambda c: (_bump(c, reg_alpha=1.5, reg_lambda=0.01),
               "LGBNUDGE: push L1/L2 regularization further along the Optuna-found "
               "direction (reg_alpha 0.80->1.5, reg_lambda 0.001->0.01)"),
]

FEAT_QUEUE = [
    lambda c: (dict(c, features={"drop": c["features"]["drop"] + ["bound_so2"]}),
               "FEAT: further trim bound_so2 (redundant with free_so2_ratio + the two raw "
               "SO2 columns already present) -> 16 feats"),
    lambda c: (dict(c, features={"drop": [f for f in c["features"]["drop"] if f != "acid_ph_ratio"]}),
               "FEAT: restore acid_ph_ratio only, isolate its marginal effect vs the "
               "aggressive 17-feature trim"),
]

CATORIG_QUEUE = [
    lambda c: (_bump(c, depth=6, learning_rate=0.02, l2_leaf_reg=5.0),
               "CATORIG: nudge toward the tuned-CAT region from the untuned default (depth "
               "6 kept, lr 0.03->0.02, l2_leaf_reg 3.0->5.0) -- cheap probe of whether the "
               "vanilla config is leaving easy gains on the table"),
]

LGBORIG_QUEUE = [
    lambda c: (_bump(c, num_leaves=45, min_child_samples=25, reg_lambda=2.0),
               "LGBORIG: nudge toward more regularization from the untuned default "
               "(num_leaves 31->45 capacity + min_child_samples/reg_lambda up) -- cheap "
               "probe distinct from the Optuna-tuned lineages"),
]

SOLO_QUEUES = {"CAT": CAT_QUEUE, "XGB": XGB_QUEUE, "LGBNUDGE": LGBNUDGE_QUEUE,
               "FEAT": FEAT_QUEUE, "CATORIG": CATORIG_QUEUE, "LGBORIG": LGBORIG_QUEUE}
ROOT_QUEUE = [
    lambda c: (_bump(c, n_estimators=600, learning_rate=0.01),
               "ROOT: slower/longer pace (n_estimators 345->600, lr 0.0169->0.01) around "
               "the Optuna optimum"),
    lambda c: (_bump(c, num_leaves=45),
               "ROOT: num_leaves 57->45, probe less capacity around the Optuna optimum"),
]
LINEAGE_NAMES = {}


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES[lineage_id]
    idx = harness.lineage_size(tree, lineage_id) - 1
    if name == "BLEND":
        result = None
        if idx < len(BLEND_QUEUE):
            result = BLEND_QUEUE[idx](tree, parent_node["config"])
        if result is None:
            attempt = max(idx - len(BLEND_QUEUE), 0)
            result = blend_fallback(tree, parent_node["config"], attempt)
        child_cfg, desc = result
    elif name == "ROOTQ":
        queue = ROOT_QUEUE
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    else:
        queue = SOLO_QUEUES[name]
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, child_cfg, is_root=False):
    nid = harness.next_id(tree)
    r = ev.evaluate(child_cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
    stored_cfg = child_cfg
    if r["status"] == "evaluated" and r.get("result") is not None:
        stored_cfg = {**child_cfg, "result": r["result"]}
    if is_root:
        real_nid = harness.add_root(tree, mutation, stored_cfg, r["score"], r["status"], r["wall_s"])
    else:
        real_nid = harness.add_node(tree, parent_id, mutation, stored_cfg, r["score"], r["status"], r["wall_s"])
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    harness.save(tree, TREE_PATH)
    return real_nid, r


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = harness.load(TREE_PATH)
        print(f"Resuming existing tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = harness.new_tree(COMP)

    def n_evaluated():
        return sum(1 for n in tree["nodes"] if n["status"] == "evaluated")

    # --- root ---
    if not tree["nodes"]:
        nid, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} QWK={qwk_of(r)} wall_s={r['wall_s']}")

    # --- a small ROOTQ lineage off the root itself, so the tuned-LGB optimum keeps being
    #     locally explored independent of LGBNUDGE's alternate path ---
    already_rootq = any(n["mutation"].startswith("[ROOTQ]")
                         for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_rootq:
        p = dc(ROOT_CONFIG["params"])
        p.update(reg_alpha=0.4)
        cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
        desc = ("[ROOTQ] first ROOTQ-lineage child: reg_alpha 0.80->0.40, probe less L1 "
                f"regularization around the Optuna optimum (parent=#{tree['root_id']})")
        nid, r = eval_and_add(tree, tree["root_id"], desc, cfg)
        print(f"[ROOTQ seed] #{nid} QWK={qwk_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # --- seed the 6 first-generation SOLO lineages ---
    for name, seed_fn in SOLO_SEEDS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} QWK={qwk_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # --- seed the BLEND lineage (needs CAT + XGB solo ids) ---
    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} QWK={qwk_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = ["ROOTQ"] + [n for n, _ in SOLO_SEEDS] + ["BLEND"]
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in all_names:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name

    # --- adaptive tree-search loop ---
    iterations = 0
    while n_evaluated() < TARGET_NODES and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > 100:
            print("Safety cap on iterations reached, stopping.")
            break
        parent_id, lineage_id = harness.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted (no more eligible expansions). Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        nid, r = eval_and_add(tree, parent_id, mutation, child_cfg)
        gb = harness.global_best(tree)
        gb_qwk = -gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"QWK={qwk_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_QWK={gb_qwk} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = harness.global_best(tree)
    gb_qwk = -gb["score"] if gb else None
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} QWK={gb_qwk} mutation={gb['mutation']}")
    print(f"Linear-iteration best: {LINEAR_BEST}  (tree {'BEAT' if gb_qwk > LINEAR_BEST else 'did not beat'} it)")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for e in tree["search_state"]["backtrack_log"]:
        print(" ", e)


if __name__ == "__main__":
    main()
