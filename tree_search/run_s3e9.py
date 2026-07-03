"""tree_search/run_s3e9.py — drives the ERA-inspired candidate-tree search for
playground-series-s3e9 using tree_search/harness.py (selection/plateau/backtrack) and
tree_search/eval_s3e9.py (config -> CV RMSE).

The agent (Claude) is the mutation proposer, per the design: mutations are authored
below as small, hand-written "nudge" functions grounded in STATUS.md / knowledge/
experience.md (label-noise ceiling, what linear iteration already proved dead:
dup-group smoothing and explicit interaction features — NOT retested here). Which
lineage/node actually gets expanded next, and when a backtrack happens, is decided
dynamically by harness.select_next_parent() from the scores as they come in — that is
the part that is *not* scripted, i.e. the actual "tree search replacing linear
iteration."

Five first-generation lineages fan out from the root (root = s3e9's current
hand-regularized single-LGB config, OOF ~12.11061):
  CAT      - model-type switch to CatBoost solo (known strongest single learner)
  XGB      - model-type switch to XGB solo (known weakest of the three, explore if
             regularization/robust-loss nudges can close the gap)
  FEAT     - feature-subset trimming (subtraction, not addition — addition of
             interaction terms was already proven dead in exp #3)
  ROBUST   - objective swap to a robust loss (huber/fair) — untested direction;
             motivated by the 56%-duplicate-feature-row label-noise ceiling
  REG      - hyperparam nudges along the already-proven-good "regularize over
             increase capacity" axis, exploring both shrink and expand directions

Each lineage has a short authored queue of further mutations; when a lineage runs out
of authored ideas it falls back to seed-variation (proven variance-reduction lever
from Phase B of the linear-iteration run) rather than stalling.
"""
import copy
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness  # noqa: E402
import eval_s3e9 as ev  # noqa: E402

COMP = "playground-series-s3e9"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 20          # aim 18-22 evaluated nodes
MAX_WALL_S = 40 * 60       # hard stop, matches the ~40 min training budget
EVAL_TIMEOUT_S = 90


def dc(cfg):
    return copy.deepcopy(cfg)


# ---------------------------------------------------------------------------
# Root config: exact reproduction of scripts/train.py's regularized LGB (OOF 12.11061)
# ---------------------------------------------------------------------------
ROOT_CONFIG = {
    "model": "lgb",
    "params": dict(num_leaves=15, max_depth=5, min_child_samples=25, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=4.0,
                   learning_rate=0.02, n_estimators=3000, early_stopping_rounds=150),
    "features": {"drop": []},
    "postprocess": {"clip_min": 0, "clip_max": None, "clip_oof": False},
}
ROOT_MUTATION = ("root: s3e9 hand-regularized single-LGB config from scripts/train.py "
                  "(num_leaves=15,depth=5,L1=2,L2=4) -- OOF solo ~12.11061 per STATUS.md; "
                  "the fair single-model reference point for this tree-search run")


# ---------------------------------------------------------------------------
# First-generation lineage seeds (5 candidate directions)
# ---------------------------------------------------------------------------
def seed_cat():
    cfg = {
        "model": "cat",
        "params": dict(depth=6, l2_leaf_reg=6.0, iterations=4000, learning_rate=0.03,
                       early_stopping_rounds=200),
        "features": {"drop": []},
        "postprocess": dc(ROOT_CONFIG["postprocess"]),
    }
    desc = ("model-type switch: CatBoost solo (depth6,l2=6) -- known strongest single "
            "learner on this data (STATUS.md 12.07459 vs LGB 12.11061/XGB 12.12086); "
            "ordered-boosting regularization resists the 56%-dup-row label noise best")
    return cfg, desc


def seed_xgb():
    cfg = {
        "model": "xgb",
        "params": dict(max_depth=4, min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=2.0, reg_lambda=4.0, learning_rate=0.02, n_estimators=3000,
                       early_stopping_rounds=150),
        "features": {"drop": []},
        "postprocess": dc(ROOT_CONFIG["postprocess"]),
    }
    desc = ("model-type switch: XGB solo (depth4,min_child_weight=8,L1=2,L2=4) -- "
            "currently the weakest of the three solo learners (12.12086); branch "
            "explores whether extra regularization/robust-loss nudges can close the gap")
    return cfg, desc


def seed_feat():
    drop = ["sqrt_age", "total_mass", "fine_coarse_ratio", "flyash_cement_ratio",
            "slag_cement_ratio", "has_superplasticizer"]
    cfg = {
        "model": "lgb",
        "params": dc(ROOT_CONFIG["params"]),
        "features": {"drop": drop},
        "postprocess": dc(ROOT_CONFIG["postprocess"]),
    }
    desc = ("feature-subset SUBTRACTION (not addition -- exp #3 already proved adding "
            "interaction terms dead, do not retest): drop 6 weak/redundant members "
            "(sqrt_age redundant w/ log_age per EDA; total_mass a raw sum with low "
            "marginal info; fine_coarse/flyash_cement/slag_cement ratios weaker than "
            "the binder-based ratios; has_superplasticizer a low-info flag) -- 16 of 22 "
            "features remain; 'too many features hurts' is a recurring cross-comp "
            "finding (s3e7, s3e9 exp #3) but subtraction on THIS feature set is untested")
    return cfg, desc


def seed_robust():
    p = dc(ROOT_CONFIG["params"])
    p["objective"] = "huber"
    p["alpha"] = 10.0
    cfg = {
        "model": "lgb",
        "params": p,
        "features": {"drop": []},
        "postprocess": dc(ROOT_CONFIG["postprocess"]),
    }
    desc = ("objective swap: LGB huber loss (alpha=10) instead of plain L2 -- untested "
            "direction; hypothesis is that a loss less sensitive to individual residual "
            "magnitude should help given the 56%-duplicate-feature-row label-noise "
            "ceiling (different Strength values for identical recipes), unlike "
            "dup-group target smoothing (STATUS exp #5) which tried to fix the noise "
            "at the DATA level and failed -- this fixes it at the LOSS level instead")
    return cfg, desc


def seed_reg():
    p = dc(ROOT_CONFIG["params"])
    p.update(num_leaves=10, max_depth=4, learning_rate=0.03)
    cfg = {
        "model": "lgb",
        "params": p,
        "features": {"drop": []},
        "postprocess": dc(ROOT_CONFIG["postprocess"]),
    }
    desc = ("hyperparam nudge: shrink capacity further along the already-proven "
            "'regularize over capacity' axis (num_leaves 15->10, depth 5->4, "
            "lr 0.02->0.03 to compensate for fewer boosting steps needed) -- tests "
            "whether root's leaves=15/depth=5 undershot or overshot the noise-ceiling "
            "sweet spot")
    return cfg, desc


LINEAGE_SEEDS = [seed_cat, seed_xgb, seed_feat, seed_robust, seed_reg]


# ---------------------------------------------------------------------------
# Per-lineage authored mutation queues: fn(parent_config) -> (child_config, mutation_desc)
# Indexed by lineage_size(tree, lineage_id) - 1 (0 for the first post-seed child).
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"].update(params_update)
    return c


CAT_QUEUE = [
    lambda c: (_bump(c, l2_leaf_reg=9.0),
               "CAT: l2_leaf_reg 6->9, more L2 given the label-noise ceiling"),
    lambda c: (_bump(c, depth=5),
               "CAT: depth 6->5, less capacity to probe the over/underfit boundary"),
    lambda c: (_bump(c, bagging_temperature=1.0),
               "CAT: add bagging_temperature=1.0 (Bayesian-bootstrap row weighting) -- "
               "untested regularization axis for CatBoost on this data"),
    lambda c: (_bump(c, learning_rate=0.015, iterations=6000),
               "CAT: lr 0.03->0.015 with iterations 4000->6000, finer optimization pace"),
    lambda c: (_bump(c, random_strength=2.0),
               "CAT: add random_strength=2.0 (extra split-score stochasticity) as a "
               "second, independent regularization lever vs label noise"),
]

XGB_QUEUE = [
    lambda c: (_bump(c, max_depth=3),
               "XGB: max_depth 4->3, lean harder into regularization since XGB is "
               "currently the weakest solo learner"),
    lambda c: (_bump(c, reg_alpha=4.0, reg_lambda=8.0),
               "XGB: double L1/L2 (2->4 / 4->8)"),
    lambda c: (_bump(c, learning_rate=0.01),
               "XGB: lr 0.02->0.01, finer pace with the same early-stopping patience"),
    lambda c: (_bump(c, colsample_bytree=0.5, subsample=0.6),
               "XGB: heavier row/feature subsampling (colsample 0.7->0.5, subsample "
               "0.8->0.6) for variance reduction"),
    lambda c: (_bump(c, objective="reg:pseudohubererror"),
               "XGB: robust loss reg:pseudohubererror instead of squared error, same "
               "motivation as the LGB ROBUST lineage"),
]

def _feat(c, drop_list):
    child = dc(c)
    child["features"] = {"drop": sorted(set(drop_list))}
    return child


_FEAT_CORE13 = {"CementComponent", "BlastFurnaceSlag", "FlyAshComponent", "WaterComponent",
                 "SuperplasticizerComponent", "CoarseAggregateComponent",
                 "FineAggregateComponent", "AgeInDays", "log_age", "binder",
                 "water_binder_ratio", "has_slag", "has_flyash"}

FEAT_QUEUE = [
    lambda c: (_feat(c, c["features"]["drop"] + ["agg_binder_ratio"]),
               "FEAT: further trim agg_binder_ratio (15 feats left) -- testing if the "
               "aggregate/binder ratio is also low marginal value"),
    lambda c: (_feat(c, [f for f in c["features"]["drop"] if f != "has_superplasticizer"]
                      + ["water_cement_ratio"]),
               "FEAT: drop the naive water_cement_ratio (EDA: water_binder_ratio "
               "Pearson -0.227 dominates the naive -0.151 version) and restore "
               "has_superplasticizer"),
    lambda c: (_feat(c, [f for f in ev.ALL_FEATURES if f not in _FEAT_CORE13]),
               "FEAT: aggressive trim to a 13-feature core (raw8 + log_age + binder + "
               "water_binder_ratio + has_slag/has_flyash) -- minimal-sufficient-set test"),
    lambda c: (_feat(c, [f for f in c["features"]["drop"] if f != "sqrt_age"]),
               "FEAT: restore sqrt_age only on top of the current best trim, to isolate "
               "its marginal effect in isolation from the other 5 drops"),
]

ROBUST_QUEUE = [
    lambda c: (_bump(c, objective="fair", fair_c=1.0),
               "LGB: swap huber->fair loss (fair_c=1.0), alternate robust-loss family"),
    lambda c: (_bump(c, alpha=5.0),
               "LGB: huber alpha 10->5, more aggressive down-weighting of large residuals"),
    lambda c: (_bump(c, alpha=15.0),
               "LGB: huber alpha 10->15, milder (closer to plain L2) to bracket the "
               "alpha sweet spot from both sides"),
    lambda c: (_bump(c, reg_lambda=6.0),
               "LGB: combine the best huber alpha found so far with reg_lambda 4->6 "
               "(robust loss + extra L2 together)"),
]

REG_QUEUE = [
    lambda c: (_bump(c, num_leaves=7),
               "LGB: num_leaves 10->7, continue shrinking capacity along the same axis"),
    lambda c: (_bump(c, min_child_samples=40),
               "LGB: min_child_samples 25->40, stronger leaf-occupancy regularization "
               "(untested axis -- only leaves/depth/L1L2 were tuned in the linear run)"),
    lambda c: (_bump(c, reg_lambda=6.0),
               "LGB: reg_lambda 4->6, push L2 further along the known-good direction"),
    lambda c: (_bump(c, num_leaves=20, subsample=0.7),
               "LGB: opposite-direction probe -- num_leaves back up to 20 (more "
               "capacity) + subsample 0.8->0.7 (more row bagging), to check whether "
               "root's leaves=15 was already near the sweet spot or whether both "
               "directions from it help"),
]

QUEUES = {
    "CAT": CAT_QUEUE, "XGB": XGB_QUEUE, "FEAT": FEAT_QUEUE,
    "ROBUST": ROBUST_QUEUE, "REG": REG_QUEUE,
}
LINEAGE_NAMES = {}  # lineage_id (node id of the first-gen node) -> short name, filled at runtime


def fallback_mutation(parent_config, attempt):
    """Used once a lineage's authored queue is exhausted: seed-variation, the one
    lever proven to keep paying off after tuning/denoising plateaued in the linear
    iteration (STATUS.md Phase B, exp #6->#8)."""
    c = dc(parent_config)
    seed_key = "random_state" if c["model"] in ("lgb", "xgb") else "random_seed"
    c["params"][seed_key] = 1000 + attempt
    return c, (f"fallback seed-variation (lineage's authored mutation queue exhausted): "
               f"same config, alt {seed_key}={1000 + attempt} -- proven variance-"
               f"reduction lever from Phase B of the linear-iteration run")


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES[lineage_id]
    queue = QUEUES[name]
    idx = harness.lineage_size(tree, lineage_id) - 1  # 0-indexed position of the next mutation
    if idx < len(queue):
        child_cfg, desc = queue[idx](parent_node["config"])
    else:
        child_cfg, desc = fallback_mutation(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


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
        r = ev.evaluate(ROOT_CONFIG, timeout_s=EVAL_TIMEOUT_S)
        harness.add_root(tree, ROOT_MUTATION, ROOT_CONFIG, r["score"], r["status"], r["wall_s"])
        harness.save(tree, TREE_PATH)
        print(f"[root] #{0} score={r['score']} wall_s={r['wall_s']}")

    # --- seed the 5 first-generation lineages directly off the root ---
    names = ["CAT", "XGB", "FEAT", "ROBUST", "REG"]
    for name, seed_fn in zip(names, LINEAGE_SEEDS):
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        # resume-safety: skip seeds already present (matched by mutation prefix)
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            for n in already:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name
            continue
        cfg, desc = seed_fn()
        r = ev.evaluate(cfg, timeout_s=EVAL_TIMEOUT_S)
        nid = harness.add_node(tree, tree["root_id"], f"[{name}] {desc}", cfg,
                                r["score"], r["status"], r["wall_s"])
        LINEAGE_NAMES[nid] = name
        harness.save(tree, TREE_PATH)
        print(f"[{name} seed] #{nid} score={r['score']} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully in case of resume (in case above loop's `continue`
    # branch didn't cover every existing node, e.g. mid-run resume)
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in names:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name

    # --- adaptive tree-search loop ---
    iterations = 0
    while n_evaluated() < TARGET_NODES and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > 60:
            print("Safety cap on iterations reached, stopping.")
            break
        parent_id, lineage_id = harness.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted (no more eligible expansions). Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        r = ev.evaluate(child_cfg, timeout_s=EVAL_TIMEOUT_S)
        nid = harness.add_node(tree, parent_id, mutation, child_cfg,
                                r["score"], r["status"], r["wall_s"])
        harness.save(tree, TREE_PATH)
        gb = harness.global_best(tree)
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"score={r['score']} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best={gb['score'] if gb else None} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = harness.global_best(tree)
    print(f"\nDone. {n_evaluated()} evaluated nodes, {len(tree['nodes'])} total, "
          f"wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} score={gb['score']} mutation={gb['mutation']}")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for ev_ in tree["search_state"]["backtrack_log"]:
        print(" ", ev_)


if __name__ == "__main__":
    main()
