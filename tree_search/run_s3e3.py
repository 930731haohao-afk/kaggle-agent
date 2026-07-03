"""tree_search/run_s3e3.py — drives the harness_v2 (Phase D-2) candidate-tree search for
playground-series-s3e3 (Employee Attrition, ROC-AUC, maximize).

Sweep question (same for every Phase D comp): can harness v2 (ensemble-default node
space + experience-library priors + adaptive plateau + child dedup) match or beat the
linear-iteration best (0.83814, exp #7, 6-way rank-average blend) in FEWER evaluations
than linear iteration took rounds (7 experiments across 4 self-improvement rounds)?

--- Idea-injection experiment (recommendation #4) ---
`harness_v2.suggest_priors(COMP_META)` is called once at startup with this comp's
metric/tag keywords; the returned bullets (verbatim from knowledge/experience.md, no
LLM call inside the harness) are printed, saved into the tree under `tree["priors"]`,
and labelled P0..P(n-1). Every mutation queue entry below tags its description with
`[PRIOR Pk]` when it was directly shaped by that bullet, or `[PRIOR none]` when it comes
from this comp's own STATUS.md "next ideas" list or generic ensemble hygiene instead. A
post-run pass (`prior_usage_summary`) compares the two groups' win rates (child beat its
own parent's score) so the report can say whether prior-informed mutations actually did
better than uninformed ones, not just how many there were.

--- v2-specific plumbing vs run_s3e14.py/run_s3e5.py (v1) ---
harness_v2.add_root/add_node return the SAME two-return-value contract as v1 EXCEPT
add_node returns `(nid, dup_id)` (v1 returned a bare int) because of child dedup
(recommendation #3): if `dup_id is not None`, no node was added. Because this run's
convention (inherited from every prior run_*.py) merges the evaluator's `result` dict
into the STORED config after evaluation (so blend mutations like "remove weakest member"
can read prior weights straight off `parent_cfg["result"]`), the raw stored config is
never byte-identical across two different nodes even when the *proposed* config was --
so hv2.add_node's own built-in dedup (which hashes the full stored config, result
included) would never actually fire. This driver therefore does its own dedup check
(`find_dup`, comparing result-STRIPPED config hashes) BEFORE spending any compute on an
eval, and calls hv2.add_node(..., allow_duplicate=True) to skip the (here-meaningless)
built-in check afterwards. Each rejection is logged to `DEDUP_REJECTIONS` and bumps a
per-lineage retry offset so the next propose_child() call for that lineage advances to
the NEXT mutation-queue item / fallback attempt instead of re-proposing an identical
config forever.

Root: LGB, Optuna direct-full-5-fold-CV-AUC-objective tuned (STATUS.md exp #4, solo OOF
0.837305) -- "the linear winner's tuned-LGB config" per the task brief, and the single
strongest learner in the linear-iteration run (used at weight 0.7 in its final blend).
"""
import copy
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import eval_s3e3 as ev  # noqa: E402

COMP = "playground-series-s3e3"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 22          # aim >=18 evaluated nodes (mix of solo + blend)
MAX_WALL_S = 25 * 60      # hard stop, matches the ~25 min training budget (tiny data)
EVAL_TIMEOUT_S = 120
LINEAR_BEST = 0.838140    # linear-iteration's best (6-way blend, rank-average, STATUS.md exp #7)
LINEAR_ROUNDS = 7         # STATUS.md exp #1-#7 (4 self-improvement rounds after baseline+v1/v2)

COMP_META = {"metric": "auc", "tags": ["10k", "small_sample"]}  # matches "### ROC-AUC(排名指標)"
                                                                  # + "### 小樣本(<10k 列)" headers

DEDUP_REJECTIONS = []  # (attempted_mutation, existing_dup_id) log for the report
_dedup_offset = {}      # lineage_id -> extra propose_child idx offset from rejections


def dc(cfg):
    return copy.deepcopy(cfg)


def strip_result(cfg):
    """v1/v2 convention: eval_and_add merges the evaluator's OUTPUT `result` dict into
    the stored config for record-keeping (so e.g. _blend_remove_weakest can read prior
    weights straight off parent_cfg["result"]). That merge means hv2.add_node's own
    built-in dedup (which hashes the whole stored config) would never fire -- two nodes
    proposing the identical INPUT config always differ in their (result-dependent)
    stored config. This driver's OWN dedup check strips "result" first so it compares
    only the part of the config that was actually PROPOSED, matching recommendation #3's
    intent."""
    return {k: v for k, v in cfg.items() if k != "result"}


def find_dup(tree, cfg):
    target = hv2.config_hash(strip_result(cfg))
    for n in tree["nodes"]:
        if hv2.config_hash(strip_result(n["config"])) == target:
            return n["id"]
    return None


# ---------------------------------------------------------------------------
# Root: exact reproduction of STATUS.md exp #4 (Optuna direct-full-CV-AUC-tuned LGB)
# ---------------------------------------------------------------------------
LGB_TUNED_PARAMS = dict(
    objective="binary", metric="auc", n_estimators=2000,
    learning_rate=0.050253069663926536, num_leaves=3, max_depth=4,
    min_child_samples=60, subsample=0.8995656675058548,
    colsample_bytree=0.7246802411328122, reg_alpha=0.06815791273889091,
    reg_lambda=0.0011513245661312597, random_state=42,
)
ROOT_CONFIG = {
    "kind": "solo", "model": "lgb", "params": dc(LGB_TUNED_PARAMS),
    "features": {"drop": []},
}
ROOT_MUTATION = ("root: s3e3 linear-winner's tuned-LGB config -- Optuna direct-full-5-"
                 "fold-CV-AUC-objective tuning (STATUS.md exp #4), solo OOF AUC "
                 "~0.837305, strongest single learner (weight 0.7 in the linear-"
                 "iteration's final blend) [PRIOR P1: Optuna-direct-final-metric-on-"
                 "full-CV transfers cleanly to AUC, no discretization trap like QWK]")


# ---------------------------------------------------------------------------
# Six first-generation SOLO lineage seeds.
# ---------------------------------------------------------------------------
def seed_lgborig():
    cfg = {"kind": "solo", "model": "lgb",
           "params": dict(objective="binary", metric="auc", n_estimators=2000,
                          learning_rate=0.03, num_leaves=7, min_child_samples=30,
                          subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
                          reg_alpha=1.0, reg_lambda=2.0, random_state=42),
           "features": {"drop": []}}
    desc = ("model-variant: exp #3's original (non-Optuna) LGB, num_leaves=7 heavily "
            "regularized -- solo ~0.832925, cheap diversity source distinct from the "
            "Optuna optimum, known non-zero blend member (weight 0.3 in linear best) "
            "[PRIOR P0: AUC is a ranking metric -- no imbalance weighting, "
            "regularization over capacity is what worked here]")
    return cfg, desc


def seed_xgbtuned():
    cfg = {"kind": "solo", "model": "xgb",
           "params": dict(objective="binary:logistic", n_estimators=2000, eval_metric="auc",
                          early_stopping_rounds=100, random_state=42,
                          learning_rate=0.0468274656210896, max_depth=2,
                          min_child_weight=13, subsample=0.6879111669199495,
                          colsample_bytree=0.8287002153933888,
                          reg_alpha=0.0022483066052387973, reg_lambda=0.8448951877348988),
           "features": {"drop": []}}
    desc = ("model-type: Optuna-tuned XGB (30 trials, TPE, full 5-fold CV AUC objective, "
            "86.5s, one-time pre-tune -- NEVER done for XGB in the linear-iteration run, "
            "which only ever Optuna-tuned LGB) -- solo OOF AUC 0.835633 vs vanilla XGB "
            "0.807634, closing most of the gap to LGB_tuned's 0.837305 "
            "[PRIOR P1: apply the direct-final-metric-on-full-CV Optuna recipe to a "
            "model type it was never tried on]")
    return cfg, desc


def seed_catorig():
    cfg = {"kind": "solo", "model": "cat",
           "params": dict(loss_function="Logloss", eval_metric="AUC", iterations=2000,
                          learning_rate=0.03, depth=5, l2_leaf_reg=8.0, random_seed=42),
           "features": {"drop": []}}
    desc = ("model-type: CatBoost, label/freq-encoded features, exp #3 unchanged config "
            "-- solo ~0.762684, weakest learner; kept ONLY as an inert blend-diversity "
            "seed, no further solo-direction mutation queue (see CATORIG_QUEUE=[]) "
            "[PRIOR P3: CatBoost structurally weak on this ~1.7k-row dataset, zeroed "
            "by weight search twice already -- do not retry as a solo direction]")
    return cfg, desc


def seed_catnative():
    cfg = {"kind": "solo", "model": "cat",
           "params": dict(loss_function="Logloss", eval_metric="AUC", iterations=2000,
                          learning_rate=0.05, depth=3, l2_leaf_reg=16.0, random_seed=42),
           "features": {"drop": [], "native_cat": True}}
    desc = ("model-type: CatBoost w/ native categorical handling (round-3 winning "
            "depth=3/l2=16/lr=0.05 config) -- solo ~0.814259, best CatBoost variant but "
            "still 2.3pt below LGB_tuned; CLOSED item per STATUS.md, no further mutation "
            "queue (CATNATIVE_QUEUE=[]), kept only as a blend-diversity seed "
            "[PRIOR P4: native cat_features narrows the CatBoost gap but cannot flip "
            "the ranking -- necessary but not sufficient, do not retry again]")
    return cfg, desc


def seed_seedbag():
    p = dc(LGB_TUNED_PARAMS)
    p["random_state"] = 2024
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("seed variation: LGB_tuned params, random_state 42->2024 -- STATUS.md exp #5's "
            "5th blend member (solo ~same order as root, seed-bag of the Optuna optimum) "
            "[PRIOR P2: 'add tuned to pool + seed bag' compounds small but real gains on "
            "AUC, diminishing with iteration count]")
    return cfg, desc


def seed_feat():
    drop = ["YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager",
            "YearsSinceLastPromotion", "TotalWorkingYears"]
    cfg = {"kind": "solo", "model": "lgb", "params": dc(LGB_TUNED_PARAMS),
           "features": {"drop": drop}}
    desc = ("feature-subset SUBTRACTION on the tuned-LGB hyperparams: drop the 5 raw "
            "tenure columns now condensed into role_tenure_ratio/mgr_tenure_ratio/"
            "promo_ratio/company_tenure_ratio/income_per_year_worked/age_at_join "
            "(computed upstream in features.py before this drop, so still valid) -- "
            "43-feature test, UNTESTED in the linear-iteration run [PRIOR none -- "
            "sourced from STATUS.md's own 'Next ideas': 'drop redundant raw tenure "
            "columns, keep engineered ratios only -- untested hypothesis']")
    return cfg, desc


SOLO_SEEDS = [("LGBORIG", seed_lgborig), ("XGBTUNED", seed_xgbtuned), ("CATORIG", seed_catorig),
              ("CATNATIVE", seed_catnative), ("SEEDBAG", seed_seedbag), ("FEAT", seed_feat)]


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
        if hv2.lineage_of(tree, n["id"]) == lineage_id:
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
    """All evaluated kind=='solo' node ids, best (highest AUC == lowest -AUC score) first."""
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["XGBTUNED"], ids["LGBORIG"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet", "space": "prob"}
    desc = (f"ensemble seed: 3-way blend of root LGB_tuned(#{tree['root_id']}) + "
            f"XGBTUNED(#{ids['XGBTUNED']}) + LGBORIG(#{ids['LGBORIG']}) CACHED OOF, "
            f"dirichlet weight search, prob-space -- this tree-search's entry point into "
            f"ensemble space; no retraining, evaluates in <1s [PRIOR none -- generic "
            f"ensemble-default node-space seed, the recommendation #1 mechanism itself]")
    return cfg, desc


def _mk_add_named(name, prior_tag):
    def fn(tree, parent_cfg):
        add_id = _id_for(tree, name)
        if add_id in parent_cfg["members"]:
            return None
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        desc = (f"BLEND: add {name} solo member (#{add_id}) -> {len(child['members'])}-way, "
                f"dirichlet weight search, {child.get('space','prob')}-space {prior_tag}")
        return child, desc
    return fn


def _blend_space_swap(tree, parent_cfg):
    child = dc(parent_cfg)
    cur = child.get("space", "prob")
    child["space"] = "rank" if cur == "prob" else "prob"
    child.pop("result", None)
    desc = (f"BLEND: weight-blend SPACE comparison {cur}->{child['space']} on the same "
            f"{len(child['members'])} members -- re-check STATUS.md's 'next idea' "
            f"(linear iteration's rank-average beat prob-blend by only +0.000364, "
            f"flagged as possibly noise-level) [PRIOR none -- STATUS.md next-idea]")
    return child, desc


def _blend_remove_weakest(tree, parent_cfg):
    prior = parent_cfg.get("result") or {}
    weights = prior.get("weights")
    members = parent_cfg["members"]
    if not weights or len(members) <= 2:
        return None
    idx = int(np.argmin(weights))
    weak_id = members[idx]
    child = dc(parent_cfg)
    child["members"] = [m for m in members if m != weak_id]
    child.pop("result", None)
    desc = (f"BLEND: remove lowest-weight member #{weak_id} (w={weights[idx]:.3f}) -> "
            f"leaner {len(child['members'])}-way, test whether it was truly adding value "
            f"or just being tolerated by the weight search [PRIOR none -- generic "
            f"ensemble hygiene, see knowledge/experience.md Ensemble/後處理 section]")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("CATORIG", "[PRIOR none -- STATUS.md exp #7 pattern: let weight "
                              "search decide even for a closed-solo-direction model]"),
    _mk_add_named("CATNATIVE", "[PRIOR P3/P4 -- same closed-item logic, weight search "
                                "is expected to zero it per experience.md]"),
    _mk_add_named("SEEDBAG", "[PRIOR P2 -- add-to-pool + seed-bag]"),
    _blend_space_swap,
    _mk_add_named("FEAT", "[PRIOR none -- untested feature-subset diversity member]"),
    _blend_remove_weakest,
]


def blend_fallback(tree, parent_cfg, attempt):
    pool = solo_pool(tree)
    unused = [nid for nid in pool if nid not in parent_cfg["members"]]
    if unused:
        add_id = unused[0]
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        desc = (f"BLEND fallback: add next-best unused pool member #{add_id} -> "
                f"{len(child['members'])}-way [PRIOR none -- fallback]")
        return child, desc
    child = dc(parent_cfg)
    child.pop("result", None)
    desc = f"BLEND fallback: queue exhausted & pool fully included, re-run weight search (attempt {attempt}) [PRIOR none -- fallback]"
    return child, desc


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    seed_key = "random_state" if c["model"] in ("lgb", "xgb") else "random_seed"
    c["params"][seed_key] = 3000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): same "
            f"config, alt {seed_key}={3000 + attempt} [PRIOR none -- fallback]")
    return c, desc


# ---------------------------------------------------------------------------
# Per-lineage authored mutation queues (SOLO): fn(parent_config) -> (child_config, desc)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"].update(params_update)
    return c


LGBORIG_QUEUE = [
    lambda c: (_bump(c, reg_lambda=4.0),
               "LGBORIG: reg_lambda 2.0->4.0, push L2 further along the known-good "
               "direction [PRIOR P5: small sample + correlated engineered features -> "
               "regularize over capacity]"),
    lambda c: (_bump(c, num_leaves=5, min_child_samples=50),
               "LGBORIG: num_leaves 7->5, min_child_samples 30->50 -- further "
               "regularization nudge [PRIOR P5]"),
]

XGBTUNED_QUEUE = [
    lambda c: (_bump(c, random_state=2024),
               "XGBTUNED: seed variation on the Optuna-tuned params (random_state "
               "42->2024) -- independent seed-bag source for the newly-tuned XGB "
               "[PRIOR P2: add-to-pool + seed-bag]"),
    lambda c: (_bump(c, min_child_weight=25, reg_lambda=2.0),
               "XGBTUNED: min_child_weight 13->25, reg_lambda 0.84->2.0 -- push "
               "regularization further along the Optuna-found direction [PRIOR P5]"),
]

CATORIG_QUEUE = []   # closed item (P3) -- no further solo-direction mutation, fallback only
CATNATIVE_QUEUE = []  # closed item (P4) -- no further solo-direction mutation, fallback only

SEEDBAG_QUEUE = [
    lambda c: (_bump(c, reg_lambda=0.01),
               "SEEDBAG: reg_lambda 0.00115->0.01, push L2 further along the tuned "
               "direction while keeping the seed=2024 bag [PRIOR P5]"),
    lambda c: (_bump(c, random_state=777),
               "SEEDBAG: third seed variant (2024->777) [PRIOR P2: add-to-pool + "
               "seed-bag, diminishing but real gains]"),
]

FEAT_QUEUE = [
    lambda c: (dict(c, features={"drop": c["features"]["drop"] + ["income_per_year_worked"]}),
               "FEAT: further trim income_per_year_worked (redundant with "
               "income_per_joblevel + MonthlyIncome itself) -- 42 feats [PRIOR none -- "
               "untested extension of the STATUS.md next-idea]"),
    lambda c: (dict(c, features={"drop": [f for f in c["features"]["drop"] if f != "TotalWorkingYears"]}),
               "FEAT: restore TotalWorkingYears only, isolate its marginal effect vs the "
               "aggressive 5-column tenure trim [PRIOR none]"),
]

SOLO_QUEUES = {"LGBORIG": LGBORIG_QUEUE, "XGBTUNED": XGBTUNED_QUEUE, "CATORIG": CATORIG_QUEUE,
               "CATNATIVE": CATNATIVE_QUEUE, "SEEDBAG": SEEDBAG_QUEUE, "FEAT": FEAT_QUEUE}
LINEAGE_NAMES = {}  # lineage_id (node id of the first-gen node) -> short name


def propose_child(tree, parent_id, lineage_id):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = LINEAGE_NAMES[lineage_id]
    idx = hv2.lineage_size(tree, lineage_id) - 1 + _dedup_offset.get(lineage_id, 0)
    if name == "BLEND":
        result = None
        if idx < len(BLEND_QUEUE):
            result = BLEND_QUEUE[idx](tree, parent_node["config"])
        if result is None:
            attempt = max(idx - len(BLEND_QUEUE), 0)
            result = blend_fallback(tree, parent_node["config"], attempt)
        child_cfg, desc = result
    else:
        queue = SOLO_QUEUES[name]
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            child_cfg, desc = solo_fallback(parent_node["config"], idx - len(queue))
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


# ---------------------------------------------------------------------------
def eval_and_add(tree, parent_id, mutation, child_cfg, is_root=False, lineage_id=None):
    """Returns (nid_or_None, dup_id_or_None, result_or_None). On dedup rejection, nid is
    None, dup_id is the pre-existing node's id, no eval was run."""
    dup_id = None if is_root else find_dup(tree, child_cfg)
    if dup_id is not None:
        DEDUP_REJECTIONS.append(dict(mutation=mutation, dup_id=dup_id))
        if lineage_id is not None:
            _dedup_offset[lineage_id] = _dedup_offset.get(lineage_id, 0) + 1
        return None, dup_id, None

    nid = hv2.next_id(tree)
    r = ev.evaluate(child_cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
    stored_cfg = child_cfg
    if r["status"] == "evaluated" and r.get("result") is not None:
        stored_cfg = {**child_cfg, "result": r["result"]}
    if is_root:
        real_nid = hv2.add_root(tree, mutation, stored_cfg, r["score"], r["status"], r["wall_s"])
    else:
        real_nid, add_dup = hv2.add_node(tree, parent_id, mutation, stored_cfg, r["score"],
                                          r["status"], r["wall_s"], allow_duplicate=True)
        assert add_dup is None
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    hv2.save(tree, TREE_PATH)
    return real_nid, None, r


def auc_of(r):
    if r is None:
        return None
    if r.get("result") and "auc" in r["result"]:
        return r["result"]["auc"]
    return -r["score"] if r.get("score") is not None else None


# ---------------------------------------------------------------------------
# Prior-usage analysis (idea-injection experiment): compare prior-informed vs
# uninformed mutation win rates. "Win" = child's score beat its direct parent's score
# (lower -AUC = higher AUC = improvement), among evaluated non-root, non-seed nodes
# (mutations proposed by propose_child, i.e. everything the search LOOP added, not the
# hand-authored first-generation seeds -- seeds don't have a comparable "parent" score
# in the same lineage sense).
# ---------------------------------------------------------------------------
_PRIOR_TAG_RE = re.compile(r"\[PRIOR (P\d+|none)")


def prior_usage_summary(tree):
    by_id = {n["id"]: n for n in tree["nodes"]}
    informed, uninformed = [], []
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["parent_id"] is None or n["parent_id"] == tree["root_id"]:
            continue  # skip root and first-gen seeds (no in-lineage parent to compare against)
        m = _PRIOR_TAG_RE.search(n["mutation"])
        if not m:
            continue
        tag = m.group(1)
        parent = by_id.get(n["parent_id"])
        if parent is None or parent["score"] is None or n["score"] is None:
            continue
        won = n["score"] < parent["score"]
        (informed if tag != "none" else uninformed).append(dict(node_id=n["id"], tag=tag, won=won))
    def rate(lst):
        return (sum(1 for x in lst if x["won"]) / len(lst)) if lst else None
    return dict(informed=informed, uninformed=uninformed,
                informed_win_rate=rate(informed), uninformed_win_rate=rate(uninformed))


def main():
    t_start = time.time()
    if os.path.exists(TREE_PATH):
        tree = hv2.load(TREE_PATH)
        print(f"Resuming existing tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv2.new_tree(COMP)
        priors = hv2.suggest_priors(COMP_META)
        tree["priors"] = priors
        print(f"suggest_priors({COMP_META}) -> {len(priors)} bullets:")
        for i, p in enumerate(priors):
            print(f"  P{i}: {p[:200]}")

    def n_evaluated():
        return sum(1 for n in tree["nodes"] if n["status"] == "evaluated")

    # --- root ---
    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} AUC={auc_of(r)} wall_s={r['wall_s']}")

    # --- seed the 6 first-generation SOLO lineages ---
    for name, seed_fn in SOLO_SEEDS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # --- seed the BLEND lineage (needs XGBTUNED + LGBORIG solo ids) ---
    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = [n for n, _ in SOLO_SEEDS] + ["BLEND"]
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in all_names:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name

    # --- adaptive tree-search loop ---
    iterations = 0
    while n_evaluated() < TARGET_NODES and (time.time() - t_start) < MAX_WALL_S:
        iterations += 1
        if iterations > 80:
            print("Safety cap on iterations reached, stopping.")
            break
        parent_id, lineage_id = hv2.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted (no more eligible expansions). Stopping.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id)
        nid, dup, r = eval_and_add(tree, parent_id, mutation, child_cfg, lineage_id=lineage_id)
        if nid is None:
            print(f"DEDUP REJECTED <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"(identical to existing node #{dup}); retrying with next mutation-queue slot")
            continue
        gb = hv2.global_best(tree)
        gb_auc = -gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_AUC={gb_auc} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    gb_auc = -gb["score"] if gb else None
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    prior_stats = prior_usage_summary(tree)
    tree["prior_usage_summary"] = {
        "n_informed": len(prior_stats["informed"]), "n_uninformed": len(prior_stats["uninformed"]),
        "informed_win_rate": prior_stats["informed_win_rate"],
        "uninformed_win_rate": prior_stats["uninformed_win_rate"],
    }
    tree["dedup_rejections"] = DEDUP_REJECTIONS
    hv2.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} AUC={gb_auc} mutation={gb['mutation']}")
    print(f"Linear-iteration best: {LINEAR_BEST} over {LINEAR_ROUNDS} experiments "
          f"(tree {'BEAT' if gb_auc > LINEAR_BEST else 'did not beat'} it)")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for e in tree["search_state"]["backtrack_log"]:
        print(" ", e)
    print(f"Dedup rejections ({len(DEDUP_REJECTIONS)}):")
    for e in DEDUP_REJECTIONS:
        print(" ", e)
    print(f"Prior usage: informed={len(prior_stats['informed'])} "
          f"(win rate={prior_stats['informed_win_rate']}), "
          f"uninformed={len(prior_stats['uninformed'])} "
          f"(win rate={prior_stats['uninformed_win_rate']})")


if __name__ == "__main__":
    main()
