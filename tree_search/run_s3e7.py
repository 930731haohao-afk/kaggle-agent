"""tree_search/run_s3e7.py — drives the harness_v2 (Phase D-3) candidate-tree search for
playground-series-s3e7 (Hotel Reservation Cancellation, ROC-AUC, maximize).

Sweep question (same for every Phase D comp): can harness v2 (ensemble-default node
space + experience-library priors + adaptive plateau + child dedup) match or beat the
linear-iteration best (0.899893, exp #6, LGB(Optuna-tuned)/XGB/CAT 0.54/0.40/0.06 prob
blend, 0.01 grid) in FEWER evaluations than linear iteration took rounds (6 experiments:
#1 baseline, #2-#3 feature iteration, #4-#6 Phase B self-improvement)?

s3e7 is flagged as the HARDEST sweep case: the linear run squeezed only +0.0005 total
over its generic baseline (0.89882 -> 0.899893) across 3 self-improvement rounds, one of
which (Round 2, Optuna-tuned XGB) was REJECTED because it regressed the blend via lost
diversity. Headroom may be near zero here — an honest "tree search also can't move it"
outcome is a valid, useful result. Do NOT force fake gains (no cherry-picked seeds, no
silently-changed folds/features to manufacture a delta).

--- Idea-injection experiment (recommendation #4) ---
harness_v2.suggest_priors(COMP_META) is called once at startup; COMP_META's keywords are
chosen to hit knowledge/experience.md's "### ROC-AUC(排名指標)" (metric="auc"),
"### 高共線性特徵" / "## 特徵工程模式" (tags containing "特徵"/"共線性" — the section
that documents s3e7's OWN "特徵過多反而退步" feature-pruning lesson, exp #2 vs #3),
"## Ensemble/後處理" (tag "ensemble"), and "## 超參調校(Optuna)" (tag "optuna"). Every
mutation queue entry tags its description with `[PRIOR Pk]` when directly shaped by one
of those returned bullets, or `[PRIOR none]` otherwise. `prior_usage_summary` compares
the two groups' win rates at the end.

--- Feature-pruning mutation direction (the brief's specific ask) ---
The root LGB is trained once with `want_importance=True`; its average-gain importance
over the 8 ENGINEERED features (computed on this run, not assumed from EDA) is used to
build the FEATPRUNE seed's drop list from the *actual* weakest members, rather than
guessing. Iter1's regressed 14-feature ("full" variant) set is never reintroduced.

--- v2-specific plumbing vs run_s3e14.py/run_s3e5.py (v1) --- (identical convention to
run_s3e3.py — see that file's docstring for the full rationale) `find_dup` compares
result-STRIPPED config hashes before spending compute on an eval; rejections are logged
to DEDUP_REJECTIONS and bump a per-lineage retry offset.

Root: LGB, Optuna fold-0-proxy-tuned (STATUS.md exp #4, Phase B Round 1) — "the linear
winner's strongest solo config" per the task brief, solo OOF 0.899215, weight 0.54 in the
final 0.899893 blend (the single strongest base learner throughout the linear run).
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
import eval_s3e7 as ev  # noqa: E402

COMP = "playground-series-s3e7"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 22          # aim >=18 evaluated nodes (mix of solo + blend)
MAX_WALL_S = 30 * 60      # hard stop, matches the ~30 min training budget
EVAL_TIMEOUT_S = 200
LINEAR_BEST = 0.899893    # linear-iteration's best (exp #6, prob blend 0.01grid, STATUS.md)
LINEAR_ROUNDS = 6         # STATUS.md exp #1-#6 (baseline + 2 feature iters + 3 Phase-B rounds)

COMP_META = {"metric": "auc", "tags": ["特徵", "共線性", "ensemble", "optuna"]}
# matches "### ROC-AUC(排名指標)", "## 特徵工程模式", "### 高共線性特徵",
# "## Ensemble/後處理", "## 超參調校(Optuna)" headers

DEDUP_REJECTIONS = []  # (attempted_mutation, existing_dup_id) log for the report
_dedup_offset = {}      # lineage_id -> extra propose_child idx offset from rejections


def dc(cfg):
    return copy.deepcopy(cfg)


def strip_result(cfg):
    """See run_s3e3.py's identical helper docstring: eval_and_add merges the evaluator's
    OUTPUT `result` dict into the stored config for record-keeping, which would make
    hv2.add_node's own built-in dedup never fire (two nodes proposing the identical INPUT
    config always differ in their result-dependent stored config). This driver's own
    dedup check strips "result" (and "want_importance", a root-only bookkeeping flag)
    first so it compares only the part of the config that was actually PROPOSED. Blend
    member lists are additionally SORTED before hashing so two blends over the same
    member SET reached in different append-order are recognized as semantic duplicates
    (the dirichlet weight search is symmetric in member order — same set == same search
    space)."""
    out = {k: v for k, v in cfg.items() if k not in ("result", "want_importance")}
    if out.get("kind") == "blend" and "members" in out:
        out = dict(out, members=sorted(out["members"]))
    return out


def find_dup(tree, cfg):
    target = hv2.config_hash(strip_result(cfg))
    for n in tree["nodes"]:
        if hv2.config_hash(strip_result(n["config"])) == target:
            return n["id"]
    return None


# ---------------------------------------------------------------------------
# Root: exact reproduction of STATUS.md exp #4 (Optuna fold-0-proxy-tuned LGB, Phase B R1)
# ---------------------------------------------------------------------------
LGB_TUNED_PARAMS = dict(
    objective="binary", metric="auc", n_estimators=3000,
    learning_rate=0.06811817027632358, num_leaves=178, max_depth=3,
    min_child_samples=47, subsample=0.8888159684496226, subsample_freq=1,
    colsample_bytree=0.5316042232727975, reg_alpha=2.142043500176781,
    reg_lambda=0.011502513321845967, random_state=42, early_stopping_rounds=150,
)
ROOT_CONFIG = {
    "kind": "solo", "model": "lgb", "params": dc(LGB_TUNED_PARAMS),
    "features": {"drop": []}, "want_importance": True,
}
ROOT_MUTATION = ("root: s3e7 linear-winner's strongest solo config -- Optuna fold-0-"
                 "proxy-tuned LGB (50 trials, STATUS.md exp #4 / Phase B Round 1), solo "
                 "OOF AUC 0.899215, single strongest learner (weight 0.54 in the linear-"
                 "iteration's final 0.899893 blend) [PRIOR: AUC is a continuous ranking "
                 "metric, Optuna-direct-final-metric-on-full-CV transfers cleanly, no "
                 "discretization trap like QWK]")

XGB_HAND = dict(objective="binary:logistic", n_estimators=3000, learning_rate=0.03,
                max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42,
                eval_metric="auc", early_stopping_rounds=150)
CAT_HAND = dict(loss_function="Logloss", eval_metric="AUC", iterations=4000,
                learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                random_seed=42, thread_count=-1, early_stopping_rounds=200)


# ---------------------------------------------------------------------------
# Seven first-generation SOLO lineage seeds.
# ---------------------------------------------------------------------------
def seed_xgbhand():
    cfg = {"kind": "solo", "model": "xgb", "params": dc(XGB_HAND), "features": {"drop": []}}
    desc = ("model-type: hand-set XGB, unchanged from STATUS.md exp #4/#6 base models -- "
            "solo OOF 0.898765, 2nd-strongest learner (weight 0.40 in linear best), kept "
            "verbatim (NOT re-Optuna-tuned -- exp #5 showed tuned XGB improves solo but "
            "REGRESSES the blend via lost diversity, so this lineage's mutation queue "
            "nudges regularization only, never re-tunes toward the AUC objective) "
            "[PRIOR none -- STATUS.md exp #4 base model, verbatim]")
    return cfg, desc


def seed_cathand():
    cfg = {"kind": "solo", "model": "cat", "params": dc(CAT_HAND), "features": {"drop": []}}
    desc = ("model-type: hand-set CatBoost, unchanged from STATUS.md exp #4/#6 base "
            "models -- solo OOF 0.896709, weakest learner (weight only 0.06 in linear "
            "best) [PRIOR P19 -- weight search often zeroes weak members and that is a "
            "feature not a bug; kept as blend-diversity seed with a light nudge queue, "
            "weight search arbitrates]")
    return cfg, desc


def seed_seedbag():
    p = dc(LGB_TUNED_PARAMS)
    p["random_state"] = 2024
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("seed variation: LGB_tuned params, random_state 42->2024 -- seed-bagging the "
            "Optuna optimum, UNTESTED in the linear-iteration run (linear never seed-"
            "bagged on this comp) [PRIOR P13 -- 'Optuna fold-proxy tune -> add to pool "
            "(not replace) -> seed bag' is a 4-competition-validated default starting "
            "move; s3e7's linear run stopped before the seed-bag step]")
    return cfg, desc


def seed_regnudge():
    p = dc(LGB_TUNED_PARAMS)
    p["reg_alpha"] = 4.0
    p["num_leaves"] = 90
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("regularization nudge on the tuned-LGB hyperparams: reg_alpha 2.14->4.0, "
            "num_leaves 178->90 (push further along the Optuna-found shallow+regularized "
            "direction) -- this comp is 42k rows, NOT the <10k-row small-sample regime "
            "experience.md's regularization prior is validated on, so this is a direct "
            "test of whether that prior still transfers at medium scale [PRIOR P6 -- "
            "shallow+strong-regularization was Optuna's own optimum on this comp; push "
            "the same direction further by hand]")
    return cfg, desc


def seed_featprune(weak_feats):
    cfg = {"kind": "solo", "model": "lgb", "params": dc(LGB_TUNED_PARAMS),
           "features": {"drop": list(weak_feats)}}
    desc = (f"feature-PRUNING on the tuned-LGB hyperparams: drop the {len(weak_feats)} "
            f"lowest-gain-importance engineered features measured on THIS run's root "
            f"({', '.join(weak_feats)}) -> {len(ev.ALL_FEATURES) - len(weak_feats)} feats "
            f"[PRIOR P3 -- importance-probe pruning of low-value columns; also s3e7's OWN "
            f"exp #2->#3 lesson (14 eng feats 0.89788 REGRESSED below baseline -> trimmed "
            f"to 8 eng feats 0.89939) and s3e3's independent Phase D-2 replication -- "
            f"pushes the SAME direction one step further using measured, not guessed, "
            f"importances]")
    return cfg, desc


def seed_xgbdiv():
    p = dc(XGB_HAND)
    p["max_depth"] = 8
    p["colsample_bytree"] = 0.6
    p["learning_rate"] = 0.05
    cfg = {"kind": "solo", "model": "xgb", "params": p, "features": {"drop": []}}
    desc = ("model-variant: XGB pushed DEEPER/higher-lr (depth 6->8, lr 0.03->0.05, "
            "colsample 0.8->0.6) -- deliberately diversifying AWAY from LGB_tuned's "
            "shallow direction rather than re-tuning toward it, since exp #5 showed "
            "tuning XGB toward the AUC objective converges it onto LGB_tuned's shallow "
            "structure and REGRESSES the blend via lost diversity (-0.000169) [PRIOR "
            "none -- direct response to STATUS.md exp #5's diversity-loss finding, this "
            "run's own 'next idea' rather than an experience.md transfer]")
    return cfg, desc


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["XGBHAND"], ids["CATHAND"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet", "space": "prob"}
    desc = (f"ensemble seed: 3-way blend of root LGB_tuned(#{tree['root_id']}) + "
            f"XGBHAND(#{ids['XGBHAND']}) + CATHAND(#{ids['CATHAND']}) CACHED OOF, "
            f"dirichlet weight search, prob-space -- reproduces the linear-iteration's "
            f"exp #4/#6 base-model trio under harness_v2's ensemble machinery, no "
            f"retraining [PRIOR none -- generic ensemble-default node-space seed, the "
            f"recommendation #1 mechanism itself]")
    return cfg, desc


SOLO_SEED_SPECS = [("XGBHAND", seed_xgbhand), ("CATHAND", seed_cathand),
                    ("SEEDBAG", seed_seedbag), ("REGNUDGE", seed_regnudge),
                    ("XGBDIV", seed_xgbdiv)]  # FEATPRUNE seeded separately (needs importances)


# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    out = {}
    all_names = [n for n, _ in SOLO_SEED_SPECS] + ["FEATPRUNE", "BLEND"]
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name in all_names:
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
            f"{len(child['members'])} members -- re-check STATUS.md exp #6 (rank-average "
            f"0.899884 vs prob 0.899893, a -0.000009 noise-level gap on the linear run's "
            f"3-model trio; re-verify under harness_v2's own blend seeds/members) [PRIOR "
            f"P18 -- blend-layer fine-tuning (rank-average) is noise-level on AUC per "
            f"experience.md, this mutation is a deliberate re-verification of that prior]")
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
            f"or just being tolerated by the weight search [PRIOR P16 -- removing "
            f"(near-)zero-weight members is zero-cost, per experience.md Ensemble/後處理]")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("SEEDBAG", "[PRIOR P2 -- add-to-pool + seed-bag stacks small gains on "
                              "AUC, let weight search decide]"),
    _mk_add_named("FEATPRUNE", "[PRIOR P9 -- tuned/variant members should be ADDED to "
                                "the pool, never replace; different feature subset is a "
                                "genuine diversity source]"),
    _mk_add_named("REGNUDGE", "[PRIOR P9 -- add-to-pool-not-replace, weight search "
                               "arbitrates at zero cost]"),
    _mk_add_named("XGBDIV", "[PRIOR P9 -- add-to-pool-not-replace; deliberate-diversity "
                             "XGB as blend member]"),
    _blend_space_swap,
    _blend_remove_weakest,
]


def blend_fallback(tree, parent_cfg, attempt):
    """Post-queue blend mutations. Every candidate is find_dup-checked HERE, before being
    returned, so the fallback can never live-lock re-proposing the same duplicate (the
    first full run of this driver did exactly that: 73 consecutive dedup rejections of
    'add member #4' against node #14 until the iteration safety cap fired, because the
    old fallback always picked unused[0] regardless of rejection history). Order:
    (1) each unused pool member best-first, (2) space swap on the current member set,
    (3) None -> caller marks the lineage exhausted (forced backtrack)."""
    pool = solo_pool(tree)
    for add_id in [nid for nid in pool if nid not in parent_cfg["members"]]:
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            desc = (f"BLEND fallback: add next-best unused pool member #{add_id} -> "
                    f"{len(child['members'])}-way [PRIOR none -- fallback]")
            return child, desc
    child = dc(parent_cfg)
    cur = child.get("space", "prob")
    child["space"] = "rank" if cur == "prob" else "prob"
    child.pop("result", None)
    if find_dup(tree, child) is None:
        desc = (f"BLEND fallback: all member additions duplicate existing nodes; space "
                f"swap {cur}->{child['space']} on the current {len(child['members'])} "
                f"members instead [PRIOR none -- fallback]")
        return child, desc
    return None  # genuinely exhausted -> forced backtrack


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


XGBHAND_QUEUE = [
    lambda c: (_bump(c, reg_alpha=1.5, reg_lambda=2.0),
               "XGBHAND: reg_alpha 0.5->1.5, reg_lambda 1.0->2.0 -- regularization nudge "
               "[PRIOR none -- generic hand-nudge, no matching evidence bullet at this "
               "42k-row scale]"),
    lambda c: (_bump(c, max_depth=4),
               "XGBHAND: max_depth 6->4 -- test whether LGB_tuned's shallow direction "
               "transfers to XGB too (watch for the exp #5 diversity-loss trap if this "
               "ever gets tuned rather than hand-nudged) [PRIOR none]"),
]

CATHAND_QUEUE = [
    lambda c: (_bump(c, l2_leaf_reg=10.0),
               "CATHAND: l2_leaf_reg 5.0->10.0 -- regularization nudge on the weakest "
               "learner [PRIOR none -- generic hand-nudge]"),
    lambda c: (_bump(c, depth=5),
               "CATHAND: depth 7->5 -- shallower CatBoost variant [PRIOR none]"),
]

SEEDBAG_QUEUE = [
    lambda c: (_bump(c, random_state=777),
               "SEEDBAG: second seed variant (2024->777) [PRIOR P2 -- add-to-pool + "
               "seed-bag, diminishing but real gains on AUC]"),
    lambda c: (_bump(c, reg_lambda=0.05),
               "SEEDBAG: reg_lambda 0.0115->0.05, push L2 further while keeping the "
               "seed=2024 bag [PRIOR none -- generic hand-nudge]"),
]

REGNUDGE_QUEUE = [
    lambda c: (_bump(c, reg_alpha=6.0, num_leaves=60),
               "REGNUDGE: push further -- reg_alpha 4.0->6.0, num_leaves 90->60 [PRIOR "
               "P6 -- shallow+strong-regularization is the direction Optuna itself found "
               "on this very comp (s3e7 exp #4), push one step further]"),
    lambda c: (_bump(c, min_child_samples=80),
               "REGNUDGE: min_child_samples 47->80 -- regularize via leaf-size instead of "
               "leaf-count [PRIOR P6 -- same shallow+regularized direction, different knob]"),
]

FEATPRUNE_QUEUE = [
    None,  # filled in main() once weak_feats/next_weak are known (needs root importances)
    None,
]

XGBDIV_QUEUE = [
    lambda c: (_bump(c, random_state=2024),
               "XGBDIV: seed variation (random_state 42->2024) on the deliberate-"
               "diversity params [PRIOR P2 -- add-to-pool + seed-bag]"),
    lambda c: (_bump(c, learning_rate=0.08, colsample_bytree=0.5),
               "XGBDIV: push diversity direction further -- lr 0.05->0.08, colsample "
               "0.6->0.5 [PRIOR none -- this run's own diversity-seeking direction]"),
]

SOLO_QUEUES = {"XGBHAND": XGBHAND_QUEUE, "CATHAND": CATHAND_QUEUE, "SEEDBAG": SEEDBAG_QUEUE,
               "REGNUDGE": REGNUDGE_QUEUE, "FEATPRUNE": FEATPRUNE_QUEUE, "XGBDIV": XGBDIV_QUEUE}
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
        if result is None:  # blend mutation space genuinely exhausted
            return None, None
        child_cfg, desc = result
    else:
        queue = SOLO_QUEUES[name]
        if idx < len(queue) and queue[idx] is not None:
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
# Prior-usage analysis (idea-injection experiment): see run_s3e3.py's identical helper
# for the full rationale. "Win" = child's score beat its direct parent's score.
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

    # --- root (also computes engineered-feature importances for FEATPRUNE) ---
    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} AUC={auc_of(r)} wall_s={r['wall_s']}")

    root_node = tree["nodes"][tree["root_id"]]
    importance = (root_node["config"].get("result") or {}).get("importance") or {}
    eng_importance = {k: v for k, v in importance.items() if k in ev.ENGINEERED_FEATURES}
    ranked_weak = sorted(eng_importance, key=lambda k: eng_importance[k]) if eng_importance else []
    print(f"Engineered-feature importance (ascending, root LGB): "
          f"{[(k, eng_importance[k]) for k in ranked_weak]}")
    weak2 = ranked_weak[:2] if len(ranked_weak) >= 2 else ranked_weak
    _weak3 = ranked_weak[:3] if len(ranked_weak) >= 3 else ranked_weak

    def seed_featprune_fn():
        return seed_featprune(weak2)

    # --- fill FEATPRUNE's queue now that we know the real weakest features ---
    if len(ranked_weak) >= 3:
        third = ranked_weak[2]
        FEATPRUNE_QUEUE[0] = (lambda c, _third=third: (
            dict(c, features={"drop": c["features"]["drop"] + [_third]}),
            f"FEATPRUNE: further prune next-weakest engineered feature ({_third}) -> "
            f"{len(ev.ALL_FEATURES) - len(c['features']['drop']) - 1} feats [PRIOR P3 -- "
            f"importance-probe pruning is a standing lever, push one more step]"
        ))
    if weak2:
        restore = weak2[0]
        FEATPRUNE_QUEUE[1] = (lambda c, _restore=restore: (
            dict(c, features={"drop": [f for f in c["features"]["drop"] if f != _restore]}),
            f"FEATPRUNE: restore {_restore} only, isolate its marginal effect vs the "
            f"multi-feature trim [PRIOR none -- isolation test]"
        ))

    # --- seed the 5 hand-authored first-generation SOLO lineages + FEATPRUNE + BLEND ---
    all_named_seeds = list(SOLO_SEED_SPECS) + [("FEATPRUNE", seed_featprune_fn)]
    for name, seed_fn in all_named_seeds:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")

    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} AUC={auc_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = [n for n, _ in SOLO_SEED_SPECS] + ["FEATPRUNE", "BLEND"]
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
        if child_cfg is None:  # lineage's mutation space exhausted -> forced backtrack
            st = tree["search_state"]
            if lineage_id not in st["plateaued"]:
                st["plateaued"].append(lineage_id)
                st["backtrack_log"].append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=(f"lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)}'s "
                            f"mutation proposals exhausted (queue done, every remaining "
                            f"candidate is a duplicate of an existing node) -> forced "
                            f"backtrack")))
            hv2.save(tree, TREE_PATH)
            print(f"FORCED BACKTRACK: lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"mutation space exhausted; plateaued={tree['search_state']['plateaued']}")
            continue
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
    # first evaluated-node index (1-based) at which global_best first reached/beat LINEAR_BEST
    evals_to_match = None
    running_best = None
    for i, n in enumerate([n for n in tree["nodes"] if n["status"] == "evaluated"], start=1):
        s = -n["score"]
        running_best = s if running_best is None else max(running_best, s)
        if running_best >= LINEAR_BEST and evals_to_match is None:
            evals_to_match = i
    tree["evals_to_match_linear_best"] = evals_to_match
    hv2.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} AUC={gb_auc} mutation={gb['mutation']}")
    print(f"Linear-iteration best: {LINEAR_BEST} over {LINEAR_ROUNDS} experiments "
          f"(tree {'BEAT' if gb_auc > LINEAR_BEST else ('MATCHED' if gb_auc == LINEAR_BEST else 'did not beat')} it)")
    print(f"Evals to match/beat linear best: {evals_to_match}")
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
