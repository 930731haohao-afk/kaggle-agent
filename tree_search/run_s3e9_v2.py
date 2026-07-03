"""tree_search/run_s3e9_v2.py — drives the harness_v2 (Phase E-1, REVENGE MATCH)
candidate-tree search for playground-series-s3e9 (Concrete Compressive Strength, RMSE,
minimize).

--- Why a revenge match ---
v1 (tree_search/run_s3e9.py + eval_s3e9.py, single-model-only nodes) LOST on this exact
comp: best node #1 (CatBoost solo, 12.07459) never beat the linear iteration's 7-way
seed-bagged blend (12.07003) -- STATUS.md's Phase C-2a verdict named the structural
cause explicitly: "to actually beat this competition's current best score, Stage 4
needs the search space widened beyond single-model nodes ... the real historical gains
... came entirely from seed-bagging/blending, a move this single-model-per-node
prototype structurally cannot make." harness_v2's ensemble-default node space (`kind`:
"solo"|"blend", recommendation #1) is exactly that widening. This run answers: does it
flip the verdict on THIS SAME comp/data/CV?

v1's own experiments_tree.json is left completely untouched (a separate, first-ever
two-tree comparison for this task-runner setup, noted in STATUS.md); this run writes to
its own experiments_tree_v2.json.

--- Budget strategy (task brief) ---
Root + 4 legacy pool members are REUSED from scripts/_round23_pool.npz (Round 2/3's
already-trained OOF/pred columns: LGB, XGB, CAT, LGB_tuned, LGB_tuned_seed2) via
eval_s3e9_v2.load_legacy_solo() -- load + digit-for-digit RMSE recompute (verified
against STATUS.md's historical values BEFORE any node is added, see main()'s startup
log), NOT a retrain. This is the "verify root reproduces digit-for-digit BEFORE
searching" check, done essentially for free. The freed training budget goes to:
  - seed-family expansion (linear iteration's own STATUS.md finding: "seed bagging is
    the only lever that reliably improved this comp's blend after tuning/denoising both
    plateaued" -- CAT_SEEDFAM, LGB_SEEDFAM, LGB_TUNED_SEEDFAM lineages below)
  - boundary-push probes on CAT depth/l2 (CAT_DEPTH, CAT_L2 -- s3e11's winning lever:
    at 360k-row scale CatBoost's Optuna optimum reversed to MORE capacity than small-
    data intuition suggests; s3e9 is small (5.4k rows) + label-noise-capped, the
    opposite regime from s3e11, so the honest expectation here is that MORE capacity
    should NOT help and the existing depth=6/l2=6 hand-set config is already near the
    right boundary -- this is a confirm-or-deny probe, not assumed to reverse)
  - one deliberately-diverse deep/low-reg LGB solo (DEEPLGB, mirrors s3e11's own
    DEEPLGB pattern, applied here as a diversity source for the blend pool rather than
    an expected-to-win solo, since s3e9's own evidence is capacity generally hurts)
  - blend-heavy search once the solo pool exists: composition variants (add/remove
    member, a z-score-standardized "rank-average"-family composition variant per the
    task brief), weight-search-method comparison (dirichlet vs grid_simplex, members
    permitting)

Explicitly NOT re-tested (per task brief + STATUS.md's own "tried, don't retry" list):
duplicate-group target smoothing (STATUS.md exp #5, harmful: 12.07347->12.08122),
explicit interaction-term features (STATUS.md exp #3, harmful: 12.07347->12.09483),
robust-loss (huber/fair) objectives (v1's ROBUST lineage, harmful: 12.11061->12.16-22),
feature-subtraction sweeps (v1's FEAT lineage already covered this ground: plateaued at
12.10-12.11, no gain in either direction).

--- Idea-injection experiment (recommendation #4) ---
harness_v2.suggest_priors(COMP_META) is called once at startup; COMP_META's keywords
hit "### RMSE/極偏態目標" (metric="rmse"), "### 重複列/標籤噪音" (tag "重複列"),
"### 小樣本(<10k 列)" (tag "小樣本"), "## 超參調校(Optuna)" (tag "optuna"), "##
Ensemble/後處理" (tag "ensemble") -- 20 bullets total. Every mutation queue entry tags
its description with `[PRIOR Pk]` (k = the bullet's index in that returned list) when
directly shaped by one of those bullets, or `[PRIOR none]` otherwise, so prior-usage
win-rate can be computed post-hoc exactly as run_s3e11.py does.

--- v2-specific plumbing (identical convention to run_s3e11.py) ---
`find_dup` compares result-STRIPPED config hashes (with blend member lists SORTED)
before spending compute on an eval; rejections are logged to DEDUP_REJECTIONS and bump
a per-lineage retry offset so a rejected proposal doesn't stall the lineage's queue
position.

Root: hand-set CatBoost (depth=6, l2_leaf_reg=6, iterations=4000, lr=0.03) -- the
single strongest solo in the ENTIRE linear-iteration pool (STATUS.md: 12.07459, no
mutation in v1's 20-node search ever beat it), matching the task brief's explicit
instruction to seed "the known-good solos (CAT root family, hand-reg LGB, tuned LGB
from linear)".
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
import eval_s3e9_v2 as ev  # noqa: E402

COMP = "playground-series-s3e9"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree_v2.json")
TARGET_NODES = 26       # task brief: >=18 evaluated nodes; margin for backtracks/dedup
                        # (bumped 22->26 after adding the 2 historical-seed-repro solos)
MAX_WALL_S = 25 * 60     # hard stop -- well under the ~30 min budget (per-node cost here
                        # is seconds, not minutes: legacy reuse is ~0s, new CAT solos
                        # ~3s, new LGB solos ~7s, blend nodes are pure weight-search)
EVAL_TIMEOUT_S = 90
LINEAR_BEST = 12.070034  # linear iteration's best, FULL PRECISION (exp #8 in
                        # experiments.json: "score": 12.070034; STATUS.md's prose only
                        # displays the 5-decimal-rounded 12.07003 -- using the full-
                        # precision value here so "beat"/"tie"/"lose" is judged fairly
                        # against the actual stored number, not its rounded display)
V1_TREE_BEST = 12.07459  # v1 (single-model-only harness) best -- the score v2 must beat
                        # to flip the revenge-match verdict

COMP_META = {"metric": "rmse", "tags": ["重複列", "小樣本", "optuna", "ensemble"]}
# matches "### RMSE/極偏態目標", "### 重複列/標籤噪音", "### 小樣本(<10k 列)",
# "## 超參調校(Optuna)", "## Ensemble/後處理"

DEDUP_REJECTIONS = []
_dedup_offset = {}


def dc(cfg):
    return copy.deepcopy(cfg)


def strip_result(cfg):
    """See run_s3e11.py's identical helper docstring: eval_and_add merges the
    evaluator's OUTPUT `result` dict into the stored config for record-keeping, which
    would make hv2.add_node's own built-in dedup never fire. This driver's own dedup
    check strips "result" first so it compares only the part of the config that was
    actually PROPOSED. Blend member lists are additionally SORTED before hashing so two
    blends over the same member SET reached in different append-order are recognized as
    semantic duplicates."""
    out = {k: v for k, v in cfg.items() if k != "result"}
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
# Root + reused-pool params: exact reproduction of the linear iteration's known-good
# solos (STATUS.md / scripts/train.py / scripts/train_optuna_pool.py).
# ---------------------------------------------------------------------------
CAT_ROOT_PARAMS = dict(depth=6, l2_leaf_reg=6.0, iterations=4000, learning_rate=0.03,
                        random_seed=42)
ROOT_CONFIG = {"kind": "solo", "model": "cat", "params": dc(CAT_ROOT_PARAMS),
               "features": {"drop": []}}
ROOT_MUTATION = ("root: s3e9's strongest known solo -- hand-set CatBoost (depth=6, "
                 "l2_leaf_reg=6) -- OOF RMSE 12.07459 (STATUS.md; also v1's global-best "
                 "node #1, no mutation across v1's 20-node single-model-only search "
                 "ever beat it), REUSED from scripts/_round23_pool.npz (verified "
                 "digit-for-digit, no retrain) [PRIOR P5 -- CatBoost's ordered-boosting "
                 "regularization resists the 56%-dup-row label-noise ceiling better "
                 "than LGB/XGB's default capacity]")

LGB_ORIG_PARAMS = dict(num_leaves=15, max_depth=5, min_child_samples=25, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.7, reg_alpha=2.0,
                       reg_lambda=4.0, learning_rate=0.02, n_estimators=3000,
                       early_stopping_rounds=150, random_state=42)
XGB_ORIG_PARAMS = dict(max_depth=4, min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=2.0, reg_lambda=4.0, learning_rate=0.02, n_estimators=3000,
                       early_stopping_rounds=150, random_state=42)
LGB_TUNED_PARAMS = dict(n_estimators=3000, learning_rate=0.019795655587585677,
                        num_leaves=11, max_depth=3, min_child_samples=31,
                        subsample=0.6018504151952752, subsample_freq=1,
                        colsample_bytree=0.5006835428888052,
                        reg_alpha=0.1120861378640153, reg_lambda=0.3020804107290036)

LEGACY_SPECS = {
    # name -> (legacy_cache_column_name, expected historical RMSE, config)
    "LGB_ORIG": ("LGB", 12.11061,
                 {"kind": "solo", "model": "lgb", "params": dc(LGB_ORIG_PARAMS),
                  "features": {"drop": []}}),
    "XGB_ORIG": ("XGB", 12.12086,
                 {"kind": "solo", "model": "xgb", "params": dc(XGB_ORIG_PARAMS),
                  "features": {"drop": []}}),
    "LGB_TUNED": ("LGB_tuned", 12.12074,
                  {"kind": "solo", "model": "lgb",
                   "params": dict(dc(LGB_TUNED_PARAMS), random_state=42),
                   "features": {"drop": []}}),
    "LGB_TUNED_SEED2": ("LGB_tuned_seed2", 12.10843,
                        {"kind": "solo", "model": "lgb",
                         "params": dict(dc(LGB_TUNED_PARAMS), random_state=1042),
                         "features": {"drop": []}}),
}
LEGACY_DESCS = {
    "LGB_ORIG": ("model-type: hand-regularized LGB (num_leaves=15,depth=5,L1=2,L2=4) "
                 "-- STATUS.md's root single-model reference, solo OOF 12.11061, "
                 "REUSED from scripts/_round23_pool.npz [PRIOR P6 -- regularization + "
                 "features together is what fixed LGB's initial overfit on this "
                 "dup-noisy data]"),
    "XGB_ORIG": ("model-type: hand-regularized XGB (depth=4,min_child_weight=8,L1=2,"
                 "L2=4) -- weakest of the 3 base learners (12.12086) but kept as a "
                 "free diversity source for the blend pool per 'weight search often "
                 "zeroes weak members, that is a feature not a bug' [PRIOR P19 -- "
                 "zero-weight members cost nothing to keep, REUSED from "
                 "scripts/_round23_pool.npz]"),
    "LGB_TUNED": ("model-type: Optuna-tuned LGB (60-trial full-5fold-CV objective, "
                  "STATUS.md Round 2) -- solo OOF 12.12074, WORSE than hand-reg LGB "
                  "solo yet its seed variants earned real blend weight in the linear "
                  "run [PRIOR P12 -- add tuned model to pool, don't replace the "
                  "original, REUSED from scripts/_round23_pool.npz]"),
    "LGB_TUNED_SEED2": ("seed variation: Optuna-tuned LGB, random_state 42->1042 -- "
                        "STATUS.md Round 3's seed-bag member, solo OOF 12.10843, "
                        "earned 22.7% blend weight in the 5-way pool [PRIOR P8/P13 -- "
                        "seed bagging is the single most reliable lever on this "
                        "label-noise-capped comp, REUSED from "
                        "scripts/_round23_pool.npz]"),
}


# ---------------------------------------------------------------------------
# Six NEW first-generation SOLO lineages -- the only ones needing real training among
# the first-gen seeds (root + 4 legacy members above cost ~0s each).
# ---------------------------------------------------------------------------
def seed_cat_seedfam():
    p = dict(dc(CAT_ROOT_PARAMS), random_seed=7)
    cfg = {"kind": "solo", "model": "cat", "params": p, "features": {"drop": []}}
    desc = ("seed-family expansion off the root CAT config (seed 42->7) -- the linear "
            "run's own Phase B found seed-bagging the dominant CatBoost member gave "
            "its single largest round-4 gain (12.07143->12.07003); this run extends "
            "the seed family with NEW seeds (not the linear run's exact seed=1042) so "
            "the blend pool gets independently-random members rather than reproducing "
            "already-known numbers [PRIOR P8 -- seed bagging is the only reliable "
            "lever left once tuning/denoising both plateaued on this comp]")
    return cfg, desc


def seed_cat_depth():
    p = dict(dc(CAT_ROOT_PARAMS), depth=8)
    cfg = {"kind": "solo", "model": "cat", "params": p, "features": {"drop": []}}
    desc = ("BOUNDARY-PUSH probe (s3e11's winning lever, transplanted): push CAT depth "
            "6->8 (more capacity) while holding l2/seed fixed -- s3e9 is small "
            "(5.4k rows) + label-noise-capped, the OPPOSITE regime from s3e11's "
            "360k-row 'more capacity wins' reversal, so the honest expectation is this "
            "should NOT help (confirm, not assumed) [PRIOR P17 -- direct test of "
            "whether the large-data capacity reversal transfers to small noisy data; "
            "prior evidence (P5/P3) says it should not]")
    return cfg, desc


def seed_cat_l2():
    p = dict(dc(CAT_ROOT_PARAMS), l2_leaf_reg=10.0)
    cfg = {"kind": "solo", "model": "cat", "params": p, "features": {"drop": []}}
    desc = ("BOUNDARY-PUSH probe (opposite axis from CAT_DEPTH): push l2_leaf_reg "
            "6.0->10.0 (more regularization) while holding depth/seed fixed -- tests "
            "whether root's l2=6 undershot the noise-ceiling sweet spot in the "
            "'regularize harder' direction v1's own CAT lineage only nudged once "
            "(l2 6->9, tied/lost) [PRIOR P3/P5 -- small noisy data rewards "
            "regularization over capacity, push further to find the true boundary]")
    return cfg, desc


def seed_lgb_seedfam():
    p = dict(dc(LGB_ORIG_PARAMS), random_state=2024)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("seed-family expansion of the hand-reg LGB baseline (random_state "
            "42->2024) -- the linear run never seed-bagged the ORIGINAL LGB (only the "
            "Optuna-tuned one), untested extension of the validated seed-bagging "
            "recipe to a different base config [PRIOR P8 -- seed bagging, applied to "
            "a pool member the linear run itself never tried it on]")
    return cfg, desc


def seed_deeplgb():
    p = dict(num_leaves=63, max_depth=8, min_child_samples=10, subsample=0.9,
             subsample_freq=1, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.1,
             learning_rate=0.03, n_estimators=3000, early_stopping_rounds=150,
             random_state=42)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("model-variant: DELIBERATELY diverse deep/high-capacity/light-reg LGB "
            "(num_leaves 15->63, max_depth 5->8, reg_alpha 2.0->0.1, reg_lambda "
            "4.0->0.1, min_child_samples 25->10) -- s3e9's own evidence (P3/P5/P6) "
            "says capacity should HURT here (opposite of s3e11's large-data finding), "
            "so this is a deliberate diversity source for the blend pool (mirrors "
            "s3e11's own DEEPLGB pattern) rather than an expected solo winner -- worth "
            "testing since a genuinely different bias/variance profile can still add "
            "blend value even with a worse solo score [PRIOR none -- diversity-for-"
            "blend hypothesis, not itself evidenced on this comp]")
    return cfg, desc


def seed_lgb_tuned_seedfam():
    p = dict(dc(LGB_TUNED_PARAMS), random_state=3000)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("seed-family expansion of the Optuna-tuned LGB, 3rd independent seed "
            "(random_state=3000, distinct from the linear run's own 42/1042/2042) -- "
            "STATUS.md's Round 4 noted diminishing-but-still-positive returns on the "
            "3rd seed (-0.00140 there); honest test of whether a 3rd tuned-LGB seed "
            "here is still marginally useful or genuinely exhausted [PRIOR P8/P13 -- "
            "seed bagging after tuning is the cheapest residual gain, extending the "
            "seed family one further]")
    return cfg, desc


def seed_cat_histrepro():
    p = dict(dc(CAT_ROOT_PARAMS), random_seed=1042)
    cfg = {"kind": "solo", "model": "cat", "params": p, "features": {"drop": []}}
    desc = ("EXACT reproduction of the linear iteration's CAT_seed2 (STATUS.md Round 4, "
            "same depth=6/l2=6 as root, random_seed 42->1042) -- the linear run's own "
            "single largest Phase-B round-4 gain came from seed-bagging exactly this "
            "config; this member's individual OOF was NOT separately cached in "
            "scripts/_round23_pool.npz (only the final 7-way blend was), so unlike the "
            "4 legacy pool members it needs a real (cheap, ~2-3s) retrain rather than a "
            "load -- included so the v2 blend pool has direct access to the exact "
            "member composition that produced 12.07003, not just this run's own new "
            "seeds [PRIOR P8/P13 -- seed bagging, direct reproduction of the comp's own "
            "best-evidenced instance of the lever]")
    return cfg, desc


def seed_lgbtuned_histrepro():
    p = dict(dc(LGB_TUNED_PARAMS), random_state=2042)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("EXACT reproduction of the linear iteration's LGB_tuned_seed3 (STATUS.md "
            "Round 4, same Optuna-tuned hyperparams, random_state 42->2042) -- the "
            "other new member of the historical 7-way 12.07003 blend; individual OOF "
            "not separately cached (only the final blend was), real retrain needed "
            "[PRIOR P8/P13 -- seed bagging, direct reproduction]")
    return cfg, desc


NEW_SOLO_SEED_SPECS = [
    ("CAT_SEEDFAM", seed_cat_seedfam), ("CAT_DEPTH", seed_cat_depth),
    ("CAT_L2", seed_cat_l2), ("LGB_SEEDFAM", seed_lgb_seedfam),
    ("DEEPLGB", seed_deeplgb), ("LGB_TUNED_SEEDFAM", seed_lgb_tuned_seedfam),
    ("CAT_HISTREPRO", seed_cat_histrepro), ("LGBT_HISTREPRO", seed_lgbtuned_histrepro),
]
ALL_SOLO_NAMES = list(LEGACY_SPECS) + [n for n, _ in NEW_SOLO_SEED_SPECS]


# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    out = {}
    all_names = ALL_SOLO_NAMES + ["BLEND"]
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"]:
            for name in all_names:
                if name not in out and n["mutation"].startswith(f"[{name}]"):
                    out[name] = n["id"]
    return out


def solo_pool(tree):
    """All evaluated kind=='solo' node ids, best (lowest RMSE) first."""
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    members = solo_pool(tree)
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet", "compose": "raw"}
    desc = (f"ensemble seed: {len(members)}-way blend of the FULL first-generation "
            f"solo pool {members} (root CAT + 4 legacy-reused + 6 newly-trained "
            f"seed-family/boundary-push/diversity solos) CACHED OOF, dirichlet weight "
            f"search -- the recommendation #1 mechanism itself: this is the exact move "
            f"v1's single-model-only harness could not make [PRIOR none -- generic "
            f"ensemble-default node-space seed]")
    return cfg, desc


def _mk_add_named(name, prior_tag):
    def fn(tree, parent_cfg):
        seed_ids = _lineage_seed_ids(tree)
        if name not in seed_ids:
            return None
        add_id = seed_ids[name]
        if add_id in parent_cfg["members"]:
            return None
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        desc = (f"BLEND: add {name} solo member (#{add_id}) -> {len(child['members'])}-way, "
                f"dirichlet weight search {prior_tag}")
        return child, desc
    return fn


def _blend_add_next_unused(tree, parent_cfg):
    pool = solo_pool(tree)
    for add_id in pool:
        if add_id not in parent_cfg["members"]:
            child = dc(parent_cfg)
            child["members"] = parent_cfg["members"] + [add_id]
            child.pop("result", None)
            desc = (f"BLEND: add next-best unused solo pool member #{add_id} -> "
                    f"{len(child['members'])}-way [PRIOR none -- generic pool-growth, "
                    f"weight search arbitrates]")
            return child, desc
    return None


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
            f"leaner {len(child['members'])}-way, test whether it was truly adding "
            f"value or just being tolerated [PRIOR P19 -- removing (near-)zero-weight "
            f"members is zero-cost, s3e11's own R1 validated this]")
    return child, desc


def _blend_compose_zscore(tree, parent_cfg):
    if parent_cfg.get("compose") == "zscore":
        return None  # already tried on this exact member set
    child = dc(parent_cfg)
    child["compose"] = "zscore"
    child.pop("result", None)
    desc = ("BLEND: composition-variant probe -- z-score-standardize each member's OOF "
            "(own mean/std) before weight search, rescale via the target's own "
            "mean/std before scoring -- a distinct blend parameterization from plain "
            "weighted-average-of-raw-values (the task brief's 'rank-average' "
            "composition-variant lever, adapted for a continuous RMSE target rather "
            "than a ranking metric) [PRIOR none -- experience.md's only related "
            "finding (s3e7 rank-average blend tuning) was noise-level on AUC, an even "
            "weaker case exists here since RMSE is not rank-based; honest single-node "
            "test, not assumed to help]")
    return child, desc


def _blend_method_swap(tree, parent_cfg):
    n_members = len(parent_cfg["members"])
    if n_members > 5:
        return None  # grid_simplex only supported up to 5 members (harness_v2 constraint)
    child = dc(parent_cfg)
    cur = child.get("weight_search", "dirichlet")
    child["weight_search"] = "grid_simplex" if cur == "dirichlet" else "dirichlet"
    child.pop("result", None)
    desc = (f"BLEND: weight-SEARCH-METHOD comparison {cur}->{child['weight_search']} on "
            f"the same {n_members} members -- grid_simplex is exhaustive (0.05-step "
            f"simplex, similar spirit to the linear run's own 0.05-step grid) vs "
            f"dirichlet's random+refine search [PRIOR none -- generic ensemble-"
            f"mechanism re-check]")
    return child, desc


BLEND_QUEUE = [
    _blend_remove_weakest,
    _blend_compose_zscore,
    _blend_add_next_unused,
    _blend_remove_weakest,
    _blend_method_swap,
]


def blend_fallback(tree, parent_cfg, attempt):
    """Post-queue blend mutations, find_dup-checked HERE before being returned (see
    run_s3e11.py's identical rationale: never live-lock re-proposing the same
    duplicate). Alternates add/remove until genuinely exhausted -> forced backtrack."""
    pool = solo_pool(tree)
    for add_id in [nid for nid in pool if nid not in parent_cfg["members"]]:
        child = dc(parent_cfg)
        child["members"] = parent_cfg["members"] + [add_id]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            desc = (f"BLEND fallback: add next-best unused pool member #{add_id} -> "
                    f"{len(child['members'])}-way [PRIOR none -- fallback]")
            return child, desc
    prior = parent_cfg.get("result") or {}
    weights = prior.get("weights")
    members = parent_cfg["members"]
    if weights and len(members) > 2:
        idx = int(np.argmin(weights))
        child = dc(parent_cfg)
        child["members"] = [m for i, m in enumerate(members) if i != idx]
        child.pop("result", None)
        if find_dup(tree, child) is None:
            desc = (f"BLEND fallback: remove weakest member #{members[idx]} (all "
                    f"additions duplicate) [PRIOR none -- fallback]")
            return child, desc
    if parent_cfg.get("weight_search") == "dirichlet" and len(members) <= 5:
        child = dc(parent_cfg)
        child["weight_search"] = "grid_simplex"
        child.pop("result", None)
        if find_dup(tree, child) is None:
            return child, "BLEND fallback: method swap to grid_simplex [PRIOR none]"
    return None  # genuinely exhausted -> forced backtrack


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    seed_key = "random_state" if c["model"] in ("lgb", "xgb") else "random_seed"
    c["params"] = dc(c["params"])
    c["params"][seed_key] = 5000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): "
            f"same config, alt {seed_key}={5000 + attempt} [PRIOR P8 -- seed bagging "
            f"as a universal fallback lever on this comp]")
    return c, desc


# ---------------------------------------------------------------------------
# Per-lineage authored mutation queues (SOLO): fn(parent_config) -> (child_config, desc)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"] = dict(c["params"])
    c["params"].update(params_update)
    return c


CAT_SEEDFAM_QUEUE = [
    lambda c: (_bump(c, random_seed=99),
               "CAT_SEEDFAM: 3rd new seed (99) -- honest test of whether the seed pool "
               "is exhausted after 2 new seeds or still has marginal juice [PRIOR P8]"),
]

CAT_DEPTH_QUEUE = [
    lambda c: (_bump(c, depth=10),
               "CAT_DEPTH: push capacity FURTHER (depth 8->10) -- if depth=8 already "
               "lost to root, this brackets whether the trend is monotonic or the "
               "boundary sits somewhere between 6 and 8 [PRIOR P17 -- continue the "
               "boundary-push probe in the same direction]"),
    lambda c: (_bump(c, depth=4),
               "CAT_DEPTH: OPPOSITE direction from the seed -- depth 6->4 (less "
               "capacity than root), bracket the boundary from both sides [PRIOR P3 -- "
               "small noisy data rewards regularization, test whether root's depth=6 "
               "itself already undershot"),
]

CAT_L2_QUEUE = [
    lambda c: (_bump(c, l2_leaf_reg=14.0),
               "CAT_L2: push regularization FURTHER (l2 10->14) -- brackets whether "
               "more L2 keeps helping or has already turned [PRIOR P3/P5]"),
    lambda c: (_bump(c, l2_leaf_reg=3.0),
               "CAT_L2: OPPOSITE direction -- l2 6->3 (less regularization than root), "
               "bracket from both sides [PRIOR P17 -- boundary-push, opposite arm]"),
]

LGB_SEEDFAM_QUEUE = [
    lambda c: (_bump(c, random_state=3000),
               "LGB_SEEDFAM: 2nd new seed (3000) for the hand-reg LGB family [PRIOR P8]"),
]

DEEPLGB_QUEUE = [
    lambda c: (_bump(c, num_leaves=127, reg_alpha=0.01, reg_lambda=0.01),
               "DEEPLGB: push capacity further -- num_leaves 63->127, reg_alpha "
               "0.1->0.01, reg_lambda 0.1->0.01 [PRIOR none -- continue the deliberate-"
               "diversity direction, bracket how far capacity can go before this "
               "comp's noise ceiling makes it actively worse]"),
    lambda c: (_bump(c, num_leaves=31, reg_alpha=0.5, reg_lambda=0.5),
               "DEEPLGB: pull back toward the middle (num_leaves 63->31, more reg than "
               "the seed but still less than root) [PRIOR P3 -- test an intermediate "
               "capacity point between root's num_leaves=15 and the deep seed's 63]"),
]

LGB_TUNED_SEEDFAM_QUEUE = [
    lambda c: (_bump(c, random_state=4000),
               "LGB_TUNED_SEEDFAM: 4th independent seed (4000) of the Optuna-tuned "
               "LGB -- extends the family past the linear run's own 3-seed stopping "
               "point (which cited diminishing gains) [PRIOR P8/P13 -- honest test of "
               "whether the 4th seed is genuinely exhausted"),
]

SOLO_QUEUES = {
    "CAT_SEEDFAM": CAT_SEEDFAM_QUEUE, "CAT_DEPTH": CAT_DEPTH_QUEUE,
    "CAT_L2": CAT_L2_QUEUE, "LGB_SEEDFAM": LGB_SEEDFAM_QUEUE,
    "DEEPLGB": DEEPLGB_QUEUE, "LGB_TUNED_SEEDFAM": LGB_TUNED_SEEDFAM_QUEUE,
    # legacy-reused lineages + the 2 historical-repro seeds: no authored queue ->
    # immediate solo_fallback (seed variant)
    "LGB_ORIG": [], "XGB_ORIG": [], "LGB_TUNED": [], "LGB_TUNED_SEED2": [],
    "CAT_HISTREPRO": [], "LGBT_HISTREPRO": [],
}
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
        if result is None:
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
def reuse_legacy(tree, mutation, cfg, legacy_name, expected_score, is_root=False):
    """Adds a node WITHOUT calling ev.evaluate() -- loads the already-trained OOF/pred
    from scripts/_round23_pool.npz via ev.load_legacy_solo (which itself asserts
    digit-for-digit consistency with the historical STATUS.md score), caches it into
    this driver's own harness_v2 cache dir under the predicted node id, and records
    wall_s=0.0 with a `reused_from` marker in the stored result."""
    nid = hv2.next_id(tree)
    oof, pred, score = ev.load_legacy_solo(legacy_name, expected_score)
    ev.hv2.cache_oof(ev.CACHE_DIR, nid, oof, pred=pred, rmse=score)
    result = {"n_feats": len(ev.ALL_FEATURES) - len(cfg.get("features", {}).get("drop", [])),
              "rmse": round(score, 6), "reused_from": f"scripts/_round23_pool.npz:{legacy_name}"}
    stored_cfg = {**cfg, "result": result}
    if is_root:
        real_nid = hv2.add_root(tree, mutation, stored_cfg, round(score, 6), "evaluated", 0.0)
    else:
        real_nid, dup = hv2.add_node(tree, tree["root_id"], mutation, stored_cfg,
                                      round(score, 6), "evaluated", 0.0, allow_duplicate=True)
        assert dup is None
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    hv2.save(tree, TREE_PATH)
    return real_nid, result


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


def rmse_of(r):
    if r is None:
        return None
    if r.get("result") and "rmse" in r["result"]:
        return r["result"]["rmse"]
    return r.get("score")


# ---------------------------------------------------------------------------
# Prior-usage analysis (idea-injection experiment): "win" = child's score beat its
# direct parent's score (lower RMSE = improvement, no sign flip).
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


def verify_digit_for_digit():
    """Task brief: 'verify root digit-for-digit BEFORE searching'. Loads every legacy
    column from scripts/_round23_pool.npz and asserts RMSE reproduces the historical
    STATUS.md value, printing confirmation -- run BEFORE any node is added to the tree."""
    print("=== digit-for-digit legacy verification (BEFORE searching) ===")
    checks = [("CAT", 12.07459), ("LGB", 12.11061), ("XGB", 12.12086),
              ("LGB_tuned", 12.12074), ("LGB_tuned_seed2", 12.10843)]
    for name, expected in checks:
        oof, pred, score = ev.load_legacy_solo(name, expected)
        print(f"  {name}: recomputed RMSE={score:.6f} vs historical {expected} -> OK "
              f"(|diff|={abs(score - expected):.2e})")
    print("=== verification passed ===\n")


def main():
    t_start = time.time()
    verify_digit_for_digit()

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

    # --- root: REUSED from scripts/_round23_pool.npz's CAT column ---
    if not tree["nodes"]:
        nid, r = reuse_legacy(tree, ROOT_MUTATION, ROOT_CONFIG, "CAT", 12.07459, is_root=True)
        print(f"[root] #{nid} RMSE={r['rmse']} wall_s=0.0 (REUSED, verified digit-for-digit)")

    # --- 4 legacy pool members: REUSED from scripts/_round23_pool.npz ---
    for name, (legacy_name, expected, cfg) in LEGACY_SPECS.items():
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        mutation = f"[{name}] {LEGACY_DESCS[name]}"
        nid, r = reuse_legacy(tree, mutation, cfg, legacy_name, expected)
        print(f"[{name} seed] #{nid} RMSE={r['rmse']} wall_s=0.0 (REUSED, verified digit-for-digit)")

    # --- 6 NEW first-generation SOLO lineages (real training) ---
    for name, seed_fn in NEW_SOLO_SEED_SPECS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} RMSE={rmse_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # --- BLEND lineage: seeded with the FULL first-generation solo pool ---
    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} RMSE={rmse_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = ALL_SOLO_NAMES + ["BLEND"]
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
                            f"mutation proposals exhausted -> forced backtrack")))
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
        gb_rmse = gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"RMSE={rmse_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_RMSE={gb_rmse} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    gb_rmse = gb["score"] if gb else None
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    n_reused = sum(1 for n in tree["nodes"] if n["status"] == "evaluated"
                   and (n["config"].get("result") or {}).get("reused_from"))
    prior_stats = prior_usage_summary(tree)
    tree["prior_usage_summary"] = {
        "n_informed": len(prior_stats["informed"]), "n_uninformed": len(prior_stats["uninformed"]),
        "informed_win_rate": prior_stats["informed_win_rate"],
        "uninformed_win_rate": prior_stats["uninformed_win_rate"],
    }
    tree["dedup_rejections"] = DEDUP_REJECTIONS
    # first evaluated-node index (1-based) at which global_best first reached/beat LINEAR_BEST
    evals_to_match = None
    evals_to_beat_v1 = None
    running_best = None
    for i, n in enumerate([n for n in tree["nodes"] if n["status"] == "evaluated"], start=1):
        s = n["score"]
        running_best = s if running_best is None else min(running_best, s)
        if running_best <= LINEAR_BEST and evals_to_match is None:
            evals_to_match = i
        if running_best < V1_TREE_BEST and evals_to_beat_v1 is None:
            evals_to_beat_v1 = i
    tree["evals_to_match_linear_best"] = evals_to_match
    tree["evals_to_beat_v1_tree_best"] = evals_to_beat_v1
    hv2.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend, "
          f"{n_reused} reused-from-legacy-cache), {len(tree['nodes'])} total, "
          f"wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} RMSE={gb_rmse} mutation={gb['mutation'][:150]}")
    print(f"v1 tree-search best: {V1_TREE_BEST} (tree {'BEAT' if gb_rmse < V1_TREE_BEST else ('MATCHED' if gb_rmse == V1_TREE_BEST else 'did not beat')} it, evals_to_beat={evals_to_beat_v1})")
    print(f"Linear-iteration best: {LINEAR_BEST} (tree {'BEAT' if gb_rmse < LINEAR_BEST else ('MATCHED' if gb_rmse == LINEAR_BEST else 'did not beat')} it, evals_to_match={evals_to_match})")
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
