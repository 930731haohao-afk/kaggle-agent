"""tree_search/run_s3e19.py — drives the harness_v2 (Phase D-5) candidate-tree search for
playground-series-s3e19 (Forecast Mini-course Sales, SMAPE, minimize), the sweep's
TIME-SERIES case (TimeSeriesSplit CV, not the KFold every prior Phase-D comp used).

Sweep question: v2 is 3/3 BEAT on KFold-CV comps (s3e3 #12, s3e7 #13, s3e1 #9). Does the
harness generalize to a TimeSeriesSplit-CV comp where (a) the OOF vector only covers a
SUBSET of rows (the first ~306/1826 unique dates are train-only in every fold, never
validated) and (b) the linear-iteration best (10.01946, exp #7) is itself flagged with an
honest "somewhat optimistic" caveat (fold 5 double-dips as both the Optuna objective and
1/5 of the blend-weight-search OOF)?

--- Idea-injection experiment (recommendation #4) ---
harness_v2.suggest_priors(COMP_META) is called once at startup; COMP_META's keywords hit
knowledge/experience.md's "### SMAPE/時序" (metric="smape" — s3e19 is itself the sole
evidence source for every bullet in that section, including the fold-5-proxy lesson and
the rejected ratio-decomposition finding), "## CV 設計" (tag "cv" — the TimeSeriesSplit
CV-scheme-comparison lesson, "絕不跨 CV 方案比較分數"), "## 超參調校(Optuna)" (tag
"optuna"), "## Ensemble/後處理" (tag "ensemble"). Every mutation queue entry tags its
description with `[PRIOR Pk]` when directly shaped by one of those returned bullets, or
`[PRIOR none]` otherwise.

--- Known constraints from the task brief (STATUS.md + experiments.json) ---
- CV MUST be TimeSeriesSplit(n_splits=5) on the 1826 unique dates, IDENTICAL to
  scripts/train.py's get_time_folds() — verified digit-for-digit in eval_s3e19.py
  (114000/136950 rows covered, matching the 5*304*75-series arithmetic) BEFORE this
  driver started searching.
- Do NOT re-test ratio decomposition (STATUS.md Round 1: solo OOF SMAPE 14.75 vs LGB
  10.18, weight-searched to 0 — the structural country/product-share signal does NOT
  beat GBDT here because the real bottleneck, 1-year-ahead grand-total extrapolation
  under a 5-year COVID-disrupted series, affects any method equally).
- Do NOT re-test XGB solo (weight-searched to 0 twice already, dropped from the pool
  per the linear run's own instruction).
- Untried/deprioritized ideas this run DOES test (STATUS.md "Potential improvements"
  + this task's brief): (1) a flat global-multiplier ("auto_scale") postprocess node —
  cheap, honest, structurally distinct from the rejected ratio-decomposition (no
  country/date-conditional structure, just a scalar level-fit on OOF); (2) seed-bagging
  the Optuna-tuned LGB itself (STATUS.md explicitly flags this as untried and notes
  experience.md's s3e5 counterexample that seed-bagging a directly-tuned config can be
  neutral — cheap to test either way); (3) a deliberately-diverse deep/high-capacity LGB
  variant (XGB is banned as a diversity source here, so diversity is sought via LGB
  capacity/regularization direction instead, mirroring s3e7's diversity-preservation
  lesson); (4) calendar-feature-subset probes (feature SUBTRACTION, not addition).

--- Digit-for-digit reproduction note ---
eval_s3e19.py loads the linear run's own on-disk train_processed.csv/test_processed.csv
(scripts/features.py's already-written artifact, not recomputed) so every base-model
score this run produces should closely match STATUS.md/experiments.json's historical
numbers for identical configs (LGB_s42 hand-set params ~10.17758, CAT_s42 ~10.73064,
etc.) — verified as part of building this evaluator (see eval_s3e19.py's IDX-coverage
assertion, 114000 rows, matching exactly).

--- v2-specific plumbing vs run_s3e1.py --- (identical convention — see that file's
docstring for the full rationale) `find_dup` compares result-STRIPPED config hashes
(with blend members SORTED) before spending compute on an eval; rejections are logged
to DEDUP_REJECTIONS and bump a per-lineage retry offset.

Root: LGB, Optuna fold-5-proxy tuned (STATUS.md exp #7, Phase B Round 4) — solo OOF
10.14833, the single strongest member in the linear pool, picked as ROOT to match the
established convention from run_s3e1.py/run_s3e3.py/run_s3e7.py (root = the
Optuna-found config itself).
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
import eval_s3e19 as ev  # noqa: E402

COMP = "playground-series-s3e19"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 22        # aim >=18 evaluated nodes (mix of solo + blend)
MAX_WALL_S = 30 * 60     # hard stop, matches the ~30 min training budget
EVAL_TIMEOUT_S = 200
LINEAR_BEST = 10.01946   # linear-iteration's best (exp #7, 6-way grid-search blend, STATUS.md)
LINEAR_ROUNDS = 7        # STATUS.md exp #1-#7 (baseline + engineered + diagnostic + 4 Phase-B rounds)

COMP_META = {"metric": "smape", "tags": ["cv", "optuna", "ensemble", "時序"]}
# matches "### SMAPE/時序", "## CV 設計", "## 超參調校(Optuna)", "## Ensemble/後處理"

DEDUP_REJECTIONS = []  # (attempted_mutation, existing_dup_id) log for the report
_dedup_offset = {}      # lineage_id -> extra propose_child idx offset from rejections


def dc(cfg):
    return copy.deepcopy(cfg)


def strip_result(cfg):
    """See run_s3e1.py's identical helper docstring: eval_and_add merges the
    evaluator's OUTPUT `result` dict into the stored config for record-keeping, which
    would make hv2.add_node's own built-in dedup never fire. This driver's own dedup
    check strips "result" first so it compares only the part of the config that was
    actually PROPOSED. Blend member lists are additionally SORTED before hashing so two
    blends over the same member SET reached in different append-order are recognized
    as semantic duplicates."""
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
# Root: exact reproduction of STATUS.md exp #7 (Optuna fold-5-proxy-tuned LGB)
# ---------------------------------------------------------------------------
LGB_TUNED_PARAMS = dict(
    learning_rate=0.037125798221945665, num_leaves=20, min_child_samples=38,
    subsample=0.6105071416341405, colsample_bytree=0.9049138633231821,
    reg_alpha=0.0010341384364001556, reg_lambda=0.43011494608925666,
    n_estimators=2000, random_state=42, verbosity=-1,
)
ROOT_CONFIG = {
    "kind": "solo", "model": "lgb", "params": dc(LGB_TUNED_PARAMS),
    "features": {"drop": []},
}
ROOT_MUTATION = ("root: s3e19 linear-winner's strongest solo config -- Optuna fold-5-"
                 "proxy-tuned LGB (40 trials, 42.0s, STATUS.md exp #7/Phase B Round 4), "
                 "solo OOF SMAPE 10.14833 (best single model, vs LGB_s42 10.17758), "
                 "weight 0.5 in the linear run's 6-way blend (10.01946) [PRIOR: this "
                 "run's own fold-5-proxy Optuna recipe, s3e19 is itself the sole "
                 "evidence source for the 'TimeSeriesSplit proxy = LAST fold, not "
                 "fold-0' lesson in experience.md's SMAPE/時序 section]")

LGB_ORIG_PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                       min_child_samples=20, subsample=0.9, colsample_bytree=0.8,
                       reg_lambda=1.0, random_state=42, verbosity=-1, objective="rmse")
CAT_ORIG_PARAMS = dict(iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                       random_seed=42, loss_function="RMSE", verbose=False)


# ---------------------------------------------------------------------------
# Eight first-generation SOLO lineage seeds.
# ---------------------------------------------------------------------------
def seed_lgb_s42():
    cfg = {"kind": "solo", "model": "lgb", "params": dc(LGB_ORIG_PARAMS), "features": {"drop": []}}
    desc = ("model-variant: exp #2's original (non-Optuna) hand-set LGB, seed=42 -- "
            "solo OOF 10.17758, the linear pool's LGB_s42 member (weight 0 in the final "
            "6-way blend, but kept as diversity/reproduction anchor) [PRIOR none -- "
            "STATUS.md base model, verbatim]")
    return cfg, desc


def seed_lgb_s2024():
    p = dc(LGB_ORIG_PARAMS)
    p["random_state"] = 2024
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("seed variation: hand-set LGB params, random_state 42->2024 -- STATUS.md "
            "exp #5/#7's LGB_s2024 member, solo OOF 10.21059, weight 0.1 in the linear "
            "6-way blend [PRIOR P29 -- seed bagging is effective under time-series "
            "extrapolation too, s3e19 exp #5/#6 are themselves the validating evidence]")
    return cfg, desc


def seed_lgb_s7():
    p = dc(LGB_ORIG_PARAMS)
    p["random_state"] = 7
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("seed variation: hand-set LGB params, random_state 42->7 -- STATUS.md "
            "exp #6/#7's LGB_s7 member, solo OOF 10.19446, LARGEST weight (0.4) of any "
            "non-tuned member in the linear 6-way blend [PRIOR P29 -- seed bagging, "
            "same evidence as LGB_s2024]")
    return cfg, desc


def seed_cat_s42():
    cfg = {"kind": "solo", "model": "cat", "params": dc(CAT_ORIG_PARAMS), "features": {"drop": []}}
    desc = ("model-type: hand-set CatBoost, seed=42 -- STATUS.md exp #2/#7's CAT_s42 "
            "member, solo OOF 10.73064, weakest-but-kept-for-diversity learner (weight "
            "0 in the final blend) [PRIOR none -- STATUS.md base model, verbatim]")
    return cfg, desc


def seed_cat_s2024():
    p = dc(CAT_ORIG_PARAMS)
    p["random_seed"] = 2024
    cfg = {"kind": "solo", "model": "cat", "params": p, "features": {"drop": []}}
    desc = ("seed variation: hand-set CatBoost, random_seed 42->2024 -- STATUS.md "
            "exp #6/#7's CAT_s2024 member, solo OOF 10.98383 (weakest single member), "
            "weight 0 in the linear blend [PRIOR P29 -- seed bagging, kept as pool "
            "diversity even though individually weak, per experience.md's 'weight "
            "search often zeroes weak members, that is a feature not a bug']")
    return cfg, desc


def seed_seedbag_tuned():
    p = dc(LGB_TUNED_PARAMS)
    p["random_state"] = 2024
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("UNTRIED per STATUS.md 'Potential improvements': seed-bag the Optuna-tuned "
            "LGB itself, random_state 42->2024 -- experience.md flags a real "
            "counter-example (s3e5: seed-bagging a config whose Optuna objective WAS "
            "the final metric can be neutral/negative, since the tuned config already "
            "converged to a shallow, low-variance solution) but s3e19's LGB_tuned was "
            "tuned on a FOLD-5 PROXY (not the full-OOF final metric directly), so its "
            "variance profile may differ -- honest test, not assumed to help [PRIOR P8 "
            "-- seed bagging is the cheapest reliable post-tuning residual gain, but "
            "flagged with the s3e5 counter-example caveat for directly-tuned configs]")
    return cfg, desc


def seed_deeplgb():
    p = dc(LGB_ORIG_PARAMS)
    p.update(num_leaves=127, min_child_samples=10, subsample=0.85,
             colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.05, learning_rate=0.05)
    cfg = {"kind": "solo", "model": "lgb", "params": p, "features": {"drop": []}}
    desc = ("model-variant: DELIBERATELY diverse deep/high-capacity/light-reg LGB "
            "(num_leaves 63->127, reg_lambda 1.0->0.05, reg_alpha 0->0, min_child "
            "20->10) -- pushed AWAY from LGB_tuned's shallow-and-regularized optimum "
            "(num_leaves 20) rather than toward it, since XGB is banned as a diversity "
            "source here (weight-searched to 0 twice, dropped per instruction). Direct "
            "application of s3e7's diversity-preservation lesson using LGB itself as "
            "the diversity axis [PRIOR none -- this run's own diversity-seeking probe, "
            "no experience.md precedent for LGB-vs-LGB capacity diversity specifically]")
    return cfg, desc


def seed_calsubset():
    p = dc(LGB_ORIG_PARAMS)
    cfg = {"kind": "solo", "model": "lgb", "params": p,
           "features": {"drop": ["weekofyear", "day"]}}
    desc = ("calendar-feature-SUBSET probe: drop weekofyear (redundant with "
            "day_of_year/month) and day (raw day-of-month, weak signal vs "
            "dow/cyclical) from the hand-set LGB_s42 baseline -- tests whether "
            "trimming redundant calendar columns helps time-series extrapolation "
            "(regularization-via-feature-reduction, per experience.md's collinearity/"
            "excess-feature-hurts pattern, applied here to the calendar-feature "
            "context specifically) [PRIOR P value -- 'features過多反而退步' pattern "
            "from s3e7/s3e14/s3e9, untested on s3e19's calendar-feature set]")
    return cfg, desc


SOLO_SEED_SPECS = [
    ("LGB_S42", seed_lgb_s42), ("LGB_S2024", seed_lgb_s2024), ("LGB_S7", seed_lgb_s7),
    ("CAT_S42", seed_cat_s42), ("CAT_S2024", seed_cat_s2024),
    ("SEEDBAG_TUNED", seed_seedbag_tuned), ("DEEPLGB", seed_deeplgb),
    ("CALSUBSET", seed_calsubset),
]


# ---------------------------------------------------------------------------
def _lineage_seed_ids(tree):
    out = {}
    all_names = [n for n, _ in SOLO_SEED_SPECS] + ["BLEND"]
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
    """All evaluated kind=='solo' node ids, best (lowest SMAPE) first."""
    nodes = [n for n in tree["nodes"]
             if n["status"] == "evaluated" and n["config"].get("kind") == "solo"]
    nodes.sort(key=lambda n: n["score"])
    return [n["id"] for n in nodes]


def seed_blend(tree):
    ids = _lineage_seed_ids(tree)
    members = [tree["root_id"], ids["LGB_S42"], ids["LGB_S2024"], ids["LGB_S7"],
               ids["CAT_S42"], ids["CAT_S2024"]]
    cfg = {"kind": "blend", "members": members, "weight_search": "dirichlet"}
    desc = (f"ensemble seed: 6-way blend of root LGB_tuned(#{tree['root_id']}) + "
            f"LGB_S42(#{ids['LGB_S42']}) + LGB_S2024(#{ids['LGB_S2024']}) + "
            f"LGB_S7(#{ids['LGB_S7']}) + CAT_S42(#{ids['CAT_S42']}) + "
            f"CAT_S2024(#{ids['CAT_S2024']}) CACHED OOF, dirichlet weight search -- "
            f"reproduces the linear-iteration's exp #7 6-member pool under "
            f"harness_v2's ensemble machinery, no retraining, scored on the IDENTICAL "
            f"IDX-masked OOF rows [PRIOR none -- generic ensemble-default node-space "
            f"seed, the recommendation #1 mechanism itself]")
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
                f"dirichlet weight search {prior_tag}")
        return child, desc
    return fn


def _blend_method_swap(tree, parent_cfg):
    child = dc(parent_cfg)
    cur = child.get("weight_search", "dirichlet")
    child["weight_search"] = "grid_simplex" if cur == "dirichlet" else "dirichlet"
    child.pop("result", None)
    n_members = len(child["members"])
    if child["weight_search"] == "grid_simplex" and n_members > 5:
        return None  # grid_simplex only supported up to 5 members (harness_v2 constraint)
    desc = (f"BLEND: weight-SEARCH-METHOD comparison {cur}->{child['weight_search']} on "
            f"the same {n_members} members -- grid_simplex is exhaustive (0.05-step "
            f"simplex, similar spirit to the linear run's own 0.1-step grid) vs "
            f"dirichlet's random+refine search [PRIOR none -- generic "
            f"ensemble-mechanism re-check]")
    return child, desc


def _blend_scale_toggle(tree, parent_cfg):
    child = dc(parent_cfg)
    cur_pp = child.get("postprocess") or {}
    cur_auto = bool(cur_pp.get("auto_scale", False))
    child["postprocess"] = {"auto_scale": not cur_auto}
    child.pop("result", None)
    desc = (f"BLEND: UNTRIED global-multiplier postprocess toggle auto_scale="
            f"{cur_auto}->{not cur_auto} -- grid-search a flat scalar multiplier on "
            f"the blended OOF (applied inside metric_fn per harness_v2.eval_blend's "
            f"contract, so the weight search itself is scale-aware) -- STRUCTURALLY "
            f"DIFFERENT from the already-REJECTED ratio-decomposition (STATUS.md "
            f"Round 1, RD weight 0): no country/date-conditional structure at all, "
            f"just a level-fit; task brief flags this as cheap+untried, linear "
            f"iteration never tried it [PRIOR none -- this run's own untried lever]")
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
            f"leaner {len(child['members'])}-way, test whether it was truly adding "
            f"value or just being tolerated by the weight search [PRIOR P14 -- "
            f"removing (near-)zero-weight members is zero-cost, per experience.md "
            f"Ensemble/後處理]")
    return child, desc


BLEND_QUEUE = [
    _mk_add_named("SEEDBAG_TUNED", "[PRIOR P8 -- add-to-pool-not-replace, untried "
                                    "seed-bag-the-tuned-config idea, weight search "
                                    "arbitrates whether it helps at zero extra cost]"),
    _mk_add_named("DEEPLGB", "[PRIOR none -- deliberate-diversity LGB as blend member, "
                              "s3e7's diversity lesson applied via LGB capacity instead "
                              "of a banned second model family]"),
    _blend_scale_toggle,
    _mk_add_named("CALSUBSET", "[PRIOR none -- calendar-feature-subset member, let "
                                "weight search decide whether the leaner feature set "
                                "adds value even if its solo score is worse]"),
    _blend_method_swap,
    _blend_remove_weakest,
]


def blend_fallback(tree, parent_cfg, attempt):
    """Post-queue blend mutations, each find_dup-checked HERE before being returned
    (see run_s3e1.py's identical rationale: never live-lock re-proposing the same
    duplicate)."""
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
    cur_pp = child.get("postprocess") or {}
    child["postprocess"] = {"auto_scale": not bool(cur_pp.get("auto_scale", False))}
    child.pop("result", None)
    if find_dup(tree, child) is None:
        desc = ("BLEND fallback: all member additions duplicate; auto_scale toggle "
                "instead [PRIOR none -- fallback]")
        return child, desc
    return None  # genuinely exhausted -> forced backtrack


def solo_fallback(parent_cfg, attempt):
    c = dc(parent_cfg)
    seed_key = "random_state" if c["model"] == "lgb" else "random_seed"
    c["params"] = dc(c["params"])
    c["params"][seed_key] = 3000 + attempt
    desc = (f"fallback seed-variation (lineage's authored mutation queue exhausted): "
            f"same config, alt {seed_key}={3000 + attempt} [PRIOR none -- fallback]")
    return c, desc


# ---------------------------------------------------------------------------
# Per-lineage authored mutation queues (SOLO): fn(parent_config) -> (child_config, desc)
# ---------------------------------------------------------------------------
def _bump(cfg, **params_update):
    c = dc(cfg)
    c["params"].update(params_update)
    return c


LGB_S42_QUEUE = [
    lambda c: (_bump(c, reg_lambda=2.0),
               "LGB_S42: reg_lambda 1.0->2.0 -- push MORE regularization on the "
               "un-tuned baseline (137k rows is medium-large scale; test whether "
               "extra L2 helps the extrapolation gap) [PRIOR P6 -- direct test of "
               "capacity-direction, extrapolation rewards regularization per this "
               "comp's own Optuna finding]"),
    lambda c: (_bump(c, num_leaves=31, min_child_samples=40),
               "LGB_S42: num_leaves 63->31, min_child_samples 20->40 -- shallower/more "
               "regularized variant, same direction as LGB_tuned's own optimum "
               "(num_leaves 20) but starting from hand-set params [PRIOR P6]"),
]

LGB_S2024_QUEUE = [
    lambda c: (_bump(c, reg_lambda=0.3),
               "LGB_S2024: reg_lambda 1.0->0.3 -- test the OPPOSITE capacity "
               "direction (less reg) on this seed variant, since LGB_S42's queue "
               "tests more-reg [PRIOR none -- deliberate counter-direction probe]"),
    lambda c: (_bump(c, learning_rate=0.05),
               "LGB_S2024: learning_rate 0.03->0.05 -- faster learning rate variant "
               "[PRIOR none -- generic hand-nudge]"),
]

LGB_S7_QUEUE = [
    lambda c: (_bump(c, min_child_samples=40),
               "LGB_S7: min_child_samples 20->40 -- regularize via leaf-size on the "
               "blend's largest-weighted non-tuned member [PRIOR P6]"),
    lambda c: (_bump(c, subsample=0.7),
               "LGB_S7: subsample 0.9->0.7 -- more row subsampling [PRIOR none]"),
]

CAT_S42_QUEUE = [
    lambda c: (_bump(c, l2_leaf_reg=6.0),
               "CAT_S42: l2_leaf_reg 3.0->6.0 -- regularization nudge [PRIOR none]"),
    lambda c: (_bump(c, depth=6),
               "CAT_S42: depth 8->6 -- shallower CatBoost variant [PRIOR none]"),
]

CAT_S2024_QUEUE = [
    lambda c: (_bump(c, depth=10),
               "CAT_S2024: depth 8->10 -- test whether CatBoost (currently the "
               "weakest member) benefits from MORE capacity given 137k rows is "
               "fairly large scale (s3e11's 'large data wants more capacity' finding "
               "vs the shallower-favors-extrapolation finding this comp's own Optuna "
               "run found for LGB) [PRIOR P6 -- direct test of which capacity-"
               "direction prior applies to CatBoost specifically at this scale]"),
    lambda c: (_bump(c, learning_rate=0.03),
               "CAT_S2024: learning_rate 0.05->0.03 -- slower learning rate variant "
               "[PRIOR none]"),
]

SEEDBAG_TUNED_QUEUE = [
    lambda c: (_bump(c, random_state=777),
               "SEEDBAG_TUNED: second seed variant on the tuned config (2024->777) "
               "-- further test of whether seed-bagging a fold-5-proxy-tuned (not "
               "directly-final-metric-tuned) config behaves like a normal seed-bag "
               "or like s3e5's neutral counter-example [PRIOR P8 -- continue testing "
               "the seed-bag-the-tuned-config idea]"),
    lambda c: (_bump(c, reg_lambda=0.6),
               "SEEDBAG_TUNED: reg_lambda 0.430->0.6, push L2 further while keeping "
               "seed=2024 [PRIOR none -- generic hand-nudge]"),
]

DEEPLGB_QUEUE = [
    lambda c: (_bump(c, num_leaves=200, colsample_bytree=0.7),
               "DEEPLGB: push capacity further -- num_leaves 127->200, "
               "colsample_bytree 0.9->0.7 [PRIOR none -- continue the diversity-"
               "seeking direction]"),
    lambda c: (_bump(c, learning_rate=0.08, subsample=0.7),
               "DEEPLGB: push diversity direction further -- lr 0.05->0.08, subsample "
               "0.85->0.7 [PRIOR none -- this run's own diversity-seeking direction, "
               "mirrors run_s3e1.py's XGBDIV_QUEUE pattern applied to LGB]"),
]

CALSUBSET_QUEUE = [
    lambda c: (dict(c, features={"drop": ["weekofyear", "day", "is_month_start",
                                            "is_month_end"]}),
               "CALSUBSET: more aggressive subset -- also drop is_month_start/"
               "is_month_end (rare/near-constant flags) on top of weekofyear/day "
               "[PRIOR none -- continue the feature-reduction probe]"),
    lambda c: (dict(c, features={"drop": ["month_sin", "month_cos"]}),
               "CALSUBSET: alternate subset -- drop ONLY the month cyclical pair "
               "(keep raw month; test whether trees can recover the cyclical "
               "boundary from month+day alone without explicit sin/cos) [PRIOR none "
               "-- experience.md notes cyclical encoding depends on whether target "
               "is truly calendar-driven; s3e19's target IS calendar-driven, so this "
               "tests whether the EXPLICIT encoding is still needed on top of the "
               "raw column trees can already split on]"),
]

SOLO_QUEUES = {
    "LGB_S42": LGB_S42_QUEUE, "LGB_S2024": LGB_S2024_QUEUE, "LGB_S7": LGB_S7_QUEUE,
    "CAT_S42": CAT_S42_QUEUE, "CAT_S2024": CAT_S2024_QUEUE,
    "SEEDBAG_TUNED": SEEDBAG_TUNED_QUEUE, "DEEPLGB": DEEPLGB_QUEUE,
    "CALSUBSET": CALSUBSET_QUEUE,
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


def smape_of(r):
    if r is None:
        return None
    if r.get("result") and "smape" in r["result"]:
        return r["result"]["smape"]
    return r.get("score")


# ---------------------------------------------------------------------------
# Prior-usage analysis (idea-injection experiment): "win" = child's score beat its
# direct parent's score (lower SMAPE = improvement, no sign flip).
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
        print(f"[root] #{nid} SMAPE={smape_of(r)} wall_s={r['wall_s']}")

    # --- seed the 8 first-generation SOLO lineages ---
    for name, seed_fn in SOLO_SEED_SPECS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} SMAPE={smape_of(r)} status={r['status']} wall_s={r['wall_s']}")

    already_blend = any(n["mutation"].startswith("[BLEND]")
                        for n in tree["nodes"] if n["parent_id"] == tree["root_id"])
    if not already_blend:
        cfg, desc = seed_blend(tree)
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
        print(f"[BLEND seed] #{nid} SMAPE={smape_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES fully (resume-safety)
    all_names = [n for n, _ in SOLO_SEED_SPECS] + ["BLEND"]
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
        gb_smape = gb["score"] if gb else None
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"SMAPE={smape_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best_SMAPE={gb_smape} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']}")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    gb_smape = gb["score"] if gb else None
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
        s = n["score"]
        running_best = s if running_best is None else min(running_best, s)
        if running_best <= LINEAR_BEST and evals_to_match is None:
            evals_to_match = i
    tree["evals_to_match_linear_best"] = evals_to_match
    hv2.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} SMAPE={gb_smape} mutation={gb['mutation']}")
    print(f"Linear-iteration best: {LINEAR_BEST} over {LINEAR_ROUNDS} experiments "
          f"(tree {'BEAT' if gb_smape < LINEAR_BEST else ('MATCHED' if gb_smape == LINEAR_BEST else 'did not beat')} it)")
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
