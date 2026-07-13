"""tree_search/prior_wiring.py — mechanical prior→candidate wiring (arm A of the
prior-attribution experiment).

WHY THIS EXISTS (the write-only hole)
-------------------------------------
`suggest_priors` / `suggest_priors_v4` compute per-competition priors and drivers store
them in `tree["priors"]` — but nothing downstream ever READS that field (verified by
grep across harness v2/v3/v4 and every driver on 2026-07-13: zero consumers). The
candidates the search actually evaluates all come from hand-written `seed_*` functions,
whose descriptions merely *cite* priors ("[PRIOR P18]") that a human matched by eye at
driver-authoring time. Stage-5 attribution therefore measured injection as a no-op:
the priors never reached the queue.

This module closes that gap MECHANICALLY and DETERMINISTICALLY: it classifies each
prior dict (the `suggest_priors_v4` schema: text/provenance/source) against an ordered
rule table and, where a rule is actionable, translates the prior into one or more
concrete candidate configs (the evaluator schema: kind/model/params/features/rounder),
each tagged `[PRIOR-<PROV>-<n>]` so tree nodes are traceable back to the exact prior
that generated them.

HONESTY CONTRACT
----------------
Not every prior is executable. The output ledger classifies every prior into exactly
one of:
  candidate  — translated into >=1 enqueued config (the wire is live)
  satisfied  — the advice is already structurally enforced (e.g. s3e5's evaluator bakes
               OptimizedRounder into every node's score; the root IS the direct-QWK
               Optuna config). Emitting a duplicate would be noise, not signal.
  policy     — search-process advice (stop rules, "quit while ahead"): consumed by the
               search policy layer, not representable as a node.
  n/a        — precondition fails for this competition (e.g. Platt scaling needs a
               probability metric; native categorical handling needs categorical
               columns). Recorded with the failing precondition.
  unmapped   — no rule matched; an honest gap in the template library.
The ledger is the deliverable as much as the candidates: it is the proof that every
prior was *delivered* and *dispositioned*, which is exactly what the stage-5 no-op
finding said was missing.

DETERMINISM
-----------
Rule order is fixed, template outputs are pure functions of (prior, champion_config,
ctx), and candidates are deduped against the tree via harness_v2.config_hash — so the
same tree + same priors + same ctx always yields the same candidates in the same
order. No RNG, no LLM (that's arm B), fully compatible with the bit-reproduction gate.
"""

from __future__ import annotations

import os
import re
import sys
from copy import deepcopy

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import harness_v2 as hv2  # noqa: E402  (config_hash)

# ---------------------------------------------------------------------------
# Template builders. Each takes (prior, champion_cfg, ctx) and returns a list of
# (config, short_desc) pairs. They are DEFENSIVE: if the champion config does not
# carry the knobs the template needs, they return [] (the rule then falls through
# to "n/a" with reason "template preconditions not met by champion config").
#
# ctx keys (all optional, driver-provided):
#   has_categorical: bool  — competition has raw categorical columns
#   prob_metric:     bool  — metric consumes probabilities (logloss/Brier/calibrated AUC)
#   blend_members:   list  — node ids whose OOF is cached and eligible for blending
#   alt_tuned:       dict  — {"cat": {...params...}, "xgb": {...}} alternative tuned params
# ---------------------------------------------------------------------------

# per-model candidate seed-key names (sklearn-style AND native-API style — different
# comps' evaluators use different naming schemes, e.g. s3e5 random_state vs s6e1 seed)
_SEED_KEYS = {"lgb": ("random_state", "seed"), "xgb": ("random_state", "seed"),
              "cat": ("random_seed",)}


def _t_rounder_fold_avg(prior, champ, ctx):
    """Nested/cutpoint-overfit priors -> the fold-averaged-cutpoints rounder variant."""
    if champ.get("kind") != "solo" or champ.get("rounder") != "full_oof":
        return []
    cfg = deepcopy(champ)
    cfg.pop("result", None)
    cfg["rounder"] = "fold_avg"
    return [(cfg, "champion config, rounder full_oof->fold_avg (per-fold cutpoints, averaged)")]


def _t_regularize_push(prior, champ, ctx):
    """'Small sample -> prefer regularization over capacity' -> shrink capacity knobs."""
    if champ.get("kind") != "solo":
        return []
    params = champ.get("params", {})
    model = champ.get("model")
    patch = {}
    if model == "lgb":
        if "min_child_samples" in params:
            patch["min_child_samples"] = params["min_child_samples"] * 2
        if "num_leaves" in params:
            patch["num_leaves"] = max(7, params["num_leaves"] // 2)
    elif model == "cat":
        if "min_data_in_leaf" in params:
            patch["min_data_in_leaf"] = params["min_data_in_leaf"] * 2
    elif model == "xgb":
        if "min_child_weight" in params:
            patch["min_child_weight"] = params["min_child_weight"] * 2
        if "max_depth" in params and params["max_depth"] > 2:
            patch["max_depth"] = params["max_depth"] - 1
    if not patch:
        return []
    cfg = deepcopy(champ)
    cfg.pop("result", None)
    cfg["params"] = dict(params, **patch)
    knobs = ", ".join(f"{k} {params[k]}->{v}" for k, v in patch.items())
    return [(cfg, f"regularization push on champion ({knobs})")]


def _t_seed_bag(prior, champ, ctx):
    """Seed-bagging priors -> same config, two alternative seeds (members for a later blend)."""
    if champ.get("kind") != "solo":
        return []
    key = next((k for k in _SEED_KEYS.get(champ.get("model"), ())
                if k in champ.get("params", {})), None)
    if key is None:
        return []
    out = []
    for seed in (101, 202):  # fixed, deterministic
        cfg = deepcopy(champ)
        cfg.pop("result", None)
        cfg["params"] = dict(champ["params"], **{key: seed})
        out.append((cfg, f"seed-bag member: champion config, {key}={seed}"))
    return out


def _t_blend_pool(prior, champ, ctx):
    """Blend/weight-search priors -> pool blend over the driver-supplied member ids
    (the weight search arbitrates membership; zeroing a member is a valid outcome —
    that is literally what the s3e3 'CAT zeroed' prior predicts)."""
    members = ctx.get("blend_members") or []
    if len(members) < 3:
        return []
    cfg = {"kind": "blend", "members": sorted(members), "weight_search": "dirichlet"}
    if "rounder" in champ:
        cfg["rounder"] = champ.get("rounder", "full_oof")
    return [(cfg, f"pool blend over {len(members)} cached members, dirichlet weight search")]


# ---------------------------------------------------------------------------
# Rule table. Ordered; FIRST match wins (deterministic disposition per prior).
# status: candidate -> builder runs; satisfied/policy -> recorded, no configs;
#         na -> precondition callable decides (True = genuinely n/a).
# Patterns are matched lowercase against the prior's full text (zh + en keys).
# ---------------------------------------------------------------------------

RULES = [
    # --- satisfied: structurally enforced by this evaluator/root already ---
    dict(key="rounder_enforced", status="satisfied",
         patterns=["optimizedrounder(對 oof qwk 調切點)", "naive round", "遠勝 naive"],
         reason="evaluator bakes OptimizedRounder into every node score (eval docstring)"),
    dict(key="regression_head", status="satisfied",
         patterns=["迴歸頭", "多分類頭"],
         reason="champion/root already uses the regression head"),
    dict(key="optuna_direct_metric", status="satisfied",
         patterns=["optuna 目標函式直接設", "直接設為「後處理", "direct-post-rounder"],
         reason="root IS the Optuna direct-post-rounder-QWK-tuned config"),

    # --- candidates: mechanically translatable ---
    dict(key="rounder_fold_avg", status="candidate", builder=_t_rounder_fold_avg,
         patterns=["巢狀", "切點對全 oof 過擬", "切點若直接對整份 oof", "nested", "cutpoint"]),
    dict(key="regularize_push", status="candidate", builder=_t_regularize_push,
         patterns=["優先加正則", "正則化而非加容量", "regulariz", "加正則"]),
    dict(key="seed_bag", status="candidate", builder=_t_seed_bag,
         patterns=["seed bag", "換不同的隨機種子", "種子平均", "seed averaging"]),
    dict(key="blend_pool", status="candidate", builder=_t_blend_pool,
         patterns=["權重搜尋歸零", "blend 永遠有益", "weight search", "rank averag", "先看各模型 oof"]),

    # --- policy: consumed by the search loop, not a node ---
    dict(key="stop_discipline", status="policy",
         patterns=["見好就收", "plateau", "停止準則", "噪音上過度搜尋"],
         reason="search-policy advice: patience/stop rule, not a candidate config"),

    # --- n/a: competition-level preconditions ---
    dict(key="needs_prob_metric", status="na",
         patterns=["platt", "sigmoid 校準", "isotonic", "機率校準"],
         precond=lambda ctx: not ctx.get("prob_metric", False),
         reason="metric does not consume probabilities"),
    dict(key="needs_categorical", status="na",
         patterns=["cat_features", "原生類別", "native categorical", "類別欄"],
         precond=lambda ctx: not ctx.get("has_categorical", False),
         reason="competition has no raw categorical columns"),
]

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.lower())


def _match_rule(text_lc: str):
    for rule in RULES:
        if any(p in text_lc for p in rule["patterns"]):
            return rule
    return None


def _prior_tag(prior, idx: int) -> str:
    src = prior.get("source", "")
    if prior.get("provenance") == "EXT" and src:
        return f"PRIOR-EXT-{src.split('-')[-1]}"
    return f"PRIOR-INT-{idx:02d}"


def wire_priors(priors: list, champion_cfg: dict, tree: dict, ctx: dict = None,
                max_candidates: int = 8):
    """Classify every prior and translate the actionable ones into candidate configs.

    Returns {"candidates": [...], "ledger": [...]}:
      candidates: [{config, mutation, tag, rule, prior_text}]  (deduped vs tree + batch,
                  capped at max_candidates, order = prior order then template order)
      ledger:     one row per prior: {tag, provenance, source, rule, status, reason,
                  n_candidates, text} — every prior dispositioned exactly once.
    """
    ctx = ctx or {}
    # stored node configs carry a "result" payload the fresh candidates don't; strip it
    # (and any other non-recipe key) before hashing or tree-level dedupe never fires
    def _recipe_hash(cfg):
        return hv2.config_hash({k: v for k, v in cfg.items() if k != "result"})
    existing = {_recipe_hash(n["config"]) for n in tree["nodes"]}
    candidates, ledger = [], []

    for idx, prior in enumerate(priors):
        tag = _prior_tag(prior, idx)
        text_lc = _norm(prior.get("text", ""))
        rule = _match_rule(text_lc)

        if rule is None:
            ledger.append(dict(tag=tag, provenance=prior.get("provenance"),
                               source=prior.get("source"), rule=None, status="unmapped",
                               reason="no template rule matched (honest gap)",
                               n_candidates=0, text=prior.get("text", "")[:160]))
            continue

        status, reason = rule["status"], rule.get("reason", "")
        n_emitted = 0

        if status == "na" and not rule["precond"](ctx):
            # precondition actually HOLDS -> the prior is applicable but we have no
            # template for the applicable case; record honestly as unmapped.
            status, reason = "unmapped", f"applicable (precond holds) but no template ({rule['key']})"

        if status == "candidate":
            built = rule["builder"](prior, champion_cfg, ctx)
            if not built:
                status, reason = "na", "template preconditions not met by champion config"
            for cfg, short in built:
                h = _recipe_hash(cfg)
                if h in existing:
                    reason = (reason + "; " if reason else "") + "deduped vs existing node"
                    continue
                if len(candidates) >= max_candidates:
                    reason = (reason + "; " if reason else "") + "max_candidates cap hit"
                    break
                existing.add(h)
                snippet = prior.get("text", "").split("|")[0].strip()[:90]
                candidates.append(dict(
                    config=cfg, tag=tag, rule=rule["key"], prior_text=prior.get("text", ""),
                    mutation=f"[{tag}] {short} <= wired from prior: {snippet}"))
                n_emitted += 1
            if n_emitted == 0 and status == "candidate":
                status = "candidate-deduped"

        ledger.append(dict(tag=tag, provenance=prior.get("provenance"),
                           source=prior.get("source"), rule=rule["key"], status=status,
                           reason=reason, n_candidates=n_emitted,
                           text=prior.get("text", "")[:160]))

    return dict(candidates=candidates, ledger=ledger)


def ledger_summary(ledger: list) -> str:
    """One-line-per-status human summary for logs/reports."""
    from collections import Counter
    c = Counter(row["status"] for row in ledger)
    parts = [f"{k}={v}" for k, v in sorted(c.items())]
    total_cand = sum(row["n_candidates"] for row in ledger)
    return f"{len(ledger)} priors dispositioned ({', '.join(parts)}); {total_cand} candidate configs emitted"
