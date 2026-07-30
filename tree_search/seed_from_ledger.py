"""Consume the injection ledger's config-only operators as real search nodes.

Why this file exists. `external_data/apply.py` turns `objective` / `blend_member` /
`encoding` operators into node configurations, appends them to `plan["node_configs"]`, and
books them "realized". Until 2026-07-30 nothing read that list: a repo-wide grep found the
write and the docstring describing the contract, and no reader. The judgment layer's
model-level decisions therefore changed nothing about a run while the ledger reported
success — the same produced-with-no-consumer failure the operator contract was introduced to
stop, one layer higher.

This module closes the loop. It translates each emitted config into concrete solo node
configs the evaluators actually accept (`{kind, model, params, features}` — see
`tree_search/eval_s3e19.py:7`), and reports back what it could NOT translate, with a reason,
so an unconsumable emission is recorded rather than silently dropped.

What translates, and what does not:
  objective      -> one solo node per model family present in `params_by_model`, base params
                    merged with the mapped objective and `forbid_params` removed.
  blend_member   -> one shallow, heavily-regularized solo node; the driver's existing reblend
                    step then picks it up out of `solo_pool`, which is what "add to the pool,
                    never replace a member" means operationally.
  encoding       -> NOT a node config. native is already the evaluators' default (categorical
                    dtype is passed through); count / ordinal / crosses belong in the data
                    layer (`external_data/apply.py`); target needs per-fold support inside the
                    evaluator and onehot_sparse needs a linear model family. All are returned
                    as unconsumable with the specific reason.

Usage (library):
    from seed_from_ledger import load_emitted, materialize
    specs = load_emitted(workspace_dir)
    nodes, skipped = materialize(specs, base_cfg, whitelist)
"""
from __future__ import annotations

import copy
import json
import os

# knobs the blend_member archetype uses, per model family, in the evaluators' own vocabulary
_ARCHETYPE_KNOBS = {
    "shallow_regularized": {
        "lgb": {"num_leaves": 7, "max_depth": 3, "reg_alpha": 3.0, "reg_lambda": 10.0,
                "min_child_samples": 120, "learning_rate": 0.05},
        "xgb": {"max_depth": 3, "reg_alpha": 3.0, "reg_lambda": 10.0,
                "min_child_weight": 40, "learning_rate": 0.05},
        "cat": {"depth": 3, "l2_leaf_reg": 12.0, "learning_rate": 0.05},
    },
}
_DEPTH_KEY = {"lgb": "max_depth", "xgb": "max_depth", "cat": "depth"}


def load_emitted(workspace: str) -> list[dict]:
    """Read the arm's emitted node configs from its manifest (or bare ledger)."""
    for name, path in (("manifest", os.path.join(workspace, "data", "arm_manifest.json")),
                       ("ledger", os.path.join(workspace, "data", "injection_ledger.json"))):
        if not os.path.exists(path):
            continue
        obj = json.load(open(path))
        plan = obj.get("ledger", obj) if name == "manifest" else obj
        cfgs = plan.get("node_configs") or []
        if cfgs:
            return cfgs
    return []


def _seed_objective(spec: dict, base: dict, whitelist: dict) -> tuple[list[dict], list[dict]]:
    by_model = spec.get("params_by_model") or {}
    forbid = set(spec.get("forbid_params") or [])
    out, skipped = [], []
    families = [m for m in by_model if m in (whitelist or {})] or list(by_model)
    for fam in families:
        p = dict(base.get("params") or {}) if base.get("model") == fam else {}
        p.update(by_model[fam])
        for k in forbid:
            p.pop(k, None)
        out.append({"kind": "solo", "model": fam, "params": p,
                    "features": copy.deepcopy(base.get("features") or {"drop": []}),
                    "seed_role": "objective", "metric_family": spec.get("metric_family")})
    if not out:
        skipped.append({"operator": "objective", "reason":
                        f"no model family in params_by_model ({sorted(by_model)}) is available"})
    return out, skipped


def _seed_blend_member(spec: dict, base: dict, whitelist: dict) -> tuple[list[dict], list[dict]]:
    arch = spec.get("archetype", "shallow_regularized")
    knobs = _ARCHETYPE_KNOBS.get(arch)
    if knobs is None:
        return [], [{"operator": "blend_member",
                     "reason": f"archetype {arch!r} has no knob mapping "
                               f"(known: {sorted(_ARCHETYPE_KNOBS)})"}]
    fam = base.get("model") or "lgb"
    if fam not in knobs:
        fam = next(iter(knobs))
    p = dict(base.get("params") or {}) if base.get("model") == fam else {}
    p.update(knobs[fam])
    depth = spec.get("depth")
    if depth is not None and _DEPTH_KEY.get(fam):
        p[_DEPTH_KEY[fam]] = int(depth)
        if fam == "lgb":
            p["num_leaves"] = max(2, 2 ** int(depth) - 1)
    return ([{"kind": "solo", "model": fam, "params": p,
              "features": copy.deepcopy(base.get("features") or {"drop": []}),
              "seed_role": "blend_member", "archetype": arch}], [])


# encoding schemes and why each one is or is not a node config
_ENCODING_REASON = {
    "native": "already the evaluators' default: categorical dtype is passed to the model, "
              "so no node is needed (advisory, not a gap)",
    "count": "data-layer transform: belongs in external_data/apply.py as an added column, "
             "not in a node config",
    "ordinal": "data-layer transform: belongs in external_data/apply.py as an added column",
    "crosses": "data-layer transform: belongs in external_data/apply.py as added columns",
    "target": "requires per-fold target encoding computed on the model's own folds "
              "(requires_evaluator_support: per_fold_target_encoding); no evaluator "
              "implements it, and approximating it out-of-fold is the s4e1 leakage path",
    "onehot_sparse": "requires a sparse linear model family, which the evaluators do not have",
}


def materialize(specs: list[dict], base_cfg: dict,
                whitelist: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Translate emitted configs into node configs. Returns (nodes, unconsumable)."""
    nodes, skipped = [], []
    for spec in specs or []:
        role = spec.get("seed_role")
        if role == "objective":
            n, s = _seed_objective(spec, base_cfg, whitelist or {})
        elif role == "blend_member":
            n, s = _seed_blend_member(spec, base_cfg, whitelist or {})
        elif role == "encoding":
            scheme = spec.get("scheme", "native")
            n, s = [], [{"operator": "encoding", "scheme": scheme,
                         "reason": _ENCODING_REASON.get(scheme, "unknown scheme")}]
        else:
            n, s = [], [{"operator": spec.get("seed_role") or "unknown",
                         "reason": "no translation rule for this seed_role"}]
        nodes.extend(n)
        skipped.extend(s)
    return nodes, skipped


def write_consumption_report(workspace: str, seeded: list[dict], skipped: list[dict]) -> str:
    """Record what the driver actually seeded, so the ledger's claim is auditable."""
    path = os.path.join(workspace, "injection_consumed.json")
    json.dump({"seeded": seeded, "unconsumable": skipped,
               "seeded_count": len(seeded), "unconsumable_count": len(skipped)},
              open(path, "w"), indent=2)
    return path


if __name__ == "__main__":  # smoke: translate an arm's emissions without running anything
    import sys
    ws = sys.argv[1]
    specs = load_emitted(ws)
    base = {"kind": "solo", "model": "lgb", "params": {"num_leaves": 31, "n_estimators": 400},
            "features": {"drop": []}}
    nodes, skipped = materialize(specs, base, {"lgb": [], "cat": []})
    print(json.dumps({"emitted": len(specs), "nodes": nodes, "unconsumable": skipped}, indent=2))
