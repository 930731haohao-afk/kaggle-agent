"""Validate a dossier's typed injection ideas against the operator contract.

The contract in `knowledge/injection_operators.md` is only worth something if conformance is
machine-checkable: a dossier that emits prose, or an operator name the dispatcher has never
heard of, must fail loudly here rather than be silently dropped downstream (the failure the
contract exists to prevent). This module is also the end-to-end test of the contract's
EMISSION half — the execution half is covered by external_data/selftest.py.

Usage:
  VIRTUAL_ENV= uv run python3 external_data/validate_dossier.py competitions/*/dossier.json
Exit status is non-zero if any dossier fails, so it can gate a run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# operator -> (required params, optional params)
SPEC: dict[str, tuple[set[str], set[str]]] = {
    "join_feature":  ({"source", "join"}, {"as"}),
    "ratio_target":  ({"source", "join"}, {"space", "carry_forward"}),
    "log_offset":    ({"source", "join"}, {"space", "carry_forward"}),
    "trend_term":    (set(), {"unit", "degree", "centered"}),
    "flag_feature":  ({"source", "join"}, {"as", "window", "per_name"}),
    "sample_weight": ({"predicate"}, {"weight", "also_flag"}),
    "split_policy":  ({"scheme"}, {"time_col", "n_splits", "forbid", "group_col"}),
    "postprocess":   (set(), {"round_to_int", "clip_min", "clip_max", "global_scale"}),
}
SOURCES = {"worldbank:gdp_per_capita", "worldbank:gdp_per_capita_const",
           "worldbank:gdp_per_capita_ppp", "worldbank:gdp", "worldbank:gdp_const",
           "worldbank:gdp_growth_pct", "worldbank:population", "worldbank:cpi_inflation",
           "worldbank:unemployment_pct", "worldbank:urban_pop_pct",
           "worldbank:internet_users_pct", "holidays", "iso_country_table",
           "ecb_fx", "owid_covid"}


def validate(path: Path) -> dict:
    d = json.loads(path.read_text())
    comp = path.parent.name
    out = {"comp": comp, "schema": None, "n_ideas": 0, "valid": 0,
           "errors": [], "operators": [], "not_recorded": len(d.get("not_recorded") or [])}
    ideas = d.get("injection_ideas")
    if isinstance(ideas, dict):
        out["schema"] = "legacy_prose"
        out["errors"].append("injection_ideas is a dict of prose lists; the contract requires "
                             "a list of typed operators")
        out["n_ideas"] = sum(len(v) for v in ideas.values() if isinstance(v, list))
        return out
    if not isinstance(ideas, list):
        out["schema"] = "missing"
        out["errors"].append("injection_ideas absent or not a list")
        return out

    out["schema"] = "typed"
    out["n_ideas"] = len(ideas)
    for i, idea in enumerate(ideas):
        tag = f"idea[{i}]"
        if not isinstance(idea, dict):
            out["errors"].append(f"{tag}: not an object"); continue
        op = idea.get("operator")
        out["operators"].append(op)
        if op not in SPEC:
            out["errors"].append(f"{tag}: unknown operator {op!r} (vocabulary: {sorted(SPEC)})")
            continue
        req, opt = SPEC[op]
        params = idea.get("params")
        if not isinstance(params, dict):
            out["errors"].append(f"{tag} [{op}]: params missing or not an object"); continue
        missing = req - set(params)
        if missing:
            out["errors"].append(f"{tag} [{op}]: missing required params {sorted(missing)}")
        unknown = set(params) - req - opt
        if unknown:
            out["errors"].append(f"{tag} [{op}]: unknown params {sorted(unknown)}")
        src = params.get("source")
        if src is not None and src not in SOURCES:
            out["errors"].append(f"{tag} [{op}]: source {src!r} is not whitelisted")
        if op in ("ratio_target", "log_offset", "join_feature") and isinstance(params.get("join"), dict):
            if "keys" not in params["join"]:
                out["errors"].append(f"{tag} [{op}]: join lacks 'keys'")
        for prov in ("rationale", "source_prior"):
            if not idea.get(prov):
                out["errors"].append(f"{tag} [{op}]: missing {prov} (provenance is required so a "
                                     "prior's origin stays auditable)")
        if not [e for e in out["errors"] if e.startswith(tag)]:
            out["valid"] += 1
    return out


def main(paths: list[str]) -> int:
    rows = [validate(Path(p)) for p in paths]
    bad = 0
    print(f"{'comp':38s} {'schema':12s} {'ideas':>5s} {'valid':>5s} {'not_rec':>7s}  operators")
    for r in rows:
        ops = ",".join(str(o) for o in r["operators"][:5]) or "-"
        print(f"{r['comp'][:38]:38s} {str(r['schema']):12s} {r['n_ideas']:>5d} "
              f"{r['valid']:>5d} {r['not_recorded']:>7d}  {ops}")
        if r["errors"]:
            bad += 1
            for e in r["errors"][:6]:
                print(f"     ! {e}")
    tot, ok = sum(r["n_ideas"] for r in rows), sum(r["valid"] for r in rows)
    print(f"\n{ok}/{tot} ideas conform; {len(rows) - bad}/{len(rows)} dossiers clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
