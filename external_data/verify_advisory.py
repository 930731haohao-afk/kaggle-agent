"""Check whether an evaluator actually implements the advisory operators a dossier requested.

Why this exists. `apply.py` books `split_policy` and `postprocess` into the ledger's `advisory`
bucket UNCONDITIONALLY -- it runs on dataframes and has no handle on the evaluator that would
honor them, so it cannot check. For months the ledger therefore said "the evaluator owns this"
for requests no evaluator implemented, and nothing surfaced it. The concrete case that exposed
it (2026-07-30): s3e19's dossier asked for `round_to_int` / `clip_min` / `global_scale`, while
`eval_s3e19.maybe_postprocess` recognized only `auto_scale` / `scale` -- a pure vocabulary
mismatch, silently dropped. That evaluator was fixed; the general hole was not.

This module closes the general hole by making the claim checkable AFTER the fact, statically:
given a competition, find its evaluator and inspect whether the specific parameters the dossier
asked for are actually referenced in that evaluator's source. It is deliberately a static
source check rather than a runtime one, because it must work without loading competition data
or training anything.

A "verified" advisory entry means: the evaluator exists, has the mechanism, and the mechanism
reads the exact parameter names requested. Anything less is reported, not assumed.

Usage:
  VIRTUAL_ENV= uv run python3 external_data/verify_advisory.py                    # all dossiers
  VIRTUAL_ENV= uv run python3 external_data/verify_advisory.py <comp-slug> ...    # specific ones
Exit status is non-zero if any requested advisory parameter is unimplemented, so it can gate a run.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_TREE = _ROOT / "tree_search"

# competition slug -> its evaluator module stem. Kept explicit rather than guessed: a wrong
# guess here would report a competition as verified against someone else's evaluator.
EVALUATOR_FOR = {
    "playground-series-s3e19": "eval_s3e19",
    "playground-series-s5e1": "eval_s5e1",
    "playground-series-s5e10": "eval_s5e10",
    "cat-in-the-dat": "eval_citd",
    "tabular-playground-series-sep-2022": "eval_sep22",
    "playground-series-s3e1": "eval_s3e1",
    "playground-series-s3e3": "eval_s3e3",
    "playground-series-s3e5": "eval_s3e5",
    "playground-series-s3e7": "eval_s3e7",
    "playground-series-s3e9": "eval_s3e9",
    "playground-series-s3e11": "eval_s3e11",
    "playground-series-s3e14": "eval_s3e14",
    "playground-series-s3e16": "eval_s3e16",
    "playground-series-s3e20": "eval_s3e20",
    "playground-series-s4e11": "eval_s4e11",
    "playground-series-s6e1": "eval_s6e1",
    "playground-series-s6e2": "eval_s6e2",
    "afsis-soil-properties": "eval_afsis",
    "conway-s-reverse-game-of-life": "eval_conway",
    "tabular-playground-series-aug-2022": "eval_aug22",
    "tabular-playground-series-jan-2022": "eval_tpsjan22",
}

# For each advisory operator, the mechanism function that must exist in the evaluator, and how
# each requested parameter is expected to appear in that evaluator's source.
MECHANISM = {
    "postprocess": "maybe_postprocess",
    "split_policy": None,       # folds are built at import time, not via a named entry point
}


def _param_implemented(src: str, op: str, param: str) -> bool:
    """Is `param` actually read by the evaluator's source?

    Looks for the parameter name being fetched off a config/params dict -- `pp.get("clip_min")`,
    `postprocess["clip_min"]`, `cfg.get('clip_min')` etc. A bare mention inside a comment or
    docstring does not count as an implementation, so the match requires a subscript or .get().
    """
    pat = rf"""(\.get\(\s*["']{re.escape(param)}["']|\[\s*["']{re.escape(param)}["']\s*\])"""
    return re.search(pat, src) is not None


def verify_competition(comp: str, dossier_path: Path) -> dict:
    out = {"comp": comp, "requests": [], "evaluator": None, "errors": []}
    d = json.loads(dossier_path.read_text())
    ideas = d.get("injection_ideas")
    if not isinstance(ideas, list):
        return out                        # legacy prose dossier: nothing typed to verify

    advisory = [i for i in ideas if i.get("operator") in MECHANISM]
    if not advisory:
        return out

    mod = EVALUATOR_FOR.get(comp)
    out["evaluator"] = mod
    if mod is None:
        out["errors"].append(f"no evaluator mapping for {comp!r}; cannot verify its advisory "
                             f"requests -- add it to EVALUATOR_FOR")
        for i in advisory:
            out["requests"].append({"operator": i["operator"], "params": i.get("params", {}),
                                    "status": "unverifiable", "detail": "no evaluator mapping"})
        return out

    path = _TREE / f"{mod}.py"
    if not path.exists():
        out["errors"].append(f"evaluator {mod}.py does not exist")
        for i in advisory:
            out["requests"].append({"operator": i["operator"], "params": i.get("params", {}),
                                    "status": "NOT_IMPLEMENTED",
                                    "detail": f"{mod}.py does not exist"})
        return out

    src = path.read_text()
    for i in advisory:
        op, params = i["operator"], (i.get("params") or {})
        mech = MECHANISM[op]
        if mech and f"def {mech}" not in src:
            out["requests"].append({"operator": op, "params": params, "status": "NOT_IMPLEMENTED",
                                    "detail": f"{mod}.py has no {mech}() at all -- every "
                                              f"requested parameter is silently dropped"})
            continue
        if op == "split_policy":
            # folds are frozen at import time and asserted against the pipeline's own splitter,
            # so there is no parameter to match; report it as structurally-honored-but-unchecked
            out["requests"].append({"operator": op, "params": params, "status": "STRUCTURAL",
                                    "detail": "folds are built at evaluator import time and "
                                              "asserted against the pipeline's splitter; this "
                                              "checker cannot confirm the scheme matches, so "
                                              "confirm it in the evaluator's fold assertion"})
            continue
        missing = [k for k in params if not _param_implemented(src, op, k)]
        if missing:
            out["requests"].append({"operator": op, "params": params, "status": "PARTIAL",
                                    "detail": f"{mod}.{mech}() does not read: {sorted(missing)}"})
        else:
            out["requests"].append({"operator": op, "params": params, "status": "verified",
                                    "detail": f"{mod}.{mech}() reads every requested parameter"})

    # Runtime attestation beats the static grep: if the evaluator has actually executed the
    # mechanism (tree_search/eval_support.write_attestation, written from evaluate_solo), the
    # attestation records which parameters were honoured in a real evaluation. A static
    # "verified" is a claim about source text; "verified (runtime)" is a consumer trace.
    att_path = dossier_path.parent / "evaluator_attestation.json"
    if att_path.exists():
        try:
            att = json.loads(att_path.read_text())
        except Exception:  # noqa: BLE001
            att = {}
        for q in out["requests"]:
            entry = att.get(q["operator"])
            if not isinstance(entry, dict):
                continue
            honored = set(entry.get("honored_params") or entry.keys())
            if q["status"] in ("verified", "PARTIAL") and set(q["params"]) <= honored:
                q["status"] = "verified (runtime)"
                q["detail"] = ("evaluator_attestation.json records these exact parameters "
                               "honoured in a real evaluation")
    return out


def main(argv: list[str]) -> int:
    if argv:
        paths = [(c, _ROOT / "competitions" / c / "dossier.json") for c in argv]
    else:
        paths = [(Path(p).parent.name, Path(p))
                 for p in sorted(glob.glob(str(_ROOT / "competitions" / "*" / "dossier.json")))]
        # Generated arm workspaces ("<comp>-v5-<arm>" from make_v5_arm.py, "<comp>.v5arms-*"
        # from arm drivers) carry verbatim copies of their parent competition's dossier;
        # auditing the copy double-counts the parent and always reports "no evaluator
        # mapping". The parent's own dossier stays in the audit.
        paths = [(c, p) for c, p in paths if "-v5-" not in c and ".v5arms-" not in c]

    rows = [verify_competition(c, p) for c, p in paths if p.exists()]
    rows = [r for r in rows if r["requests"] or r["errors"]]

    bad = 0
    print(f"{'competition':40s} {'operator':13s} {'status':16s} detail")
    print("-" * 118)
    for r in rows:
        for q in r["requests"]:
            if q["status"] in ("NOT_IMPLEMENTED", "PARTIAL", "unverifiable"):
                bad += 1
            print(f"{r['comp'][:40]:40s} {q['operator']:13s} {q['status']:16s} {q['detail'][:52]}")
        for e in r["errors"]:
            print(f"{r['comp'][:40]:40s} {'-':13s} {'ERROR':16s} {e[:52]}")

    total = sum(len(r["requests"]) for r in rows)
    ok = total - bad
    print(f"\n{ok}/{total} advisory requests verified as actually implemented.")
    if bad:
        print(f"{bad} request(s) are NOT implemented, only partially implemented, or "
              f"unverifiable.\nThe ledger books all of these as 'advisory: the evaluator owns "
              f"this' regardless -- that\nclaim is what this checker exists to keep honest.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
