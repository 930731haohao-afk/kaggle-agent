#!/usr/bin/env python3
"""bug_demo_prior_injection.py — runnable proof of the write-only-priors bug.

THE BUG (found 2026-07-08, fixed 2026-07-13 — this script proves the bug itself):
    Stage-5 "external idea injection" computed priors and stored them in
    tree["priors"], but NO search operator ever read that field. All candidates
    actually searched came from hand-written seed functions in each driver. So
    every earlier "injection has no effect" measurement measured non-delivery,
    not uselessness.

WHAT THIS SCRIPT DOES (one command, ~2 seconds, no training):

  Proof 1 — static (code scan):
      Scans every harness/driver .py in tree_search/ for occurrences of
      tree["priors"] / tree['priors'] and classifies each as WRITE / PRINT
      versus READ-BY-SEARCH. Expected: zero read-by-search sites.

  Proof 2 — dynamic (bit-identity on committed artifacts):
      Loads the two committed single-variable attribution runs on s3e3
      (competitions/playground-series-s3e3/experiments_tree_v4_{off,ext}.json):
      identical seeds/folds/budget, the ONLY difference is injection off/on.
      Shows the write side DID differ (8 vs 10 priors stored) while every
      searched node and every cached OOF vector is bit-identical. If the two
      extra injected priors had been read anywhere downstream, something would
      have had to differ; nothing does — a constructive proof of non-delivery.

  Verdict — prints PASS/FAIL for the bug claim.

Regenerate the artifacts from scratch (optional, ~minutes):
    V4_MODE=off uv run python3 tree_search/run_s3e3_v4.py
    V4_MODE=ext uv run python3 tree_search/run_s3e3_v4.py

Fix & follow-up experiment live in tree_search/prior_wiring.py and
docs/prior_wiring_findings.md — deliberately out of scope here.
"""

import glob
import io
import json
import os
import re
import sys
import tokenize

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
COMP_DIR = os.path.join(REPO, "competitions", "playground-series-s3e3")

PRIORS_RE = re.compile(r"tree\s*\[\s*['\"]priors['\"]\s*\]")
# a hit is a WRITE/PRINT if the same line assigns into the field or just formats it
WRITE_RE = re.compile(r"tree\s*\[\s*['\"]priors['\"]\s*\]\s*=")


def _code_subscript_lines(src):
    """Line numbers where 'priors' appears as a REAL subscript string literal in CODE
    (a bare STRING token '\\'priors\\''), as opposed to merely being mentioned inside a
    comment or docstring (COMMENT token / long STRING containing prose)."""
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING and tok.string.strip("'\"") == "priors" \
                and len(tok.string) <= len("'priors'") + 2:
            lines.add(tok.start[0])
    return lines


def proof1_static():
    print("=" * 72)
    print("Proof 1 — static: who touches tree['priors'] across ALL search code?")
    print("=" * 72)
    hits, readers = [], []
    for path in sorted(glob.glob(os.path.join(_HERE, "*.py"))):
        base = os.path.basename(path)
        if base == os.path.basename(__file__) or base.startswith(("prior_wiring", "run_wire",
                                                                  "run_s3e5_wire", "run_s6e1_wire",
                                                                  "wire_bootstrap")):
            continue  # exclude the FIX (2026-07-13) — we are proving the pre-fix state
        src = open(path, encoding="utf-8").read()
        code_lines = _code_subscript_lines(src)
        for lineno, line in enumerate(src.splitlines(), 1):
            if not PRIORS_RE.search(line):
                continue
            if lineno not in code_lines:
                kind = "COMMENT"          # mention inside comment/docstring, not code
            elif WRITE_RE.search(line):
                kind = "WRITE"
            elif "print" in line or "len(" in line:
                kind = "PRINT/LOG"
            else:
                kind = "READ?"
            hits.append((base, lineno, kind, line.strip()[:80]))
            if kind == "READ?":
                readers.append((base, lineno))
    for base, lineno, kind, text in hits:
        print(f"  {kind:9s} {base}:{lineno:<5d} {text}")
    print(f"\n  total touches: {len(hits)}   classified as consumed-by-search: {len(readers)}")
    ok = len(readers) == 0
    print(f"  => {'PASS' if ok else 'FAIL'}: no search operator reads tree['priors'] "
          f"(pre-fix code)")
    return ok


def proof2_dynamic():
    print("\n" + "=" * 72)
    print("Proof 2 — dynamic: single-variable off vs ext runs are bit-identical")
    print("=" * 72)
    t_off = json.load(open(os.path.join(COMP_DIR, "experiments_tree_v4_off.json")))
    t_ext = json.load(open(os.path.join(COMP_DIR, "experiments_tree_v4_ext.json")))

    p_off, p_ext = t_off.get("priors", []), t_ext.get("priors", [])
    print(f"  write side  : priors stored off={len(p_off)}  ext={len(p_ext)}  "
          f"(injection DID add {len(p_ext) - len(p_off)} [EXT] entries)")

    n_off, n_ext = t_off["nodes"], t_ext["nodes"]
    same_n = len(n_off) == len(n_ext)
    max_dscore, diffs = 0.0, 0
    for a, b in zip(n_off, n_ext):
        if a["mutation"] != b["mutation"] or a["status"] != b["status"]:
            diffs += 1
        if isinstance(a["score"], (int, float)) and isinstance(b["score"], (int, float)):
            max_dscore = max(max_dscore, abs(a["score"] - b["score"]))
    print(f"  search side : nodes off={len(n_off)} ext={len(n_ext)} same_count={same_n}; "
          f"structural diffs={diffs}; max|Δscore|={max_dscore}")

    oof_max = 0.0
    n_oof = 0
    off_dir = os.path.join(_HERE, "cache_s3e3_v4_off")
    ext_dir = os.path.join(_HERE, "cache_s3e3_v4_ext")
    for f in sorted(os.listdir(off_dir)):
        if not f.endswith(".npz"):
            continue
        a = np.load(os.path.join(off_dir, f))["oof"]
        b = np.load(os.path.join(ext_dir, f))["oof"]
        oof_max = max(oof_max, float(np.max(np.abs(a - b))))
        n_oof += 1
    print(f"  OOF caches  : {n_oof} pairs compared, max|Δ|={oof_max}")

    ok = same_n and diffs == 0 and max_dscore == 0.0 and oof_max == 0.0
    print(f"  => {'PASS' if ok else 'FAIL'}: the 2 extra injected priors changed NOTHING "
          f"downstream — they were never read")
    return ok


if __name__ == "__main__":
    ok1 = proof1_static()
    ok2 = proof2_dynamic()
    print("\n" + "=" * 72)
    if ok1 and ok2:
        print("VERDICT: BUG PROVEN — stage-5 injection was write-only (never delivered "
              "to the search). Earlier 'injection = null' results measured non-delivery.")
    else:
        print("VERDICT: proofs incomplete — inspect the FAIL lines above.")
    print("=" * 72)
    sys.exit(0 if (ok1 and ok2) else 1)
