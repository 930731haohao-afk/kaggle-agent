#!/usr/bin/env python3
"""Fail if anything a benchmark lane can read states a benchmark competition's own result.

WHY THIS EXISTS. Rounds 4-8 of the pre-re-run audit each closed a leak by naming the file
that held it -- experience.md, then the instruction files, then the pinned eval modules, then
root STATUS.md, then config.yaml, then tree_search/llm_proposer_input_*.json, then
competitions/_batch_results.json, then the recorded OOF matrices left inside data/. Every
round the fix was "add another glob", and every round the next round found a file the glob
list did not know about. Enumeration by hand lost five times in a row.

So this is the check, not another glob: walk everything a run can read and flag any line that
puts a benchmark competition's name next to a number precise enough to be a score, or next to
leaderboard language. The archive script's job is to make this pass; this script's job is to
say whether it did. `--simulate-archive` answers that BEFORE the destructive move.

  python3 benchmark_infra/verify_clean_slate.py --simulate-archive   # would the plan suffice?
  python3 benchmark_infra/verify_clean_slate.py                      # after --execute
  python3 benchmark_infra/verify_clean_slate.py --root /some/tree    # arbitrary tree

Exit 0 clean, 3 leaks found, 4 usage error.

WHAT IS EXEMPT, AND WHY (the list is short on purpose -- an exemption is a promise that some
OTHER mechanism keeps a run out of that file, and each one names its mechanism):
  knowledge/knowledge_base.json  structured base; served only through task_priors_for.py,
                                 whose 8 selftest contracts are the isolation guarantee.
  knowledge/experience*.md       served only through query_library.py / suggest_priors.py,
                                 which self-exclude; raw reads forbidden by the launcher
                                 prompt and by SKILL.md.
  knowledge/idea_bank.md         same doors; consumer channel dormant (round 6).
  knowledge/archive_pre_structured/, archive/   quarantined, not on any read path.
  tests/, benchmark_infra/       the audit's own machinery: these files describe leaks in
                                 order to prevent them, and a lane has no reason to read the
                                 harness that judges it. Kept narrow deliberately.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)

# Read-path exemptions. Each entry is a prefix relative to the root.
EXEMPT_PREFIXES = (
    ".git/", "archive/", "node_modules/", ".venv/", "__pycache__/",
    "knowledge/knowledge_base.json", "knowledge/experience", "knowledge/idea_bank.md",
    "knowledge/vision_experience.md", "knowledge/archive_pre_structured/",
    "tests/", "benchmark_infra/", ".feb-archive/",
)
TEXT_EXT = {".md", ".py", ".json", ".txt", ".yaml", ".yml", ".sh", ".tex", ".cfg", ".toml",
            ".ipynb", ".log", ".rst"}
MAX_BYTES = 8 * 1024 * 1024

# A score: 4+ decimal places is far past anything a data description states in passing.
SCORE = re.compile(r"\b\d+\.\d{4,}\b")
# Leaderboard language is a leak even without a number attached.
LB = re.compile(r"private\s+(?:lb|leaderboard)|public\s+(?:lb|leaderboard)|champion_metric",
                re.I)


def benchmark_slugs(root: str) -> dict[str, re.Pattern]:
    """slug -> regex matching that competition by full slug or by short form (s3e16, sep-2022)."""
    mf = os.path.join(root, "docs/rerun_manifest.json")
    if os.path.exists(mf):
        comps = sorted(json.load(open(mf))["competitions"])
    else:  # a bare tree (the unit test's fixture) still has to be checkable
        comps = ["playground-series-s3e16"]
    out = {}
    for c in comps:
        forms = {re.escape(c)}
        m = re.search(r"(s\d+e\d+)$", c)
        if m:
            forms.add(rf"(?<![0-9a-z]){m.group(1)}(?![0-9a-z])")
        m = re.search(r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)-\d{4})$", c)
        if m:
            forms.add(re.escape(m.group(1)))
        out[c] = re.compile("|".join(sorted(forms)), re.I)
    return out


def archive_plan(root: str) -> set[str]:
    script = os.path.join(root, "benchmark_infra/archive_workspaces_for_rerun.sh")
    if not os.path.exists(script):
        return set()
    r = subprocess.run(["bash", script], capture_output=True, text=True, cwd=root, check=True)
    return {ln[len("DRY  mv "):].split(" -> ")[0]
            for ln in r.stdout.splitlines() if ln.startswith("DRY  mv ")}


def is_exempt(rel: str) -> bool:
    return any(rel == p or rel.startswith(p) for p in EXEMPT_PREFIXES)


def under_plan(rel: str, plan: set[str]) -> bool:
    if rel in plan:
        return True
    return any(rel.startswith(p + "/") for p in plan)


def scan_file(path: str, rel: str, slugs: dict[str, re.Pattern]) -> list[str]:
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if len(line) > 4000:
            line = line[:4000]
        score, lb = SCORE.search(line), LB.search(line)
        if not (score or lb):
            continue
        for comp, pat in slugs.items():
            if pat.search(line):
                what = score.group(0) if score else lb.group(0)
                hits.append(f"{rel}:{i}: {comp} :: {what} :: {line.strip()[:110]}")
                break
        if len(hits) >= 5:          # 5 per file is plenty to act on
            break
    return hits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--simulate-archive", action="store_true",
                    help="also skip everything the archive script's dry run would move")
    ap.add_argument("--max-report", type=int, default=40)
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"no such root: {root}", file=sys.stderr)
        return 4

    slugs = benchmark_slugs(root)
    plan = archive_plan(root) if args.simulate_archive else set()
    findings, n_files = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        # never follow a symlinked directory: those are the manifest-verified clean roots
        dirnames[:] = [d for d in sorted(dirnames)
                       if not os.path.islink(os.path.join(dirpath, d))
                       and not is_exempt(rel_dir + d + "/")
                       and not (plan and under_plan(rel_dir + d, plan))]
        for fn in sorted(filenames):
            rel = rel_dir + fn
            if is_exempt(rel) or os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                continue
            if plan and under_plan(rel, plan):
                continue
            full = os.path.join(dirpath, fn)
            if os.path.islink(full) or os.path.getsize(full) > MAX_BYTES:
                continue
            n_files += 1
            findings += scan_file(full, rel, slugs)

    label = "post-archive (simulated)" if args.simulate_archive else "current tree"
    if not findings:
        print(f"clean slate OK — {n_files} readable files scanned, {label}, "
              f"{len(slugs)} benchmark competitions")
        return 0
    by_file: dict[str, int] = {}
    for f in findings:
        by_file[f.split(":", 1)[0]] = by_file.get(f.split(":", 1)[0], 0) + 1
    print(f"CLEAN-SLATE FAILED — {len(findings)} line(s) in {len(by_file)} file(s) state a "
          f"benchmark competition's own result ({label}, {n_files} files scanned)\n")
    for f in findings[:args.max_report]:
        print("  " + f)
    if len(findings) > args.max_report:
        print(f"  ... {len(findings) - args.max_report} more")
    print("\nEach file must be archived by benchmark_infra/archive_workspaces_for_rerun.sh, "
          "redacted, or given an EXEMPT_PREFIXES entry naming what keeps a run out of it.")
    return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
