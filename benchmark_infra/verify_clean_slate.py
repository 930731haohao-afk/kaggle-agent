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

WHAT IS EXEMPT, AND WHY (short on purpose -- an exemption is a promise that some OTHER
mechanism keeps a run out of that file, and each one names its mechanism):
  knowledge/knowledge_base.json  the structured base. Served only through
                                 task_priors_for.py, whose 8 selftest contracts ARE the
                                 isolation guarantee; a raw read is forbidden by
                                 SKILL.md's HARD RULE and by 00_problem_dossier step 3
                                 ("there is no raw prose file to read"), and
                                 tests/test_reverification_repros.py asserts both texts.
  knowledge/experience*.md       served only through query_library.py / suggest-priors,
                                 which self-exclude; raw reads forbidden by the launcher
                                 prompt and by SKILL.md.
  knowledge/idea_bank.md         same doors; consumer channel dormant (round 6).
  archive/, knowledge/archive_pre_structured/, .feb-archive/, .git/
                                 quarantined records. In the RUN ROOT none of these exist
                                 at all -- build_myagent_run_root.py copies by allowlist,
                                 so absence is the mechanism, not a rule.
  competitions_vision/           a different workstream: not among the 20 tabular benchmark
                                 competitions and not copied into the run root. 390k files
                                 and 3 GB of image arrays, none of it on a tabular lane's
                                 read path.
  tests/, benchmark_infra/       the audit's own machinery: these files describe leaks in
                                 order to prevent them. Same allowlist mechanism -- neither
                                 is copied into the run root. They are exempt only so the
                                 repo-scoped --simulate-archive mode stays usable.

The run root is the real boundary; this scan is how you find out whether it held.
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
CLEAN_DATA_ROOT = os.path.join(os.path.expanduser("~"), "ai_agents/bench-comps")

# Read-path exemptions. Each entry is a prefix relative to the root.
EXEMPT_PREFIXES = (
    ".git/", "archive/", "node_modules/", ".venv/", "__pycache__/", ".ruff_cache/",
    ".pytest_cache/", ".mypy_cache/",
    "knowledge/knowledge_base.json", "knowledge/experience", "knowledge/idea_bank.md",
    "knowledge/vision_experience.md", "knowledge/archive_pre_structured/",
    "tests/", "benchmark_infra/", ".feb-archive/", "competitions_vision/",
)
MAX_BYTES = 64 * 1024 * 1024
BINARY_SNIFF = 8 * 1024 * 1024     # how much of a large file is searched

# A score. 3 decimals, not 4: the report that defeated the round-8 gate wrote 0.955 / 0.790 /
# 8.729, and three decimals is already past anything a data description states in passing.
SCORE = re.compile(r"\b\d+\.\d{3,}\b")
# Leaderboard language leaks even with no number attached.
LB = re.compile(r"private\s+(?:lb|leaderboard)|public\s+(?:lb|leaderboard)"
                r"|(?<![\w.])champion_metric(?![\w])|private\s+(?:mape|smape|rmse|auc)",
                re.I)
# How far apart a competition name and a score may be before they stop being about
# each other. A table header is 1-3 lines from its rows; a docstring is hundreds of
# lines from a hyper-parameter constant.
WINDOW = 10
# Numbers that are code, not results: a hyper-parameter grid ([0.001, 0.003, 0.01, ...]), a
# coordinate-descent step tuple, or an ALL_CAPS constant. Without this the 3-decimal
# threshold turns every search space in a pinned evaluator into a finding.
_NUM_SEQ = re.compile(r"[\[(]\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+\s*[\])]")
_CONST = re.compile(r"^\s*[A-Z][A-Z0-9_]{2,}\s*=\s*-?\d+\.\d+\s*(?:#.*)?$")


_SEQ_CONT = re.compile(r"^[\s\-0-9.,()\[\]]+:?\)?:?$")


def _score_in(line: str):
    """The first score-looking number on `line` that is not plainly code."""
    if _CONST.match(line) or (line.strip() and _SEQ_CONT.match(line)):
        return None
    spans = [m.span() for m in _NUM_SEQ.finditer(line)]
    for m in SCORE.finditer(line):
        if not any(a <= m.start() and m.end() <= b for a, b in spans):
            return m
    return None

# Hand-written display names the slug rules cannot derive. Each maps to its slug.
DISPLAY_NAMES = {
    "afsis": "afsis-soil-properties",
    "cat in the dat": "cat-in-the-dat", "cat-in-the-dat": "cat-in-the-dat",
    "citd": "cat-in-the-dat",
    "conway": "conway-s-reverse-game-of-life",
    "reverse game of life": "conway-s-reverse-game-of-life",
    "tps jan 2022": "tabular-playground-series-jan-2022",
    "tpsjan22": "tabular-playground-series-jan-2022",
    "tps aug 2022": "tabular-playground-series-aug-2022",
    "tpsaug22": "tabular-playground-series-aug-2022",
    "tps sep 2022": "tabular-playground-series-sep-2022",
    "tpssep22": "tabular-playground-series-sep-2022",
    "sep22": "tabular-playground-series-sep-2022",
}


def benchmark_slugs(root: str) -> dict[str, re.Pattern]:
    """slug -> regex matching that competition by full slug, short form, or display name."""
    mf = os.path.join(root, "docs/rerun_manifest.json")
    if os.path.exists(mf):
        comps = sorted(json.load(open(mf))["competitions"])
    else:  # a bare tree (the unit tests' fixtures) still has to be checkable
        mf = os.path.join(DEFAULT_ROOT, "docs/rerun_manifest.json")
        comps = (sorted(json.load(open(mf))["competitions"]) if os.path.exists(mf)
                 else ["playground-series-s3e16"])
    out = {}
    for c in comps:
        # RIGHT boundary on the full slug too: 'playground-series-s3e1' is a prefix of
        # 'playground-series-s3e19', so without it every s3e19 hit was filed under s3e1.
        forms = {rf"{re.escape(c)}(?![0-9a-z])"}
        m = re.search(r"(s\d+e\d+)$", c)
        if m:
            forms.add(rf"(?<![0-9a-z]){m.group(1)}(?![0-9a-z])")
        m = re.search(r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)-\d{4})$", c)
        if m:
            forms.add(re.escape(m.group(1)))
        for name, slug in DISPLAY_NAMES.items():
            if slug == c:
                forms.add(rf"(?<![0-9a-z]){re.escape(name)}(?![0-9a-z])")
        out[c] = re.compile("|".join(sorted(forms)), re.I)
    return out


def archive_plan(root: str) -> set[str]:
    script = os.path.join(root, "benchmark_infra/archive_workspaces_for_rerun.sh")
    if not os.path.exists(script):
        return set()
    r = subprocess.run(["bash", script], capture_output=True, text=True, cwd=root, check=True)
    return {ln[len("DRY  mv "):].split(" -> ")[0]
            for ln in r.stdout.splitlines() if ln.startswith("DRY  mv ")}


# competitions/<comp>/{data,data_official}/ — the official competition files. Mechanism:
# nothing reaches a
# data directory except by being present in ~/ai_agents/bench-comps/<comp>/data, which
# build_isolated_roots.py populates from Kaggle's own `competitions files` manifest. That is
# what makes a file official rather than a previous run's intermediate, and it is the exact
# distinction round 8 found the archive getting wrong (OOF matrices and Optuna champions
# sitting in data/). data_official/ is the same set from the other side: every entry of
# bench-comps/<comp>/data is a symlink INTO it. Scanning 1.4 GB of raw competition data on
# every gate run buys nothing.
_OFFICIAL_DATA = re.compile(r"^competitions/[^/]+/(data|data_official)/")


# Machine-generated caches, exempt wherever they appear (tree_search/__pycache__/ is not
# matched by a top-level "__pycache__/" prefix).
EXEMPT_DIR_NAMES = {"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}


def is_exempt(rel: str) -> bool:
    if _OFFICIAL_DATA.match(rel):
        return True
    if EXEMPT_DIR_NAMES.intersection(rel.split("/")):
        return True
    return any(rel == p or rel.startswith(p) for p in EXEMPT_PREFIXES)


def under_plan(rel: str, plan: set[str]) -> bool:
    if rel in plan:
        return True
    return any(rel.startswith(p + "/") for p in plan)


def _read(path: str) -> tuple[str, bool] | None:
    """Text if we can get it, else a lossy decode of the head. None only on an OS error.

    Extension whitelists were the round-8 gate's largest hole: .db, .csv, .html and
    extensionless files were never opened, which is exactly where mlflow.db's 284 runs and
    the sibling tree's .out lane logs lived. Everything gets looked at now; unparseable
    bytes are searched as latin-1 over the first BINARY_SNIFF bytes, which is enough to
    find a slug next to a score in a SQLite page or a CSV header.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    try:
        with open(path, "rb") as fh:
            raw = fh.read(BINARY_SNIFF if size > BINARY_SNIFF else size)
    except OSError:
        return None
    binary = b"\x00" in raw[:8192]
    try:
        return raw.decode("utf-8"), binary
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace"), True


def scan_file(path: str, rel: str, slugs: dict[str, re.Pattern]) -> list[str]:
    """Flag a file where a benchmark competition is named WITHIN `WINDOW` lines of a score or
    leaderboard language.

    Not same-line: a markdown table puts 'Private LB' in the header row and the competition
    in a data row, so the round-8 same-line rule passed a private-LB table for nine lanes --
    the same positional channel rounds 5-8 kept finding in the knowledge views. Not
    whole-file either: source code mentions a competition in its docstring and a
    hyper-parameter 300 lines later, which is not a leak. A window is what actually
    distinguishes "this document is about this competition's result" from a coincidence.
    """
    got = _read(path)
    if got is None:
        return []
    text, binary = got
    lines = text.splitlines()
    named: dict[str, list[int]] = {}
    for i, line in enumerate(lines, 1):
        for comp, pat in slugs.items():
            if pat.search(line):
                named.setdefault(comp, []).append(i)
    if not named:
        return []
    signals = []
    for i, ln in enumerate(lines, 1):
        m = _score_in(ln) or LB.search(ln)
        if m:
            signals.append((i, m.group(0)))
    if not signals:
        if binary:
            # A binary artifact naming a competition cannot be cleared by reading it:
            # mlflow.db stores its 284 runs' metrics as SQLite REALs, so the slug shows in
            # the page text and the score does not. Unverifiable is not clean (round 9).
            first = min(named, key=lambda c: named[c][0])
            return [f"{rel}:{named[first][0]}: {first} :: BINARY artifact naming a benchmark "
                    f"competition — cannot be cleared by inspection; keep it out of the run "
                    f"root or justify it explicitly"]
        return []
    hits = []
    for comp in sorted(named, key=lambda c: named[c][0]):
        best = None
        for cline in named[comp]:
            for sline, what in signals:
                d = abs(sline - cline)
                if d <= WINDOW and (best is None or d < best[0]):
                    best = (d, cline, sline, what)
        if best:
            _d, cline, sline, what = best
            hits.append(f"{rel}:{cline}: {comp} :: {what} (line {sline}) :: "
                        f"{lines[sline - 1].strip()[:110]}")
        if len(hits) >= 5:
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
    skipped: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        keep = []
        for d in sorted(dirnames):
            rel = rel_dir + d
            if is_exempt(rel + "/") or (plan and under_plan(rel, plan)):
                continue
            full = os.path.join(dirpath, d)
            if os.path.islink(full):
                # A symlinked directory used to be pruned on the assumption that it is a
                # manifest-verified clean root. That is true of the 5 competitions/<c>/data
                # links and of nothing else, and it was enforced nowhere (round 9).
                target = os.path.realpath(full)
                if target.startswith(os.path.realpath(CLEAN_DATA_ROOT) + os.sep):
                    continue
                skipped.append(f"{rel}/ -> {target} (symlinked directory outside the "
                               f"manifest-verified data root)")
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            rel = rel_dir + fn
            if is_exempt(rel) or (plan and under_plan(rel, plan)):
                continue
            full = os.path.join(dirpath, fn)
            if os.path.islink(full):
                target = os.path.realpath(full)
                if not os.path.exists(target):
                    skipped.append(f"{rel} -> {target} (DANGLING symlink)")
                    continue
                if target.startswith(os.path.realpath(CLEAN_DATA_ROOT) + os.sep):
                    continue
                # a lane reads straight through a symlink; follow it
            try:
                size = os.path.getsize(full)
            except OSError as exc:
                skipped.append(f"{rel} (unstatable: {exc})")
                continue
            if size > MAX_BYTES:
                skipped.append(f"{rel} ({size // 1024 // 1024} MB > "
                               f"{MAX_BYTES // 1024 // 1024} MB cap — only the first "
                               f"{BINARY_SNIFF // 1024} KB would be searched)")
            n_files += 1
            findings += scan_file(full, rel, slugs)

    label = "post-archive (simulated)" if args.simulate_archive else "current tree"
    if skipped:
        print(f"NOT FULLY EXAMINED — {len(skipped)} path(s):")
        for s in skipped[:20]:
            print("  " + s)
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")
        print()
    if not findings:
        # A skip is not a pass. The round-8 gate counted only files it read and printed that
        # count as if it were coverage, so "clean" and "never opened" looked identical.
        if skipped:
            print(f"INCONCLUSIVE — {n_files} files examined and clean, but {len(skipped)} "
                  f"path(s) above were not fully examined ({label})")
            return 3
        print(f"clean slate OK — {n_files} readable files examined, {label}, "
              f"{len(slugs)} benchmark competitions")
        return 0
    by_file: dict[str, int] = {}
    for f in findings:
        by_file[f.split(":", 1)[0]] = by_file.get(f.split(":", 1)[0], 0) + 1
    print(f"CLEAN-SLATE FAILED — {len(findings)} finding(s) in {len(by_file)} file(s) state a "
          f"benchmark competition's own result ({label}, {n_files} files examined)\n")
    for f in findings[:args.max_report]:
        print("  " + f)
    if len(findings) > args.max_report:
        print(f"  ... {len(findings) - args.max_report} more")
    print("\nEach file must be kept out of the run root, redacted, or given an "
          "EXEMPT_PREFIXES entry naming what keeps a run out of it.")
    return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
