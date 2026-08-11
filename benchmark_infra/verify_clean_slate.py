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
import hashlib
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

# A score, two ways. Precision alone is the wrong axis -- what makes a number a score is
# the metric word beside it, and MAE ~340 / SMAPE ~48.24 are routinely written with 0-2
# decimals (2026-08-10 round-10). So: 3+ decimals anywhere, OR any number adjacent to a
# metric word.
SCORE = re.compile(r"\b\d+\.\d{3,}\b")
METRIC = (r"cv|oof|lb|leaderboard|score|auc|roc[_ ]?auc|rmsle|rmse|mae|mape|smape|kappa|qwk|"
          r"r2|accuracy|logloss|mcrmse|private|public")
# ADJACENT, not merely same-line: "MAE 340" and "SMAPE 48.24" are scores; `random_state=42`
# and a 2022 in a competition name are not, and a same-line rule called both a score.
SCORE_NEAR_METRIC = re.compile(
    rf"(?:\b(?:{METRIC})\b[^\w\n]{{0,4}}(\d+(?:\.\d+)?)\b)"
    rf"|(?:\b(\d+(?:\.\d+)?)[^\w\n]{{0,4}}\b(?:{METRIC})\b)", re.I)
# Leaderboard language leaks even with no number attached.
LB = re.compile(r"private\s+(?:lb|leaderboard)|public\s+(?:lb|leaderboard)"
                r"|(?<![\w.])champion_metric(?![\w])|private\s+(?:mape|smape|rmse|auc)",
                re.I)
# How far apart a competition name and a score may be before they stop being about each
# other. A table header is 1-3 lines from its rows; a docstring is hundreds of lines from a
# hyper-parameter constant.
WINDOW = 10
# A file that names a competition and carries this many HIGH-PRECISION values anywhere is a
# result table however it is laid out -- a legend at the top and rows below defeats any
# proximity rule, and that is the most natural way to write one (round 10).
TABLE_SCORES = 3
# Numbers that are code, not results: a hyper-parameter grid of three or more values.
_NUM_SEQ = re.compile(r"[\[(]\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){2,}\s*[\])]")
# a continuation line of a multi-line grid: only numbers, and three or more of them, so the
# two-line "name / single indented value" listing is still seen
_SEQ_CONT = re.compile(r"^[\s\-0-9.,()\[\]]+:?\)?:?$")
# An ALL_CAPS constant is code -- unless the constant's own NAME carries a competition, which
# is exactly the `S5E1_PRIVATE = 0.06253` shape (round 10).
_CONST = re.compile(r"^\s*([A-Z][A-Z0-9_]{2,})\s*=\s*-?\d+\.\d+\s*(?:#.*)?$")


def _score_in(line: str):
    """(match, strong) for the first score-looking number on `line`, else None.

    Strong = 3+ decimal places, the shape nothing states in passing. Weak = any number
    sitting against a metric word, which is how large-scale metrics get written. Round 9
    suppressed whole lines that looked like code (ALL_CAPS constants, bare numeric lines);
    that swallowed `S5E1_PRIVATE = 0.06253` and the universal two-line "name / indented
    value" listing, so only genuine multi-value grids are excluded now (round 10).
    """
    if _SEQ_CONT.match(line) and len(re.findall(r"\d+\.\d+", line)) >= 3:
        return None
    mc = _CONST.match(line)
    if mc and not re.search(r"(?<![0-9A-Z])(S\d+E\d+|AFSIS|CONWAY|CITD)(?![0-9A-Z])",
                            mc.group(1)):
        return None
    spans = [m.span() for m in _NUM_SEQ.finditer(line)]

    def outside(m):
        return not any(a <= m.start() and m.end() <= b for a, b in spans)

    for m in SCORE.finditer(line):
        if outside(m):
            return m, True
    for m in SCORE_NEAR_METRIC.finditer(line):
        if not outside(m):
            continue
        num = next(g for g in m.groups() if g)
        # a weak hit must at least look like a measurement: 2+ decimals, or 3+ digits
        if re.fullmatch(r"\d+\.\d{2,}", num) or re.fullmatch(r"\d{3,}", num):
            return m, False
    return None


# Hand-written short forms the slug rules cannot derive.
DISPLAY_NAMES = {
    "afsis": "afsis-soil-properties",
    "cat in the dat": "cat-in-the-dat", "citd": "cat-in-the-dat",
    "conway": "conway-s-reverse-game-of-life",
    "reverse game of life": "conway-s-reverse-game-of-life",
    "tps jan 2022": "tabular-playground-series-jan-2022",
    "tpsjan22": "tabular-playground-series-jan-2022",
    "tps aug 2022": "tabular-playground-series-aug-2022",
    "tpsaug22": "tabular-playground-series-aug-2022",
    "tps sep 2022": "tabular-playground-series-sep-2022",
    "tpssep22": "tabular-playground-series-sep-2022", "sep22": "tabular-playground-series-sep-2022",
}


def _titles_from_configs(root: str, comp: str) -> set[str]:
    """Human titles for `comp`, read from its own config.yaml.

    A hand-written abbreviation list covered no competition's actual TITLE -- "Crab Age
    Dataset", "Africa Soil Property Prediction Challenge" -- while the run root ships
    config.yaml as the decoder ring that lets a reader join those titles to a slug
    (2026-08-10 round-10). Deriving them from the same file closes the gap and cannot go
    stale.
    """
    out: set[str] = set()
    cfg = os.path.join(root, "competitions", comp, "config.yaml")
    if not os.path.exists(cfg):
        return out
    for line in open(cfg, encoding="utf-8", errors="replace").read().splitlines():
        m = re.match(r"\s*(title|name|description)\s*:\s*(.+?)\s*$", line)
        if not m:
            continue
        val = m.group(2).strip().strip('"\'')
        for phrase in re.findall(r"[A-Z][\w'-]*(?:\s+[A-Z0-9][\w'-]*){1,5}", val):
            if len(phrase) >= 12:
                out.add(phrase)
        m2 = re.search(r"\(([^)]{8,60})\)", val)
        if m2:
            out.add(m2.group(1).strip())
    return out


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
        for title in _titles_from_configs(root, c):
            forms.add(re.escape(title))
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
_OFFICIAL_DATA = re.compile(r"^competitions/([^/]+)/(?:data|data_official)/(.+)$")


def _is_official_file(rel: str) -> bool:
    """True only for a data file whose NAME is in that competition's clean root.

    A prefix exemption was a hole the run itself writes into: the pinned evaluators read
    competitions/<comp>/data/train_processed.csv, which Stage 2 writes there, and the gate
    runs once before lane 1 of 20 in a shared root -- so a leftover from an aborted attempt
    at the same competition sat in an exempt directory (2026-08-10 round-10). Membership in
    the Kaggle-manifest-verified clean root is the actual mechanism; check it.
    """
    m = _OFFICIAL_DATA.match(rel)
    if not m:
        return False
    comp, tail = m.group(1), m.group(2)
    root = os.path.join(CLEAN_DATA_ROOT, comp, "data")
    return os.path.exists(os.path.join(root, tail))


# Machine-generated caches, exempt wherever they appear (tree_search/__pycache__/ is not
# matched by a top-level "__pycache__/" prefix).
EXEMPT_DIR_NAMES = {"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}


BASELINE_FILE = ".rerun_baseline.json"


def sidecar_path(root: str) -> str:
    """Where the baseline's own hash lives: BESIDE the root, not inside it.

    The run root is bound writable into every lane, and the manifest inside it is exempt from
    scanning and absent from its own file list -- so it could be rewritten, or emptied, and
    every frozen-file check would pass. The sandbox blanks the root's parent, so a lane can
    neither read nor write this (round 12).
    """
    return os.path.abspath(root).rstrip(os.sep) + ".baseline.sha256"


def baseline_tampered(root: str) -> str | None:
    """Non-None if the manifest does not match the hash recorded outside the root."""
    side = sidecar_path(root)
    bpath = os.path.join(root, BASELINE_FILE)
    if not os.path.exists(side):
        return None
    if not os.path.exists(bpath):
        return f"{BASELINE_FILE} is missing but {os.path.basename(side)} records one"
    want = open(side).read().strip()
    got = hashlib.sha256(open(bpath, "rb").read()).hexdigest()
    if got != want:
        return (f"{BASELINE_FILE} does not match the hash recorded beside the root: the "
                f"manifest that defines which files are frozen has itself been rewritten")
    return None


def load_baseline(root: str) -> dict[str, str] | None:
    """The frozen surface, as recorded by build_myagent_run_root.py at build time.

    Absent for a tree that was never built by the builder -- the repo itself under
    --simulate-archive, and the unit fixtures -- in which case every file is treated as
    frozen and the gate behaves exactly as it did before.
    """
    p = os.path.join(root, BASELINE_FILE)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError):
        return None


def _own_comp(rel: str, slugs: dict) -> str | None:
    parts = rel.split("/")
    if len(parts) >= 2 and parts[0] == "competitions" and parts[1] in slugs:
        # NOT data/. That directory holds official Kaggle files and nothing else, so a
        # score-bearing file there is a leftover from an aborted attempt, not a deliverable
        # -- the one thing a relaunched lane must not find (round 10).
        if len(parts) >= 3 and parts[2] == "data":
            return None
        return parts[1]
    return None


def slugs_for(rel: str, slugs: dict, baseline: dict[str, str] | None) -> dict:
    """Which competitions this file may not name.

    A file the builder put there may name none of them. A file the RUN produced under
    competitions/<comp>/ may name <comp> -- that is the deliverable the launcher orders each
    lane to write ("a 3-line summary ... with the final CV score"), and gating startup on a
    rule that rejects it made every relaunch exit 3 before lane 2 began (round 11). It may
    still name no OTHER competition: the exemption is per-lane, so a lane that writes a
    sibling's score is caught by exactly the rule it was always caught by.
    """
    if baseline is None or rel in baseline:
        return slugs
    own = _own_comp(rel, slugs)
    return {c: p for c, p in slugs.items() if c != own} if own else slugs


def baseline_findings(root: str, baseline: dict[str, str] | None) -> list[str]:
    """The frozen surface must still be what the builder wrote.

    Round 9 put the memory guard and round 10 the auto-memory setting in the BUILDER, but
    run_myagent_headless.sh never runs the builder -- it checks `[ -d $RUN_ROOT ]` and then
    this gate. So a root reused across nights, built by an older revision, or clobbered by a
    lane started all 20 lanes with those guards silently absent (round 11). Hashing what the
    builder froze is what makes a build-time guard enforceable at START time.
    """
    if not baseline:
        return []
    out = []
    for rel, want in sorted(baseline.items()):
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            out.append(f"{rel}: frozen at build time and now MISSING — the run root is not "
                       f"the tree the builder verified")
            continue
        try:
            got = hashlib.sha256(open(full, "rb").read()).hexdigest()
        except OSError as exc:
            out.append(f"{rel}: frozen at build time and unreadable now ({exc})")
            continue
        if got != want:
            out.append(f"{rel}: frozen at build time and MODIFIED since — a guard the "
                       f"builder wrote cannot be assumed to still be there")
    return out


def is_exempt(rel: str) -> bool:
    if rel == BASELINE_FILE:
        return True
    if _is_official_file(rel):
        return True
    if EXEMPT_DIR_NAMES.intersection(rel.split("/")):
        return True
    return any(rel == p or rel.startswith(p) for p in EXEMPT_PREFIXES)


def under_plan(rel: str, plan: set[str]) -> bool:
    if rel in plan:
        return True
    return any(rel.startswith(p + "/") for p in plan)


def _read(path: str) -> tuple[str, bool, bool] | None:
    """(text, is_binary, truncated) — or None if the file could not be opened.

    UTF-16 was invisible to every rule: it decodes under latin-1 as NUL-interleaved
    characters, so no slug matched and the file returned before even reaching the binary
    rule (2026-08-10 round-10). BOM-marked UTF-16 is now decoded properly, and a
    NUL-interleaved body is retried as UTF-16 before being called binary.
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
    truncated = size > BINARY_SNIFF
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16"), False, truncated
        except UnicodeDecodeError:
            pass
    head = raw[:8192]
    if b"\x00" in head:
        # every other byte NUL is UTF-16 text, not a binary artifact
        nul_even = head[1::2].count(0)
        nul_odd = head[0::2].count(0)
        if max(nul_even, nul_odd) > len(head) // 4:
            for enc in ("utf-16-le", "utf-16-be"):
                try:
                    return raw.decode(enc), False, truncated
                except UnicodeDecodeError:
                    continue
    try:
        return raw.decode("utf-8"), b"\x00" in head, truncated
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace"), True, truncated


def names_any_competition(path: str, slugs: dict) -> bool | None:
    """Does this file contain ANY competition token, anywhere in its bytes?

    For files too large to decode whole. A finding requires a competition to be NAMED, so a
    file naming none cannot state a result however big it is -- and that is decidable by
    streaming. Without this, every file over 8 MB became a `skipped` entry and therefore
    INCONCLUSIVE, and since the launcher gates EVERY start on this, the first lane to write a
    40 MB submission made all subsequent relaunches impossible (round 12).

    Returns None if the file could not be read at all -- that is a real skip.
    """
    tokens = set()
    for comp in slugs:
        tokens.add(comp.lower().encode())
        m = re.search(r"(s\d+e\d+)$", comp)
        if m:
            tokens.add(m.group(1).lower().encode())
    longest = max(len(t) for t in tokens) if tokens else 0
    chunk = 4 * 1024 * 1024
    try:
        with open(path, "rb") as f:
            tail = b""
            while True:
                buf = f.read(chunk)
                if not buf:
                    return False
                hay = (tail + buf).lower()
                if any(t in hay for t in tokens):
                    return True
                tail = hay[-longest:] if longest else b""
    except OSError:
        return None


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
    text, binary, _trunc = got
    lines = text.splitlines()
    named: dict[str, list[int]] = {}
    for i, line in enumerate(lines, 1):
        for comp, pat in slugs.items():
            if pat.search(line):
                named.setdefault(comp, []).append(i)
    if not named:
        return []
    signals, strong = [], []
    for i, ln in enumerate(lines, 1):
        got = _score_in(ln)
        if got:
            m, is_strong = got
            signals.append((i, m.group(0).strip()))
            if is_strong:
                strong.append((i, m.group(0)))
            continue
        m = LB.search(ln)
        if m:
            signals.append((i, m.group(0)))
            strong.append((i, m.group(0)))
    if binary:
        # Unconditional now: round 9 fired this only when the file had NO signal anywhere,
        # so one unrelated decimal (a version string, a timestamp) turned a hard finding
        # into silence (round 10).
        first = min(named, key=lambda c: named[c][0])
        return [f"{rel}:{named[first][0]}: {first} :: BINARY artifact naming a benchmark "
                f"competition — cannot be cleared by inspection; keep it out of the run "
                f"root or justify it explicitly"]
    if not signals:
        if False:
            # A binary artifact naming a competition cannot be cleared by reading it:
            # mlflow.db stores its 284 runs' metrics as SQLite REALs, so the slug shows in
            # the page text and the score does not. Unverifiable is not clean (round 9).
            first = min(named, key=lambda c: named[c][0])
            return [f"{rel}:{named[first][0]}: {first} :: BINARY artifact naming a benchmark "
                    f"competition — cannot be cleared by inspection; keep it out of the run "
                    f"root or justify it explicitly"]
        return []
    # A legend at the top and a sorted table below defeats any proximity rule, and it is the
    # most natural way to write a multi-lane result table (round 10). If the file names a
    # competition at all and carries TABLE_SCORES score-shaped numbers, the layout does not
    # matter.
    if len(strong) >= TABLE_SCORES:
        first = min(named, key=lambda c: named[c][0])
        return [f"{rel}:{named[first][0]}: {first} :: {len(strong)} high-precision values in "
                f"one file naming a benchmark competition (first at line {strong[0][0]}: "
                f"{strong[0][1]}) — a result table, whatever its layout"]
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
    ap.add_argument("--require-baseline", action="store_true",
                    help="fail unless the root carries the builder's baseline manifest")
    ap.add_argument("--simulate-archive", action="store_true",
                    help="also skip everything the archive script's dry run would move")
    ap.add_argument("--max-report", type=int, default=40)
    ap.add_argument("--lane-isolated", action="store_true",
                    help="each lane sees only its own competitions/<comp>/ (lane_sandbox.sh "
                         "with LANE_COMP). Files THIS RUN wrote inside a workspace are then "
                         "readable by one lane only -- the one that wrote them -- so they "
                         "are not scanned for sibling competitions. Without this the gate "
                         "scans them, which rejects a finished lane for citing the "
                         "experience library exactly as SKILL.md requires.")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"no such root: {root}", file=sys.stderr)
        return 4

    slugs = benchmark_slugs(root)
    baseline = load_baseline(root)
    tampered = baseline_tampered(root)
    if tampered:
        print(f"BASELINE TAMPERED — {tampered}", file=sys.stderr)
        return 3
    if args.require_baseline and not os.path.exists(sidecar_path(root)):
        # baseline_tampered() cannot speak without the sidecar -- it early-returns None, which
        # reads as "not tampered". So the check an attacker has to defeat is the one that only
        # fires if they leave the witness in place, and deleting it is easier than forging it.
        # A root built before the sidecar existed lands here too, and that root is stale by
        # construction: rebuild rather than trust a manifest nothing vouches for.
        print(f"no sidecar at {sidecar_path(root)}: nothing outside the root vouches for "
              f"{BASELINE_FILE}, so the manifest that defines which files are frozen cannot "
              f"be checked. Rebuild with build_myagent_run_root.py.", file=sys.stderr)
        return 3
    if args.require_baseline and baseline is None:
        # The launcher gates startup on this. A root with no baseline is one the builder
        # never produced -- stale, hand-made, or half-copied -- and the quarantine then
        # refuses every competition while the loop logs a completion marker anyway (round 12).
        print(f"no {BASELINE_FILE} in {root}: this root was not produced by "
              f"build_myagent_run_root.py, so its frozen surface cannot be verified",
              file=sys.stderr)
        return 3
    plan = archive_plan(root) if args.simulate_archive else set()
    findings, n_files = [], 0
    own_workspace = 0
    skipped: list[str] = []
    # os.walk's default swallows a listdir failure and yields NOTHING for that subtree, so an
    # unreadable DIRECTORY was the one unexaminable path that never reached `skipped` -- it
    # took the "no findings, no skips" branch and printed a normal pass, while every
    # unreadable FILE correctly forced INCONCLUSIVE. A skip is not a pass (round 11).
    for dirpath, dirnames, filenames in os.walk(
            root, followlinks=False,
            onerror=lambda e: skipped.append(
                f"{os.path.relpath(getattr(e, 'filename', root), root)}/ "
                f"(directory could not be listed: {e})")):
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
            if size > BINARY_SNIFF:
                # round 9 only reported files past MAX_BYTES, but _read stops at
                # BINARY_SNIFF, so everything in between was searched partially and
                # certified whole (round 10). Round 12: that made every relaunch impossible
                # once a lane wrote a 40 MB submission, so decide it by streaming first --
                # a file naming no competition at all cannot state one's result.
                named_anywhere = names_any_competition(full, slugs)
                if named_anywhere is False:
                    n_files += 1
                    continue
                skipped.append(f"{rel} ({size // 1024 // 1024} MB — names a benchmark "
                               f"competition and only the first "
                               f"{BINARY_SNIFF // 1024 // 1024} MB could be searched)"
                               if named_anywhere else f"{rel} (could not be opened)")
            got = _read(full)
            if got is None:
                skipped.append(f"{rel} (could not be opened)")
                continue
            n_files += 1
            # A file THIS RUN produced inside a competition workspace, when every lane is
            # confined to its own workspace, can be read by exactly one lane: the one that
            # wrote it. Scanning it against the sibling slugs rejected the lane for obeying
            # the skill -- SKILL.md requires a library_hits trace on every experiment and
            # query_library.py serves evidence from OTHER competitions by design -- so one
            # finished lane made every relaunch exit 3 before lane 2 began.
            #
            # The exemption is granted only under --lane-isolated, and only the launcher
            # passes it, together with the LANE_COMP that lane_sandbox.sh now requires. Run
            # by hand the gate stays strict, and a test binds the two so neither can drift
            # alone. What this does NOT cover is the lane's own workspace holding a PREVIOUS
            # run's answer; that is the quarantine's job, per-lane, before the lane starts.
            if args.lane_isolated and baseline is not None and rel not in baseline \
                    and _own_comp(rel, slugs):
                own_workspace += 1
                continue
            findings += scan_file(full, rel, slugs_for(rel, slugs, baseline))

    findings += baseline_findings(root, baseline)

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
        # Never silent about what was exempted. A count printed as coverage while files went
        # unscanned is the round-8 shape this gate's own INCONCLUSIVE branch exists to avoid.
        extra = (f", {own_workspace} of them this run's own lane output inside its workspace "
                 f"(readable by that lane alone under --lane-isolated)"
                 if own_workspace else "")
        print(f"clean slate OK — {n_files} readable files examined{extra}, {label}, "
              f"{len(slugs)} benchmark competitions")
        return 0
    by_file: dict[str, int] = {}
    for f in findings:
        by_file[f.split(":", 1)[0]] = by_file.get(f.split(":", 1)[0], 0) + 1
    print(f"CLEAN-SLATE FAILED — {len(findings)} finding(s) in {len(by_file)} file(s): a "
          f"benchmark competition's own result is stated where the lane could read it, or "
          f"the builder's frozen surface no longer matches ({label}, {n_files} files "
          f"examined)\n")
    for f in findings[:args.max_report]:
        print("  " + f)
    if len(findings) > args.max_report:
        print(f"  ... {len(findings) - args.max_report} more")
    print("\nEach file must be kept out of the run root, redacted, or given an "
          "EXEMPT_PREFIXES entry naming what keeps a run out of it.")
    return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
