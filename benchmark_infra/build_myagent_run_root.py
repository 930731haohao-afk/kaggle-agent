#!/usr/bin/env python3
"""Build the isolated root my-agent runs the 20-competition benchmark from.

WHY A ROOT INSTEAD OF AN ARCHIVE. Rounds 4-9 all tried to define the clean slate by
SUBTRACTION: take the repo, remove everything that leaks. Round 9 showed the boundary itself
was wrong — the leak surface does not stop at the repo:

  * the harness auto-injects ~/.claude/projects/<key-derived-from-cwd>/memory/MEMORY.md,
    which for the recorded run's cwd names 8 lanes' CV scores and winning recipes, before
    the lane's first tool call;
  * the archive is a `mv`, so `git show` retrieves every archived file, and .git/ is the one
    directory no scan can meaningfully redact;
  * /home/tjyen/ai_agents/ — the repo's parent, which the launcher sources a lock script
    from and which 5 workspaces' data/ symlinks point into — holds the all-20 public/private
    table and the frozen AIDE/NVIDIA references;
  * documents/ (a near-homograph of the swept docs/) holds a private-LB table for 9 lanes;
  * mlflow.db mirrors 16 lanes' metric, score history and hyper-parameters.

Each of those is one more entry on a denylist that lost five rounds running. This script
inverts it: the run root contains what a lane NEEDS and nothing else, and everything above is
absent because it was never copied — not because a glob remembered it. Same structural fix
`build_isolated_roots.py` applied to the other agents' data roots after four runs were voided
by my own intermediates; this applies it to my-agent's own workspace.

Competition data costs nothing to include: every official file is HARDLINKED into the run
root from ~/ai_agents/bench-comps/<comp>/data, the Kaggle-manifest-verified
official-files-only root every lane shares. Not symlinked -- each entry under bench-comps is
itself a symlink back into kaggle/competitions/<comp>/data_official/, so a symlinked data/
resolves into the repo and leaves the lane one `..` from the tree this root excludes.

Usage:
    python3 benchmark_infra/build_myagent_run_root.py                 # dry run, list plan
    python3 benchmark_infra/build_myagent_run_root.py --write         # build
    python3 benchmark_infra/build_myagent_run_root.py --write --root /tmp/x   # elsewhere

Then verify:  python3 benchmark_infra/verify_clean_slate.py --root <run-root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# NOT under ~/ai_agents. This file's own header names that directory as one of the five leak
# surfaces that justified moving the run out of the repo -- MASTER_TODO.md and LANE_CLAIMS.md
# carry the all-20 public/private table -- and then the default put the run root one `..`
# from it, while the gate only ever walks INSIDE --root (round 11). A neutral parent holding
# nothing but run roots is the point, and _check_parent below enforces it.
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "benchruns/myagent-rerun")
CLEAN_DATA = os.path.join(os.path.expanduser("~"), "ai_agents/bench-comps")

# --- the allowlist ----------------------------------------------------------------------
# Whole directories a lane needs in full.
DIRS = [
    ".claude/skills/kaggle-agent",          # the pipeline itself
    "external_data",                        # rules gate, dossier, operators, source caches
    "utils",                                # experiment log, evaluation helpers
    "templates",
]
# Never copied out of an allowlisted directory, whatever else that directory holds.
DIR_EXCLUDE = ["kaggle_auth.sh", "__pycache__", "*.pyc"]
# Individual files, path-for-path.
FILES = [
    "knowledge/knowledge_base.json",        # served ONLY through task_priors_for.py
    "knowledge/task_priors_for.py",
    "knowledge/query_library.py",
    "knowledge/experience.md",              # served ONLY through query_library.py
    # the ONE piece of kaggle-safe-submit the pipeline uses (06_submission.md step 5). The
    # rest of that skill prescribes `kaggle competitions submissions -c <comp>`, which
    # returns this competition's own prior public LB scores (round 10).
    ".claude/skills/kaggle-safe-submit/scripts/validate_submission.py",
    "docs/rerun_manifest.json",
    "pyproject.toml",
    "uv.lock",
]
# tree_search: the harness and the competition-agnostic tools. The pinned evaluators are
# added per competition from the manifest. Everything else in that directory -- the
# reproduction drivers, the frozen proposer inputs, the non-pinned eval_*.py holding recorded
# arm scores, the lane logs -- is left behind (2026-08-10 round-8, round-9 #10/#18).
TREE_TOOLS = [
    "harness.py", "harness_v2.py", "harness_v3.py", "eval_support.py",
    "stage2_inputs.py", "make_v5_arm.py", "run_template_v3.py",
]
# Inside a competition workspace, only these are inputs.
WS_FILES = ["config.yaml", "rules_verdict.json"]

# Explicitly NOT copied, and why (each one leaked in an audit round):
NOT_COPIED = """
  .git/                     git show retrieves anything ever committed (round 9 #20)
  docs/ (except manifest)   every lane's scores for all three agents (round 8)
  documents/                private-LB table for 9 lanes (round 9 #1/#5/#17)
  benchmark_results/        the three-way comparison tables (round 8)
  mlflow.db                 16 lanes' metric, score history, hyper-parameters (round 9 #4)
  tests/                    quote recorded champion scores (round 9 #8)
  benchmark_infra/          the audit harness; a lane has no reason to read its judge
  competitions/*/           experiments.json, STATUS.md, scripts/, submissions/, facts.json
  competitions/_batch*      the recorded batch run's baselines and blend weights (round 8)
  tree_search/run_*_v3.py   recorded root configs and digit-verify targets (round 6)
  tree_search/llm_*.json    champion metric + champion config + self-citing priors (round 8)
  tree_search/cache_*/      OOF vectors keyed by node id (round 8)
  .claude/skills/kaggle-agent-self-improvement/   prescribes scanning competitions/*/
                            experiments.json, which is how sibling workspaces leaked (round 8)
  .superpowers/             overnight plan files quoting per-lane four-stage score ladders
  utils/kaggle_auth.sh      points at ~/.kaggle/huang_token, the account holding every
                            benchmark submission for all three lanes; with it,
                            `kaggle competitions submissions -c <comp>` returns this
                            competition's own prior public LB scores (round 10)
  .claude/skills/kaggle-safe-submit/SKILL.md   prescribes exactly that call at its quota
                            check and back-fill steps; only its validator script is copied
"""


def project_key(root: str) -> str:
    """How the harness derives its per-project state directory from a working directory.

    The CLI replaces EVERY non-alphanumeric character, not just the separator, and truncates
    at 200 characters. Guarding `.replace("/", "-")` guarded a path nothing can ever create,
    so the one check standing between the re-run and the recorded run's memory file always
    printed "absent" (2026-08-10 round-10). The recorded run's own directory proves the rule:
    /home/tjyen/ai_agents/kaggle -> -home-tjyen-ai-agents-kaggle, underscore to dash.
    """
    key = re.sub(r"[^a-zA-Z0-9]", "-", os.path.abspath(root))
    if len(key) > 200:                      # the CLI appends a hash of the full path
        key = key[:200] + "-" + hashlib.sha256(os.path.abspath(root).encode()).hexdigest()[:8]
    return key


def memory_dir(root: str) -> str:
    return os.path.join(os.path.expanduser("~"), ".claude/projects", project_key(root),
                        "memory")


def _materialize_data(src_dir: str, dst_dir: str) -> None:
    """Give the run root its own directory entry for every official file.

    NOT a symlink to the shared clean root: every entry under bench-comps/<comp>/data is
    itself a symlink back into kaggle/competitions/<comp>/data_official/, so a symlinked
    data/ resolves *into the repo* and an agent following its own data path lands one `..`
    away from the tree the run root exists to exclude (2026-08-10 round-9). Hardlinks cost
    no bytes on the same filesystem and cannot be followed anywhere: the file simply exists
    here too. Copy is the cross-device fallback.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        src = os.path.realpath(os.path.join(src_dir, name))
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            _materialize_data(src, dst)
            continue
        # The gate exempts competitions/<comp>/data/ on the grounds that only
        # manifest-verified official files get here. That promise is kept HERE: the source
        # is always an entry of bench-comps/<comp>/data, which build_isolated_roots.py fills
        # from Kaggle's own file manifest. Nothing else may be written into a data dir.
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _check_parent(root: str) -> list[str]:
    """The builder is the only thing that ever looks OUTSIDE the run root.

    verify_clean_slate.py is a single os.walk(root), so nothing above the root is examined,
    ever. That is why the default root sitting inside ~/ai_agents went unnoticed for two
    rounds while MASTER_TODO.md, one `..` away, carried the all-20 public/private table
    (round 11). A lane is separated from the parent directory by prose alone, and rounds 1-4
    established that prose is not a boundary.

    Files directly in the parent only: a recursive scan would walk sibling run roots, and
    those are checked by their own gate.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import verify_clean_slate as gate

    parent = os.path.dirname(os.path.abspath(root))
    slugs = gate.benchmark_slugs(REPO)
    out = []
    try:
        entries = sorted(os.listdir(parent))
    except OSError:
        return out
    for name in entries:
        full = os.path.join(parent, name)
        if not os.path.isfile(full) or os.path.islink(full):
            continue
        out += [f"../{f}" for f in gate.scan_file(full, name, slugs)]
    return out


def _write_baseline(root: str) -> None:
    """Record the frozen surface: every file the gate scans, by sha256.

    Two things follow from this file existing. The gate can tell what the builder produced
    from what the RUN produced, so a lane's own STATUS.md no longer fails it -- without which
    the run could never restart, since the launcher gates startup on the gate and orders each
    lane to write its final CV score there (round 11). And the guards the builder writes --
    the stripped submit routes, .claude/settings.json -- become checkable at START time,
    where the launcher actually looks, instead of only at build time, where it does not.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import verify_clean_slate as gate

    files = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = [d for d in dirnames if not gate.is_exempt(rel_dir + d + "/")]
        for fn in filenames:
            rel = rel_dir + fn
            if gate.is_exempt(rel):
                continue
            full = os.path.join(dirpath, fn)
            if os.path.islink(full) or not os.path.isfile(full):
                continue
            try:
                files[rel] = hashlib.sha256(open(full, "rb").read()).hexdigest()
            except OSError:
                continue
    bpath = os.path.join(root, gate.BASELINE_FILE)
    json.dump({"built_from": REPO, "files": files}, open(bpath, "w"), indent=2, sort_keys=True)
    # AND a sidecar beside the root, which the lane cannot see: the sandbox blanks the run
    # root's parent. Without it the manifest is its own only witness -- exempt from scanning,
    # absent from its own file list, and writable by every lane, so rewriting it (or just
    # emptying it to {"files":{}}) disabled every frozen-file check at once (round 12).
    digest = hashlib.sha256(open(bpath, "rb").read()).hexdigest()
    open(gate.sidecar_path(root), "w").write(digest + "\n")


def _drop_submit_routes(root: str) -> None:
    """Rewrite the copied skill's routes to the submit skill, which is not in the root.

    A lane that submits can read back `kaggle competitions submissions -c <comp>` — its own
    recorded public LB, and the other two agents' (round 10). The benchmark does not need it:
    the lane's contract ends at submission.csv, and scoring happens outside the run.
    """
    note = ("validate the CSV with `uv run python "
            ".claude/skills/kaggle-safe-submit/scripts/validate_submission.py` — this run "
            "does NOT submit to Kaggle and has no credentials; the operator scores "
            "submission.csv after the run")
    for rel in (".claude/skills/kaggle-agent/SKILL.md",
                ".claude/skills/kaggle-agent/references/06_submission.md"):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        out = []
        for line in open(p, encoding="utf-8").read().splitlines():
            low = line.lower()
            if "kaggle competitions submit" in low or "competitions submissions -c" in low \
                    or "kaggle_auth.sh" in low or "kaggle_api_token" in low:
                continue
            if "kaggle-safe-submit` skill" in line or "use `kaggle-safe-submit`" in line:
                line = note
            out.append(line)
        open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")


def plan(root: str) -> list[tuple[str, str]]:
    """[(source, destination)] — sources relative to REPO, destinations to root."""
    man = json.load(open(os.path.join(REPO, "docs/rerun_manifest.json")))["competitions"]
    out = []
    for d in DIRS:
        out.append((d, d))
    for f in FILES:
        out.append((f, f))
    for t in TREE_TOOLS:
        out.append((f"tree_search/{t}", f"tree_search/{t}"))
    for comp, spec in sorted(man.items()):
        out.append((f"tree_search/{spec['eval']}", f"tree_search/{spec['eval']}"))
        for f in WS_FILES:
            src = f"competitions/{comp}/{f}"
            if os.path.exists(os.path.join(REPO, src)):
                out.append((src, src))
    return out


RUN_ROOT_CLAUDE_MD = """# my-agent benchmark run root

This directory is the ONLY tree this run may read. It was built by allowlist
(benchmark_infra/build_myagent_run_root.py in the source repo): it holds the skill, the
knowledge tools, the search harness, each competition's pinned evaluator, and each
workspace's config plus the competition's official data files.

Nothing here records how any competition turned out before. That is the point of the root,
and it is the one thing you must not undo:

- **Do not read outside this directory.** Not the parent, not ~/ai_agents/*, not any other
  checkout of this project, not ~/.claude/. Nothing here needs anything out there: even the
  competition data files are present in this tree.
- **Do not consult git history** — there is no repository here, and there must not be one.
- **Follow the skill**: `.claude/skills/kaggle-agent/SKILL.md` and its references define the
  pipeline. Read the knowledge library only through `knowledge/query_library.py` and
  `knowledge/task_priors_for.py`; never open `knowledge/experience.md` or
  `knowledge/knowledge_base.json` directly.
- **Package management is uv** — `uv run python3 ...`, never pip.
- **This run does not submit to Kaggle and holds no credentials.** No network calls at all.
  Your contract ends at `competitions/<comp>/submission.csv`; the operator scores it
  afterwards. Do not look for a token, and do not try to read any leaderboard: your own
  competition's prior submissions are exactly the answer this root exists to withhold.
- Write experiments to `competitions/<comp>/experiments.json` and finish with
  `competitions/<comp>/submission.csv` plus a 3-line `STATUS.md` summary.
"""


def build(root: str, write: bool) -> int:
    parent_leaks = _check_parent(root)
    if parent_leaks:
        print(f"REFUSING: {os.path.dirname(os.path.abspath(root))} — the run root's own "
              f"parent directory states a benchmark competition's result, and a lane reaches "
              f"it with one `..`. The gate never looks outside --root.\n", file=sys.stderr)
        for f in parent_leaks[:10]:
            print("  " + f, file=sys.stderr)
        if len(parent_leaks) > 10:
            print(f"  ... {len(parent_leaks) - 10} more", file=sys.stderr)
        print(f"\nBuild into a directory whose parent holds nothing but run roots "
              f"(default: {DEFAULT_ROOT}).", file=sys.stderr)
        return 4
    steps = plan(root)
    if write:
        if os.path.exists(root):
            print(f"refusing to build over an existing {root} — remove it first, so a stale "
                  f"root can never masquerade as a fresh one", file=sys.stderr)
            return 4
        os.makedirs(root)
    n = 0
    for src, dst in steps:
        s, d = os.path.join(REPO, src), os.path.join(root, dst)
        if not os.path.exists(s):
            print(f"MISSING  {src}")
            continue
        if write:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks=False,
                                ignore=shutil.ignore_patterns(*DIR_EXCLUDE))
            else:
                shutil.copy2(s, d)
        else:
            print(f"COPY  {src}")
        n += 1

    man = json.load(open(os.path.join(REPO, "docs/rerun_manifest.json")))["competitions"]
    missing_data = []
    for comp in sorted(man):
        target = os.path.join(CLEAN_DATA, comp, "data")
        dst = os.path.join(root, "competitions", comp, "data")
        if not os.path.isdir(target):
            missing_data.append(comp)
            continue
        if write:
            _materialize_data(target, dst)
        else:
            n_esc = sum(1 for f in sorted(os.listdir(target))
                        if os.path.realpath(os.path.join(target, f)).startswith(REPO + os.sep))
            print(f"DATA  competitions/{comp}/data  <- {target} "
                  f"({len(os.listdir(target))} official files"
                  + (f", {n_esc} of which resolve back into the repo" if n_esc else "") + ")")
        n += 1
    if missing_data:
        print(f"\nNO CLEAN DATA ROOT for {len(missing_data)}: {missing_data}", file=sys.stderr)
        print("build them with benchmark_infra/build_isolated_roots.py first", file=sys.stderr)
        return 4

    if write:
        open(os.path.join(root, "CLAUDE.md"), "w").write(RUN_ROOT_CLAUDE_MD)
        _drop_submit_routes(root)
        # Auto-memory is not a static artifact you clear once before the run: the harness
        # WRITES to it during a session, and all 20 lanes share this one working directory,
        # so lane k's notes would be injected into lanes k+1..20 -- and a lane relaunched
        # after the 6 h cap or the stall watchdog would read its own previous attempt's CV
        # score and champion recipe (2026-08-10 round-10). Off for this project entirely.
        os.makedirs(os.path.join(root, ".claude"), exist_ok=True)
        json.dump({"autoMemoryEnabled": False},
                  open(os.path.join(root, ".claude/settings.json"), "w"), indent=2)
        _write_baseline(root)
    n += 1

    # The memory directory is derived from the working directory, so a fresh root gets a
    # fresh (absent) one. Check rather than assume: this is the channel that put 8 lanes'
    # CV scores into the session before its first tool call (round 9 #15).
    md = memory_dir(root)
    if os.path.exists(md):
        entries = [e for e in os.listdir(md) if not e.startswith(".")]
        if entries:
            print(f"\nREFUSING: {md} already exists and is non-empty ({entries[:4]}). The "
                  f"harness injects it into every session started from {root}. Move it "
                  f"aside before the run.", file=sys.stderr)
            return 4
    print(f"\nmemory directory for this root: {md} (absent — nothing auto-injected)")
    print(f"{n} item(s) {'copied' if write else 'would be copied'} -> {root}")
    if not write:
        print("\nNOT copied, and why:" + NOT_COPIED)
        print("(dry run; pass --write to build)")
    else:
        print("\nnext: python3 benchmark_infra/verify_clean_slate.py --root " + root)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    return build(os.path.abspath(args.root), args.write)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
