#!/usr/bin/env python3
"""Move an aborted attempt's artifacts out of the run root before the lane restarts.

WHY THIS EXISTS. run_myagent_headless.sh skips a competition whose submission.csv already
exists, so the only lanes it ever STARTS are ones that did not finish -- killed by the 6 h
cap, by the stall watchdog, by a reboot, or by the operator. What such a lane finds waiting
in competitions/<comp>/ is its own aborted attempt: a STATUS.md whose first three lines are
"the final CV score", an experiments_tree_v3.json holding every node it evaluated, and the
scripts that produced them. Reading them is a warm start from its own previous answer.

Round 10 closed exactly this channel for the harness's auto-memory ("a lane relaunched after
the 6 h cap or the stall watchdog would read its own previous attempt's CV score and champion
recipe"). The same thing arrives by a second road, through the filesystem, and closing one
road is not closing the channel -- which has been this audit's recurring lesson.

WHAT IS MOVED. Everything under competitions/<comp>/ that the BUILDER did not put there, as
recorded in the run root's .rerun_baseline.json, except data/. The builder's files -- the
config, the rules verdict, the official data -- are what the lane needs to start at all;
moving them would break the lane this is meant to protect. Nothing is deleted: the attempt is
moved intact to --dest, outside the run root, so it stays available to the operator for the
report and stays unreadable to the lane.

Usage:
  python3 quarantine_partial_attempt.py --root <run-root> --comp <slug> --dest <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_clean_slate as gate  # noqa: E402


def _cache_dirs(root: str, comp: str) -> list[str]:
    """The competition's OOF caches, which live OUTSIDE competitions/<comp>/.

    harness_v2.cache_oof keys an entry by NODE ID alone, and a relaunch starts node ids at 0
    again because the tree is quarantined with everything else. So attempt 2's node 5 loads
    attempt 1's node 5 vectors -- a different config, same key -- and blends on them.

    The guards built for exactly this are inert on both sides: nothing passes `config=` to
    cache_oof, so no identity is ever stamped, and load_oof only compares the hash when one
    is present, so a MISSING stamp skips the check instead of failing it. A guard that
    defaults to trusting is not a guard, so the cache is moved instead (round 11 / stages).
    """
    out = []
    ts = os.path.join(root, "tree_search")
    if not os.path.isdir(ts):
        return out
    short = comp.replace("playground-series-", "").replace("tabular-playground-series-", "")
    for name in sorted(os.listdir(ts)):
        if not name.startswith("cache_"):
            continue
        tail = name[len("cache_"):]
        if tail.startswith(short) or tail.startswith(comp) or short.startswith(tail):
            out.append(f"tree_search/{name}")
    return out


def partial_paths(root: str, comp: str) -> list[str]:
    """Run-produced paths under competitions/<comp>/, plus its OOF caches."""
    baseline = gate.load_baseline(root) or {}
    out = []
    base = os.path.join(root, "competitions", comp)
    if os.path.isdir(base):
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
            if rel_dir == f"competitions/{comp}":
                dirnames[:] = [d for d in dirnames if d != "data"]  # official Kaggle files
            for fn in filenames:
                rel = f"{rel_dir}/{fn}"
                if rel not in baseline:
                    out.append(rel)
    for cache in _cache_dirs(root, comp):
        full = os.path.join(root, cache)
        for dirpath, _dirnames, filenames in os.walk(full):
            rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
            for fn in filenames:
                rel = f"{rel_dir}/{fn}"
                if rel not in baseline:
                    out.append(rel)
    return sorted(out)


def quarantine(root: str, comp: str, dest: str, *, dry_run: bool = False) -> int:
    rels = partial_paths(root, comp)
    if not rels:
        print(f"no partial attempt for {comp} — nothing to move")
        return 0
    if not gate.load_baseline(root):
        # Without a baseline every file looks run-produced, and this would move the config
        # the lane needs. Refuse rather than guess.
        print(f"REFUSING: {root} has no {gate.BASELINE_FILE}, so the builder's files cannot "
              f"be told from the run's. Rebuild the root with build_myagent_run_root.py.",
              file=sys.stderr)
        return 4
    target = os.path.join(dest, comp)
    print(f"{'would move' if dry_run else 'moving'} {len(rels)} path(s) -> {target}")
    for rel in rels:
        print("  " + rel)
        if dry_run:
            continue
        dst = os.path.join(target, os.path.relpath(rel, f"competitions/{comp}"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(root, rel), dst)
    if not dry_run:
        # leave the tree tidy: drop directories the move emptied, never the ones with files
        for dirpath, dirnames, filenames in os.walk(
                os.path.join(root, "competitions", comp), topdown=False):
            if dirpath.endswith("/data") or "/data/" in dirpath:
                continue
            if not dirnames and not filenames and os.path.basename(dirpath) != comp:
                os.rmdir(dirpath)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="the run root")
    ap.add_argument("--comp", required=True, help="competition slug")
    ap.add_argument("--dest", required=True, help="where to move it (OUTSIDE the run root)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root, dest = os.path.abspath(a.root), os.path.abspath(a.dest)
    if dest == root or dest.startswith(root + os.sep):
        print(f"REFUSING: --dest {dest} is inside the run root; the lane would still read it",
              file=sys.stderr)
        return 4
    return quarantine(root, a.comp, dest, dry_run=a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
