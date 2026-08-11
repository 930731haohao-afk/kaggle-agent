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


def _const_strings(tree, here: str) -> dict:
    """Fold module-level string assignments, including os.path.join of resolvable parts.

    Enough to resolve CACHE_DIR without importing the evaluator -- importing one costs a full
    feature build, and this runs between every pair of lanes.
    """
    import ast
    env = {"_HERE": here}

    def val(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "attr", getattr(f, "id", ""))
            if name == "join":
                parts = [val(a) for a in node.args]
                if all(p is not None for p in parts):
                    return os.path.join(*parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = val(node.left), val(node.right)
            if a is not None and b is not None:
                return a + b
        return None

    for n in tree.body:
        if isinstance(n, ast.Assign):
            v = val(n.value)
            if v is None:
                continue
            for t in n.targets:
                if isinstance(t, ast.Name):
                    env[t.id] = v
    return env


def cache_dirs_for(repo_or_root: str, comp: str) -> list[str]:
    """Where this competition's OOF cache actually lives, per its evaluator's own source.

    NOT a naming heuristic on the slug. The previous version derived a prefix from the
    competition name, and the evaluators simply do not name CACHE_DIR that way:
    cat-in-the-dat writes cache_citd, aug-2022 cache_aug22, jan-2022 cache_tpsjan22, sep-2022
    cache_tssep22_main, and s3e5 nests its under cache_s3e5/v2 through a variable. Four lanes
    were therefore left with the previous attempt's vectors -- the exact failure this module
    exists to prevent -- and four unrelated competitions had their caches moved instead.
    """
    import ast
    man_path = os.path.join(repo_or_root, "docs/rerun_manifest.json")
    if not os.path.exists(man_path):
        return []
    entry = json.load(open(man_path))["competitions"].get(comp) or {}
    mod = None
    for k in ("eval_module", "evaluator", "eval"):
        if isinstance(entry, dict) and entry.get(k):
            mod = str(entry[k]).replace(".py", "")
            break
    if not mod:
        return []
    src_path = os.path.join(repo_or_root, "tree_search", f"{mod}.py")
    if not os.path.exists(src_path):
        return []
    here = os.path.join(repo_or_root, "tree_search")
    env = _const_strings(ast.parse(open(src_path, encoding="utf-8").read()), here)
    out = []
    for name, v in env.items():
        if name.endswith("CACHE_DIR") and isinstance(v, str) and "cache" in v:
            rel = os.path.relpath(v, repo_or_root) if os.path.isabs(v) else v
            out.append(rel.replace(os.sep, "/"))
    # Collapse ancestor/descendant pairs. eval_s3e5_v2.py declares BOTH
    # V1_CACHE_DIR = cache_s3e5 and CACHE_DIR = cache_s3e5/v2, and both names end in
    # CACHE_DIR, so this returned the pair -- partial_paths then walked the parent (which
    # already descends into v2/) and walked v2/ again, listing every file under it twice.
    # shutil.move succeeded on the first copy and raised FileNotFoundError on the second,
    # which the launcher turns into "REFUSING to start playground-series-s3e5 ... continue".
    # Keeping the ancestor is what subsumes the descendant; dropping it would leave v1's
    # directory unswept. Both names come from THIS competition's own evaluator, so there is
    # no other lane's cache to protect here.
    dirs = sorted(set(out))
    return [d for d in dirs
            if not any(d != o and d.startswith(o.rstrip("/") + "/") for o in dirs)]


def partial_paths(root: str, comp: str) -> list[str]:
    """Run-produced paths under competitions/<comp>/, plus its OOF caches."""
    baseline = gate.load_baseline(root) or {}
    out = []
    base = os.path.join(root, "competitions", comp)
    # Whether data/ can be swept by MEMBERSHIP rather than by name. Pruning the directory
    # wholesale left the aborted attempt's Stage-2 tables in place -- train_processed.csv,
    # test_processed.csv, level_table.csv, holidays.csv -- and four pinned evaluators read
    # exactly those paths. The safe test is the gate's own: a file is official iff its name
    # is in this competition's Kaggle-manifest-verified clean root. If that root is not on
    # this machine we cannot tell official from run-produced, and moving the official data
    # would break the lane this module exists to protect, so fall back to the name prune.
    clean = os.path.join(gate.CLEAN_DATA_ROOT, comp, "data")
    by_membership = os.path.isdir(clean)
    if os.path.isdir(base):
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
            if rel_dir == f"competitions/{comp}" and not by_membership:
                dirnames[:] = [d for d in dirnames if d != "data"]  # official Kaggle files
            for fn in filenames:
                rel = f"{rel_dir}/{fn}"
                if rel in baseline or gate._is_official_file(rel):
                    continue
                out.append(rel)
    for cache in cache_dirs_for(root, comp):
        full = os.path.join(root, cache)
        for dirpath, _dirnames, filenames in os.walk(full):
            rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
            for fn in filenames:
                rel = f"{rel_dir}/{fn}"
                if rel not in baseline:
                    out.append(rel)
    return sorted(out)


def quarantine(root: str, comp: str, dest: str, *, dry_run: bool = False) -> int:
    # A cache that cannot be located is a cache that will not be cleared, and the lane then
    # blends on its previous attempt's vectors with nothing to say so. Refuse loudly instead:
    # silently missing four of twenty is precisely what the name heuristic did.
    man = os.path.join(root, "docs/rerun_manifest.json")
    if os.path.exists(man) and comp in json.load(open(man)).get("competitions", {}):
        if not cache_dirs_for(root, comp):
            print(f"REFUSING: cannot resolve the OOF cache directory for {comp} from its "
                  f"pinned evaluator. A cache left in place lets this lane blend on its "
                  f"previous attempt's vectors -- same node ids, different configs.",
                  file=sys.stderr)
            return 4
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
        prefix = f"competitions/{comp}/"
        sub = rel[len(prefix):] if rel.startswith(prefix) else os.path.join("_outside", rel)
        dst = os.path.join(target, sub)
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
