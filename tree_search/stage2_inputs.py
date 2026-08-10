"""What a pinned evaluator needs from THIS run's Stage 2, and what to say when it is absent.

Five of the manifest-pinned evaluators do not build their own features: they read a table
the run's Stage 2 wrote into `competitions/<comp>/data/`. Before the 2026-08-10 round-8
clean-slate fix those tables were the RECORDED run's, left in place and silently consumed —
a fresh lane scored on the previous run's feature engineering while believing it had built
its own. The archive now removes them, which turns the dependency into a bare
`FileNotFoundError` from inside pandas at import time.

An import-time crash naming a path is not an instruction. This module makes the evaluator
say what artifact is missing, which columns it must carry, which stage produces it, and why
the file is not simply sitting there.
"""
from __future__ import annotations

import os


def require(path: str, *, comp: str, artifact: str, columns: list[str],
            produced_by: str = "Stage 2 feature engineering") -> str:
    """Return `path`, or raise with an actionable message if it is missing."""
    if os.path.exists(path):
        return path
    raise RuntimeError(
        f"{comp}: the pinned evaluator needs `{artifact}`, which is not present at\n"
        f"    {path}\n"
        f"It is produced by {produced_by} — it is NOT a competition input, which is why the "
        f"clean slate removed the copy left behind by an earlier run "
        f"(benchmark_infra/archive_workspaces_for_rerun.sh; 2026-08-10 round-8). Write this "
        f"run's own table there, carrying at least: {', '.join(columns)}.\n"
        f"Do not restore the archived copy: scoring this run on a previous run's feature "
        f"matrix is the contamination the archive exists to prevent.")


def require_module(scripts_dir: str, module: str, *, comp: str, exposes: list[str]) -> None:
    """Check that THIS run's Stage 2 wrote `scripts/<module>.py`, before importing it.

    Sixteen of the twenty pinned evaluators do `sys.path.insert(scripts_dir)` then
    `import features` — feature code the run itself is supposed to have written. From an
    isolated run root that import raises a bare `ModuleNotFoundError: No module named
    'features'`, which tells a lane nothing about what it is expected to produce
    (2026-08-10 round-9). docs/rerun_manifest.json's `stage2_module_contract` records the
    module and the signature of every callable; this makes the failure point at it.
    """
    if os.path.exists(os.path.join(scripts_dir, module + ".py")):
        return
    raise RuntimeError(
        f"{comp}: the pinned evaluator imports `{module}` from\n"
        f"    {scripts_dir}\n"
        f"which is THIS run's own Stage 2 feature code — not a competition input, and not "
        f"something to copy from a previous run. Write {module}.py there exposing: "
        f"{', '.join(exposes)}.\n"
        f"The full signature contract is docs/rerun_manifest.json -> competitions[{comp!r}]"
        f".stage2_module_contract.")
