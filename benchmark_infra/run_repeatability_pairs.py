#!/usr/bin/env python3
"""Item 8 — AIDE between-run variance, measured as pairs against the existing baseline.

The paired significance test in the report covers test-set sampling noise but not AIDE's
run-to-run variance: its code is re-sampled from an LLM at every step, so two runs of the
same competition can follow different search paths. This measures that.

Design note — why one new run per competition, not two. The 07-23/24 batch runs already
satisfy the benchmark conditions exactly (20 steps, exec.timeout=1800, isolated root,
same shim), so each is a valid first replicate. Pairing a fresh run against it gives n=2
per competition for half the machine time. s3e11 and s3e14 already have two conforming
runs each and need nothing new.

Conditions are pinned explicitly here rather than inherited. The first attempt at this
experiment (chain_v5_repeat.sh, 2026-07-30) called run_comp.py from an inline shell loop
and omitted exec.timeout, so AIDE fell back to its 3600 s default and 4 of 20 steps ran
longer than the benchmark ever allowed. Nothing warned: the value is passed on the command
line by each launcher, and it is recorded only inside each run's own config.yaml, which no
tool reads. Run audit_aide_conditions.py after this finishes.

Usage:  setsid nohup python3 run_repeatability_pairs.py > repeatability_pairs.log 2>&1 < /dev/null &
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

HOME = Path.home()
AIDEML = HOME / "ai_agents/aideml"
RUNS = HOME / "ai_agents/aideml-runs"
BENCH = HOME / "ai_agents/bench-comps"
STATUS = RUNS / "REPEATABILITY_STATUS.md"

STEPS = 20
EXEC_TIMEOUT = 1800          # must match the benchmark; see the module docstring
COMP_TIMEOUT = 5 * 3600      # wall-clock safety net above the slowest baseline (s6e2, 192 min)

# Competitions needing one fresh run to pair against their 07-23/24 baseline. Ordered
# cheapest first so a partial night still yields complete pairs.
COMPS = [
    "playground-series-s4e11",   # baseline 24.9 min
    "playground-series-s3e16",   # baseline 44.1 min
    "playground-series-s5e10",   # baseline 170.1 min
    "playground-series-s6e2",    # baseline 192.1 min
]


def log(msg: str) -> None:
    line = f"- `{datetime.now():%m-%d %H:%M}` {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def scalar(text: str, key: str):
    m = re.search(rf"^\s+{key}:\s*(\S+)", text, re.M)
    return m.group(1) if m else None


def conforming_runs(comp: str):
    """Every run of this competition that matches the benchmark conditions, with its best."""
    out = []
    logs = RUNS / comp / "logs"
    if not logs.is_dir():
        return out
    for d in sorted(logs.iterdir()):
        cfg, jrn = d / "config.yaml", d / "journal.json"
        if not (cfg.is_file() and jrn.is_file()):
            continue
        t = cfg.read_text()
        if (scalar(t, "timeout"), scalar(t, "steps")) != (str(EXEC_TIMEOUT), str(STEPS)):
            continue
        try:
            data = json.loads(jrn.read_text())
        except json.JSONDecodeError:
            continue
        nodes = data["nodes"] if isinstance(data, dict) else data
        vals = [n["metric"]["value"] for n in nodes
                if not n.get("is_buggy")
                and isinstance((n.get("metric") or {}).get("value"), (int, float))]
        if not vals:
            continue
        maximize = (nodes[0].get("metric") or {}).get("maximize")
        out.append((d.name, max(vals) if maximize else min(vals), len(nodes)))
    return out


def main() -> None:
    if not STATUS.exists():
        STATUS.write_text(
            "# Item 8 — AIDE repeatability, paired against the existing baseline\n\n"
            f"One fresh run per competition at {STEPS} steps, exec.timeout={EXEC_TIMEOUT}, "
            "isolated root `bench-comps/`. Each pairs with its 07-23/24 baseline run, which "
            "already meets the same conditions.\n\n")
    log(f"queue: {', '.join(c.replace('playground-series-', '') for c in COMPS)}")

    import sys
    sys.path.insert(0, str(HOME / "ai_agents"))
    from lane_lock import lane_acquire, lane_release   # noqa: E402 -- needs the path above

    for i, comp in enumerate(COMPS, 1):
        before = {r[0] for r in conforming_runs(comp)}
        log(f"[{i}/{len(COMPS)}] START {comp} (existing conforming runs: {len(before)})")
        started = time.time()

        # Held in THIS process, not a subshell. The previous form acquired inside a throwaway
        # `bash -c`, whose EXIT trap fired lane_release the moment it printed OK -- so the lock
        # was taken and dropped in the same second and every competition below ran unprotected
        # (lane_lock.log, 2026-07-30 11:35:02, both lines same timestamp). See lane_lock.py.
        if not lane_acquire(f"repeat-{comp}"):
            log(f"[{i}/{len(COMPS)}] WARNING: proceeding without the lane lock (timed out "
                f"waiting); this run's conditions are contended and must be labelled as such")

        env = dict(os.environ, AIDE_COMP_ROOT=str(BENCH), AIDE_EXEC_MEM_LIMIT_GB="32")
        cmd = [str(AIDEML / ".venv/bin/python"), str(AIDEML / "run_comp.py"),
               comp, "--shim", "--steps", str(STEPS), f"exec.timeout={EXEC_TIMEOUT}"]
        with (RUNS / f"{comp}_repeat.log").open("w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, cwd=str(AIDEML),
                                    env=env, start_new_session=True)
            try:
                rc = proc.wait(timeout=COMP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = "TIMEOUT"

        lane_release()

        mins = (time.time() - started) / 60
        runs = conforming_runs(comp)
        new = [r for r in runs if r[0] not in before]
        detail = (f"new run best={new[0][1]} ({new[0][2]} nodes)" if new else "NO conforming new run")
        log(f"[{i}/{len(COMPS)}] DONE {comp}: {detail} ({mins:.0f} min, rc={rc}); "
            f"pair now n={len(runs)}")

    # Summarise every competition that ended up with a usable pair.
    log("--- paired summary ---")
    for comp in COMPS + ["playground-series-s3e11", "playground-series-s3e14"]:
        runs = conforming_runs(comp)
        if len(runs) < 2:
            log(f"{comp}: only {len(runs)} conforming run — no pair")
            continue
        bests = [r[1] for r in runs]
        spread = max(bests) - min(bests)
        log(f"{comp}: n={len(runs)} bests={bests} spread={spread:.6g}")

    log("REPEATABILITY PAIRS COMPLETE")


if __name__ == "__main__":
    main()
