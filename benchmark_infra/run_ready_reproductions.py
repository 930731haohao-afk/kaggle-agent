#!/usr/bin/env python3
"""Execute NVIDIA-lane reproduction scripts that have been authored and smoke-tested.

Splits the lane into the part that needs judgement and the part that does not. Choosing a
public kernel and porting it is authoring work — an agent does that, once. Running the port
is mechanical, so it belongs in a script that survives the session ending.

Only scripts with a `READY` marker beside them are run: the marker means the author smoke-
tested that script end to end. Anything half-written is skipped rather than burning a slot.

Runs sequentially, after the AIDE lanes, to keep the machine's load comparable to how the
original 17-competition benchmark was run.

Usage:  setsid nohup python run_ready_reproductions.py > run_ready.log 2>&1 < /dev/null &
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

NV = Path.home() / "ai_agents/nvidia-kaggle-runs"
PY = Path.home() / "ai_agents/kaggle/.venv/bin/python"
STATUS = NV / "RUN_READY_STATUS.md"

COMPS = [
    "afsis-soil-properties",
    "cat-in-the-dat",
    "conway-s-reverse-game-of-life",
    "tabular-playground-series-aug-2022",
    "tabular-playground-series-jan-2022",
]
COMP_TIMEOUT = 4 * 3600  # safety net above the observed max (~2.5 h), not a budget
MAX_WAIT = 14 * 3600


def log(msg: str) -> None:
    line = f"- `{datetime.now():%m-%d %H:%M}` {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def aide_running() -> bool:
    """A real AIDE interpreter, not the wrapper shell that launched it."""
    out = subprocess.run(["pgrep", "-af", "run_comp.py"], capture_output=True, text=True, check=False)
    for line in out.stdout.splitlines():
        pid, _, _ = line.partition(" ")
        try:
            exe = os.path.realpath(f"/proc/{pid}/exe")
        except OSError:
            continue
        if "python" in os.path.basename(exe):
            return True
    return False


def main() -> None:
    STATUS.write_text("# NVIDIA lanes — executing smoke-tested reproductions\n\n"
                      "Only scripts with a `READY` marker run; the marker means the authoring "
                      "agent smoke-tested that script.\n\n")

    waited = 0
    while aide_running() and waited < MAX_WAIT:
        if waited % 1800 == 0:
            log(f"waiting for the AIDE lanes to finish ({waited // 60} min)")
        time.sleep(120)
        waited += 120
    if waited >= MAX_WAIT:
        log("WARNING: AIDE lanes still running after the cap — starting anyway, contended")

    ran = skipped = 0
    for comp in COMPS:
        script = NV / comp / "reproduce.py"
        ready = NV / comp / "READY"
        if not script.is_file() or not ready.exists():
            log(f"{comp}: SKIP — {'no reproduce.py' if not script.is_file() else 'not smoke-tested'}")
            skipped += 1
            continue

        log(f"START {comp}")
        started = time.time()
        env = dict(os.environ, OMP_NUM_THREADS="10", MKL_NUM_THREADS="10")
        with (NV / comp / "run.log").open("w") as lf:
            proc = subprocess.Popen([str(PY), "reproduce.py"], cwd=str(NV / comp),
                                    stdout=lf, stderr=lf, env=env, start_new_session=True)
            try:
                rc = proc.wait(timeout=COMP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = "TIMEOUT"

        mins = (time.time() - started) / 60
        sub = NV / comp / "submission.csv"
        state = f"submission written ({sub.stat().st_size} bytes)" if sub.is_file() else "NO SUBMISSION"
        log(f"DONE {comp}: {state} ({mins:.1f} min, rc={rc})")
        ran += 1

    log(f"NVIDIA LANES COMPLETE — {ran} run, {skipped} skipped. "
        "Submission to Kaggle and folding into the score table still pending.")


if __name__ == "__main__":
    main()
