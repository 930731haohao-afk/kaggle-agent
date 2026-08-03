#!/usr/bin/env python3
"""Re-run the July-subset competitions with AIDE's default budget (20 steps,
default 5 drafts). Results go to BATCH20_STATUS.md / batch20_results.json;
per-run scores are taken from each comp's NEWEST journal so the 20-step run is
reported separately from the earlier 5-step one.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_export import export_comp  # noqa: E402
from run_comp import load_comp_cfg  # noqa: E402

AIDEML = Path.home() / "ai_agents/aideml"
COMP_ROOT = Path.home() / "ai_agents/kaggle/competitions"
RUNS_ROOT = Path.home() / "ai_agents/aideml-runs"
STATUS_MD = RUNS_ROOT / "BATCH20_STATUS.md"
RESULTS_JSON = RUNS_ROOT / "batch20_results.json"

STEPS = 20
EXEC_TIMEOUT = 1800
COMP_TIMEOUT = 4 * 3600
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from lane_lock import lane  # noqa: E402


def node_count(journal_path: Path) -> int:
    try:
        return len(json.loads(journal_path.read_text()).get("nodes", []))
    except Exception:
        return 0


def seed_journal(comp: str) -> Path | None:
    """Newest journal for this comp — a crashed 20-step run's partial journal
    (continue where it stopped) or last night's 5-step one."""
    journals = sorted((RUNS_ROOT / comp / "logs").glob("*/journal.json"),
                      key=lambda p: p.stat().st_mtime)
    return journals[-1] if journals else None

COMPS = [
    "playground-series-s3e1", "playground-series-s3e3", "playground-series-s3e5",
    "playground-series-s3e7", "playground-series-s3e9", "playground-series-s3e11",
    "playground-series-s3e14", "playground-series-s3e16", "playground-series-s3e19",
    "playground-series-s3e20", "playground-series-s4e11", "playground-series-s5e10",
    "playground-series-s6e1", "playground-series-s6e2", "playground-series-s6e7",
    "spaceship-titanic", "home-data-for-ml-course",
]


def log(line: str) -> None:
    with STATUS_MD.open("a") as f:
        f.write(f"- `{time.strftime('%m-%d %H:%M')}` {line}\n")
    print(line, flush=True)


def ensure_shim() -> None:
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/v1/models", timeout=5)
        return
    except Exception:
        pass
    log("shim down — restarting")
    subprocess.Popen(
        [sys.executable, str(AIDEML / "claude_shim.py"), "--port", "8765"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(2)
    urllib.request.urlopen("http://127.0.0.1:8765/v1/models", timeout=5)


def newest_run_result(comp: str, minimize: bool) -> dict:
    journals = sorted((RUNS_ROOT / comp / "logs").glob("*/journal.json"),
                      key=lambda p: p.stat().st_mtime)
    if not journals:
        return {"best": None, "good_nodes": 0, "total_nodes": 0}
    nodes = json.loads(journals[-1].read_text()).get("nodes", [])
    vals = []
    for n in nodes:
        m = n.get("metric")
        v = m.get("value") if isinstance(m, dict) else m
        if not n.get("is_buggy") and isinstance(v, (int, float)):
            vals.append(v)
    best = (min(vals) if minimize else max(vals)) if vals else None
    return {"best": best, "good_nodes": len(vals), "total_nodes": len(nodes),
            "run": journals[-1].parent.name}


def main() -> None:
    # Hold the lane mutex for the batch: without it an AIDE re-run batch could overlap
    # my-agent's or NVIDIA's lane, which is the contention that voided RUN1 (2026-08-03).
    with lane("aide-rerun-20steps"):
        _main_locked()


def _main_locked() -> None:
    STATUS_MD.write_text(
        f"# AIDE 20-step re-run — started {time.strftime('%Y-%m-%d %H:%M')}\n\n")
    t0 = time.time()
    log(f"queue: {len(COMPS)} competitions — {', '.join(COMPS)}")
    results = []
    for i, comp in enumerate(COMPS, 1):
        ensure_shim()
        cfg = load_comp_cfg(COMP_ROOT / comp / "config.yaml")
        minimize = str(cfg.get("optimization_direction", "")).lower().startswith(
            ("min", "low"))
        seed = seed_journal(comp)
        if seed and node_count(seed) >= STEPS:
            res = {"comp": comp, "metric": cfg.get("evaluation_metric"),
                   "direction": "min" if minimize else "max", "rc": 0,
                   "minutes": None, "note": "already complete",
                   **newest_run_result(comp, minimize)}
            results.append(res)
            RESULTS_JSON.write_text(json.dumps(results, indent=2))
            log(f"[{i}/{len(COMPS)}] SKIP {comp}: already has "
                f"{res['total_nodes']} nodes (best={res['best']})")
            continue
        log(f"[{i}/{len(COMPS)}] START {comp}"
            + (f" (resume from {seed.parent.name}, {node_count(seed)} nodes)"
               if seed else " (fresh)"))
        logfile = RUNS_ROOT / comp / "batch20.log"
        cmd = [str(AIDEML / ".venv/bin/python"), str(AIDEML / "run_comp.py"),
               comp, "--shim", "--steps", str(STEPS), f"exec.timeout={EXEC_TIMEOUT}"]
        if seed:
            cmd.append(f"initial_journal={seed}")
        tc = time.time()
        with logfile.open("w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, start_new_session=True)
            try:
                rc = proc.wait(timeout=COMP_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                rc = -9
        res = {"comp": comp, "metric": cfg.get("evaluation_metric"),
               "direction": "min" if minimize else "max", "rc": rc,
               "minutes": round((time.time() - tc) / 60, 1),
               **newest_run_result(comp, minimize)}
        try:
            export_comp(comp)
        except Exception as e:
            log(f"export failed for {comp}: {e!r}")
        results.append(res)
        RESULTS_JSON.write_text(json.dumps(results, indent=2))
        log(f"[{i}/{len(COMPS)}] DONE {comp}: best={res['best']} "
            f"({res['good_nodes']}/{res['total_nodes']} ok, {res['minutes']} min, rc={rc})")
    log(f"20-STEP RE-RUN COMPLETE — {len(results)} comps in "
        f"{(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
