"""Lane mutex for Python drivers — the same lock as lane_lock.sh, held by the right process.

Why this exists. `lane_lock.sh` is correct for a long-lived shell: it writes `$$` into the
lock and releases on that shell's EXIT trap. A Python driver that acquires it via
`subprocess.run(["bash", "-c", "source lane_lock.sh && lane_acquire ..."])` gets the opposite
of what it wants: the throwaway subshell writes ITS OWN pid, exits as soon as the command
finishes, and its EXIT trap fires `lane_release`, whose ownership test (`pid == $$`) passes
because the subshell is indeed the recorded owner. The lock is acquired and released in the
same second, and the driver then runs its entire competition unprotected.

That is not hypothetical. `lane_lock.log` recorded, on 2026-07-30:

    ACQUIRED by repeat-playground-series-s4e11 (pid 917918)
    RELEASED by repeat-playground-series-s4e11 (pid 917918)

both stamped 11:35:02, while pids 917917/917935 went on training for another 95 minutes with
`.lane_lock` absent. No overlap actually occurred, only because nothing else tried to start.

This module implements the same protocol in-process, so the owner pid is the driver that
actually holds the machine. It is deliberately wire-compatible with lane_lock.sh — same lock
directory, same `pid`/`owner` files, same log format — so shell drivers and Python drivers
mutually exclude each other rather than each respecting a private lock.

    from lane_lock import lane
    with lane("repeat-s4e11"):
        ...run one competition...          # released on exit, including on exception

Or, if a context manager does not fit the control flow:

    lane_acquire("repeat-s4e11"); try: ...; finally: lane_release()
"""
from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

LANE_LOCK_DIR = Path("/home/tjyen/ai_agents/.lane_lock")
LANE_LOCK_LOG = Path("/home/tjyen/ai_agents/lane_lock.log")

_POLL_SECONDS = 60
_DEFAULT_MAX_WAIT = 18 * 3600
_held_by_us = False


def _log(msg: str) -> None:
    # matches lane_lock.sh's `date -Is` format so both writers produce one readable log
    stamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    stamp = stamp[:-2] + ":" + stamp[-2:]
    with LANE_LOCK_LOG.open("a") as f:
        f.write(f"{stamp} {msg}\n")


def _read(name: str) -> str:
    try:
        return (LANE_LOCK_DIR / name).read_text().strip()
    except OSError:
        return ""


def _alive(pid: str) -> bool:
    """Is the recorded owner still running? Mirrors lane_lock.sh's `kill -0` reclamation."""
    if not pid.isdigit():
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True        # exists but owned by another user — treat as held, never steal


def lane_acquire(name: str, max_wait: int = _DEFAULT_MAX_WAIT) -> bool:
    """Block until the machine is free. Returns True if the lock is held, False on timeout.

    On timeout the caller proceeds WITHOUT the lock (matching lane_lock.sh), and the log says
    so explicitly — a contended run that is labelled contended is recoverable, a silent one is
    not.
    """
    global _held_by_us
    waited = 0
    while True:
        try:
            LANE_LOCK_DIR.mkdir()                  # atomic: the whole mutex rests on this
        except FileExistsError:
            pass
        else:
            (LANE_LOCK_DIR / "pid").write_text(f"{os.getpid()}\n")
            (LANE_LOCK_DIR / "owner").write_text(f"{name}\n")
            _held_by_us = True
            _log(f"ACQUIRED by {name} (pid {os.getpid()})")
            return True

        holder_pid, holder_name = _read("pid"), _read("owner") or "?"

        if holder_pid and not _alive(holder_pid):
            _log(f"STALE lock from {holder_name} (pid {holder_pid} gone) — reclaiming for {name}")
            shutil.rmtree(LANE_LOCK_DIR, ignore_errors=True)
            continue

        if waited >= max_wait:
            _log(f"TIMEOUT: {name} waited {waited // 60}min for {holder_name} — proceeding "
                 f"WITHOUT the lock, conditions are contended")
            return False

        if waited % 1800 == 0:
            _log(f"{name} waiting for {holder_name} ({waited // 60}min)")
        time.sleep(_POLL_SECONDS)
        waited += _POLL_SECONDS


def lane_release() -> None:
    """Release only if we actually own it — never delete another driver's lock."""
    global _held_by_us
    if not _held_by_us:
        return
    owner = _read("owner") or "?"
    if _read("pid") == str(os.getpid()):
        shutil.rmtree(LANE_LOCK_DIR, ignore_errors=True)
        _log(f"RELEASED by {owner} (pid {os.getpid()})")
    _held_by_us = False


@contextlib.contextmanager
def lane(name: str, max_wait: int = _DEFAULT_MAX_WAIT):
    """Preferred form: the lock cannot outlive the block, including on exception."""
    got = lane_acquire(name, max_wait)
    try:
        yield got
    finally:
        lane_release()


atexit.register(lane_release)      # backstop for drivers that exit without unwinding
