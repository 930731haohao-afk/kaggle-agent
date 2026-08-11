#!/bin/bash
# Benchmark watchdog — notices when the machine is idle while work is still queued.
#
# On 2026-07-24 a session died with RUN2 never started and the machine sat unused until
# 2026-07-27: roughly 72 hours lost, discovered only by being asked. Nothing was watching.
# A stalled scheduler cost another 139 minutes the same way. This runs from systemd, not
# from a session, so it survives exactly the failure it is meant to catch.
#
# It does not restart anything on its own — a benchmark run has fairness preconditions
# (exclusive machine, clean workspaces) that a script cannot verify. It notifies, loudly.
set -uo pipefail

RUNS=/home/tjyen/ai_agents/aideml-runs
LOG=/home/tjyen/ai_agents/bench_watchdog.log
STATE=/home/tjyen/ai_agents/.watchdog_idle_since
# The my-agent re-run happens in an isolated root outside the repo (round 9), and the
# launcher writes both its status file and every lane's output under it. Same default and
# same override as run_myagent_headless.sh -- a watchdog pointed anywhere else supervises a
# tree nothing writes to, which is the failure it exists to catch (round 11).
RUN_ROOT=${RUN_ROOT:-/home/tjyen/benchruns/myagent-rerun}

log(){ echo "$(date -Is) $*" >> "$LOG"; }

# Two independent signals, because either alone lies.
#
#   alive    — a real interpreter is running (not a wrapper shell that outlived its child;
#              that mistake cost 139 minutes on 2026-07-27)
#   advancing — some benchmark output was written recently
#
# Process existence alone cannot distinguish work from a hung run: a wedged process reads
# as busy forever and no alarm ever fires. Output freshness alone cannot distinguish a
# finished queue from a dead one. Requiring both catches the hang, which is the failure
# mode that most resembles "the machine is fine" while nothing is produced.
STALE_MIN=30

alive=0
while read -r pid _; do
  exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
  # `claude -p` is launched under `timeout`, so its /proc exe resolves to /usr/bin/timeout,
  # not to an interpreter — the exe check alone made every my-agent lane invisible here.
  case "$exe" in *python*|*node*|*timeout*) alive=1; break;; esac
done < <(pgrep -af "run_comp.py|reproduce.py|train.py|mlebench prepare|claude -p" 2>/dev/null | awk '{print $1, $2}')

# An ABSOLUTE timestamp, computed here. `find` on this machine is bfs, which rejects the
# relative form GNU find accepts -- `-newermt "-30 minutes"` exited with "Invalid timestamp",
# and with stderr on /dev/null that was indistinguishable from "nothing was written". So
# `advancing` was 0 on every tick, which with alive=1 is the critical HUNG branch: the
# watchdog cried wolf continuously while a run worked, and `exit 0`ed there, before the
# per-driver check that reports a driver that died (round 11).
SINCE=$(date -d "-${STALE_MIN} minutes" '+%F %T') || { log "cannot compute the staleness cutoff"; exit 0; }

advancing=0
fresh=$(find "$RUNS" /home/tjyen/ai_agents/nvidia-kaggle-runs "$RUN_ROOT/competitions" \
     -type f \( -name "journal.json" -o -name "*.log" -o -name "submission.csv" \
                -o -name "STATUS.md" -o -name "experiments_tree_v3.json" \) \
     -newermt "$SINCE" 2>&1 | head -1)
case "$fresh" in
  # A find that cannot run is not evidence of a stall. Say so instead of alarming.
  *"error"*|*"Invalid"*|*"unknown predicate"*) log "freshness probe failed: $fresh"; exit 0;;
  ?*) advancing=1;;
esac

if [ "$alive" = "1" ] && [ "$advancing" = "0" ]; then
  log "HUNG: interpreter alive but no benchmark output written in ${STALE_MIN}min"
  notify-send -u critical "Benchmark hung" \
    "A run is alive but has produced nothing for ${STALE_MIN}min" 2>/dev/null || true
  exit 0
fi

busy=$alive

# Per-driver check — the blind spot that let a whole lane sit out the night.
#
# Machine-level "alive + advancing" says nothing about *which* work is running. On
# 2026-07-28 the my-agent driver died at 17:58 and never ran a single competition, while AIDE
# lanes kept the machine busy all night; every check passed and no alarm fired. A driver that
# is supposed to be running and isn't is a silent failure exactly like a hang.
#
# Each entry is: <driver process pattern>|<status file>|<completion marker>
DRIVERS=(
  "rerun_contaminated4.py|$RUNS/RERUN4_STATUS.md|RE-RUN COMPLETE"
  "phase9a_aide_lanes.py|$RUNS/PHASE9A_STATUS.md|PHASE 9A AIDE LANES COMPLETE"
  "run_ready_reproductions.py|/home/tjyen/ai_agents/nvidia-kaggle-runs/RUN_READY_STATUS.md|NVIDIA LANES COMPLETE"
  "run_myagent_headless.sh|$RUN_ROOT/MYAGENT_LANES_STATUS.md|MY-AGENT LANES COMPLETE"
  "auto_submit_reruns.py|$RUNS/AUTO_SUBMIT.md|AUTO-SUBMIT COMPLETE"
  "run2_cells.py|$RUNS/RUN2_STATUS.md|RUN2 CELLS 2-9 COMPLETE"
)

dead=""
for entry in "${DRIVERS[@]}"; do
  pat=${entry%%|*}; rest=${entry#*|}; sf=${rest%%|*}; marker=${rest##*|}
  [ -f "$sf" ] || continue                                    # never started — not its turn yet
  grep -q "$marker" "$sf" 2>/dev/null && continue             # finished cleanly
  # Match the interpreter, not a wrapper shell whose command line merely mentions the script.
  running=0
  while read -r pid _; do
    exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    case "$exe" in *python*|*bash*) running=1; break;; esac
  done < <(pgrep -af "$pat" 2>/dev/null | awk '{print $1, $2}')
  [ "$running" = "0" ] && dead="$dead ${pat%%.*}"
done

if [ -n "$dead" ]; then
  log "DRIVER DEAD:$dead — status file has no completion marker and no process is running"
  notify-send -u critical "Benchmark driver died" "Not running, not finished:$dead" 2>/dev/null || true
fi

# Work is pending if any queue file still lists an unfinished item.
pending=""
[ -f "$RUNS/RERUN4_STATUS.md" ] && ! grep -q "RE-RUN COMPLETE" "$RUNS/RERUN4_STATUS.md" 2>/dev/null \
  && pending="$pending rerun4"
[ -f /home/tjyen/ai_agents/MASTER_TODO.md ] && grep -q "^| 7 |" /home/tjyen/ai_agents/MASTER_TODO.md 2>/dev/null \
  && pending="$pending RUN2"

if [ "$busy" = "1" ]; then
  rm -f "$STATE"
  exit 0
fi

if [ -z "$pending" ]; then
  rm -f "$STATE"
  log "idle, nothing pending — fine"
  exit 0
fi

now=$(date +%s)
[ -f "$STATE" ] || echo "$now" > "$STATE"
since=$(cat "$STATE")
mins=$(( (now - since) / 60 ))

log "IDLE ${mins}min with work pending:$pending"

# Escalate once past 30 minutes; a benchmark cell never has that much dead air mid-run.
if [ "$mins" -ge 30 ]; then
  notify-send -u critical "Benchmark idle ${mins}min" "Nothing running, pending:$pending" 2>/dev/null || true
  log "NOTIFIED (idle ${mins}min)"
fi
