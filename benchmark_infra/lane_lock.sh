#!/bin/bash
# Lane mutex — makes "one lane at a time" a mechanism instead of a convention.
#
# The rule was only written down as "RUN2 needs an exclusive machine", so nothing stopped
# the my-agent driver from starting while the AIDE lanes were still running on 2026-07-27.
# AIDE had run afsis alone in 70.9 min; my-agent began the same competition at load average
# 17. Same competition, unequal conditions — the defect that voided RUN1, repeated because
# correctness depended on every driver's wait condition being right.
#
# Any driver that runs benchmark work sources this file and calls `lane_acquire <name>`.
# The lock is a directory (atomic to create) holding the owner's PID, so a driver that dies
# does not wedge the queue: a stale lock whose PID is gone is reclaimed automatically.
#
#   source /home/tjyen/ai_agents/lane_lock.sh
#   lane_acquire "phase9a-aide"     # blocks until the machine is free
#   ...run one competition...
#   lane_release
#
# Deliberately not a queue: order comes from the drivers' own upstream markers. This only
# guarantees that two lanes never overlap.

LANE_LOCK_DIR=/home/tjyen/ai_agents/.lane_lock
LANE_LOCK_LOG=/home/tjyen/ai_agents/lane_lock.log

_lane_log(){ echo "$(date -Is) $*" >> "$LANE_LOCK_LOG"; }

lane_acquire() {
  local me="${1:-unknown}" waited=0 max="${2:-$((18*3600))}"
  while :; do
    if mkdir "$LANE_LOCK_DIR" 2>/dev/null; then
      echo "$$" > "$LANE_LOCK_DIR/pid"
      echo "$me" > "$LANE_LOCK_DIR/owner"
      _lane_log "ACQUIRED by $me (pid $$)"
      return 0
    fi

    local holder_pid holder_name
    holder_pid=$(cat "$LANE_LOCK_DIR/pid" 2>/dev/null || echo "")
    holder_name=$(cat "$LANE_LOCK_DIR/owner" 2>/dev/null || echo "?")

    # Reclaim a lock whose owner is gone — otherwise one crashed driver stops every lane.
    if [ -n "$holder_pid" ] && ! kill -0 "$holder_pid" 2>/dev/null; then
      _lane_log "STALE lock from $holder_name (pid $holder_pid gone) — reclaiming for $me"
      rm -rf "$LANE_LOCK_DIR"
      continue
    fi

    if [ "$waited" -ge "$max" ]; then
      _lane_log "TIMEOUT: $me waited $((waited/60))min for $holder_name — proceeding WITHOUT the lock, conditions are contended"
      return 1
    fi
    [ $((waited % 1800)) -eq 0 ] && _lane_log "$me waiting for $holder_name ($((waited/60))min)"
    sleep 60; waited=$((waited+60))
  done
}

lane_release() {
  local owner
  owner=$(cat "$LANE_LOCK_DIR/owner" 2>/dev/null || echo "?")
  if [ "$(cat "$LANE_LOCK_DIR/pid" 2>/dev/null)" = "$$" ]; then
    rm -rf "$LANE_LOCK_DIR"
    _lane_log "RELEASED by $owner (pid $$)"
  fi
}

# Release on exit however the driver ends — a lock outliving its owner is the failure this
# is meant to prevent.
trap lane_release EXIT INT TERM
