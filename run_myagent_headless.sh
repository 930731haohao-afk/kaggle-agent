#!/bin/bash
# my-agent lanes, unattended — headless Claude Code sessions, one competition at a time.
#
# The earlier claim that this lane "cannot run unattended" was wrong, and AIDE is the
# counterexample: an LLM makes every decision there too, it just does so from a detached
# process. `claude -p` gives my-agent the same property — the full skill-driven loop
# (EDA -> features -> modeling -> evaluation -> iterate) runs inside one headless session
# per competition. What a session loses by having no human is recoverable: the watchdog
# flags stalls, and a dead session leaves its workspace for the next one to pick up.
#
# THE RUN DOES NOT HAPPEN IN THE REPO. RUN_ROOT is an isolated tree built by allowlist
# (benchmark_infra/build_myagent_run_root.py --write) holding the skill, the knowledge tools,
# the harness, each competition's pinned evaluator, and each workspace's config.yaml plus a
# data/ symlink to the Kaggle-manifest-verified bench-comps root. Nine audit rounds tried to
# reach a clean slate by deleting things from the repo; round 9 showed the boundary was wrong.
# The harness derives its memory directory from the working directory and auto-injects it, and
# the repo's own key carries a MEMORY.md naming eight lanes' CV scores; `git show` retrieves
# every archived file; documents/, mlflow.db and the repo's parent all hold per-lane private
# scores. A fresh working directory has none of that, because none of it was ever copied in.
#
# Usage:  setsid nohup bash run_myagent_headless.sh > myagent_lanes.log 2>&1 < /dev/null &
set -uo pipefail

REPO=/home/tjyen/ai_agents/kaggle
# NOT under ~/ai_agents: MASTER_TODO.md and LANE_CLAIMS.md there carry the all-20
# public/private table, one `..` from where this used to default (round 11).
RUN_ROOT=${RUN_ROOT:-/home/tjyen/benchruns/myagent-rerun}
BASE=$RUN_ROOT
NV_MARKER=/home/tjyen/ai_agents/nvidia-kaggle-runs/RUN_READY_STATUS.md
AIDE_MARKER=/home/tjyen/ai_agents/aideml-runs/PHASE9A_STATUS.md
# OUTSIDE the run root, like $ATTEMPTS and $TRANSCRIPTS, and for the same reason.
#
# It used to live at $RUN_ROOT/MYAGENT_LANES_STATUS.md, which --chdir puts in the lane's
# first `ls` and which --tmpfs "$RUN_ROOT/competitions" does not hide, being one level up.
# The launcher appends quarantine_partial_attempt.py's stdout to it, and that stdout is one
# line per moved path -- including submissions/submission_top8_equalw_rank_0.849907_....csv,
# because the pipeline names submissions after their CV score. So the module that exists to
# stop a relaunched lane reading its own previous attempt's score wrote that score into a
# file the lane opens on arrival. The gate does not catch it either: its SCORE pattern needs
# a word boundary and _0.849907_ has none.
STATUS=${STATUS:-$RUN_ROOT.status.md}
# Aborted attempts are moved here — OUTSIDE the run root, so a relaunched lane cannot read
# what its previous attempt scored, while the operator keeps every byte of it.
ATTEMPTS=${ATTEMPTS:-$RUN_ROOT.attempts}
# Each lane gets its OWN transcript directory, bound over ~/.claude/projects inside the
# sandbox. The lane's transcript survives on real disk (it is the evidence for the post-run
# audit of what each lane actually opened), while the existing transcripts — and any sibling
# lane's — stay invisible to it.
TRANSCRIPTS=${TRANSCRIPTS:-$RUN_ROOT.transcripts}
# EXPORTED, because lane_sandbox.sh reads them from the environment, not from this shell. The
# first launch of the fixed pipeline died on exactly this: RUN_ROOT was set but not exported,
# so all 20 lanes hit the sandbox's own guard and exited rc=1 within a second of each other.
export RUN_ROOT TRANSCRIPTS

# A competition takes tens of minutes. A lane that returns in seconds did not run one -- it
# failed to start, and the cause is the same for every lane behind it. Marching on turns one
# broken invocation into 20 empty workspaces and a COMPLETE marker that the watchdog reads as
# "finished cleanly", which is the failure this whole file's header is about.
MIN_PLAUSIBLE_SECS=${MIN_PLAUSIBLE_SECS:-60}
CLAUDE=/home/tjyen/.local/bin/claude
PER_COMP_SECS=${PER_COMP_SECS:-21600}   # 6 h safety net — original method was uncapped; observed singles 0.4-4 h, so the cap must sit above the max, not inside the range. Overridable ONLY so a smoke can be bounded; a real lane must keep the default, since a cap inside the observed range truncates the method rather than protecting it.
STALL_MIN=30                # kill a session that has written nothing for this long — longest observed legitimate quiet gap is a single training epoch, well under this
MAX_WAIT_SECS=$((16*3600))

# Kill a process and all descendants. `kill $pid` alone leaves the claude (node) process
# orphaned under init, still holding its dead HTTP stream: the subshell dies, timeout dies,
# and the actual hung process survives — the one thing the stall handler exists to remove.
kill_tree(){ local p; for p in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$p"; done; kill -9 "$1" 2>/dev/null; }

# DERIVED, never hand-listed. A stale hand-written list ran 5 of the 20 competitions and then
# logged "MY-AGENT LANES COMPLETE" — the same "reports success in minutes" shape the archive
# script's header warns about, one layer up (2026-08-10 round-9). The manifest is the single
# definition of which competitions the re-run covers.
COMPS=$(python3 -c "import json,sys;print(' '.join(sorted(json.load(open(sys.argv[1]))['competitions'])))" "$REPO/docs/rerun_manifest.json") || {
  echo "cannot read the competition list from $REPO/docs/rerun_manifest.json" >&2; exit 4; }

# SMOKE_COMPS — run a subset, to exercise THIS file end to end before committing a night to
# it. Every launch bug found so far (RUN_ROOT unexported, COMPLETE after zero lanes, SIGTERM
# booking a live lane as finished) was found by launching all 20 and watching them fail
# together, because there was no way to run one.
#
# The subset is checked against the manifest, so it cannot become the stale hand-list that
# ran 5 of 20 and declared victory. And a smoke NEVER writes the completion marker:
# bench_watchdog.sh reads that marker as "this driver finished cleanly", and a marker earned
# by one competition would retire the other nineteen.
SMOKE=0
if [ -n "${SMOKE_COMPS:-}" ]; then
  for c in $SMOKE_COMPS; do
    case " $COMPS " in
      *" $c "*) ;;
      *) echo "SMOKE_COMPS names '$c', which is not in the manifest" >&2; exit 4;;
    esac
  done
  COMPS=$SMOKE_COMPS
  SMOKE=1
fi

# PER_COMP_SECS is overridable ONLY for a smoke. A real lane inheriting a short cap from the
# environment -- a leftover export, a cron wrapper -- would truncate all twenty runs inside
# the 0.4-4 h band the method actually occupies, and the record would show twenty completed
# lanes with no sign that any of them was cut off. Bind it to SMOKE and log the value, so the
# cap in force is a fact in the run record rather than a property of someone's shell.
if [ "$SMOKE" != "1" ] && [ "$PER_COMP_SECS" != "21600" ]; then
  echo "PER_COMP_SECS=$PER_COMP_SECS but this is not a smoke run; a real lane keeps the 6 h "\
"default. Re-run with SMOKE_COMPS set, or unset PER_COMP_SECS." >&2
  exit 4
fi

# The run root must exist and must pass the clean-slate gate before a single lane starts.
[ -d "$RUN_ROOT" ] || { echo "no run root at $RUN_ROOT — build it with python3 $REPO/benchmark_infra/build_myagent_run_root.py --write" >&2; exit 4; }
# --lane-isolated because every lane below runs under lane_sandbox.sh with LANE_COMP set, so
# it sees ONLY its own competitions/<comp>/. Without the flag the gate scans a finished
# lane's deliverables against the sibling slugs and fails on the library_hits trace SKILL.md
# requires -- which made one completed lane enough to block every relaunch for all 20.
# The flag and the LANE_COMP below are one mechanism; a test asserts they move together.
python3 "$REPO/benchmark_infra/verify_clean_slate.py" --root "$RUN_ROOT" \
        --require-baseline --lane-isolated || {
  echo "run root is not clean — refusing to start" >&2; exit 3; }

# The root can be clean and still unable to run a competition. Its .venv was built by `uv
# sync`, and torch is not in uv.lock (pyproject.toml says so) -- so conway's pinned
# evaluator, which imports a Stage-2 module that imports torch, would have failed at import
# and produced a fabricated loss on a competition my-agent wins. A missing package deflates
# my-agent exactly as leakage inflates it, and nothing checked either the packages or
# whether the GPU the sandbox deliberately preserves is actually usable.
python3 "$REPO/benchmark_infra/preflight_run_root_env.py" --root "$RUN_ROOT" || {
  echo "run root's environment cannot run every competition — refusing to start" >&2; exit 3; }

log(){ echo "- \`$(date '+%m-%d %H:%M')\` $*" | tee -a "$STATUS"; }

# SIGTERM used to make `wait` return, after which the launcher booked the still-running
# lane as finished, released its lock and started the next competition -- so stopping the
# run took two kills, and the first one corrupted the record of the lane it interrupted.
run_pid=''; stall_pid=''
on_signal(){
  trap - TERM INT
  log "stopping on signal — killing the running lane; no COMPLETE marker will be written"
  [ -n "$stall_pid" ] && kill "$stall_pid" 2>/dev/null
  [ -n "$run_pid" ] && kill_tree "$run_pid"
  lane_release 2>/dev/null
  exit 6
}
trap on_signal TERM INT

# Append on relaunch — truncating would erase the record of lanes already run.
[ -f "$STATUS" ] || printf '# my-agent lanes — headless overnight runs\n\n' > "$STATUS"

# Wait for EVERY upstream lane, not just one of them.
#
# This originally waited only on NVIDIA. NVIDIA finished at 23:42 on 2026-07-27 while the
# AIDE lanes ran until the next morning, so my-agent started against a loaded machine: AIDE
# had run afsis alone in 70.9 min, and my-agent began the same competition at load average
# 17. Same competition, different conditions per lane — precisely the defect that voided
# RUN1. Bounded, because an unbounded wait turns one false reading into a night of idling.
waited=0
until grep -q "NVIDIA LANES COMPLETE" "$NV_MARKER" 2>/dev/null \
   && grep -q "PHASE 9A AIDE LANES COMPLETE" "$AIDE_MARKER" 2>/dev/null; do
  [ $waited -ge $MAX_WAIT_SECS ] && { log "WARNING: upstream lanes unfinished after $((MAX_WAIT_SECS/3600))h — starting anyway, conditions were contended"; break; }
  [ $((waited % 1800)) -eq 0 ] && log "waiting for the NVIDIA and AIDE lanes ($((waited/60)) min)"
  sleep 120; waited=$((waited+120))
done

source /home/tjyen/ai_agents/lane_lock.sh
# lane_lock.sh ends with `trap lane_release EXIT INT TERM`, and `source` runs it in THIS
# shell -- so it silently replaced the handler installed above, and on_signal became
# unreachable for the whole run. Re-arm. Keep lane_lock's EXIT trap: releasing the lock on
# any exit is the property it exists for, and on_signal releases explicitly before exiting.
#
# The handler this restores is not a nicety. Without it a SIGTERM makes `wait` return 143
# while the bwrap/claude tree keeps running orphaned; the loop books the lane as
# "NO SUBMISSION", starts the next competition on top of the still-running one, and writes
# MY-AGENT LANES COMPLETE at the end. That is the same unequal-load condition this file's
# header says voided RUN1 -- and it is verbatim the failure on_signal was written to fix,
# reintroduced twenty lines below the fix.
trap on_signal TERM INT

lanes_ran=0
audit_fail_streak=0
for c in $COMPS; do
  # Relaunch-safe: a competition that already produced its submission is done.
  if [ -f "$BASE/competitions/$c/submission.csv" ]; then
    log "SKIP $c: submission already present"
    continue
  fi

  # Anything still here has no submission, so it is an ABORTED attempt — killed by the 6h
  # cap, the stall watchdog, a reboot or the operator. Its STATUS.md opens with "the final CV
  # score" and its tree holds every node it evaluated, so a lane restarting on top of it gets
  # a warm start from its own previous answer. Round 10 closed that channel for the harness's
  # auto-memory; this is the same channel arriving through the filesystem (round 11). Moved,
  # never deleted: the attempt stays available to the operator, outside the lane's reach.
  python3 "$REPO/benchmark_infra/quarantine_partial_attempt.py" \
      --root "$RUN_ROOT" --comp "$c" --dest "$ATTEMPTS" >> "$STATUS" 2>&1 || {
    log "REFUSING to start $c: could not clear its aborted attempt"; continue; }

  # PER LANE, not just before lane 1. The startup preflight proves the environment at that
  # moment and cannot see a change made after it ran -- and torch is out-of-lock, so a single
  # `uv sync` anywhere removes it (reproduced). UV_NO_SYNC in the sandbox is the prevention;
  # this is the detection, and it costs about a second. Catching it here means one lane is
  # affected instead of every competition after the one that broke it.
  python3 "$REPO/benchmark_infra/preflight_run_root_env.py" --root "$RUN_ROOT" >/dev/null || {
    log "ABORTING at $c: the run root's environment no longer satisfies the preflight — "\
"something removed a package the remaining lanes need. Earlier lanes keep their results."
    lane_release 2>/dev/null; exit 5; }
  # ...including the transcript of that attempt, which records its scores turn by turn.
  #
  # mkdir -p first, and CHECK the move. quarantine_partial_attempt creates $ATTEMPTS only
  # when it actually moves something, so an attempt that died before writing a file -- but
  # after a session had already reported scores in its transcript -- left $ATTEMPTS absent,
  # the mv failed into a discarded stderr, and the transcript stayed at $TRANSCRIPTS/$c,
  # which lane_sandbox.sh then binds over ~/.claude/projects for the replacement lane. The
  # one case where the workspace quarantine has nothing to say is exactly the case where the
  # transcript is the only surviving record of the previous attempt's answer.
  if [ -d "$TRANSCRIPTS/$c" ]; then
    mkdir -p "$ATTEMPTS"
    mv "$TRANSCRIPTS/$c" "$ATTEMPTS/$c.transcript.$(date +%s)" || {
      log "REFUSING to start $c: its previous attempt's transcript could not be moved out "\
"of $TRANSCRIPTS, and the new lane would be handed it"; continue; }
  fi

  # Hold the machine for exactly one competition, then hand it on. The upstream-marker wait
  # above establishes ordering; this makes non-overlap a mechanism rather than a convention
  # that every driver has to implement correctly.
  lane_acquire "my-agent/$c" || log "WARNING: started $c without the lane lock — contended"
  log "START $c (cap ${PER_COMP_SECS}s, stall ${STALL_MIN}min)"
  start=$(date +%s)

  PROMPT="You are running ONE benchmark competition with the kaggle-agent skill: $c.

Work in competitions/$c/ (config.yaml present; data/ symlinks to the official competition files and nothing else). This whole tree is an isolated run root built for this benchmark: it holds the skill, the tools and the data, and no record of how any competition turned out before.

Follow the kaggle-agent skill as written — read SKILL.md and its references and do what they say. Do not treat this prompt as the definition of the pipeline; the skill is. In particular the skill designates **tree search as the preferred Stage 4 optimisation loop** (references/07_tree_search.md, harness tree_search/harness_v3.py), to be entered once the linear Iteration Protocol has produced a baseline solo model plus at least one blend, with the linear protocol kept only as the first-pass fallback and for competitions where a ~60-node budget is not worth it. An earlier run of this benchmark listed the stages in the prompt and silently omitted tree search; every competition then finished in 5-32 minutes having never entered it, which is not this agent's method. If you judge a competition too cheap to justify the search, say so explicitly in STATUS.md with the reason.

Also per the skill: consult the experience library ONLY through its filtered tool — \`python3 knowledge/query_library.py --query <terms> --comp $c\` — before EDA and before modeling. NEVER open knowledge/experience.md directly: the raw file carries results outside the tagged bullets that no reading rule can filter. Append every experiment to competitions/$c/experiments.json; write validated new insights back through the library tools, never by hand-appending prose.

CRITICAL — run every training job in the FOREGROUND and wait for it. No '&', nohup or setsid, and never end a turn while a job is still running: this session ends the moment you stop calling tools, so a backgrounded job dies unfinished and the competition produces nothing. A previous conway attempt failed exactly this way. If a configuration would not finish in the budget, shrink it until it completes in the foreground.

Constraints:
- STRICT lane isolation: never read or reference anything under ~/ai_agents/aideml*, ~/ai_agents/nvidia-kaggle*, or other agents' outputs. STAY INSIDE THIS ROOT: do not read the parent directory, ~/ai_agents/* (other than the data/ symlink target), any other checkout of this project, or ~/.claude/. There is no git repository here and you must not create or consult one. Everything you need is in this tree; anything outside it is either another agent's lane or a record of a previous run of this same competition.
- Machine is shared: cap threads at 10 (LightGBM num_threads, OMP). Fix seeds; deterministic=true, force_row_wise=true for LightGBM.
- Budget: work at your normal pace; the pipeline decides when it is done (6 h hard safety net). A completed modest pipeline beats an unfinished ambitious one.
- Finish by writing competitions/$c/submission.csv (columns/id order per sample_submission.csv) and a 3-line summary at the top of competitions/$c/STATUS.md with the final CV score.

Work autonomously; never ask questions; take documented fallbacks when blocked."

  # Stall watchdog for this one competition.
  #
  # Neither file output nor process state alone can tell a stall from work:
  #   - conway 2026-07-28 looked hung by every process signal (2h47m elapsed, 1m42s CPU,
  #     idle TCP) while a foreground cnn_v2 training was legitimately running — and that
  #     training wrote NO file for 70 minutes, so an output-only check would kill it too.
  #     It was killed by hand on those signals; the training survived as an orphan and the
  #     result was salvaged, but the kill itself was a misdiagnosis.
  #   - a genuinely wedged session (dead HTTP stream) also writes nothing, but burns no CPU.
  # So a stall requires BOTH: no file written under the workspace for STALL_MIN AND no CPU
  # consumed by the process tree across the check interval. Training always burns CPU;
  # a dead stream never does.
  # Through the sandbox, always. Round 10 made the run root credential-free and stripped the
  # submit routes from the skill; round 11 showed the lane never ran in the root's
  # environment. ~/.bashrc exports a live KAGGLE_API_TOKEN into every Bash tool call,
  # ~/.kaggle/huang_token is the account that submitted every competition for all three
  # lanes, ~/ai_agents holds the all-20 public/private table, and ~/.claude/projects holds
  # 1050 transcripts naming a benchmark competition. None of that is reachable by redaction.
  ( LANE_TRANSCRIPTS="$TRANSCRIPTS/$c" LANE_COMP="$c" \
    timeout -k 60 $PER_COMP_SECS bash "$REPO/benchmark_infra/lane_sandbox.sh" \
      "$CLAUDE" -p "$PROMPT" --dangerously-skip-permissions \
      > "$BASE/competitions/$c/headless_run.log" 2>&1 ) &
  run_pid=$!

  (
    tree_cpu(){ # total CPU jiffies of $1 and all descendants
      local p sum=0 jif
      for p in $1 $(pgrep -P "$1" 2>/dev/null); do
        [ "$p" = "$1" ] || { jif=$(tree_cpu "$p"); sum=$((sum + jif)); }
      done
      jif=$(awk '{print $14+$15}' "/proc/$1/stat" 2>/dev/null || echo 0)
      echo $((sum + jif))
    }
    prev_cpu=0
    while kill -0 $run_pid 2>/dev/null; do
      sleep 300
      kill -0 $run_pid 2>/dev/null || break
      cur_cpu=$(tree_cpu $run_pid)
      quiet_files=0
      # ABSOLUTE: bfs rejects a relative -newermt and the error goes to /dev/null, so this
      # half of the test silently answered "quiet" every time (same bug as bench_watchdog).
      stale_since=$(date -d "-${STALL_MIN} minutes" '+%F %T')
      [ -z "$(find "$BASE/competitions/$c" -newermt "$stale_since" -type f 2>/dev/null | head -1)" ] && quiet_files=1
      # Under 5 CPU-seconds across 5 minutes = idle; an API round-trip alone costs more.
      if [ "$quiet_files" = "1" ] && [ $((cur_cpu - prev_cpu)) -lt 500 ]; then
        log "STALLED $c: no output for ${STALL_MIN}min and no CPU in 5min — killing the session"
        kill_tree $run_pid
        break
      fi
      prev_cpu=$cur_cpu
    done
  ) &
  stall_pid=$!

  wait $run_pid
  rc=$?
  kill $stall_pid 2>/dev/null

  lane_release
  lanes_ran=$((lanes_ran + 1))
  secs=$(( $(date +%s) - start ))
  mins=$(( secs / 60 ))

  # Did it RUN, or did it fail to start? A competition takes tens of minutes; seconds means
  # the invocation is broken, and it is broken identically for every lane behind this one.
  if [ "$rc" -ne 0 ] && [ "$secs" -lt "$MIN_PLAUSIBLE_SECS" ]; then
    log "ABORTING: $c exited rc=$rc after ${secs}s — under ${MIN_PLAUSIBLE_SECS}s means the "\
"lane never started, and the next 19 would fail the same way. Last lines of its log:"
    tail -5 "$BASE/competitions/$c/headless_run.log" 2>/dev/null | sed 's/^/    /' | tee -a "$STATUS"
    log "no COMPLETE marker written — this run did not happen"
    exit 5
  fi

  if [ -f "$BASE/competitions/$c/submission.csv" ]; then
    log "DONE $c: submission written (${mins} min, rc=$rc)"
  else
    log "DONE $c: NO SUBMISSION (${mins} min, rc=$rc) — needs a human look"
  fi

  # WHILE THE TRANSCRIPT IS STILL ONE LANE. Three of this run's rules are held by instruction
  # and not by the sandbox -- no network fetches, the library only through query_library.py,
  # nothing outside the run root -- because claude needs the network and the library files
  # live in the root. audit_lane_transcript.py reads what the lane actually reached for.
  #
  # Per lane, not only at the end: an audit that runs after 20 lanes tells you the run is
  # void after you have spent it.
  audit_out=$(python3 "$REPO/benchmark_infra/audit_lane_transcript.py" \
      --transcripts "$TRANSCRIPTS" --root "$RUN_ROOT" --comp "$c" 2>&1)
  audit_rc=$?
  if [ "$audit_rc" -ne 0 ]; then
    log "TRANSCRIPT AUDIT FAILED for $c — this lane's result cannot be used as recorded:"
    printf '%s\n' "$audit_out" | sed 's/^/    /' | tee -a "$STATUS"
    audit_fail_streak=$((audit_fail_streak + 1))
    # One lane can misbehave on its own; two in a row is the pipeline doing it, and every
    # remaining lane will do it too. Stopping then costs 18 lanes of compute instead of
    # producing 20 results that have to be thrown away.
    if [ "$audit_fail_streak" -ge 2 ]; then
      log "ABORTING: two consecutive lanes failed the transcript audit — this is the "\
"pipeline, not one lane. Earlier lanes keep their results."
      exit 5
    fi
  else
    audit_fail_streak=0
    printf '%s\n' "$audit_out" | grep -E '^(MINOR|MAJOR|BLOCKER):' | tr '\n' ' ' \
      | sed "s/^/- \`$(date '+%m-%d %H:%M')\` audit $c: /" >> "$STATUS"
    echo >> "$STATUS"
  fi
done

if [ "$lanes_ran" -eq 0 ]; then
  log "NO LANE RAN — refusing to write the completion marker, which bench_watchdog.sh "\
"reads as 'this driver finished cleanly'"
  exit 5
fi
if [ "$SMOKE" = "1" ]; then
  log "SMOKE RUN over $lanes_ran competition(s) — no completion marker; the other lanes "\
"have not run"
  exit 0
fi
log "MY-AGENT LANES COMPLETE ($lanes_ran/20 lanes ran)"
