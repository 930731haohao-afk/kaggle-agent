#!/bin/bash
# Run one benchmark lane with the out-of-root surfaces removed from its filesystem.
#
# WHY A SANDBOX AND NOT A RULE. Eleven audit rounds tried to make the run root clean by
# subtraction, and round 11 showed the root was never the boundary. A lane does not run in
# the root's environment; it runs in the operator's login shell:
#
#   ~/.bashrc exports KAGGLE_API_TOKEN unconditionally, so every Bash tool call in every lane
#   already holds a live token. ~/.kaggle/huang_token is the account that submitted EVERY
#   benchmark competition for ALL THREE lanes, readable at a canonical path one `cat` away.
#   `kaggle` is on PATH, and `kaggle competitions submissions <comp>` takes the competition
#   POSITIONALLY -- so the literal round 10 stripped from the skill was not even the form a
#   lane would type. Removing the instructions removed no capability.
#
#   ~/ai_agents/ holds MASTER_TODO.md and LANE_CLAIMS.md, which carry the all-20 public and
#   private table, one `..` from where the run root used to default.
#
#   ~/.claude/projects/ holds the session transcripts. These are NOT the memory directory
#   round 10 disabled -- they sit beside it, at a fixed path, and 1050 of them name a
#   benchmark competition.
#
# None of that is reachable by redaction, and rounds 1-4 already established that prose is
# not a boundary. So: it is a filesystem boundary or it is nothing.
#
# WHAT SURVIVES. Everything the agent genuinely needs -- its own binary and credentials, uv,
# python, the caches -- and the run root, writable. What the lane cannot see, it cannot read
# by accident, by instruction, or by trying.
#
# Usage:  lane_sandbox.sh <command> [args...]      (RUN_ROOT must be set)
set -uo pipefail

: "${RUN_ROOT:?RUN_ROOT must be set}"
RUN_ROOT=$(cd "$RUN_ROOT" 2>/dev/null && pwd) || { echo "no such RUN_ROOT: $RUN_ROOT" >&2; exit 4; }

case "$RUN_ROOT" in
  "$HOME"/ai_agents/*)
    echo "REFUSING: $RUN_ROOT is inside ~/ai_agents, which this sandbox blanks" >&2; exit 4;;
esac

command -v bwrap >/dev/null || { echo "bwrap is not installed; refusing to run a lane with the credential and transcript surfaces exposed" >&2; exit 4; }

# The lane's OWN transcript must survive -- it is the evidence for the post-run audit of what
# each lane actually opened, which is what replaces another round of static leak-hunting. So
# ~/.claude/projects is not blanked but REPLACED: an empty per-lane directory is bound over
# it, so the 1050 existing transcripts that name a benchmark competition are gone while this
# lane's own writes land on real disk. Per-lane, not per-run, so a relaunched lane cannot
# read the transcript of its own aborted attempt either -- the same reason its STATUS.md and
# tree are quarantined before it restarts.
LANE_TRANSCRIPTS=${LANE_TRANSCRIPTS:-$RUN_ROOT.transcripts/default}
mkdir -p "$LANE_TRANSCRIPTS" || exit 4
case "$LANE_TRANSCRIPTS" in
  "$RUN_ROOT"|"$RUN_ROOT"/*)
    echo "REFUSING: LANE_TRANSCRIPTS is inside the run root, where the lane would read it" >&2
    exit 4;;
esac

# --tmpfs over a path replaces it with an empty writable directory INSIDE the sandbox only;
# nothing on the real filesystem is touched or deleted. Order matters: bwrap applies these
# left to right, so a later --bind can re-expose one subtree of an earlier --tmpfs.
#
# THREE THINGS THIS MUST NOT BREAK, each of which the first version did:
#
#   --dev-bind, NOT --dev. `--dev /dev` mounts a MINIMAL devtmpfs, which has no /dev/nvidia*,
#   so every lane would have trained on CPU while the two frozen reference lanes had the GPU.
#   That does not leak anything -- it silently handicaps my-agent and makes the whole
#   comparison meaningless in the direction that looks like a fair loss.
#
#   NO --tmpfs /run. /etc/resolv.conf is a symlink into /run/systemd/resolve/, so blanking
#   /run leaves it dangling: DNS fails and `claude -p` cannot reach the API at all. /run holds
#   no benchmark result; it was blanked out of tidiness and cost the run its network.
#
#   ~/.claude is blanked WHOLE, not just its projects/ subdirectory. The rest of it --
#   history.jsonl, file-history/, debug/, backups/, cache/ -- holds hundreds of files stating
#   benchmark results, and a live Kaggle token. Only .credentials.json is bound back, so the
#   agent can authenticate; everything else it needs it recreates in the tmpfs.
#
# ~/benchruns is blanked too, with only THIS run root bound back: the aborted attempts, the
# per-lane transcripts, and any earlier build (including one made before the prose scrubs)
# all live there under their real paths.
BENCHRUNS=$(dirname "$RUN_ROOT")

# ONE WORKSPACE PER LANE.
#
# The run root holds all twenty competition workspaces, and binding it whole handed every
# lane the other nineteen. By the time lane 20 starts, nineteen STATUS.md files are sitting
# beside it, each opening with its competition's final CV score and champion recipe, and the
# only thing between them is a sentence in the prompt. That is true on a clean FIRST pass --
# it needs no abort, no relaunch, no leftover. Twelve rounds of this audit have been about
# the difference between a rule and a boundary; this was the outermost layer still relying
# on a rule.
#
# It is also what made the run unrestartable. The startup gate scans the whole root, and a
# finished lane's own deliverables cite sibling competitions by name and number -- because
# SKILL.md REQUIRES a library_hits trace and query_library.py serves evidence from other
# competitions by design. So the gate failed on the lane doing exactly what it was told, and
# every relaunch exited 3. With the siblings unreachable, their content is no longer
# something the gate has to protect against.
#
# LANE_COMP is REQUIRED, with no default. A default would be the fail-open shape this whole
# audit keeps closing: a caller that forgets it would silently get the old, unbounded view.
#   LANE_COMP=<slug>  bind only that workspace; the other nineteen do not exist
#   LANE_COMP=none    no workspace at all -- for maintenance (uv sync, read-only probes)
: "${LANE_COMP:?LANE_COMP must be set: a competition slug, or 'none' for maintenance}"
comp_isolation=(--tmpfs "$RUN_ROOT/competitions")
if [ "$LANE_COMP" != "none" ]; then
  ws=$RUN_ROOT/competitions/$LANE_COMP
  [ -d "$ws" ] || { echo "no workspace at $ws — LANE_COMP names a competition this root does not have" >&2; exit 4; }
  comp_isolation+=(--bind "$ws" "$ws")
fi

exec bwrap \
  --ro-bind / / \
  --dev-bind /dev /dev --proc /proc --tmpfs /tmp \
  --tmpfs "$HOME/.kaggle" \
  --tmpfs "$HOME/ai_agents" \
  --tmpfs "$HOME/Documents" \
  --tmpfs "$HOME/.claude" \
  --ro-bind "$HOME/.claude/.credentials.json" "$HOME/.claude/.credentials.json" \
  --tmpfs "$BENCHRUNS" \
  --bind "$RUN_ROOT" "$RUN_ROOT" \
  "${comp_isolation[@]}" \
  --bind "$LANE_TRANSCRIPTS" "$HOME/.claude/projects" \
  --bind "$HOME/.cache" "$HOME/.cache" \
  --setenv MPLCONFIGDIR "$RUN_ROOT/.mplconfig" \
  --unsetenv KAGGLE_API_TOKEN \
  --unsetenv KAGGLE_USERNAME \
  --unsetenv KAGGLE_KEY \
  --unsetenv KAGGLE_CONFIG_DIR \
  --setenv KAGGLE_CONFIG_DIR "$HOME/.kaggle" \
  --setenv VIRTUAL_ENV "" \
  --chdir "$RUN_ROOT" \
  --die-with-parent \
  "$@"
