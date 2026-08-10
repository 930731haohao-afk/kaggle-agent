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
exec bwrap \
  --ro-bind / / \
  --dev /dev --proc /proc --tmpfs /tmp --tmpfs /run \
  --tmpfs "$HOME/.kaggle" \
  --tmpfs "$HOME/ai_agents" \
  --tmpfs "$HOME/Documents" \
  --bind "$LANE_TRANSCRIPTS" "$HOME/.claude/projects" \
  --bind "$HOME/.cache" "$HOME/.cache" \
  --bind "$RUN_ROOT" "$RUN_ROOT" \
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
