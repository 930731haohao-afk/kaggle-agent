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
# Optional dotfiles, bound back only if present: bwrap fails the whole mount if a --bind
# source is missing, and a machine without one of these must not lose its sandbox.
#
# NOT .bashrc, and NOT .profile. Both were bound back for PATH, and .bashrc:123 is
# `export GITHUB_PERSONAL_ACCESS_TOKEN=<literal>` -- so the --unsetenv above was undone by
# the allowlist entry beside it: the variable was empty and the credential was one `cat`
# away in cleartext. PATH is set explicitly below instead, which is what the file was
# wanted for. (.bashrc:142's `alias rm='trash-put'` is also why ~/.local/share/Trash holds
# the answer key; the file has now cost this sandbox two findings.)
home_files=()
for f in .gitconfig; do
  [ -e "$HOME/$f" ] && home_files+=(--ro-bind "$HOME/$f" "$HOME/$f")
done

comp_isolation=(--tmpfs "$RUN_ROOT/competitions")
if [ "$LANE_COMP" != "none" ]; then
  # LANE_COMP must be a plain slug. Unvalidated, a value like ../other-comp or an absolute
  # path resolves back out of the isolation and silently restores the view this exists to
  # remove -- and it would look like it worked.
  case "$LANE_COMP" in
    */*|.*|"") echo "LANE_COMP must be a bare competition slug, not $LANE_COMP" >&2; exit 4;;
  esac
  ws=$RUN_ROOT/competitions/$LANE_COMP
  [ -d "$ws" ] || { echo "no workspace at $ws — LANE_COMP names a competition this root does not have" >&2; exit 4; }
  comp_isolation+=(--bind "$ws" "$ws")

  # THE OOF CACHES ARE THE SAME CHANNEL, ONE DIRECTORY OVER.
  #
  # Isolating competitions/ left tree_search/ bound whole, and all 20 pinned evaluators
  # cache there: cache_afsis, cache_citd, cache_conway, ... one per competition. Each .npz
  # holds ['oof', 'pred', <metric>] -- the out-of-fold vectors, the test predictions AND the
  # score. So lane k could read lanes 1..k-1's cached vectors and their metric values, on a
  # clean first pass, with no abort and no leftover: exactly the exposure the workspace
  # isolation closed, in the directory next to it.
  #
  # Resolved by the SAME reader the quarantine uses -- each evaluator's own CACHE_DIR, from
  # its source. Not a name heuristic: those guessed wrong for 4 of 20 (cat-in-the-dat writes
  # cache_citd, sep-2022 cache_tssep22_main), and a lane whose own cache was tmpfs'd by
  # mistake would silently recompute every vector or, worse, blend on an empty cache.
  own_caches=$(python3 -c "
import sys
sys.path.insert(0, '$(dirname "$(readlink -f "$0")")')
import quarantine_partial_attempt as q
print(' '.join(q.cache_dirs_for('$RUN_ROOT', '$LANE_COMP')))" 2>/dev/null)
  if [ -z "$own_caches" ]; then
    echo "REFUSING: cannot resolve $LANE_COMP's OOF cache directory from its pinned "\
"evaluator. Blanking the others without knowing which is this lane's would either expose "\
"a sibling's vectors or destroy this lane's." >&2
    exit 4
  fi
  for rel in $own_caches; do mkdir -p "$RUN_ROOT/$rel"; done
  for d in "$RUN_ROOT"/tree_search/cache_*; do
    [ -d "$d" ] || continue
    keep=0
    for rel in $own_caches; do
      case "$d" in "$RUN_ROOT/$rel"|"$RUN_ROOT/$rel"/*) keep=1;; esac
    done
    [ "$keep" = "1" ] || comp_isolation+=(--tmpfs "$d")
  done
fi

# $HOME IS AN ALLOWLIST, NOT A DENY LIST.
#
# It used to be four named directories blanked under a blanket --ro-bind / /: ~/.kaggle,
# ~/ai_agents, ~/Documents, ~/.claude, plus ~/benchruns. That shape is fail-open by
# construction -- every path NOT named stays readable -- and the benchmark's answer key was
# sitting in three of them. Measured with this project's own detector (verify_clean_slate's
# scan_file, against the 20 real slugs):
#
#   ~/aide-benchmark-package/    12 files, incl. results/three_way_scores.csv: one row per
#                                competition with mine_local, nvidia_local, aide_local,
#                                mine_pub, mine_priv, mine_rank and both reference lanes'
#   ~/Desktop/                  179 files, incl. kaggle_competition_scripts/competitions/
#                                <slug>/scripts/ for all 20 -- the previous run's tuned
#                                scripts, i.e. exactly the warm start quarantine_partial_
#                                attempt.py exists to prevent, reachable without ever
#                                touching the run root
#   ~/Downloads/, ~/kaggle_aapb/   4 more
#
# Naming those three would leave the shape intact and the next tree would reopen it. So
# blank $HOME and bind back only what a lane needs to run. Everything omitted here is
# invisible by default, which is the property a deny list can never have.
#
# What is bound back, and why each one:
#   .local/bin            the uv binary and the claude symlink
#   .local/share/claude   what that symlink points at
#   .local/share/uv       uv's own data directory (its managed pythons and tool state).
#                         WITHOUT IT, uv does not merely lose a cache -- it REBUILDS the run
#                         root's .venv on the next `uv run`, and it rebuilds it from uv.lock,
#                         which is how 228 packages became 3 and how torch "disappeared"
#                         twice today. That was never a sync pruning an out-of-lock package;
#                         it was the whole environment being recreated. Scanned with this
#                         project's own detector: 15986 files, 0 name a benchmark competition.
#   .local/state/claude   the agent's own state, writable
#
#   NOT ~/.local WHOLE, which is what this first said, to obtain those same two binaries.
#   ~/.local/share/Trash is the GNOME trash, and ~/.bashrc:142 is `alias rm='trash-put'`, so
#   every tree the operator has ever deleted from $HOME is in there -- including
#   nvidia-kaggle-runs/ (the frozen reference lane, with a per-competition result.json
#   carrying BOTH lanes' numbers) and two copies of a previous run's tuned per-competition
#   scripts. This project's own detector flags 314 files under ~/.local, 310 of them in the
#   trash: MORE than the 195 this allowlist was written to close. The channel refills itself
#   every time the operator deletes something. An allowlist entry that is a whole tree is a
#   deny list again, one level down.
#   .cache          uv's package cache; writable, or every lane re-downloads its wheels
#   .bashrc/.profile  PATH. The token export at .bashrc:122 reads ~/.kaggle/kaggle.json,
#                   which no longer exists inside, so it yields an empty string.
#   .gitconfig      harmless, and tooling complains without it
#   .claude/.credentials.json   the agent authenticates with this and nothing else
#
# UV_NO_SYNC=1 because torch is NOT in uv.lock (pyproject.toml records why), and a plain
# `uv sync` REMOVES out-of-lock packages -- reproduced: torch present, `uv sync`, torch gone.
# The run root's venv is provisioned by us with `sync --inexact`; a lane re-syncing it would
# silently delete the package conway's evaluator imports, and the startup preflight cannot
# see a removal that happens after it ran. The skill never tells a lane to sync, but that is
# an instruction, and this file's whole premise is that instructions are not boundaries.
# Deliberately NOT bound back: .ssh, .gnupg, .config (which holds a GitHub token under gh/).
#
# WHAT THIS STILL DOES NOT CLOSE, stated plainly because it cannot be closed here. The lane
# needs the network to reach the Anthropic API, so there is no --unshare-net, and this
# repository is published. A lane that chose to could fetch its own competition's recorded
# STATUS.md over HTTPS. The credentials are gone (KAGGLE_* and the GitHub PAT are unset
# above, ~/.config/gh is not bound), so this is unauthenticated fetching of a public repo,
# and the compensating control is the post-run transcript audit: every lane's transcript is
# preserved outside the sandbox and records every tool call it made. That is a DETECTIVE
# control, not a boundary, and it is the one place in this design where prose still does
# work. Closing it properly needs an egress proxy that allows only api.anthropic.com.
exec bwrap \
  --ro-bind / / \
  --dev-bind /dev /dev --proc /proc --tmpfs /tmp \
  --tmpfs "$HOME" \
  --ro-bind "$HOME/.local/bin" "$HOME/.local/bin" \
  --ro-bind "$HOME/.local/share/claude" "$HOME/.local/share/claude" \
  --ro-bind "$HOME/.local/share/uv" "$HOME/.local/share/uv" \
  --bind "$HOME/.local/state/claude" "$HOME/.local/state/claude" \
  --bind "$HOME/.cache" "$HOME/.cache" \
  "${home_files[@]}" \
  --ro-bind "$HOME/.claude/.credentials.json" "$HOME/.claude/.credentials.json" \
  --bind "$RUN_ROOT" "$RUN_ROOT" \
  "${comp_isolation[@]}" \
  --bind "$LANE_TRANSCRIPTS" "$HOME/.claude/projects" \
  --setenv MPLCONFIGDIR "$RUN_ROOT/.mplconfig" \
  --setenv UV_NO_SYNC 1 \
  --setenv PATH "$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
  --unsetenv KAGGLE_API_TOKEN \
  --unsetenv KAGGLE_USERNAME \
  --unsetenv KAGGLE_KEY \
  --unsetenv GITHUB_PERSONAL_ACCESS_TOKEN \
  --unsetenv GITHUB_TOKEN \
  --unsetenv GH_TOKEN \
  --unsetenv KAGGLE_CONFIG_DIR \
  --setenv KAGGLE_CONFIG_DIR "$HOME/.kaggle" \
  --setenv VIRTUAL_ENV "" \
  --chdir "$RUN_ROOT" \
  --die-with-parent \
  "$@"
