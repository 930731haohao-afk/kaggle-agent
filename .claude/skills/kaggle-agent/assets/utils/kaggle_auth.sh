#!/bin/bash
# Source this script to set KAGGLE_API_TOKEN, then run a kaggle command in the same call.
# Usage: source utils/kaggle_auth.sh && uv run kaggle competitions list
#        KAGGLE_TOKEN_FILE=~/.kaggle/huang_token source utils/kaggle_auth.sh && ...
#
# (2026-08-03 audit) This used to read a hardcoded Windows path
# ('C:/Users/user/.kaggle/kaggle.json') on a Linux host. Two failures stacked:
#   1. kaggle.json does not work with the new KGAT_ tokens at all — CLAUDE.md says the
#      KAGGLE_API_TOKEN env var is the only working method.
#   2. Being sourced, the failed $(...) exported an EMPTY KAGGLE_API_TOKEN and left $?
#      at 0, so `source utils/kaggle_auth.sh && kaggle ...` happily ran the command and
#      every kaggle call died with an auth error pointing nowhere.
# It now reads the token file this project actually uses and fails loudly instead.

# `return` when sourced (the supported use), `exit` when executed directly.
(return 0 2>/dev/null) && _kaggle_auth_die=return || _kaggle_auth_die=exit

# ~/.kaggle/kaggle_api_token.txt holds the plain KGAT_ token for this repo's account
# (tjyen1975) — the location REPRODUCE.md, .claude/skills/README.md and the competition
# STATUS.md files all cite. ~/.kaggle/huang_token is a DIFFERENT account (AAPB); select
# it with KAGGLE_TOKEN_FILE, never by editing this default.
KAGGLE_TOKEN_FILE="${KAGGLE_TOKEN_FILE:-$HOME/.kaggle/kaggle_api_token.txt}"

if [ ! -r "$KAGGLE_TOKEN_FILE" ]; then
    echo "kaggle_auth: token file not found or unreadable: $KAGGLE_TOKEN_FILE" >&2
    echo "kaggle_auth: create it with the KGAT_ token from https://www.kaggle.com/settings" >&2
    echo "kaggle_auth: (or point KAGGLE_TOKEN_FILE elsewhere). NOT exporting KAGGLE_API_TOKEN." >&2
    unset KAGGLE_API_TOKEN
    $_kaggle_auth_die 1
fi

KAGGLE_API_TOKEN=$(tr -d '[:space:]' < "$KAGGLE_TOKEN_FILE")

if [ -z "$KAGGLE_API_TOKEN" ]; then
    echo "kaggle_auth: $KAGGLE_TOKEN_FILE is empty. NOT exporting KAGGLE_API_TOKEN." >&2
    unset KAGGLE_API_TOKEN
    $_kaggle_auth_die 1
fi

# The new-style tokens start with KGAT_; a kaggle.json 'key' pasted in here would be
# accepted by this script but rejected by the API, so say something instead of guessing.
case "$KAGGLE_API_TOKEN" in
    KGAT_*) ;;
    *) echo "kaggle_auth: warning — token in $KAGGLE_TOKEN_FILE does not start with KGAT_;" >&2
       echo "kaggle_auth: old kaggle.json keys do not work with the current API." >&2 ;;
esac

# Name the source on success. A token can also be *present and rotated* — the CLI then
# returns a bare 401 and, without this line, you again cannot tell which credential to
# refresh. (Checked 2026-08-03: this file's KGAT_7... token 401s; a fresh token from
# https://www.kaggle.com/settings pasted into it restores submissions.)
echo "kaggle_auth: KAGGLE_API_TOKEN set from $KAGGLE_TOKEN_FILE" >&2
export KAGGLE_API_TOKEN
unset _kaggle_auth_die
