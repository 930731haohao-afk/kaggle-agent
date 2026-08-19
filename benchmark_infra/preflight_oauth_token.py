#!/usr/bin/env python3
"""Refuse to start a long lane on an OAuth token that will expire during it.

WHY THIS EXISTS. Two of this benchmark's three multi-hour outages were the same failure:
a lane started, ran correctly for 39-113 minutes, and then died with

    Failed to authenticate. API Error: 401 OAuth access token has been revoked.

taking its work with it. The access token's lifetime is on the order of hours; a lane's cap
is six. The sandbox now binds .credentials.json writable so a session can refresh in place,
which is the actual fix -- but refresh can still fail (a revoked refresh token, a clock skew,
a network blip at exactly the wrong minute), and when it does the failure is silent until the
lane is already deep into its budget.

This is the cheap guard in front of that: read the expiry, compare it to what the lane is
allowed to consume, and refuse BEFORE the lane starts. A refusal at second zero is legible
and costs nothing; the same refusal 75 minutes in costs 75 minutes and looks like a crash.

WHAT IT DOES NOT DO. It does not refresh anything, write anything, or contact any server. It
reads one local JSON file and exits. It also never treats "I could not read the expiry" as
"the token is fine": an unreadable or malformed credentials file is rc 3, not rc 0, because a
guard that fails open is not a guard. The one deliberate exception is a missing `expiresAt`
field -- some credential shapes do not carry one -- which is rc 0 with a note, since refusing
every lane on a file we simply do not understand would be worse than the failure we are
preventing.

Exit codes:
    0  the token outlives the lane budget (or the file carries no expiry to check)
    2  the token expires before the lane budget does -- do not start
    3  the credentials file is missing, unreadable, or malformed

Usage:
    preflight_oauth_token.py --budget-secs 21600
    preflight_oauth_token.py --budget-secs 21600 --credentials ~/.claude/.credentials.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

DEFAULT_CREDENTIALS = os.path.expanduser("~/.claude/.credentials.json")

# Where the access-token expiry lives, in the shapes this project has actually seen. Each is
# a path of dict keys. Checked in order; the first one present wins.
EXPIRY_PATHS = (
    ("claudeAiOauth", "expiresAt"),
    ("oauth", "expiresAt"),
    ("expiresAt",),
)
# Same, for the refresh token. Reported for the operator's benefit, never used to pass or
# fail: a live refresh token is what makes in-place renewal possible, so an operator reading
# a refusal wants to know whether re-auth means "one command" or "log in again".
REFRESH_EXPIRY_PATHS = (
    ("claudeAiOauth", "refreshTokenExpiresAt"),
    ("oauth", "refreshTokenExpiresAt"),
    ("refreshTokenExpiresAt",),
)


def _dig(doc: dict, path: tuple[str, ...]):
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _as_epoch_seconds(raw) -> float | None:
    """Accept seconds or milliseconds; these files have used both.

    A bare int is ambiguous, so it is disambiguated by magnitude rather than by assumption:
    anything past the year 2100 in seconds is milliseconds. Getting this backwards would make
    a valid token look 1000x expired (refusing every lane) or an expired one look valid for
    millennia (refusing none) -- the second is the dangerous direction, so the comparison is
    written to be obvious rather than clever.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if raw <= 0:
        return None
    return raw / 1000.0 if raw > 4_102_444_800 else float(raw)


def check(credentials_path: str, budget_secs: float, now: float | None = None) -> tuple[int, str]:
    now = time.time() if now is None else now
    try:
        with open(credentials_path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        return 3, f"no credentials file at {credentials_path}"
    except (OSError, json.JSONDecodeError) as exc:
        return 3, f"credentials file at {credentials_path} is unreadable: {exc}"
    if not isinstance(doc, dict):
        return 3, f"credentials file at {credentials_path} is not a JSON object"

    raw = next((v for p in EXPIRY_PATHS if (v := _dig(doc, p)) is not None), None)
    if raw is None:
        return 0, "credentials carry no access-token expiry field — nothing to check"
    expires_at = _as_epoch_seconds(raw)
    if expires_at is None:
        return 3, f"access-token expiry is not a usable timestamp: {raw!r}"

    remaining = expires_at - now
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(expires_at))

    refresh_note = ""
    refresh_raw = next(
        (v for p in REFRESH_EXPIRY_PATHS if (v := _dig(doc, p)) is not None), None)
    refresh_at = _as_epoch_seconds(refresh_raw)
    if refresh_at is not None:
        if refresh_at > now:
            refresh_note = (" The refresh token is still valid until "
                            f"{time.strftime('%Y-%m-%d', time.localtime(refresh_at))}, so a "
                            "running session should be able to renew in place; if lanes are "
                            "still dying on 401, the refresh path itself is broken.")
        else:
            refresh_note = (" The refresh token has ALSO expired — in-place renewal cannot "
                            "work and the operator must re-authenticate before any lane runs.")

    if remaining <= 0:
        return 2, (f"the access token expired at {stamp}. Re-authenticate before starting a "
                   f"lane.{refresh_note}")
    if remaining < budget_secs:
        return 2, (f"the access token expires at {stamp}, in {remaining / 3600:.1f} h, but a "
                   f"lane may run for {budget_secs / 3600:.1f} h. A lane started now can "
                   f"outlive its own credentials and die mid-run.{refresh_note}")
    return 0, (f"access token valid for {remaining / 3600:.1f} h, longer than the "
               f"{budget_secs / 3600:.1f} h lane budget")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget-secs", type=float, required=True,
                    help="the wall-clock a single lane is allowed to consume")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    a = ap.parse_args(argv)
    rc, msg = check(a.credentials, a.budget_secs)
    prefix = {0: "oauth preflight OK", 2: "OAUTH PREFLIGHT FAILED",
              3: "OAUTH PREFLIGHT INCONCLUSIVE"}[rc]
    print(f"{prefix} — {msg}", file=sys.stderr if rc else sys.stdout)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
