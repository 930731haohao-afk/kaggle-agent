"""The OAuth token preflight: refuse a lane the credentials cannot outlive.

Two of the 20-lane rerun's three outages were a lane dying on `401 OAuth access token has
been revoked` after 39 and 75 minutes of correct work. The sandbox now binds
.credentials.json writable so a session can refresh in place; this guard is what runs when
refresh does not happen. These tests pin the properties that make it worth having:

  * it refuses when the token expires inside the lane's budget, and passes when it does not
  * it does NOT fail open on a missing, corrupt or nonsense credentials file
  * it reads both second and millisecond timestamps, because this project's files use both
  * it never writes to the credentials file it reads

The last one matters more than it looks: a preflight that mutates the credential it is
checking would be a new way to break exactly what it protects.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "benchmark_infra", "preflight_oauth_token.py")
SIX_HOURS = 21600


def _write_creds(tmp_path, expires_at, refresh_at=None, shape="claudeAiOauth"):
    inner = {"accessToken": "sk-ant-oat-fake", "expiresAt": expires_at}
    if refresh_at is not None:
        inner["refreshTokenExpiresAt"] = refresh_at
    doc = inner if shape is None else {shape: inner}
    p = tmp_path / ".credentials.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def _run(creds, budget=SIX_HOURS):
    return subprocess.run(
        [sys.executable, SCRIPT, "--budget-secs", str(budget), "--credentials", creds],
        capture_output=True, text=True, check=False)


class TestTheDecision:
    def test_token_outliving_the_budget_passes(self, tmp_path):
        creds = _write_creds(tmp_path, (time.time() + 8 * 3600) * 1000)
        r = _run(creds)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "valid for" in r.stdout

    def test_token_expiring_inside_the_budget_is_refused(self, tmp_path):
        # The exact shape of the real failure: 3.1 h of token, 6 h of lane.
        creds = _write_creds(tmp_path, (time.time() + 3.1 * 3600) * 1000)
        r = _run(creds)
        assert r.returncode == 2, r.stdout + r.stderr
        assert "can outlive its own credentials" in r.stderr

    def test_already_expired_token_is_refused(self, tmp_path):
        creds = _write_creds(tmp_path, (time.time() - 60) * 1000)
        r = _run(creds)
        assert r.returncode == 2
        assert "expired at" in r.stderr

    def test_the_boundary_is_the_budget_not_the_clock(self, tmp_path):
        """A token with 4 h left is fine for a 1 h lane and not for a 6 h lane.

        The refusal is a statement about the lane, not about the token in isolation, and a
        guard that ignored the budget would either block every short lane or pass every
        long one.
        """
        creds = _write_creds(tmp_path, (time.time() + 4 * 3600) * 1000)
        assert _run(creds, budget=3600).returncode == 0
        assert _run(creds, budget=SIX_HOURS).returncode == 2


class TestItDoesNotFailOpen:
    """Every one of these could plausibly have been written as 'assume fine and continue'."""

    def test_missing_file_is_inconclusive_not_pass(self, tmp_path):
        r = _run(str(tmp_path / "does-not-exist.json"))
        assert r.returncode == 3
        assert "no credentials file" in r.stderr

    def test_corrupt_json_is_inconclusive_not_pass(self, tmp_path):
        p = tmp_path / ".credentials.json"
        p.write_text("{not json at all", encoding="utf-8")
        assert _run(str(p)).returncode == 3

    def test_non_object_json_is_inconclusive_not_pass(self, tmp_path):
        p = tmp_path / ".credentials.json"
        p.write_text('["a list, not an object"]', encoding="utf-8")
        assert _run(str(p)).returncode == 3

    def test_nonsense_expiry_value_is_inconclusive_not_pass(self, tmp_path):
        p = tmp_path / ".credentials.json"
        p.write_text(json.dumps({"claudeAiOauth": {"expiresAt": "tomorrow"}}), encoding="utf-8")
        r = _run(str(p))
        assert r.returncode == 3
        assert "not a usable timestamp" in r.stderr

    def test_absent_expiry_field_passes_with_a_note(self, tmp_path):
        """The one deliberate pass-through, and it is narrow.

        Some credential shapes carry no expiry. Refusing every lane because we do not
        understand a file would be worse than the failure being prevented — but it must say
        so rather than claiming the token was checked.
        """
        p = tmp_path / ".credentials.json"
        p.write_text(json.dumps({"claudeAiOauth": {"accessToken": "x"}}), encoding="utf-8")
        r = _run(str(p))
        assert r.returncode == 0
        assert "no access-token expiry field" in r.stdout


class TestTimestampUnits:
    def test_milliseconds_are_understood(self, tmp_path):
        creds = _write_creds(tmp_path, int((time.time() + 8 * 3600) * 1000))
        assert _run(creds).returncode == 0

    def test_seconds_are_understood(self, tmp_path):
        creds = _write_creds(tmp_path, int(time.time() + 8 * 3600))
        assert _run(creds).returncode == 0

    def test_seconds_and_milliseconds_agree_on_the_verdict(self, tmp_path):
        """The dangerous direction is reading ms as s: an expired token then looks valid
        for millennia and the guard silently passes everything."""
        soon = time.time() + 600  # ten minutes: refused under a 6 h budget either way
        assert _run(_write_creds(tmp_path, int(soon))).returncode == 2
        assert _run(_write_creds(tmp_path, int(soon * 1000))).returncode == 2


class TestRefreshTokenIsReported:
    def test_live_refresh_token_is_mentioned(self, tmp_path):
        creds = _write_creds(tmp_path, (time.time() + 600) * 1000,
                             refresh_at=(time.time() + 20 * 86400) * 1000)
        r = _run(creds)
        assert r.returncode == 2
        assert "refresh token is still valid" in r.stderr

    def test_dead_refresh_token_says_reauthenticate(self, tmp_path):
        creds = _write_creds(tmp_path, (time.time() - 600) * 1000,
                             refresh_at=(time.time() - 86400) * 1000)
        r = _run(creds)
        assert r.returncode == 2
        assert "must re-authenticate" in r.stderr


class TestItIsReadOnly:
    def test_the_credentials_file_is_never_modified(self, tmp_path):
        """It reads the file the whole system authenticates with. It must not touch it."""
        creds = _write_creds(tmp_path, (time.time() + 3.1 * 3600) * 1000,
                             refresh_at=(time.time() + 20 * 86400) * 1000)
        before = hashlib.sha256(open(creds, "rb").read()).hexdigest()
        _run(creds, budget=SIX_HOURS)   # the refusing path
        _run(creds, budget=60)          # the passing path
        after = hashlib.sha256(open(creds, "rb").read()).hexdigest()
        assert before == after


class TestTheSandboxLetsRefreshHappen:
    def test_credentials_are_bound_writable_not_readonly(self):
        """The guard above is the backstop; THIS is the actual fix.

        A --ro-bind here means a session cannot write a refreshed token back, so a lane
        outliving its access token dies mid-run however good the preflight was. Pinned as a
        property of the file because it reads as the safer choice and silently is not.
        """
        src = open(os.path.join(REPO, "benchmark_infra", "lane_sandbox.sh"),
                   encoding="utf-8").read()
        assert '--bind "$HOME/.claude/.credentials.json"' in src
        assert '--ro-bind "$HOME/.claude/.credentials.json"' not in src


class TestTheLauncherUsesIt:
    def test_launcher_runs_the_preflight_per_lane_and_aborts_on_rc2(self):
        src = open(os.path.join(REPO, "run_myagent_headless.sh"), encoding="utf-8").read()
        assert "preflight_oauth_token.py" in src
        # Budget-aware, not a hardcoded window.
        assert '--budget-secs "$PER_COMP_SECS"' in src
        # rc 2 stops the run; rc 3 must not.
        assert 'oauth_rc" = "2"' in src
        assert "the token check is advisory" in src
