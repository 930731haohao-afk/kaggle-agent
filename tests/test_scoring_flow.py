"""End-to-end tests for benchmark_infra/score_lane_submissions.py — the operator scoring step.

Every test drives the real CLI as a subprocess against a fake run root and a fake `kaggle`
binary, because the dominant defect shape in this project is a correct rule wired to
nothing: a gate asserted in a docstring that no code path can actually reach. Each test
pins one gate to observable behavior — the exit code, stdout, the fake binary's call log,
or the append-only scores CSV. The competition slugs are REAL manifest slugs on purpose:
the script reads docs/rerun_manifest.json from the repo it lives in, which a tmp_path
fixture cannot (and must not) fake.
"""
import csv
import hashlib
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "benchmark_infra", "score_lane_submissions.py")

# Real slugs from docs/rerun_manifest.json — see the module docstring for why.
COMP_A = "playground-series-s3e3"
COMP_B = "playground-series-s3e7"

MARKER = "MY-AGENT LANES COMPLETE"
PREFIX = "[my-agent isolated-rerun]"
HEADER = "ref,fileName,date,description,status,publicScore,privateScore\n"

# The canned submit output reproduces the 2026-07-27 incident shape: the CLI's
# carriage-return progress bar is the LAST line, after the success line. Tail-line
# matching marks this as a failure; whole-output matching does not.
SUBMIT_OK = ("Successfully submitted to Playground Series\n"
             " 98%|##########| 4.50M/4.62M [00:01<00:00, 3.2MB/s]\n")


def _subrow(comp, status="COMPLETE", pub="0.71234", priv="0.70987"):
    """One row of `kaggle competitions submissions --csv` carrying this run's message
    prefix. The date suffix is deliberately NOT today's: matching must use the prefix
    alone, so a rerun on a later day still recognises its own earlier submission."""
    return (f'42,submission.csv,2026-01-01 10:00:00,'
            f'"{PREFIX} {comp}, 2026-01-01",{status},{pub},{priv}\n')


def _mk_root(tmp_path, comps, marker=True):
    """A fake finished run root: competitions/<comp>/{submission.csv,data/sample_...}
    plus the sibling <root>.status.md the launcher would have written."""
    root = tmp_path / "run-root"
    for c in comps:
        d = root / "competitions" / c
        (d / "data").mkdir(parents=True, exist_ok=True)
        (d / "data" / "sample_submission.csv").write_text(
            "id,target\n1,0.5\n2,0.5\n3,0.5\n", encoding="utf-8")
        (d / "submission.csv").write_text(
            "id,target\n1,0.11\n2,0.42\n3,0.77\n", encoding="utf-8")
    (tmp_path / "run-root.status.md").write_text(
        f"# smoke\n\n{MARKER}\n" if marker else "# smoke\n\nstill running\n",
        encoding="utf-8")
    return root


def _mk_kaggle(tmp_path, comps, before=None, after=None, submit_out=SUBMIT_OK,
               hang_submit_for=None, hang_seconds=3):
    """A fake `kaggle` binary: logs its argv to calls.log, answers `submissions` with a
    per-comp canned CSV (a different one once `submit` has been called for that comp,
    so a fresh submit becomes visible to the poll loop), and answers `submit` with
    canned output. State is per-competition so multi-comp runs stay independent.

    Besides argv, every call dumps KAGGLE_CONFIG_DIR and KAGGLE_API_TOKEN from its OWN
    environment as `env:`-prefixed lines — the only way a test can see what the child
    process actually received, which is where the wrong-account guard lives. The
    prefix keeps env lines distinguishable from argv lines: the token is SUPPOSED to
    be in the environment and in nothing else.

    hang_submit_for=<comp> makes that one comp's `submit` sleep hang_seconds (stdout
    detached, so a killed parent is not held open by the orphan) — a hung CLI for the
    timeout tests."""
    before, after = before or {}, after or {}
    kdir = tmp_path / "fakekaggle"
    kdir.mkdir(exist_ok=True)
    log = kdir / "calls.log"
    for c in comps:
        (kdir / f"before_{c}.csv").write_text(before.get(c, HEADER), encoding="utf-8")
        (kdir / f"after_{c}.csv").write_text(
            after.get(c, HEADER + _subrow(c)), encoding="utf-8")
    (kdir / "submit_out.txt").write_text(submit_out, encoding="utf-8")
    hang = (f'if [ "$4" = "{hang_submit_for}" ]; then sleep {hang_seconds} '
            f'> /dev/null 2>&1; fi') if hang_submit_for else ":"
    bin_ = kdir / "kaggle"
    bin_.write_text(f"""#!/bin/sh
echo "$@" >> "{log}"
echo "env:KAGGLE_CONFIG_DIR=$KAGGLE_CONFIG_DIR" >> "{log}"
echo "env:KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN" >> "{log}"
case "$2" in
  submissions)
    if [ -f "{kdir}/submitted_$4" ]; then cat "{kdir}/after_$4.csv"
    else cat "{kdir}/before_$4.csv"; fi
    ;;
  submit)
    {hang}
    : > "{kdir}/submitted_$4"
    cat "{kdir}/submit_out.txt"
    ;;
esac
exit 0
""", encoding="utf-8")
    bin_.chmod(0o755)
    return str(bin_), log


def _mk_token(tmp_path, value="KGAT_faketoken_abc123"):
    t = tmp_path / "huang_token"
    t.write_text(value + "\n", encoding="utf-8")
    return str(t)


def _args(tmp_path, root, kbin, token, comps, extra=()):
    a = ["--root", str(root), "--kaggle-bin", kbin, "--token-file", token,
         "--sleep", "0",
         "--out", str(tmp_path / "scores.csv"),
         "--status-log", str(tmp_path / "scoring.md")]
    for c in comps:
        a += ["--comp", c]
    return a + list(extra)


def _run(args, env=None):
    """env, when given, is overlaid on os.environ — used to shrink the script's
    documented SCORE_*_TIMEOUT caps, which a monkeypatch cannot reach across the
    process boundary."""
    return subprocess.run([sys.executable, SCRIPT, *args],
                          capture_output=True, text=True, check=False,
                          env={**os.environ, **env} if env else None)


def _rows(tmp_path):
    with open(tmp_path / "scores.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class TestDryRunIsTheDefault:
    def test_without_yes_submit_no_kaggle_call_no_credential_read_no_file_written(self, tmp_path):
        """The default invocation must be incapable of submitting. The token file here
        does not even exist: if the dry run so much as opened it, the run would fail,
        so rc 0 is the proof that no credential was read. The fake binary's call log
        staying absent is the proof that no kaggle CLI call happened."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, str(tmp_path / "no-such-token"), [COMP_A]))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "DRY RUN — nothing was submitted" in r.stdout, r.stdout
        assert not log.exists(), "dry run called the fake kaggle binary: " + log.read_text()
        assert not (tmp_path / "scores.csv").exists(), "dry run wrote the scores CSV"
        assert not (tmp_path / "scoring.md").exists(), "dry run wrote the status log"

    def test_dry_run_prints_the_sha256_of_what_would_be_submitted(self, tmp_path):
        """The dry run's product is its stdout, and the sha256 is the part that lets the
        operator tie the eventual score row back to the exact file previewed here."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, str(tmp_path / "no-such-token"), [COMP_A]))
        expected = _sha(root / "competitions" / COMP_A / "submission.csv")
        assert expected in r.stdout, r.stdout
        assert f"{PREFIX} {COMP_A}" in r.stdout, r.stdout

    def test_dry_run_with_a_missing_submission_exits_partial(self, tmp_path):
        """'0 = dry-run fully clean' — a dry run over a root with a hole must say so in
        its exit code, or the operator reads the preview as a green light."""
        root = _mk_root(tmp_path, [COMP_A])
        os.remove(root / "competitions" / COMP_A / "submission.csv")
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, str(tmp_path / "no-such-token"), [COMP_A]))
        assert r.returncode == 3, r.stdout + r.stderr
        assert "missing: 1" in r.stdout, r.stdout


class TestCompletionMarkerGate:
    def test_missing_marker_refuses_before_any_validation(self, tmp_path):
        """Scoring a run the audit has not passed launders an invalid run into a results
        table, so the refusal must come BEFORE any per-comp work: no validator run (its
        'Validating' banner absent), no kaggle call, no CSV row."""
        root = _mk_root(tmp_path, [COMP_A], marker=False)
        kbin, log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 2, r.stdout + r.stderr
        assert "completion marker" in r.stderr, r.stderr
        assert "Validating" not in r.stdout, "the validator ran before the marker gate"
        assert not log.exists(), "the marker gate did not stop the kaggle CLI"
        assert not (tmp_path / "scores.csv").exists()

    def test_allow_incomplete_lifts_the_marker_gate(self, tmp_path):
        """--allow-incomplete exists so a smoke root can be scored; with it, the same
        unmarked root goes all the way through to a scored row."""
        root = _mk_root(tmp_path, [COMP_A], marker=False)
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit", "--allow-incomplete"]))
        assert r.returncode == 0, r.stdout + r.stderr
        rows = _rows(tmp_path)
        assert len(rows) == 1 and rows[0]["status"] == "scored", rows

    def test_a_slug_not_in_the_manifest_is_refused(self, tmp_path):
        """Slugs come only from docs/rerun_manifest.json; a typo'd --comp must fail
        loudly rather than silently score nothing."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path),
                       ["not-a-real-competition"]))
        assert r.returncode == 2, r.stdout + r.stderr
        assert "not-a-real-competition" in r.stderr, r.stderr


class TestMissingSubmission:
    def test_missing_file_is_recorded_and_the_other_comps_still_run(self, tmp_path):
        """One lane leaving no submission.csv must not abort the scoring of the other
        nineteen: the missing comp gets a 'missing' row, the rest proceed, and the exit
        code says partial (3)."""
        root = _mk_root(tmp_path, [COMP_A, COMP_B])
        os.remove(root / "competitions" / COMP_A / "submission.csv")
        kbin, log = _mk_kaggle(tmp_path, [COMP_A, COMP_B])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A, COMP_B],
                       extra=["--yes-submit"]))
        assert r.returncode == 3, r.stdout + r.stderr
        by_comp = {row["comp"]: row for row in _rows(tmp_path)}
        assert by_comp[COMP_A]["status"] == "missing", by_comp
        assert by_comp[COMP_B]["status"] == "scored", by_comp
        assert "missing: 1" in r.stdout and "scored: 1" in r.stdout, r.stdout
        submit_lines = [ln for ln in log.read_text().splitlines()
                        if ln.startswith("competitions submit ")]
        assert len(submit_lines) == 1 and COMP_B in submit_lines[0], submit_lines


class TestValidatorGate:
    def test_structural_failure_blocks_the_submit_for_that_comp_only(self, tmp_path):
        """A malformed file burns a daily submission; the validator (rc 1, structural)
        must stop the upload before it starts. The fake binary's log is the evidence:
        no submit call for the broken comp, exactly one for the good one."""
        root = _mk_root(tmp_path, [COMP_A, COMP_B])
        (root / "competitions" / COMP_A / "submission.csv").write_text(
            "foo,bar\n1,2\n", encoding="utf-8")
        kbin, log = _mk_kaggle(tmp_path, [COMP_A, COMP_B])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A, COMP_B],
                       extra=["--yes-submit"]))
        assert r.returncode == 3, r.stdout + r.stderr
        by_comp = {row["comp"]: row for row in _rows(tmp_path)}
        assert by_comp[COMP_A]["status"] == "validation-blocked", by_comp
        assert by_comp[COMP_B]["status"] == "scored", by_comp
        submit_lines = [ln for ln in log.read_text().splitlines()
                        if ln.startswith("competitions submit ")]
        assert len(submit_lines) == 1, submit_lines
        assert COMP_A not in submit_lines[0] and COMP_B in submit_lines[0], submit_lines
        assert "blocked: 1" in r.stdout, r.stdout

    def test_a_suspicious_submission_is_blocked_without_a_waiver(self, tmp_path):
        """Constant predictions are a legal CSV that trips the validator's SUSPICIOUS
        tier (rc 2, almost certainly a bug). Without --allow-suspicious the submit
        gate must treat rc 2 exactly like rc 1 — no upload, a validation-blocked
        row — because a gate written 'if vrc == 1' waves every suspicious file
        straight through to a spent daily submission."""
        root = _mk_root(tmp_path, [COMP_A])
        (root / "competitions" / COMP_A / "submission.csv").write_text(
            "id,target\n1,0.5\n2,0.5\n3,0.5\n", encoding="utf-8")
        kbin, log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 3, r.stdout + r.stderr
        rows = _rows(tmp_path)
        assert rows[0]["status"] == "validation-blocked", rows
        assert "suspicious, not waived" in rows[0]["note"], rows
        submit_lines = [ln for ln in (log.read_text().splitlines() if log.exists() else [])
                        if ln.startswith("competitions submit ")]
        assert submit_lines == [], "a suspicious file was submitted: " + str(submit_lines)

    def test_allow_suspicious_submits_and_the_row_records_the_waiver(self, tmp_path):
        """--allow-suspicious REASON is the operator's own written waiver, passed
        through to the validator verbatim; with it the same constant file submits,
        and the CSV note must carry the reason so the results table shows the score
        was taken under waiver rather than through a clean validation."""
        reason = "constant baseline is the intended submission"
        root = _mk_root(tmp_path, [COMP_A])
        (root / "competitions" / COMP_A / "submission.csv").write_text(
            "id,target\n1,0.5\n2,0.5\n3,0.5\n", encoding="utf-8")
        kbin, log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit", "--allow-suspicious", reason]))
        assert r.returncode == 0, r.stdout + r.stderr
        rows = _rows(tmp_path)
        assert rows[0]["status"] == "scored", rows
        assert f"suspicious waived: {reason}" in rows[0]["note"], rows
        submit_lines = [ln for ln in log.read_text().splitlines()
                        if ln.startswith("competitions submit ")]
        assert len(submit_lines) == 1, submit_lines


class TestIdempotence:
    def test_an_existing_prefix_row_skips_the_submit_and_records_its_scores(self, tmp_path):
        """Rerunning after a crash must not spend quota on a duplicate. The pre-submit
        listing already holds a row with this run's message prefix (dated a different
        day — matching is prefix-only), so no submit happens and that row's scores are
        what gets recorded."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, log = _mk_kaggle(tmp_path, [COMP_A],
                               before={COMP_A: HEADER + _subrow(COMP_A)})
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 0, r.stdout + r.stderr
        submit_lines = [ln for ln in log.read_text().splitlines()
                        if ln.startswith("competitions submit ")]
        assert submit_lines == [], "a duplicate submit happened: " + str(submit_lines)
        rows = _rows(tmp_path)
        assert rows[0]["status"] == "already-submitted", rows
        assert rows[0]["public"] == "0.71234" and rows[0]["private"] == "0.70987", rows
        assert "already: 1" in r.stdout, r.stdout

    def test_an_already_submitted_row_does_not_attest_the_current_files_hash(self, tmp_path):
        """The scores on an already-submitted row were earned by an EARLIER upload,
        and the submission.csv on disk may have been regenerated since. Writing the
        CURRENT file's sha256 next to those scores fabricates exactly the provenance
        link the hash exists to guarantee: here the file changes between the two runs,
        and the second row must leave the sha empty and say why."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A])
        args = _args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                     extra=["--yes-submit"])
        r1 = _run(args)
        assert r1.returncode == 0, r1.stdout + r1.stderr
        sub = root / "competitions" / COMP_A / "submission.csv"
        sub.write_text("id,target\n1,0.93\n2,0.08\n3,0.51\n", encoding="utf-8")
        r2 = _run(args)
        assert r2.returncode == 0, r2.stdout + r2.stderr
        rows = _rows(tmp_path)
        assert rows[1]["status"] == "already-submitted", rows
        assert rows[1]["sha256"] == "", \
            "the rerun attested the current file's hash against an earlier upload's score"
        assert "upload predates this run; local file hash not attested" in rows[1]["note"], rows
        assert rows[0]["sha256"] != "", rows


class TestHungOrDeadCli:
    def test_a_submit_that_hangs_past_the_cap_costs_only_its_own_comp(self, tmp_path):
        """An uncaught subprocess.TimeoutExpired used to abort the whole loop with a
        traceback and rc 1 — outside the 0/2/3 exit contract — leaving every later
        comp unexamined and no row recorded. A hung kaggle submit must instead be
        caught: its comp gets a submit-failed row and the loop continues. The caps
        are the module's SCORE_*_TIMEOUT constants, env-overridable by design so this
        test can shrink them below the fake binary's sleep."""
        root = _mk_root(tmp_path, [COMP_A, COMP_B])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A, COMP_B], hang_submit_for=COMP_A)
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A, COMP_B],
                       extra=["--yes-submit"]),
                 env={"SCORE_SUBMIT_TIMEOUT": "1"})
        assert r.returncode == 3, r.stdout + r.stderr
        assert "Traceback" not in r.stderr, r.stderr
        by_comp = {row["comp"]: row for row in _rows(tmp_path)}
        assert by_comp[COMP_A]["status"] == "submit-failed", by_comp
        assert "TimeoutExpired" in by_comp[COMP_A]["note"], by_comp
        assert by_comp[COMP_B]["status"] == "scored", by_comp
        assert "failed: 1" in r.stdout and "scored: 1" in r.stdout, r.stdout

    def test_an_unreadable_token_file_refuses_with_rc_2_not_a_traceback(self, tmp_path):
        """A token file that exists but cannot be opened used to escape as an uncaught
        OSError traceback (rc 1). Precondition reads must refuse like every other
        precondition: rc 2 and one plain sentence, before any kaggle call."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, log = _mk_kaggle(tmp_path, [COMP_A])
        token = _mk_token(tmp_path)
        os.chmod(token, 0o000)
        r = _run(_args(tmp_path, root, kbin, token, [COMP_A], extra=["--yes-submit"]))
        os.chmod(token, 0o600)
        assert r.returncode == 2, r.stdout + r.stderr
        assert "Traceback" not in r.stderr, r.stderr
        assert "REFUSED" in r.stderr, r.stderr
        assert not log.exists() and not (tmp_path / "scores.csv").exists()


class TestSubmitDetection:
    def test_success_is_read_from_the_whole_output_not_the_tail_line(self, tmp_path):
        """The CLI's progress bar is the LAST line of its output; on 2026-07-27
        tail-line matching marked every successful upload as failed. The canned output
        here puts the progress bar after 'Successfully submitted' — it must count as
        a success and go on to a scored row."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, log = _mk_kaggle(tmp_path, [COMP_A], submit_out=SUBMIT_OK)
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "SUBMIT FAILED" not in r.stdout, r.stdout
        rows = _rows(tmp_path)
        assert rows[0]["status"] == "scored", rows
        submit_lines = [ln for ln in log.read_text().splitlines()
                        if ln.startswith("competitions submit ")]
        assert len(submit_lines) == 1, submit_lines

    def test_output_without_a_success_line_is_a_submit_failure(self, tmp_path):
        """The mirror case: no 'Successfully' anywhere means the upload did not happen,
        whatever else the CLI printed, and the comp must be recorded submit-failed."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A],
                                submit_out="403 Forbidden - permission denied\n")
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 3, r.stdout + r.stderr
        rows = _rows(tmp_path)
        assert rows[0]["status"] == "submit-failed", rows
        assert "failed: 1" in r.stdout, r.stdout


class TestScoreRecord:
    def test_the_row_carries_the_exact_files_sha256_and_the_canned_scores(self, tmp_path):
        """The sha256 ties the score to the exact bytes uploaded, so a later regenerated
        submission.csv cannot silently claim an old score; public/private must be the
        values Kaggle reported, verbatim."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 0, r.stdout + r.stderr
        row = _rows(tmp_path)[0]
        assert row["sha256"] == _sha(root / "competitions" / COMP_A / "submission.csv"), row
        assert row["public"] == "0.71234" and row["private"] == "0.70987", row
        assert row["message"].startswith(f"{PREFIX} {COMP_A}"), row
        assert row["comp"] == COMP_A and row["status"] == "scored", row

    def test_an_unpublished_private_score_is_recorded_empty_never_fabricated(self, tmp_path):
        """If Kaggle has not published the private score, the cell stays empty with the
        'needs a human look' note — a guessed number would corrupt the results table."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A],
                                after={COMP_A: HEADER + _subrow(COMP_A, priv="")})
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 0, r.stdout + r.stderr
        row = _rows(tmp_path)[0]
        assert row["status"] == "scored" and row["private"] == "", row
        assert "private pending — needs a human look" in row["note"], row

    def test_a_literal_None_private_score_is_treated_as_empty_not_a_number(self, tmp_path):
        """The Kaggle CLI prints the literal string 'None' (and pandas-shaped 'nan')
        in a score cell it has not published — record_s6e7_private.sh already treats
        "None" as absent. Recorded verbatim it poisons the results table with a
        non-number that later tooling will try to float(); it must be treated exactly
        like an empty cell: private recorded empty, with the pending note."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A],
                                after={COMP_A: HEADER + _subrow(COMP_A, priv="None")})
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 0, r.stdout + r.stderr
        row = _rows(tmp_path)[0]
        assert row["status"] == "scored", row
        assert row["private"] == "", "the literal string 'None' was recorded as a score: " \
            + str(row)
        assert "private pending — needs a human look" in row["note"], row

    def test_a_pre_existing_empty_out_file_still_gets_the_header(self, tmp_path):
        """An operator pre-creating the out file (`touch scores.csv`) used to make the
        existence check skip the header, so every row landed headerless and the whole
        CSV parsed with the first data row as its column names. Missing OR zero-size
        must both mean 'write the header first'."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A])
        (tmp_path / "scores.csv").write_text("", encoding="utf-8")
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 0, r.stdout + r.stderr
        rows = _rows(tmp_path)
        assert len(rows) == 1, "the header row is missing — the first data row was " \
            "consumed as column names: " + str(rows)
        assert rows[0].get("comp") == COMP_A and rows[0].get("status") == "scored", rows

    def test_a_second_run_appends_rows_and_never_clobbers_the_first(self, tmp_path):
        """Both output files are append-only. The rerun sees its own earlier submission
        (idempotence), adds an 'already-submitted' row, and the first run's 'scored'
        row must survive byte-for-byte in place — and the status log after run 1 must
        be a byte-for-byte prefix of the log after run 2, or the 'earlier lines are
        the record of earlier invocations' promise is a docstring wired to nothing."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A])
        args = _args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                     extra=["--yes-submit"])
        r1 = _run(args)
        assert r1.returncode == 0, r1.stdout + r1.stderr
        first = _rows(tmp_path)[0]
        md_after_first = (tmp_path / "scoring.md").read_text(encoding="utf-8")
        r2 = _run(args)
        assert r2.returncode == 0, r2.stdout + r2.stderr
        rows = _rows(tmp_path)
        assert len(rows) == 2, rows
        assert rows[0] == first, "the rerun rewrote the first run's row"
        assert rows[1]["status"] == "already-submitted", rows
        md_after_second = (tmp_path / "scoring.md").read_text(encoding="utf-8")
        assert md_after_second.startswith(md_after_first), \
            "the rerun truncated or rewrote the status log"


class TestCredentialGate:
    def test_a_wrong_prefix_token_is_refused_and_its_value_never_echoed(self, tmp_path):
        """Whatever is in the token file is secret-shaped and this script's stdout ends
        up in logs: the refusal must name the file, never the value, and nothing may
        have been attempted (rc 2, empty kaggle log, no CSV)."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, log = _mk_kaggle(tmp_path, [COMP_A])
        token = _mk_token(tmp_path, value="WRONG_secretvalue12345")
        r = _run(_args(tmp_path, root, kbin, token, [COMP_A], extra=["--yes-submit"]))
        assert r.returncode == 2, r.stdout + r.stderr
        assert "secretvalue12345" not in r.stdout + r.stderr, "the token value leaked"
        assert "KGAT_" in r.stderr, r.stderr
        assert not log.exists() and not (tmp_path / "scores.csv").exists()


class TestWrongAccountGuard:
    def test_the_cli_child_env_carries_the_token_and_an_empty_config_dir(self, tmp_path):
        """The CLI must authenticate from the fixture token ALONE. With
        KAGGLE_CONFIG_DIR absent from the child environment the CLI silently falls
        back to ~/.kaggle/kaggle.json — the WRONG account (tjyen1975) — so the fake
        binary dumps both variables from its own environment: the config dir must be
        set, exist, and be EMPTY, and the token must be exactly the fixture value."""
        root = _mk_root(tmp_path, [COMP_A])
        kbin, log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A],
                       extra=["--yes-submit"]))
        assert r.returncode == 0, r.stdout + r.stderr
        lines = log.read_text().splitlines()
        conf_dirs = {ln.split("=", 1)[1] for ln in lines
                     if ln.startswith("env:KAGGLE_CONFIG_DIR=")}
        tokens = {ln.split("=", 1)[1] for ln in lines
                  if ln.startswith("env:KAGGLE_API_TOKEN=")}
        assert conf_dirs and tokens, "the fake binary saw no env dump lines: " + str(lines)
        assert tokens == {"KGAT_faketoken_abc123"}, \
            "the child env token is not the token file's value: " + str(tokens)
        conf_dir = conf_dirs.pop()
        assert not conf_dirs, "KAGGLE_CONFIG_DIR changed between calls"
        assert conf_dir, "KAGGLE_CONFIG_DIR was missing from the child env — the CLI " \
            "falls back to ~/.kaggle/kaggle.json (the wrong account)"
        assert os.path.isdir(conf_dir), conf_dir
        assert os.listdir(conf_dir) == [], \
            "the config dir is not empty — a config file could override the token"


class TestTokenHygieneOnTheAcceptedPath:
    def test_the_token_value_reaches_only_the_child_environment(self, tmp_path):
        """The token travels ONLY inside the child environment. After a fully
        successful --yes-submit run — the path where the token is actually used —
        its value must appear in none of: the script's stdout, the scores CSV, the
        status log, or the fake binary's argv lines. The env: lines are the one
        deliberate exception; that is the sanctioned channel."""
        token_value = "KGAT_faketoken_abc123"
        root = _mk_root(tmp_path, [COMP_A])
        kbin, log = _mk_kaggle(tmp_path, [COMP_A])
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path, value=token_value),
                       [COMP_A], extra=["--yes-submit"]))
        assert r.returncode == 0, r.stdout + r.stderr
        assert token_value not in r.stdout + r.stderr, "the token leaked to stdout/stderr"
        assert token_value not in (tmp_path / "scores.csv").read_text(encoding="utf-8"), \
            "the token leaked into the scores CSV"
        assert token_value not in (tmp_path / "scoring.md").read_text(encoding="utf-8"), \
            "the token leaked into the status log"
        argv_lines = [ln for ln in log.read_text().splitlines()
                      if not ln.startswith("env:")]
        assert all(token_value not in ln for ln in argv_lines), \
            "the token appeared on the kaggle CLI's command line: " + str(argv_lines)


class TestSleepIsRespected:
    def test_sleep_zero_makes_a_two_comp_run_fast(self, tmp_path):
        """--sleep 0 must reach both the between-submits pause and the poll pause: a
        hardcoded 30 s anywhere makes this two-comp run blow the bound."""
        root = _mk_root(tmp_path, [COMP_A, COMP_B])
        kbin, _log = _mk_kaggle(tmp_path, [COMP_A, COMP_B])
        t0 = time.monotonic()
        r = _run(_args(tmp_path, root, kbin, _mk_token(tmp_path), [COMP_A, COMP_B],
                       extra=["--yes-submit"]))
        elapsed = time.monotonic() - t0
        assert r.returncode == 0, r.stdout + r.stderr
        assert "scored: 2" in r.stdout, r.stdout
        assert elapsed < 30, f"--sleep 0 was not respected: {elapsed:.1f}s for two comps"
