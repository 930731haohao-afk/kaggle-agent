#!/usr/bin/env python3
"""Submit each lane's submission.csv and record its scores — the operator half of the credential-free run root.

WHY THIS EXISTS. The run root ships no Kaggle credentials and no submit routes, on purpose:
build_myagent_run_root.py strips them because `kaggle competitions submissions -c <comp>`
returns that competition's own prior leaderboard scores — the exact answer the isolation
protocol exists to withhold from a lane (round 10). The run root's CLAUDE.md therefore ends
with "the operator scores it afterwards". This file is that afterwards: it runs OUTSIDE the
sandbox, with the operator's own token, against a finished root, and folds the public and
private scores into an append-only CSV the report can cite.

WHY IT CANNOT FIRE BY ACCIDENT. A submission spends one of the day's quota and stamps the
account's history, so every path to an upload is gated:

  dry run      Without --yes-submit the script is purely local: it reads no credential,
               invokes the kaggle CLI zero times, and only validates each submission.csv and
               prints what WOULD be submitted (path, sha256, message). The default invocation
               therefore cannot submit anything, ever.
  the marker   It refuses (rc 2) a root whose <root>.status.md does not carry the literal
               completion marker the launcher writes only after the run-level audit. Scoring
               a run the audit has not passed launders an invalid run into a results table;
               --allow-incomplete exists solely so a smoke root can be scored, and says so.
  idempotence  Before submitting it lists the competition's existing submissions; any row
               whose description starts with this run's message prefix means the upload
               already happened (a previous, possibly crashed, invocation), so the submit is
               skipped and that row's scores are recorded instead. Rerunning after a partial
               failure is safe and costs no quota.

WHAT IT DOES NOT DO. It renders no verdict, computes no percentile, edits no report, and
never merges into the three-way score table — it records, one row per competition, and
stops. It also never fabricates: a private score Kaggle has not published yet is recorded
empty with the note "private pending — needs a human look", not guessed.

CREDENTIAL HYGIENE. The token is read from --token-file, whitespace-stripped, and refused
unless it starts with "KGAT_". It travels only inside the child environment, never in argv,
stdout, the status log or the CSV. KAGGLE_CONFIG_DIR is pointed at a fresh EMPTY temp
directory so the CLI cannot silently fall back to ~/.kaggle/kaggle.json, which belongs to
the WRONG account (tjyen1975); the benchmark account is huangweihaohuang.

LIVENESS IS REPORTED, NOT ASSUMED. Every count in the end summary is printed, including
zero. A slug not in docs/rerun_manifest.json, or an invocation that would examine zero
competitions, is a loud rc-2 failure — a scorer that scored nothing must never look like a
scorer that found nothing wrong.

Usage:
    score_lane_submissions.py                                        # dry run, all 20 lanes
    score_lane_submissions.py --comp playground-series-s3e3          # dry run, one lane
    score_lane_submissions.py --yes-submit                           # the real thing
    score_lane_submissions.py --root <smoke-root> --allow-incomplete --yes-submit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Slugs come ONLY from the manifest. A hand-kept list here is the drift the launcher already
# paid for once: the round-9 pinned launcher listed 5 of the 20 competitions and then logged
# the completion marker anyway.
MANIFEST = os.path.join(REPO, "docs", "rerun_manifest.json")
VALIDATOR = os.path.join(REPO, ".claude", "skills", "kaggle-safe-submit", "scripts",
                         "validate_submission.py")

# The literal string run_myagent_headless.sh appends only after the run-level
# --require-all transcript audit has passed. Its absence means the run is unfinished or
# unaudited, and both mean: do not score.
COMPLETION_MARKER = "MY-AGENT LANES COMPLETE"

# Message identity for this run. Matching uses ONLY this prefix plus the competition slug;
# the ", <YYYY-MM-DD>" suffix on the submitted message is informational, so a rerun on a
# later day still recognises its own earlier submission.
MSG_PREFIX = "[my-agent isolated-rerun]"

POLL_TRIES = 20
# Time caps for every child process, in seconds. Env-overridable BY DESIGN, read once at
# import: the test suite drives this script as a subprocess, so no monkeypatch can reach
# these, and pinning "a hung CLI does not abort the run" under a 600 s cap would make the
# pin unrunnable. Operators never need to set them; the defaults are the contract.
FETCH_TIMEOUT = float(os.environ.get("SCORE_FETCH_TIMEOUT", "120"))
SUBMIT_TIMEOUT = float(os.environ.get("SCORE_SUBMIT_TIMEOUT", "600"))
VALIDATE_TIMEOUT = float(os.environ.get("SCORE_VALIDATE_TIMEOUT", "600"))

# `kaggle competitions submissions --csv` header, verbatim; kept as a constant so the row
# parsing below is checked against something written down rather than remembered.
SUBMISSIONS_HEADER = "ref,fileName,date,description,status,publicScore,privateScore"

CSV_COLUMNS = ("comp", "file", "sha256", "message", "submitted_at", "status",
               "public", "private", "note")
STATUSES = ("scored", "already-submitted", "missing", "validation-blocked",
            "submit-failed", "score-timeout")

DEFAULT_ROOT = os.path.expanduser("~/benchruns/myagent-rerun")
DEFAULT_KAGGLE = "/home/tjyen/ai_agents/mle-bench/.venv/bin/kaggle"
DEFAULT_TOKEN = os.path.expanduser("~/.kaggle/huang_token")


def _sha256(path: str) -> str:
    """Digest of the exact bytes uploaded, so the score row is tied to one specific file
    and a later re-generated submission.csv cannot silently claim an old score."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _log(status_log: str, msg: str) -> None:
    """House status-log line, printed and appended. Append-only: earlier lines are the
    record of earlier invocations and must survive a rerun."""
    line = f"- `{time.strftime('%m-%d %H:%M')}` {msg}"
    print(line, flush=True)
    with open(status_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _record(out_csv: str, row: dict) -> None:
    """Append one score row. Append-only for the same reason as the status log: a rerun
    adds rows, it never rewrites history."""
    # Missing OR zero-size: a pre-created empty file (an operator's `touch`) has no
    # header either, and headerless rows make the first data row parse as column names.
    is_new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        if is_new:
            w.writeheader()
        w.writerow(row)


def _fetch_rows(kaggle_bin: str, comp: str, env: dict) -> list[dict]:
    """This competition's submission history, parsed. Stale CLI builds print a version
    warning BEFORE the CSV header, so parsing starts at the header line rather than at
    byte zero; no header found means no rows, and the caller treats that as 'no match'."""
    out = subprocess.run([kaggle_bin, "competitions", "submissions", "-c", comp, "--csv"],
                         capture_output=True, text=True, env=env,
                         timeout=FETCH_TIMEOUT, check=False)
    lines = out.stdout.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(SUBMISSIONS_HEADER.split(",")[0] + ","):
            return list(csv.DictReader(io.StringIO("\n".join(lines[i:]))))
    return []


def _score(value: str | None) -> str:
    """A score cell as the CLI printed it, normalised: the literal strings 'None' and
    'nan' (any case) are what the CLI prints for a score Kaggle has not published —
    record_s6e7_private.sh already treats "None" as absent — and must be handled like
    an empty cell everywhere, never recorded as if they were numbers."""
    v = (value or "").strip()
    return "" if v.lower() in ("none", "nan") else v


def _match(rows: list[dict], prefix: str) -> dict | None:
    """The row belonging to this run, by description prefix ONLY. fileName is
    'submission.csv' on nearly every upload these pipelines have ever made, and the date
    column couples the match to when the operator happened to run this — neither
    identifies the run. The prefix names the run and the competition, which does."""
    for r in rows:
        if (r.get("description") or "").startswith(prefix):
            return r
    return None


def _validate(sub_path: str, sample_path: str, allow_suspicious: str | None) -> tuple[int, str]:
    """Run the skill's validator exactly as the skill does: rc 0 pass, rc 1 structural
    (never overridable), rc 2 suspicious (waivable only by the operator's own written
    reason, passed through verbatim — this script never invents a waiver)."""
    cmd = [sys.executable, VALIDATOR, sub_path, sample_path]
    if allow_suspicious:
        cmd += ["--allow-suspicious", allow_suspicious]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=VALIDATE_TIMEOUT,
                         check=False)
    return out.returncode, out.stdout + out.stderr


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="the finished run root to score")
    ap.add_argument("--comp", action="append",
                    help="score only this competition (repeatable); must be a manifest slug")
    ap.add_argument("--yes-submit", action="store_true",
                    help="actually submit. WITHOUT this flag the script is a dry run: no "
                         "kaggle CLI call, no credential read, nothing written")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="skip the completion-marker precondition — for smoke roots only")
    ap.add_argument("--allow-suspicious", metavar="REASON",
                    help="pass the operator's waiver reason through to the validator's "
                         "SUSPICIOUS tier; never applied on the script's own initiative")
    ap.add_argument("--kaggle-bin", default=DEFAULT_KAGGLE)
    ap.add_argument("--token-file", default=DEFAULT_TOKEN)
    ap.add_argument("--out", help="score CSV (default <root>.scores.csv)")
    ap.add_argument("--status-log", help="human log (default <root>.scoring.md)")
    ap.add_argument("--sleep", type=float, default=30.0,
                    help="pause between submits and between score polls (tests pass 0)")
    a = ap.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(a.root))

    # Precondition reads refuse like every other precondition: rc 2 and one plain
    # sentence, never an uncaught traceback outside the 0/2/3 exit contract.
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            slugs = sorted(json.load(f)["competitions"])
    except (OSError, json.JSONDecodeError) as e:
        print(f"REFUSED — could not read the manifest {MANIFEST}: {e}.", file=sys.stderr)
        return 2
    if a.comp:
        unknown = sorted(set(a.comp) - set(slugs))
        if unknown:
            print(f"REFUSED — not in docs/rerun_manifest.json: {unknown}. The manifest is "
                  f"the only source of slugs; a typo here must fail, not silently score "
                  f"nothing.", file=sys.stderr)
            return 2
        comps = [s for s in slugs if s in set(a.comp)]
    else:
        comps = slugs
    if not comps:
        print("REFUSED — zero competitions to score. A scorer that examined nothing must "
              "not exit like one that found nothing wrong.", file=sys.stderr)
        return 2

    # ---- preconditions, before anything else -------------------------------------------
    if not os.path.isdir(root):
        print(f"REFUSED — run root does not exist: {root}", file=sys.stderr)
        return 2
    status_md = root + ".status.md"
    if not a.allow_incomplete:
        try:
            marker_ok = os.path.isfile(status_md) and \
                COMPLETION_MARKER in open(status_md, encoding="utf-8").read()
        except OSError as e:
            print(f"REFUSED — could not read {status_md}: {e}.", file=sys.stderr)
            return 2
        if not marker_ok:
            print(f"REFUSED — {status_md} does not carry the completion marker "
                  f"{COMPLETION_MARKER!r}. Scoring a run whose audit has not passed "
                  f"launders an invalid run into a results table; --allow-incomplete "
                  f"exists for smoke roots only.", file=sys.stderr)
            return 2

    out_csv = a.out or root + ".scores.csv"
    status_log = a.status_log or root + ".scoring.md"
    today = time.strftime("%Y-%m-%d", time.gmtime())

    # ---- dry run: the default, and deliberately incapable of submitting ----------------
    # Purely local. No kaggle CLI call, no credential read, and nothing written to disk —
    # the dry run's whole product is its stdout, so running it twice out of curiosity
    # leaves no trace in the append-only records the real run keeps.
    if not a.yes_submit:
        n_would, n_missing, n_blocked = 0, 0, 0
        for comp in comps:
            sub = os.path.join(root, "competitions", comp, "submission.csv")
            sample = os.path.join(root, "competitions", comp, "data", "sample_submission.csv")
            if not os.path.isfile(sub):
                n_missing += 1
                print(f"{comp}: MISSING — no submission.csv at {sub}")
                continue
            try:
                vrc, vout = _validate(sub, sample, a.allow_suspicious)
            except (subprocess.TimeoutExpired, OSError) as e:
                n_blocked += 1
                print(f"{comp}: BLOCKED — validator did not run "
                      f"({type(e).__name__}); would NOT be submitted")
                continue
            if vrc != 0:
                n_blocked += 1
                tier = "structural" if vrc == 1 else \
                    "suspicious, not waived" if vrc == 2 else f"validator rc {vrc}"
                print(f"{comp}: BLOCKED — {tier}; would NOT be submitted")
                print(vout.rstrip())
                continue
            n_would += 1
            print(f"{comp}: would submit {sub}")
            print(f"    sha256  {_sha256(sub)}")
            print(f"    message {MSG_PREFIX} {comp}, {today}")
        print(f"\nwould submit: {n_would}")
        print("DRY RUN — nothing was submitted (no kaggle call, no credential read; "
              "pass --yes-submit to score for real)")
        print(f"comps: {len(comps)}  scored: 0  already: 0  missing: {n_missing}  "
              f"blocked: {n_blocked}  failed: 0")
        return 0 if (n_missing == 0 and n_blocked == 0) else 3

    # ---- credentials: read only now, after every refusal that makes them unnecessary ---
    if not os.path.isfile(a.token_file):
        print(f"REFUSED — token file missing: {a.token_file}", file=sys.stderr)
        return 2
    try:
        token = "".join(open(a.token_file, encoding="utf-8").read().split())
    except OSError as e:
        print(f"REFUSED — could not read the token file {a.token_file}: {e}.",
              file=sys.stderr)
        return 2
    if not token.startswith("KGAT_"):
        # The offending value is deliberately NOT echoed: whatever is in that file, it is
        # secret-shaped, and this script's stdout ends up in logs.
        print(f"REFUSED — {a.token_file} does not hold a KGAT_ token (its value is not "
              f"echoed on purpose). The benchmark account's token lives there; check the "
              f"file, do not paste the token on a command line.", file=sys.stderr)
        return 2
    # A fresh EMPTY config dir: with KAGGLE_CONFIG_DIR unset the CLI falls back to
    # ~/.kaggle/kaggle.json, which authenticates the WRONG account (tjyen1975). The
    # benchmark account is huangweihaohuang, and it must come from the token alone.
    conf_dir = tempfile.mkdtemp(prefix="kaggle-noconf-")
    env = dict(os.environ, KAGGLE_API_TOKEN=token, KAGGLE_CONFIG_DIR=conf_dir)

    counts = dict.fromkeys(STATUSES, 0)
    submits_done = 0
    for comp in comps:
        prefix = f"{MSG_PREFIX} {comp}"
        message = f"{prefix}, {today}"
        sub = os.path.join(root, "competitions", comp, "submission.csv")
        sample = os.path.join(root, "competitions", comp, "data", "sample_submission.csv")
        base = {"comp": comp, "file": sub, "sha256": "", "message": message,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "", "public": "", "private": "", "note": ""}

        if not os.path.isfile(sub):
            counts["missing"] += 1
            _record(out_csv, {**base, "status": "missing",
                              "note": "no submission.csv — the lane left nothing to score"})
            _log(status_log, f"{comp}: MISSING — no submission.csv; continuing with the rest")
            continue
        base["sha256"] = _sha256(sub)

        # Validate BEFORE any upload. A malformed file burns a daily submission and puts a
        # garbage row in the account history; the validator is the same gate the skill
        # itself runs, so "passed validation" means the same thing on both sides. A
        # validator that hangs or cannot start costs only this comp, never the loop: an
        # uncaught TimeoutExpired here used to abort the whole run with rc 1, outside
        # the 0/2/3 exit contract.
        try:
            vrc, vout = _validate(sub, sample, a.allow_suspicious)
        except (subprocess.TimeoutExpired, OSError) as e:
            counts["submit-failed"] += 1
            _record(out_csv, {**base, "status": "submit-failed",
                              "note": f"validator did not run ({type(e).__name__}) — "
                                      f"not submitted"})
            _log(status_log, f"{comp}: VALIDATOR DID NOT RUN ({type(e).__name__}) — not "
                             f"submitted; continuing with the rest")
            continue
        if vrc != 0:
            counts["validation-blocked"] += 1
            tier = "structural" if vrc == 1 else \
                "suspicious, not waived" if vrc == 2 else f"validator rc {vrc}"
            _record(out_csv, {**base, "status": "validation-blocked",
                              "note": f"validator rc {vrc} ({tier})"})
            _log(status_log, f"{comp}: BLOCKED by the validator ({tier}) — not submitted")
            print(vout.rstrip())
            continue

        # Idempotence: a row already carrying this run's prefix IS this run's submission,
        # from a previous or crashed invocation. Submitting again would spend quota to
        # produce a duplicate; record the existing row's scores instead. If the listing
        # itself hangs or dies there is no proof a submit would not duplicate, so nothing
        # is uploaded and only this comp is charged.
        try:
            pre = _match(_fetch_rows(a.kaggle_bin, comp, env), prefix)
        except (subprocess.TimeoutExpired, OSError) as e:
            counts["submit-failed"] += 1
            _record(out_csv, {**base, "status": "submit-failed",
                              "note": f"submissions fetch did not return "
                                      f"({type(e).__name__}) — idempotence unprovable, "
                                      f"nothing uploaded"})
            _log(status_log, f"{comp}: FETCH FAILED ({type(e).__name__}) — not submitted; "
                             f"continuing with the rest")
            continue
        already = pre is not None
        if already:
            _log(status_log, f"{comp}: a submission with this run's prefix already exists "
                             f"— skipping the upload, recording its scores")
        else:
            if submits_done and a.sleep > 0:
                time.sleep(a.sleep)
            try:
                s = subprocess.run([a.kaggle_bin, "competitions", "submit", "-c", comp,
                                    "-f", sub, "-m", message],
                                   capture_output=True, text=True, env=env,
                                   timeout=SUBMIT_TIMEOUT, check=False)
            except (subprocess.TimeoutExpired, OSError) as e:
                counts["submit-failed"] += 1
                _record(out_csv, {**base, "status": "submit-failed",
                                  "note": f"kaggle submit did not return "
                                          f"({type(e).__name__}) — may or may not have "
                                          f"uploaded; a rerun is idempotent"})
                _log(status_log, f"{comp}: SUBMIT FAILED ({type(e).__name__}) — "
                                 f"continuing with the rest")
                continue
            # Success is "Successfully" ANYWHERE in the whole output, never on the tail
            # line: the CLI's carriage-return progress bar is the last line it prints, and
            # tail-line matching marked every successful upload as FAILED on 2026-07-27.
            combined = s.stdout + s.stderr
            if "Successfully" not in combined:
                counts["submit-failed"] += 1
                tail = " / ".join(combined.strip().splitlines()[-2:])[:200]
                _record(out_csv, {**base, "status": "submit-failed",
                                  "note": f"kaggle CLI gave no success line: {tail}"})
                _log(status_log, f"{comp}: SUBMIT FAILED — {tail}")
                continue
            submits_done += 1
            _log(status_log, f"{comp}: submitted, polling for the score")

        # Poll until Kaggle has scored it. These competitions are all closed, so a late
        # submission normally gets public AND private together in one COMPLETE row.
        row = None
        for attempt in range(POLL_TRIES):
            if attempt and a.sleep > 0:
                time.sleep(a.sleep)
            try:
                r = _match(_fetch_rows(a.kaggle_bin, comp, env), prefix)
            except (subprocess.TimeoutExpired, OSError):
                # One lost poll, not a lost run: the next attempt may still land, and
                # the score-timeout row records the comp if none does.
                continue
            # A literal 'None'/'nan' publicScore is an unpublished score, not a score.
            if r and "COMPLETE" in (r.get("status") or "") \
                    and _score(r.get("publicScore")):
                row = r
                break
        if row is None:
            counts["score-timeout"] += 1
            _record(out_csv, {**base, "status": "score-timeout",
                              "note": f"no COMPLETE row with a public score after "
                                      f"{POLL_TRIES} polls — needs a human look"})
            _log(status_log, f"{comp}: SCORE TIMEOUT after {POLL_TRIES} polls")
            continue

        pub = _score(row.get("publicScore"))
        priv = _score(row.get("privateScore"))
        notes = []
        if a.allow_suspicious and "WAIVED by --allow-suspicious" in vout:
            notes.append(f"suspicious waived: {a.allow_suspicious}")
        sha = base["sha256"]
        if already:
            # The scores belong to an EARLIER upload, and the submission.csv on disk may
            # have been regenerated since: attesting the current file's hash against
            # them would fabricate the very provenance link the hash exists to prove.
            sha = ""
            notes.append("upload predates this run; local file hash not attested")
        if not priv:
            # Never fabricate. An empty private cell with a note is a fact; a guessed
            # number is a corruption of the results table.
            notes.append("private pending — needs a human look")
        status = "already-submitted" if already else "scored"
        counts[status] += 1
        # For an already-submitted row the EXISTING description is what produced the
        # score, so that is what gets recorded; for a fresh submit it is this message.
        _record(out_csv, {**base, "sha256": sha, "status": status,
                          "message": (row.get("description") or message) if already else message,
                          "public": pub, "private": priv, "note": "; ".join(notes)})
        _log(status_log, f"{comp}: {status} — public {pub} / private {priv or '(pending)'}")

    print(f"\ncomps: {len(comps)}  scored: {counts['scored']}  "
          f"already: {counts['already-submitted']}  missing: {counts['missing']}  "
          f"blocked: {counts['validation-blocked']}  "
          f"failed: {counts['submit-failed'] + counts['score-timeout']}")
    n_bad = counts["missing"] + counts["validation-blocked"] \
        + counts["submit-failed"] + counts["score-timeout"]
    if n_bad:
        print(f"PARTIAL — {n_bad} of {len(comps)} competition(s) not scored; rows and "
              f"notes are in {out_csv}", file=sys.stderr)
        return 3
    print(f"all {len(comps)} competition(s) scored — rows appended to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
