# Benchmark infrastructure (canonical copies)

Snapshot copies of the `ai_agents/`-level scripts used to run the three-way
agent benchmark (my-agent vs AIDE vs NVIDIA). Canonical locations on the
benchmark machine are listed in Appendix B.2 of `docs/MIDTERM_REPORT_202607.pdf`
(that report has since been superseded by `docs/REPORT.tex` as the project's
main report, but B.2 remains the appendix that records the canonical paths).

**These are snapshots, and they can drift.** The originals live outside this
repository, so a copy here is only as current as the day it was taken. Verified
on 2026-08-27: `diff -q benchmark_infra/lane_lock.sh ../lane_lock.sh` reports the
two files identical. Re-run that comparison before trusting any copy in this
directory as the as-run version, and note that `run_myagent_headless.sh` at the
repo root sources the *out-of-repo* `~/ai_agents/lane_lock.sh`, not the copy here.

## What is here (23 scripts + this README, all tracked)

One-line summaries below are taken from each script's own header; open the header
for the incident that motivated it.

Run roots, environments and mutual exclusion:

- `build_myagent_run_root.py` — build the isolated root my-agent runs the 20-competition benchmark from (`--write` to apply)
- `build_isolated_roots.py` — build clean, agent-neutral competition roots so a reference agent is not handed my-agent's working directory
- `provision_run_root_venv.sh` — build the run root's virtualenv completely, and prove it, before a lane depends on it
- `lane_sandbox.sh` — run one lane with the out-of-root surfaces removed from its filesystem
- `lane_lock.sh` / `lane_lock.py` — lane mutex ("one lane at a time" as a mechanism, not a convention); the `.py` form is for Python drivers, which cannot hold the shell version's lock correctly
- `archive_workspaces_for_rerun.sh` — archive the 20 benchmark workspaces ahead of the re-run
- `bench_watchdog.sh` — notice when the machine sits idle while work is still queued

Preflight and isolation checks (before a lane starts):

- `preflight_oauth_token.py` — refuse to start a long lane on an OAuth token that will expire during it
- `preflight_run_root_env.py` — prove the run root's environment can run every competition before lane 1 starts
- `verify_clean_slate.py` — fail if anything a lane can read states that competition's own result

Execution:

- `run_comp.py` — launch an AIDE run on one competition
- `rerun_20steps.py` — re-run the July-subset competitions at AIDE's default budget (20 steps)
- `run_repeatability_pairs.py` — measure AIDE's between-run variance as pairs against the existing baseline
- `run_ready_reproductions.py` — execute the NVIDIA-lane reproduction scripts that have already been authored and smoke-tested

Audit, scoring and fact collection (after a lane finishes):

- `audit_lane_transcript.py` — read what a lane actually did and say so; the evidence half of the isolation claim
- `split_legality_check.py` — flag single-step metric jumps that coincide with a validation-split change
- `verify_reproduction.py` — compare a reproduced kernel's submitted public score against the score its source kernel claims
- `quarantine_partial_attempt.py` — move an aborted attempt's artifacts out of the run root before the lane restarts
- `score_lane_submissions.py` — submit each lane's `submission.csv` and record its scores (the run root ships no credentials on purpose)
- `build_rerun_three_way.py` — set the rerun's fresh Kaggle scores beside the frozen AIDE/NVIDIA facts; copies, never computes
- `collect_aide_facts.py` — deterministic fact extraction from an AIDE run's journal
- `collect_dl_facts.py` — deterministic fact extraction for special / non-GBDT lanes (see the DL path in `.claude/skills/kaggle-mlspec-report/SKILL.md`)

The resulting score tables live in `benchmark_results/`; `rerun_three_way.csv` is
the authoritative per-competition table.
