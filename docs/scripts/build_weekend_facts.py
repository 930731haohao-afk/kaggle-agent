"""docs/scripts/build_weekend_facts.py — deterministic fact merger for
docs/weekend_report.md (the supervisor-facing comprehensive weekend report).

Merges three sources into a single docs/weekend_facts.json so that
verify_report.py has one file to check every number in the report against:

  1. docs/benchmark_facts.json   — the four-tier (generic/skill/iterated/tree)
     cross-competition comparison, produced by build_benchmark_table.py.
  2. docs/tree_facts.json        — the 15-run tree-search fact base, produced
     by build_tree_facts.py.
  3. a hand-listed dict (WEEKEND_TIMELINE_FACTS below) of timeline/count
     numbers that live only in git history and in the ledger
     (.superpowers/weekend-plan.md), not in any experiments.json: phase
     durations, commit counts, test counts, knowledge-base entry count.
     Every number in this dict is transcribed verbatim from either
     `git log --pretty='%ad %h %s' --date=format:'%Y-%m-%d %H:%M'`
     (re-run and diffed against the literal figures below — see the comment
     next to each block) or from the task brief's own authoritative timing
     table (A 1.5h/B 5.4h/C 1.9h/D 3.2h/E 2.7h/F 2.5h/G 41m/H 20m, ~18.5h
     total, 59 commits, 82 tests).

No number is invented here: the per-phase commit counts are the literal
`git log --oneline <first-commit-of-phase>..<last-commit-of-phase>` counts
(inclusive), verified against the checked ("[x]") ledger lines in
.superpowers/weekend-plan.md. Note one reconciled discrepancy: docs/weekend_summary.md's
phase table states "C(4 單元)" but the ledger's own checked lines for Phase C
total 5 (C-1, C-2a, C-2b, C-2c, C-3), which also matches the actual git log
commit count for that phase (2920ae5, e48871a, 0301958, c6094bd, 3662a5a).
The ledger's final tally line "43 單元" (Phase A-H total) is only internally
consistent with the sum 8+11+5+7+5+3+2+2=43 when Phase C uses 5, not 4 — so
this script (and the new report) uses 5 for Phase C, and docs/weekend_report.md
notes the correction explicitly rather than silently propagating the older
summary's smaller figure.

Usage:
    uv run python3 docs/scripts/build_weekend_facts.py
Writes docs/weekend_facts.json.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCS = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_DOCS)

# ---------------------------------------------------------------------------
# Hand-listed timeline/count facts (no experiments.json backs these numbers;
# they live in git log timestamps + the weekend-plan.md ledger).
# ---------------------------------------------------------------------------
WEEKEND_TIMELINE_FACTS = {
    "duration": {
        "total_hours_phase_A_to_H": 18.5,
        "phase_hours": {
            "A": 1.5, "B": 5.4, "C": 1.9, "D": 3.2,
            "E": 2.7, "F": 2.5, "G": 0.683, "H": 0.333,
        },
        "phase_minutes_short": {"G": 41, "H": 20},
        "phase_commit_time_range": {
            # first commit timestamp -> last commit timestamp of each phase,
            # from `git log --pretty='%ad %h %s' --date=format:'%Y-%m-%d %H:%M'`
            "pre_phase_infra": ["2026-07-03 16:49", "2026-07-03 18:07"],
            "A": ["2026-07-03 18:35", "2026-07-03 19:58"],
            "B": ["2026-07-03 20:04", "2026-07-04 01:27"],
            "C": ["2026-07-04 01:38", "2026-07-04 03:18"],
            "D": ["2026-07-04 03:27", "2026-07-04 06:41"],
            "E": ["2026-07-04 07:00", "2026-07-04 09:23"],
            "F": ["2026-07-04 09:35", "2026-07-04 11:54"],
            "G": ["2026-07-04 12:13", "2026-07-04 12:35"],
            "H": ["2026-07-04 12:45", "2026-07-04 12:55"],
        },
    },
    "commits": {
        "total": 59,
        "pre_phase_infra": 15,
        "phase_A_to_H_subtotal": 43,
        "final_wrap_commit": 1,
        "per_phase": {
            "A": 8, "B": 11, "C": 5, "D": 7,
            "E": 5, "F": 3, "G": 2, "H": 2,
        },
    },
    "tests": {
        "final_total": 82,
        "progression": {
            "pre_phase_baseline": 20,
            "after_phase_D1_v2_harness": 39,
            "after_phase_F1_v3_harness": 63,
            "after_phase_H1_online_hardening": 82,
        },
        "new_tests_added": {"D1": 19, "F1": 24, "H1": 19},
    },
    "knowledge_base": {
        "experience_md_entries": 52,
    },
    "tree_search": {
        "total_runs": 15,
        "competitions_covered": 10,
        "runs_by_generation": {"v1": 3, "v2": 9, "scale": 1, "v3": 2},
        "verdict_15_runs": {"win": 12, "tie": 2, "loss": 1},
        "verdict_10_comp_coverage": {"win": 9, "tie": 1, "loss": 0},
    },
    "benchmark": {
        "competitions": 10,
    },
    "incidents": {
        "api_overload_http_status": 529,
        "api_overload_occurrence_count": 1,
    },
}


def merge():
    with open(os.path.join(_DOCS, "benchmark_facts.json")) as f:
        benchmark_facts = json.load(f)
    with open(os.path.join(_DOCS, "tree_facts.json")) as f:
        tree_facts = json.load(f)
    return {
        "generated_by": "docs/scripts/build_weekend_facts.py",
        "sources": [
            "docs/benchmark_facts.json",
            "docs/tree_facts.json",
            "hand-listed WEEKEND_TIMELINE_FACTS (git log timestamps + "
            ".superpowers/weekend-plan.md ledger)",
        ],
        "weekend_timeline": WEEKEND_TIMELINE_FACTS,
        "benchmark_facts": benchmark_facts,
        "tree_facts": tree_facts,
    }


def main():
    out = merge()
    out_path = os.path.join(_DOCS, "weekend_facts.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
