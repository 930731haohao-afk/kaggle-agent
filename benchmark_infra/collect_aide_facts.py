#!/usr/bin/env python3
"""Deterministic fact extraction from an AIDE run's journal — the number-integrity
backbone for AIDE competition reports.

Mirrors the contract of the kaggle-mlspec-report collector: every number a report
cites must come from here, never from an LLM's reading of the code or log. Writes
one facts_aide.json per competition under aideml-runs/<comp>/.

Usage:
    python collect_aide_facts.py                 # all comps under aideml-runs
    python collect_aide_facts.py <comp> [...]    # named comps only
"""

import json
import sys
from pathlib import Path

RUNS = Path.home() / "ai_agents/aideml-runs"
LB = RUNS / "lb_submissions.json"
SCORES = RUNS / "three_way_scores.csv"


def newest_journal(comp_dir: Path) -> Path | None:
    """Newest run's journal for this competition (runs are resumed, so latest wins)."""
    js = sorted(comp_dir.glob("logs/*/journal.json"), key=lambda p: p.stat().st_mtime)
    return js[-1] if js else None


def lb_entry(comp: str) -> dict:
    """AIDE's leaderboard record for this comp, if it was submitted."""
    if not LB.is_file():
        return {}
    try:
        rows = json.loads(LB.read_text())
    except json.JSONDecodeError:
        return {}
    hits = [r for r in rows if r.get("comp") == comp and r.get("method") == "aide"]
    return hits[-1] if hits else {}


def lb_scores(comp: str) -> dict:
    """Public/private LB columns from the three-way score table."""
    if not SCORES.is_file():
        return {}
    import csv

    key = comp.replace("playground-series-", "")
    for row in csv.DictReader(SCORES.open()):
        if row.get("comp") in (comp, key):
            return {
                k: row[k]
                for k in ("metric", "direction", "aide_local", "aide_pub", "aide_priv",
                          "aide_pr", "aide_rank", "mine_priv", "nvidia_priv",
                          "local_winner", "lb_winner")
                if row.get(k)
            }
    return {}


def collect(comp: str) -> dict | None:
    comp_dir = RUNS / comp
    jpath = newest_journal(comp_dir)
    if jpath is None:
        return None

    journal = json.loads(jpath.read_text())
    nodes = journal["nodes"] if isinstance(journal, dict) else journal

    scored = [n for n in nodes if not n.get("is_buggy") and (n.get("metric") or {}).get("value") is not None]
    values = [n["metric"]["value"] for n in scored]
    maximize = bool(scored[0]["metric"].get("maximize")) if scored else None
    best_node = None
    if scored:
        best_node = max(scored, key=lambda n: n["metric"]["value"]) if maximize \
            else min(scored, key=lambda n: n["metric"]["value"])

    exec_times = [n["exec_time"] for n in nodes if n.get("exec_time")]
    buggy = [n for n in nodes if n.get("is_buggy")]
    # Why each buggy node failed — the search's failure profile, not just its count.
    exc_types = {}
    for n in buggy:
        exc_types[n.get("exc_type") or "unknown"] = exc_types.get(n.get("exc_type") or "unknown", 0) + 1

    return {
        "competition": comp,
        "journal_path": str(jpath),
        "run_name": jpath.parent.name,
        "steps_total": len(nodes),
        "steps_scored": len(scored),
        "steps_buggy": len(buggy),
        "buggy_exc_types": exc_types,
        "metric_maximize": maximize,
        "best_metric": best_node["metric"]["value"] if best_node else None,
        "best_step": best_node.get("step") if best_node else None,
        "first_scored_metric": values[0] if values else None,
        "metric_trajectory": [
            {"step": n.get("step"), "value": n["metric"]["value"]} for n in scored
        ],
        "exec_time_total_s": round(sum(exec_times), 1) if exec_times else None,
        "exec_time_mean_s": round(sum(exec_times) / len(exec_times), 1) if exec_times else None,
        "exec_time_max_s": round(max(exec_times), 1) if exec_times else None,
        "best_solution_path": str(jpath.parent / "best_solution.py"),
        "aide_report_path": str(jpath.parent / "report.md"),
        "leaderboard": lb_entry(comp),
        "score_table": lb_scores(comp),
    }


def main() -> None:
    comps = sys.argv[1:] or sorted(
        d.name for d in RUNS.iterdir() if d.is_dir() and (d / "logs").is_dir()
    )
    written = 0
    for comp in comps:
        facts = collect(comp)
        if facts is None:
            print(f"skip {comp}: no journal")
            continue
        out = RUNS / comp / "facts_aide.json"
        out.write_text(json.dumps(facts, indent=2))
        written += 1
        print(f"{comp}: steps={facts['steps_total']} scored={facts['steps_scored']} "
              f"best={facts['best_metric']} -> {out.name}")
    print(f"\nwrote {written} facts_aide.json")


if __name__ == "__main__":
    main()
