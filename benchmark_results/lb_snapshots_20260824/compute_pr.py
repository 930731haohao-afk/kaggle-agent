#!/usr/bin/env python3
"""Recompute the percentile-rank (PR) column of REPORT_v7 Table 2 from the
2026-08-24 public-leaderboard snapshots in this directory.

Definition: PR = percent of public-leaderboard teams whose public score is
at-or-below the agent's public score, direction-aware (for a minimized metric,
teams scoring >= the agent; for a maximized metric, teams scoring <= the agent).
Agent public scores and metric directions come from ../rerun_three_way.csv.

Note: the aide_pr / nvidia_pr columns already inside rerun_three_way.csv are a
frozen record computed earlier under a different definition and snapshot date;
they are intentionally left untouched. REPORT_v7 Table 2 uses the values this
script prints. See README.md in this directory.
"""
import csv
import glob
import math
import os
import statistics


def round1(v):
    """Round half-up to one decimal, matching the table's printed cells."""
    return math.floor(v * 10 + 0.5) / 10

HERE = os.path.dirname(os.path.abspath(__file__))
RERUN = os.path.join(HERE, "..", "rerun_three_way.csv")

AGENTS = [("mine", "mine_rerun_pub"), ("nvidia", "nvidia_pub"), ("aide", "aide_pub")]


def snapshot_scores(comp):
    paths = glob.glob(os.path.join(HERE, f"{comp}-publicleaderboard-*.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected exactly one snapshot for {comp}, got {paths}")
    with open(paths[0], encoding="utf-8-sig") as f:
        return [float(r["Score"]) for r in csv.DictReader(f)]


def pr(scores, agent_score, direction):
    if direction == "min":
        beaten = sum(1 for s in scores if s >= agent_score)
    else:
        beaten = sum(1 for s in scores if s <= agent_score)
    return 100.0 * beaten / len(scores)


def main():
    with open(RERUN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # the scoring log appends; keep the last row per competition
    dedup = {}
    for r in rows:
        dedup[r["comp"]] = r
    per_agent = {name: [] for name, _ in AGENTS}
    print(f"{'competition':<28}" + "".join(f"{n:>10}" for n, _ in AGENTS))
    for comp, r in dedup.items():
        scores = snapshot_scores(comp)
        # the direction column is the LOCAL metric's direction; where the
        # leaderboard scores in the opposite direction, the row's note says so
        # (s6e1: local R2 is max, the LB reports an error, min)
        direction = "min" if "LB direction min" in r.get("note", "") else r["direction"]
        cells = []
        for name, col in AGENTS:
            v = round1(pr(scores, float(r[col]), direction))
            per_agent[name].append(v)
            cells.append(v)
        print(f"{comp:<28}" + "".join(f"{c:>10.1f}" for c in cells))
    # the table's summary row is the median of the printed (rounded) cells
    print(f"{'median (range)':<28}", end="")
    for name, _ in AGENTS:
        vs = per_agent[name]
        print(f"  {round1(statistics.median(vs)):.1f} ({min(vs):.1f}-{max(vs):.1f})", end="")
    print()


if __name__ == "__main__":
    main()
