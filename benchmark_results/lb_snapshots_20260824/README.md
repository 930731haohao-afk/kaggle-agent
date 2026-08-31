# Public-leaderboard snapshots, 2026-08-24

The 20 CSVs are Kaggle public-leaderboard snapshots (one per benchmark
competition) downloaded 2026-08-24, the source of the percentile-rank (PR)
column of REPORT Table 2.

`compute_pr.py` recomputes every PR cell and the median/range summary row from
these snapshots plus the agents' public scores in `../rerun_three_way.csv`.
Definition: percent of public-leaderboard teams scoring at-or-below the agent,
direction-aware; cells are rounded half-up to one decimal and the summary row
is the median of the rounded cells. Running it reproduces Table 2's 60 PR
cells and the summary row exactly.

Note on `rerun_three_way.csv`: its `aide_pr` / `nvidia_pr` columns are a
frozen record computed before these snapshots existed, under an earlier
definition and leaderboard state. They intentionally remain untouched;
REPORT Table 2 uses the snapshot-based values from this directory.
