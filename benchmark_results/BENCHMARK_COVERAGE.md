# Benchmark coverage — which competitions each agent has actually run

Generated 2026-07-27. 37 competitions have been touched by at least one agent. Coverage is
what determines how many of them can carry a conclusion, and it is much narrower than the
raw count suggests.

- **16** have all three agents with AIDE at its full 20-step budget.
- **10** of those also have a private leaderboard and are uncontaminated — the set
  the paired significance test can actually run on.

The 20 competitions with 5 AIDE nodes are from an earlier pilot at a quarter of AIDE's
budget; they also have no NVIDIA run. Bringing any of them into the comparison means running
two lanes, not one.

`private LB` is read from Kaggle's own submission records (`privateScore` populated), not
inferred from the competition category — three Getting Started / Playground competitions
score every test row publicly and have no private split at all, which is invisible from the
category alone.

⚠ marks a competition where the harness fed AIDE another agent's cached intermediates; those
AIDE figures are void pending the re-run in progress.

| Competition | my agent | NVIDIA | AIDE@20 | AIDE nodes | private LB | Category |
|---|:--:|:--:|:--:|--:|---|---|
| playground-series-s3e11 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s3e14 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s3e16 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s3e19 ⚠ | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s3e3 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s3e5 ⚠ | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s3e7 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s3e9 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s4e11 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s5e10 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s6e1 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s6e2 | ✓ | ✓ | ✓ | 20 | ✅ | Playground |
| playground-series-s3e20 | ✓ | ✓ | ✓ | 20 | ? untested | Playground |
| home-data-for-ml-course | ✓ | ✓ | ✓ | 20 | ❌ public only | Getting Started |
| playground-series-s6e7 ⚠ | ✓ | ✓ | ✓ | 20 | ❌ public only | Playground |
| spaceship-titanic | ✓ | ✓ | ✓ | 20 | ❌ public only | Getting Started |
| playground-series-s3e1 ⚠ | ✓ | ✓ | · | 9 | ✅ | Playground |
| playground-series-s4e1 | ✓ | ✓ | · | 5 | ✅ | Playground |
| afsis-soil-properties | ✓ | · | · | 5 | ✅ | Research |
| bike-sharing-demand | ✓ | · | · | 5 | ✅ | Playground |
| cat-in-the-dat | ✓ | · | · | 5 | ✅ | Playground |
| conway-s-reverse-game-of-life | ✓ | · | · | 5 | ✅ | Playground |
| tabular-playground-series-aug-2022 | ✓ | · | · | 5 | ✅ | Playground |
| tabular-playground-series-jan-2022 | ✓ | · | · | 5 | ✅ | Playground |
| tmdb-box-office-prediction | ✓ | · | · | 5 | ✅ | Playground |
| child-mind-institute-detect-sleep-states | ✓ | · | · | 5 | ? untested | Featured |
| emvic | ✓ | · | · | 5 | ? untested | Research |
| forest-cover-type | ✓ | · | · | 5 | ? untested | Playground |
| icdar2013-stroke-recovery-from-offline-data | ✓ | · | · | 5 | ? untested | Research |
| linking-writing-processes-to-writing-quality | ✓ | · | · | 5 | ? untested | Featured |
| llm-classification-finetuning | ✓ | · | · | 5 | ? untested | Getting Started |
| predict-energy-behavior-of-prosumers | ✓ | · | · | 5 | ? untested | Featured |
| the-icml-2013-whale-challenge-right-whale-redux | ✓ | · | · | 5 | ? untested | Research |
| us-patent-phrase-to-phrase-matching | · | · | ✓ | 20 | ? untested | Featured |
| house-prices-advanced-regression-techniques | ✓ | · | · | 5 | ❌ public only | Getting Started |
| store-sales-time-series-forecasting | ✓ | · | · | 5 | ❌ public only | Recruitment |
| titanic | ✓ | · | · | 5 | ❌ public only | Getting Started |
