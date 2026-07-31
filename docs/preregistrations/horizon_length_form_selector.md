# Pre-registration: horizon length as the injection-form selector

**Status: REGISTERED 2026-07-31 — awaiting the next firing-class competition. No leaderboard
for any new firing-class competition may be consulted before the prediction below is copied
into that competition's dossier and committed.**

## Why this document exists

Every local signal tested for choosing the external-data injection form is 0-for-3
(CV, held-out-year, proportionality diagnostic — the last killed by its own pre-registered
prediction on s5e1; see `knowledge/task_priors.md`, TASK-TS-FUTURE). One post-hoc pattern
survives in the n=3 evidence and earns belief the only way the dead selector did not:
by predicting in advance.

## Hypothesis (H-HORIZON)

On a firing-class competition (country-panel time series where the dossier proposes external
data), the winning injection form is decided by the test-horizon length:

- horizon **≤ 1 year** → `join_feature` beats `ratio_target` on the private leaderboard;
- horizon **> 1 year** → `ratio_target` beats `join_feature`.

Evidence that motivated it (all post-hoc, hence this registration):

| competition | horizon | winner | private scores |
|---|---|---|---|
| s3e19 | 1 year | join_feature | 48.497 vs 52.073 (SMAPE) |
| tps-sep-2022 | 1 year | join_feature | 23.390 vs 24.091 (SMAPE) |
| s5e1 | 3 years | ratio_target | 0.12417 vs 0.15626 (MAPE) |

## Protocol (binding, copied from the s5e1 falsification run)

1. **Trigger**: the Stage 0.5 dossier of a new competition fires external-data need in the
   country-panel family (TASK-TS-FUTURE class).
2. **Before any model is fitted and before any leaderboard is consulted**, write into that
   competition's `dossier.json` a `preregistration` object:
   `{"hypothesis": "H-HORIZON", "horizon_years": <n>, "predicted_winner": <form>,
   "falsifier": "the predicted loser wins the paired test, or the paired test returns a tie"}`
   and commit it to version control.
3. Race both forms as usual — **the prediction never changes what runs** (the operating rule
   "race both arms, always" stands regardless of this hypothesis's fate; pre-registration is
   epistemics, not a gate).
4. Score both arms on the real leaderboard; judge by the paired test.

## Pre-committed consequences

- **Falsifier fires** → H-HORIZON is demoted to a recorded note in `task_priors.md` (exactly
  as the proportionality diagnostic was), the form-selection question stays open, and no new
  post-hoc pattern may be promoted without its own registration here first.
- **Prediction holds** → H-HORIZON gains one confirmation at n=1 pre-registered; it still does
  not gate racing until it has survived **two** pre-registered tests, and even then it selects
  the racing order (which arm gets the first budget), never a single-arm run.

## Log

- 2026-07-31 — registered. No firing-class competition pending; next one triggers step 2.
