# Frozen versions

The benchmark's comparability rests on knowing exactly which version produced which number.
This file records the freezes; it is append-only.

## my-agent

| Freeze | Commit | Covers | Notes |
|---|---|---|---|
| v3 (benchmark baseline) | pre-`14a77c4` lineage | the 18 original baseline competitions | six-stage pipeline, no Stage 0.5 |
| v5 (injection layer) | `9a30d8a` and earlier | s3e19 / sep-2022 / s5e1 controlled arms, and the two firing-class three-way lanes | Stage 0.5 dossier, typed operator contract, external-data layer |
| **v5.2 (current, frozen 2026-08-03)** | **`8416b4d`** | nothing yet — the freeze exists so the next run has a stated version | adds: per-fold target encoding + sparse-linear family (`eval_support.py`), runtime attestation, the three quality gates, and the 15 execution-path bug fixes of the 08-03 sweep |

**Rule.** Any new competition lane states the my-agent commit it ran at, in its lane log and
in the report row it produces. A lane that cannot state its commit is not comparable and its
number is not admitted to the baseline.

## Reference methods

Both are frozen for the duration of the study and carry no version table: AIDE runs its
released code at its default 20 steps, the NVIDIA agent runs its shipped skill. Improvements
built for either (e.g. `benchmark_infra/verify_reproduction.py`, the reproduce-then-verify
check) are explicitly **post-freeze**: they gate nothing retroactively and are not part of any
reported run.

## Log

- 2026-08-03 — v5.2 frozen at `8416b4d` after the adversarial execution-path sweep. No run
  has used it yet. The 20-competition baseline is closed at its measured versions; expansion
  was considered and declined (see the batch decision in the engineering log and the power
  analysis: separating the observed 57%/48% win rates would need ~388 competitions, and the
  only available pool is entirely Playground Series, which would push the evaluation set from
  70% to 80–89% Playground and dilute the very composition contrast the report's central
  claim rests on).
