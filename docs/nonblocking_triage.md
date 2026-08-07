# Triage of the 58 non-blocking architecture-gate findings — 2026-08-07

Source: the 131-agent Stage 1–6 gate (2026-08-04). These were confirmed real but judged
non-blocking for the 20-competition re-run by their verifiers. Triage into three buckets:

- **A — fix before the run.** On the fresh-run execution path; cheap relative to 30–50h of
  compute they could distort.
- **B — fix during/after the run.** Real, but touch paths a fresh run does not exercise, or
  only degrade convenience/observability.
- **C — document as a known limitation.** Contract-hygiene findings (written-but-never-read
  artifacts) whose fix is a design decision, not a patch; or already fixed en route.

## Already fixed en route (verified by the round-2/3 work)

- #47 sep-2022 blend unpack — fixed 6f0e49c, re-verified genuine.
- #54 champion selector drops the true best on "diagnostic" substrings — fixed 5b2c7af,
  the real s3e19 champion is now a test.
- #7 operator contract documents pre-07-31 `encoding` behaviour — superseded by the
  03_features.md §1.5 rewrite (0f0692d).

## Bucket A — fix before the run (12)

| # | finding | why it can distort the run |
|---|---|---|
| 1 | eda_template KeyError on train-only columns | aborts leak-correlation EDA on any comp with a train-only col |
| 4 | data_loader.py divergent copies (dtype fix in one only) | same class as the experiment_log triplication that bit round 2 |
| 13 | get_best_experiment defaults to maximize on the v1 schema | silently inverts champion choice on minimize metrics |
| 17 | free-form `direction` string overrides the correct minimize flag | same inversion, different door |
| 18/55 | champion comparison across different metrics | naive baseline can outrank the champion when metrics mix |
| 19 | evaluation.py triplicated, identity test covers experiment_log only | extend the byte-identity test to every duplicated util |
| 25/40/52 | OOF-cache identity stamp write-only (no producer passes config, no reader checks) | stale-cache reads scored a model that never trained — the exact defect the stamp was built for |
| 36 | Stage 2 validation inspects only train and only warns | test_processed can ship broken |
| 49 | suggest_priors fails open when comp= is the workspace dirname | isolation door reopens under the launcher's naming |
| 51 | v3 drivers resume committed already-stopped trees (0 nodes evaluated) | a "re-run" that instantly resumes to DONE |

## Bucket B — during/after the run (18)

5 (10-col truncated drift check), 8 (flag_feature guard order), 12 (shallow_variant feature
reset), 14 (generic driver bypasses subprocess timeout), 15 (budget re-extension on resume),
16 (GBDT determinism settings location), 21 (submit_template 2-col limit), 22 (two Range
references), 23 (validator prose polish), 26/46 (explore-burst budget reservation), 32
(template categorical collapse), 35 (lin-node attestation), 38 (metric filter reads
eval_metric), 39 (04_modeling schema contradiction), 42/43/44/45 (Stage 4→5 recording schema
family), 48 (DEDUP_REJECTIONS identifier), 50 (verify_advisory workspace filter), 56/58
(query_library zero-hit silent exit).

## Bucket C — known limitations, documented not patched (deliberate)

The written-but-never-read contract family: 2/3/28/33/34 (eda_verdict / validation_decision /
EDA hand-off artifact), 9/29/53 (split policy triple-recorded, no Stage 3 consumer), 30
(rules verdict unread by the dossier schema — note: the verdict IS now enforced at
apply_operators, which is the binding point; the dossier field is bookkeeping), 31 (EDA
number for external_data.needed), 37 (Auto-ML block), 41 (07_5 prior query), 57 (sign
convention enforced at one seam), 6 (validate_data PASS wording), 10 (retrieval gate
asserted/contradicted across documents), 11/27 (stale train_template fork; harness_v4 /
make_v5_arm unreferenced by the skill program).

Rationale: each of these is "an artifact is produced that nothing consumes". The honest fix
is either to build the consumer (a design decision that changes the pipeline) or to stop
producing the artifact (loses the record). Neither belongs in the pre-run window; both are
recorded here so the re-run's readers know these contracts are aspirational.

#20 (Range cannot catch an un-inverted transform even with --train in some shapes) and #24
(ratio on the linear family lacks its inversion in the generated evaluator) are under
round-3 adversarial verification right now; they move to bucket A if confirmed.
