# Report Validation (Plan §5)

> Evidence for the three report-validation mechanisms in plan §5. Date: 2026-07-08.
> 1. Scoring rubric / 2. Code consistency check / 3. Report-only reproduction test.

## 1. Scoring rubric (§5 report validation 1)

**3 independent LLM subagents** (not the report authors; the plan's "can first be compared by another LLM") scored the 15 competitions
per `docs/report_rubric.md` (10 items / max 20 / threshold ≥16 and no item at 0).

| Competition | Total/20 | Passed | Missing items (items scoring 1) |
|---|---|---|---|
| s3e1 | 20 | ✅ | — |
| s3e3 | 19 | ✅ | Item 4: KITCHENBLEND champion ensemble members/weights/solo not recorded |
| s3e5 | 19 | ✅ | Item 5: no concrete post-tuning hyperparameters (only Optuna meta-settings) |
| s3e7 | 20 | ✅ | — |
| s3e9 | 20 | ✅ | — |
| s3e11 | 20 | ✅ | — |
| s3e14 | 19 | ✅ | Item 9: §7 missing Stage 1 baseline reproduction instructions |
| s3e16 | 18 | ✅ | Item 3: §3.1 does not state missingness status; Item 5: no post-tuning hyperparameters |
| s3e19 | 20 | ✅ | — |
| s3e20 | 20 | ✅ | — |
| s4e1 | 20 | ✅ | — |
| s4e11 | 19 | ✅ | Item 5: CV seed not explicitly listed + post-tuning hyperparameters not listed |
| s5e10 | 20 | ✅ | — |
| s6e1 | 20 | ✅ | — |
| s6e2 | 19 | ✅ | Item 9: §7 missing Stage 1 reproduction instructions |

**Result: 15/15 all passed** (threshold ≥16 and no single item at 0); average **19.5/20**.

**Missing-item types + revision status (honest disposition)**:
| Type | Competitions | Disposition |
|---|---|---|
| §3.1 does not state missingness status | s3e16 | ✅ Added (verified: no missing values, no duplicate rows) |
| §7 missing Stage 1 reproduction instructions | s3e14, s6e2 | ⓘ Stage 1 = the generic batch / the pre-existing February baseline, **not a runnable step of this pipeline** (it is the baseline anchor being compared against, no reproduction needed); this rubric item is on the strict side, we do not add fake commands, and the report still passes at 19/20 |
| Champion ensemble weights not recorded | s3e3 | ⚠️ The early-batch KITCHENBLEND did not save member weights — a record limitation, we do not fabricate |
| Concrete post-tuning hyperparameters not recorded | s3e5, s3e16, s4e11 | ⚠️ The early batch did not save Optuna best params — a record limitation, honestly noted (part of "the S3 early batch has less instrumentation," see plan_compliance_audit) |

**Summary**: all 15 competitions passed (average 19.5/20). Missing items are mostly **early-batch record limitations** (hyperparameters/weights not saved), not report errors;
the only clearly fixable factual gap (s3e16 missingness) has been added. This also corroborates the audit's "the S3 early batch has less instrumentation."

## 2. Code consistency check (§5 report validation 2)

**3 independent audit subagents** compared "the methods described in the report ↔ the actual code" (five points: CV / model / post-processing / tree config / champion blend);
I **verified every finding item by item against the tree files/code** (manual spot-check) before disposition.

**Result: 10 of 15 competitions fully consistent; 5 competitions found 5 genuine inconsistencies in total (all verified, wording corrected, no decision scores touched)**:

| Competition | Severity | Inconsistency | Correction |
|---|---|---|---|
| s5e10 | med | §3.3 "no failed node" is a genuine error — boundary-push node#5 pushed min_child_samples to an illegal negative value → failure, cascading to 4 failed blends (+6 dedup placeholders) | ✅ Changed to honestly disclose the failed node |
| s3e3 | med | The Stage 5 quick-eval member pool took the base-tree 3-way (node#7), not the _scale-tree 36-way true champion (node#55); stage5_sweep did not read _scale | ✅ Note added to §5; the conclusion (Stage 5 = Stage 4) is unaffected |
| s3e11 | low | §3.2 DEEPLGB "3rd-largest weight" is actually 4th-largest (0.1783, 4th in descending order) | ✅ Changed to 4th-largest |
| s6e1 | low | §3.3 "no failed node" is imprecise — 2 status=failed are dedup placeholders | ✅ Changed to "no genuine evaluation failures" |
| s3e9 | low | §2.3/§3.4 "no post-processing" but the code clips the submission (≥0) | ✅ Disclosed (no-op) |

**Assessment**: the consistency check successfully caught 5 genuine report↔code gaps (2 med / 3 low), mostly wording precision; the most important,
s5e10's "no failed node," was originally a misstatement and has been made honest (a boundary push does produce illegal settings that fail). All corrections **touch no reported
number**, only bringing the narrative in line with the code.

## 3. Report-only reproduction test (§5 report validation 3)

Dispatched **2 independent agents to read the report + raw data only** (strictly forbidden from reading any code/scripts/experiment logs/processed CSV),
to reconstruct the process from scratch, run the scores, and measure the gap. Chose 2 representative competitions: s3e1 (RMSE, clean), s3e5 (QWK ordinal, complex post-processing).

| Competition | Reproduced stage | Reproduced score | Report claim | Gap | Conclusion |
|---|---|---|---|---|---|
| s3e1 | Stage 2 blend | 0.559287 | 0.558768 | ~0.0005 (~0.09%) | Partial |
| s3e5 | Stage 3 (tuning + OptimizedRounder) | 0.56265 | 0.56769 | ~0.005 | Partial |

**Shared conclusion: "partial" — enough to reconstruct the process + approach the score (gap ~0.09–0.9%), not enough to reproduce bit-for-bit.**
- **Yes (methodology)**: the problem / metric / CV scheme / model family / blend structure / **post-processing** are all fully described. s3e5's OptimizedRounder
  cut-point post-processing (the biggest gain source in this competition) is especially well documented — the cut points the reproducing agent independently searched out match every cut point given in the report
  to within ~0.06, strong evidence the method is reconstructable; s3e1's EDA layer matches exactly.
- **No (bit-level)**: because the **model baseline hyperparameters and exact engineered feature set** are not recorded one by one (§3.3 often admits "not recorded"), the residual gap
  comes precisely from this. Consistent with the rubric (§1) finding "early batch did not save hyperparameters" and the audit's "the S3 early batch has less instrumentation."

**Verdict for Goal 2**: the reports achieve "**major decisions and process reconstructable from the report alone, landing near the score**" (the core of Goal 2);
"reproducing the numbers bit-for-bit" is limited by the early batch's unrecorded hyperparameters — not a report error, but a matter of record granularity. This is an honest boundary of completeness.

---

## Summary (§5 three report validations)

| Mechanism | Result |
|---|---|
| 1 Scoring rubric | 15/15 all passed (average 19.5/20); missing items mostly early-batch record limitations, s3e16 missingness added |
| 2 Code consistency check | 15 competitions, 10 consistent; caught 5 genuine inconsistencies (2 med/3 low) all verified + corrected, no decision scores touched |
| 3 Report-only reproduction test | 2 representative competitions both "partial": process reconstructable, score approached ~0.09–0.9%, bit-level limited by unrecorded hyperparameters |

**Overall**: report quality (Goal 2) meets the target on all four fronts — **process reconstructable, methods faithful, numbers traceable, key decisions reconstructable from report only**;
the only systematic limitation is "the early batch (S3) did not record baseline hyperparameters one by one," which is honestly delimited and not a report error.
