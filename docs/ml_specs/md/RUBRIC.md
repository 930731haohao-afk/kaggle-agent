# Report Rubric Validation (plan-aligned)

The summer-project plan (目標二) requires a rubric checklist as the acceptance criterion, and the plan's §5.1 defines report
validation as: covers the competition purpose + the five process components, is clear, reproducible-without-code,
number-traceable, and honest.

The kaggle-report skill's own rubric (R1–R19) targets that skill's **7-section** format (e.g. R1 "7 節齊備",
R17 "must not quote facts.json"), which is a *different* document structure from the 顏佐榕 **5-section** ML-spec
framework used here. So these reports are validated against a **plan-aligned rubric** derived directly from Goal 2
and the plan's §5.1, not the skill's format-specific rubric.

## Checklist (8 items)

| # | Check | Standard |
|---|-------|----------|
| P1 | Purpose — what | Overview + Purpose of Data state the problem being solved |
| P2 | Purpose — why (重要性) | An explicit "Why it matters" states why the problem is important (non-canned) |
| P3 | Five components complete | Data / Models / Training / Inference / Evaluation all present, each with substance or explicit "not recorded" |
| P4 | Reproducible decisions | Splitting strategy + Feature Set + Reproducibility Standards documented so a data scientist could rebuild without code |
| P5 | Numbers traceable | Every figure grounded in the competition's structured records (grounding line present; no invented numbers) |
| P6 | Decision trajectory | Champion node identified and the score progression (stage/tier table) shown — the "breakthrough" trail |
| P7 | Honesty | CV-only / LB-closed stated where true; unavailable items marked "not recorded"; source conflicts surfaced |
| P8 | Performance benchmarking | Cross-agent comparison (vs NVIDIA reproduce-agent) present, supporting Goal 3 |

## Result — all 25 validated reports pass 8/8

(15 original reports validated 2026-07-22; the 7 benchmark-baseline additions — afsis, cat-in-the-dat,
conway, tps-aug-2022, tps-jan-2022, s5e1, tps-sep-2022 — drafted and number-audited 2026-08-17, every
figure traced to the competition's workspace artifacts or the frozen three-way score table; the 3
special deep-learning-fallback lanes — ventilator, us-patent, siim-isic — added below, every figure
traced to that competition's `facts.json` and the offline MLE-bench records in `benchmark_results/run2/`.)

Two of the rows below, **s3e20** and **s4e1**, sit outside the frozen baseline and their reports have
been moved to `../archive_non_benchmark/`; the `md/` folder itself holds **23** reports — the 20
baseline competitions plus the 3 special lanes.

| Comp | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
|------|----|----|----|----|----|----|----|----|
| s3e1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e11 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e14 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e19 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s3e20 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s4e1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s4e11 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s5e10 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s6e1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s6e2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| afsis-soil-properties | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cat-in-the-dat | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| conway-s-reverse-game-of-life | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| tps-aug-2022 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| tps-jan-2022 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| s5e1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| tps-sep-2022 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ventilator-pressure-prediction | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| us-patent-phrase-to-phrase-matching | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| siim-isic-melanoma-classification | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

For the three special lanes, P8 is satisfied by an offline MLE-bench three-way grade table (our agent vs
NVIDIA vs AIDE) rather than a leaderboard comparison, and P7 additionally carries the "no live
public/private leaderboard exists for this lane" statement.

**Not covered by these reports** (broader plan items, out of report scope): the plan's §5.2 code-consistency check
(LLM compares report vs code), §5.3 reproducibility test (a third party rebuilds from report only),
Goal 5 Docker containerization.
