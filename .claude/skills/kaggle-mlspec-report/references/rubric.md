# ML-Spec Report Rubric (plan-aligned, 8 items)

The summer-project plan (Goal 2) requires a rubric checklist as the acceptance criterion;
plan §5.1 defines report validation as: covers the competition purpose + the five process
components, is clear, reproducible-without-code, number-traceable, and honest. Walk every
item before finalizing; the committed set in `docs/ml_specs/` passes all 15 at 8/8
(`docs/ml_specs/RUBRIC.md`).

| # | Check | Standard |
|---|-------|----------|
| P1 | Purpose — what | Overview + Purpose of Data state the problem being solved |
| P2 | Purpose — why (importance) | An explicit "Why it matters" states why the problem is important (non-canned) |
| P3 | Five components complete | Data / Models / Training / Inference / Evaluation all present, each with substance or an explicit `not recorded` |
| P4 | Reproducible decisions | Splitting strategy + Feature Set + Reproducibility Standards documented so a data scientist could rebuild without code |
| P5 | Numbers traceable | Every figure grounded in the competition's structured records (grounding line present; no invented numbers; verify_report.py exits 0) |
| P6 | Decision trajectory | Champion node identified and the score progression (stage/tier table) shown — the "breakthrough" trail |
| P7 | Honesty | CV-only / LB-closed stated where true; unavailable items marked `not recorded`; source conflicts surfaced |
| P8 | Performance benchmarking | Cross-agent comparison (vs the NVIDIA reproduce-agent) present, supporting plan Goal 3 — or an explicit note that no benchmark record exists |

## Not covered by these reports (broader plan items, out of report scope)

- plan §5.2 code-consistency check (LLM compares report vs code)
- plan §5.3 reproducibility test (a third party rebuilds from report only)
- Goal 5 Docker containerization

## Note vs the retired 7-section skill

This rubric is the **plan-aligned 8-item** rubric for the **5-section Tso-Jung Yen ML-spec**
format. It deliberately differs from the retired `kaggle-report` skill's format-specific
R1–R19 rubric (which targeted a 7-section structure). Validate ML-spec reports against P1–P8
here, not against the old R-series.
