---
name: kaggle-report
description: |
  Generate a structured, reproducible data-analysis report (REPORT.md + PDF) for a completed
  Kaggle competition run. Numbers are extracted deterministically into facts.json by collect.py;
  the agent writes only the what/why narrative and never rewrites a number.

  Use when the user asks to: generate a competition report, 產報告, 產出分析報告, summarize a
  finished competition run, or after completing the kaggle-agent pipeline (Stage 5).

  Trigger phrases: "report", "報告", "REPORT.md", "分析報告", "kaggle report".
---

# Kaggle Report

Turns a finished competition folder into a report a data scientist can use to
reproduce the pipeline **without reading the code**.

## Hard Rules (non-negotiable)

1. **Every number in REPORT.md must be copied verbatim from facts.json.**
   Never recompute, re-round, or invent a number. Derived stats (improvement %,
   gaps) may only appear if the underlying numbers are in facts.json and the
   arithmetic is shown inline (e.g. "1.34356 − 1.33812 = 0.00544").
2. **STATUS.md is narrative context only** — you may read it to understand *why*
   decisions were made, but numbers still come from facts.json.
3. **material_level == "baseline-only"** → say so explicitly: the run used only
   the generic batch baseline; mark EDA/feature sections as 未執行. Never
   fabricate analysis that did not happen.
4. **facts.missing / facts.unparsed** → write 無紀錄 for missing items and list
   unparsed entries verbatim in section 7. Never guess.

## Pipeline

Run from project root (`/home/tjyen/ai_agents/kaggle`), in order:

1. **Collect facts (deterministic)**
   ```bash
   uv run python3 .claude/skills/kaggle-report/assets/collect.py <competition-name>
   ```
   Writes `competitions/<name>/facts.json`. On error: report the message to the
   user and stop — do not improvise around missing inputs.

2. **Write the report (you)**
   - Read `competitions/<name>/facts.json` and [references/report_structure.md](references/report_structure.md).
   - Copy `assets/report_template.md` to `competitions/<name>/REPORT.md`, fill every
     section per the structure spec, embedding numbers/tables from facts.json verbatim.

3. **Rubric self-check** — walk [references/rubric.md](references/rubric.md) item by
   item; fix any gap before proceeding.

4. **Verify number traceability (deterministic)**
   ```bash
   uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
       competitions/<name>/REPORT.md competitions/<name>/facts.json
   ```
   Must exit 0. If it flags numbers: fix the report. Do not add numbers to
   facts.json by hand — facts.json is generated only by collect.py.

5. **PDF (deterministic)**
   ```bash
   bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/<name>/REPORT.md
   ```
   If chromium is unavailable, keep REPORT.md as the deliverable and tell the
   user the PDF step was skipped (MD is the primary artifact).

6. Show the user where REPORT.md / REPORT.pdf landed and summarize rubric result.
