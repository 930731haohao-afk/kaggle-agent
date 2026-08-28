---
name: kaggle-mlspec-report
description: |
  Generate a structured, reproducible ML Specification Report (5-section 顏佐榕 framework:
  Data / Models & Architecture / Training / Inference / Evaluation & Benchmarking) plus an
  academic-serif PDF for a completed Kaggle competition run. Numbers are extracted
  deterministically into facts.json by collect.py; the agent writes only the what/why
  narrative and never invents or rewrites a number. Includes the vs-NVIDIA reproduce-agent
  benchmark and a plan-aligned 8-item rubric self-check.

  Use when the user asks to: generate an ML-spec / competition report, 產 ML 規格報告,
  產出競賽報告, 產出分析報告, write up a finished competition run, or after completing the
  kaggle-agent pipeline.

  Trigger phrases: "ml-spec report", "ML 規格報告", "競賽報告", "分析報告", "report", "報告".

  Do NOT use for: general ML questions, or a competition with no experiments.json /
  facts sources yet — there must be a finished run under competitions/<name>/ to report on.
---

# Kaggle ML-Spec Report

Turns a finished competition folder into a **5-section ML Specification Report** — the
顏佐榕 ML-spec framework aligned to the summer-plan's Goal 1 ((甲) what/why + (乙) the five
process components) — that a data scientist can read to rebuild the pipeline **without
reading the code**. Output lives in `docs/ml_specs/` (canonical report set), not inside the
competition folder.

## Hard Rules (non-negotiable — the number-integrity contract)

1. **Every number in the report must trace to that competition's structured records**
   (`facts.json`, `eda_summary.json`, `STATUS.md`, `experiments_tree_v3.json`,
   `config.yaml`, and the benchmark record for the vs-NVIDIA line). Never recompute,
   re-round, or invent a number. Derived stats (improvement %, gaps) may appear only if
   the underlying numbers are grounded and the arithmetic is shown inline in a fenced code
   block (e.g. `1.34356 − 1.33812 = 0.00544`) — verify_report.py exempts code blocks.
2. **STATUS.md is narrative context only** — read it to understand *why* decisions were
   made; numbers still come from facts.json.
3. **DL→GBDT translation convention**: the 顏佐榕 framework has DL-shaped fields (Loss,
   Optimization Algorithm, Learning Rate, **Learning Rate Scheduler**, **Batch Size**,
   **Transfer Learning**, **Data Augmentation**). For a GBDT pipeline, fill the real ones
   (loss objective, gradient boosting, boosting shrinkage) and mark the DL-only ones
   **`N/A (reason)`** with the GBDT analogue named — e.g. *"no Learning Rate Scheduler
   (N/A — GBDT convergence is governed by CV early-stopping, not a schedule)"*,
   *"Transfer Learning N/A (no pretrained weights); its analogue here is cross-competition
   experience injection"*, *"Data Augmentation N/A (tabular); the analogues are feature
   engineering and seed-bagging"*. Never leave a field blank and never fabricate a DL
   component that did not exist.
4. **Unavailable items → write "not recorded"** (e.g. peak memory, inference duration), and
   **surface source conflicts honestly** rather than picking one (e.g. "eda_summary reports
   562 duplicate rows, STATUS.md reports none; not reconciled").
5. **CV-only honesty**: if the run was never submitted / the leaderboard is closed, state
   "CV-only, no public/private LB" explicitly; only cite a real LB number when one exists.

## The 5-section structure (see references/mlspec_structure.md for the field checklist)

Title line: `# ML Specification Report — <comp>` + a subtitle `### <Task> · our from-scratch
agent (<architecture>)`, then a grounding blockquote listing the source files.

1. **Overview** — what the task is + **"Why it matters"** (non-canned) + the champion CV
   score + a one-line vs-NVIDIA outcome (winner / tie / behind).
2. **Data** — Purpose of Data · Data Format · Data Volume · Data Quality (missing/dupes/
   shift + the defining wrinkle) · strongest EDA signals · Annotation Guidelines · Feature
   Set (grouped table) · Splitting strategy (CV scheme + why + LB status).
3. **Models & Architecture** — Purpose · Architecture Type · Input Format/Dimension ·
   Architecture Description (representative hyperparams) · Model Complexity (trees×leaves,
   not a dense parameter count) · the champion node id + member weights.
4. **Training procedures** — the staged trajectory table (each stage → OOF score) · Loss
   Function · Optimization Algorithm · Learning Rate · Learning Rate Scheduler (N/A) · Batch
   Size (N/A) · Training Duration · Training Memory · Transfer Learning analogue · Data
   Augmentation analogue · Reproducibility Standards (seeds, determinism, replay/rebuild gate).
5. **Inference procedures** — Decision Threshold (or N/A for ranking/regression) ·
   post-processing (clip / round / OptimizedRounder / snap-to-grid) · Inference Duration ·
   Inference Memory.
6. **Evaluation & Benchmarking** — the task · Performance Metrics (with the stage
   progression) · **Performance Benchmarking**: the vs-NVIDIA reproduce-agent table
   (approach + score + winner note), with honest caveats (de-leaking, unified CV schemes,
   CV-only).

(Sections 2–6 are the 五大元件 (乙): Data spec / Model spec / Training spec / Inference
procedure / Evaluation metric. Section 1 is 目的 (甲) what/why.)

## Pipeline

Run from the repository root, in order:

1. **Summarize EDA (deterministic, optional but recommended)**
   ```bash
   uv run python3 .claude/skills/kaggle-mlspec-report/assets/eda_summary.py <competition-name>
   ```
   Writes `competitions/<name>/eda_summary.json` (target distribution, feature stats,
   correlations, collinearity, train/test shift, validation hint). Skip if data absent.

2. **Collect facts (deterministic)**
   ```bash
   uv run python3 .claude/skills/kaggle-mlspec-report/assets/collect.py <competition-name>
   ```
   Writes `competitions/<name>/facts.json`, wiring `config.yaml`, `experiments.json`,
   `STATUS.md`, `eda_summary.json`. On error: report to the user and stop.

3. **Write the report (you)** — read `facts.json` +
   [references/mlspec_structure.md](references/mlspec_structure.md), then write
   `docs/ml_specs/md/playground-series-<comp>.md` following the 5-section structure,
   embedding numbers/tables from facts.json verbatim and applying the DL→GBDT convention.

4. **Rubric self-check** — walk [references/rubric.md](references/rubric.md) (P1–P8) item by
   item; fix any gap before proceeding.

5. **Verify number traceability (deterministic)**
   ```bash
   uv run python3 .claude/skills/kaggle-mlspec-report/assets/verify_report.py \
       docs/ml_specs/md/playground-series-<comp>.md competitions/<name>/facts.json
   ```
   Must exit 0. Fix the report if it flags numbers; never hand-edit facts.json.

   Note the scope mismatch: Hard Rule 1 allows a number to come from any of that
   competition's structured records, but `verify_report.py` only compares against the
   `facts.json` you pass it. A number legitimately taken from `STATUS.md`,
   `eda_summary.json`, `experiments_tree_v3.json` or the benchmark record will therefore
   be flagged. Do not delete such a number and do not copy it into facts.json — trace it
   to its record, and if it has no record, remove the claim or write "not recorded".
   Also check the sign character: the checker only recognises an ASCII `-` as a minus, so
   a typographic `−` turns `−0.048` into the unmatched token `0.048`.

6. **PDF (deterministic)** — grayscale academic serif (`assets/spec_style.css`, the default:
   Times-family serif, no accent colours), auto TOC + page numbers. Every per-competition
   spec ships in this style; do not pass a different stylesheet:
   ```bash
   bash .claude/skills/kaggle-mlspec-report/assets/md2pdf.sh \
       docs/ml_specs/md/playground-series-<comp>.md \
       docs/ml_specs/pdf/playground-series-<comp>.pdf
   ```
   weasyprint primary (page numbers), chromium fallback (no page numbers — say so if it
   fired). Exit 0 = PDF written, 2 = no engine (keep the .md, say PDF skipped), 3 = claimed
   success but no file.

7. Show the user where the .md / .pdf landed and summarize the rubric result. If reporting a
   full set, regenerate `docs/ml_specs/README.md`'s results table and confirm `RUBRIC.md`
   still passes 8/8.

## DL / special-lane path (ANY non-GBDT / complex-modality lane)

Some lanes do not run the GBDT tree-search pipeline — deep-learning fallbacks of **any
modality**: vision, NLP, audio, video, multimodal, sequence/time-series models, and any
future complex competition. Their records differ, so `collect.py` does not apply.

**Decision rule — which extractor?**
- Workspace contains GBDT tree-search records (`experiments_tree*.json`) → it is a GBDT
  pipeline lane → use `collect.py` (steps above).
- Otherwise (DL fallback / complex lane, whatever the modality) → use
  `benchmark_infra/collect_dl_facts.py`. The tool enforces this itself: pointed at a GBDT
  lane it exits with a clear "use collect.py" message instead of overwriting that
  pipeline's facts.json (override: `--force`, ideally with `--out`).

1. **Collect facts with the generic DL/complex extractor**:
   ```bash
   VIRTUAL_ENV= uv run python benchmark_infra/collect_dl_facts.py competitions/<name> \
       [--grade-record PATH] [--out PATH] [--force]
   ```
   Nothing is hardcoded to specific competitions or filenames — records are **discovered
   by glob** in the workspace: `config.yaml|yml`, `experiments*.json` (schema-v2 entries
   verbatim; primary = `experiments.json`), `STATUS.md` (parsed sections),
   `headless*.log` (verbatim), `cv_result*.json`, `res_*.json`, `*_results.json`,
   `discovery*.json`, `dossier.json`, `eda_summary*.json`, `scripts/*_cv.json`,
   `scripts/*decision*.json`, `scripts/*_results.json`, plus deterministic greps of
   `scripts/*.py` (SEED constant, loss / optimizer / LR-scheduler identifiers actually
   referenced) and row counts of `submission*.csv` / `submissions/*.csv`. Offline grade
   records are looked up generically: `benchmark_results/run2/<comp>__*.json` first, then
   any `<comp>__*.json` under `benchmark_results/`; none found → grading is recorded as
   `"not recorded"` (or pass `--grade-record PATH` explicitly). **Metric name/direction
   come only from records** (config → dossier; config → experiment entries → grade
   record `is_lower_better`) — never from a lookup table of known competitions. Absent
   records land in `missing` → the report writes "not recorded". Nothing is ever guessed.
2. **DL fields are NATIVE, not translated** — Hard Rule 3's DL→GBDT convention does not
   apply. Fill backbone, learning-rate schedule, batch size, epochs, augmentation, and
   transfer learning directly from the records (e.g. OneCycleLR / cosine+warmup,
   pretrained DeBERTa-v3 / ImageNet timm backbones, dihedral augmentation). Only mark a
   DL field N/A when the records genuinely show it was absent (e.g. no augmentation used).
3. **Grading is offline (MLE-bench-style), not a live leaderboard** — the grade record
   (score + medal thresholds/flags) is the only external anchor. Say so explicitly in the
   header blockquote, the Splitting-strategy LB status, and the Benchmarking section, and
   benchmark against **all** graded agents found in the record set (e.g. NVIDIA, AIDE),
   noting their approaches are not recorded in the grade files.

The rest of the pipeline (5-section structure, rubric, `verify_report.py` against the DL
facts.json, `md2pdf.sh`) is unchanged. Reports live at `docs/ml_specs/<comp>.md` (+ `md/`,
`pdf/` copies) like every other lane. Existing examples: `ventilator-pressure-prediction`
(time-series BiLSTM), `us-patent-phrase-to-phrase-matching` (NLP transformers),
`siim-isic-melanoma-classification` (vision) — examples, not an exhaustive list.

## Notes

- The **vs-NVIDIA reproduce-agent** benchmark (Evaluation §) draws on the benchmark record
  (`docs/benchmark_facts.json` / the per-comp `ext_facts.json`); if no benchmark exists for a
  competition, say so and omit the comparison table rather than inventing a comparator.
- This skill replaced the older 7-section `kaggle-report` skill (2026-07-16); the shared
  engine (`collect.py` / `verify_report.py` / `md2pdf.sh` / `eda_summary.py` /
  `report_style.css`) was carried over unchanged.
