# ML Specification Reports — 22 Competitions (full 20-competition benchmark baseline + s3e20, s4e1)

Each competition is documented as a **report** against the 顏佐榕 ML-spec framework, satisfying the summer-project
plan's Goal 1: **(甲) 競賽目的 — what + why** (Overview + "Why it matters") and **(乙) 五大流程元件** — Data /
Models / Training / Inference / Evaluation. Presentation is narrative-first with tables only where genuinely tabular;
every figure is grounded in that competition's `facts.json` / `config.yaml` / `eda_summary.json` / `STATUS.md` /
`experiments_tree_v3.json`, and DL-only items with no tabular-GBDT analogue are marked *N/A (reason)*.

Each report has a Markdown source (`playground-series-<comp>.md`) and an individual academic-serif PDF
(`pdf/playground-series-<comp>.pdf`). Rubric validation: `RUBRIC.md` (all 15 pass 8/8 plan-aligned checks).

| # | Competition | Task | Metric | Our best (CV) | NVIDIA | Winner |
|---|-------------|------|--------|--------------:|-------:|--------|
| 1 | s3e1  | California Housing — median house value | RMSE ↓ | 0.5571 | 0.5086 | NVIDIA |
| 2 | s3e3  | Employee attrition | AUC ↑ | 0.8378 | 0.8758 | NVIDIA |
| 3 | s3e5  | Wine quality (ordinal) | QWK ↑ | 0.5677 | 0.5478 | **our agent** |
| 4 | s3e7  | Hotel reservation cancellation | AUC ↑ | 0.8999 | 0.9014 | tie |
| 5 | s3e9  | Concrete compressive strength | RMSE ↓ | 12.07 | 12.026 | tie |
| 6 | s3e11 | Media campaign cost | RMSLE ↓ | 0.29565 | 0.2956 | tie |
| 7 | s3e14 | Wild blueberry yield | MAE ↓ | 340.6 | 337.28 | NVIDIA |
| 8 | s3e16 | Crab age | MAE ↓ | 1.3356 | 1.3387 | **our agent** |
| 9 | s3e19 | Forecast mini-course sales | SMAPE ↓ | 10.02 | 13.72 | **our agent** |
| 10 | s3e20 | Rwanda CO2 emission | RMSE ↓ | 21.15 | 19.648 | NVIDIA |
| 11 | s4e1  | Bank customer churn | AUC ↑ | 0.894393 | 0.8984 | NVIDIA |
| 12 | s4e11 | Depression prediction | Accuracy ↑ | 0.9402 | 0.9399 | tie |
| 13 | s5e10 | Road accident risk | RMSE ↓ | 0.05597 | 0.05609 | tie |
| 14 | s6e1  | Student exam-score | R² ↑ | 0.78718 | 0.7863 | tie |
| 15 | s6e2  | Heart disease | AUC ↑ | 0.95553 | 0.9554 | tie |

Record vs NVIDIA reproduce-agent: **3 our-agent wins / 7 ties / 5 NVIDIA wins.** Scores are local CV (all
competitions closed → no live leaderboard); benchmark methodology and caveats (de-leaking, unified CV schemes) are
documented per report and in the four-axis evaluation.

> **Do not mix this CV-based record with the midterm report's LB-based standings.** The midterm report
> (`../MIDTERM_REPORT_202607.md`) scores the same competitions by *late-submission leaderboard* results with a
> paired significance test; this README's table is *local CV*. The two disagree in 10 of 13 comparable
> competitions (s3e19 fully reverses: CV says our agent wins by 27%, private LB says NVIDIA wins), because local
> CV schemes differ per agent and can leak (see the midterm report §3.4). Cite LB numbers for any
> between-agent claim; CV numbers here describe each pipeline's internal view only.

## Benchmark-baseline additions (added 2026-08-17)

The seven competitions below complete coverage of the frozen 20-competition benchmark baseline
(REPORT_v3/v4 §The Three-Way Benchmark). Unlike the CV-based table above, these rows quote the
**real private leaderboard** from the study's frozen three-way score table, per the rule stated in
the note — LB numbers are the ones fit for between-agent claims. PR = percentile rank.

| # | Competition | Metric | our private LB (PR) | NVIDIA | AIDE | LB winner |
|---|-------------|--------|--------------------:|-------:|-----:|-----------|
| 16 | afsis-soil-properties | MCRMSE ↓ | **0.49517** (46.6) | 0.69855 | 0.49861 | **our agent** |
| 17 | cat-in-the-dat | ROC-AUC ↑ | **0.80241** (73.1) | 0.77084 | 0.80217 | **our agent** |
| 18 | conway-s-reverse-game-of-life | MAE ↓ | **0.10875** (98.6) | 0.11189 | 0.11065 | **our agent** |
| 19 | tps-aug-2022 | ROC-AUC ↑ | 0.59055 (66.0) | 0.58730 | **0.59106** | AIDE |
| 20 | tps-jan-2022 | SMAPE ↓ | 6.46156 (71.5) | **4.63651** | 10.66638 | NVIDIA |
| 21 | s5e1 | MAPE ↓ | **0.12752** (76.5) | 0.14594 | 0.17917 | **our agent** |
| 22 | tps-sep-2022 | SMAPE ↓ | 28.54859 (14.6) | **5.48351** | 9.72095 | NVIDIA |

s5e1 and tps-sep-2022 ran the full v5 pipeline (dossier + external-data injection, knowledge library
pinned) and joined the frozen baseline on 2026-07-31; the other five ran in the 9a coverage phase.
tps-sep-2022 is our agent's known failure mode (wrong country-level regime bet, flagged as
undecidable-from-data before submission — see that report's Evaluation section).
