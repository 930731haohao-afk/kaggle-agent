# ML Specification Reports — 15 Playground-Series Competitions

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
