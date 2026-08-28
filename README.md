# Sinica Kaggle Agent

An end-to-end AI agent that runs a Kaggle competition from raw data to submitted prediction with **no human in the loop**, benchmarked against two frozen reference agents across 23 competitions.

> **Result: best score on 13 of 23 competitions (13 wins, 10 losses)** — the best private score on 10 of the 20 ordinary competitions, and the three-way best on all 3 special ones. The reference agents split the remaining 10 (NVIDIA 7, AIDE 3).

Built during a summer internship at the Institute of Statistical Science, Academia Sinica (July–August 2026). PI: Dr. Tso-Jung Yen.

---

## How it works

An LLM (Claude Code) reasons and decides at every stage — problem identification, EDA, validation design, feature engineering, model choice, next step — while LightGBM / XGBoost / CatBoost / Optuna are invoked through the shell for fitting, and a tree search over complete candidate solutions is the optimization loop.

Three elements distinguish it from the reference agents:

**Stage 0.5, a problem dossier.** Before EDA the agent commits in writing to what kind of problem this is — task family, train/test window relation, split policy, external-data need — written to `dossier.json` and grounded in a structured library of task-level priors built from our own past runs. Reading competition-specific discussions or public kernels is forbidden, which keeps the method distinct from reproduction-based agents. The dossier is a prior, not a conclusion: EDA must verify its hypotheses and record a verdict.

**A cross-competition experience library.** Every insight carries an evidence field (competition, experiment, score before → score after). Entries without a measured delta are rejected, and an experiment record without a library-query trace is invalid — checked mechanically, not by convention.

**Self-documentation.** Every completed run emits a five-section ML specification report whose numbers are extracted deterministically from structured records; the LLM writes only the narrative and cannot introduce a figure. Fields the records don't contain are marked "not recorded".

## The benchmark

Two frozen reference agents: **AIDE** (LLM tree search over code, no cross-competition memory) and the **NVIDIA Kaggle Agent** (reproduces the highest-voted public kernel). Neither was modified, even where we found weaknesses.

23 competitions in two groups. The **20 ordinary** ones are tabular Kaggle competitions scored on the real private leaderboard via late submission — an agent's own cross-validation cannot arbitrate between agents, so no local CV number is quoted as evidence anywhere in the results tables. The **3 special** ones (NLP text pairing, physiological time series, medical imaging) are graded offline by MLE-bench.

Fairness was enforced structurally rather than by convention: each agent read only its own data root, lanes held no Kaggle credentials so a run could not reach the leaderboard, one lane executed at a time under a mutex, and a per-lane transcript audit read every tool call before a score was accepted. Two lanes were discarded by that audit and re-run clean.

## Repository map

| Path | What's in it |
|---|---|
| [`.claude/skills/kaggle-agent/`](.claude/skills/kaggle-agent/) | The agent itself: pipeline stages, rules it must follow, gates it cannot bypass |
| [`tree_search/`](tree_search/) | The optimization loop — `harness_v3.py` is the harness every benchmark run used |
| [`knowledge/`](knowledge/) | Experience library and the structured task-prior library the dossier consults |
| [`external_data/`](external_data/) | The injection layer: whitelisted sources, leakage-guarded as-of joins |
| [`benchmark_infra/`](benchmark_infra/) | Benchmark harness — isolated roots, lane mutex, transcript audit, scoring, fact collectors |
| [`competitions/<name>/`](competitions/) | One workspace per competition: scripts, `experiments*.json`, `dossier.json`, the scored submission |
| [`docs/ml_specs/`](docs/ml_specs/) | The 23 per-competition specification reports (Markdown + PDF) |
| [`benchmark_results/`](benchmark_results/) | Cross-agent score tables, coverage status, and 13 reports re-documenting the AIDE runs |
| [`docs/`](docs/) | The written report and its companions, the poster and talk, and research notes |
| [`tests/`](tests/) | The maintained test suite (380 tests) |

## Reading the results

- **[`docs/REPORT_v7.tex`](docs/REPORT_v7.tex)** — the main report: method, three-way benchmark, injection-layer experiments, limitations.
- **[`docs/ENGINEERING_LOG.tex`](docs/ENGINEERING_LOG.tex)** and **[`docs/CASE_STUDIES.tex`](docs/CASE_STUDIES.tex)** — companions carrying the depth the report condenses.
- **[`benchmark_results/rerun_three_way.csv`](benchmark_results/rerun_three_way.csv)** — the authoritative per-competition score table every headline number traces to.
- **[`docs/presentation/`](docs/presentation/)** — the A0 poster and the four-minute talk.

Earlier research documents, kept because they carry detail the report condenses and because later documents cite them — all predate the three-way benchmark, so where a competition count or headline differs, `REPORT_v7` is current:
[`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) (results brief),
[`docs/TECH_REPORT.md`](docs/TECH_REPORT.md) (the 15-competition ablation study),
[`docs/pipeline_stages_detail.md`](docs/pipeline_stages_detail.md) (per-stage operations),
[`docs/statistical_rigor.md`](docs/statistical_rigor.md) (the paired significance apparatus used during development),
[`docs/phase_j_j3_findings.md`](docs/phase_j_j3_findings.md) (external-injection attribution),
[`docs/aygun_comparison.md`](docs/aygun_comparison.md) (method comparison with the ERA system),
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) (the running experiment diary).

## Running it

```bash
bash setup.sh          # uv sync from the frozen uv.lock, then a self-check of the ML stack
uv run pytest -q       # the test suite
```

Full instructions, including how to re-derive the benchmark tables and how to check that every number in a specification report traces to its records, are in **[REPRODUCE.md](REPRODUCE.md)**.

To run the agent on a competition, invoke the `kaggle-agent` skill from Claude Code inside this repository; the skill file documents the stages and the gates.

## Honest notes

The three special competitions ran a documented fallback rather than the full method — one node there is a deep-learning fine-tune rather than a gradient-boosting refit, so a tree search over dozens of nodes would not fit the compute budget. Offline grading also yields one score per submission, with no public/private pair to corroborate it.

The sample is small and the standing is sensitive to its composition: reproduction-based and search-based methods win on disjoint kinds of competitions, so adding or removing a few competitions of one kind moves the win count. Reproduced kernels were also tuned by their authors against the realized public leaderboard, which makes the comparison least symmetric exactly where reproduction is strongest.

Two negative results are part of the record: no local signal we tested selects the form an external-data injection should take, and the transcript audit discarded a contaminated lane whose tainted score looked *better* than its clean replacement's.

---

*Private repository; access granted on request. Every number in the report traces to the structured records in this repository.*
