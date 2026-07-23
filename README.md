# Kaggle AI Agent — Hybrid Autonomous Competition Agent

A hybrid AI agent built from "Claude Code (LLM reasoning) + AutoML tools (LightGBM/XGBoost/CatBoost)"
that autonomously runs the full pipeline of a Kaggle tabular competition (understanding the problem → EDA → CV design → features → modeling →
ensembling → submission), and quantifies the contribution of each autonomous capability through **controlled ablation**. The method aligns with the ERA system of Aygün et al. (2026,
Nature).

## Where to Start Reading

| What you want | Read this |
|------|--------|
| Understand the whole project's results in 5 minutes | [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) (results brief) |
| The 15 competition reports + cross-competition results table | [docs/ml_specs/](docs/ml_specs/) (ML-spec reports, 顏佐榕 5-section framework) |
| Shared method, five stages, and per-stage operations | [docs/pipeline_stages_detail.md](docs/pipeline_stages_detail.md) |
| Research-style synthesis (method + results + honest limitations) | [docs/TECH_REPORT.md](docs/TECH_REPORT.md) (draft technical report) |
| Full analysis of a single competition | `docs/ml_specs/md/playground-series-<comp>.md` |
| How the agent operates a competition | [.claude/skills/kaggle-agent/SKILL.md](.claude/skills/kaggle-agent/SKILL.md) |

In-depth research documents: statistical significance [docs/statistical_rigor.md](docs/statistical_rigor.md); external-injection
attribution findings [docs/phase_j_j3_findings.md](docs/phase_j_j3_findings.md); method comparison with the ERA system of Aygün et al.
[docs/aygun_comparison.md](docs/aygun_comparison.md).

Every report has a corresponding PDF (with table of contents and page numbers).

## Core Method: Five-Stage Ablation

By solving the same competition multiple times, granting one additional capability each time, and then comparing scores, we can isolate the contribution of each capability stage by stage:

1. **Stage 1** no-skill baseline → 2. **+ kaggle-agent skill** (structured six-stage) →
3. **+ linear self-iteration** (experience-library priors, Optuna, seed bagging) →
4. **+ tree search** (candidate trees replace the single linear path; ERA's first pillar) →
5. **+ external idea injection** (literature idea bank; ERA's second pillar, injection mechanism built and empirically tested — but external injection shows no measurable systematic gain on the test competitions (honest null, see docs/phase_j_j3_findings.md))

## Main Results

- The Stage 1→4 ladder across **15 competitions** (10 same-season S3 main benchmark + 5 cross-season S4–S6, five metrics)
  **holds in every case**, with no competition regressing overall at any stage.
- On s3e16, the only competition with a real Kaggle leaderboard reference, the tree-search version improves in the same direction on both boards — external evidence of credibility.
- The recipe still holds across seasons (with shrinking gains); the best cross-competition solutions all pass the OOF bit-by-bit reproduction gate (some at the bit level).

## Repo Structure

```
competitions/playground-series-<comp>/   Per-competition workspace: data/ (gitignore), scripts/,
                                        experiments.json (raw records), facts.json,
                                        STATUS.md, config.yaml
tree_search/                            Self-built tree search: harness_v2/v3/v4 + per-competition driver
                                        run_<comp>_v3.py + eval_<comp>.py
knowledge/                              experience.md (internal experience library [INT]),
                                        idea_bank.md (external idea bank [EXT])
.claude/skills/                         kaggle-agent, kaggle-agent-self-improvement,
                                        kaggle-mlspec-report, kaggle-safe-submit, kaggle-vision-agent
docs/ml_specs/                          The 15 ML-spec competition reports (md + pdf) + results table + RUBRIC
docs/                                   Results brief, technical report, experiment log, pipeline-stage detail,
                                        benchmark_facts.json + build_benchmark_table.py
CLAUDE.md                               Project instructions (incl. uv package management, Kaggle CLI setup)
```

## How to Reproduce

**One-command environment build + self-check** (a lightweight alternative to Docker, freezing `uv.lock`):

```bash
bash setup.sh          # build the core environment + self-check the core ML stack
bash setup.sh --torch  # additionally install torch (only needed for vision/NLP competitions)
```

For full reproduction steps, data/credentials, known limitations, and "why not Docker," see **[REPRODUCE.md](REPRODUCE.md)**.

The following are the common reproduction instructions (all Python is run via **uv**, see CLAUDE.md):

```bash
# rebuild the cross-competition benchmark facts table (extracted from each competition's experiments.json, with consistency asserts)
uv run python3 docs/scripts/build_benchmark_table.py

# verify that all numbers in a report are traceable to facts.json (gate, exit 0 = pass)
uv run python3 .claude/skills/kaggle-mlspec-report/assets/verify_report.py \
    docs/ml_specs/md/playground-series-s6e1.md competitions/playground-series-s6e1/facts.json

# generate a PDF with table of contents / page numbers from Markdown
bash .claude/skills/kaggle-mlspec-report/assets/md2pdf.sh \
    docs/ml_specs/md/playground-series-s6e1.md docs/ml_specs/pdf/playground-series-s6e1.pdf

# reproduce a single competition's best solution (including the digit-for-digit OOF reproduction gate; produces a submission only on pass), using s5e10 as an example
uv run python3 competitions/playground-series-s5e10/scripts/06_rebuild_tree_best.py
```

Each competition's end-to-end reproduction is its `scripts/` (`04_train_blend.py` → `05_iterate.py` → `06_rebuild_tree_best.py`) plus the tree-search driver in `tree_search/`; see **[REPRODUCE.md](REPRODUCE.md)** for the consolidated steps.

## How to Use the Agent

In Claude Code, trigger the kaggle-agent skill (trigger words and workflow are in
[.claude/skills/kaggle-agent/SKILL.md](.claude/skills/kaggle-agent/SKILL.md)) and point it at a
competition workspace; a human can intervene and override decisions at every stage. For report generation, see the kaggle-mlspec-report skill.

The **packaging, installation steps, and dependencies** of the skills are in [.claude/skills/README.md](.claude/skills/README.md).

## Data and Credentials

- Competition data goes in each competition's `data/`, which is gitignored and not version-controlled.
- The Kaggle API token is always injected via the environment variable `KAGGLE_API_TOKEN`, and is **never written to a file** (see the security rules in CLAUDE.md).
