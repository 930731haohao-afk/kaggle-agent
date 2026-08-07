---
name: kaggle-agent
description: 'Hybrid AI agent for Kaggle competitions combining Claude Code reasoning with Auto-ML tools (LightGBM, XGBoost, CatBoost, AutoGluon). Guides the full competition pipeline (ingestion, EDA, feature engineering, modeling, evaluation, submission). Use when the user wants to work on a Kaggle competition, perform EDA on competition data, engineer features for a Kaggle dataset, train and evaluate models, generate or improve a Kaggle submission, review experiment history, or optimize competition performance. Trigger phrases include "kaggle", "competition", "submission", "leaderboard", "kaggle agent", "train model", "feature engineering", "EDA", "cross-validation", "ensemble". Supports both tabular (tree models) and NLP (transformers) competitions. Do NOT use this skill for general ML or statistics concept questions, for pandas/NumPy/Python debugging, or for data analysis not tied to a specific Kaggle competition workspace — answer those directly without engaging the pipeline; only invoke it when the task targets an actual competition workspace under competitions/.'
---

# Kaggle Agent

A hybrid AI agent that combines Claude Code's reasoning with Auto-ML tools for systematic optimization.

## Project Structure

This skill operates within a structured project workspace:

```
kaggle/                              # Project root
├── .claude/skills/kaggle-agent/     # This skill
│   ├── SKILL.md                     # Skill definition (this file)
│   ├── references/                  # Detailed stage instructions
│   │   ├── 01_setup.md             # Competition setup
│   │   ├── 02_eda.md               # Exploratory data analysis
│   │   ├── 03_features.md          # Feature engineering
│   │   ├── 04_modeling.md          # Model training
│   │   ├── 05_evaluation.md        # Evaluation & iteration
│   │   └── 06_submission.md        # Submission generation
│   └── assets/                      # Bundled resources
│       ├── templates/               # Python script templates
│       └── utils/                   # Utility scripts
└── competitions/                    # Per-competition workspaces
    └── <competition-name>/
        ├── config.yaml              # Competition metadata
        ├── STATUS.md                # Competition status doc (created after first submission)
        ├── data/                    # Raw and processed data
        ├── scripts/                 # Generated Python scripts
        ├── submissions/             # Generated submission files
        └── experiments.json         # Experiment tracking
```

## Configuration

- **Python**: Managed by `uv` — run all scripts with `uv run python3 <script>`
- **Dependencies**: Tracked in `pyproject.toml` at project root
- **Kaggle API**: Requires `KAGGLE_API_TOKEN` environment variable

### Kaggle API Authentication

The Kaggle CLI requires the `KAGGLE_API_TOKEN` environment variable. Always chain the export with kaggle commands:

```bash
source utils/kaggle_auth.sh && uv run kaggle <command>
```

Environment variables don't persist across separate Bash tool invocations in Claude Code.

### Pre-run lint gate (new scripts only)

Before executing any NEWLY WRITTEN competition script for the first time, lint it and fix
every finding — these rules flag code that produces wrong numbers silently:

```bash
uvx ruff@0.16.1 check <script> --select F821,F841,B023,B006,E722,PLW1510,RUF059 --isolated
```

Timing matters: a script is free to fix BEFORE its first logged run; after a run's score is
logged the script is FROZEN as the as-run record — never lint or edit it again (this is why
`competitions/` is excluded from the repo-level ruff config; the 2026-07-31 audit of 22
historical B023 hits found all benign, so frozen stays frozen). B023 in particular: a
closure defined in a loop is safe only if called in the same iteration; if it escapes
(stored callback, delayed execution), it silently uses the last iteration's values.

## Core Workflow

Follow these stages sequentially. The user may start at any stage or repeat stages as needed.

**Progressive disclosure — read on demand**: The per-stage detail lives in `references/NN_*.md`. Read each
reference file **only when you actually reach that stage** — do not preload them all. This keeps the working
context lean: metadata is always loaded, this SKILL.md loads when the skill triggers, and a reference file is
read only when its stage begins.

### Experience Library (check first)

Before Stage 1 (EDA) and Stage 3 (Modeling), consult the experience library — ONLY through
`python3 knowledge/query_library.py --query <terms> --comp <competition-slug>`. Do not open
`knowledge/experience.md` directly during a benchmark run: the raw file carries results in its
preamble and section headers that sit outside any tagged bullet, so no reading rule applied by
hand can filter them; the tool's whole-entry self-exclusion can (2026-08-07 audit). The library lives at the project root
for validated cross-competition insights (indexed by metric/data-type, each entry evidence-backed).
Apply what transfers; log new validated insights back into it after major score changes.

**RETRIEVAL GATE (binding, 2026-07-31).** Before creating any NEW experiment, run
`knowledge/query_library.py --query <metric/data-type/idea terms>` and record the query terms in
the experiment's `library_query` field (and top hits, or the literal string `none`, in
`library_hits`). An experiment record without a `library_query` trace is invalid;
`query_library.py --audit <experiments.json>` checks this. Why binding: on a benchmark
competition we re-derived a result the library already held because nothing forced a query,
and separately retrieved an over-claiming entry when a correct one sat beside it —
evidence-delta-ranked retrieval plus a mandatory trace is the fix for both (main report,
Open Problems).
Task-LEVEL priors (problem identification: split policy, external-data need) live in
`knowledge/knowledge_base.json` and are consumed by Stage 0.5 — **and the HARD RULE below
applies to them too**. An earlier design held them in a raw prose file outside the rule,
which made it an unfiltered door into the library: the lane-isolation constraint is enforced
on directories, but other agents' per-competition results were quoted inside the file itself,
so a lane could read another lane's winning mechanism without touching its workspace
(2026-08-04 architecture gate). Stage 0.5 reads the library ONLY through
`knowledge/task_priors_for.py <comp>`, which RENDERS a per-competition view from structured
facts: a prior whose evidence rests solely on the competition being solved is dropped
whole, Action included, because the Action is the answer.

**HARD RULE — never read the answers this competition already produced.** Every bullet ends in a
`證據: <competition>, exp #N, scoreA -> scoreB` citation. **Skip every bullet whose 證據 names the
competition you are currently solving**, even when the metric and tags match perfectly. By July 2026
the library held the recorded benchmark runs' own final feature sets and ensemble choices, so a
re-run that reads them unfiltered improves its score by recalling its own answer, and the
improvement is credited to whatever else changed. `harness_v2.suggest_priors` now enforces this
automatically — it requires a `comp` key in `comp_meta` and drops self-citing bullets, printing how
many it dropped — but the rule binds you too when you read the file directly. State in the STATUS
log which bullets you excluded for this reason.

### Stage 0: Competition Setup
**Goal**: Establish workspace and understand the competition.

See [references/01_setup.md](references/01_setup.md) for detailed instructions.

Key actions: Create workspace, config.yaml, inspect data files.

### Stage 0.5: Problem Dossier (upstream knowledge injection)
**Goal**: Identify what kind of problem this is before deep EDA — task family, train–test
window relation, split policy, external-data need.

See [references/00_problem_dossier.md](references/00_problem_dossier.md) for detailed instructions.

Key actions: Match the competition against the FILTERED prior library — `python3
knowledge/task_priors_for.py <comp>` ([TASK-*] entries; never the raw file during a benchmark run),
write `competitions/<comp>/dossier.json` (task family, split policy, external-data candidates
from the whitelist only, data-level and model-level injection ideas). The dossier is a prior,
not a conclusion — Stage 1 EDA must verify its hypotheses and record an `eda_verdict`.
Never read competition-specific discussions or kernels for this step.

### Stage 1: Exploratory Data Analysis (EDA)
**Goal**: Understand the data deeply before modeling.

See [references/02_eda.md](references/02_eda.md) for detailed instructions.

Key actions: Generate EDA script, analyze distributions/correlations, identify issues, recommend validation strategy.

### Stage 2: Feature Engineering
**Goal**: Create informative features based on EDA insights.

See [references/03_features.md](references/03_features.md) for detailed instructions.

Key actions: Propose features, get approval, implement and validate features, check importance.

### Stage 3: Modeling
**Goal**: Train models using Auto-ML and custom approaches.

See [references/04_modeling.md](references/04_modeling.md) for detailed instructions.

Key actions: Establish baseline, run Auto-ML (AutoGluon/FLAML), log experiments, report CV scores.

### Stage 4: Evaluation & Iteration
**Goal**: Analyze results and decide on next steps.

See [references/05_evaluation.md](references/05_evaluation.md) for detailed instructions.

Key actions: Review experiments, analyze errors, propose improvements, track trajectory.

**Tree Search (preferred for optimization)**: Once the linear Iteration Protocol above
has produced a baseline solo model plus at least one blend, switch to tree search — see
[references/07_tree_search.md](references/07_tree_search.md). Evidence: tree search beats
the linear iteration best in 9 of 10 benchmarked competitions (1 exact tie, 0 losses;
the archived prototype report (do not open during a run)). Harness: `tree_search/harness_v3.py`; example driver:
`tree_search/run_s3e7_v3.py`. Keep the linear protocol above as the fallback for the very
first iteration pass (before a baseline + blend exist) and for small/cheap-eval
competitions where a ~60-node search budget isn't worth it.

### Stage 5: Submission
**Goal**: Generate competition-ready submission file.

See [references/06_submission.md](references/06_submission.md) for detailed instructions.

Key actions: Retrain on full data, generate predictions, format submission, validate output.

## Behavioral Guidelines

### Transparency
- **Explain every decision**: Before running code, state what you're doing and why
- **Report results clearly**: After each script, summarize key takeaways
- **Flag concerns**: Warn about leakage, overfitting, or data issues immediately

### Iteration Protocol
- **Never repeat failed approaches**: Change something each iteration
- **Track everything**: Log all experiments with parameters, features, and scores
- **Experiment logging is MANDATORY via `experiment_log.log_experiment_v2()`** (assets/utils/experiment_log.py).
  Never hand-roll experiment dicts in training scripts — past hand-rolled entries created three
  incompatible schemas. Import it by path in generated scripts:

  ```python
  import importlib.util
  _spec = importlib.util.spec_from_file_location(
      "experiment_log",
      ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
  experiment_log = importlib.util.module_from_spec(_spec)
  _spec.loader.exec_module(experiment_log)
  experiment_log.log_experiment_v2(comp_dir, model=..., metric=..., direction=..., score=..., ...)
  ```
- **Compare against baseline**: Always report improvement relative to baseline

### Safety
- **Confirm before externally-visible / irreversible actions**: Before ANY Kaggle submission (each consumes a
  daily quota slot), dataset upload, or creating a **public** dataset, restate the competition + file + message
  to the user and get **explicit approval** first. Datasets default to private. For the actual submit, prefer the
  `kaggle-safe-submit` skill (it validates the CSV and checks remaining quota before uploading).
- **Never print/log/echo `KAGGLE_API_TOKEN`**: treat it as a secret; pass it only via the environment variable,
  never hardcode it into scripts, files, or messages.
- **Ask before long operations**: Confirm if training will take >5 minutes
- **Use timestamped filenames**: Never overwrite good submissions
- **Validate before submitting**: Check format, shape, value ranges (use `kaggle-safe-submit`)

### Resource Awareness
- **Start small**: Use samples for initial experiments, scale up when validated
- **Mind submission limits**: Don't waste daily submission quotas
- **Use appropriate tools**: Lighter models first, heavy ensembles when justified

## Quick Reference: Available Resources

### Templates (assets/templates/)
- `eda_template.py` — EDA script scaffold
- `feature_template.py` — Feature engineering scaffold
- `train_template.py` — Model training scaffold
- `submit_template.py` — Submission generation scaffold

### Utils (assets/utils/)
- `data_loader.py` — Load and validate competition data
- `evaluation.py` — Local scoring and CV utilities
- `experiment_log.py` — Log experiments to JSON
- `kaggle_auth.sh` — Kaggle API authentication helper

### Reference Files (references/)
- `00_problem_dossier.md` — Stage 0.5 problem dossier (upstream knowledge injection; consumes the filtered library via `knowledge/task_priors_for.py`)
- `01_setup.md` — Competition setup instructions
- `02_eda.md` — EDA instructions
- `03_features.md` — Feature engineering instructions
- `04_modeling.md` — Modeling instructions
- `05_evaluation.md` — Evaluation instructions
- `06_submission.md` — Submission instructions
- `07_tree_search.md` — Tree search: agent-driven Stage 4 optimization loop (preferred once a baseline + blend exist)

## Important Notes

- **Environment**: Linux with NVIDIA GPU, Python 3.11, PyTorch 2.11.0 + CUDA 13.0
- **Path format**: Use forward slashes for paths
- **Python command**: Use `python3` (not `python`)
- **Package manager**: Always use `uv` for Python package management
- **Git**: Project is version controlled, use standard git workflow
