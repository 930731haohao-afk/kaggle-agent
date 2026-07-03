---
name: kaggle-agent
description: |
  Hybrid AI agent for Kaggle competitions combining Claude Code reasoning with Auto-ML tools (LightGBM, XGBoost, CatBoost, AutoGluon).
  Guides through full competition pipeline: ingestion, EDA, feature engineering, modeling, evaluation, and submission.

  Use this skill when the user wants to: work on a Kaggle competition, perform exploratory data analysis on competition data,
  engineer features for a Kaggle dataset, train and evaluate models for a competition, generate or improve a Kaggle submission,
  review experiment history and decide next steps, or optimize competition performance.

  Trigger phrases: "kaggle", "competition", "submission", "leaderboard", "kaggle agent", "train model", "feature engineering",
  "EDA", "cross-validation", "ensemble".

  Supports both tabular (tree models) and NLP (transformers) competitions.
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
export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('/home/tjyen/.kaggle/kaggle.json'))['key'])") && uv run kaggle <command>
```

Environment variables don't persist across separate Bash tool invocations in Claude Code.

## Core Workflow

Follow these stages sequentially. The user may start at any stage or repeat stages as needed.

### Experience Library (check first)

Before Stage 1 (EDA) and Stage 3 (Modeling), consult `knowledge/experience.md` at the project root
for validated cross-competition insights (indexed by metric/data-type, each entry evidence-backed).
Apply what transfers; log new validated insights back into it after major score changes.

### Stage 0: Competition Setup
**Goal**: Establish workspace and understand the competition.

See [references/01_setup.md](references/01_setup.md) for detailed instructions.

Key actions: Create workspace, config.yaml, inspect data files.

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
- **Ask before long operations**: Confirm if training will take >5 minutes
- **Use timestamped filenames**: Never overwrite good submissions
- **Validate before submitting**: Check format, shape, value ranges

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
- `01_setup.md` — Competition setup instructions
- `02_eda.md` — EDA instructions
- `03_features.md` — Feature engineering instructions
- `04_modeling.md` — Modeling instructions
- `05_evaluation.md` — Evaluation instructions
- `06_submission.md` — Submission instructions

## Important Notes

- **Environment**: Linux with NVIDIA GPU, Python 3.11, PyTorch 2.11.0 + CUDA 13.0
- **Path format**: Use forward slashes for paths
- **Python command**: Use `python3` (not `python`)
- **Package manager**: Always use `uv` for Python package management
- **Git**: Project is version controlled, use standard git workflow
