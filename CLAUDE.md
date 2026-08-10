# Claude Instructions — Kaggle AI Agent Project

## Project Overview

This project builds a **hybrid AI agent for Kaggle competitions** implemented as a **Claude Code Skill**. Claude Code itself acts as the agent, guided by the skill's structured instructions and domain knowledge. The hybrid approach combines:
- **LLM reasoning** (Claude Code) for creative decision-making, EDA interpretation, feature engineering, and strategy
- **Auto-ML tools** (AutoGluon, FLAML, etc.) invoked via Bash for systematic model search, hyperparameter tuning, and ensembling

The user triggers the skill, points it at a competition, and Claude Code autonomously works through the data science pipeline — from understanding the problem to generating a submission — while the user provides oversight and intervention at each stage.

## Architecture

### How It Works
The Kaggle skill provides Claude Code with a structured workflow and domain knowledge for competitions. Claude Code uses its built-in tools to execute every step:
- **Read / Glob / Grep** — Explore data files, configs, and past experiment logs
- **Bash** — Run Python scripts for EDA, training, Auto-ML, and submission generation
- **Write / Edit** — Create and modify Python scripts, configs, and reports
- **Task** — Delegate parallel work (e.g., run EDA while reading competition rules)
- **WebFetch** — Pull competition descriptions or dataset documentation

### Core Workflow (Skill-Guided)
```
1. Ingest competition  →  Read rules, metric, data description
2. EDA                 →  Generate & run analysis scripts, interpret output
3. Feature engineering →  Reason about features, write transformation code
4. Modeling            →  Invoke Auto-ML tools, review results
5. Evaluation          →  Compare experiments, decide next steps
6. Iterate             →  Refine features/models based on scores
7. Submit              →  Format and save submission file
```

### Hybrid Approach
- **Claude Code handles**: Problem understanding, validation strategy, creative feature engineering, interpreting results, deciding next steps, writing all code on-the-fly
- **Auto-ML tools handle**: Model selection, hyperparameter optimization, stacking/ensembling (invoked by Claude Code via Bash)
- **Human handles**: Approving strategies, providing domain knowledge, overriding decisions, setting compute budgets

## Directory Structure
```
kaggle\
├── CLAUDE.md                  # This file — project instructions
├── pyproject.toml             # Python dependencies managed by uv
├── .claude\
│   └── skills\
│       └── kaggle-agent\      # The Kaggle agent skill
│           ├── SKILL.md       # Skill definition and instructions
│           └── instructions\  # Detailed workflow instructions per stage
├── templates\                 # Reusable Python script templates
│   ├── eda_template.py        # Common EDA patterns
│   ├── feature_template.py    # Feature engineering scaffolding
│   ├── train_template.py      # Model training scaffolding
│   └── submit_template.py     # Submission formatting
├── utils\                     # Shared Python utility scripts
│   ├── data_loader.py         # Load and validate competition data
│   ├── evaluation.py          # Local scoring and CV utilities
│   └── experiment_log.py      # Log experiments to JSON/CSV
└── competitions\              # Per-competition workspaces
    └── <competition_name>\
        ├── config.yaml        # Competition-specific settings (metric, target, etc.)
        ├── data\              # Raw and processed data (gitignored)
        ├── scripts\           # Generated Python scripts for this competition
        ├── submissions\       # Generated submission files
        └── experiments.json   # Experiment history (params, scores, notes)
```

## Skill Design

### Skill Responsibilities
The skill (`SKILL.md`) should instruct Claude Code to:
1. **Ask** the user for the competition name/URL and download location of data
2. **Read** the competition config or create one interactively
3. **Follow the core workflow** step by step, reporting findings at each stage
4. **Track experiments** — log every model run with parameters, features, and scores
5. **Recommend next steps** — after each iteration, explain what to try and why
6. **Stay within bounds** — respect competition rules (external data, internet, etc.)

### Skill Invocation
The user should be able to trigger the skill with a slash command like:
- `/kaggle` — Start or resume work on a competition
- With options to target a specific stage (e.g., "run EDA", "try a new model", "generate submission")

## Guidelines

### Development Approach
- **Build the skill incrementally** — Start with a minimal SKILL.md, test it, then add sophistication
- **Templates are optional scaffolding** — Claude Code can write scripts from scratch, but templates speed up common patterns
- **Utils are convenience, not requirements** — Only create utility scripts when a pattern repeats across multiple competitions

### Code Standards
- **Language**: Python (all generated scripts)
- **Style**: PEP 8, type hints for function signatures, docstrings for public functions
- **Package Manager**: **uv** — Use `uv` for all Python package management (install, sync, lock, run)
  - `uv init` to initialize the project
  - `uv add <package>` to add dependencies (replaces `pip install`)
  - `uv sync` to install all dependencies from the lock file
  - `uv run <script>` to run scripts within the managed environment
  - Dependencies are tracked in `pyproject.toml` (not `requirements.txt`)
- **Logging**: Use Python's `logging` module in generated scripts

### Kaggle-Specific Rules
- **Respect competition rules** — Always check if external data, pretrained models, or internet access is allowed
- **Validation matters most** — A good local CV scheme is more important than a good model
- **Track public vs. private leaderboard** — Don't overfit to public LB
- **Keep submission count in mind** — Don't waste daily submissions on trivial changes
- **Start with tabular competitions** — Most common, well-understood; tackle vision/NLP later

### Data Handling
- **Never commit large data files** — Add `data\` folders to `.gitignore`
- **Document data sources** — Note where data came from and any preprocessing applied
- **Validate data early** — Check for nulls, duplicates, leakage before modeling

## Kaggle CLI Setup

### Prerequisites
- Install the Kaggle CLI: `uv add kaggle`
- Kaggle username: `tjyen1975`

### Credential Setup
The new-style Kaggle API tokens (`KGAT_` prefix) require the `KAGGLE_API_TOKEN` environment variable.

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings) > API > "Create New Token"
2. Copy the generated `KGAT_...` token
3. Set it as an environment variable before running Kaggle CLI commands:
   ```bash
   export KAGGLE_API_TOKEN=<your_token>
   ```
4. For persistent access, add the export line to your shell profile (`~/.bashrc` or `~/.zshrc`)

**Note**: The older `kaggle.json` approach (`~\.kaggle\kaggle.json`) does **not** work with `KGAT_` tokens. Always use the environment variable method.

### Security Rules
- **NEVER** store API tokens in project files, scripts, CLAUDE.md, or conversation history
- **NEVER** hardcode credentials in Python scripts — always use the `KAGGLE_API_TOKEN` environment variable
- If a token is accidentally exposed, rotate it immediately on Kaggle
- The environment variable must be set in the shell session before running any `kaggle` or `uv run kaggle` commands

### Common Kaggle CLI Commands
All commands require `KAGGLE_API_TOKEN` to be set in the environment.
```bash
# Download competition data
uv run kaggle competitions download -c <competition-name> -p competitions\<name>\data

# List competition files
uv run kaggle competitions files -c <competition-name>

# Submit predictions
uv run kaggle competitions submit -c <competition-name> -f <submission-file> -m "<message>"

# Check submission status
uv run kaggle competitions submissions -c <competition-name>

# List active competitions
uv run kaggle competitions list

# Download a dataset (non-competition)
uv run kaggle datasets download -d <owner>/<dataset-name> -p <path>
```

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-10 | Hybrid LLM + Auto-ML approach | Combines creative reasoning with systematic optimization |
| 2026-02-10 | Claude Code Skill (not standalone app) | Leverages existing Claude Code tools and reasoning loop; simpler to build; human-in-the-loop by default |
| 2026-02-10 | Python as primary language | Best ecosystem for ML/Kaggle (scikit-learn, pandas, AutoGluon, etc.) |
| 2026-02-10 | uv for package management | Fast, modern Python package manager; handles venv, dependencies, and lock files in one tool |

## Important Notes
- This is an experimental project — prioritize working code over perfect engineering
- The agent should be transparent about its reasoning — explain why it makes each decision
- Human-in-the-loop is a feature, not a limitation — allow easy intervention at every stage
- The skill can evolve — start simple and add workflow stages as we learn what works
