# kaggle-agent

A Claude Code skill that acts as a hybrid AI agent for Kaggle competitions, combining LLM reasoning with Auto-ML tools (LightGBM, XGBoost, CatBoost, AutoGluon).

## What It Does

Point the agent at a Kaggle competition and it autonomously works through the full data science pipeline:

1. **Setup** — Download data, read competition rules, configure workspace
2. **EDA** — Generate and run exploratory analysis, interpret findings
3. **Feature Engineering** — Design and implement features based on EDA insights
4. **Modeling** — Train models via Auto-ML, log experiments, report CV scores
5. **Evaluation** — Analyze errors, compare experiments, propose improvements
6. **Submission** — Retrain on full data, generate and validate submission files

Claude Code handles the reasoning (problem understanding, feature design, strategy decisions) while Auto-ML tools handle systematic optimization (model selection, hyperparameter tuning, ensembling).

## Installation

1. Clone this repo into your Claude Code skills directory:

   ```bash
   git clone https://github.com/tjyen/kaggle-agent.git ~/.claude/skills/kaggle-agent
   ```

2. Create the skill registration file:

   ```bash
   cp ~/.claude/skills/kaggle-agent/SKILL.md ~/.claude/skills/kaggle-agent.skill
   ```

   Or create `~/.claude/skills/kaggle-agent.skill` manually — see the [Claude Code docs](https://docs.anthropic.com/en/docs/claude-code) for details.

3. Ensure Python dependencies are available (the agent uses `uv` to manage packages at runtime).

4. Set up Kaggle API access:

   ```bash
   export KAGGLE_API_TOKEN=<your-kaggle-api-token>
   ```

## Usage

In Claude Code, invoke the skill with:

```
/kaggle
```

Or describe what you want naturally — the skill triggers on phrases like "kaggle", "competition", "submission", "train model", "feature engineering", "EDA", etc.

## Repo Structure

```
├── SKILL.md              # Skill definition and core workflow
├── assets/
│   ├── templates/        # Python script scaffolds
│   │   ├── eda_template.py
│   │   ├── feature_template.py
│   │   ├── train_template.py
│   │   └── submit_template.py
│   └── utils/            # Shared utility scripts
│       ├── data_loader.py
│       ├── evaluation.py
│       ├── experiment_log.py
│       └── kaggle_auth.sh
└── references/           # Detailed instructions per workflow stage
    ├── 01_setup.md
    ├── 02_eda.md
    ├── 03_features.md
    ├── 04_modeling.md
    ├── 05_evaluation.md
    └── 06_submission.md
```

## Supported Competition Types

- **Tabular** — Tree-based models (LightGBM, XGBoost, CatBoost) with AutoGluon/FLAML
- **NLP** — Transformer fine-tuning (DistilBERT, etc.)

## Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Kaggle API token

## License

This project is provided as-is for personal use.
