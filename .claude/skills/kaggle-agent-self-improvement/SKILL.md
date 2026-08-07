---
name: kaggle-agent-self-improvement
description: |
  DEPRECATED -- DO NOT USE. Superseded by the `kaggle-agent` skill; kept only because a
  handful of historical run logs cite this path. It carries a stale Stage 3 that instructs
  reading MEMORY.md and scanning competition directories, which is exactly the
  cross-competition contamination path the benchmark's isolation protocol forbids, and it
  predates Stage 0.5 (problem dossier), the typed injection contract, and the binding
  experience-library retrieval gate.

  If a Kaggle competition task arrives, invoke `kaggle-agent` instead. Never invoke this one.
---

# Kaggle Agent

> **DEPRECATED (2026-08-03).** Use `.claude/skills/kaggle-agent/` instead. This copy
> is retained for historical log references only; its Stage 3 instructions predate the
> isolation protocol and would read other competitions' records.

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

## Self-Improvement Strategies

This agent has 5 self-improvement strategies it can deploy at its own discretion. These are **not mandatory steps** — the agent evaluates the situation and decides whether a strategy would help. See [references/07_self_improvement.md](references/07_self_improvement.md) for full details.

### Decision Framework

At each stage transition and after each experiment, pause and ask yourself these questions:

**Before choosing an approach** (Stages 0-3):
- *"Have I seen a competition like this before?"* → If yes or unsure, consult **Experience Library** (MEMORY.md, past `competitions/*/STATUS.md`). If the competition type is familiar and you remember what worked, skip this.
- *"Am I confident in one approach, or should I hedge?"* → If uncertain, use **Best-of-N** to evaluate 3-5 diverse candidates. If one approach is clearly best from experience, go direct.

**After each experiment** (Stages 3-4):
- *"Did the score improve?"* → Track with **Verifiable Rewards**. Always compute: absolute score, delta vs baseline, delta vs best, trend over last 3 experiments. This is lightweight and should always happen.
- *"Did the result surprise me?"* → If yes (score went the wrong direction, or by an unexpected amount), apply **Reflexion** — write a structured self-critique to understand why. If the result was expected, a brief note suffices.
- *"Am I stuck?"* → If 3+ experiments with no improvement, trigger **Adaptive Search** — look externally for new ideas. If you still have untried ideas from your own reasoning, keep going.

**Before moving to submission** (Stage 4→5):
- *"Have I left obvious value on the table?"* → If the best model was found on iteration 1 and no alternatives were explored, consider whether Best-of-N or Adaptive Search would uncover something better. If multiple strategies have been tried and scores have plateaued, move on.

### Strategy Summary

| Strategy | The agent should use this when... | Skip when... |
|----------|----------------------------------|-------------|
| **Experience Library** | Unfamiliar competition type, or want to avoid past mistakes | Highly familiar domain with clear approach |
| **Verifiable Rewards** | Always — lightweight score tracking is always useful | Never skip this |
| **Reflexion** | Score degrades, unexpected result, or repeated failure | Score improved as expected |
| **Best-of-N** | Uncertain which approach to take, or starting fresh | One approach is clearly dominant |
| **Adaptive Search** | Stuck 3+ experiments, or completely new problem domain | Still have untried ideas from own reasoning |

### Autonomous Iteration

When the user doesn't specify a particular stage, the agent can run 3-5 improvement iterations autonomously, applying whichever strategies are relevant at each step. Stop when: plateau detected, max iterations reached, or a decision needs user input.

## Core Workflow

Follow these stages sequentially. The user may start at any stage or repeat stages as needed.

### Experience Library (check first)

Before Stage 1 (EDA) and Stage 3 (Modeling), consult the experience library ONLY via `python3 knowledge/query_library.py --query <terms> --comp <competition-slug>` — do not open `knowledge/experience.md` directly during a benchmark run (2026-08-10 sync with the primary skill).
for validated cross-competition insights (indexed by metric/data-type, each entry evidence-backed).
Apply what transfers; log new validated insights back into it after major score changes.

### Stage 0: Competition Setup
**Goal**: Establish workspace and understand the competition.

See [references/01_setup.md](references/01_setup.md) for detailed instructions.

Key actions: Create workspace, config.yaml, inspect data files. Consider consulting experience library if the competition type overlaps with past work.

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

Key actions: Establish baseline, run Auto-ML (AutoGluon/FLAML), log experiments, report CV scores. Consider Best-of-N if uncertain about approach; track verifiable rewards (score deltas) after each experiment.

### Stage 4: Evaluation & Iteration
**Goal**: Analyze results and decide on next steps.

See [references/05_evaluation.md](references/05_evaluation.md) for detailed instructions.

Key actions: Review experiments, analyze errors, propose improvements, track trajectory. Apply reflexion when results surprise you; use verifiable rewards to detect plateaus; consider adaptive search if stuck 3+ experiments.

**Tree Search (preferred for optimization)**: Once the linear Iteration Protocol above
has produced a baseline solo model plus at least one blend, switch to tree search — see
[references/07_tree_search.md](references/07_tree_search.md). Evidence: tree search beats
the linear iteration best in 9 of 10 benchmarked competitions (1 exact tie, 0 losses;
per-competition numbers live in the archived prototype report — do not open during a run).
Harness: `tree_search/harness_v3.py`; template driver: `tree_search/run_template_v3.py`
(competition-agnostic — never copy a per-competition run_*_v3.py; they embed recorded
configs). Keep the linear protocol above as the fallback for the very
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
      ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py")
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
- `07_self_improvement.md` — Self-improvement strategies (experience library, verifiable rewards, reflexion, best-of-N, adaptive search)
- `07_tree_search.md` — Tree search: agent-driven Stage 4 optimization loop (preferred once a baseline + blend exist)

## Important Notes

- **Environment**: Linux with NVIDIA GPU, Python 3.11, PyTorch 2.11.0 + CUDA 13.0
- **Path format**: Use forward slashes for paths
- **Python command**: Use `python3` (not `python`)
- **Package manager**: Always use `uv` for Python package management
- **Git**: Project is version controlled, use standard git workflow
