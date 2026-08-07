# Stage 3: Modeling

## Contents
1. [Establish Baseline](#1-establish-baseline)
2. [Auto-ML Search](#2-auto-ml-search)
3. [Log Experiments](#3-log-experiments)
4. [Analyze Results](#4-analyze-results)
5. [Manual Model Tuning (Optional)](#5-manual-model-tuning-optional)
6. [Ensembling](#6-ensembling)
7. [Recommend Next Steps](#7-recommend-next-steps)

## Objective
Train models using both a quick baseline and Auto-ML tools, tracking all experiments systematically.

## Steps

### 0. Retrieval Gate (before every new experiment)
Run `VIRTUAL_ENV= uv run python3 knowledge/query_library.py --query <terms>` with the
experiment's metric, data-type, and idea keywords; record `library_query` (the terms) and
`library_hits` (top results, or `none`) in the experiment entry BEFORE running it. The
self-exclusion HARD RULE from SKILL.md applies to the hits: skip any entry whose 證據 cites
the current competition. No trace, no experiment — `query_library.py --audit` enforces.

### 1. Establish Baseline
Before any Auto-ML, train a simple baseline to set a reference point:

**For classification:**
- Majority class predictor (accuracy baseline)
- Logistic Regression with default params (real baseline)
- Single LightGBM with default params

**For regression:**
- Mean/median predictor (naive baseline)
- Linear Regression (real baseline)
- Single LightGBM with default params

Log the baseline CV score to `experiments.json`:
```json
{
    "experiment_id": 1,
    "timestamp": "2026-02-10T14:30:00",
    "stage": "baseline",
    "model": "LightGBM-default",
    "features": "processed_v1",
    "params": {},
    "cv_strategy": "5-fold-stratified",
    "cv_scores": [0.81, 0.79, 0.82, 0.80, 0.81],
    "cv_mean": 0.806,
    "cv_std": 0.011,
    "notes": "Default LightGBM on processed features"
}
```

### 2. Auto-ML Search
Run Auto-ML to systematically search model space. Choose based on availability:

**AutoGluon (preferred for tabular):**
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label='<target>',
    eval_metric='<metric>',
    path='competitions/<name>/models/autogluon'
).fit(
    train_data,
    time_limit=<seconds>,  # Start with 600 (10 min), increase if needed
    presets='best_quality'  # or 'medium_quality' for faster iteration
)

leaderboard = predictor.leaderboard(test_data)
```

**FLAML (lighter alternative):**
```python
from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train, y_train,
    task='<classification|regression>',
    metric='<metric>',
    time_budget=<seconds>,
    eval_method='cv',
    n_splits=5
)
```

**Ask user before running** if time_limit > 600 seconds (10 minutes).

### 3. Log Experiments
After Auto-ML completes, log each model to `experiments.json`:
- Model type and hyperparameters
- CV score (mean and per-fold)
- Training time
- Feature set used

### 4. Analyze Results
Report to the user:
- **Leaderboard**: All models ranked by CV score
- **Baseline comparison**: How much did Auto-ML improve over baseline?
- **Model diversity**: Are the top models all tree-based, or is there variety?
- **Score distribution**: How much variance across folds?
- **Diminishing returns**: Is the gap between #1 and #5 small? (suggests feature engineering matters more than model tuning)

### 5. Manual Model Tuning (Optional)
If Auto-ML results suggest a particular model family works well, offer to fine-tune:
- **LightGBM/XGBoost**: learning_rate, num_leaves, max_depth, min_child_samples, reg_alpha, reg_lambda
- **CatBoost**: depth, l2_leaf_reg, learning_rate, iterations
- **Neural nets**: Only if data is large enough and Auto-ML flagged them as competitive

Use Optuna or manual grid search for focused tuning:
```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        # ... more params
    }
    # Train with CV and return mean score
    return cv_mean_score

study = optuna.create_study(direction='<maximize|minimize>')
study.optimize(objective, n_trials=50)
```

### 6. Ensembling
If multiple strong models exist, consider:
- **Simple averaging**: Average predictions from top N models (good starting point)
- **Weighted averaging**: Weight models by CV performance
- **Stacking**: Train a meta-model on out-of-fold predictions from base models

Log ensemble experiments the same way as individual models.

### 7. Recommend Next Steps
Based on results, recommend one of:
- **Iterate features** — If model scores are similar, features are the bottleneck
- **Iterate models** — If one model family clearly dominates, tune it further
- **Ensemble** — If multiple diverse models perform well
- **Submit** — If CV score is satisfactory and we've iterated enough

## Script Organization
Save modeling scripts to `competitions/<name>/scripts/`:
- `baseline.py` — Baseline models
- `automl_run.py` — Auto-ML execution
- `tune_<model>.py` — Manual tuning scripts
- `ensemble.py` — Ensembling script

## Completion Criteria
- Baseline model has been trained and logged
- Auto-ML has been run and results logged
- All experiments are in `experiments.json`
- Results have been analyzed and presented to the user
- Next step has been recommended and approved
