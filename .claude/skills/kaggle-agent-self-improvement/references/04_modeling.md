# Stage 3: Modeling

## Contents
1. [Consult Experience Library](#1-consult-experience-library)
2. [Establish Baseline](#2-establish-baseline)
3. [Best-of-N Candidate Generation](#3-best-of-n-candidate-generation)
4. [Auto-ML Search](#4-auto-ml-search)
5. [Log Experiments](#5-log-experiments)
6. [Analyze Results](#6-analyze-results)
7. [Manual Model Tuning (Optional)](#7-manual-model-tuning-optional)
8. [Ensembling](#8-ensembling)
9. [Recommend Next Steps](#9-recommend-next-steps)

## Objective
Train models using both a quick baseline and Auto-ML tools, tracking all experiments systematically.

## Steps

### 1. Consult Experience Library
Before choosing an approach, check past competition results for similar problems:
- Read MEMORY.md for patterns from completed competitions
- Scan `competitions/*/STATUS.md` for competitions with same problem type/metric
- Note which models, features, and tricks worked best on similar data
- Present recommendations to user before proceeding

### 2. Establish Baseline
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

### 3. Best-of-N Candidate Generation
Instead of trying one approach at a time, generate 3-5 diverse candidates and evaluate all:

1. **Generate diverse candidates** — Ensure diversity across at least 2 axes:
   - Model family: LightGBM, XGBoost, CatBoost, Ridge/ElasticNet, Neural Network
   - Feature set: basic, advanced interactions, selected subset
   - Preprocessing: raw vs. scaled vs. transformed target

2. **Evaluate all candidates** with the same CV strategy

3. **Report ranked results** to the user:
   ```
   Candidate Results (5-fold CV):
   1. CatBoost + basic:     CV = 0.823  ← Best
   2. LGB + interactions:   CV = 0.819
   3. XGBoost + basic:      CV = 0.812
   Selected: CatBoost for iteration, LGB for ensemble diversity.
   ```

4. **Expand the top 1-2 candidates** — Tune hyperparameters, add features, etc.

This is especially valuable at the start of modeling and when stuck (combine with Adaptive Search results from evaluation stage).

### 4. Auto-ML Search
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

### 5. Log Experiments
After Auto-ML completes, log each model to `experiments.json`:
- Model type and hyperparameters
- CV score (mean and per-fold)
- Training time
- Feature set used

### 6. Analyze Results
Report to the user:
- **Leaderboard**: All models ranked by CV score
- **Baseline comparison**: How much did Auto-ML improve over baseline?
- **Model diversity**: Are the top models all tree-based, or is there variety?
- **Score distribution**: How much variance across folds?
- **Diminishing returns**: Is the gap between #1 and #5 small? (suggests feature engineering matters more than model tuning)

### 7. Manual Model Tuning (Optional)
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

### 8. Ensembling
If multiple strong models exist, consider:
- **Simple averaging**: Average predictions from top N models (good starting point)
- **Weighted averaging**: Weight models by CV performance
- **Stacking**: Train a meta-model on out-of-fold predictions from base models

Log ensemble experiments the same way as individual models.

### 9. Recommend Next Steps
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
