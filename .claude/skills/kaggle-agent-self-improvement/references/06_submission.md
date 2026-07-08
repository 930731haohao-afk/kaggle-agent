# Stage 5: Submission

## Contents
1. [Select Final Model(s)](#1-select-final-models)
2. [Retrain on Full Training Data](#2-retrain-on-full-training-data)
3. [Generate Predictions](#3-generate-predictions)
4. [Apply Post-Processing](#4-apply-post-processing)
5. [Format Submission](#5-format-submission)
6. [Validate Submission](#6-validate-submission)
7. [Save Submission](#7-save-submission)
8. [Report to User](#8-report-to-user)
9. [Document Competition Status](#9-document-competition-status)
10. [Update Project-Level Status](#10-update-project-level-status)

## Objective
Generate a valid, competition-ready submission file from the best model(s).

## Steps

### 1. Select Final Model(s)
Review `experiments.json` and select the model(s) to use for submission:
- **Single best model** — Simplest approach, use the model with the best CV score
- **Ensemble** — If ensembling was done in Stage 3, use the ensemble
- **Multiple submissions** — If daily limit allows, submit both single model and ensemble

Present the selection to the user for confirmation.

### 2. Retrain on Full Training Data
**Important**: The final model should be trained on ALL training data (not just the training folds):
- Combine all K folds (no validation holdout)
- Use the same hyperparameters as the best CV experiment
- For ensembles, retrain each base model on full data

**Caveat**: If the competition has a temporal component, retraining on full data might not be appropriate. Discuss with the user.

```python
# Example: retrain best LightGBM on full data
import lightgbm as lgb

best_params = {}  # From experiments.json
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train_full, y_train_full)
predictions = model.predict(X_test)  # or predict_proba
```

### 3. Generate Predictions
- Run the final model on the test set
- Apply the same feature engineering pipeline used during training
- For classification with probabilities: generate both hard predictions and probabilities (user chooses which to submit)
- For regression: generate raw predictions

### 4. Apply Post-Processing
If any post-processing was found helpful during evaluation:
- Threshold optimization (classification)
- Prediction clipping (regression)
- Rounding (if target is discrete)
- Calibration

### 5. Format Submission
Read the sample submission file and match its format exactly:
- Same column names
- Same column order
- Same ID column values (and same order)
- Correct dtype (int vs. float vs. string)
- No missing values
- Correct number of rows (must match test set)

```python
import pandas as pd

sample_sub = pd.read_csv('competitions/<name>/data/sample_submission.csv')
submission = pd.DataFrame({
    sample_sub.columns[0]: test_ids,       # ID column
    sample_sub.columns[1]: predictions      # Prediction column
})

# Validate
assert submission.shape == sample_sub.shape, f"Shape mismatch: {submission.shape} vs {sample_sub.shape}"
assert submission.isnull().sum().sum() == 0, "Submission contains NaN values"
assert (submission[sample_sub.columns[0]] == sample_sub[sample_sub.columns[0]]).all(), "ID mismatch"
```

### 6. Validate Submission
Run validation checks before saving:
- **Shape**: Matches sample submission exactly
- **Columns**: Same names and order as sample submission
- **IDs**: All test IDs are present, no extras, correct order
- **Values**: No NaN, within expected range, correct dtype
- **Sanity check**: Prediction distribution is similar to training target distribution (not required to match, but large deviations are suspicious)

### 7. Save Submission
Save with a descriptive, timestamped filename:
```
competitions/<name>/submissions/submission_<model>_<score>_<YYYYMMDD_HHMMSS>.csv
```

Example: `submission_lgbm_ensemble_0.856_20260210_143000.csv`

Also save a metadata file or add to `experiments.json`:
```json
{
    "submission_file": "submission_lgbm_ensemble_0.856_20260210_143000.csv",
    "model": "LightGBM ensemble (3 models)",
    "cv_score": 0.856,
    "features": "processed_v3",
    "retrained_on_full_data": true,
    "post_processing": "threshold=0.42",
    "notes": "Best ensemble from iteration round 3"
}
```

### 8. Report to User
Present:
- Submission file location
- Model used and its CV score
- Prediction distribution summary
- Reminder of daily submission limit
- Suggestion: submit and compare public LB score to CV score
  - If public LB >> CV: possible LB overfitting or validation issue
  - If public LB << CV: possible train-test distribution shift
  - If public LB ~ CV: validation strategy is working well

### 9. Document Competition Status
After each submission, create or update a `STATUS.md` file in the competition workspace (`competitions/<name>/STATUS.md`). This serves as a persistent summary of all work done on the competition.

The STATUS.md should include:

```markdown
# <Competition Name> — Competition Status

## Competition Info
- URL, problem description, metric, data size, train/test split details

## Current Best Score
- CV score and Public LB score (table format)

## Pipeline Summary
### Features
- Number of features, key engineered features, dropped features with reasons

### Models
- Table of all models trained with CV scores, ranked best to worst

### Submission
- Method used (single model / ensemble), post-processing, submission filename

## Key Observations
- Important patterns found during EDA and modeling
- CV vs. LB gap analysis

## Potential Improvements
- Ideas not yet tried that could improve the score

## Files
- Directory tree of all files in the competition workspace
```

**Rules:**
- Create `STATUS.md` after the first submission to Kaggle
- Update it after each subsequent submission with new scores and findings
- Keep it concise but complete enough to resume work in a new session

### 10. Update Project-Level Status
After updating the competition-level `STATUS.md`, also update the **project-level** `STATUS.md` in the project root (`kaggle/STATUS.md`).

This file tracks all competitions across sessions. Update it with:
- Add the new competition to the **All Competitions Summary** table (with Public LB, Private LB, status)
- Add a session entry or append to the current session with competition details (type, metric, CV score, Public LB, key notes)
- Update the session file list with the new competition workspace

**Rules:**
- Always update the project-level `STATUS.md` after every Kaggle submission
- Follow the existing format and table structure already in the file
- If a new session section is needed (first competition of the day), create one with the next session number

## Completion Criteria
- Submission file has been generated and saved
- All validation checks have passed
- Submission metadata has been logged
- User has been informed of the file location and model details
- STATUS.md has been created or updated in the competition workspace
- Project-level STATUS.md has been updated with the new competition entry
