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
- **Single best model** — Simplest approach, use the model with the best CV score **among
  submission candidates only**
- **Ensemble** — If ensembling was done in Stage 3, use the ensemble
- **Multiple submissions** — If daily limit allows, submit both single model and ensemble

**Exclude diagnostics before ranking.** Not every logged experiment is a submission
candidate. Leakage probes, illegal-split runs and sanity checks are logged deliberately
and they usually score *highest*, because the thing being diagnosed is exactly what
inflates CV. Picking the top row blind is the study's documented silent failure #4: a
leakage probe explicitly marked not-for-submission won the selection.

Use the helper rather than sorting the log yourself — it already drops them:
```python
from utils.experiment_log import get_best_experiment, _entry_is_diagnostic

best = get_best_experiment('competitions/<name>', minimize=<True if lower is better>)
assert best is not None and not _entry_is_diagnostic(best), "champion is a diagnostic run"
```
`_entry_is_diagnostic` matches these markers, case-insensitively, across an entry's
`model` / `model_name` / `notes` / `tag` / `label` fields: `diagnostic`,
`not used for submission`, `not for submission`, `leakage check`, `leak check`,
`sanity probe`. So a diagnostic run is only excluded if it was *tagged* — when you log
one, put one of those strings in `notes`.

Verify explicitly before moving on: state which experiment ID was chosen, its score, and
that it is not diagnostic-tagged. If the only experiments left are diagnostics, stop and
say so — do not submit one. (2026-08-03 audit)

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

best_params = winning_entry["params"]   # experiments.json, v2 schema `params` field
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train_full, y_train_full)
predictions = model.predict(X_test)  # or predict_proba
```

### 3. Generate Predictions
- Run the final model on the test set
- Apply the same feature engineering pipeline used during training
- For classification with probabilities: generate both hard predictions and probabilities (user chooses which to submit)
- For regression: generate raw predictions

**INVERT THE TARGET TRANSFORM BEFORE WRITING ANYTHING.** This step used to say "generate raw
predictions" and stop, which is correct only when the model was trained on the raw target. If
Stage 2's injection ledger carries a `target_transform`, the model was NOT:

| `plan["target_transform"]["kind"]` | trained on | invert with |
|---|---|---|
| `ratio_log` | `log(target / covariate)` | `exp(pred) * covariate` |
| `ratio_linear` | `target / covariate` | `pred * covariate` |
| `log_offset` | `log1p(target)` | `expm1(pred)` |
| (absent)    | the raw target | nothing |

**A kind not in this table is a STOP, not a shrug.** `apply.py` writes
`ratio_{space}`, so the space parameter mints new kinds — `ratio_linear` was missing from the
first version of this table, and a dossier that chose `"space": "linear"` fell through the
"(absent)" row and shipped a submission in `y/covariate` units (~1e-4 of the target's scale)
that the validator passed because no `--train` was supplied (2026-08-07 re-verification). If
the kind you read is not listed here, do not guess and do not submit: the inversion is defined
by the transform, and an unlisted transform means this table is stale.

**Run the validator WITH the training target** — `--train train.csv --target <col>` — every
time. The Range check compares the submission against the training target's own range; without
`--train` it silently SKIPs, which is exactly how an un-inverted ratio submission
(values at ~1e-4 of the target's scale) got its "PASS — all checks clean".

The covariate column must be joined onto the TEST frame for the inversion, with the same
leakage rule the training join used — an inversion that reaches for a covariate value the
training side could not see is a leak introduced at the last step.

A submission written without its inversion is not a slightly worse submission; it is in the
wrong units, and every downstream check that compares it against the training target's range
is exactly the check that catches it. Read `competitions/<name>/injection_ledger.json`, state
in your report which transform you inverted (or that there was none), and run the submission
validator — its Range check exists for this.

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
Shape / columns / NaN / IDs are necessary but **not sufficient** — all four pass on inf
values, constant predictions, all-zero predictions, out-of-range values, and
probabilities sent to a hard-label metric, every one of which wastes a submission
(2026-08-03 audit). Run the full gate:

```bash
uv run python .claude/skills/kaggle-safe-submit/scripts/validate_submission.py \
  <submission.csv> <sample_submission.csv> --metric <metric> \
  --train <train.csv> --target <target_col>
```
Exit 0 = pass, 1 = structural (never submit), 2 = suspicious (stop, report to the user,
waive only with their agreement). The same two-tier gate is inlined in
`assets/templates/submit_template.py`.

- **Shape**: Matches sample submission exactly
- **Columns**: Same names and order as sample submission
- **IDs**: All test IDs are present, no extras, no duplicates, correct order
- **Values**: No NaN, no ±inf, within expected range, correct dtype
- **Not degenerate**: More than one unique value, not all zeros
- **Right value kind**: Probabilities for AUC/logloss, hard labels for accuracy/F1/QWK
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
