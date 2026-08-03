"""
Submission Generation Template for Kaggle Competitions
=======================================================
Usage: Modify the CONFIG section below, then run the script.
Retrains on full data and generates a formatted submission file.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — Modify these for each competition
# ============================================================
COMPETITION_DIR = "competitions/<name>"
TRAIN_FILE = "data/train_processed.csv"
TEST_FILE = "data/test_processed.csv"
SAMPLE_SUB_FILE = "data/sample_submission.csv"
TARGET_COL = "<target>"
ID_COL = "<id>"
PROBLEM_TYPE = "binary_classification"  # binary_classification, multiclass_classification, regression
USE_PROBABILITIES = True  # For classification: submit probabilities instead of hard labels
RANDOM_SEED = 42

# Escape hatch for the SUSPICIOUS validation tier below (see VALIDATION section).
# A constant / out-of-range / hard-label submission is legal on some competitions but is
# almost always a bug, so it blocks by default. To override you must set the flag AND
# write down why — an empty reason leaves the block in place, so the override cannot be
# flipped on reflexively (2026-08-03 audit).
ALLOW_SUSPICIOUS_PREDICTIONS = False
SUSPICIOUS_OVERRIDE_REASON = ""
# ============================================================

# Load data
train = pd.read_csv(os.path.join(COMPETITION_DIR, TRAIN_FILE))
test = pd.read_csv(os.path.join(COMPETITION_DIR, TEST_FILE))
sample_sub = pd.read_csv(os.path.join(COMPETITION_DIR, SAMPLE_SUB_FILE))

y_train = train[TARGET_COL]
X_train = train.drop(columns=[TARGET_COL, ID_COL])
test_ids = test[ID_COL]
X_test = test.drop(columns=[ID_COL])

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
print(f"Sample submission columns: {list(sample_sub.columns)}")


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================
# Replace this section with the best model and params from experiments.json

import lightgbm as lgb

best_params = {
    # Fill in from experiments.json
}

print("\nTraining final model on full training data...")

if PROBLEM_TYPE == "binary_classification":
    model = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **best_params)
elif PROBLEM_TYPE == "multiclass_classification":
    model = lgb.LGBMClassifier(verbosity=-1, random_state=RANDOM_SEED, **best_params)
else:
    model = lgb.LGBMRegressor(verbosity=-1, random_state=RANDOM_SEED, **best_params)

model.fit(X_train, y_train)
print("Model trained.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

if PROBLEM_TYPE in ["binary_classification", "multiclass_classification"]:
    if USE_PROBABILITIES:
        predictions = model.predict_proba(X_test)
        if PROBLEM_TYPE == "binary_classification":
            predictions = predictions[:, 1]
    else:
        predictions = model.predict(X_test)
else:
    predictions = model.predict(X_test)

print(f"\nPrediction stats:")
print(f"  Shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
print(f"  Min: {np.min(predictions):.6f}")
print(f"  Max: {np.max(predictions):.6f}")
print(f"  Mean: {np.mean(predictions):.6f}")


# ============================================================
# POST-PROCESSING (customize as needed)
# ============================================================

# Example: clip predictions to valid range
# predictions = np.clip(predictions, 0, 1)

# Example: threshold optimization for binary classification
# predictions = (predictions > 0.42).astype(int)


# ============================================================
# FORMAT SUBMISSION
# ============================================================

submission = pd.DataFrame({
    sample_sub.columns[0]: test_ids,
    sample_sub.columns[1]: predictions
})


# ============================================================
# VALIDATION
# ============================================================
# The gate has two tiers (2026-08-03 audit). The original gate was shape / columns /
# NaN / IDs only — and all four of those PASS on inf values, all-constant predictions,
# all-zero predictions, out-of-range values, and probabilities sent to a competition
# scored on hard labels. Each of those is a wasted daily submission, so they are checked
# here too:
#   STRUCTURAL — the file is malformed; Kaggle rejects it or scores the wrong rows.
#                Never overridable.
#   SUSPICIOUS — the file is a legal CSV that is almost certainly a bug. A constant
#                column IS a valid submission on a handful of competitions (and is the
#                correct answer for a degenerate metric), so this tier is overridable —
#                but loudly and explicitly, via ALLOW_SUSPICIOUS_PREDICTIONS plus a
#                written reason, never by passing silently.
# Canonical standalone implementation of the same rules (usable on any CSV, not just
# one this template produced):
#   .claude/skills/kaggle-safe-submit/scripts/validate_submission.py
# Keep the two in sync when a rule changes.

print("\nValidation checks:")

structural_failures = []
suspicious_failures = []


def check(failures, name, ok, detail=""):
    """Record + print one check. `failures` is the tier this check belongs to."""
    print(f"  {name}: {'PASS' if ok else 'FAIL'} {detail}".rstrip())
    if not ok:
        failures.append(f"{name} — {detail}" if detail else name)
    return ok


id_col, pred_col = sample_sub.columns[0], sample_sub.columns[1]

# ---- STRUCTURAL ----
check(structural_failures, "Shape", submission.shape == sample_sub.shape,
      f"(got {submission.shape}, expected {sample_sub.shape})")
check(structural_failures, "Columns", list(submission.columns) == list(sample_sub.columns),
      f"(got {list(submission.columns)}, expected {list(sample_sub.columns)})")
nan_count = int(submission.isnull().sum().sum())
check(structural_failures, "NaN", nan_count == 0, f"({nan_count} NaN values)")
id_match = (len(submission) == len(sample_sub) and
            (submission[id_col].values == sample_sub[id_col].values).all())
check(structural_failures, "IDs", bool(id_match),
      "" if id_match else "(order and values must match the sample submission)")

preds_out = submission[pred_col]
is_numeric = pd.api.types.is_numeric_dtype(preds_out)
finite_mask = (np.isfinite(preds_out.to_numpy(dtype=float)) if is_numeric
               else np.ones(len(preds_out), bool))
# inf survives isnull() untouched — that is exactly how it reached Kaggle through a
# NaN-only gate. isfinite() is False for NaN as well, so subtract the NaNs back out to
# report inf on its own.
pred_nan = int(preds_out.isnull().sum())
n_inf = int((~finite_mask).sum()) - pred_nan
check(structural_failures, "Finite", n_inf <= 0, f"({max(n_inf, 0)} +/-inf values)")

# ---- SUSPICIOUS ----
n_unique = int(preds_out.nunique(dropna=False))
check(suspicious_failures, "Variance", n_unique > 1,
      f"({n_unique} unique value(s)"
      + (f", constant = {preds_out.iloc[0]}" if n_unique == 1 else "") + ")")

all_zero = bool(is_numeric and len(preds_out) and (preds_out.to_numpy(dtype=float) == 0).all())
check(suspicious_failures, "Non-zero", not all_zero,
      "(every prediction is exactly 0 — a placeholder array was probably never filled in)")

# Reference range: probabilities live in [0, 1]; anything else is judged against the
# training target's own range, widened by half its span so that a genuinely wider test
# distribution does not trip the gate but a unit/scale error (log-space, un-inverted
# transform, wrong column) does.
expects_probability = (PROBLEM_TYPE in ("binary_classification", "multiclass_classification")
                       and USE_PROBABILITIES)
y_is_numeric = pd.api.types.is_numeric_dtype(y_train)
if is_numeric and finite_mask.any() and (expects_probability or y_is_numeric):
    vals = preds_out.to_numpy(dtype=float)[finite_mask]
    if expects_probability:
        lo, hi, range_desc = 0.0, 1.0, "[0, 1] probabilities"
    else:
        t_lo, t_hi = float(np.min(y_train)), float(np.max(y_train))
        span = (t_hi - t_lo) or max(abs(t_hi), 1.0)
        lo, hi = t_lo - 0.5 * span, t_hi + 0.5 * span
        range_desc = f"[{lo:.6g}, {hi:.6g}] (train target range +/-50% span)"
    n_out = int(((vals < lo) | (vals > hi)).sum())
    check(suspicious_failures, "Range", n_out == 0,
          f"({n_out} values outside {range_desc}; observed "
          f"[{vals.min():.6g}, {vals.max():.6g}])")
else:
    print("  Range: SKIP (no numeric reference range)")

# Hard-label vs probability mismatch. Submitting 0.73 to a competition scored on the
# label itself, or a hard 0/1 to an AUC/logloss competition, both validate as clean CSVs
# and both throw the score away.
if is_numeric and finite_mask.any():
    vals = preds_out.to_numpy(dtype=float)[finite_mask]
    train_labels = set(np.unique(y_train).tolist())
    if PROBLEM_TYPE != "regression" and not USE_PROBABILITIES:
        stray = sorted({v for v in np.unique(vals).tolist() if v not in train_labels})[:5]
        check(suspicious_failures, "Label set", not stray,
              f"(hard labels expected, got values outside the training label set: {stray})")
    elif expects_probability:
        looks_binary = set(np.unique(vals).tolist()) <= {0.0, 1.0}
        check(suspicious_failures, "Probability-valued", not looks_binary,
              "(metric expects probabilities but every value is 0 or 1 — "
              "predict_proba was probably replaced by predict)")

# Distribution sanity — reported, never gated (06_submission.md step 6).
if is_numeric and y_is_numeric and finite_mask.any():
    print(f"  [info] pred mean {np.mean(preds_out.to_numpy(dtype=float)[finite_mask]):.6f} "
          f"vs train target mean {float(np.mean(y_train)):.6f}")

override_active = bool(ALLOW_SUSPICIOUS_PREDICTIONS and SUSPICIOUS_OVERRIDE_REASON.strip())
if suspicious_failures and ALLOW_SUSPICIOUS_PREDICTIONS and not override_active:
    print("\nALLOW_SUSPICIOUS_PREDICTIONS is set but SUSPICIOUS_OVERRIDE_REASON is empty — "
          "override ignored.")

all_pass = not structural_failures and (not suspicious_failures or override_active)


# ============================================================
# SAVE
# ============================================================

if all_pass:
    if suspicious_failures:
        print(f"\nOVERRIDE: {len(suspicious_failures)} suspicious check(s) bypassed — "
              f"{SUSPICIOUS_OVERRIDE_REASON.strip()}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}.csv"
    output_path = os.path.join(COMPETITION_DIR, "submissions", filename)
    os.makedirs(os.path.join(COMPETITION_DIR, "submissions"), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")
    print(f"Rows: {len(submission)}")
else:
    print("\nWARNING: Validation failed! Submission NOT saved.")
    for f in structural_failures:
        print(f"  [structural] {f}")
    for f in suspicious_failures:
        print(f"  [suspicious] {f}")
    if suspicious_failures and not structural_failures:
        print("Fix the issues above, or — only if this really is the intended submission —"
              " set ALLOW_SUSPICIOUS_PREDICTIONS = True and state why in "
              "SUSPICIOUS_OVERRIDE_REASON.")
    else:
        print("Fix the issues above before saving.")
