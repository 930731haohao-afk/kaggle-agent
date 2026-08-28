# Submission Checklist

<!--
 Note: Third-layer, on-demand resource for SKILL.md step 2 — read it only when validation
       fails or the competition type is unfamiliar. Before the 2026-08-03 audit
       SKILL.md pointed here but the file did not exist.
-->

Items marked **[auto]** are enforced by `scripts/validate_submission.py`; **[manual]**
items need the agent's own judgement (competition page / config.yaml).

## The two tiers

The validator separates failures into two tiers with distinct exit codes:

| Tier | Exit | Meaning | Waivable |
|---|---|---|---|
| STRUCTURAL | 1 | The file itself is broken — Kaggle will reject it, or rows align to the wrong IDs | never |
| SUSPICIOUS | 2 | The CSV is legal but almost certainly a bug (constant, all zeros, out of range, probabilities/labels swapped) | only by explicit waiver with `--allow-suspicious "reason"` |
| PASS | 0 | Passed | — |

Constant predictions are legal — and correct — on a handful of competitions (a degenerate
baseline, a single-class metric), which is why this tier can be waived; they are also the
single most common symptom of a broken pipeline, which is why it blocks by default and the
waiver must be written down rather than assumed — never passed silently.

## STRUCTURAL — must pass

- **[auto] Row count** exactly matches sample_submission (= number of test rows).
- **[auto] Column names and order** exactly match sample_submission, including case.
- **[auto] No duplicate IDs**.
- **[auto] ID alignment** — not just the same set, **the order must match too**. A misordered
  file is still accepted on most competitions and scored by ID alignment, but any ID type
  coercion (`1` vs `1.0` vs `"1"`) shifts the whole file out of alignment.
- **[auto] No NaN**.
- **[auto] No ±inf** — `isnull()` returns False for inf, so a gate that only checks NaN lets it through.
  inf usually comes from an unclipped log/division inverse transform.
- **[manual] File encoding and line endings** — UTF-8, no BOM, no trailing blank line (`to_csv(..., index=False)` is enough).
- **[manual] No extra index column** — forgetting `index=False` adds a column; the column check catches it.

## SUSPICIOUS — blocks by default

- **[auto] Constant predictions** (only one distinct value). Most common causes: the model was
  never fit, the wrong column was predicted, or the whole pipeline returned a fill value.
- **[auto] All zeros**. A placeholder `np.zeros(len(test))` that never got filled in.
- **[auto] Out of range**. Probabilities must lie in [0, 1]; otherwise the bound is the training
  target's range ±50% span, and exceeding it usually means a scale error (log space not restored,
  target transform not inverted, wrong column matched).
- **[auto] Probabilities / hard labels swapped** (both directions silently throw away score):
  - The metric takes hard labels (accuracy / F1 / QWK) but 0.73 was submitted → threshold or argmax first.
  - The metric takes probabilities (AUC / logloss) but 0/1 was submitted → `predict_proba` was
    written as `predict`; AUC collapses to a step value near 0.5.
- **[manual] Distribution clearly deviates from the training target**. The validator only prints
  `pred mean` vs `train target mean` without blocking; when they differ by an order of magnitude
  or more, go back and investigate yourself.

## Per-problem-type rules

### Binary classification
| Metric | Submit | Common failure |
|---|---|---|
| AUC / ROC-AUC | Positive-class probability (calibration optional; only ranking matters) | Submitting 0/1 hard labels — all ranking information is lost |
| LogLoss | Calibrated probabilities, recommended clipped to `[1e-15, 1-1e-15]` | Submitting 0 or 1 → infinite loss |
| Accuracy / F1 | Hard labels, threshold tuned on OOF | Submitting probabilities; or tuning the threshold on the test set (leakage) |

`--metric auc --expect probability` makes the validator enable the [0,1] check and the
"must not be all 0/1" check.

### Multiclass classification
- One probability column per class: column order **must** match sample_submission, and each row must sum to ≈ 1
  (**[manual]** — the validator currently checks only the last column; verify the row-sum yourself for multi-column probabilities).
- Single hard-label column: values must lie within the training label set (checkable only with `--train/--target`).
- Label type: if the sample uses string class names, submit strings — not 0/1/2 integer encodings.

### Regression
- If the target was transformed during training (log1p, Box-Cox, standardise), **always invert the transform after predicting**.
  This is what the range check catches most often.
- RMSLE competitions: predictions must not be negative (`np.clip(pred, 0, None)`), or Kaggle errors out immediately.
- If the target is an integer count, whether to round depends on the metric; for RMSE, usually do **not** round.

### Ordinal / QWK
- Submit integer grades, not continuous values. Threshold boundaries must be optimised on OOF and
  then frozen — never tuned on the test set.

## Three questions before the upload

1. Was this file produced by the best **non-diagnostic** experiment?
   (See step 1 of `references/06_submission.md` and `_entry_is_diagnostic` in `utils/experiment_log.py`.)
2. Is there enough of today's quota left? (`daily_submission_limit`)
3. Is this run in experiments.json, and will the LB score map back to it when it arrives?
