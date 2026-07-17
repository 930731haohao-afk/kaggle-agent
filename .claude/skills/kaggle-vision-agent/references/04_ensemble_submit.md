# Stage V4 — Ensemble & submit

## The weight solve (thin and convex — no search tree needed)

Finding ensemble weights over cached OOFs is a small convex problem: minimize the metric of
`sum(w_i * OOF_i)` with `w >= 0, sum w = 1`. Solve it directly:

- Differentiable/continuous metric (logloss, RMSE): `scipy.optimize.minimize(..., method="SLSQP")`
  with the simplex constraint, or NNLS + renormalize.
- Discrete metric (accuracy): SLSQP on a smooth surrogate (logloss) first, then a small local
  search (coordinate steps) scored on the TRUE metric. Post-processing (argmax/threshold/rounding)
  MUST be inside the scoring function so every candidate weight vector is judged on the real
  metric — the discretized-metric lesson from tabular (s3e5/s3e16) applies verbatim.
- Sanity checks: best single model as the baseline; equal weights as a second baseline. If the
  solved blend does not beat the best solo on OOF, submit the solo (evidence: tabular s3e20 —
  unsearched blends can be worse).

## TTA (test-time augmentation)

Cheap, reliable gain: average predictions over a few deterministic views (identity + flips/crops
appropriate to the domain — again, no h-flip for digits/text). Validate TTA on OOF first: apply
the same TTA to validation folds and confirm it helps before applying to test.

## Post-processing

- Probability calibration only if the metric rewards it (logloss) AND OOF shows miscalibration.
- Class-prior adjustment if train/test distributions provably differ.
- Every post-processing decision validated on OOF with the full-pipeline score, logged.

## Submission

- Format per sample_submission.csv exactly (column names, id order, dtypes).
- Validate before writing: shape, no NaNs, label space matches, spot-check a few rows.
- Save to `competitions/<name>/submissions/` with a descriptive name; log the submission entry
  (score, config, blend weights) in experiments.json.
- Actual upload goes through the kaggle-safe-submit skill / CLI with the env-var token. Track the
  CV<->LB gap on the first submission — it calibrates trust in the local CV for the rest of the run.
