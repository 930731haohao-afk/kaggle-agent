"""
Evaluation Utility
===================
Scoring, cross-validation helpers, and threshold optimization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from typing import List, Optional, Callable


def _f1(y_true, y_pred) -> float:
    """F1 with the averaging the target actually calls for.

    This entry used to hardcode average="macro" for every problem. On a binary target
    that reports the unweighted mean of the positive AND negative class F1 -- a different
    (usually much higher) number than the F1 a binary competition is scored on, silently
    substituted under the same metric name (2026-08-03 audit). Two classes -> "binary" on
    the higher label; three or more -> "macro" as before.
    """
    labels = np.unique(y_true)
    if len(labels) == 2:
        pos_label = 1 if 1 in labels else labels[1]
        return f1_score(y_true, y_pred, average="binary", pos_label=pos_label)
    return f1_score(y_true, y_pred, average="macro")


METRIC_FUNCTIONS = {
    "auc": lambda y, pred, prob: roc_auc_score(y, prob),
    "accuracy": lambda y, pred, prob: accuracy_score(y, pred),
    "f1": lambda y, pred, prob: _f1(y, pred),
    "log_loss": lambda y, pred, prob: log_loss(y, prob),
    "rmse": lambda y, pred, prob: np.sqrt(mean_squared_error(y, pred)),
    "mae": lambda y, pred, prob: mean_absolute_error(y, pred),
    "r2": lambda y, pred, prob: r2_score(y, pred),
}

# Metrics where lower is better
MINIMIZE_METRICS = {"log_loss", "rmse", "mae"}

# Metrics computed from `prob` alone: the hard labels `pred` never enter them, so they
# take the SAME value at every threshold. find_best_threshold refuses these rather than
# returning the first grid point as if it had won (2026-08-03 audit).
THRESHOLD_INDEPENDENT_METRICS = {"auc", "log_loss"}


def compute_metric(y_true, y_pred, y_prob, metric_name: str) -> float:
    """Compute a single metric by name."""
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRIC_FUNCTIONS.keys())}")
    return METRIC_FUNCTIONS[metric_name](y_true, y_pred, y_prob)


# The problem_type vocabulary config.yaml offers (references/01_setup.md), split by the
# splitter each one needs. Anything outside these two sets is rejected -- see below.
STRATIFIED_PROBLEM_TYPES = {"binary_classification", "multiclass_classification"}
UNSTRATIFIED_PROBLEM_TYPES = {"regression", "multilabel", "ranking"}


def get_cv_splitter(
    problem_type: str,
    n_splits: int = 5,
    random_seed: int = 42,
    groups: Optional[np.ndarray] = None
):
    """Get the appropriate CV splitter based on problem type.

    `problem_type` must be one of the values config.yaml documents. The `else` branch used
    to swallow everything it didn't recognise into a plain unstratified KFold, so a typo
    ("binary", "classification") or a capitalised value silently dropped stratification
    from a classification competition's CV -- the validation scheme was then wrong for
    every experiment that used it, with nothing printed (2026-08-03 audit). Unknown values
    now raise, the same way compute_metric already rejects an unknown metric name.

    `multilabel` and `ranking` DO map to plain KFold, but as a stated decision rather than
    a fallthrough: sklearn ships no multilabel-stratified splitter, and a ranking task
    should normally pass `groups` (its query/session ids) instead.

    Note `groups` selects GroupKFold, which takes no shuffle/random_state -- `random_seed`
    has no effect on that branch (GroupKFold is deterministic). The array itself is only
    inspected for None here; it must be passed again to `splitter.split(X, y, groups)`.
    """
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    if problem_type in STRATIFIED_PROBLEM_TYPES:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    if problem_type in UNSTRATIFIED_PROBLEM_TYPES:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    raise ValueError(
        f"Unknown problem_type: {problem_type!r}. Available: "
        f"{sorted(STRATIFIED_PROBLEM_TYPES | UNSTRATIFIED_PROBLEM_TYPES)}. "
        f"(Pass `groups=` for a grouped split.)")


def find_best_threshold(y_true, y_prob, metric_name: str = "f1", n_steps: int = 100) -> dict:
    """
    Find the optimal classification threshold for binary classification.

    Raises ValueError for a threshold-independent metric (auc, log_loss). Those are
    computed from the probabilities alone, so every grid point scores identically, argmax
    returns index 0, and the function used to hand back thresholds[0] == 0.01 -- a number
    indistinguishable from a real optimum, and one that would binarise almost everything
    to 1 if a caller acted on it (2026-08-03 audit).

    When several thresholds tie for the best score the SMALLEST of them is returned; a tie
    is common on small validation sets, so treat the value as one of several equally good
    cut points rather than a sharp optimum.

    Returns:
        Dict with best_threshold, best_score, and all thresholds/scores tested.
    """
    if metric_name in THRESHOLD_INDEPENDENT_METRICS:
        raise ValueError(
            f"{metric_name!r} does not depend on the threshold (it is computed from the "
            f"probabilities, not the hard labels), so there is no threshold to optimize. "
            f"Pick a label metric -- "
            f"{sorted(set(METRIC_FUNCTIONS) - THRESHOLD_INDEPENDENT_METRICS)} -- or score "
            f"{metric_name!r} directly with compute_metric.")
    thresholds = np.linspace(0.01, 0.99, n_steps)
    scores = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = compute_metric(y_true, y_pred, y_prob, metric_name)
        scores.append(score)

    scores = np.array(scores)
    if metric_name in MINIMIZE_METRICS:
        best_idx = np.argmin(scores)
    else:
        best_idx = np.argmax(scores)

    return {
        "best_threshold": thresholds[best_idx],
        "best_score": scores[best_idx],
        "default_score": compute_metric(y_true, (y_prob >= 0.5).astype(int), y_prob, metric_name),
    }


def print_classification_report(y_true, y_pred, y_prob=None):
    """Print a detailed classification report."""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(pd.DataFrame(cm))

    if y_prob is not None and len(np.unique(y_true)) == 2:
        print(f"\nROC AUC: {roc_auc_score(y_true, y_prob):.6f}")
        print(f"Log Loss: {log_loss(y_true, y_prob):.6f}")


def print_regression_report(y_true, y_pred):
    """Print a detailed regression report."""
    residuals = y_true - y_pred

    print("\nRegression Report:")
    print(f"  RMSE:  {np.sqrt(mean_squared_error(y_true, y_pred)):.6f}")
    print(f"  MAE:   {mean_absolute_error(y_true, y_pred):.6f}")
    print(f"  R2:    {r2_score(y_true, y_pred):.6f}")
    print(f"\nResidual Statistics:")
    print(f"  Mean:  {residuals.mean():.6f}")
    print(f"  Std:   {residuals.std():.6f}")
    print(f"  Min:   {residuals.min():.6f}")
    print(f"  Max:   {residuals.max():.6f}")
