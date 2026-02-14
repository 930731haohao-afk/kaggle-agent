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


METRIC_FUNCTIONS = {
    "auc": lambda y, pred, prob: roc_auc_score(y, prob),
    "accuracy": lambda y, pred, prob: accuracy_score(y, pred),
    "f1": lambda y, pred, prob: f1_score(y, pred, average="macro"),
    "log_loss": lambda y, pred, prob: log_loss(y, prob),
    "rmse": lambda y, pred, prob: np.sqrt(mean_squared_error(y, pred)),
    "mae": lambda y, pred, prob: mean_absolute_error(y, pred),
    "r2": lambda y, pred, prob: r2_score(y, pred),
}

# Metrics where lower is better
MINIMIZE_METRICS = {"log_loss", "rmse", "mae"}


def compute_metric(y_true, y_pred, y_prob, metric_name: str) -> float:
    """Compute a single metric by name."""
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRIC_FUNCTIONS.keys())}")
    return METRIC_FUNCTIONS[metric_name](y_true, y_pred, y_prob)


def get_cv_splitter(
    problem_type: str,
    n_splits: int = 5,
    random_seed: int = 42,
    groups: Optional[np.ndarray] = None
):
    """Get the appropriate CV splitter based on problem type."""
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    elif problem_type in ["binary_classification", "multiclass_classification"]:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def find_best_threshold(y_true, y_prob, metric_name: str = "f1", n_steps: int = 100) -> dict:
    """
    Find the optimal classification threshold for binary classification.

    Returns:
        Dict with best_threshold, best_score, and all thresholds/scores tested.
    """
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
