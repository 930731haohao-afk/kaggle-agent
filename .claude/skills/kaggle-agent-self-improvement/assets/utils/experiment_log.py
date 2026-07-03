"""
Experiment Logger Utility
==========================
Track and manage experiments in experiments.json.
"""

import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any


def load_experiments(competition_dir: str) -> list:
    """Load experiment history from experiments.json."""
    exp_file = os.path.join(competition_dir, "experiments.json")
    if not os.path.exists(exp_file):
        return []
    with open(exp_file, "r") as f:
        return json.load(f)


def save_experiments(competition_dir: str, experiments: list):
    """Save experiment history to experiments.json."""
    exp_file = os.path.join(competition_dir, "experiments.json")
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)


def log_experiment(
    competition_dir: str,
    model_name: str,
    params: dict,
    cv_scores: List[float],
    eval_metric: str,
    features_file: str = "",
    n_features: int = 0,
    cv_strategy: str = "5-fold",
    notes: str = ""
) -> int:
    """
    Log a new experiment and return its ID.

    Returns:
        The experiment ID.
    """
    experiments = load_experiments(competition_dir)
    exp_id = len(experiments) + 1

    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "features": features_file,
        "n_features": n_features,
        "params": {k: str(v) for k, v in params.items()},
        "cv_strategy": cv_strategy,
        "cv_scores": [round(s, 6) for s in cv_scores],
        "cv_mean": round(float(sum(cv_scores) / len(cv_scores)), 6),
        "cv_std": round(float(
            (sum((s - sum(cv_scores)/len(cv_scores))**2 for s in cv_scores) / len(cv_scores))**0.5
        ), 6),
        "eval_metric": eval_metric,
        "notes": notes
    }

    experiments.append(experiment)
    save_experiments(competition_dir, experiments)
    return exp_id


def get_best_experiment(
    competition_dir: str,
    metric: Optional[str] = None,
    minimize: bool = False
) -> Optional[dict]:
    """
    Get the best experiment by CV mean score.

    Args:
        competition_dir: Path to competition directory.
        metric: Filter by eval_metric (optional).
        minimize: If True, lower is better.

    Returns:
        The best experiment dict, or None if no experiments exist.
    """
    experiments = load_experiments(competition_dir)
    if not experiments:
        return None

    if metric:
        experiments = [e for e in experiments if e.get("eval_metric") == metric]

    if not experiments:
        return None

    if minimize:
        return min(experiments, key=lambda e: e["cv_mean"])
    else:
        return max(experiments, key=lambda e: e["cv_mean"])


def print_leaderboard(competition_dir: str, top_n: int = 10):
    """Print a formatted leaderboard of experiments."""
    experiments = load_experiments(competition_dir)
    if not experiments:
        print("No experiments logged yet.")
        return

    metric = experiments[0].get("eval_metric", "unknown")
    minimize = metric in {"log_loss", "rmse", "mae"}
    sorted_exps = sorted(experiments, key=lambda e: e["cv_mean"], reverse=not minimize)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT LEADERBOARD ({metric}, {'lower' if minimize else 'higher'} is better)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'ID':<5} {'Model':<25} {'CV Mean':<10} {'CV Std':<10} {'Features':<8}")
    print(f"{'-'*70}")

    for rank, exp in enumerate(sorted_exps[:top_n], 1):
        print(f"{rank:<4} {exp['experiment_id']:<5} {exp['model'][:24]:<25} "
              f"{exp['cv_mean']:<10.6f} {exp['cv_std']:<10.6f} {exp.get('n_features', '?'):<8}")

    print(f"\nTotal experiments: {len(experiments)}")


def log_experiment_v2(
    competition_dir: str,
    model: str,
    metric: str,
    direction: str,
    score: float,
    cv: Optional[dict] = None,
    features: Optional[List[str]] = None,
    base_models: Optional[List[dict]] = None,
    ensemble: Optional[dict] = None,
    postprocess: Optional[List[str]] = None,
    submission: Optional[str] = None,
    leaderboard: Optional[dict] = None,
    notes: str = "",
) -> int:
    """
    Log an experiment in the canonical v2 schema (spec:
    docs/superpowers/specs/2026-07-03-kaggle-report-skill-design.md §6).

    Required: model, metric, direction ("minimize"/"maximize"),
    score (final representative score, post-processed).
    Optional fields are omitted from the JSON entry when not given.
    Returns the experiment_id.
    """
    experiments = load_experiments(competition_dir)
    entry = {
        "schema_version": 2,
        "experiment_id": len(experiments) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "metric": metric,
        "direction": direction,
        "score": round(float(score), 6),
    }
    if features is not None:
        entry["features"] = list(features)
        entry["n_features"] = len(features)
    for key, val in (("cv", cv), ("base_models", base_models), ("ensemble", ensemble),
                     ("postprocess", postprocess), ("submission", submission),
                     ("leaderboard", leaderboard)):
        if val is not None:
            entry[key] = val
    if notes:
        entry["notes"] = notes
    experiments.append(entry)
    save_experiments(competition_dir, experiments)
    return entry["experiment_id"]
