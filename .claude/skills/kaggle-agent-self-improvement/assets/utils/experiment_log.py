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
    # `open(w)` truncates first: a crash or full disk mid-dump leaves truncated JSON and the
    # run's whole history is gone. Write beside it and os.replace (2026-08-03 audit).
    tmp = f"{exp_file}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(experiments, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, exp_file)


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


_DIAGNOSTIC_MARKERS = ("diagnostic", "not used for submission", "not for submission",
                       "leakage check", "leak check", "sanity probe")


def _entry_score(entry: dict):
    """Score under either schema (v2 `score`, v1 `cv_mean`)."""
    for k in ("score", "cv_mean"):
        v = entry.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _entry_is_diagnostic(entry: dict) -> bool:
    """Did the run label this a diagnostic rather than a submission candidate?

    Champion selection that ignores the label reproduces the study's silent failure #4:
    a leakage probe explicitly marked not-for-submission won the aggregation.
    """
    hay = " ".join(str(entry.get(k, "")) for k in
                   ("model", "model_name", "notes", "tag", "label")).lower()
    return any(m in hay for m in _DIAGNOSTIC_MARKERS)


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

    # Reading only `cv_mean` raised KeyError on the v2 schema SKILL.md mandates, and the
    # maximize default silently inverted every minimize-metric competition; v2 records the
    # direction per entry, so use it when the log agrees (2026-08-03 audit).
    scored = [e for e in experiments if _entry_score(e) is not None]
    if not scored:
        return None
    scored = [e for e in scored if not _entry_is_diagnostic(e)] or scored
    dirs = {e.get("direction") for e in scored if e.get("direction")}
    if len(dirs) == 1:
        minimize = dirs.pop() == "minimize"
    return (min if minimize else max)(scored, key=_entry_score)


def print_leaderboard(competition_dir: str, top_n: int = 10):
    """Print a formatted leaderboard of experiments."""
    experiments = load_experiments(competition_dir)
    if not experiments:
        print("No experiments logged yet.")
        return

    metric = experiments[0].get("metric") or experiments[0].get("eval_metric", "unknown")
    dirs = {e.get("direction") for e in experiments if e.get("direction")}
    minimize = (dirs.pop() == "minimize") if len(dirs) == 1 else \
        metric in {"log_loss", "rmse", "mae", "rmsle", "smape", "mape", "mcrmse"}
    scored = [e for e in experiments if _entry_score(e) is not None]
    sorted_exps = sorted(scored, key=_entry_score, reverse=not minimize)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT LEADERBOARD ({metric}, {'lower' if minimize else 'higher'} is better)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'ID':<5} {'Model':<25} {'CV Mean':<10} {'CV Std':<10} {'Features':<8}")
    print(f"{'-'*70}")

    for rank, exp in enumerate(sorted_exps[:top_n], 1):
        std = exp.get("cv_std")
        std_s = f"{std:<10.6f}" if isinstance(std, (int, float)) else f"{'-':<10}"
        tag = "  [diagnostic]" if _entry_is_diagnostic(exp) else ""
        print(f"{rank:<4} {exp['experiment_id']:<5} {str(exp.get('model', '?'))[:24]:<25} "
              f"{_entry_score(exp):<10.6f} {std_s} {exp.get('n_features', '?'):<8}{tag}")

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
    **extra: Any,
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
    # Free-form extras. This is how the binding retrieval gate
    # (references/04_modeling.md §0) records `library_query` / `library_hits`, and how a
    # run records its own bookkeeping (stage, params, cv_strategy, cv_scores) without
    # hand-rolling an entry dict and re-forking the schema.
    # Canonical fields are reserved: an extras key that collides would silently rewrite the
    # entry's identity or its score, which every reader downstream trusts (2026-08-03 audit).
    clashing = sorted(set(entry) & set(extra))
    if clashing:
        raise ValueError(f"extras may not overwrite canonical fields: {clashing}")
    for key, val in extra.items():
        if val is not None:
            entry[key] = val
    experiments.append(entry)
    save_experiments(competition_dir, experiments)
    return entry["experiment_id"]
