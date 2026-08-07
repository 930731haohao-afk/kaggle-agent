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


# Markers that ASSERT an entry is not a submission candidate. Split into two tiers because the
# single list matched free-text notes: an entry whose notes said "passed the leakage check" was
# classified a diagnostic and silently dropped from champion selection, so a legitimate winner
# lost to a weaker model for mentioning a check it had passed (2026-08-04 architecture gate).
#
# Tier 1 asserts exclusion and is safe anywhere, including prose.
_EXCLUSION_PHRASES = ("not used for submission", "not for submission", "diagnostic only",
                      "do not submit", "excluded from selection", "not a candidate")
# Tier 2 are topic words. A short, deliberate LABEL field of "diagnostic" means it; the same
# word inside a sentence does not. Never matched against `notes`.
_DIAGNOSTIC_LABELS = ("diagnostic", "leakage check", "leak check", "sanity probe", "probe")


def _jsonable(v):
    """Coerce to something json.dump accepts, losing as little as possible."""
    if isinstance(v, (bool, int, float, str, type(None))):
        return v
    if hasattr(v, "item") and not hasattr(v, "__len__"):     # numpy scalar
        try:
            unwrapped = v.item()
        except Exception:  # noqa: BLE001
            return str(v)
        # np.datetime64.item() returns datetime.date/datetime, np.timedelta64.item() returns
        # timedelta -- none JSON-serializable, so the "sanitized" entry still crashed
        # json.dump after training (2026-08-07 round-4). Recurse: non-JSON unwraps fall
        # through to str().
        if isinstance(unwrapped, (bool, int, float, str, type(None))):
            return unwrapped
        return _jsonable(unwrapped)
    if isinstance(v, dict):
        return {str(k) if not isinstance(k, str) else k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):                                  # numpy array
        try:
            return v.tolist()
        except Exception:  # noqa: BLE001
            return str(v)
    import datetime as _dt
    if isinstance(v, (_dt.date, _dt.datetime, _dt.timedelta)):
        return str(v)
    return str(v)


def _entry_score(entry: dict):
    """Score under either schema (v2 `score`, v1 `cv_mean`)."""
    import math as _math
    for k in ("score", "cv_mean"):
        v = entry.get(k)
        # Non-finite counts as no score: nan compares False with everything, so letting it
        # through made the champion depend on entry ORDER (2026-08-07 round-3).
        if isinstance(v, (int, float)) and _math.isfinite(v):
            return float(v)
    return None


def _entry_is_diagnostic(entry: dict) -> bool:
    """Did the run label this a diagnostic rather than a submission candidate?

    Champion selection that ignores the label reproduces the study's silent failure #4:
    a leakage probe explicitly marked not-for-submission won the aggregation.
    """
    # 1. An explicit flag is authoritative -- and it arrives as bool, 0/1, or "true"/"false"
    #    depending on which code path wrote the entry. isinstance(flag, bool) alone skipped
    #    {"diagnostic": 0}, fell through to tier 2, and excluded an entry whose flag SAID it
    #    was not a diagnostic (2026-08-07 re-verification).
    flag = entry.get("diagnostic")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (int, float)) and flag in (0, 1):
        return bool(flag)
    if isinstance(flag, str) and flag.strip().lower() in ("true", "false", "yes", "no", "0", "1"):
        return flag.strip().lower() in ("true", "yes", "1")
    # 2. Dedicated LABEL fields only. model/model_name were scanned here too, but in this repo
    #    they hold long free-text blend DESCRIPTIONS -- the real s3e19 champion is
    #    "tree-search v2 blend (10-way: ... calendar-probe ...)" and the bare word "probe"
    #    demoted it to second place (2026-08-07). A label is deliberate; a description is not.
    labels = " ".join(str(entry.get(k, "")) for k in ("tag", "label")).lower()
    if any(m in labels for m in _DIAGNOSTIC_LABELS):
        return True
    # 3. Free-text notes: a phrase that ASSERTS exclusion counts anywhere; a diagnostic LABEL
    #    counts only as the notes' opening prefix ("leakage check: target leaks via row
    #    ordering" is the documented way to mark a leak probe), never mid-prose ("passed the
    #    leakage check" is evidence the entry is sound).
    notes = str(entry.get("notes", "")).strip().lower()
    if any(p in notes for p in _EXCLUSION_PHRASES):
        return True
    return any(notes.startswith(m + ":") for m in _DIAGNOSTIC_LABELS)


# Direction spellings seen in real logs, and metric families whose direction is known. Both
# exist because "direction is a free-form string" and "no direction at all" each silently
# resolved to MAXIMIZE -- inverting champion selection on every minimize-metric competition
# whose log spelled it differently (2026-08-07, bucket-A #13/#17).
_MINIMIZE_WORDS = {"minimize", "minimise", "min", "lower", "lower_is_better", "smaller"}
_MAXIMIZE_WORDS = {"maximize", "maximise", "max", "higher", "higher_is_better", "greater"}
_MINIMIZE_METRICS = {"rmse", "mae", "mse", "rmsle", "smape", "mape", "logloss", "log_loss",
                     "mcrmse", "medae", "crps", "wrmsse", "brier"}
_MAXIMIZE_METRICS = {"auc", "roc_auc", "rocauc", "aucroc", "accuracy", "acc", "f1", "qwk",
                     "kappa", "r2", "map", "ndcg", "precision", "recall", "gini"}


def _norm_direction(val) -> Optional[bool]:
    """direction value -> minimize? (None = unrecognized)."""
    s = str(val or "").strip().lower()
    if s in _MINIMIZE_WORDS:
        return True
    if s in _MAXIMIZE_WORDS:
        return False
    return None


def _metric_minimize(name) -> Optional[bool]:
    s = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in _MINIMIZE_METRICS:
        return True
    if s in _MAXIMIZE_METRICS:
        return False
    return None


def _entry_metric(e: dict):
    return e.get("metric") or e.get("eval_metric")


def get_best_experiment(
    competition_dir: str,
    metric: Optional[str] = None,
    minimize: Optional[bool] = None
) -> Optional[dict]:
    """
    Get the best non-diagnostic experiment.

    Direction resolution, in order: the caller's explicit `minimize`; the entries' own
    recorded direction (normalized -- "min"/"minimise"/"MINIMIZE" all count); the metric
    family. If none of those can decide, this RAISES rather than silently maximizing:
    a wrong champion is a submission, and the old maximize default inverted every
    minimize-metric competition whose log lacked a direction field.

    Entries with different metrics are refused unless `metric=` filters to one -- ranking
    raw scores across metrics compares apples against oranges (bucket-A #18).
    """
    experiments = load_experiments(competition_dir)
    if isinstance(experiments, dict):
        # the dict schema ({"competition": ..., "experiments": [...]}) exists in this repo
        # (child-mind workspace); iterating its KEYS raised AttributeError from _entry_score
        # (2026-08-07 round-4)
        experiments = experiments.get("experiments", [])
    if not experiments:
        return None

    if metric:
        m_norm = str(metric).strip().lower()
        experiments = [e for e in experiments
                       if str(_entry_metric(e) or "").strip().lower() == m_norm]
    if not experiments:
        return None

    scored = [e for e in experiments if _entry_score(e) is not None]
    if not scored:
        return None
    scored = [e for e in scored if not _entry_is_diagnostic(e)] or scored

    metrics_present = {str(_entry_metric(e)).strip().lower()
                       for e in scored if _entry_metric(e)}
    if len(metrics_present) > 1:
        raise ValueError(
            f"experiments.json mixes metrics {sorted(metrics_present)}; ranking raw scores "
            f"across metrics is meaningless. Pass metric=<one of them> to select the family "
            f"to rank within.")

    if minimize is None:
        dirs = {_norm_direction(e.get("direction")) for e in scored if e.get("direction")}
        dirs.discard(None)
        if len(dirs) == 1:
            minimize = dirs.pop()
        elif len(dirs) > 1:
            raise ValueError(
                "experiments disagree on direction (some minimize, some maximize); the log "
                "is inconsistent and picking a side silently would invert the champion for "
                "half the entries.")
        else:
            only_metric = next(iter(metrics_present), None)
            minimize = _metric_minimize(only_metric)
            if minimize is None:
                raise ValueError(
                    f"cannot determine optimization direction: no entry records one, and "
                    f"metric {only_metric!r} is not in the known families. Pass "
                    f"minimize=True/False explicitly -- defaulting silently is how a "
                    f"minimize-metric competition got a maximized champion.")
    return (min if minimize else max)(scored, key=_entry_score)


def print_leaderboard(competition_dir: str, top_n: int = 10):
    """Print a formatted leaderboard of experiments."""
    experiments = load_experiments(competition_dir)
    if isinstance(experiments, dict):
        experiments = experiments.get("experiments", [])
    if not experiments:
        print("No experiments logged yet.")
        return

    metric = experiments[0].get("metric") or experiments[0].get("eval_metric", "unknown")
    # One direction logic, not two: this function used to re-derive direction from its own
    # 7-metric inline set and a raw string compare, so a wrmsse log printed "higher is
    # better" and ranked the WORSE entry first while get_best_experiment picked the better
    # one (2026-08-07 round-4).
    dirs = {_norm_direction(e.get("direction")) for e in experiments if e.get("direction")}
    dirs.discard(None)
    if len(dirs) == 1:
        minimize = dirs.pop()
    else:
        inferred = _metric_minimize(metric)
        minimize = inferred if inferred is not None else False
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
    params: Optional[dict] = None,
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
    # A non-finite score is not an experiment result, it is a broken CV -- and it used to
    # poison everything downstream: json.dump emits a bare NaN token (invalid JSON), every
    # comparison with nan is False so min()/max() return whichever entry comes FIRST, and the
    # champion became order-dependent (2026-08-07 round-3). Refuse loudly at log time.
    import math as _math
    if not _math.isfinite(entry["score"]):
        raise ValueError(
            f"score={score!r} is not finite; a NaN/inf score means the CV itself failed. "
            f"Fix the evaluation and log a real number -- logging NaN poisons champion "
            f"selection order-dependently and writes invalid JSON.")
    if features is not None:
        # `features` crosses the Stage 2 -> Stage 3 seam as a comma-separated STRING in one
        # contract and as a list in the other. list("a,b,c") explodes a string into single
        # CHARACTERS, so the logged feature set became ['a', ',', 'b', ...] and n_features
        # became the character count -- silently, in the artifact Stage 5 reads back
        # (2026-08-04 architecture gate). Normalize instead of coercing blindly.
        if isinstance(features, str):
            feats = [f.strip() for f in features.split(",") if f.strip()]
        elif isinstance(features, (list, tuple, set)):
            feats = [str(f) for f in features]
        else:
            raise TypeError(
                f"features must be a list or a comma-separated string, got "
                f"{type(features).__name__}; blind list() on anything else silently produces "
                f"a per-character feature list")
        entry["features"] = feats
        entry["n_features"] = len(feats)
    if params is not None:
        # The hyperparameters that produced this score. Absent until 2026-08-04; the first fix
        # stringified top-level VALUES only, so a numpy scalar value or key -- which is what
        # sklearn/lightgbm actually hand back -- raised TypeError from json.dump AFTER
        # training, discarding the experiment and leaving an orphan .tmp file
        # (2026-08-07 re-verification). Sanitize recursively: keys become str, numpy scalars
        # unwrap via .item(), arrays become lists, anything else falls back to str().
        entry["params"] = _jsonable(params)
    for key, val in (("cv", cv), ("base_models", base_models), ("ensemble", ensemble),
                     ("postprocess", postprocess), ("submission", submission),
                     ("leaderboard", leaderboard)):
        if val is not None:
            # Sanitized like params: a numpy array anywhere here (fold scores from sklearn,
            # blend weights) raised TypeError from json.dump AFTER training, discarding the
            # experiment -- the exact class the params fix claimed closed (2026-08-07).
            entry[key] = _jsonable(val)
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
    extra = _jsonable(extra)
    for key, val in extra.items():
        if val is not None:
            entry[key] = val
    experiments.append(entry)
    save_experiments(competition_dir, experiments)
    return entry["experiment_id"]
