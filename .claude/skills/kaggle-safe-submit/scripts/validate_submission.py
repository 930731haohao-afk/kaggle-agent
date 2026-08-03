#!/usr/bin/env python3
"""
Deterministic Kaggle submission validator.

Created 2026-08-03 (audit): SKILL.md step 2 has always told the agent to run this file,
but the file did not exist — the skill's core instruction was a dangling reference, so
"validation passed" could only ever have been asserted, never executed.

Two tiers, mirroring assets/templates/submit_template.py in the kaggle-agent skill:

  STRUCTURAL  The file is malformed — wrong shape/columns/IDs, NaN, +/-inf. Kaggle
              rejects it or scores the wrong rows. Never overridable. -> exit 1
  SUSPICIOUS  A legal CSV that is almost certainly a bug: constant predictions, all
              zeros, values outside the plausible range, probabilities sent to a
              hard-label metric (or hard labels sent to a probability metric). Each of
              these passes a shape/column/NaN/ID gate and wastes a daily submission.
              Constant IS a legal submission on rare competitions, so this tier can be
              waived — but only with --allow-suspicious REASON, never silently. -> exit 2

Usage:
    python3 scripts/validate_submission.py <submission.csv> <sample_submission.csv>
    python3 scripts/validate_submission.py --submission sub.csv --sample sample.csv \
        --id-col id --metric auc --train data/train.csv --target target

The range and label-set checks need a reference. Pass --metric (config.yaml
`evaluation_metric`) and/or --train/--target whenever they exist; without them those
checks report SKIP rather than guessing, because guessing produces false alarms that
train the agent to ignore the gate.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

EXIT_OK, EXIT_STRUCTURAL, EXIT_SUSPICIOUS = 0, 1, 2

# Metric -> what the prediction column is supposed to contain. Normalised by _norm().
_PROBABILITY_METRICS = {
    "auc", "rocauc", "areaunderroccurve", "logloss", "binarylogloss", "multilogloss",
    "crossentropy", "averageprecision", "map", "prauc", "aucpr", "brier", "brierscore",
}
_LABEL_METRICS = {
    "accuracy", "balancedaccuracy", "f1", "f1macro", "f1micro", "f1weighted", "fbeta",
    "precision", "recall", "mcc", "matthewscorrcoef", "kappa", "cohenkappa",
    "quadraticweightedkappa", "qwk", "jaccard", "dice", "iou", "hammingloss",
}
_CONTINUOUS_METRICS = {
    "rmse", "mse", "mae", "rmsle", "msle", "smape", "mape", "r2", "mcrmse", "medae",
    "poisson", "tweedie", "crps", "spearman", "pearson", "gini",
}


def _norm(name: str) -> str:
    """Lowercase and strip separators so 'ROC-AUC', 'roc_auc', 'roc auc' all collide."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class Report:
    """Collects check results in the two tiers and prints them as they run."""

    def __init__(self) -> None:
        self.structural: list[str] = []
        self.suspicious: list[str] = []

    def check(self, tier: list, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {name:<20} {'PASS' if ok else 'FAIL'} {detail}".rstrip())
        if not ok:
            tier.append(f"{name} — {detail}" if detail else name)
        return ok

    def skip(self, name: str, why: str) -> None:
        print(f"  {name:<20} SKIP {why}")


def _resolve_expectation(metric: Optional[str], expect: str,
                         y_train: Optional[pd.Series]) -> str:
    """What should the prediction column hold: 'probability', 'label', 'continuous'?

    Explicit --expect wins; then the metric; then a conservative read of the training
    target. Returns 'unknown' when nothing is knowable — the caller then skips the
    checks that need a reference instead of inventing one.
    """
    if expect != "auto":
        return expect
    if metric:
        m = _norm(metric)
        if m in _PROBABILITY_METRICS:
            return "probability"
        if m in _LABEL_METRICS:
            return "label"
        if m in _CONTINUOUS_METRICS:
            return "continuous"
    if y_train is not None and pd.api.types.is_numeric_dtype(y_train):
        uniq = np.unique(y_train.dropna().to_numpy())
        integral = np.all(uniq == np.round(uniq))
        if not integral or len(uniq) > 20:
            return "continuous"
    return "unknown"


def validate(sub: pd.DataFrame, sample: pd.DataFrame, id_col: Optional[str] = None,
             y_train: Optional[pd.Series] = None, metric: Optional[str] = None,
             expect: str = "auto") -> Report:
    """Run every check. Caller decides what to do with the two failure lists."""
    rep = Report()
    id_col = id_col or sample.columns[0]
    pred_col = sample.columns[-1]

    # ---- STRUCTURAL ----
    rep.check(rep.structural, "Row count", len(sub) == len(sample),
              f"(got {len(sub)}, expected {len(sample)})")
    rep.check(rep.structural, "Columns", list(sub.columns) == list(sample.columns),
              f"(got {list(sub.columns)}, expected {list(sample.columns)})")

    if id_col not in sub.columns or id_col not in sample.columns:
        rep.check(rep.structural, "ID column", False, f"('{id_col}' missing)")
        return rep

    sub_ids, sample_ids = sub[id_col].astype(str), sample[id_col].astype(str)
    rep.check(rep.structural, "ID duplicates", not sub_ids.duplicated().any(),
              f"({int(sub_ids.duplicated().sum())} duplicate IDs)")
    same_set = set(sub_ids) == set(sample_ids)
    same_order = len(sub_ids) == len(sample_ids) and (sub_ids.values == sample_ids.values).all()
    rep.check(rep.structural, "ID alignment", bool(same_order), "" if same_order else (
        "(IDs match as a set but not in order — rows are permuted)" if same_set
        else f"({len(set(sample_ids) - set(sub_ids))} missing, "
             f"{len(set(sub_ids) - set(sample_ids))} unexpected)"))

    if pred_col not in sub.columns:
        rep.check(rep.structural, "Prediction column", False, f"('{pred_col}' missing)")
        return rep

    preds = sub[pred_col]
    n_nan = int(preds.isnull().sum())
    rep.check(rep.structural, "NaN", n_nan == 0, f"({n_nan} NaN values)")

    numeric = pd.api.types.is_numeric_dtype(preds)
    if numeric:
        arr = preds.to_numpy(dtype=float)
        finite = np.isfinite(arr)
        # isnull() is False for +/-inf, which is exactly how inf reaches Kaggle through a
        # NaN-only gate. isfinite() is also False for NaN, so subtract those out.
        n_inf = int((~finite).sum()) - n_nan
        rep.check(rep.structural, "Finite", n_inf <= 0, f"({max(n_inf, 0)} +/-inf values)")
        vals = arr[finite]
    else:
        rep.skip("Finite", "(non-numeric prediction column)")
        vals = np.array([])

    # ---- SUSPICIOUS ----
    n_unique = int(preds.nunique(dropna=False))
    rep.check(rep.suspicious, "Variance", n_unique > 1,
              f"({n_unique} unique value(s)"
              + (f", constant = {preds.iloc[0]}" if n_unique == 1 and len(preds) else "")
              + ")")

    all_zero = bool(numeric and len(vals) and (vals == 0).all())
    rep.check(rep.suspicious, "Non-zero", not all_zero,
              "(every prediction is exactly 0 — a placeholder array was never filled in)")

    kind = _resolve_expectation(metric, expect, y_train)
    labels = (set(np.unique(y_train.dropna().to_numpy()).tolist())
              if y_train is not None and pd.api.types.is_numeric_dtype(y_train) else None)

    # Range. Probabilities are [0, 1]; everything else is judged against the training
    # target's own range widened by half its span, so a wider test distribution passes
    # but a scale error (log space, un-inverted transform, wrong column) does not.
    if not numeric or not len(vals):
        rep.skip("Range", "(non-numeric prediction column)")
    elif kind == "probability":
        n_out = int(((vals < 0) | (vals > 1)).sum())
        rep.check(rep.suspicious, "Range", n_out == 0,
                  f"({n_out} values outside [0, 1]; observed "
                  f"[{vals.min():.6g}, {vals.max():.6g}])")
    elif y_train is not None and pd.api.types.is_numeric_dtype(y_train):
        t_lo, t_hi = float(y_train.min()), float(y_train.max())
        span = (t_hi - t_lo) or max(abs(t_hi), 1.0)
        lo, hi = t_lo - 0.5 * span, t_hi + 0.5 * span
        n_out = int(((vals < lo) | (vals > hi)).sum())
        rep.check(rep.suspicious, "Range", n_out == 0,
                  f"({n_out} values outside [{lo:.6g}, {hi:.6g}] "
                  f"(train target +/-50% span); observed "
                  f"[{vals.min():.6g}, {vals.max():.6g}])")
    else:
        rep.skip("Range", "(no reference — pass --metric and/or --train/--target)")

    # Hard label vs probability. Both directions throw the score away silently: 0.73 on
    # an accuracy competition is rounded or rejected, a hard 0/1 on AUC/logloss loses all
    # ranking information.
    if not numeric or not len(vals):
        rep.skip("Value kind", "(non-numeric prediction column)")
    elif kind == "label":
        if labels is not None:
            stray = sorted({v for v in np.unique(vals).tolist() if v not in labels})[:5]
            rep.check(rep.suspicious, "Value kind", not stray,
                      f"(metric '{metric}' scores hard labels, but values outside the "
                      f"training label set were found: {stray})")
        else:
            fractional = np.any(vals != np.round(vals))
            rep.check(rep.suspicious, "Value kind", not fractional,
                      f"(metric '{metric}' scores hard labels, but the column holds "
                      f"fractional values, e.g. {vals[vals != np.round(vals)][:3].tolist()})")
    elif kind == "probability":
        rep.check(rep.suspicious, "Value kind", not set(np.unique(vals).tolist()) <= {0.0, 1.0},
                  f"(metric '{metric}' scores probabilities but every value is 0 or 1 — "
                  "predict_proba was probably replaced by predict)")
    else:
        rep.skip("Value kind", "(no metric/expectation given)")

    if numeric and len(vals) and y_train is not None and pd.api.types.is_numeric_dtype(y_train):
        print(f"  [info] pred mean {vals.mean():.6f} vs train target mean "
              f"{float(y_train.mean()):.6f}")
    return rep


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("positional", nargs="*", metavar="SUBMISSION SAMPLE",
                   help="submission.csv and sample_submission.csv")
    p.add_argument("--submission")
    p.add_argument("--sample")
    p.add_argument("--id-col", help="default: first column of the sample submission")
    p.add_argument("--train", help="training CSV, for the range / label-set reference")
    p.add_argument("--target", help="target column name inside --train")
    p.add_argument("--metric", help="config.yaml evaluation_metric, e.g. auc / rmse / accuracy")
    p.add_argument("--expect", default="auto",
                   choices=["auto", "probability", "label", "continuous"])
    p.add_argument("--allow-suspicious", metavar="REASON",
                   help="waive the SUSPICIOUS tier; the reason is required and is echoed")
    a = p.parse_args(argv)

    sub_path = a.submission or (a.positional[0] if len(a.positional) > 0 else None)
    sample_path = a.sample or (a.positional[1] if len(a.positional) > 1 else None)
    if not sub_path or not sample_path:
        p.error("need a submission and a sample_submission (positional or --submission/--sample)")

    sub, sample = pd.read_csv(sub_path), pd.read_csv(sample_path)
    y_train = None
    if a.train and a.target:
        y_train = pd.read_csv(a.train, usecols=[a.target])[a.target]

    print(f"Validating {sub_path}\n      against {sample_path}")
    rep = validate(sub, sample, a.id_col, y_train, a.metric, a.expect)

    if rep.structural:
        print(f"\nSTRUCTURAL FAILURE ({len(rep.structural)}) — do NOT submit:")
        for f in rep.structural:
            print(f"  - {f}")
        return EXIT_STRUCTURAL
    if rep.suspicious:
        waiver = (a.allow_suspicious or "").strip()
        print(f"\nSUSPICIOUS ({len(rep.suspicious)}) — legal CSV, almost certainly a bug:")
        for f in rep.suspicious:
            print(f"  - {f}")
        if waiver:
            print(f"\nWAIVED by --allow-suspicious: {waiver}")
            print("PASS (with waiver)")
            return EXIT_OK
        print("\nDo NOT submit. Fix it, or re-run with "
              '--allow-suspicious "<why this really is the intended submission>".')
        return EXIT_SUSPICIOUS
    print("\nPASS — all checks clean.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
