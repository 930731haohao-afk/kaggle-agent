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
    "auc", "rocauc", "aucroc", "areaunderroccurve", "areaundertheroccurve",
    "logloss", "binarylogloss", "multilogloss", "binarycrossentropy",
    "crossentropy", "averageprecision", "map", "prauc", "aucpr", "brier", "brierscore",
    "gini", "normalizedgini",
}
# 'aucroc' is here because config.yaml on at least one competition spells the metric
# "auc-roc": _norm() flattened that to "aucroc", which was in no set, so the value-kind check
# resolved to "unknown" and the probability-range gate silently downgraded to PASS. Word order
# must never decide whether a gate runs (2026-08-04 architecture gate).
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
    """Lowercase, NFKC-normalize and strip separators so 'ROC-AUC', 'roc_auc', 'roc auc' --
    and fullwidth/unicode spellings like 'ＲＯＣ－ＡＵＣ' -- all collide. Non-ASCII spellings
    silently disabled the probability gates before normalization (2026-08-07 round-3)."""
    import unicodedata
    flat = unicodedata.normalize("NFKC", str(name)).lower()
    return re.sub(r"[^a-z0-9]", "", flat)


def _norm_variants(name: str) -> set:
    """Every spelling of a metric name that should be treated as the same metric.

    Stripping separators is not enough: it maps 'roc_auc' to 'rocauc' but 'auc-roc' to
    'aucroc', which was in no set, so the value-kind check resolved to "unknown" and the
    probability-range test silently downgraded to PASS on a competition whose config.yaml
    happened to spell it the other way round (2026-08-04 architecture gate). Also emit the
    token-sorted form so word order cannot decide whether a gate runs.
    """
    flat = _norm(name)
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", str(name).lower()) if tok]
    return {flat, "".join(sorted(tokens))} - {""}


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
        variants = _norm_variants(metric)
        if variants & _PROBABILITY_METRICS:
            return "probability"
        if variants & _LABEL_METRICS:
            return "label"
        if variants & _CONTINUOUS_METRICS:
            return "continuous"
        # Token-level fallback: "area under the ROC curve (AUC)" and
        # "categorization_accuracy" -- both verbatim from this repo's config.yaml files --
        # match no whole-name variant, resolved to "unknown", and the value gate silently
        # downgraded to SKIP (2026-08-07 re-verification). Any individual token that is
        # itself a known metric name decides; probability outranks label outranks continuous
        # because its range check is the strictest.
        import unicodedata
        flat_metric = unicodedata.normalize("NFKC", str(metric)).lower()
        tokens = {tok for tok in re.split(r"[^a-z0-9]+", flat_metric) if tok}
        if tokens & _PROBABILITY_METRICS:
            return "probability"
        if tokens & _LABEL_METRICS:
            return "label"
        if tokens & _CONTINUOUS_METRICS:
            return "continuous"
    if y_train is not None and pd.api.types.is_numeric_dtype(y_train):
        uniq = np.unique(y_train.dropna().to_numpy())
        integral = np.all(uniq == np.round(uniq))
        if not integral or len(uniq) > 20:
            return "continuous"
    return "unknown"


def validate(sub: pd.DataFrame, sample: pd.DataFrame, id_col: Optional[str] = None,
             y_train: Optional[pd.Series] = None, metric: Optional[str] = None,
             expect: str = "auto",
             y_train_frame: Optional[pd.DataFrame] = None) -> Report:
    """Run every check. Caller decides what to do with the two failure lists."""
    rep = Report()
    id_col = id_col or sample.columns[0]
    # The prediction column is the last NON-ID column. sample.columns[-1] was taken
    # unconditionally, so a submission whose id sits last had its IDs value-checked as
    # predictions while the real prediction column was never examined
    # (2026-08-07 re-verification).
    non_id = [c for c in sample.columns if c != id_col]
    pred_col = non_id[-1] if non_id else sample.columns[-1]

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

    # EVERY prediction column, not just the last. afsis (13 targets) and conway (400) submit
    # wide: checking sample.columns[-1] alone let an all-NaN or all-inf column in any other
    # position pass the structural gate untouched (2026-08-04 architecture gate).
    pred_cols = [c for c in sample.columns if c != id_col and c in sub.columns]
    if not pred_cols:
        pred_cols = [pred_col]
    n_nan_by_col, n_inf_by_col = {}, {}
    for c in pred_cols:
        col = sub[c]
        n_nan_c = int(col.isnull().sum())
        n_nan_by_col[c] = n_nan_c
        if pd.api.types.is_numeric_dtype(col):
            a = col.to_numpy(dtype=float)
            # isnull() is False for +/-inf, which is exactly how inf reaches Kaggle through a
            # NaN-only gate. isfinite() is also False for NaN, so subtract those out.
            n_inf_by_col[c] = max(int((~np.isfinite(a)).sum()) - n_nan_c, 0)
    bad_nan = {c: n for c, n in n_nan_by_col.items() if n}
    bad_inf = {c: n for c, n in n_inf_by_col.items() if n}
    rep.check(rep.structural, "NaN", not bad_nan,
              f"({sum(bad_nan.values())} NaN across {len(bad_nan)}/{len(pred_cols)} column(s): "
              f"{dict(list(bad_nan.items())[:5])})" if bad_nan else "")
    rep.check(rep.structural, "Finite", not bad_inf,
              f"({sum(bad_inf.values())} +/-inf across {len(bad_inf)}/{len(pred_cols)} "
              f"column(s): {dict(list(bad_inf.items())[:5])})" if bad_inf else "")
    # A probability/continuous metric scored on an object-dtype column means the predictions
    # are strings ("0.5"): Kaggle may coerce or reject, and every numeric gate here silently
    # SKIPped, so it passed clean (2026-08-07 round-3). Structural, because the file is wrong
    # as submitted whatever the values decode to.
    _kind_now = _resolve_expectation(metric, expect, y_train)
    if _kind_now in ("probability", "continuous"):
        obj_cols = [c for c in pred_cols if not pd.api.types.is_numeric_dtype(sub[c])]
        rep.check(rep.structural, "Numeric dtype", not obj_cols,
                  f"({len(obj_cols)} prediction column(s) are non-numeric on a {_kind_now} "
                  f"metric: {obj_cols[:5]})" if obj_cols else "")
    all_null_cols = [c for c in pred_cols if sub[c].isnull().all()]
    rep.check(rep.structural, "No empty target column", not all_null_cols,
              f"({len(all_null_cols)} column(s) entirely null: {all_null_cols[:5]})"
              if all_null_cols else "")

    preds = sub[pred_col]
    numeric = pd.api.types.is_numeric_dtype(preds)
    if numeric:
        arr = preds.to_numpy(dtype=float)
        vals = arr[np.isfinite(arr)]
    else:
        vals = np.array([])

    # ---- SUSPICIOUS ----
    # Variance and non-zero run PER COLUMN. They used to read only the last column, so a
    # multi-target submission (afsis: 6 columns, conway: 401, emvic: 37) with a constant or
    # all-zero placeholder anywhere else passed untouched (2026-08-07 re-verification).
    const_cols, zero_cols = {}, []
    for c in pred_cols:
        col = sub[c]
        nu = int(col.nunique(dropna=False))
        if nu <= 1:
            const_cols[c] = col.iloc[0] if len(col) else None
        if pd.api.types.is_numeric_dtype(col):
            a = col.to_numpy(dtype=float)
            fin = a[np.isfinite(a)]
            if len(fin) and (fin == 0).all():
                zero_cols.append(c)
    rep.check(rep.suspicious, "Variance", not const_cols,
              f"({len(const_cols)}/{len(pred_cols)} column(s) constant: "
              f"{dict(list(const_cols.items())[:5])})" if const_cols else
              f"(all {len(pred_cols)} column(s) vary)")
    rep.check(rep.suspicious, "Non-zero", not zero_cols,
              f"({len(zero_cols)} column(s) entirely 0 — a placeholder array was never "
              f"filled in: {zero_cols[:5]})" if zero_cols else "")

    kind = _resolve_expectation(metric, expect, y_train)
    labels = (set(np.unique(y_train.dropna().to_numpy()).tolist())
              if y_train is not None and pd.api.types.is_numeric_dtype(y_train) else None)

    # Range, PER COLUMN. Probabilities are [0, 1]; everything else is judged against that
    # target column's OWN training range widened by half its span. Two defects made this
    # single-column before (2026-08-07 round-4): only the last column was inspected (a 1000x
    # scale error in afsis's non-last 'P' passed clean), and the one --target reference was
    # compared against whatever column happened to be last -- a cross-target comparison that
    # only catches errors when the targets share a scale. y_train_frame carries the training
    # columns matched BY NAME.
    def _ref_for(col):
        if y_train_frame is not None and col in y_train_frame.columns \
                and pd.api.types.is_numeric_dtype(y_train_frame[col]):
            return y_train_frame[col]
        if y_train is not None and pd.api.types.is_numeric_dtype(y_train) \
                and (getattr(y_train, "name", None) in (col, None) or len(pred_cols) == 1):
            return y_train
        return None

    range_fails, range_skips = [], []
    for col in pred_cols:
        colvals = sub[col]
        if not pd.api.types.is_numeric_dtype(colvals):
            continue
        cv = colvals.to_numpy(dtype=float)
        cv = cv[np.isfinite(cv)]
        if not len(cv):
            continue
        if kind == "probability":
            n_out = int(((cv < 0) | (cv > 1)).sum())
            if n_out:
                range_fails.append(f"{col}: {n_out} outside [0,1], observed "
                                   f"[{cv.min():.6g}, {cv.max():.6g}]")
            continue
        ref = _ref_for(col)
        if ref is None:
            range_skips.append(col)
            continue
        t_lo, t_hi = float(ref.min()), float(ref.max())
        span = (t_hi - t_lo) or max(abs(t_hi), 1.0)
        lo, hi = t_lo - 0.5 * span, t_hi + 0.5 * span
        n_out = int(((cv < lo) | (cv > hi)).sum())
        if n_out:
            range_fails.append(f"{col}: {n_out} outside [{lo:.6g}, {hi:.6g}] "
                               f"(its own train range +/-50%), observed "
                               f"[{cv.min():.6g}, {cv.max():.6g}]")
    if kind == "probability" or any(_ref_for(c) is not None for c in pred_cols):
        rep.check(rep.suspicious, "Range", not range_fails,
                  ("(" + "; ".join(range_fails[:4]) + ")") if range_fails else
                  (f"({len(pred_cols) - len(range_skips)}/{len(pred_cols)} column(s) "
                   f"checked" + (f"; no reference for {range_skips[:3]}" if range_skips
                                 else "") + ")"))
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
    y_train, y_train_frame = None, None
    if a.train and not a.target:
        # Half the mandated pair silently dropped the range reference and passed clean --
        # the same silent-downgrade class as omitting --train entirely (2026-08-07 round-4).
        # For multi-target submissions we can do better than an error: load every training
        # column that matches a prediction column BY NAME.
        train_df = pd.read_csv(a.train)
        pred_names = [c for c in sample.columns if c != (a.id_col or sample.columns[0])]
        matched = [c for c in pred_names if c in train_df.columns]
        if matched:
            y_train_frame = train_df[matched]
            print(f"[validate] --target omitted; matched {len(matched)} training column(s) "
                  f"by name: {matched[:6]}")
        else:
            print("ERROR: --train given without --target, and no training column matches a "
                  "prediction column by name. The Range gate would silently skip — refusing "
                  "instead. Pass --target <col>.")
            return EXIT_STRUCTURAL
    elif a.train and a.target:
        train_df = pd.read_csv(a.train)
        if a.target not in train_df.columns:
            print(f"ERROR: --target {a.target!r} not in --train columns "
                  f"{list(train_df.columns)[:8]}")
            return EXIT_STRUCTURAL
        y_train = train_df[a.target]
        # multi-target: every other prediction column that matches by name gets its own ref
        pred_names = [c for c in sample.columns if c != (a.id_col or sample.columns[0])]
        matched = [c for c in pred_names if c in train_df.columns]
        if len(matched) > 1:
            y_train_frame = train_df[matched]

    print(f"Validating {sub_path}\n      against {sample_path}")
    rep = validate(sub, sample, a.id_col, y_train, a.metric, a.expect,
                   y_train_frame=y_train_frame)

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
