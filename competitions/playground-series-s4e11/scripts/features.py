"""Feature engineering module for playground-series-s4e11 (Depression, Accuracy).

Reused unmodified by scripts/04_train_blend.py (tier2), scripts/05_iterate.py (tier3),
and tree_search/eval_s4e11.py (tier4) so that all four tiers train on byte-identical
features and the SAME CV folds (StratifiedKFold(5, shuffle=True, seed=42) on Depression).

Carries forward the Feb-2026 EDA/feature set documented in STATUS.md and
scripts/02_baseline.py's engineer_features() (28 features after label-encoding: sleep/
diet bucket cleanup, is_student/suicidal_thoughts/family_history/is_male binary flags,
structured-NaN filled with 0 for academic/work columns that are structurally absent for
the other Working-Professional-or-Student group, age features, 5 interaction terms,
label-encoded City/Profession/Degree/...).

Leakage audit (cross-season lesson from s4e1's surname target encoding, which used an
INDEPENDENT fold seed from the model's own training folds and leaked through the shared
grouping key -- see competitions/playground-series-s4e1/scripts/features.py docstring):
this competition's Feb feature set used ONLY label encoding for its high-cardinality
categoricals (City/Profession/Degree), never target encoding, so there is no analogous
fold-inconsistency leak here. This module is a faithful, directly-comparable port of the
Feb baseline's engineer_features(), refactored into make_folds/build_all so tiers 2-4
share one definition -- no correction needed vs the Feb version (unlike s4e1).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET, ID = "Depression", "id"
N_SPLITS, SEED = 5, 42
DROP_RAW = [ID, TARGET, "Name"]

SLEEP_MAP = {
    "Less than 5 hours": "less_than_5",
    "5-6 hours": "5_6",
    "7-8 hours": "7_8",
    "More than 8 hours": "more_than_8",
}
DIET_MAP = {"Healthy": "Healthy", "Moderate": "Moderate", "Unhealthy": "Unhealthy"}


def make_folds(y):
    """Canonical CV folds -- IDENTICAL across tier2/tier3/tier4. Call once per process
    with the training target vector; every eval/train script must use this, not its own
    StratifiedKFold call, so folds are guaranteed byte-identical."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def engineer_base(df):
    """Deterministic (no-leakage) feature engineering -- verbatim port of Feb's
    engineer_features() (scripts/02_baseline.py)."""
    out = df.copy()
    out["Sleep Duration"] = out["Sleep Duration"].map(SLEEP_MAP).fillna("other")
    out["Dietary Habits"] = out["Dietary Habits"].map(DIET_MAP).fillna("Other")
    out["is_student"] = (out["Working Professional or Student"] == "Student").astype(int)
    for col in ["Academic Pressure", "Work Pressure", "Study Satisfaction", "Job Satisfaction"]:
        out[col] = out[col].fillna(0)
    out["CGPA"] = out["CGPA"].fillna(0)
    out["suicidal_thoughts"] = (out["Have you ever had suicidal thoughts ?"] == "Yes").astype(int)
    out["family_history"] = (out["Family History of Mental Illness"] == "Yes").astype(int)
    out["is_male"] = (out["Gender"] == "Male").astype(int)
    out["Financial Stress"] = out["Financial Stress"].fillna(out["Financial Stress"].median())
    out["age_decade"] = out["Age"] // 10
    out["is_young"] = (out["Age"] < 30).astype(int)
    out["pressure"] = out["Academic Pressure"] + out["Work Pressure"]
    out["satisfaction"] = out["Study Satisfaction"] + out["Job Satisfaction"]
    out["stress_x_pressure"] = out["Financial Stress"] * out["pressure"]
    out["suicidal_x_student"] = out["suicidal_thoughts"] * out["is_student"]
    out["age_x_pressure"] = out["Age"] * out["pressure"]
    out = out.drop(columns=["Name"], errors="ignore")
    return out


def label_encode_strings(train_e, test_e):
    """Label-encode every remaining object-dtype column (union of train+test
    categories). Matches the Feb baseline's `string_cols` loop exactly -- whatever is
    still string-typed after engineer_base gets encoded (Gender, City,
    Working Professional or Student, Profession, Sleep Duration, Dietary Habits, Degree,
    the suicidal-thoughts and family-history raw Yes/No columns)."""
    train_e = train_e.copy()
    test_e = test_e.copy()
    string_cols = train_e.select_dtypes(include="object").columns.tolist()
    for c in string_cols:
        cats = pd.Index(pd.concat([train_e[c].astype(str), test_e[c].astype(str)]).unique())
        mp = {v: i for i, v in enumerate(cats)}
        train_e[c] = train_e[c].astype(str).map(mp).astype("int32")
        test_e[c] = test_e[c].astype(str).map(mp).astype("int32")
    return train_e, test_e


def build_all(train_raw, test_raw, y, folds):
    """One-call convenience: engineer + label-encode. `folds` accepted for API symmetry
    with competitions/playground-series-s4e1/scripts/features.py (no fold-safe encoding
    step is needed here -- see module docstring's leakage audit) and is unused."""
    del folds
    train_e = engineer_base(train_raw)
    test_e = engineer_base(test_raw)
    train_e, test_e = label_encode_strings(train_e, test_e)
    feature_cols = [c for c in train_e.columns if c not in DROP_RAW]
    return train_e, test_e, feature_cols


def best_threshold_accuracy(y_true, prob, lo=0.10, hi=0.90, step=0.01):
    """Metric-aware post-processing for the Accuracy metric (this comp's analogue of
    s3e16's integer-rounding / s3e5's OptimizedRounder): Accuracy needs a hard 0/1 label,
    so every decision score in this competition is computed AFTER a threshold sweep on
    the continuous OOF probability, never on the raw prob vector directly, matching
    knowledge/experience.md's post-processing rule ("決策永遠對後處理後的分數做"). Sweep
    range/step (0.10..0.90 step 0.01) matches Feb's scripts/02_baseline.py exactly for
    direct comparability. Global (non-nested) sweep on the full OOF vector -- same
    convention Feb used; honest caveat carried into the report (single-scalar fit, and
    Feb's own optimized-threshold CV 0.93738 was, if anything, exceeded by its Public LB
    0.94093, i.e. no sign of overfitting for this comp's version of the technique).
    Vectorized (numpy broadcasting over all thresholds at once) rather than a Python
    loop calling sklearn's accuracy_score 80+ times per call -- this function is the
    metric_fn inside every blend weight-search (tier2-4), called thousands of times per
    blend node, so a measured ~2-3x per-call speedup compounds materially over a
    60-node tree search. Verified to reproduce the naive per-threshold-loop
    implementation bit-for-bit (same swept thresholds, same first-of-ties tie-break)
    before being adopted -- this is a performance rewrite, not a behavior change.

    Returns (best_accuracy, best_threshold)."""
    y_true_b = np.asarray(y_true).astype(bool)
    prob = np.asarray(prob, dtype=np.float64)
    thresholds = np.union1d(np.arange(lo, hi + 1e-9, step, dtype=np.float64), [0.5])
    preds = prob[:, None] >= thresholds[None, :]
    accs = (preds == y_true_b[:, None]).mean(axis=0)
    best_idx = int(np.argmax(accs))
    return float(accs[best_idx]), float(thresholds[best_idx])
