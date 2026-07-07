"""Feature engineering module for playground-series-s6e1 (exam_score, R2).

Reused unmodified by scripts/04_train_blend.py (stage 2), scripts/05_iterate.py
(stage 3), tree_search/eval_s6e1.py (stage 4) and scripts/06_rebuild_tree_best.py
(rebuild gate) so that all stages train on byte-identical features and the SAME CV
folds: KFold(5, shuffle, random_state=42) on row order.

Fold-comparability note vs the Feb-2026 prior-season run (scripts/full_pipeline.py):
Feb used exactly KFold(5, shuffle, random_state=42) on the same train.csv row order;
sklearn's KFold split depends only on n_samples and the seed, so the folds produced
by make_folds() here are IDENTICAL to Feb's -- Feb's logged scores (experiments.json
exp 1-4) are directly comparable to every score this run produces.

Feature set: verbatim port of Feb's engineering block in full_pipeline.py --
22 features = 4 raw numerics + 3 ordinal-encoded categoricals + 4 label-encoded
categoricals (11 BASE columns) + 11 engineered columns (4 pairwise products, 2
polynomials, a composite effort ratio, an ordinal study-method score, a product with
it, a sleep-deficit transform, a sleep x quality product). No missing values, no
target encoding of any kind (nothing fold-dependent); the whole transform is
deterministic and leak-free by construction. Label maps are derived from the
combined train+test unique values (target never involved), so codes are identical
across train and test (EDA confirms test has no unseen categories in any column).

INTERACTION_FEATURES lists the 11 engineered columns as one group: knowledge/
experience.md carries a negative prior ("trees learn multiplicative interactions
themselves; explicit product columns add collinearity, not information", s3e9
evidence; "too many features regress, pruning is a routine win", s3e7/s3e14) while
Feb's STATUS.md builds these interactions and lists "more complex 3-way interactions"
as an improvement. Stage 3 round 1 resolves this with a controlled same-folds on/off
comparison before the set is trusted any further (see scripts/05_iterate.py).

Metric-aware post-processing for R2 on a physically-bounded exam score: predictions
are clipped to [0, 100] before scoring -- r2_clip() is the ONLY decision metric in
this comp, used identically by every stage's weight search and every logged score
(knowledge/experience.md's "clip to the training target range as a safety net", s3e1/
s3e11; and its rule: decide on the post-processed score, never the raw one). Feb's
pipeline applied the same [0,100] clip before scoring, so scores stay comparable.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

TARGET, ID = "exam_score", "id"
N_SPLITS, SEED = 5, 42
CLIP_LO, CLIP_HI = 0.0, 100.0  # exam score physical range

SLEEP_QUALITY_MAP = {"poor": 0, "average": 1, "good": 2}
FACILITY_RATING_MAP = {"low": 0, "medium": 1, "high": 2}
EXAM_DIFFICULTY_MAP = {"easy": 0, "moderate": 1, "hard": 2}
STUDY_METHOD_SCORE_MAP = {"self-study": 0, "online videos": 1, "group study": 2,
                          "mixed": 3, "coaching": 4}
LABEL_COLS = ["gender", "course", "internet_access", "study_method"]

DROP_RAW = [TARGET, ID, "gender", "course", "internet_access", "sleep_quality",
            "study_method", "facility_rating", "exam_difficulty"]

# The 11 engineered product/poly/composite columns (everything beyond the 11 encoded
# base columns) -- the unit under test in stage 3 round 1's interaction prior check.
INTERACTION_FEATURES = [
    "study_x_attendance", "study_x_sleep_quality", "study_x_facility",
    "attendance_x_sleep", "study_sq", "attendance_sq", "total_effort",
    "study_method_score", "method_x_hours", "sleep_deficit", "sleep_x_quality",
]


def make_folds(y):
    """Canonical CV folds -- IDENTICAL across stages 2/3/4 AND identical to the
    Feb-2026 run (same KFold class, same shuffle seed, same row order; KFold ignores
    y beyond its length). Every train/eval script must call this, never its own
    KFold, so folds stay byte-identical."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(kf.split(np.zeros(len(y))))


def _label_maps(train_raw, test_raw):
    """Deterministic label codes from the combined train+test unique values (sorted;
    target never involved) -- identical to Feb's LabelEncoder-on-concat approach and
    identical across train/test (EDA: no unseen categories in test)."""
    maps = {}
    for c in LABEL_COLS:
        cats = sorted(set(train_raw[c].dropna().unique()) | set(test_raw[c].dropna().unique()))
        maps[c] = {v: i for i, v in enumerate(cats)}
    return maps


def engineer(df, label_maps):
    """Deterministic feature engineering -- verbatim port of Feb's block in
    scripts/full_pipeline.py (same formulas, same output columns)."""
    out = df.copy()
    out["sleep_quality_ord"] = out["sleep_quality"].map(SLEEP_QUALITY_MAP)
    out["facility_rating_ord"] = out["facility_rating"].map(FACILITY_RATING_MAP)
    out["exam_difficulty_ord"] = out["exam_difficulty"].map(EXAM_DIFFICULTY_MAP)
    out["gender_enc"] = out["gender"].map(label_maps["gender"])
    out["course_enc"] = out["course"].map(label_maps["course"])
    out["internet_enc"] = out["internet_access"].map(label_maps["internet_access"])
    out["study_method_enc"] = out["study_method"].map(label_maps["study_method"])
    out["study_x_attendance"] = out["study_hours"] * out["class_attendance"]
    out["study_x_sleep_quality"] = out["study_hours"] * out["sleep_quality_ord"]
    out["study_x_facility"] = out["study_hours"] * out["facility_rating_ord"]
    out["attendance_x_sleep"] = out["class_attendance"] * out["sleep_hours"]
    out["study_sq"] = out["study_hours"] ** 2
    out["attendance_sq"] = out["class_attendance"] ** 2
    out["total_effort"] = (out["study_hours"] / 8
                           + out["class_attendance"] / 100
                           + out["sleep_quality_ord"] / 2) / 3
    out["study_method_score"] = out["study_method"].map(STUDY_METHOD_SCORE_MAP)
    out["method_x_hours"] = out["study_method_score"] * out["study_hours"]
    out["sleep_deficit"] = 8 - out["sleep_hours"]
    out["sleep_x_quality"] = out["sleep_hours"] * out["sleep_quality_ord"]
    return out


def build_all(train_raw, test_raw):
    """Engineer both frames; return (train_e, test_e, feature_cols) with the 22-column
    Feb feature set. The first 11 columns are the BASE encoded set, the remaining 11
    are INTERACTION_FEATURES (in that order)."""
    label_maps = _label_maps(train_raw, test_raw)
    train_e = engineer(train_raw, label_maps)
    test_e = engineer(test_raw, label_maps)
    base = ["age", "study_hours", "class_attendance", "sleep_hours",
            "sleep_quality_ord", "facility_rating_ord", "exam_difficulty_ord",
            "gender_enc", "course_enc", "internet_enc", "study_method_enc"]
    feature_cols = base + INTERACTION_FEATURES
    return train_e, test_e, feature_cols


def r2_clip(y_true, pred, lo=CLIP_LO, hi=CLIP_HI):
    """Metric-aware post-processing for R2 on the [0,100]-bounded exam score: clip
    predictions into the target's physical range, then R2. This is this comp's
    analogue of s5e10's clip[0,1] -- every decision score in every stage goes through
    here, never through raw-R2. (Clipping can only help a squared-error-based metric
    when the target itself is bounded; Feb's pipeline applied the same [0,100] clip
    before scoring, so scores remain directly comparable.)"""
    p = np.clip(np.asarray(pred, dtype=np.float64), lo, hi)
    y = np.asarray(y_true, dtype=np.float64)
    return float(r2_score(y, p))
