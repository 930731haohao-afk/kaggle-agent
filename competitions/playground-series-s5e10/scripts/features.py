"""Feature engineering module for playground-series-s5e10 (accident_risk, RMSE).

Reused unmodified by scripts/04_train_blend.py (stage 2), scripts/05_iterate.py
(stage 3), tree_search/eval_s5e10.py (stage 4) and scripts/06_rebuild_tree_best.py
(rebuild gate) so that all stages train on byte-identical features and the SAME CV
folds: KFold(5, shuffle=True, random_state=42) on row order.

Fold-comparability note vs the Feb-2026 prior-season run (scripts/full_pipeline.py):
Feb used exactly KFold(5, shuffle=True, random_state=42) on the same train.csv row
order; sklearn's KFold split depends only on n_samples and the seed, so the folds
produced by make_folds() here are IDENTICAL to Feb's -- Feb's logged scores
(experiments.json exp 1-4) are directly comparable to every score this run produces.

Feature set: verbatim port of Feb's engineering block in full_pipeline.py --
27 features = 4 raw numerics + 4 ordinal-encoded categoricals + 4 bool->int flags
+ 7 pairwise products with the top predictors + 3 polynomial terms + 3 binary risk
combos + 2 lane interactions. No missing values anywhere, no target encoding of any
kind (nothing fold-dependent), so the whole transform is deterministic and leak-free
by construction.

INTERACTION_FEATURES lists the 15 engineered product/poly/combo columns as one
group: knowledge/experience.md carries a negative prior ("trees learn multiplicative
interactions themselves; explicit product columns add collinearity, not information",
s3e9 evidence) while Feb's STATUS.md claims these interactions are key. Stage 3
round 1 resolves this with a controlled same-folds on/off comparison before the set
is trusted any further (see scripts/05_iterate.py).

Metric-aware post-processing for RMSE on a [0,1]-bounded target: predictions are
clipped to [0,1] before scoring -- rmse_clip() is the ONLY decision metric in this
comp, used identically by every stage's weight search and every logged score
(knowledge/experience.md's rule: decide on the post-processed score, never raw).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

TARGET, ID = "accident_risk", "id"
N_SPLITS, SEED = 5, 42

ROAD_TYPE_MAP = {"rural": 0, "urban": 1, "highway": 2}
LIGHTING_MAP = {"daylight": 0, "dim": 1, "night": 2}
WEATHER_MAP = {"clear": 0, "rainy": 1, "foggy": 2}
TIME_MAP = {"morning": 0, "afternoon": 1, "evening": 2}

DROP_RAW = [TARGET, ID, "road_type", "lighting", "weather", "time_of_day",
            "road_signs_present", "public_road", "holiday", "school_season"]

# The 15 engineered interaction/polynomial/combo features (everything beyond the
# 12 encoded raw columns) -- the unit under test in stage 3 round 1's prior check.
INTERACTION_FEATURES = [
    "curvature_x_speed", "curvature_x_lighting", "speed_x_lighting",
    "curvature_x_weather", "speed_x_weather", "curvature_x_accidents",
    "speed_x_accidents", "curvature_sq", "speed_sq", "curvature_x_speed_sq",
    "night_rain", "night_fog", "high_speed_curve", "lanes_x_speed",
    "lanes_x_curvature",
]


def make_folds(y):
    """Canonical CV folds -- IDENTICAL across stages 2/3/4 AND identical to the
    Feb-2026 run (same KFold class, same shuffle seed, same row order; KFold ignores
    y beyond its length). Every train/eval script must call this, never its own
    KFold, so folds stay byte-identical."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(kf.split(np.zeros(len(y))))


def engineer(df):
    """Deterministic feature engineering -- verbatim port of Feb's block in
    scripts/full_pipeline.py (same formulas, same output column order)."""
    out = df.copy()
    out["road_type_enc"] = out["road_type"].map(ROAD_TYPE_MAP)
    out["lighting_enc"] = out["lighting"].map(LIGHTING_MAP)
    out["weather_enc"] = out["weather"].map(WEATHER_MAP)
    out["time_of_day_enc"] = out["time_of_day"].map(TIME_MAP)
    out["road_signs_int"] = out["road_signs_present"].astype(int)
    out["public_road_int"] = out["public_road"].astype(int)
    out["holiday_int"] = out["holiday"].astype(int)
    out["school_season_int"] = out["school_season"].astype(int)
    out["curvature_x_speed"] = out["curvature"] * out["speed_limit"]
    out["curvature_x_lighting"] = out["curvature"] * out["lighting_enc"]
    out["speed_x_lighting"] = out["speed_limit"] * out["lighting_enc"]
    out["curvature_x_weather"] = out["curvature"] * out["weather_enc"]
    out["speed_x_weather"] = out["speed_limit"] * out["weather_enc"]
    out["curvature_x_accidents"] = out["curvature"] * out["num_reported_accidents"]
    out["speed_x_accidents"] = out["speed_limit"] * out["num_reported_accidents"]
    out["curvature_sq"] = out["curvature"] ** 2
    out["speed_sq"] = out["speed_limit"] ** 2
    out["curvature_x_speed_sq"] = out["curvature"] * out["speed_limit"] ** 2
    out["night_rain"] = ((out["lighting_enc"] == 2) & (out["weather_enc"] == 1)).astype(int)
    out["night_fog"] = ((out["lighting_enc"] == 2) & (out["weather_enc"] == 2)).astype(int)
    out["high_speed_curve"] = out["curvature"] * (out["speed_limit"] >= 60).astype(int)
    out["lanes_x_speed"] = out["num_lanes"] * out["speed_limit"]
    out["lanes_x_curvature"] = out["num_lanes"] * out["curvature"]
    return out


def build_all(train_raw, test_raw):
    """Engineer both frames; return (train_e, test_e, feature_cols) with the same
    27-column order Feb's pipeline used."""
    train_e = engineer(train_raw)
    test_e = engineer(test_raw)
    feature_cols = [c for c in train_e.columns if c not in DROP_RAW]
    return train_e, test_e, feature_cols


def rmse_clip(y_true, pred, lo=0.0, hi=1.0):
    """Metric-aware post-processing for RMSE on the [0,1]-bounded target: clip
    predictions into the target's physical range, then RMSE. This is this comp's
    analogue of s4e11's threshold sweep / s3e16's integer rounding -- every decision
    score in every stage goes through here, never through raw-RMSE. (Clipping can
    only help RMSE when the target itself is bounded; Feb's pipeline applied the
    same clip before scoring, so scores remain directly comparable.)"""
    p = np.clip(np.asarray(pred, dtype=np.float64), lo, hi)
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def snap_to_grid(pred, step=0.01, lo=0.0, hi=1.0):
    """Round predictions to the target's 0.01 grid (train target has only 98 unique
    values, all on this grid). Kept as a UTILITY for stage 3 round 1's second prior
    check only -- the snap-to-grid prior in knowledge/experience.md was earned on
    MAE (s3e14); whether it transfers to a squared-error metric is exactly what the
    controlled check decides before anyone uses it."""
    return np.clip(np.round(np.asarray(pred, dtype=np.float64) / step) * step, lo, hi)
