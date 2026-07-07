"""Feature engineering module for playground-series-s6e2 (Heart Disease, ROC-AUC).

Reused unmodified by scripts/04_train_blend.py (stage 2), scripts/05_iterate.py
(stage 3), tree_search/eval_s6e2.py (stage 4) and scripts/06_rebuild_tree_best.py
(rebuild gate) so that all stages train on byte-identical features and the SAME CV
folds: StratifiedKFold(5, shuffle, random_state=42) on the (Presence=1 / Absence=0)
target.

Fold-comparability note vs the Feb-2026 prior-season run: Feb's v1 baseline
(scripts/02_train_ensemble.py) used exactly StratifiedKFold(5, shuffle,
random_state=42) on the same train.csv row order + target; StratifiedKFold's split
depends only on the target vector and the seed, so make_folds() here reproduces Feb
v1's folds -- Feb's v1 ensemble OOF AUC (0.95498) is the fold-identical stage-1 record
and is directly comparable to every score this run produces. (Feb's later v2-v5 used
different, non-canonical CV schemes -- 10-fold, 5-seed x 5-fold and 10-seed x 10-fold
averaging -- so they are NOT byte-comparable to this single-fold ladder and are kept as
context only, never as ladder rungs; cross-CV comparison is the one cardinal sin per
knowledge/experience.md.)

Feature set: verbatim port of Feb's focused v3 engineering block
(scripts/04_train_multiseed_v3.py, which Feb's STATUS.md concluded optimal: "focused
33 features" beat the 53- and 58-feature variants). 33 features = 13 raw UCI numeric
columns (BASE_FEATURES) + 20 engineered columns (ENGINEERED_FEATURES): 5 medical-
threshold / heart-rate features, 11 pairwise interaction products, 1 domain risk
composite, 2 age ratios, 1 Thallium==7 flag. All 13 raw columns are already numeric
(no categorical strings, no missing values, no target encoding of any kind -- nothing
fold-dependent), so the whole transform is deterministic and leak-free by construction;
codes are identical across train and test.

ENGINEERED_FEATURES is listed as one group so stage 3 round 1 can run a controlled
same-folds on/off check of the experience-library prior ("trees learn multiplicative
interactions themselves; explicit product columns add collinearity, not information",
s3e9; "too many features regress, pruning is a routine win", s3e7/s3e14) against Feb's
STATUS.md claim that these engineered interactions are what carry the signal -- the
exact same prior s6e1 REJECTED and s5e10 CONFIRMED, re-verified here on this comp.

Metric-aware post-processing: NONE. ROC-AUC is a pure ranking metric -- any monotone
transform of the scores (including clipping probabilities to [0,1], which predict_proba
already guarantees) leaves the AUC unchanged -- so the submission carries the raw
positive-class probabilities and there is no threshold/rounding step (submitting hard
0/1 labels would destroy the ranking AUC needs). auc() below is the ONLY decision
metric, used identically by every stage's weight search and every logged score.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET, ID, POS_LABEL = "Heart Disease", "id", "Presence"
N_SPLITS, SEED = 5, 42

# 13 raw UCI Heart-Disease numeric columns (train.csv order), all already numeric.
BASE_FEATURES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression", "Slope of ST",
    "Number of vessels fluro", "Thallium",
]

# The 20 engineered columns (Feb focused v3 set) -- the unit under test in stage 3
# round 1's interaction/feature-count prior check.
ENGINEERED_FEATURES = [
    "Age_decade", "BP_high", "Chol_high", "HR_reserve", "HR_pct_max",
    "Age_x_MaxHR", "Age_x_STdep", "Age_x_Vessels", "Chol_x_Age", "BP_x_Chol",
    "BP_x_Age", "STdep_x_Slope", "STdep_x_MaxHR", "Vessels_x_Thallium",
    "Angina_x_STdep", "ChestPain_x_Angina", "Risk_score", "Chol_per_Age",
    "BP_per_Age", "Thallium_7",
]


def make_folds(y):
    """Canonical CV folds -- IDENTICAL across stages 2/3/4 AND identical to Feb v1's
    StratifiedKFold(5, shuffle, random_state=42) on the same target/row order.
    StratifiedKFold's split depends only on y and the seed, so every train/eval script
    must call this (never its own splitter) to keep folds byte-identical."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def encode_target(train_raw):
    """Presence -> 1, Absence -> 0 (int64). The submission still carries continuous
    positive-class probabilities; this integer target is only for training/scoring."""
    return (train_raw[TARGET] == POS_LABEL).astype(np.int64).to_numpy()


def engineer(df):
    """Deterministic feature engineering -- verbatim port of Feb's focused v3 block in
    scripts/04_train_multiseed_v3.py (same formulas, same output columns)."""
    out = df.copy()
    out["Age_decade"] = out["Age"] // 10
    out["BP_high"] = (out["BP"] >= 140).astype(int)
    out["Chol_high"] = (out["Cholesterol"] > 240).astype(int)
    out["HR_reserve"] = 220 - out["Age"] - out["Max HR"]
    out["HR_pct_max"] = out["Max HR"] / (220 - out["Age"] + 1)
    out["Age_x_MaxHR"] = out["Age"] * out["Max HR"]
    out["Age_x_STdep"] = out["Age"] * out["ST depression"]
    out["Age_x_Vessels"] = out["Age"] * out["Number of vessels fluro"]
    out["Chol_x_Age"] = out["Cholesterol"] * out["Age"]
    out["BP_x_Chol"] = out["Cholesterol"] * out["BP"]
    out["BP_x_Age"] = out["BP"] * out["Age"]
    out["STdep_x_Slope"] = out["ST depression"] * out["Slope of ST"]
    out["STdep_x_MaxHR"] = out["ST depression"] * out["Max HR"]
    out["Vessels_x_Thallium"] = out["Number of vessels fluro"] * out["Thallium"]
    out["Angina_x_STdep"] = out["Exercise angina"] * out["ST depression"]
    out["ChestPain_x_Angina"] = out["Chest pain type"] * out["Exercise angina"]
    out["Risk_score"] = (
        out["Age"] / 77 + out["Sex"] + out["Chest pain type"] / 4
        + out["BP"] / 200 + out["Cholesterol"] / 564 + out["FBS over 120"]
        + out["Exercise angina"] + out["ST depression"] / 6.2
        + out["Number of vessels fluro"] / 3 + (out["Thallium"] == 7).astype(int)
    )
    out["Chol_per_Age"] = out["Cholesterol"] / (out["Age"] + 1)
    out["BP_per_Age"] = out["BP"] / (out["Age"] + 1)
    out["Thallium_7"] = (out["Thallium"] == 7).astype(int)
    return out


def build_all(train_raw, test_raw):
    """Engineer both frames; return (train_e, test_e, feature_cols) with the 33-column
    focused Feb feature set. The first 13 columns are BASE_FEATURES, the remaining 20
    are ENGINEERED_FEATURES (in that order)."""
    train_e = engineer(train_raw)
    test_e = engineer(test_raw)
    feature_cols = BASE_FEATURES + ENGINEERED_FEATURES
    return train_e, test_e, feature_cols


def auc(y_true, y_score):
    """ROC-AUC -- the ONLY decision metric in this comp (ranking metric, no
    post-processing: any monotone transform of the scores leaves it unchanged)."""
    return float(roc_auc_score(y_true, y_score))
