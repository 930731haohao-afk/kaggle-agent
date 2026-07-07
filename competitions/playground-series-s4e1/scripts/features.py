"""Feature engineering module for playground-series-s4e1 (Bank Churn, ROC-AUC).

Reused unmodified by scripts/04_train_blend.py (tier2), scripts/05_iterate*.py (tier3),
and tree_search/eval_s4e1.py (tier4) so that all four tiers train on byte-identical
features and the SAME CV folds (StratifiedKFold(5, shuffle=True, seed=42) on Exited).

Carries forward the Feb-2026 EDA/feature set documented in STATUS.md (28 features:
gender/geography dummies, age/balance/products derived features, seven interaction
terms, surname target encoding) with ONE deliberate correction:

  Feb baseline (scripts/02_baseline.py) computed the surname target encoding with an
  INDEPENDENT 5-fold split (seed=99) from the model's own training folds (seed=42).
  Because encoding groups (surnames) are shared across the two different fold
  partitions, a model-fold's training rows could have their surname_target_enc value
  partly informed by rows that sit in that same model-fold's *validation* split under
  the seed=99 partition -- a subtle target leak through the shared grouping key.
  This module instead computes the encoding fold_safe with the EXACT SAME folds used
  for model training (matches the established fold_safe_te pattern in
  competitions/playground-series-s3e11/scripts/iterate2.py), which removes that leak
  path entirely. Kept smoothing=20 from the Feb version (empirically reasonable given
  ~2,932 unique surnames over 165k rows).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET, ID = "Exited", "id"
N_SPLITS, SEED = 5, 42
DROP_RAW = [ID, TARGET, "CustomerId", "Surname"]


def make_folds(y):
    """Canonical CV folds -- IDENTICAL across tier2/tier3/tier4. Call once per process
    with the training target vector; every eval/train script must use this, not its own
    StratifiedKFold call, so folds are guaranteed byte-identical."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def engineer_base(df):
    """Deterministic (no-leakage) engineered features -- everything except the
    surname target encoding, which needs the fold assignment and is added separately
    by add_surname_te() so callers control which folds it's fit_safe against."""
    out = df.copy()
    out["is_female"] = (out["Gender"] == "Female").astype(int)
    for geo in ["Germany", "Spain"]:
        out[f"geo_{geo}"] = (out["Geography"] == geo).astype(int)
    out["age_sq"] = out["Age"] ** 2
    out["age_decade"] = out["Age"] // 10
    out["age_40_60"] = ((out["Age"] >= 40) & (out["Age"] <= 60)).astype(int)
    out["has_balance"] = (out["Balance"] > 0).astype(int)
    out["balance_salary_ratio"] = out["Balance"] / out["EstimatedSalary"].clip(lower=1)
    out["products_gt2"] = (out["NumOfProducts"] > 2).astype(int)
    out["products_eq1"] = (out["NumOfProducts"] == 1).astype(int)
    out["age_x_products"] = out["Age"] * out["NumOfProducts"]
    out["age_x_balance"] = out["Age"] * out["Balance"]
    out["age_x_active"] = out["Age"] * out["IsActiveMember"]
    out["geo_gender"] = out["Geography"].astype(str) + "_" + out["Gender"].astype(str)
    out["active_x_balance"] = out["IsActiveMember"] * out["Balance"]
    out["active_x_products"] = out["IsActiveMember"] * out["NumOfProducts"]
    out["credit_age_ratio"] = out["CreditScore"] / out["Age"].clip(lower=1)
    return out


def label_encode_strings(train_e, test_e, cols=("Geography", "Gender", "geo_gender")):
    """Label-encode remaining string columns (union of train+test categories) in
    place. Returns the two frames for chaining. Default cols matches the Feb baseline's
    `string_cols` loop (any object-dtype column not in DROP_RAW)."""
    train_e = train_e.copy()
    test_e = test_e.copy()
    for c in cols:
        cats = pd.Index(pd.concat([train_e[c].astype(str), test_e[c].astype(str)]).unique())
        mp = {v: i for i, v in enumerate(cats)}
        train_e[c] = train_e[c].astype(str).map(mp).astype("int32")
        test_e[c] = test_e[c].astype(str).map(mp).astype("int32")
    return train_e, test_e


def add_surname_te(train_e, test_e, y, folds, smoothing=20):
    """Fold-safe surname target encoding: for each of `folds`, fit the smoothed
    surname->mean(Exited) map on the fold's TRAIN partition only and apply it to that
    fold's held-out VALIDATION rows (train OOF) and accumulate 1/N_SPLITS of the test
    prediction from each fold's map (test TE = average over fold maps, no leakage since
    test has no target). `folds` must be the exact list returned by make_folds(y)."""
    n_splits = len(folds)
    gm = float(y.mean())
    te_tr = np.zeros(len(train_e))
    te_te = np.zeros(len(test_e))
    surnames_tr = train_e["Surname"]
    surnames_te = test_e["Surname"]
    for tr_idx, va_idx in folds:
        stats = (pd.Series(y[tr_idx], index=surnames_tr.iloc[tr_idx].index)
                 .groupby(surnames_tr.iloc[tr_idx]).agg(["mean", "count"]))
        smoothed = (stats["mean"] * stats["count"] + gm * smoothing) / (stats["count"] + smoothing)
        te_tr[va_idx] = surnames_tr.iloc[va_idx].map(smoothed).fillna(gm).to_numpy()
        te_te += surnames_te.map(smoothed).fillna(gm).to_numpy() / n_splits
    train_e = train_e.copy()
    test_e = test_e.copy()
    train_e["surname_target_enc"] = te_tr
    test_e["surname_target_enc"] = te_te
    return train_e, test_e


def build_all(train_raw, test_raw, y, folds):
    """One-call convenience: engineer + label-encode + fold-safe surname TE. Returns
    (Xtr_df, Xte_df, feature_cols) -- Xtr_df/Xte_df still carry id/target/raw string
    cols so callers can select via feature_cols."""
    train_e = engineer_base(train_raw)
    test_e = engineer_base(test_raw)
    train_e, test_e = label_encode_strings(train_e, test_e)
    train_e, test_e = add_surname_te(train_e, test_e, y, folds)
    feature_cols = [c for c in train_e.columns if c not in DROP_RAW]
    return train_e, test_e, feature_cols
