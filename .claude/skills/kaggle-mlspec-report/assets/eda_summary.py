"""
Deterministic EDA fact extractor for kaggle-mlspec-report.

Usage (from project root):
    uv run python3 .claude/skills/kaggle-mlspec-report/assets/eda_summary.py <competition-name>

Reads  competitions/<name>/config.yaml + data/train.csv + data/test.csv
Writes competitions/<name>/eda_summary.json

This turns the previously-ephemeral EDA (eda.py stdout) into a STRUCTURED,
saved artifact — the same shape as config.yaml / experiments.json — so collect.py
can wire it into facts.json and the report's data/eval sections become traceable.

Every number here is computed straight from the data files; the LLM never
recomputes EDA — it only reads eda_summary.json via facts.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml


def _num(x):
    """JSON-safe float (handles numpy types / NaN)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 6)


def summarize(comp_dir: str) -> dict:
    cfg = yaml.safe_load(open(os.path.join(comp_dir, "config.yaml")))
    target, idc = cfg["target_column"], cfg["id_column"]
    data = os.path.join(comp_dir, "data")
    train = pd.read_csv(os.path.join(data, cfg.get("train_file", "train.csv")))
    test_path = os.path.join(data, cfg.get("test_file", "test.csv"))
    test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    feats = [c for c in train.columns if c not in (idc, target)]
    num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
    cat_feats = [c for c in feats if c not in num_feats]

    out = {
        "source": "eda_summary.py",
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)) if test is not None else None,
        "n_features": len(feats),
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "duplicate_rows_train": int(train[feats].duplicated().sum()),
    }

    # ---- target ----
    y = train[target]
    is_num_target = pd.api.types.is_numeric_dtype(y)
    tgt = {"name": target, "dtype": str(y.dtype), "n_unique": int(y.nunique()),
           "missing": int(y.isna().sum())}
    if is_num_target:
        tgt.update({k: _num(v) for k, v in {
            "mean": y.mean(), "std": y.std(), "min": y.min(), "p25": y.quantile(.25),
            "median": y.median(), "p75": y.quantile(.75), "max": y.max(),
            "skew": y.skew(), "kurtosis": y.kurtosis()}.items()})
        tgt["is_integer_valued"] = bool(np.allclose(y.dropna() % 1, 0))
        tgt["log_transform_candidate"] = bool(y.min() >= 0 and abs(y.skew()) > 1.0)
    else:
        vc = y.value_counts()
        tgt["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        tgt["imbalance_ratio"] = _num(vc.max() / vc.min()) if len(vc) > 1 else None
    if is_num_target and y.nunique() <= 30:
        tgt["value_counts"] = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    out["target"] = tgt

    # ---- missing values ----
    miss_tr = {c: int(train[c].isna().sum()) for c in feats if train[c].isna().any()}
    out["missing_train"] = miss_tr
    if test is not None:
        out["missing_test"] = {c: int(test[c].isna().sum())
                               for c in feats if c in test and test[c].isna().any()}

    # ---- numeric feature stats ----
    out["numeric_stats"] = {c: {k: _num(v) for k, v in {
        "mean": train[c].mean(), "std": train[c].std(), "min": train[c].min(),
        "max": train[c].max(), "skew": train[c].skew(),
        "zeros": int((train[c] == 0).sum())}.items()} for c in num_feats}

    # ---- categorical feature stats ----
    out["categorical_stats"] = {}
    for c in cat_feats:
        d = {"cardinality": int(train[c].nunique()),
             "top": {str(k): int(v) for k, v in train[c].value_counts().head(5).items()}}
        if test is not None and c in test:
            d["unseen_in_test"] = sorted(map(str, set(test[c].dropna().unique())
                                             - set(train[c].dropna().unique())))[:20]
        out["categorical_stats"][c] = d

    # ---- correlation with target (numeric target only) ----
    if is_num_target and num_feats:
        pear = train[num_feats + [target]].corr("pearson")[target].drop(target)
        spear = train[num_feats + [target]].corr("spearman")[target].drop(target)
        out["target_correlation"] = {c: {"pearson": _num(pear[c]), "spearman": _num(spear[c])}
                                     for c in num_feats}
        # collinear feature pairs (|r|>0.95) — flags redundancy
        cmat = train[num_feats].corr().abs()
        pairs = []
        for i in range(len(num_feats)):
            for j in range(i + 1, len(num_feats)):
                if cmat.iloc[i, j] > 0.95:
                    pairs.append([num_feats[i], num_feats[j], _num(cmat.iloc[i, j])])
        out["high_collinearity_pairs"] = pairs

    # ---- train/test distribution shift (numeric) ----
    if test is not None and num_feats:
        shift = {}
        for c in num_feats:
            if c in test:
                tm, em = train[c].mean(), test[c].mean()
                shift[c] = _num((em - tm) / tm * 100) if tm else None
        out["train_test_mean_shift_pct"] = shift

    # ---- validation hint (heuristic, factual) ----
    if is_num_target and y.nunique() <= 50:
        hint = "target is discrete/integer -> StratifiedKFold on binned target for stable folds"
    elif not is_num_target:
        hint = "classification -> StratifiedKFold on the class label"
    else:
        hint = "continuous i.i.d. target -> standard KFold (check train/test shift first)"
    date_like = [c for c in cat_feats if "date" in c.lower()]
    if date_like:
        hint = f"date column(s) {date_like} present -> consider time-based validation (avoid leakage)"
    out["validation_hint"] = hint

    return out


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: eda_summary.py <competition-name>")
    comp_dir = os.path.join("competitions", sys.argv[1])
    if not os.path.exists(os.path.join(comp_dir, "config.yaml")):
        sys.exit(f"ERROR: config.yaml not found in {comp_dir}")
    facts = summarize(comp_dir)
    out = os.path.join(comp_dir, "eda_summary.json")
    with open(out, "w") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    print(f"wrote {out}: {facts['n_features']} features, "
          f"target={facts['target']['name']}, hint={facts['validation_hint']}")


if __name__ == "__main__":
    main()
