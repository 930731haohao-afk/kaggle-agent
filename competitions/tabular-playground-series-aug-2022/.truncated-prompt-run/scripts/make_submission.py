"""Final submission: LR(C=0.02) on [loading, m17, m3_na, m5_na, m2], full-train fit,
per-product rank normalization on test predictions."""
import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, load_raw, standardize_per_product

warnings.filterwarnings("ignore")
SEED = 42
COMP_DIR = os.path.join(os.path.dirname(__file__), "..")

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    os.path.join(COMP_DIR, "..", "..", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py"))
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COLS = ["loading", "measurement_17", "m3_na", "m5_na", "measurement_2"]
SCALE_COLS = ["loading", "measurement_17", "measurement_2"]
C = 0.02
CV_SCORE = 0.591310


def main():
    train, test = load_raw()
    ftr = standardize_per_product(build_features(train), SCALE_COLS)
    fte = standardize_per_product(build_features(test), SCALE_COLS)

    model = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    model.fit(ftr[COLS], ftr["failure"])
    preds = model.predict_proba(fte[COLS])[:, 1]

    # per-product rank normalization (products disjoint; AUC pooled across them)
    out = np.zeros(len(fte))
    for pc in fte["product_code"].unique():
        m = (fte["product_code"] == pc).values
        out[m] = rankdata(preds[m]) / m.sum()

    sample = pd.read_csv(os.path.join(COMP_DIR, "data", "sample_submission.csv"))
    sub = pd.DataFrame({sample.columns[0]: test["id"].values,
                        sample.columns[1]: out})
    assert sub.shape == sample.shape, (sub.shape, sample.shape)
    assert (sub[sample.columns[0]].values == sample[sample.columns[0]].values).all()
    assert sub.isnull().sum().sum() == 0
    assert sub[sample.columns[1]].between(0, 1).all()

    path = os.path.join(COMP_DIR, "submission.csv")
    sub.to_csv(path, index=False)
    ts_path = os.path.join(COMP_DIR, "submissions",
                           f"submission_lr_rankpp_{CV_SCORE:.5f}.csv")
    sub.to_csv(ts_path, index=False)
    print(f"wrote {path} and {ts_path}")
    print(sub[sample.columns[1]].describe())

    experiment_log.log_experiment_v2(
        COMP_DIR, model=f"LogisticRegression(C={C}) full-train",
        metric="roc_auc", direction="maximize", score=CV_SCORE,
        cv={"strategy": "GroupKFold(product_code, 5)"},
        features=COLS,
        postprocess=["per-product rank normalization on test"],
        submission=os.path.basename(ts_path),
        notes="FINAL submission; raw loading + per-product std-scale; "
              "m17 Huber-imputed per product",
    )


if __name__ == "__main__":
    main()
