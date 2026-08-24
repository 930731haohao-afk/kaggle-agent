"""Stage 5 — write submission.csv from a saved prediction tag, validated against the sample.

Note on "retrain on full data": every member's test prediction is already the average of its
5 fold models (bagging), which is the standard choice for CNN ensembles and keeps the
submitted predictor identical in structure to the one the CV score measured. A single
full-data refit would change the predictor without any way to validate it, so it is not used.
"""
from __future__ import annotations

import json
import logging
import sys

import pandas as pd

import common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "blend"
    sample = pd.read_csv(C.DATA / "sample_submission.csv")
    test = pd.read_csv(C.DATA / "test.csv")
    _, pred = C.load_pred(tag)

    sub = pd.DataFrame({"image_name": test.image_name.values, "target": pred})
    # AUC is rank-invariant, but a [0,1] range is what the sample implies; min-max keeps ranks.
    sub["target"] = (sub.target - sub.target.min()) / (sub.target.max() - sub.target.min())
    sub = sample[["image_name"]].merge(sub, on="image_name", how="left")

    assert sub.shape == sample.shape, f"shape {sub.shape} != {sample.shape}"
    assert (sub.image_name.values == sample.image_name.values).all(), "id order mismatch"
    assert sub.target.notna().all(), "NaN in predictions"
    assert sub.target.between(0, 1).all(), "predictions outside [0,1]"
    assert sub.target.nunique() > len(sub) * 0.9, "predictions are suspiciously non-unique"

    path = C.COMP / "submission.csv"
    sub.to_csv(path, index=False)
    log.info("wrote %s  rows=%d  min=%.5f mean=%.5f max=%.5f",
             path, len(sub), sub.target.min(), sub.target.mean(), sub.target.max())
    log.info("head:\n%s", sub.head().to_string(index=False))
    cv = json.loads((C.COMP / "blend_results.json").read_text())["leave_fold_out"]
    C.log_exp(model=f"FINAL SUBMISSION (tag={tag})", metric="roc_auc", direction="maximize",
              score=cv, submission="submission.csv",
              notes=f"honest leave-fold-out CV of the submitted predictor; {len(sub)} rows, "
                    f"id order verified against sample_submission.csv; min-max scaled "
                    f"(rank-preserving, AUC unaffected)")


if __name__ == "__main__":
    main()
