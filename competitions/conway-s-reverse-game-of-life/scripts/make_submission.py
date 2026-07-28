"""Write submission.csv from a test-probability npz.

Usage: make_submission.py <preds_npz> <out_csv> [thresholds_json]
thresholds_json: optional per-delta {"1":0.5,...} (default 0.5 everywhere).
Column names / id order taken from sampleSubmission.csv.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "competitions/conway-s-reverse-game-of-life/scripts")
import common


def main(npz_path, out_csv, thresh=None):
    a = common.load_arrays()
    tdelta, test_ids = a["tdelta"], a["test_ids"]
    probs = np.load(npz_path)["test"]                    # (50000,20,20)
    thresh = thresh or {}
    pred = np.zeros_like(probs, dtype=np.int8)
    for d in range(1, 6):
        m = tdelta == d
        t = float(thresh.get(str(d), 0.5))
        pred[m] = (probs[m] > t).astype(np.int8)
    sample = pd.read_csv(f"{common.DATA}/sampleSubmission.csv", nrows=1)
    cols = list(sample.columns)
    assert cols[0] == "id" and len(cols) == 401
    flat = pred.reshape(len(pred), 400)
    df = pd.DataFrame(flat, columns=cols[1:])
    df.insert(0, "id", test_ids)
    # sampleSubmission id order = test.csv id order (both ascending)
    assert (df["id"].to_numpy() == np.sort(df["id"].to_numpy())).all()
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: shape {df.shape}, live-cell rate {flat.mean():.4f}")
    return df


if __name__ == "__main__":
    th = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], th)
