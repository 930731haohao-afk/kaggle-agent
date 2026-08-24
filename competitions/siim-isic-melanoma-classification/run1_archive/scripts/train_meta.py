"""Stage 3 baseline: metadata / patient-context GBDT (no pixels)."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common as C


def run(use_te: bool, model: str = "lgb", tag: str | None = None) -> dict:
    C.set_seed()
    tr, te = C.load_meta()
    Xtr, Xte = C.build_tabular(tr, te)
    y = tr.target.to_numpy()
    folds = tr.fold.to_numpy()
    te_tr, te_val, te_te = C.patient_te(tr, te)

    oof = np.zeros(len(tr))
    pred = np.zeros(len(te))
    t0 = time.time()
    for k in range(C.N_FOLDS):
        m = folds != k
        xa, xb, xc = Xtr[m].copy(), Xtr[~m].copy(), Xte.copy()
        if use_te:
            xa["pat_te"] = te_tr[m, k]
            xb["pat_te"] = te_val[~m]
            xc["pat_te"] = te_te[:, k]
        if model == "lgb":
            import lightgbm as lgb

            mdl = lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=5,
                min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=2.0, random_state=C.SEED, n_jobs=10,
                deterministic=True, force_row_wise=True, verbose=-1)
            mdl.fit(xa, y[m])
        else:
            from catboost import CatBoostClassifier

            mdl = CatBoostClassifier(
                iterations=800, learning_rate=0.03, depth=5, l2_leaf_reg=8.0,
                random_seed=C.SEED, verbose=0, allow_writing_files=False, thread_count=10)
            mdl.fit(xa.fillna(-999), y[m])
            xb, xc = xb.fillna(-999), xc.fillna(-999)
        oof[~m] = mdl.predict_proba(xb)[:, 1]
        pred += mdl.predict_proba(xc)[:, 1] / C.N_FOLDS
    sec = time.time() - t0
    score = C.auc(y, oof)
    per_fold = [C.auc(y[folds == k], oof[folds == k]) for k in range(C.N_FOLDS)]
    name = tag or f"{model}_meta{'_te' if use_te else ''}"
    C.ART.mkdir(exist_ok=True)
    np.save(C.ART / f"oof_{name}.npy", oof)
    np.save(C.ART / f"pred_{name}.npy", pred)
    print(f"{name}: OOF AUC {score:.5f}  folds {[round(v, 4) for v in per_fold]}  {sec:.0f}s")
    return {"name": name, "score": score, "per_fold": per_fold, "sec": sec,
            "n_feats": Xtr.shape[1] + int(use_te)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgb")
    a = ap.parse_args()
    res = []
    for mdl in a.models.split(","):
        for use_te in (False, True):
            res.append(run(use_te, mdl))
    (C.COMP / "meta_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
