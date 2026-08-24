"""Stage 4b — level-2 stack: LGB on [member ranks + metadata features].

A convex blend can only reweight members globally. A stacker can condition on context, e.g.
"trust the CNN less on 640x480 images from this site" or "trust it less for very old patients".
Whether that extra capacity pays for its overfitting risk is decided by the same honest
leave-fold-out protocol used for the convex blend — nested, so the level-2 model never scores
a fold it was fitted on.

Member OOFs come from models that did not see the scored row, and the level-2 model is fitted
on the other folds only, so the stack is fold-safe end to end.
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata

import common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def rk(x):
    return rankdata(x) / len(x)


def main() -> None:
    tags = sys.argv[1:]
    f = pd.read_csv(C.CACHE / "r2_folds.csv")
    y, folds, gfolds = f.target.values, f.fold.values, f.gfold.values

    tr_raw, te_raw = C.load_meta()
    tr_raw = C.make_folds(tr_raw)
    tr, te, feats = C.build_features(tr_raw, te_raw)
    ctx = ["age", "sex_n", "site", "res_id", "area", "pat_n", "gray_m", "vignette"]
    Xtr = pd.DataFrame({f"m_{t}": rk(C.load_pred(t)[0]) for t in tags})
    Xte = pd.DataFrame({f"m_{t}": rk(C.load_pred(t)[1]) for t in tags})
    for c in ctx:
        Xtr[c] = tr[c].astype(float).fillna(tr[c].median()).values
        Xte[c] = te[c].astype(float).fillna(tr[c].median()).values

    mk = lambda: LGBMClassifier(  # noqa: E731
        n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
        random_state=C.SEED, n_jobs=18, verbose=-1)

    oof = np.zeros(len(f))
    pred = np.zeros(len(te))
    for k in range(C.N_FOLDS):
        m = mk()
        idx = folds != k
        m.fit(Xtr[idx], y[idx])
        oof[~idx] = m.predict_proba(Xtr[~idx])[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / C.N_FOLDS
    auc = C.oof_auc(y, oof)
    gauc = float(np.mean([C.oof_auc(y[gfolds == k], oof[gfolds == k]) for k in range(C.N_FOLDS)]))
    log.info("stack AUC %.5f  (patient-grouped %.5f)", auc, gauc)
    C.save_pred("stack", oof, pred)
    C.log_exp(model="LGB level-2 stack (member ranks + metadata context)", metric="roc_auc",
              direction="maximize", score=auc, features=list(Xtr.columns),
              cv=dict(scheme="StratifiedKFold(5) by image, nested (members are OOF)",
                      per_fold=C.per_fold_auc(y, oof, folds), patient_grouped_auc=gauc),
              notes=f"tag=stack; members={tags}; shallow+regularised (leaves 7, mcs 80) because "
                    f"level-2 data is small in effective positives (513)")


if __name__ == "__main__":
    main()
