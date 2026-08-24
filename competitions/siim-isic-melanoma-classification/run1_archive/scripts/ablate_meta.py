"""Leak diagnostic: which metadata feature groups produce the surprisingly high 0.88 AUC?"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import common as C

GROUPS = {
    "demog": ["sex", "age"],
    "site": [c for c in ["site_torso", "site_lower_extremity", "site_upper_extremity",
                         "site_head_neck", "site_palms_soles", "site_oral_genital", "site_na"]],
    "resolution": ["orig_w", "orig_h", "orig_px", "orig_ar"],
    "colour": ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"],
    "patient_n": ["pat_n_img", "pat_n_sites"],
    "uglyduck": None,  # filled below
}


def fit(X: pd.DataFrame, y, folds) -> float:
    oof = np.zeros(len(X))
    for k in range(C.N_FOLDS):
        m = folds != k
        mdl = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=5,
                                 min_child_samples=60, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
                                 random_state=C.SEED, n_jobs=10, deterministic=True,
                                 force_row_wise=True, verbose=-1)
        mdl.fit(X[m], y[m])
        oof[~m] = mdl.predict_proba(X[~m])[:, 1]
    return C.auc(y, oof)


def main() -> None:
    C.set_seed()
    tr, te = C.load_meta()
    X, _ = C.build_tabular(tr, te)
    y = tr.target.to_numpy()
    folds = tr.fold.to_numpy()
    GROUPS["uglyduck"] = [c for c in X.columns if c.startswith(("ud_", "udz_"))]
    out = {}
    for g, cols in GROUPS.items():
        out[f"only_{g}"] = fit(X[cols], y, folds)
        out[f"drop_{g}"] = fit(X[[c for c in X.columns if c not in cols]], y, folds)
        print(f"{g:12s} only={out['only_' + g]:.5f}  drop={out['drop_' + g]:.5f}", flush=True)
    out["all"] = fit(X, y, folds)
    print("all =", round(out["all"], 5))

    # importance on the full feature set
    mdl = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=5,
                             min_child_samples=60, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
                             random_state=C.SEED, n_jobs=10, deterministic=True,
                             force_row_wise=True, verbose=-1).fit(X, y)
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("\ntop-15 gain importance:\n", imp.head(15).round(0))

    # is image resolution a data-source proxy that correlates with the label?
    res = tr.assign(res=tr.orig_w.astype(int).astype(str) + "x" + tr.orig_h.astype(int).astype(str))
    vc = res.groupby("res")["target"].agg(["mean", "size"]).sort_values("size", ascending=False)
    print("\ntarget rate by original resolution (top 12 by count):\n", vc.head(12).round(4))
    print("n distinct resolutions:", vc.shape[0])
    (C.COMP / "ablation_meta.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
