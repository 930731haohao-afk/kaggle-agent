"""Stage 3.1 — metadata + image-statistic models (no pixels through a CNN).

Establishes the non-vision baseline and ablates the three feature blocks so the vision
stage knows what the tabular side already explains. Per experience.md (ROC-AUC section),
NO class-imbalance weighting is used: AUC is a ranking metric and reweighting perturbs the
loss surface without improving ranking (validated on s3e3 / s4e1 / s6e2).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def cv_model(make, tr, te, feats, y, folds, gfolds, tag):
    """Fit `make()` once per fold; return OOF, mean test prediction and both AUCs."""
    oof = np.zeros(len(tr))
    pred = np.zeros(len(te))
    Xtr, Xte = tr[feats].astype(float), te[feats].astype(float)
    for k in range(C.N_FOLDS):
        m = make()
        idx = folds != k
        m.fit(Xtr[idx], y[idx])
        oof[~idx] = m.predict_proba(Xtr[~idx])[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / C.N_FOLDS
    auc = C.oof_auc(y, oof)
    # Honest generalisation check: same OOF vector scored within patient-disjoint folds.
    gauc = float(np.mean([C.oof_auc(y[gfolds == k], oof[gfolds == k]) for k in range(C.N_FOLDS)]))
    return oof, pred, auc, gauc


def main() -> None:
    tr_raw, te_raw = C.load_meta()
    tr_raw = C.make_folds(tr_raw)
    y = tr_raw.target.values
    folds, gfolds = tr_raw.fold.values, tr_raw.gfold.values
    tr_raw[["image_name", "patient_id", "target", "fold", "gfold"]].to_csv(
        C.CACHE / "r2_folds.csv", index=False)

    variants = {
        "full":     dict(use_res=True,  use_patient=True),
        "no_res":   dict(use_res=False, use_patient=True),
        "no_pat":   dict(use_res=True,  use_patient=False),
    }
    built = {k: C.build_features(tr_raw, te_raw, **v) for k, v in variants.items()}
    tr, te, feats = built["full"]
    for c in feats:  # a handful of stat columns can be NaN if a JPEG failed to parse
        med = tr[c].median()
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)
    log.info("features: %d  (%s ...)", len(feats), feats[:8])

    results = {}

    # --- naive + linear baselines -------------------------------------------------
    base_feats = ["sex_n", "age", "site"]
    mk_lr = lambda: make_pipeline(  # noqa: E731
        StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=C.SEED))
    for name, fs in [("lr_base3", base_feats), ("lr_allfeat", feats)]:
        t0 = time.time()
        o, p, auc, gauc = cv_model(mk_lr, tr.fillna(0), te.fillna(0), fs, y, folds, gfolds, name)
        C.save_pred(name, o, p)
        results[name] = dict(auc=auc, gauc=gauc, n_feat=len(fs), sec=round(time.time() - t0, 1))
        log.info("%-14s AUC %.5f  (patient-grouped %.5f)  %.0fs", name, auc, gauc, time.time() - t0)
        C.log_exp(model=f"LogisticRegression ({name})", metric="roc_auc", direction="maximize",
                  score=auc, features=fs,
                  cv=dict(scheme="StratifiedKFold(5) by image — regime-matched to mle-bench split",
                          per_fold=C.per_fold_auc(y, o, folds),
                          patient_grouped_auc=gauc),
                  notes=f"tag={name}; run2 metadata baseline, no class weighting (AUC is a ranking metric)")

    # --- GBDTs, full feature set --------------------------------------------------
    makers = {
        "lgb": lambda: LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
            random_state=C.SEED, n_jobs=18, verbose=-1),
        "xgb": lambda: XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=8,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist",
            random_state=C.SEED, n_jobs=18, eval_metric="auc"),
        "cat": lambda: CatBoostClassifier(
            iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
            random_seed=C.SEED, verbose=0, allow_writing_files=False, thread_count=18),
    }
    for mname, mk in makers.items():
        t0 = time.time()
        tag = f"{mname}_meta"
        o, p, auc, gauc = cv_model(mk, tr, te, feats, y, folds, gfolds, tag)
        C.save_pred(tag, o, p)
        results[tag] = dict(auc=auc, gauc=gauc, n_feat=len(feats), sec=round(time.time() - t0, 1))
        log.info("%-14s AUC %.5f  (patient-grouped %.5f)  %.0fs", tag, auc, gauc, time.time() - t0)
        C.log_exp(model=f"{mname.upper()} metadata+imagestats", metric="roc_auc",
                  direction="maximize", score=auc, features=feats,
                  cv=dict(scheme="StratifiedKFold(5) by image", per_fold=C.per_fold_auc(y, o, folds),
                          patient_grouped_auc=gauc),
                  notes=f"tag={tag}; no class weighting; features = metadata + resolution + "
                        f"prep-time colour/texture stats + patient-relative (ugly duckling)")

    # --- feature-block ablation (LGB, identical folds) -----------------------------
    abl = {}
    for vname in ["no_res", "no_pat"]:
        a_tr, a_te, a_feats = built[vname]
        for c in a_feats:
            m = a_tr[c].median()
            a_tr[c] = a_tr[c].fillna(m)
            a_te[c] = a_te[c].fillna(m)
        o, p, auc, gauc = cv_model(makers["lgb"], a_tr, a_te, a_feats, y, folds, gfolds, vname)
        abl[vname] = dict(auc=auc, gauc=gauc, n_feat=len(a_feats))
        log.info("ablation %-8s AUC %.5f (delta %+0.5f vs full)", vname, auc, auc - results["lgb_meta"]["auc"])
    # image-statistic block removed
    meta_only = ["sex_n", "age", "site", "site_na", "area", "aspect", "long_side", "res_id"]
    o, p, auc, gauc = cv_model(makers["lgb"], tr, te, meta_only, y, folds, gfolds, "no_imgstats")
    abl["no_imgstats"] = dict(auc=auc, gauc=gauc, n_feat=len(meta_only))
    log.info("ablation %-8s AUC %.5f (delta %+0.5f vs full)", "no_imgstats", auc,
             auc - results["lgb_meta"]["auc"])
    results["ablation"] = abl
    C.log_exp(model="LGB feature-block ablation", metric="roc_auc", direction="maximize",
              score=max(v["auc"] for v in abl.values()),
              notes="; ".join(f"{k}={v['auc']:.5f}" for k, v in abl.items())
                    + f"; full={results['lgb_meta']['auc']:.5f}")

    # --- feature importance --------------------------------------------------------
    m = makers["lgb"]()
    m.fit(tr[feats].astype(float), y)
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    log.info("top-20 LGB gain-split importance:\n%s", imp.head(20).to_string())
    results["top_features"] = imp.head(25).to_dict()

    C.write_json("meta_results.json", results)


if __name__ == "__main__":
    main()
