"""Metadata model v3 = v1 base features + label-free extras (+ optional patient TE).

Ablation (same folds, LGB): v1 base 0.88225 -> +label-free extras 0.88715 (+0.0049);
adding a fold-safe TE on the resolution/source key instead COLLAPSED it to 0.86530 (-0.017),
so the source-group TE is dropped -- the tree already splits orig_w/orig_h natively and the TE
only adds LOO noise on the small resolution groups.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
from train_meta2 import extra_feats


def main() -> None:
    C.set_seed()
    tr, te = C.load_meta()
    y = tr.target.to_numpy()
    folds = tr.fold.to_numpy()
    X1, X1t = C.build_tabular(tr, te)
    both = pd.concat([tr, te], ignore_index=True)
    both["res"] = both.orig_w.astype(int).astype(str) + "x" + both.orig_h.astype(int).astype(str)
    e_tr, _ = extra_feats(tr, both)
    e_te, _ = extra_feats(te, both)
    Xtr = pd.concat([X1, e_tr], axis=1)
    Xte = pd.concat([X1t, e_te], axis=1)
    pte_tr, pte_val, pte_te = C.patient_te(tr, te)

    out = []
    for use_te in (False, True):
        for model in ("lgb", "cat", "xgb"):
            oof = np.zeros(len(tr))
            pred = np.zeros(len(te))
            for k in range(C.N_FOLDS):
                m = folds != k
                xa, xb, xc = Xtr[m].copy(), Xtr[~m].copy(), Xte.copy()
                if use_te:
                    xa["pat_te"] = pte_tr[m, k]
                    xb["pat_te"] = pte_val[~m]
                    xc["pat_te"] = pte_te[:, k]
                if model == "lgb":
                    import lightgbm as lgb
                    mdl = lgb.LGBMClassifier(
                        n_estimators=700, learning_rate=0.03, num_leaves=15, max_depth=5,
                        min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                        reg_alpha=1.0, reg_lambda=2.0, random_state=C.SEED, n_jobs=8,
                        deterministic=True, force_row_wise=True, verbose=-1).fit(xa, y[m])
                elif model == "cat":
                    from catboost import CatBoostClassifier
                    mdl = CatBoostClassifier(iterations=900, learning_rate=0.03, depth=5,
                                             l2_leaf_reg=8.0, random_seed=C.SEED, verbose=0,
                                             allow_writing_files=False, thread_count=8)
                    xa, xb, xc = xa.fillna(-999), xb.fillna(-999), xc.fillna(-999)
                    mdl.fit(xa, y[m])
                else:
                    import xgboost as xgb
                    mdl = xgb.XGBClassifier(
                        n_estimators=700, learning_rate=0.03, max_depth=5, min_child_weight=10,
                        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
                        random_state=C.SEED, n_jobs=8, tree_method="hist",
                        eval_metric="auc").fit(xa, y[m])
                oof[~m] = mdl.predict_proba(xb)[:, 1]
                pred += mdl.predict_proba(xc)[:, 1] / C.N_FOLDS
            tag = f"{model}_v3{'_te' if use_te else ''}"
            s = C.auc(y, oof)
            pf = [C.auc(y[folds == k], oof[folds == k]) for k in range(C.N_FOLDS)]
            np.save(C.ART / f"oof_{tag}.npy", oof)
            np.save(C.ART / f"pred_{tag}.npy", pred)
            print(f"{tag:12s} OOF AUC {s:.5f}   folds {[round(v, 4) for v in pf]}", flush=True)
            out.append({"tag": tag, "score": s, "per_fold": pf, "n_feats": Xtr.shape[1] + int(use_te)})
            C.log_experiment(
                model=f"{model.upper()} metadata v3{' + patient TE' if use_te else ''}",
                score=s, fold_scores=pf,
                features=list(Xtr.columns) + (["pat_te"] if use_te else []),
                params={"variant": "v3 (base + label-free patient/source extras)",
                        "imbalance_weighting": "none (ROC-AUC prior)"},
                notes=f"tag={tag}; resolution-group TE deliberately excluded (ablated at -0.017)")
    (C.COMP / "meta3_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
