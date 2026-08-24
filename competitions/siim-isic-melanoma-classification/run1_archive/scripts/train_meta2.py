"""Metadata model v2: add a fold-safe target encoding of the image-source proxy (resolution).

EDA showed original JPEG resolution is a data-source proxy with a ~100x spread in positive rate
(6000x4000 -> 0.18%, 4032x3024 -> 17.8%) across 81 distinct resolutions, and the distribution is
identical in train and test. Raw orig_w/orig_h let the tree find only axis-aligned cuts; an
explicit fold-safe TE on the (w,h) pair gives it the group statistic directly.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C


def group_te(keys_tr: np.ndarray, y: np.ndarray, folds: np.ndarray, keys_te: np.ndarray,
             prior_w: float = 20.0):
    """Same fold-safe protocol as common.patient_te, for an arbitrary group key."""
    g = y.mean()
    te_tr = np.full((len(keys_tr), C.N_FOLDS), np.nan)
    te_val = np.full(len(keys_tr), np.nan)
    te_te = np.full((len(keys_te), C.N_FOLDS), np.nan)
    for k in range(C.N_FOLDS):
        m = folds != k
        s = pd.Series(y[m]).groupby(pd.Series(keys_tr[m])).agg(["sum", "size"])
        ti = np.where(m)[0]
        a = s.reindex(keys_tr[ti])
        te_tr[ti, k] = (a["sum"].to_numpy() - y[ti] + prior_w * g) / (a["size"].to_numpy() - 1.0 + prior_w)
        vi = np.where(~m)[0]
        b = s.reindex(keys_tr[vi])
        te_val[vi] = (np.nan_to_num(b["sum"].to_numpy()) + prior_w * g) / \
                     (np.nan_to_num(b["size"].to_numpy()) + prior_w)
        c = s.reindex(keys_te)
        te_te[:, k] = (np.nan_to_num(c["sum"].to_numpy()) + prior_w * g) / \
                      (np.nan_to_num(c["size"].to_numpy()) + prior_w)
    return te_tr, te_val, te_te


def extra_feats(df: pd.DataFrame, both: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.reset_index(drop=True).index)
    d = df.reset_index(drop=True)
    res = d.orig_w.astype(int).astype(str) + "x" + d.orig_h.astype(int).astype(str)
    cnt = both.res.value_counts()
    f["res_freq"] = res.map(cnt).astype("float32")          # label-free source-size prior
    # patient x source: how many distinct sources this patient's images come from
    pr = both.groupby("patient_id").res.nunique().rename("pat_n_res")
    f["pat_n_res"] = d.patient_id.map(pr).astype("float32")
    # patient-level colour spread (lesion heterogeneity within a patient)
    for c in ["r_mean", "b_mean", "r_std"]:
        mx = both.groupby("patient_id")[c].max()
        mn = both.groupby("patient_id")[c].min()
        f[f"pat_rng_{c}"] = (d.patient_id.map(mx) - d.patient_id.map(mn)).astype("float32")
        f[f"udrank_{c}"] = d.groupby("patient_id")[c].rank(pct=True).to_numpy("float32")
    f["age_x_site_hn"] = (d.age_approx.fillna(50) *
                          (d.anatom_site_general_challenge == "head/neck")).astype("float32")
    return f, res


def main() -> None:
    C.set_seed()
    tr, te = C.load_meta()
    y = tr.target.to_numpy()
    folds = tr.fold.to_numpy()
    Xtr, Xte = C.build_tabular(tr, te)
    both = pd.concat([tr, te], ignore_index=True)
    both["res"] = both.orig_w.astype(int).astype(str) + "x" + both.orig_h.astype(int).astype(str)
    e_tr, res_tr = extra_feats(tr, both)
    e_te, res_te = extra_feats(te, both)
    Xtr = pd.concat([Xtr, e_tr], axis=1)
    Xte = pd.concat([Xte, e_te], axis=1)

    pte_tr, pte_val, pte_te = C.patient_te(tr, te)
    rte_tr, rte_val, rte_te = group_te(res_tr.to_numpy(), y, folds, res_te.to_numpy())
    # site x sex x age-decade cell TE
    def cell(d):
        return (d.anatom_site_general_challenge.fillna("na").astype(str) + "|" +
                d.sex.fillna("na").astype(str) + "|" +
                (d.age_approx.fillna(-1) // 10).astype(int).astype(str)).to_numpy()
    cte_tr, cte_val, cte_te = group_te(cell(tr), y, folds, cell(te))

    variants = {
        "v2_res": [("res_te", rte_tr, rte_val, rte_te)],
        "v2_res_pat": [("res_te", rte_tr, rte_val, rte_te), ("pat_te", pte_tr, pte_val, pte_te)],
        "v2_all": [("res_te", rte_tr, rte_val, rte_te), ("pat_te", pte_tr, pte_val, pte_te),
                   ("cell_te", cte_tr, cte_val, cte_te)],
    }
    out = []
    for vname, tes in variants.items():
        for model in ("lgb", "cat"):
            oof = np.zeros(len(tr))
            pred = np.zeros(len(te))
            for k in range(C.N_FOLDS):
                m = folds != k
                xa, xb, xc = Xtr[m].copy(), Xtr[~m].copy(), Xte.copy()
                for nm, a_tr, a_val, a_te in tes:
                    xa[nm] = a_tr[m, k]
                    xb[nm] = a_val[~m]
                    xc[nm] = a_te[:, k]
                if model == "lgb":
                    import lightgbm as lgb
                    mdl = lgb.LGBMClassifier(
                        n_estimators=700, learning_rate=0.03, num_leaves=15, max_depth=5,
                        min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                        reg_alpha=1.0, reg_lambda=2.0, random_state=C.SEED, n_jobs=8,
                        deterministic=True, force_row_wise=True, verbose=-1).fit(xa, y[m])
                else:
                    from catboost import CatBoostClassifier
                    mdl = CatBoostClassifier(iterations=900, learning_rate=0.03, depth=5,
                                             l2_leaf_reg=8.0, random_seed=C.SEED, verbose=0,
                                             allow_writing_files=False, thread_count=8)
                    mdl.fit(xa.fillna(-999), y[m])
                    xb, xc = xb.fillna(-999), xc.fillna(-999)
                oof[~m] = mdl.predict_proba(xb)[:, 1]
                pred += mdl.predict_proba(xc)[:, 1] / C.N_FOLDS
            tag = f"{model}_{vname}"
            s = C.auc(y, oof)
            pf = [C.auc(y[folds == k], oof[folds == k]) for k in range(C.N_FOLDS)]
            np.save(C.ART / f"oof_{tag}.npy", oof)
            np.save(C.ART / f"pred_{tag}.npy", pred)
            print(f"{tag:16s} OOF AUC {s:.5f}   folds {[round(v, 4) for v in pf]}", flush=True)
            out.append({"tag": tag, "score": s, "per_fold": pf,
                        "n_feats": Xtr.shape[1] + len(tes)})
            C.log_experiment(
                model=("LightGBM" if model == "lgb" else "CatBoost") + f" metadata {vname}",
                score=s, fold_scores=pf,
                features=list(Xtr.columns) + [t[0] for t in tes],
                params={"variant": vname, "prior_w_res_te": 20.0, "prior_w_pat_te": 5.0},
                notes=f"tag={tag}; fold-safe group TE (LOO inside train folds, fold-out for valid, "
                      f"per-fold stats for test)")
    (C.COMP / "meta2_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
