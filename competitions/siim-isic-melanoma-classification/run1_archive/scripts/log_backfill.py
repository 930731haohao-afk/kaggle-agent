"""Backfill experiments.json for runs whose artifacts exist (metadata baselines + effnet V3)."""
from __future__ import annotations

import json

import numpy as np

import common as C


def main() -> None:
    tr, te = C.load_meta()
    y = tr.target.to_numpy()
    folds = tr.fold.to_numpy()
    Xtr, _ = C.build_tabular(tr, te)
    feat_names = list(Xtr.columns)

    meta_res = json.loads((C.COMP / "meta_results.json").read_text())
    abl = json.loads((C.COMP / "ablation_meta.json").read_text())
    for r in meta_res:
        feats = feat_names + (["pat_te (fold-safe patient target encoding, prior_w=5)"]
                              if r["name"].endswith("_te") else [])
        C.log_experiment(
            model=("LightGBM" if r["name"].startswith("lgb") else "CatBoost") + " (metadata only, no pixels)",
            score=r["score"], fold_scores=r["per_fold"], features=feats,
            params={"n_estimators": 600, "lr": 0.03, "num_leaves": 15, "max_depth": 5,
                    "imbalance_weighting": "none (ROC-AUC prior, experience.md)"},
            notes=(f"tag={r['name']}; {r['sec']:.0f}s. Feature-group ablation (LGB, same folds): "
                   f"only_resolution {abl['only_resolution']:.5f}, only_colour {abl['only_colour']:.5f}, "
                   f"only_uglyduck {abl['only_uglyduck']:.5f}, only_demog {abl['only_demog']:.5f}, "
                   f"only_patient_n {abl['only_patient_n']:.5f}, only_site {abl['only_site']:.5f}. "
                   "Original JPEG resolution is a data-source proxy: 6000x4000 -> 0.18% positive, "
                   "4032x3024 -> 17.8%; identical distribution in train and test."))

    tag = "efficientnet_b0_0.001_e8"
    oof = np.load(C.ART / f"oof_{tag}.npy")
    per_fold = [C.auc(y[folds == k], oof[folds == k]) for k in range(C.N_FOLDS)]
    C.log_experiment(
        model="efficientnet_b0", score=C.auc(y, oof), fold_scores=per_fold,
        features=["raw pixels 224px crop from 256px cache",
                  "aug=medium (dihedral + brightness/contrast)", "TTA=4 dihedral"],
        params={"lr": 1e-3, "epochs": 8, "bs": 64, "seed": 42, "optimizer": "AdamW wd=1e-4",
                "sched": "cosine + 10% warmup", "loss": "BCEWithLogits (no pos_weight)", "amp": "bf16"},
        notes=f"tag={tag}; wall 1804s; backfilled (logger signature bug on first run)")

    disc = json.loads((C.COMP / "discovery.json").read_text())
    C.log_experiment(
        model="backbone x LR discovery sweep (fold 0, 2 epochs)",
        score=max(d["auc"] for d in disc), fold_scores=None,
        features=["raw pixels 224px", "aug=medium"],
        params={"grid": "4 backbones x lr {1e-4,3e-4,1e-3}"},
        cv_scheme="fold 0 of the 5-fold split only (discovery proxy, NOT comparable to OOF scores)",
        notes="results: " + "; ".join(f"{d['backbone']}@{d['lr']:g}={d['auc']:.4f}" for d in disc))
    print("backfilled", len(meta_res) + 2, "entries")


if __name__ == "__main__":
    main()
