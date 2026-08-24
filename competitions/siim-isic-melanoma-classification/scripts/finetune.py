"""Stage 3.2b — full 5-fold fine-tune of one backbone; writes OOF + test predictions.

usage: finetune.py <backbone> <lr> <size> <epochs> <bs> [tag] [aug] [seed]
"""
from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import torch

import common as C
import vision as V

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    backbone, lr, size, epochs, bs = (sys.argv[1], float(sys.argv[2]), int(sys.argv[3]),
                                      int(sys.argv[4]), int(sys.argv[5]))
    tag = sys.argv[6] if len(sys.argv) > 6 else f"{backbone}_{size}px"
    aug = sys.argv[7] if len(sys.argv) > 7 else "medium"
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else C.SEED
    kw = {"img_size": size} if backbone.startswith("vit") else {}

    f = pd.read_csv(C.CACHE / "r2_folds.csv")
    y = f.target.values
    tr_imgs = V.gpu_images("train", size)
    te_imgs = V.gpu_images("test", size)
    n_te = te_imgs.shape[0]
    te_idx = np.arange(n_te)

    oof = np.zeros(len(f))
    pred = np.zeros(n_te)
    per_fold = []
    t0 = time.time()
    for k in range(C.N_FOLDS):
        tr_idx = f.index[f.fold != k].values
        va_idx = f.index[f.fold == k].values
        tk = time.time()
        p, model = V.train_fold(backbone, tr_imgs, y, tr_idx, va_idx, lr=lr, epochs=epochs,
                                bs=bs, seed=seed + k, aug=aug, model_kwargs=kw)
        oof[va_idx] = p
        pred += V.predict(model, te_imgs, te_idx, bs * 2, tta=True) / C.N_FOLDS
        per_fold.append(C.oof_auc(y[va_idx], p))
        log.info("fold %d AUC %.5f  %.0fs", k, per_fold[-1], time.time() - tk)
        del model
        torch.cuda.empty_cache()

    auc = C.oof_auc(y, oof)
    gfolds = pd.read_csv(C.CACHE / "r2_folds.csv").gfold.values
    gauc = float(np.mean([C.oof_auc(y[gfolds == k], oof[gfolds == k]) for k in range(C.N_FOLDS)]))
    wall = time.time() - t0
    C.save_pred(tag, oof, pred)
    log.info("%s OOF AUC %.5f  (patient-grouped %.5f)  per-fold %s  %.0fs",
             tag, auc, gauc, [round(a, 4) for a in per_fold], wall)
    C.log_exp(model=backbone, metric="roc_auc", direction="maximize", score=auc,
              cv=dict(scheme="StratifiedKFold(5) by image — regime-matched to mle-bench split",
                      per_fold=per_fold, patient_grouped_auc=gauc),
              notes=f"tag={tag}; {size}px, lr={lr:g}, {epochs}ep, bs={bs}, aug={aug}, seed={seed}, "
                    f"4x flip TTA, bf16, no class weighting; wall {wall:.0f}s")
    (C.COMP / f"res_{tag}.json").write_text(json.dumps(
        dict(tag=tag, backbone=backbone, lr=lr, size=size, epochs=epochs, bs=bs, aug=aug,
             seed=seed, auc=auc, gauc=gauc, per_fold=per_fold, sec=round(wall, 1)), indent=2))


if __name__ == "__main__":
    main()
