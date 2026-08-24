"""Stage 3.2a — cheap discovery: per-backbone LR mini-sweep, fold 0, 2 epochs, 256px.

knowledge/vision_experience.md is explicit that (a) backbone ranking is domain-specific and
must be rediscovered per competition, and (b) a single fixed LR ranks LR tolerance rather
than backbone quality, so every backbone gets its own LR grid before any of them is judged.
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd
import torch

import common as C
import vision as V

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SIZE = 256
EPOCHS = 2
GRID = [
    ("resnet18", 128, {}),
    ("resnet50", 96, {}),
    ("efficientnet_b0", 96, {}),
    ("efficientnet_b3", 64, {}),
    ("convnext_tiny", 64, {}),
    ("vit_small_patch16_224", 96, {"img_size": SIZE}),
]
LRS = [1e-4, 3e-4, 1e-3]


def main() -> None:
    f = pd.read_csv(C.CACHE / "r2_folds.csv")
    y = f.target.values
    tr_idx = f.index[f.fold != 0].values
    va_idx = f.index[f.fold == 0].values
    imgs = V.gpu_images("train", SIZE)
    results = []
    for backbone, bs, kw in GRID:
        for lr in LRS:
            t0 = time.time()
            try:
                p, model = V.train_fold(backbone, imgs, y, tr_idx, va_idx, lr=lr,
                                        epochs=EPOCHS, bs=bs, seed=C.SEED,
                                        model_kwargs=kw)
                auc = C.oof_auc(y[va_idx], p)
                del model
            except torch.cuda.OutOfMemoryError:
                auc = float("nan")
                log.warning("%s lr=%g OOM at bs=%d", backbone, lr, bs)
            torch.cuda.empty_cache()
            results.append(dict(backbone=backbone, lr=lr, bs=bs, auc=auc,
                                sec=round(time.time() - t0, 1)))
            log.info("%-24s lr=%-7g AUC %.4f  %.0fs", backbone, lr, auc, time.time() - t0)
            C.write_json("discovery.json", results)

    df = pd.DataFrame(results)
    best = df.loc[df.groupby("backbone").auc.idxmax()].sort_values("auc", ascending=False)
    log.info("best LR per backbone:\n%s", best.to_string(index=False))
    C.log_exp(model="backbone x LR discovery sweep (fold 0, 2 epochs, 256px)",
              metric="roc_auc", direction="maximize", score=float(best.auc.max()),
              cv=dict(scheme="fold 0 of StratifiedKFold(5) by image, single fold"),
              notes="; ".join(f"{r.backbone}@{r.lr:g}={r.auc:.4f}" for r in df.itertuples())
                    + " | per-backbone LR sweep is mandatory (vision_experience.md)")


if __name__ == "__main__":
    main()
