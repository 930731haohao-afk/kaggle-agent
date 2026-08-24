"""Timing/sanity probe: one backbone, one epoch, fold 0 — sizes the discovery sweep."""
from __future__ import annotations

import logging
import sys
import time

import pandas as pd

import common as C
import vision as V

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

if __name__ == "__main__":
    backbone = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    f = pd.read_csv(C.CACHE / "r2_folds.csv")
    y = f.target.values
    tr_idx = f.index[f.fold != 0].values
    va_idx = f.index[f.fold == 0].values
    t0 = time.time()
    imgs = V.gpu_images("train", size)
    log.info("images on GPU %s in %.0fs", tuple(imgs.shape), time.time() - t0)
    t1 = time.time()
    p, _ = V.train_fold(backbone, imgs, y, tr_idx, va_idx, lr=3e-4, epochs=1, bs=bs, seed=C.SEED)
    log.info("%s size=%d bs=%d  1ep fold0 AUC %.4f  %.0fs",
             backbone, size, bs, C.oof_auc(y[va_idx], p), time.time() - t1)
