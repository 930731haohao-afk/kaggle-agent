"""V3 full fine-tune: one backbone, 5 folds, saves real OOF + test predictions."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import common as C
import vision as V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--aug", default="medium")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--full", type=int, default=None)
    ap.add_argument("--crop", type=int, default=None)
    a = ap.parse_args()
    if a.full:
        V.FULL = a.full
    if a.crop:
        V.CROP = a.crop
    tag = a.tag or f"{a.backbone}_{a.lr:g}_e{a.epochs}_{V.CROP}px"

    C.set_seed(a.seed)
    C.ART.mkdir(exist_ok=True)
    tr, te = C.load_meta()
    y = tr.target.to_numpy()
    folds = tr.fold.to_numpy()
    imgs = V.gpu_images("train")
    imgs_te = V.gpu_images("test")
    te_all = np.arange(len(te))

    oof = np.full(len(tr), np.nan)
    pred = np.zeros(len(te))
    want = [int(f) for f in a.folds.split(",")]
    t00 = time.time()
    for k in want:
        tr_idx = np.where(folds != k)[0]
        va_idx = np.where(folds == k)[0]
        model, p_va, sec = V.train_fold(a.backbone, imgs, y, tr_idx, va_idx, a.epochs, a.lr,
                                        bs=a.bs, aug=a.aug, seed=a.seed + k)
        oof[va_idx] = p_va
        pred += V.predict(model, imgs_te, te_all) / len(want)
        print(f"[{tag}] fold {k}: AUC {C.auc(y[va_idx], p_va):.5f}  ({sec:.0f}s train)", flush=True)
        del model
        torch.cuda.empty_cache()

    done = ~np.isnan(oof)
    score = C.auc(y[done], oof[done])
    per_fold = [C.auc(y[folds == k], oof[folds == k]) for k in want]
    np.save(C.ART / f"oof_{tag}.npy", oof)
    np.save(C.ART / f"pred_{tag}.npy", pred)
    wall = time.time() - t00
    print(f"[{tag}] OOF AUC {score:.5f}  folds {[round(v, 5) for v in per_fold]}  {wall:.0f}s")

    C.log_experiment(
        model=a.backbone, score=score, fold_scores=per_fold,
        features=[f"raw pixels {V.CROP}px crop from {V.FULL}px cache",
                  f"aug={a.aug} (dihedral + brightness/contrast)", "TTA=4 dihedral"],
        params={"lr": a.lr, "epochs": a.epochs, "bs": a.bs, "seed": a.seed,
                "optimizer": "AdamW wd=1e-4", "sched": "cosine + 10% warmup",
                "loss": "BCEWithLogits (no pos_weight)", "amp": "bf16"},
        notes=f"tag={tag}; wall {wall:.0f}s",
    )
    (C.COMP / f"res_{tag}.json").write_text(json.dumps(
        {"tag": tag, "score": score, "per_fold": per_fold, "wall_s": wall}, indent=2))


if __name__ == "__main__":
    main()
