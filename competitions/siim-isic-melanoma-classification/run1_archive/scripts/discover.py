"""Tier-2 discovery: per-backbone LR mini-sweep on fold 0.

vision_experience.md is explicit that a single fixed-LR comparison ranks LR tolerance, not
backbone quality, and that backbone ranking is domain-specific -- so re-discover it here.
"""
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
    ap.add_argument("--backbones", default="resnet18,convnext_atto,efficientnet_b0,vit_tiny_patch16_224")
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="discovery.json")
    a = ap.parse_args()

    C.set_seed()
    tr, _ = C.load_meta()
    y = tr.target.to_numpy()
    folds = tr.fold.to_numpy()
    tr_idx = np.where(folds != 0)[0]
    va_idx = np.where(folds == 0)[0]
    imgs = V.gpu_images("train")
    print(f"images on GPU: {tuple(imgs.shape)}  "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)

    res = []
    for bb in a.backbones.split(","):
        for lr in [float(v) for v in a.lrs.split(",")]:
            t0 = time.time()
            try:
                _, p, sec = V.train_fold(bb, imgs, y, tr_idx, va_idx, a.epochs, lr, bs=a.bs)
                s = C.auc(y[va_idx], p)
            except Exception as e:  # noqa: BLE001 - a backbone/LR failing must not kill the sweep
                s, sec = float("nan"), time.time() - t0
                print(f"  !! {bb} lr={lr:g} failed: {type(e).__name__}: {e}", flush=True)
            res.append({"backbone": bb, "lr": lr, "auc": s, "sec": round(sec, 1)})
            print(f"{bb:28s} lr={lr:<7g} fold0 AUC {s:.5f}  ({sec:.0f}s)", flush=True)
            torch.cuda.empty_cache()
            (C.COMP / a.out).write_text(json.dumps(res, indent=2))

    print("\n--- best LR per backbone ---")
    for bb in a.backbones.split(","):
        sub = [r for r in res if r["backbone"] == bb and r["auc"] == r["auc"]]
        if sub:
            b = max(sub, key=lambda r: r["auc"])
            print(f"{bb:28s} lr={b['lr']:<7g} AUC {b['auc']:.5f}")


if __name__ == "__main__":
    main()
