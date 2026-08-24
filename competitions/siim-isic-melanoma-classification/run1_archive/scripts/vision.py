"""GPU-resident image training for melanoma classification.

No torch DataLoader anywhere: the whole uint8 cache lives on the GPU and augmentation runs as
batched tensor ops (vision_experience.md: a multi-worker DataLoader + CUDA fork deadlocked a
previous run for ~2 h).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

import common as C

DEV = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
CROP = int(os.environ.get('IMG_CROP', 224))
FULL = int(os.environ.get('IMG_FULL', 256))

_CACHE: dict[str, torch.Tensor] = {}


def gpu_images(split: str) -> torch.Tensor:
    """(N, 256, 256, 3) uint8 tensor resident on the GPU."""
    if split not in _CACHE:
        a = np.load(C.CACHE / f"{split}_{FULL}.npy", mmap_mode="r")
        t = torch.empty((a.shape[0], FULL, FULL, 3), dtype=torch.uint8, device=DEV)
        step = 4096
        for i in range(0, a.shape[0], step):
            t[i:i + step] = torch.from_numpy(np.ascontiguousarray(a[i:i + step])).to(DEV, non_blocking=True)
        _CACHE[split] = t
    return _CACHE[split]


def _to_float(batch_u8: torch.Tensor) -> torch.Tensor:
    x = batch_u8.permute(0, 3, 1, 2).float().div_(255.0)
    return x


def augment(x_u8: torch.Tensor, gen: torch.Generator, strength: str = "medium") -> torch.Tensor:
    """Per-sample random crop + dihedral flips + photometric jitter, all batched on GPU."""
    n = x_u8.shape[0]
    # per-sample random crop offsets (gather via unfold would be costly -> use narrow per unique off)
    off = torch.randint(0, FULL - CROP + 1, (2,), generator=gen, device=DEV)
    x = x_u8[:, off[0]:off[0] + CROP, off[1]:off[1] + CROP, :]
    x = _to_float(x)
    # per-sample dihedral group: hflip / vflip / transpose
    fh = torch.rand(n, generator=gen, device=DEV) < 0.5
    fv = torch.rand(n, generator=gen, device=DEV) < 0.5
    tp = torch.rand(n, generator=gen, device=DEV) < 0.5
    x = torch.where(fh.view(-1, 1, 1, 1), x.flip(3), x)
    x = torch.where(fv.view(-1, 1, 1, 1), x.flip(2), x)
    x = torch.where(tp.view(-1, 1, 1, 1), x.transpose(2, 3), x)
    if strength != "none":
        s = 0.2 if strength == "medium" else 0.1
        b = 1.0 + (torch.rand(n, 1, 1, 1, generator=gen, device=DEV) * 2 - 1) * s
        c = 1.0 + (torch.rand(n, 1, 1, 1, generator=gen, device=DEV) * 2 - 1) * s
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mu) * c + mu) * b
        x = x.clamp_(0, 1)
    return (x - MEAN) / STD


def eval_view(x_u8: torch.Tensor, view: int) -> torch.Tensor:
    o = (FULL - CROP) // 2
    x = _to_float(x_u8[:, o:o + CROP, o:o + CROP, :])
    if view & 1:
        x = x.flip(3)
    if view & 2:
        x = x.flip(2)
    return (x - MEAN) / STD


@torch.no_grad()
def predict(model: torch.nn.Module, imgs: torch.Tensor, idx: np.ndarray,
            bs: int = 128, tta: int = 4) -> np.ndarray:
    model.eval()
    out = np.zeros(len(idx), dtype=np.float64)
    ii = torch.as_tensor(idx, device=DEV)
    for v in range(tta):
        for i in range(0, len(idx), bs):
            b = ii[i:i + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(eval_view(imgs[b], v)).float().squeeze(1)
            out[i:i + bs] += torch.sigmoid(logit).double().cpu().numpy() / tta
    return out


def train_fold(backbone: str, imgs_tr: torch.Tensor, y: np.ndarray, tr_idx: np.ndarray,
               va_idx: np.ndarray, epochs: int, lr: float, bs: int = 64,
               aug: str = "medium", seed: int = C.SEED, drop_path: float = 0.1,
               log_every: int = 0) -> tuple[torch.nn.Module, np.ndarray, float]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    kw = {}
    if "convnext" in backbone or "vit" in backbone or "efficientnet" in backbone:
        kw["drop_path_rate"] = drop_path
    model = timm.create_model(backbone, pretrained=True, num_classes=1, **kw).to(DEV)
    model = model.to(memory_format=torch.channels_last)

    yt = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    ii = torch.as_tensor(tr_idx, device=DEV)
    n = len(tr_idx)
    steps = math.ceil(n / bs) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = ii[torch.randperm(n, generator=gen, device=DEV)]
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            x = augment(imgs_tr[b], gen, aug).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logit.float(), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if log_every and step % log_every == 0:
                print(f"    ep{ep} step {step}/{steps} loss {loss.item():.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
    p_va = predict(model, imgs_tr, va_idx)
    return model, p_va, time.time() - t0
