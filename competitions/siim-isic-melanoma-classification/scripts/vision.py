"""GPU-resident CNN/ViT fine-tuning for siim-isic (run2).

Engineering constraints taken from knowledge/vision_experience.md:
  * NO DataLoader / worker processes — a multi-worker DataLoader once deadlocked a run for
    ~2h against CUDA fork. Images live as one uint8 tensor on the GPU; batches are sliced,
    augmented and normalised on-device.
  * Per-backbone LR mini-sweep before judging a backbone (a single fixed LR measures LR
    tolerance, not backbone quality).

Domain-specific augmentation: dermoscopy is orientation-free, so h-flip, v-flip and 90-degree
rotations are all label-preserving (contrast with digits, where h-flip destroys 6/9). Colour
jitter is included because the contributing sites use different dermatoscope hardware.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

import common as C

log = logging.getLogger(__name__)

DEV = torch.device("cuda")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def gpu_images(split: str, size: int) -> torch.Tensor:
    """(N, H, W, 3) uint8 held on the GPU. ~5.7 GB at 256px, ~12.8 GB at 384px."""
    key = (split, size)
    if key not in _CACHE:
        arr = np.load(C.CACHE / f"r2_{split}_{size}.npy", mmap_mode="r")
        t = torch.empty(arr.shape, dtype=torch.uint8, device=DEV)
        step = 4096
        for i in range(0, len(arr), step):
            t[i:i + step] = torch.from_numpy(np.ascontiguousarray(arr[i:i + step])).to(DEV)
        _CACHE[key] = t
    return _CACHE[key]


def free_images() -> None:
    _CACHE.clear()
    torch.cuda.empty_cache()


def normalise(x_u8: torch.Tensor) -> torch.Tensor:
    x = x_u8.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def augment(x: torch.Tensor, g: torch.Generator, strength: str = "medium") -> torch.Tensor:
    """Per-sample geometric + photometric augmentation, all on GPU."""
    n = x.shape[0]
    # per-sample dihedral group: h-flip, v-flip, transpose -> all 8 orientations
    for dim in (3, 2):
        m = torch.rand(n, device=DEV, generator=g) < 0.5
        x = torch.where(m.view(-1, 1, 1, 1), x.flip(dim), x)
    m = torch.rand(n, device=DEV, generator=g) < 0.5
    x = torch.where(m.view(-1, 1, 1, 1), x.transpose(2, 3), x)

    if strength == "none":
        return x

    rot = 0.26 if strength == "medium" else 0.13          # +/- 15 or 7.5 degrees
    sc = 0.15 if strength == "medium" else 0.08
    th = (torch.rand(n, device=DEV, generator=g) * 2 - 1) * rot
    s = 1.0 + (torch.rand(n, device=DEV, generator=g) * 2 - 1) * sc
    tx = (torch.rand(n, 2, device=DEV, generator=g) * 2 - 1) * 0.06
    cos, sin = torch.cos(th) / s, torch.sin(th) / s
    theta = torch.zeros(n, 2, 3, device=DEV)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx[:, 0]
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, tx[:, 1]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    # photometric: brightness / contrast / per-channel gain (dermatoscope hardware varies)
    j = 0.25 if strength == "medium" else 0.12
    b = 1.0 + (torch.rand(n, 1, 1, 1, device=DEV, generator=g) * 2 - 1) * j
    c = 1.0 + (torch.rand(n, 1, 1, 1, device=DEV, generator=g) * 2 - 1) * j
    gain = 1.0 + (torch.rand(n, 3, 1, 1, device=DEV, generator=g) * 2 - 1) * (j * 0.4)
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mu) * c + mu) * b * gain
    return x


TTA_OPS = [lambda t: t, lambda t: t.flip(3), lambda t: t.flip(2),
           lambda t: t.flip(3).flip(2)]


@torch.no_grad()
def predict(model, imgs: torch.Tensor, idx: np.ndarray, bs: int, tta: bool) -> np.ndarray:
    model.eval()
    out = np.zeros(len(idx), dtype=np.float64)
    ops = TTA_OPS if tta else TTA_OPS[:1]
    for i in range(0, len(idx), bs):
        sl = torch.from_numpy(idx[i:i + bs]).to(DEV)
        x = normalise(imgs[sl])
        acc = 0.0
        for op in ops:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + torch.sigmoid(model(op(x)).float().squeeze(1))
        out[i:i + bs] = (acc / len(ops)).cpu().numpy()
    return out


def train_fold(backbone: str, imgs: torch.Tensor, y: np.ndarray, tr_idx: np.ndarray,
               va_idx: np.ndarray, *, lr: float, epochs: int, bs: int, seed: int,
               aug: str = "medium", drop_path: float = 0.1, tta: bool = True,
               model_kwargs: dict | None = None):
    """Fine-tune one fold; return validation predictions (TTA) and the fitted model."""
    set_seed(seed)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    model = timm.create_model(backbone, pretrained=True, num_classes=1,
                              drop_path_rate=drop_path, **(model_kwargs or {})
                              ).to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.from_numpy(y).float().to(DEV)
    steps = max(1, math.ceil(len(tr_idx) / bs)) * epochs
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr_idx)
        for i in range(0, len(order), bs):
            sl = torch.from_numpy(order[i:i + bs]).to(DEV)
            x = augment(normalise(imgs[sl]), g, aug).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # no pos_weight: AUC is a ranking metric (experience.md, ROC-AUC section)
                loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), yt[sl])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return predict(model, imgs, va_idx, bs * 2, tta), model
