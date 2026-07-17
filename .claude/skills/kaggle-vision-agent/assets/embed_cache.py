"""Tier-0 utility for the kaggle-vision-agent skill: extract and cache frozen-backbone
embeddings (the discovery substrate). Import this rather than rewriting it per competition.

Supports the two common Kaggle image formats:
  * CSV pixels  : array of flattened grayscale images (digit-recognizer style)
  * image files : a list of file paths (folder-style comps)

Usage (from a competition script):
    from embed_cache import embed_pixels, embed_files

    emb = embed_pixels("resnet18", X28, img_size=224,
                       cache="competitions_vision/<name>/data/emb_resnet18_224.npy")
    # -> (N, D) float32; computed once, then loaded from cache on later calls

Design notes:
  * atomic cache writes (tmp + os.replace) — a crash never leaves a half-written cache;
  * grayscale -> 3-channel repeat + ImageNet normalization (what pretrained backbones expect);
  * batch-streamed, so full image sets never need to fit in RAM at once;
  * deterministic: no augmentation here BY DESIGN — frozen embeddings are for tier-1 probes,
    which are blind to augmentation anyway (see SKILL.md cost ladder).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _atomic_save(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _norm_batch(t: torch.Tensor, img_size: int) -> torch.Tensor:
    """(N,C,H,W) float in [0,1] -> resized, 3-channel, ImageNet-normalized."""
    if t.shape[-1] != img_size or t.shape[-2] != img_size:
        t = F.interpolate(t, size=img_size, mode="bilinear", align_corners=False)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


@torch.no_grad()
def _extract(model_name: str, batches, cache: str | Path | None) -> np.ndarray:
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(_device())
    outs = []
    for xb in batches:  # xb: normalized (N,3,H,W) on CPU
        outs.append(model(xb.to(_device())).float().cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    emb = np.concatenate(outs)
    if cache is not None:
        _atomic_save(cache, emb)
    return emb


def embed_pixels(model_name: str, x: np.ndarray, *, img_size: int = 224,
                 batch_size: int = 256, cache: str | Path | None = None) -> np.ndarray:
    """Embed a pixel array. `x` is (N,H,W) or (N,H,W,C), uint8 or float in [0,255]."""
    if cache is not None and Path(cache).exists():
        return np.load(cache)

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None]                                   # (N,1,H,W)
    else:
        x = x.transpose(0, 3, 1, 2)                      # (N,C,H,W)

    def batches():
        for i in range(0, len(x), batch_size):
            t = torch.from_numpy(x[i : i + batch_size]) / 255.0
            yield _norm_batch(t, img_size)

    return _extract(model_name, batches(), cache)


def embed_files(model_name: str, paths: list[str | Path], *, img_size: int = 224,
                batch_size: int = 128, cache: str | Path | None = None) -> np.ndarray:
    """Embed image files (streamed; order of `paths` is preserved in the output rows)."""
    if cache is not None and Path(cache).exists():
        return np.load(cache)

    from PIL import Image

    def batches():
        for i in range(0, len(paths), batch_size):
            imgs = []
            for p in paths[i : i + batch_size]:
                with Image.open(p) as im:
                    im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
                    imgs.append(np.asarray(im, dtype=np.float32))
            t = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)) / 255.0
            yield _norm_batch(t, img_size)

    return _extract(model_name, batches(), cache)
