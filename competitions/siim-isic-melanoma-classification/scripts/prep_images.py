"""Decode JPEGs once into uint8 arrays + per-image colour/texture stats.

Rationale (knowledge/vision_experience.md): no multi-worker DataLoader during training
(CUDA-fork deadlock hazard observed on cifar-10). Everything is preloaded as uint8 and
augmented on the GPU. Decoding is done once here with PIL's `draft()` DCT downscaling,
which is what makes 6000x4000 clinical photos affordable.

Usage: prep_images.py <size>
Writes cache/r2_{train,test}_{size}.npy  (N, size, size, 3) uint8, order = csv order
       cache/r2_{train,test}_stats.csv   (only for the first size run)
"""
from __future__ import annotations

import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
CACHE = COMP / "cache"
SIZE = 256


def load_one(args: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop to square, resize to SIZE, and return pixels + summary stats."""
    split, name = args
    with Image.open(DATA / "jpeg" / split / f"{name}.jpg") as im:
        # DCT-domain downscale: decodes 6000x4000 at 1/8 scale instead of full res.
        im.draft("RGB", (SIZE * 2, SIZE * 2))
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        arr = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)

    f = arr.astype(np.float32) / 255.0
    # Central disc = lesion region, outer ring = surrounding skin. The contrast between
    # them is the clinically meaningful quantity (a lesion darker/redder than its skin).
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.sqrt((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2) / (SIZE / 2)
    core, ring = r < 0.35, r > 0.80
    gray = f.mean(2)
    gy, gx = np.gradient(gray)
    stats = np.concatenate([
        f.reshape(-1, 3).mean(0), f.reshape(-1, 3).std(0),          # global colour
        f[core].mean(0), f[core].std(0),                             # lesion colour
        f[ring].mean(0),                                             # skin colour
        f[core].mean(0) - f[ring].mean(0),                           # lesion-skin contrast
        [gray.mean(), gray.std(), np.abs(gx).mean() + np.abs(gy).mean()],  # texture
        [np.percentile(gray, 5), np.percentile(gray, 95)],
        [gray[core].mean() - gray[ring].mean()],
        # dark-corner fraction: microscope vignetting is a scanner fingerprint
        [(gray[r > 0.95] < 0.15).mean()],
    ]).astype(np.float32)
    return arr, stats


STAT_COLS = (
    [f"m_{c}" for c in "rgb"] + [f"s_{c}" for c in "rgb"]
    + [f"core_m_{c}" for c in "rgb"] + [f"core_s_{c}" for c in "rgb"]
    + [f"ring_m_{c}" for c in "rgb"] + [f"contrast_{c}" for c in "rgb"]
    + ["gray_m", "gray_s", "grad_m", "gray_p5", "gray_p95", "gray_contrast", "vignette"]
)


def run(split: str, names: list[str], write_stats: bool) -> None:
    t0 = time.time()
    out = np.lib.format.open_memmap(
        CACHE / f"r2_{split}_{SIZE}.npy", mode="w+", dtype=np.uint8,
        shape=(len(names), SIZE, SIZE, 3))
    stats = np.zeros((len(names), len(STAT_COLS)), dtype=np.float32)
    with Pool(18) as p:
        for i, (arr, st) in enumerate(
                p.imap(load_one, [(split, n) for n in names], chunksize=32)):
            out[i], stats[i] = arr, st
            if i % 5000 == 0:
                log.info("%s %d/%d  %.0fs", split, i, len(names), time.time() - t0)
    out.flush()
    if write_stats:
        df = pd.DataFrame(stats, columns=STAT_COLS)
        df.insert(0, "image_name", names)
        df.to_csv(CACHE / f"r2_{split}_stats.csv", index=False)
    log.info("%s done in %.0fs", split, time.time() - t0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SIZE = int(sys.argv[1])
    CACHE.mkdir(exist_ok=True)
    for split, csv in [("train", "train.csv"), ("test", "test.csv")]:
        run(split, pd.read_csv(DATA / csv).image_name.tolist(), write_stats=(SIZE == 256))
