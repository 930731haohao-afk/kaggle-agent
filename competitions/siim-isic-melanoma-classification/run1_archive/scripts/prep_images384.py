"""Decode every JPEG once into a fixed-size uint8 memmap cache (+ cheap per-image stats).

Rationale (vision_experience.md): no torch DataLoader at train time -- preload uint8 arrays and
do resize/aug/normalise on GPU. JPEG draft mode gives DCT-domain downscaling, which is what makes
decoding 33k images (up to 6000x4000) cheap.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COMP = Path(__file__).resolve().parents[1]
DATA = COMP / "data"
CACHE = COMP / "cache"
SIZE = 384


def load_one(args: tuple[str, str]) -> tuple[np.ndarray, tuple[int, int, float, float, float, float, float, float]]:
    path, _name = args
    with Image.open(path) as im:
        w0, h0 = im.size
        im.draft("RGB", (SIZE, SIZE))  # DCT-scaled decode, >=SIZE on both sides
        im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        a = np.asarray(im, dtype=np.uint8)
    f = a.reshape(-1, 3).astype(np.float32)
    stats = (w0, h0, *f.mean(0).tolist(), *f.std(0).tolist())
    return a, stats


def build(split: str, names: list[str], workers: int) -> None:
    CACHE.mkdir(exist_ok=True)
    out_img = CACHE / f"{split}_{SIZE}.npy"
    out_stats = CACHE / f"{split}_stats384.csv"
    if out_img.exists() and out_stats.exists():
        print(f"[{split}] cache exists, skip")
        return
    src = DATA / "jpeg" / split
    arr = np.lib.format.open_memmap(out_img, mode="w+", dtype=np.uint8,
                                    shape=(len(names), SIZE, SIZE, 3))
    rows = []
    tasks = [(str(src / f"{n}.jpg"), n) for n in names]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (a, st) in enumerate(ex.map(load_one, tasks, chunksize=32)):
            arr[i] = a
            rows.append(st)
            if (i + 1) % 4000 == 0:
                print(f"[{split}] {i + 1}/{len(names)}", flush=True)
    arr.flush()
    cols = ["orig_w", "orig_h", "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "image_name", names)
    df.to_csv(out_stats, index=False)
    print(f"[{split}] wrote {out_img} {arr.shape} and {out_stats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    build("train", tr.image_name.tolist(), a.workers)
    build("test", te.image_name.tolist(), a.workers)


if __name__ == "__main__":
    main()
