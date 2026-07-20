"""tpu-getting-started (Petals to the Metal) driver — run #4.

104-class flower classification. Data = TFRecords (jpeg 192x192). train+val are labeled
(image + class); test is image + id (hex string). Metric = accuracy (leaderboard uses
macro-F1, but accuracy is a fine proxy for backbone selection; submission = class index).
Submission: id (hex), label (predicted class index 0-103).

TFRecords parsed with the pure-python `tfrecord` package (no TensorFlow). Images decoded
to a uniform (N,192,192,3) uint8 array, cached to .npy.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tfrecord.reader import tfrecord_loader
from torchvision.transforms import v2 as T

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP.parent / "_shared"))
import vp  # noqa: E402

SIZE = 192
TFDIR = COMP / "data/tfrecords-jpeg-192x192"

CONFIG = dict(
    comp=str(COMP), img_size=SIZE, n_classes=104, classes=None, metric="accuracy",
    backbones=["resnet18", "efficientnet_b0", "mobilenetv3_large_100", "convnext_atto", "vit_tiny_patch16_224"],
    lr_grid=[1e-4, 3e-4, 1e-3],
    augs={"none": None,
          "light": T.RandomHorizontalFlip(0.5),            # flowers: h-flip fine
          "medium": T.Compose([T.RandomHorizontalFlip(0.5),
                               T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))])},
    subsample=6000, disc_folds=3, cv_folds=5, epochs=4, top_k=3,
    id_col="id", label_col="label", hflip_ok=True, batch=64, infer_batch=384,
)


def _read_split(split, want_label):
    """Decode all tfrec shards in a split -> (uint8 array [N,192,192,3], labels_or_ids)."""
    imgs, meta = [], []
    for shard in sorted(TFDIR.glob(f"{split}/*.tfrec")):
        for rec in tfrecord_loader(str(shard), None):
            imgs.append(np.asarray(Image.open(io.BytesIO(bytes(rec["image"]))).convert("RGB"), dtype=np.uint8))
            meta.append(int(rec["class"][0]) if want_label else bytes(rec["id"]).decode())
    return np.stack(imgs), meta


def _cached(split, want_label, cache):
    p = COMP / "data" / cache
    pm = COMP / "data" / (cache.replace(".npy", "_meta.npy"))
    if p.exists() and pm.exists():
        return np.load(p), np.load(pm, allow_pickle=True)
    vp.log(f"  decoding {split} tfrecords...")
    X, meta = _read_split(split, want_label)
    np.save(p, X); np.save(pm, np.array(meta, dtype=object))
    vp.log(f"  {split}: {X.shape}")
    return X, np.array(meta, dtype=object)


def load_train():
    Xtr, ytr = _cached("train", True, "train_arr.npy")
    Xva, yva = _cached("val", True, "val_arr.npy")
    X = np.concatenate([Xtr, Xva])
    y = np.concatenate([ytr, yva]).astype(np.int64)
    return X, y


def load_test():
    X, ids = _cached("test", False, "test_arr.npy")
    return X, list(ids)


def main():
    X, y = load_train()
    vp.log(f"tpu-flowers train={X.shape} n_classes={CONFIG['n_classes']} img={SIZE}")
    promote = vp.discover(X, y, CONFIG, str(COMP))
    X_test, ids = load_test()
    vp.log(f"test={X_test.shape}")
    vp.finetune(promote, X, y, X_test, CONFIG, str(COMP))
    out, score, chosen = vp.ensemble_submit(promote, y, ids, CONFIG, str(COMP))
    vp.log(f"tpu-flowers DONE: acc={score:.5f} chosen={chosen} sub={out}")


if __name__ == "__main__":
    main()
