"""dogs-vs-cats driver for the shared vision pipeline (run #3).

Binary (cat=0, dog=1), metric=LOGLOSS, submission = id,label where label = P(dog).
Variable-size JPGs -> preloaded (resized) to a uniform 160x160 uint8 array, cached.
train/ = cat.N.jpg / dog.N.jpg (label from filename); test1/ = N.jpg (id = N).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision.transforms import v2 as T

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP.parent / "_shared"))
import vp  # noqa: E402

LOAD = 160  # preload/resize size (uniform)

CONFIG = dict(
    comp=str(COMP), img_size=LOAD, n_classes=2, classes=None, metric="logloss",
    output="proba", proba_class=1,   # submit P(dog)
    backbones=["resnet18", "efficientnet_b0", "mobilenetv3_large_100", "convnext_atto", "vit_tiny_patch16_224"],
    lr_grid=[1e-4, 3e-4, 1e-3],
    augs={"none": None,
          "light": T.RandomHorizontalFlip(0.5),   # cats/dogs: h-flip is fine
          "medium": T.Compose([T.RandomHorizontalFlip(0.5),
                               T.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1))])},
    subsample=6000, disc_folds=3, cv_folds=5, epochs=3, top_k=3,
    id_col="id", label_col="label", hflip_ok=True, batch=64, infer_batch=512,
)


def _preload(names, cache):
    p = COMP / "data" / cache
    if p.exists():
        return np.load(p)
    from PIL import Image
    arr = np.zeros((len(names), LOAD, LOAD, 3), dtype=np.uint8)
    for k, fn in enumerate(names):
        with Image.open(COMP / "data" / fn) as im:
            arr[k] = np.asarray(im.convert("RGB").resize((LOAD, LOAD), Image.BILINEAR), dtype=np.uint8)
        if k % 5000 == 0:
            vp.log(f"  preload {cache} {k}/{len(names)}")
    np.save(p, arr)
    return arr


def load_train():
    files = sorted((COMP / "data/train").glob("*.jpg"), key=lambda p: (p.name.split(".")[0], int(p.name.split(".")[1])))
    names = [f"train/{f.name}" for f in files]
    y = np.array([1 if f.name.startswith("dog") else 0 for f in files], dtype=np.int64)
    X = _preload(names, "train_arr.npy")
    return X, y


def load_test():
    ids = np.arange(1, 12501)
    X = _preload([f"test1/{i}.jpg" for i in ids], "test_arr.npy")
    return X, ids


def main():
    X, y = load_train()
    vp.log(f"dogs-vs-cats train={X.shape} pos_rate={y.mean():.3f} metric=logloss img={LOAD}")
    promote = vp.discover(X, y, CONFIG, str(COMP))
    X_test, ids = load_test()
    vp.log(f"test={X_test.shape}")
    vp.finetune(promote, X, y, X_test, CONFIG, str(COMP))
    out, score, chosen = vp.ensemble_submit(promote, y, ids, CONFIG, str(COMP))
    vp.log(f"dogs-vs-cats DONE: logloss={score:.5f} chosen={chosen} sub={out}")


if __name__ == "__main__":
    main()
