"""cifar-10 driver for the shared vision pipeline (run #2).

RGB 32x32, 10 classes, string labels. 300k test images -> use img_size=128 to keep the
huge test inference affordable. Discovery on an 8k stratified subsample.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision.transforms import v2 as T

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP.parent / "_shared"))
import vp  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLS2I = {c: i for i, c in enumerate(CLASSES)}

CONFIG = dict(
    comp=str(COMP), img_size=128, n_classes=10, classes=CLASSES, metric="accuracy",
    backbones=["resnet18", "efficientnet_b0", "mobilenetv3_large_100", "convnext_atto", "vit_tiny_patch16_224"],
    lr_grid=[1e-4, 3e-4, 1e-3],
    augs={"none": None,
          "light": T.RandomAffine(degrees=8, translate=(0.08, 0.08), fill=0.0),
          "medium": T.Compose([T.RandomHorizontalFlip(0.5),
                               T.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.0)])},
    subsample=8000, disc_folds=3, cv_folds=5, epochs=3, top_k=3,
    id_col="id", label_col="label", hflip_ok=True,   # natural images: h-flip is fine
    batch=64, infer_batch=512, workers=6,
)


def _preload(split, ids, cache):
    """Decode all PNGs for a split into one (N,32,32,3) uint8 array, cached to .npy."""
    p = COMP / "data" / cache
    if p.exists():
        return np.load(p)
    from PIL import Image
    arr = np.zeros((len(ids), 32, 32, 3), dtype=np.uint8)
    for k, i in enumerate(ids):
        with Image.open(COMP / "data" / split / f"{i}.png") as im:
            arr[k] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if k % 50000 == 0:
            vp.log(f"  preload {split} {k}/{len(ids)}")
    np.save(p, arr)
    return arr


def load_train():
    lab = pd.read_csv(COMP / "data/trainLabels.csv")
    X = _preload("train", lab["id"].to_numpy(), "train_arr.npy")
    y = np.array([CLS2I[c] for c in lab["label"]], dtype=np.int64)
    return X, y


def load_test():
    ids = np.arange(1, 300001)
    X = _preload("test", ids, "test_arr.npy")
    return X, ids


def main():
    X, y = load_train()
    vp.log(f"cifar-10 train={X.shape} classes={CONFIG['n_classes']} img={CONFIG['img_size']}")
    promote = vp.discover(X, y, CONFIG, str(COMP))
    X_test, ids = load_test()
    vp.log(f"test={len(X_test)}")
    vp.finetune(promote, X, y, X_test, CONFIG, str(COMP))
    out, oof, chosen = vp.ensemble_submit(promote, y, ids, CONFIG, str(COMP))
    vp.log(f"cifar-10 DONE: OOF={oof:.5f} chosen={chosen} sub={out}")


if __name__ == "__main__":
    main()
