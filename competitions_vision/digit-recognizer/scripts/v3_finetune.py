"""V3 — full fine-tune of the promoted survivors (kaggle-vision-agent, digit-recognizer #1).

Survivors (V2 evidence, tier-2a lr sweep + tier-2b aug sweep, 8k subsample, 3-fold):
  convnext_atto  lr=1e-4  aug=medium  (0.9828)
  resnet18       lr=1e-3  aug=medium  (0.9809)
  vit_tiny_...   lr=1e-4  aug=light   (0.9581, transformer-family diversity member)

Protocol per config (references/03_finetune.md):
  full 42k train, StratifiedKFold(5, shuffle, seed=42), full unfreeze, 3 epochs with
  cosine LR decay, fp32 (matching the measured determinism gate), determinism preamble,
  eval WITHOUT augmentation. Outputs the ensemble contract per config:
    data/oof_<cfg>.npz : oof  (42000 x 10 softmax probs)
                         test (28000 x 10 softmax probs, mean over folds)
                         fold_scores, wall_s
Crash-safe: per-fold checkpoints in data/v3_ckpt_<cfg>_fold<k>.npz — a restart skips
completed folds. Each config logs to experiments.json when its CV completes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

COMP = Path(__file__).resolve().parent.parent
ROOT = COMP.parent.parent
sys.path.insert(0, str(ROOT / ".claude/skills/kaggle-vision-agent/assets"))
from torch_determinism import setup_determinism, seeded_loader_kwargs  # noqa: E402

setup_determinism(seed=42)

import timm  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ROOT / ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

SEED, N_FOLDS, IMG, BATCH, EPOCHS = 42, 5, 224, 64, 3
DEV = "cuda"

AUGS = {
    "none": None,
    "light": T.RandomAffine(degrees=10, translate=(0.07, 0.07), fill=0.0),
    "medium": T.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.9, 1.1), fill=0.0),
}
CONFIGS = [
    dict(id="convnext_atto_med", backbone="convnext_atto", lr=1e-4, aug="medium"),
    dict(id="resnet18_med", backbone="resnet18", lr=1e-3, aug="medium"),
    dict(id="vit_tiny_light", backbone="vit_tiny_patch16_224", lr=1e-4, aug="light"),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def to_batch(x28: torch.Tensor, aug=None) -> torch.Tensor:
    t = x28.unsqueeze(1) / 255.0
    if aug is not None:
        t = aug(t)
    t = F.interpolate(t, size=IMG, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std


@torch.no_grad()
def predict_probs(model, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        xb = to_batch(torch.from_numpy(X[i:i + bs])).to(DEV)
        out.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def run_config(cfg, X, y, X_test, folds):
    ckpt = lambda k: COMP / f"data/v3_ckpt_{cfg['id']}_fold{k}.npz"  # noqa: E731
    oof = np.zeros((len(X), 10), dtype=np.float32)
    test_sum = np.zeros((len(X_test), 10), dtype=np.float32)
    fold_scores, t0 = [], time.time()
    aug = AUGS[cfg["aug"]]

    for k, (tr, va) in enumerate(folds):
        if ckpt(k).exists():
            z = np.load(ckpt(k))
            oof[va], test_sum[:] = z["oof_va"], test_sum + z["test"]
            fold_scores.append(float(z["score"]))
            log(f"  {cfg['id']} fold{k}: resumed (acc={fold_scores[-1]:.4f})")
            continue
        torch.manual_seed(SEED + k)
        model = timm.create_model(cfg["backbone"], pretrained=True, num_classes=10).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EPOCHS * (len(tr) // BATCH + 1))
        dl = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                        batch_size=BATCH, shuffle=True, **seeded_loader_kwargs(seed=SEED + k))
        model.train()
        for ep in range(EPOCHS):
            for xb28, yb in dl:
                opt.zero_grad()
                F.cross_entropy(model(to_batch(xb28, aug).to(DEV)), yb.to(DEV)).backward()
                opt.step()
                sched.step()
            log(f"  {cfg['id']} fold{k} epoch{ep} done ({time.time()-t0:.0f}s elapsed)")
        model.eval()
        oof[va] = predict_probs(model, X[va])
        test_k = predict_probs(model, X_test)
        test_sum += test_k
        score = float((oof[va].argmax(1) == y[va]).mean())
        fold_scores.append(score)
        np.savez_compressed(ckpt(k), oof_va=oof[va], test=test_k, score=score)
        log(f"  {cfg['id']} fold{k}: acc={score:.4f}")
        del model
        torch.cuda.empty_cache()

    test = test_sum / N_FOLDS
    cv_acc = float((oof.argmax(1) == y).mean())
    wall = round(time.time() - t0, 1)
    np.savez_compressed(COMP / f"data/oof_{cfg['id']}.npz",
                        oof=oof, test=test, fold_scores=np.array(fold_scores), wall_s=wall)
    experiment_log.log_experiment_v2(
        str(COMP), model=f"{cfg['backbone']} full FT (lr={cfg['lr']:g}, aug={cfg['aug']}, "
                         f"{EPOCHS}ep, cosine)",
        metric="accuracy", direction="maximize", score=cv_acc,
        cv=dict(scheme="StratifiedKFold", n_splits=N_FOLDS, seed=SEED),
        features=[f"28px affine aug ({cfg['aug']}) -> 224 bilinear, ImageNet norm"],
        notes=(f"V3 full fine-tune; fold accs={[round(s,4) for s in fold_scores]}; "
               f"wall={wall}s; OOF cached data/oof_{cfg['id']}.npz; fp32; determinism on"))
    log(f"CONFIG DONE {cfg['id']}: OOF acc={cv_acc:.4f} (folds {[round(s,4) for s in fold_scores]}, {wall}s)")
    return cv_acc


def main():
    t0 = time.time()
    df = pd.read_csv(COMP / "data/train.csv")
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32).reshape(-1, 28, 28)
    X_test = pd.read_csv(COMP / "data/test.csv").to_numpy(dtype=np.float32).reshape(-1, 28, 28)
    log(f"train {X.shape}, test {X_test.shape}")
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    results = {}
    for cfg in CONFIGS:
        log(f"=== {cfg['id']} (lr={cfg['lr']:g}, aug={cfg['aug']}) ===")
        results[cfg["id"]] = run_config(cfg, X, y, X_test, folds)
    log("V3 SUMMARY: " + json.dumps(results))
    log(f"total {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
