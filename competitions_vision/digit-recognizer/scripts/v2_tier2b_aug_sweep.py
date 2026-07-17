"""V2 tier-2b — augmentation sweep (kaggle-vision-agent, digit-recognizer run #1).

Frozen probes cannot see augmentation (SKILL.md cost ladder), so augmentation is discovered
here: top backbones from tier-2a, each at ITS OWN best LR, x 3 augmentation policies.
Domain reasoning for digits: NO horizontal flip (6/9, 2/5 asymmetry); small affine
perturbations (rotate/translate/scale) match real handwriting variation.

Policies (applied to the 28x28 grayscale tensor BEFORE upscaling, train folds only):
  none   : identity
  light  : RandomAffine(rotate ±10°, translate ±7%)
  medium : RandomAffine(rotate ±15°, translate ±10%, scale 0.9-1.1)

Output: scripts/v2_tier2b_results.json + experiments.json entries.
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

SEED, N_SUB, N_FOLDS, IMG, BATCH, TOP_K = 42, 8000, 3, 224, 64, 3
DEV = "cuda"

AUGS = {
    "none": None,
    "light": T.RandomAffine(degrees=10, translate=(0.07, 0.07), fill=0.0),
    "medium": T.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.9, 1.1), fill=0.0),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_subsample():
    df = pd.read_csv(COMP / "data/train.csv")
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32).reshape(-1, 28, 28)
    rng = np.random.RandomState(SEED)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], N_SUB // 10, replace=False)
                          for c in range(10)])
    rng.shuffle(idx)
    return X[idx], y[idx]


def to_batch(x28: torch.Tensor, aug=None) -> torch.Tensor:
    """(N,28,28) float [0,255] -> normalized (N,3,224,224); aug applied at 28px, train only."""
    t = x28.unsqueeze(1) / 255.0
    if aug is not None:
        t = aug(t)
    t = F.interpolate(t, size=IMG, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std


def ft_cv_acc(name, lr, aug, X, y, folds):
    accs = []
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(SEED + k)
        model = timm.create_model(name, pretrained=True, num_classes=10).to(DEV).train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        dl = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                        batch_size=BATCH, shuffle=True, **seeded_loader_kwargs(seed=SEED + k))
        for xb28, yb in dl:
            opt.zero_grad()
            F.cross_entropy(model(to_batch(xb28, aug).to(DEV)), yb.to(DEV)).backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(va), 256):
                sel = va[i:i + 256]
                xb = to_batch(torch.from_numpy(X[sel])).to(DEV)   # NO aug at eval
                correct += int((model(xb).argmax(1).cpu().numpy() == y[sel]).sum())
        accs.append(correct / len(va))
        del model
        torch.cuda.empty_cache()
    return float(np.mean(accs))


def main():
    t0 = time.time()
    tier2a = json.loads((COMP / "scripts/v2_tier2a_results.json").read_text())
    top = [r["backbone"] for r in tier2a["ranking"][:TOP_K]]
    best_lr = {r["backbone"]: r["lr"] for r in tier2a["ranking"]}
    log(f"top-{TOP_K} from tier-2a: {top}")

    X, y = load_subsample()
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))

    resume_path = COMP / "scripts/v2_tier2b_partial.json"
    out = {"grid": json.loads(resume_path.read_text()) if resume_path.exists() else {},
           "best": {}}
    for name in top:
        out["grid"].setdefault(name, {})
        for aug_name, aug in AUGS.items():
            if aug_name in out["grid"][name]:
                continue
            t = time.time()
            # aug 'none' at best LR duplicates the tier-2a winner run — reuse its score
            if aug_name == "none":
                acc = tier2a["best"][name]["acc"]
            else:
                acc = ft_cv_acc(name, best_lr[name], aug, X, y, folds)
            out["grid"][name][aug_name] = acc
            resume_path.write_text(json.dumps(out["grid"], indent=1))
            log(f"AUG   {name:26s} {aug_name:6s} acc={acc:.4f}  ({time.time()-t:.0f}s)")
        best_aug = max(out["grid"][name], key=out["grid"][name].get)
        out["best"][name] = {"aug": best_aug, "lr": best_lr[name],
                             "acc": out["grid"][name][best_aug]}
        log(f"BEST  {name:26s} aug={best_aug} acc={out['best'][name]['acc']:.4f}")
        experiment_log.log_experiment_v2(
            str(COMP), model=f"{name} (tier-2b aug sweep, lr={best_lr[name]:g}, 1ep, {N_SUB} sub)",
            metric="accuracy", direction="maximize", score=out["best"][name]["acc"],
            cv=dict(scheme="StratifiedKFold", n_splits=N_FOLDS, seed=SEED, subsample=N_SUB),
            features=[f"28px affine aug ({best_aug}) -> 224 bilinear, ImageNet norm"],
            notes=f"aug grid={out['grid'][name]}; no h-flip (digit asymmetry); determinism on")

    out["total_wall_s"] = round(time.time() - t0, 1)
    (COMP / "scripts/v2_tier2b_results.json").write_text(json.dumps(out, indent=2))
    log("SUMMARY:")
    for name in top:
        log(f"  {name:26s} -> aug={out['best'][name]['aug']:6s} lr={best_lr[name]:g} "
            f"acc={out['best'][name]['acc']:.4f}")
    log(f"done in {out['total_wall_s']}s")


if __name__ == "__main__":
    main()
