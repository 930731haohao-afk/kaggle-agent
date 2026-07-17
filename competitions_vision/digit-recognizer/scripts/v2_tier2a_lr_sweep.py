"""V2 tier-2a — per-backbone LR mini-sweep (kaggle-vision-agent, digit-recognizer run #1).

Derisk evidence says: probes can't rank backbones, and a single-LR fine-tune is unfair
(LR sensitivity dominates). So backbone survival is decided HERE: each backbone gets its
own 3-point LR sweep (1 epoch, 8k stratified subsample, shared 3 folds), and backbones
are compared at their per-backbone best LR.

Output: scripts/v2_tier2a_results.json + experiments.json entries (one per backbone,
score = accuracy at best LR).
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

import timm  # noqa: E402  (after determinism preamble)

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ROOT / ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

SEED, N_SUB, N_FOLDS, IMG, BATCH = 42, 8000, 3, 224, 64
BACKBONES = ["resnet18", "efficientnet_b0", "mobilenetv3_large_100",
             "vit_tiny_patch16_224", "convnext_atto"]
LR_GRID = [1e-4, 3e-4, 1e-3]
DEV = "cuda"


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


def to_batch(x28):
    t = torch.from_numpy(x28).unsqueeze(1) / 255.0
    t = F.interpolate(t, size=IMG, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std


def ft_cv_acc(name, lr, X, y, folds):
    accs = []
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(SEED + k)
        model = timm.create_model(name, pretrained=True, num_classes=10).to(DEV).train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        dl = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                        batch_size=BATCH, shuffle=True, **seeded_loader_kwargs(seed=SEED + k))
        for xb28, yb in dl:
            opt.zero_grad()
            F.cross_entropy(model(to_batch(xb28.numpy()).to(DEV)), yb.to(DEV)).backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(va), 256):
                sel = va[i:i + 256]
                pred = model(to_batch(X[sel]).to(DEV)).argmax(1).cpu().numpy()
                correct += int((pred == y[sel]).sum())
        accs.append(correct / len(va))
        del model
        torch.cuda.empty_cache()
    return float(np.mean(accs))


def main():
    t0 = time.time()
    X, y = load_subsample()
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    # resume: reuse grids already completed in a previous partial run (crash-safe restart)
    resume_path = COMP / "scripts/v2_tier2a_partial.json"
    out = {"grid": {}, "best": {}}
    if resume_path.exists():
        out["grid"] = json.loads(resume_path.read_text())
        log(f"resumed grids for: {list(out['grid'])}")
    for name in BACKBONES:
        if name in out["grid"] and len(out["grid"][name]) == len(LR_GRID):
            best_lr = max(out["grid"][name], key=out["grid"][name].get)
            out["best"][name] = {"lr": float(best_lr), "acc": out["grid"][name][best_lr]}
            log(f"SKIP  {name:26s} (resumed) lr={best_lr} acc={out['best'][name]['acc']:.4f}")
            continue
        out["grid"][name] = {}
        for lr in LR_GRID:
            t = time.time()
            acc = ft_cv_acc(name, lr, X, y, folds)
            out["grid"][name][f"{lr:g}"] = acc
            resume_path.write_text(json.dumps(out["grid"], indent=1))  # crash-safe progress
            log(f"SWEEP {name:26s} lr={lr:g}  acc={acc:.4f}  ({time.time()-t:.0f}s)")
        best_lr = max(out["grid"][name], key=out["grid"][name].get)
        out["best"][name] = {"lr": float(best_lr), "acc": out["grid"][name][best_lr]}
        log(f"BEST  {name:26s} lr={best_lr}  acc={out['best'][name]['acc']:.4f}")
        experiment_log.log_experiment_v2(
            str(COMP), model=f"{name} (tier-2a LR sweep, 1ep, {N_SUB} sub)",
            metric="accuracy", direction="maximize",
            score=out["best"][name]["acc"],
            cv=dict(scheme="StratifiedKFold", n_splits=N_FOLDS, seed=SEED,
                    subsample=N_SUB),
            features=["raw pixels -> 224 bilinear, ImageNet norm, no aug"],
            notes=(f"per-backbone LR sweep {LR_GRID}, best lr={best_lr}; "
                   f"grid={out['grid'][name]}; determinism preamble on"))
    ranked = sorted(out["best"].items(), key=lambda kv: -kv[1]["acc"])
    out["ranking"] = [{"backbone": k, **v} for k, v in ranked]
    out["total_wall_s"] = round(time.time() - t0, 1)
    (COMP / "scripts/v2_tier2a_results.json").write_text(json.dumps(out, indent=2))
    log("RANKING (at per-backbone best LR):")
    for r in out["ranking"]:
        log(f"  {r['backbone']:26s} lr={r['lr']:g} acc={r['acc']:.4f}")
    log(f"done in {out['total_wall_s']}s")


if __name__ == "__main__":
    main()
