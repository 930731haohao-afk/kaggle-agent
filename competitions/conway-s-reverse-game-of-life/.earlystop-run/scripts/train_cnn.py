"""Delta-conditioned CNN for Conway's Reverse Game of Life.

Input: stop board (1ch) + delta one-hot (5ch) + xy coords (2ch) = 8 channels, 20x20.
Output: 400 logits (start board). Loss: BCE. Metric: binary MAE @ 0.5.
CV: 5-fold StratifiedKFold on delta, seed 42.
Optional synthetic data (--synth N): boards generated with the official process
(random density 1-99%, 5 warmup steps, evolve delta, discard empties).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(10)
COMP = "competitions/conway-s-reverse-game-of-life"
DEV = torch.device("cuda")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Game of Life simulation (bounded 20x20, GPU) ----------------
_KERNEL = torch.ones(1, 1, 3, 3, device=DEV)
_KERNEL[0, 0, 1, 1] = 0


def gol_step(b: torch.Tensor) -> torch.Tensor:
    """One GoL step on (N,1,20,20) float tensor, zero boundary."""
    n = F.conv2d(b, _KERNEL, padding=1)
    return ((n == 3) | ((b == 1) & (n == 2))).float()


def gen_synthetic(n_target: int, seed: int, batch: int = 100_000):
    """Generate (start, stop, delta) with the official process."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    starts, stops, deltas = [], [], []
    got = 0
    while got < n_target:
        dens = torch.rand(batch, 1, 1, 1, device=DEV, generator=g) * 0.98 + 0.01
        b = (torch.rand(batch, 1, 20, 20, device=DEV, generator=g) < dens).float()
        for _ in range(5):  # warmup
            b = gol_step(b)
        start = b
        d = torch.randint(1, 6, (batch,), device=DEV, generator=g)
        for step in range(1, 6):
            b = gol_step(b)
            m = d == step
            if step == 1:
                stop = torch.empty_like(b)
            stop[m] = b[m]
        keep = (start.sum((1, 2, 3)) > 0) & (stop.sum((1, 2, 3)) > 0)
        starts.append(start[keep, 0].to(torch.uint8).cpu())
        stops.append(stop[keep, 0].to(torch.uint8).cpu())
        deltas.append(d[keep].to(torch.uint8).cpu())
        got += int(keep.sum())
    S = torch.cat(starts)[:n_target]
    P = torch.cat(stops)[:n_target]
    D = torch.cat(deltas)[:n_target]
    return S, P, D


# ---------------- Model ----------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)


class Net(nn.Module):
    def __init__(self, ch=128, blocks=8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(8, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        return self.head(self.body(self.stem(x))).squeeze(1)


_yy, _xx = torch.meshgrid(torch.linspace(-1, 1, 20), torch.linspace(-1, 1, 20), indexing="ij")
_COORDS = torch.stack([_yy, _xx]).to(DEV)  # (2,20,20)


def make_input(stop: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """stop (N,20,20) float, delta (N,) long 1..5 -> (N,8,20,20)."""
    n = stop.shape[0]
    oh = F.one_hot(delta - 1, 5).float().view(n, 5, 1, 1).expand(n, 5, 20, 20)
    co = _COORDS.unsqueeze(0).expand(n, 2, 20, 20)
    return torch.cat([stop.unsqueeze(1), oh, co], dim=1)


def d4(board: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Apply D4 transform to (N,20,20)."""
    if flip:
        board = torch.flip(board, dims=[2])
    return torch.rot90(board, k, dims=[1, 2])


def evaluate(model, stop, delta, target, bs=4096):
    model.eval()
    errs, probs = [], []
    with torch.no_grad():
        for i in range(0, stop.shape[0], bs):
            x = make_input(stop[i:i + bs].to(DEV).float(), delta[i:i + bs].to(DEV))
            p = torch.sigmoid(model(x))
            pred = (p > 0.5).float()
            errs.append((pred - target[i:i + bs].to(DEV).float()).abs().mean(dim=(1, 2)).cpu())
            probs.append(p.cpu())
    return torch.cat(errs), torch.cat(probs)


def train_fold(tr_stop, tr_delta, tr_start, va_stop, va_delta, va_start,
               epochs, bs, lr, seed, log_prefix=""):
    set_seed(seed)
    model = Net().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = tr_stop.shape[0]
    steps_per_epoch = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")
    best_mae = 1.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        t0 = time.time()
        tot_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            stop_b = tr_stop[idx].to(DEV).float()
            start_b = tr_start[idx].to(DEV).float()
            delta_b = tr_delta[idx].to(DEV)
            # D4 augmentation: same transform for input board and target
            k = int(torch.randint(0, 4, (1,)).item())
            fl = bool(torch.randint(0, 2, (1,)).item())
            stop_b, start_b = d4(stop_b, k, fl), d4(start_b, k, fl)
            x = make_input(stop_b, delta_b)
            with torch.amp.autocast("cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x), start_b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(idx)
        if ep >= epochs - 3 or (ep + 1) % 5 == 0:
            errs, _ = evaluate(model, va_stop, va_delta, va_start)
            mae = errs.mean().item()
            best_mae = min(best_mae, mae)
            print(f"{log_prefix}ep{ep+1}/{epochs} loss={tot_loss/n:.4f} "
                  f"val_MAE={mae:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="cnn")
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    train = pd.read_csv(f"{COMP}/data/train.csv")
    start_cols = [c for c in train.columns if c.startswith("start.")]
    stop_cols = [c for c in train.columns if c.startswith("stop.")]
    S = torch.from_numpy(train[start_cols].values.astype(np.uint8).reshape(-1, 20, 20))
    P = torch.from_numpy(train[stop_cols].values.astype(np.uint8).reshape(-1, 20, 20))
    D = torch.from_numpy(train["delta"].values.astype(np.int64))

    syn = None
    if args.synth > 0:
        t0 = time.time()
        ss, sp, sd = gen_synthetic(args.synth, seed=777)
        syn = (ss, sp, sd.to(torch.int64))
        print(f"synthetic: {ss.shape[0]} boards in {time.time()-t0:.0f}s", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(D)), train["delta"].values))

    oof_mae, oof_probs = {}, {}
    fold_models = []
    for f in folds:
        tr_idx, va_idx = splits[f]
        tr_stop, tr_start = P[tr_idx], S[tr_idx]
        tr_delta = D[tr_idx]
        if syn is not None:
            tr_stop = torch.cat([tr_stop, syn[1]])
            tr_start = torch.cat([tr_start, syn[0]])
            tr_delta = torch.cat([tr_delta, syn[2]])
        model = train_fold(tr_stop, tr_delta, tr_start,
                           P[va_idx], D[va_idx], S[va_idx],
                           args.epochs, args.bs, args.lr, args.seed + f,
                           log_prefix=f"[fold{f}] ")
        errs, probs = evaluate(model, P[va_idx], D[va_idx], S[va_idx])
        oof_mae[f] = errs.mean().item()
        oof_probs[f] = (va_idx, probs)
        fold_models.append(model)
        # per-delta breakdown
        dv = D[va_idx]
        by_d = {int(d): errs[dv == d].mean().item() for d in range(1, 6)}
        print(f"[fold{f}] final MAE={oof_mae[f]:.5f} by_delta={by_d}", flush=True)
        torch.save(model.state_dict(), f"{COMP}/scripts/{args.tag}_fold{f}.pt")

    maes = [oof_mae[f] for f in folds]
    print(f"CV folds={folds} mean_MAE={np.mean(maes):.5f} std={np.std(maes):.5f}", flush=True)

    # save OOF probs for possible ensembling
    if len(folds) == 5:
        oof = np.zeros((len(D), 400), dtype=np.float32)
        for f in folds:
            va_idx, probs = oof_probs[f]
            oof[va_idx] = probs.numpy().reshape(-1, 400)
        np.save(f"{COMP}/scripts/oof_{args.tag}.npy", oof)

    if args.predict_test:
        test = pd.read_csv(f"{COMP}/data/test.csv")
        tstop_cols = [c for c in test.columns if c.startswith("stop.")]
        TP = torch.from_numpy(test[tstop_cols].values.astype(np.uint8).reshape(-1, 20, 20))
        TD = torch.from_numpy(test["delta"].values.astype(np.int64))
        acc = np.zeros((len(TD), 400), dtype=np.float32)
        for model in fold_models:
            model.eval()
            with torch.no_grad():
                for i in range(0, len(TD), 4096):
                    x = make_input(TP[i:i + 4096].to(DEV).float(), TD[i:i + 4096].to(DEV))
                    acc[i:i + 4096] += torch.sigmoid(model(x)).cpu().numpy().reshape(-1, 400)
        acc /= len(fold_models)
        np.save(f"{COMP}/scripts/test_probs_{args.tag}.npy", acc)
        print("test probs saved", flush=True)

    with open(f"{COMP}/scripts/result_{args.tag}.json", "w") as fh:
        json.dump({"folds": folds, "mae": oof_mae, "mean": float(np.mean(maes)),
                   "std": float(np.std(maes)), "args": vars(args)}, fh, indent=2)


if __name__ == "__main__":
    main()
