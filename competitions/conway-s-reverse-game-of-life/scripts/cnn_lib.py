"""Delta-conditioned CNN for reverse Game of Life.

Config-driven training with 5-fold CV, synthetic data augmentation, D4 augmentation.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "competitions/conway-s-reverse-game-of-life/scripts")
import common

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)

DEFAULT_CFG = dict(
    channels=64,
    depth=8,            # number of 3x3 conv layers in the trunk
    residual=True,      # residual pairs
    epochs=20,
    batch=512,
    lr=2e-3,
    wd=1e-4,
    synth_n=100_000,    # synthetic boards added to training each fold
    d4_aug=True,        # random D4 transform each batch
    tta=True,           # average predictions over 8 D4 transforms
    seed=42,
    per_delta=False,    # if True, train separate model per delta value
)


class Net(nn.Module):
    def __init__(self, channels: int, depth: int, residual: bool):
        super().__init__()
        self.inp = nn.Conv2d(6, channels, 3, padding=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(depth)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(depth)])
        self.out = nn.Conv2d(channels, 1, 1)
        self.residual = residual

    def forward(self, x):
        h = F.relu(self.inp(x))
        for i, (c, b) in enumerate(zip(self.convs, self.bns)):
            z = F.relu(b(c(h)))
            h = h + z if self.residual else z
        return self.out(h)[:, 0]


def make_input(stop: np.ndarray, delta: np.ndarray) -> torch.Tensor:
    """stop (N,20,20) int8, delta (N,) -> (N,6,20,20) float32 tensor."""
    n = len(stop)
    x = np.zeros((n, 6, 20, 20), np.float32)
    x[:, 0] = stop
    x[np.arange(n), delta] = 1.0  # channels 1..5 one-hot delta
    return torch.from_numpy(x)


D4 = [(0, False), (1, False), (2, False), (3, False),
      (0, True), (1, True), (2, True), (3, True)]


def d4_apply(t: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    t = torch.rot90(t, k, dims=(-2, -1))
    if flip:
        t = torch.flip(t, dims=(-1,))
    return t


def d4_invert(t: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        t = torch.flip(t, dims=(-1,))
    return torch.rot90(t, -k, dims=(-2, -1))


def _train_one(cfg, Str, Ptr, Dtr, model_seed):
    """Train one network on (start Str, stop Ptr, delta Dtr). Returns model."""
    torch.manual_seed(model_seed)
    np.random.seed(model_seed % (2**31))
    model = Net(cfg["channels"], cfg["depth"], cfg["residual"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n = len(Str)
    steps_per_epoch = (n + cfg["batch"] - 1) // cfg["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch)
    X = make_input(Ptr, Dtr)
    Y = torch.from_numpy(Str.astype(np.float32))
    g = torch.Generator().manual_seed(model_seed)
    model.train()
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=g)
        for i in range(steps_per_epoch):
            idx = perm[i * cfg["batch"]:(i + 1) * cfg["batch"]]
            xb = X[idx].to(DEVICE, non_blocking=True)
            yb = Y[idx].to(DEVICE, non_blocking=True)
            if cfg["d4_aug"]:
                k = int(torch.randint(0, 8, (1,), generator=g))
                rot, fl = D4[k]
                xb = d4_apply(xb, rot, fl)
                yb = d4_apply(yb, rot, fl)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict(model, stop, delta, cfg, batch=2048):
    """Return probs (N,20,20) float32, with optional D4 TTA."""
    model.eval()
    X = make_input(stop, delta)
    out = torch.zeros(len(stop), 20, 20)
    transforms = D4 if cfg["tta"] else [(0, False)]
    for i in range(0, len(stop), batch):
        xb = X[i:i + batch].to(DEVICE)
        acc = torch.zeros(xb.shape[0], 20, 20, device=DEVICE)
        for rot, fl in transforms:
            # transform only spatial channels; delta channels are constant maps -> safe
            logits = model(d4_apply(xb, rot, fl))
            acc += torch.sigmoid(d4_invert(logits, rot, fl))
        out[i:i + batch] = (acc / len(transforms)).cpu()
    return out.numpy()


def run_cv(cfg: dict, name: str, folds_to_run=None, save=True, log_notes=""):
    """5-fold CV. Saves OOF probs + test preds to cache/preds_{name}.npz.
    Returns (oof_mae, per_fold_scores, oof_probs)."""
    cfg = {**DEFAULT_CFG, **cfg}
    a = common.load_arrays()
    S, P, delta, T, tdelta = a["S"], a["P"], a["delta"], a["T"], a["tdelta"]
    folds = common.get_folds(delta)
    if folds_to_run is None:
        folds_to_run = list(range(common.N_SPLITS))
    oof = np.full((len(S), 20, 20), np.nan, np.float32)
    test_acc = np.zeros((len(T), 20, 20), np.float32)
    scores = []
    t0 = time.time()
    for f in folds_to_run:
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        Str, Ptr, Dtr = S[tr], P[tr], delta[tr]
        if cfg["synth_n"] > 0:
            gS, gP, gD = common.gen_synthetic(cfg["synth_n"], seed=1000 + f)
            Str = np.concatenate([Str, gS])
            Ptr = np.concatenate([Ptr, gP])
            Dtr = np.concatenate([Dtr, gD])
        if cfg["per_delta"]:
            probs = np.zeros((len(va), 20, 20), np.float32)
            tprobs = np.zeros((len(T), 20, 20), np.float32)
            for d in range(1, 6):
                md = _train_one(cfg, Str[Dtr == d], Ptr[Dtr == d], Dtr[Dtr == d],
                                cfg["seed"] + f * 10 + d)
                mva = delta[va] == d
                probs[mva] = predict(md, P[va][mva], delta[va][mva], cfg)
                mte = tdelta == d
                tprobs[mte] = predict(md, T[mte], tdelta[mte], cfg)
        else:
            model = _train_one(cfg, Str, Ptr, Dtr, cfg["seed"] + f * 10)
            probs = predict(model, P[va], delta[va], cfg)
            tprobs = predict(model, T, tdelta, cfg)
        oof[va] = probs
        test_acc += tprobs / len(folds_to_run)
        sc = common.mae((probs > 0.5).astype(np.int8), S[va])
        scores.append(sc)
        print(f"[{name}] fold {f}: MAE {sc:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    mask = ~np.isnan(oof[:, 0, 0])
    oof_mae = common.mae((oof[mask] > 0.5).astype(np.int8), S[mask])
    print(f"[{name}] OOF MAE {oof_mae:.5f}  folds {folds_to_run}  total {time.time() - t0:.0f}s",
          flush=True)
    if save:
        np.savez_compressed(f"{common.CACHE}/preds_{name}.npz",
                            oof=oof, test=test_acc, folds_run=np.array(folds_to_run))
        with open(f"{common.CACHE}/score_{name}.json", "w") as fh:
            json.dump({"name": name, "oof_mae": oof_mae, "fold_scores": scores,
                       "cfg": {k: v for k, v in cfg.items()}}, fh, indent=2)
    return oof_mae, scores, oof


if __name__ == "__main__":
    cfg_json = sys.argv[1] if len(sys.argv) > 1 else "{}"
    name = sys.argv[2] if len(sys.argv) > 2 else "cnn_default"
    folds_arg = sys.argv[3] if len(sys.argv) > 3 else "all"
    folds_to_run = None if folds_arg == "all" else [int(x) for x in folds_arg.split(",")]
    run_cv(json.loads(cfg_json), name, folds_to_run)
