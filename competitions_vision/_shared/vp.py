"""Reusable discovery-first vision pipeline (kaggle-vision-agent).

One parametrized engine for image-classification competitions, driven by a CONFIG dict +
a data adapter. Reused across competitions (digit-recognizer used bespoke scripts; from
cifar-10 onward everything runs through here). Stages: V2 discovery (tier-2a LR sweep,
tier-2b aug sweep) -> promotion -> V3 full fine-tune (OOF npz) -> V4 convex blend + submit.

A per-competition driver supplies:
  CONFIG = dict(comp=<dir>, img_size, n_classes, classes=[...], metric='accuracy',
                backbones=[...], lr_grid=[...], augs={name: transform|None},
                subsample, disc_folds, cv_folds, epochs, id_col, label_col, hflip_ok)
  loaders: load_train() -> (X, y)   load_test() -> (X_test, ids)
  where X is either an (N,H,W[,C]) uint8 array OR a list[str] of image paths.

All training runs under the bit-level determinism preamble. Fixed seed=42, shared folds.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".claude/skills/kaggle-vision-agent/assets"))
from torch_determinism import setup_determinism, seeded_loader_kwargs  # noqa: E402

setup_determinism(seed=42)
import timm  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402
from PIL import Image  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ROOT / ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

SEED = 42
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# -------------------------------------------------------------------------------------
# Data path: NO torch DataLoader (fork+CUDA worker deadlock hazard). X is a uniform uint8
# array (N,H,W,3) preloaded by the adapter; resize / augment / normalize happen in GPU
# batches. Robust (no worker processes) and fast (no per-item PNG decode, GPU resize).
# -------------------------------------------------------------------------------------
_MEAN_D = _MEAN.to(DEV)
_STD_D = _STD.to(DEV)


def _gpu_batch(x_u8, cfg, aug=None):
    """(B,H,W,3) or (B,H,W) uint8 numpy -> normalized (B,3,S,S) GPU tensor; aug on GPU."""
    x_u8 = np.ascontiguousarray(x_u8)
    if x_u8.ndim == 3:                               # grayscale (B,H,W) -> (B,H,W,1)
        x_u8 = x_u8[..., None]
    t = torch.from_numpy(x_u8).to(DEV).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    if aug is not None:
        t = aug(t)                                   # batched GPU aug (one param set per batch)
    if t.shape[-1] != cfg["img_size"]:
        t = F.interpolate(t, size=cfg["img_size"], mode="bilinear", align_corners=False)
    return (t - _MEAN_D) / _STD_D


def _make_model(name, cfg):
    """Create a pretrained timm model at cfg['img_size']. Fixed-resolution models (ViT
    patch16_224, etc.) need img_size passed so timm interpolates position embeddings;
    CNNs accept any input and reject the kwarg -> fall back."""
    n = cfg["n_classes"]
    try:
        return timm.create_model(name, pretrained=True, num_classes=n, img_size=cfg["img_size"])
    except TypeError:
        return timm.create_model(name, pretrained=True, num_classes=n)


def _train_one(name, lr, aug, X, y, tr, cfg, epochs):
    torch.manual_seed(SEED)
    model = _make_model(name, cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bs = cfg["batch"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs * (len(tr) // bs + 1), 1))
    yt = torch.from_numpy(y).to(DEV)
    model.train()
    for ep in range(epochs):
        rng = np.random.RandomState(SEED + ep)
        order = tr.copy(); rng.shuffle(order)
        for i in range(0, len(order), bs):
            bidx = order[i:i + bs]
            xb = _gpu_batch(X[bidx], cfg, aug)
            opt.zero_grad()
            F.cross_entropy(model(xb), yt[bidx]).backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def _predict(model, X, idx, cfg, softmax=True):
    model.eval()
    bs = cfg.get("infer_batch", 512)
    out = []
    for i in range(0, len(idx), bs):
        xb = _gpu_batch(X[idx[i:i + bs]], cfg, None)
        z = model(xb)
        out.append((F.softmax(z, 1) if softmax else z).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- discovery (V2)
def _cv_acc(name, lr, aug, X, y, folds, cfg, epochs):
    accs = []
    for tr, va in folds:
        m = _train_one(name, lr, aug, X, y, tr, cfg, epochs)
        p = _predict(m, X, va, cfg).argmax(1)
        accs.append(float((p == y[va]).mean()))
        del m
        torch.cuda.empty_cache()
    return float(np.mean(accs))


def discover(X, y, cfg, comp_dir):
    """tier-2a LR sweep + tier-2b aug sweep on a stratified subsample; returns promotion list."""
    rng = np.random.RandomState(SEED)
    per = cfg["subsample"] // cfg["n_classes"]
    sub = np.concatenate([rng.choice(np.where(y == c)[0], min(per, (y == c).sum()), replace=False)
                          for c in range(cfg["n_classes"])])
    rng.shuffle(sub)
    Xs = X[sub]
    ys = y[sub]
    folds = list(StratifiedKFold(cfg["disc_folds"], shuffle=True, random_state=SEED).split(np.zeros(len(ys)), ys))

    # tier-2a
    tier2a = {}
    for name in cfg["backbones"]:
        grid = {f"{lr:g}": _cv_acc(name, lr, None, Xs, ys, folds, cfg, 1) for lr in cfg["lr_grid"]}
        best = max(grid, key=grid.get)
        tier2a[name] = dict(lr=float(best), acc=grid[best], grid=grid)
        log(f"tier2a {name:24s} best lr={best} acc={grid[best]:.4f} {grid}")
        experiment_log.log_experiment_v2(
            comp_dir, model=f"{name} (tier-2a LR sweep)", metric=cfg["metric"], direction="maximize",
            score=grid[best], cv=dict(scheme="StratifiedKFold", n_splits=cfg["disc_folds"], seed=SEED,
                                      subsample=cfg["subsample"]),
            notes=f"LR sweep {cfg['lr_grid']}; grid={grid}; img={cfg['img_size']}; determinism on")
    ranked = sorted(tier2a, key=lambda n: -tier2a[n]["acc"])
    top = ranked[:cfg.get("top_k", 3)]

    # tier-2b aug on the top-k, each at its own best lr
    tier2b = {}
    for name in top:
        lr = tier2a[name]["lr"]
        grid = {}
        for an, aug in cfg["augs"].items():
            grid[an] = tier2a[name]["acc"] if an == "none" else _cv_acc(name, lr, aug, Xs, ys, folds, cfg, 1)
        best = max(grid, key=grid.get)
        tier2b[name] = dict(lr=lr, aug=best, acc=grid[best], grid=grid)
        log(f"tier2b {name:24s} best aug={best} acc={grid[best]:.4f} {grid}")
        experiment_log.log_experiment_v2(
            comp_dir, model=f"{name} (tier-2b aug sweep, lr={lr:g})", metric=cfg["metric"],
            direction="maximize", score=grid[best],
            cv=dict(scheme="StratifiedKFold", n_splits=cfg["disc_folds"], seed=SEED, subsample=cfg["subsample"]),
            notes=f"aug grid={grid}; hflip_ok={cfg['hflip_ok']}; determinism on")

    promote = [dict(id=f"{n}_{tier2b[n]['aug']}", backbone=n, lr=tier2b[n]["lr"], aug=tier2b[n]["aug"])
               for n in top]
    json.dump(dict(tier2a=tier2a, tier2b=tier2b, promote=promote),
              open(Path(comp_dir) / "scripts/discovery_results.json", "w"), indent=2)
    log(f"PROMOTE: {[p['id'] for p in promote]}")
    return promote


# ---------------------------------------------------------------- full FT (V3)
def finetune(promote, X, y, X_test, cfg, comp_dir):
    comp = Path(comp_dir)
    folds = list(StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    results = {}
    for p in promote:
        oof_path = comp / f"data/oof_{p['id']}.npz"
        if oof_path.exists():
            z = np.load(oof_path)
            results[p["id"]] = float((z["oof"].argmax(1) == y).mean())
            log(f"V3 {p['id']}: resumed OOF acc={results[p['id']]:.4f}")
            continue
        aug = cfg["augs"][p["aug"]]
        oof = np.zeros((len(y), cfg["n_classes"]), np.float32)
        test_sum = np.zeros((len(X_test) if not isinstance(X_test, list) else len(X_test), cfg["n_classes"]), np.float32)
        n_test = len(X_test)
        test_sum = np.zeros((n_test, cfg["n_classes"]), np.float32)
        fs, t0 = [], time.time()
        for k, (tr, va) in enumerate(folds):
            ck = comp / f"data/ck_{p['id']}_f{k}.npz"
            if ck.exists():
                z = np.load(ck); oof[va] = z["oof_va"]; test_sum += z["test"]; fs.append(float(z["s"])); continue
            m = _train_one(p["backbone"], p["lr"], aug, X, y, tr, cfg, cfg["epochs"])
            oof[va] = _predict(m, X, va, cfg)
            tk = _predict(m, X_test, np.arange(n_test), cfg)
            test_sum += tk
            s = float((oof[va].argmax(1) == y[va]).mean()); fs.append(s)
            np.savez_compressed(ck, oof_va=oof[va], test=tk, s=s)
            log(f"V3 {p['id']} fold{k}: acc={s:.4f} ({time.time()-t0:.0f}s)")
            del m; torch.cuda.empty_cache()
        test = test_sum / cfg["cv_folds"]
        cv = float((oof.argmax(1) == y).mean())
        np.savez_compressed(oof_path, oof=oof, test=test, fold_scores=np.array(fs))
        results[p["id"]] = cv
        experiment_log.log_experiment_v2(
            comp_dir, model=f"{p['backbone']} full FT (lr={p['lr']:g}, aug={p['aug']}, {cfg['epochs']}ep)",
            metric=cfg["metric"], direction="maximize", score=cv,
            cv=dict(scheme="StratifiedKFold", n_splits=cfg["cv_folds"], seed=SEED),
            notes=f"V3 full FT; folds={[round(x,4) for x in fs]}; img={cfg['img_size']}; determinism on")
        log(f"V3 CONFIG DONE {p['id']}: OOF={cv:.4f}")
    return results


# ---------------------------------------------------------------- ensemble + submit (V4)
def ensemble_submit(promote, y, ids, cfg, comp_dir):
    comp = Path(comp_dir)
    names = [p["id"] for p in promote]
    O = np.stack([np.load(comp / f"data/oof_{n}.npz")["oof"] for n in names])
    Tst = np.stack([np.load(comp / f"data/oof_{n}.npz")["test"] for n in names])
    solo = {n: float((O[i].argmax(1) == y).mean()) for i, n in enumerate(names)}

    def blend(w, A):
        return np.tensordot(np.asarray(w), A, axes=1)

    def negll(w):
        p = np.clip(blend(w, O), 1e-9, 1); p /= p.sum(1, keepdims=True)
        return -np.log(p[np.arange(len(y)), y]).mean()

    res = minimize(negll, np.full(len(names), 1 / len(names)), method="SLSQP",
                   bounds=[(0, 1)] * len(names),
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))

    # metric-aware scoring: 'logloss' (minimize) uses the SLSQP surrogate directly;
    # 'accuracy' (default, maximize) refines on the discrete metric with argmax in the scorer.
    from sklearn.metrics import log_loss
    metric = cfg.get("metric", "accuracy")
    is_logloss = metric == "logloss"

    def norm(P):
        P = np.clip(P, 1e-9, 1); return P / P.sum(-1, keepdims=True)

    if is_logloss:
        def score(P):        # lower is better -> return negative so "higher=better" logic holds
            return -log_loss(y, norm(P), labels=list(range(cfg["n_classes"])))
        solo = {n: -log_loss(y, norm(O[i]), labels=list(range(cfg["n_classes"]))) for i, n in enumerate(names)}
        w = np.clip(res.x, 0, 1); w /= w.sum()   # SLSQP already minimized logloss
        bw, ba = w.copy(), score(blend(w, O))
        steps = [-0.05, -0.02, 0.02, 0.05]       # light refine on true logloss
    else:
        def score(P):
            return float((P.argmax(1) == y).mean())
        solo = {n: score(O[i]) for i, n in enumerate(names)}
        w = np.clip(res.x, 0, 1); w /= w.sum()
        bw, ba = w.copy(), score(blend(w, O))
        steps = [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]
    changed = True
    while changed:
        changed = False
        for i, s in product(range(len(names)), steps):
            c = bw.copy(); c[i] = np.clip(c[i] + s, 0, 1)
            if c.sum() == 0:
                continue
            c /= c.sum(); a = score(blend(c, O))
            if a > ba + 1e-9:
                bw, ba, changed = c, a, True
    w = bw
    eq = score(blend(np.full(len(names), 1 / len(names)), O))
    best_solo = max(solo, key=solo.get)
    use_blend = ba > solo[best_solo]
    final = blend(w, Tst) if use_blend else Tst[names.index(best_solo)]
    chosen = f"blend{dict(zip(names, [round(float(x), 3) for x in w]))}" if use_blend else f"solo:{best_solo}"
    # report positive numbers (flip sign back for logloss)
    disp = (lambda v: -v) if is_logloss else (lambda v: v)
    log(f"metric={metric} solo={ {k: round(disp(v),5) for k,v in solo.items()} } "
        f"equal={disp(eq):.5f} blend={disp(ba):.5f} -> {chosen}")

    import pandas as pd
    final = norm(final)
    if cfg.get("output") == "proba":              # binary: submit P(positive class)
        col = final[:, cfg.get("proba_class", 1)]
        vals = np.clip(col, 1e-6, 1 - 1e-6)
    else:                                          # multiclass: argmax label
        idx = final.argmax(1)
        vals = [cfg["classes"][i] for i in idx] if cfg.get("classes") else idx
    sub = pd.DataFrame({cfg["id_col"]: ids, cfg["label_col"]: vals})
    out = comp / "submissions" / "vp_submission.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    ba, solo_b = disp(ba), disp(solo[best_solo])
    experiment_log.log_experiment_v2(
        comp_dir, model=f"V4 ensemble ({chosen})", metric=metric,
        direction="minimize" if is_logloss else "maximize", score=round(ba, 6),
        cv=dict(scheme="StratifiedKFold", n_splits=cfg["cv_folds"], seed=SEED),
        base_models=[dict(name=n, score=round(disp(solo[n]), 5)) for n in names],
        ensemble=dict(weights=dict(zip(names, [round(float(x), 3) for x in w])), score=round(ba, 5)),
        postprocess=["proba" if cfg.get("output") == "proba" else "argmax"], submission=out.name,
        notes=f"best_solo={solo_b:.5f} equal={disp(eq):.5f}; blend-if-better (s3e20)")
    best = min(ba, solo_b) if is_logloss else max(ba, solo_b)
    json.dump(dict(metric=metric, solo={k: disp(v) for k, v in solo.items()}, equal=disp(eq),
                   blend=ba, weights=dict(zip(names, [float(x) for x in w])),
                   chosen=chosen, oof_final=best),
              open(comp / "scripts/v4_results.json", "w"), indent=2)
    log(f"submission: {out} ({len(sub)} rows)")
    return out, best, chosen
