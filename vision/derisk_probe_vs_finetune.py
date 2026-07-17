"""De-risk experiment for the discovery-first vision pipeline (docs/vision-pipeline-
discovery-first.drawio.png).

QUESTION: does the ranking of backbones by a FROZEN linear probe (cheap discovery
substrate) agree with their ranking after a SHORT fine-tune (what we'd actually
promote)?  If yes, the cheap probe is a trustworthy Tier-A shortlister; if no, the
discovery substrate must lean on the cheap-fine-tune tier instead.

Design (kept ~1 GPU-hour on the GB10):
  data       : digit-recognizer train.csv, stratified subsample of N=8000 (seed 42)
  backbones  : 5 diverse small timm models (CNN / ViT / hybrid families)
  probe arm  : frozen backbone -> embeddings (224px) -> logistic regression,
               3-fold stratified CV -> accuracy
  finetune arm: full unfreeze, 1 epoch, AdamW lr=3e-4, batch 64, same 3 folds
               (train on 2 folds, eval on 1) -> accuracy
  verdict    : Spearman rank correlation between the two accuracy rankings,
               + whether the top-1 / top-2 backbone sets agree.

Deterministic where it matters: fixed seeds, fixed folds shared by both arms.
Output: vision/derisk_results.json + stdout log.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "competitions" / "digit-recognizer" / "data" / "train.csv"
OUT = Path(__file__).resolve().parent / "derisk_results.json"

SEED = 42
N_SUB = 8000
N_FOLDS = 3
IMG_SIZE = 224
BATCH_EMB = 256
BATCH_FT = 64
FT_EPOCHS = 1
LR = 3e-4

BACKBONES = [
    "resnet18",
    "efficientnet_b0",
    "mobilenetv3_large_100",
    "vit_tiny_patch16_224",
    "convnext_atto",
]

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_subsample() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32).reshape(-1, 28, 28)
    rng = np.random.RandomState(SEED)
    idx = []
    per_class = N_SUB // 10
    for c in range(10):
        cls = np.where(y == c)[0]
        idx.append(rng.choice(cls, per_class, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def to_batch_tensor(x28: np.ndarray) -> torch.Tensor:
    """28x28 grayscale [0,255] -> 3x224x224 ImageNet-normalized tensor batch."""
    t = torch.from_numpy(x28).unsqueeze(1) / 255.0           # N,1,28,28
    t = F.interpolate(t, size=IMG_SIZE, mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)                                  # N,3,H,W
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std


@torch.no_grad()
def extract_embeddings(name: str, X: np.ndarray) -> np.ndarray:
    model = timm.create_model(name, pretrained=True, num_classes=0)  # pooled features
    model.eval().to(DEV)
    embs = []
    for i in range(0, len(X), BATCH_EMB):
        xb = to_batch_tensor(X[i : i + BATCH_EMB]).to(DEV)
        embs.append(model(xb).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(embs)


def probe_cv_accuracy(emb: np.ndarray, y: np.ndarray, folds) -> float:
    accs = []
    for tr, va in folds:
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf.fit(emb[tr], y[tr])
        accs.append(clf.score(emb[va], y[va]))
    return float(np.mean(accs))


def finetune_cv_accuracy(name: str, X: np.ndarray, y: np.ndarray, folds) -> float:
    accs = []
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(SEED + k)
        model = timm.create_model(name, pretrained=True, num_classes=10)
        model.to(DEV).train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        ds = TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr]))
        dl = DataLoader(ds, batch_size=BATCH_FT, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED + k))
        for _ in range(FT_EPOCHS):
            for xb28, yb in dl:
                xb = to_batch_tensor(xb28.numpy()).to(DEV)
                yb = yb.to(DEV)
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(va), BATCH_EMB):
                sel = va[i : i + BATCH_EMB]
                xb = to_batch_tensor(X[sel]).to(DEV)
                pred = model(xb).argmax(1).cpu().numpy()
                correct += int((pred == y[sel]).sum())
        accs.append(correct / len(va))
        del model
        torch.cuda.empty_cache()
    return float(np.mean(accs))


def main() -> None:
    t0 = time.time()
    log(f"device={DEV} ({torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'})")
    X, y = load_subsample()
    log(f"subsample: X={X.shape} y={y.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))  # SAME folds for both arms

    results: dict = {"probe": {}, "finetune": {}, "timing_s": {}}
    for name in BACKBONES:
        t = time.time()
        emb = extract_embeddings(name, X)
        acc_p = probe_cv_accuracy(emb, y, folds)
        results["probe"][name] = acc_p
        results["timing_s"][f"probe_{name}"] = round(time.time() - t, 1)
        log(f"PROBE     {name:26s} acc={acc_p:.4f}  ({time.time()-t:.0f}s)")

    for name in BACKBONES:
        t = time.time()
        acc_f = finetune_cv_accuracy(name, X, y, folds)
        results["finetune"][name] = acc_f
        results["timing_s"][f"ft_{name}"] = round(time.time() - t, 1)
        log(f"FINETUNE  {name:26s} acc={acc_f:.4f}  ({time.time()-t:.0f}s)")

    p = np.array([results["probe"][n] for n in BACKBONES])
    f = np.array([results["finetune"][n] for n in BACKBONES])
    rho, pval = spearmanr(p, f)
    rank_p = [BACKBONES[i] for i in np.argsort(-p)]
    rank_f = [BACKBONES[i] for i in np.argsort(-f)]
    results["spearman_rho"] = float(rho)
    results["spearman_p"] = float(pval)
    results["rank_probe"] = rank_p
    results["rank_finetune"] = rank_f
    results["top1_agree"] = rank_p[0] == rank_f[0]
    results["top2_agree_as_set"] = set(rank_p[:2]) == set(rank_f[:2])
    results["config"] = dict(seed=SEED, n_sub=N_SUB, n_folds=N_FOLDS,
                             img_size=IMG_SIZE, ft_epochs=FT_EPOCHS, lr=LR,
                             backbones=BACKBONES)
    results["total_wall_s"] = round(time.time() - t0, 1)

    OUT.write_text(json.dumps(results, indent=2))
    log(f"rank_probe    = {rank_p}")
    log(f"rank_finetune = {rank_f}")
    log(f"spearman rho={rho:.3f} (p={pval:.3f})  top1_agree={results['top1_agree']}  "
        f"top2_agree={results['top2_agree_as_set']}")
    log(f"wrote {OUT}  total={results['total_wall_s']}s")


if __name__ == "__main__":
    main()
