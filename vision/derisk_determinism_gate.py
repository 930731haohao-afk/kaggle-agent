"""Layer-2 determinism gate check for the vision pipeline (GB10).

Trains the SAME tiny fine-tune config twice in the same process style the pipeline uses,
and byte-compares the resulting validation logits (the OOF analogue). Passes iff
max|diff| == 0 bit-for-bit. Also re-extracts frozen embeddings twice and compares.

This is the vision analogue of the tabular "retrain emits the exact same OOF" gate
(docs/reproducibility.md Layer 2; memory: lgbm-determinism-oof-gate).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / ".claude" / "skills" / "kaggle-vision-agent" / "assets"

CHILD = r"""
import sys, json
sys.path.insert(0, {assets!r})
from torch_determinism import setup_determinism, seeded_loader_kwargs
setup_determinism(seed=42)                     # BEFORE model/dataloader creation

import numpy as np, pandas as pd, torch, timm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv({data!r}, nrows=1200)
y = df["label"].to_numpy()
X = df.drop(columns=["label"]).to_numpy(dtype=np.float32).reshape(-1, 28, 28)
tr, va = np.arange(0, 1000), np.arange(1000, 1200)

def to_batch(x28):
    t = torch.from_numpy(x28).unsqueeze(1) / 255.0
    t = F.interpolate(t, size=224, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std

model = timm.create_model("resnet18", pretrained=True, num_classes=10).cuda().train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
dl = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                batch_size=64, shuffle=True, **seeded_loader_kwargs(seed=42))
for xb28, yb in dl:
    opt.zero_grad()
    F.cross_entropy(model(to_batch(xb28.numpy()).cuda()), yb.cuda()).backward()
    opt.step()

model.eval()
with torch.no_grad():
    logits = model(to_batch(X[va]).cuda()).cpu().numpy()
np.save({out!r}, logits)

# frozen-embedding arm
emb_model = timm.create_model("resnet18", pretrained=True, num_classes=0).cuda().eval()
with torch.no_grad():
    emb = emb_model(to_batch(X[va]).cuda()).cpu().numpy()
np.save({emb_out!r}, emb)
print("child done")
"""


def run_once(tag: str) -> tuple[Path, Path]:
    out = ROOT / "vision" / f"_det_gate_{tag}.npy"
    emb_out = ROOT / "vision" / f"_det_gate_emb_{tag}.npy"
    code = CHILD.format(assets=str(ASSETS),
                        data=str(ROOT / "competitions_vision/digit-recognizer/data/train.csv"),
                        out=str(out), emb_out=str(emb_out))
    subprocess.run([sys.executable, "-c", code], check=True)
    return out, emb_out


def main() -> None:
    import numpy as np
    a, ea = run_once("a")
    b, eb = run_once("b")
    la, lb = np.load(a), np.load(b)
    ma, mb = np.load(ea), np.load(eb)
    ft_ok = np.array_equal(la, lb)
    emb_ok = np.array_equal(ma, mb)
    print(f"fine-tune logits bit-identical across processes: {ft_ok} "
          f"(max|diff|={np.abs(la - lb).max():.3e})")
    print(f"frozen embeddings bit-identical across processes: {emb_ok} "
          f"(max|diff|={np.abs(ma - mb).max():.3e})")
    for p in (a, b, ea, eb):
        p.unlink()
    print("GATE:", "PASS" if (ft_ok and emb_ok) else "FAIL")
    sys.exit(0 if (ft_ok and emb_ok) else 1)


if __name__ == "__main__":
    main()
