# cifar-10 — Vision Pipeline Run #2 Report

**Competition:** cifar-10 (Object Recognition in Images — 10-class RGB 32×32)
**Agent:** kaggle-vision-agent (discovery-first) · **Date:** 2026-07-17 · **GPU:** NVIDIA GB10
**Metric:** accuracy (maximize) · **CV:** StratifiedKFold(5, shuffle, seed=42) · **img_size:** 128

> Second run; first on RGB image-folder data and first through the reusable `_shared/vp.py` engine.
> All numbers are OOF CV; the agent discovered its config from the data (no external recipes).

## Headline

| | accuracy |
|---|---|
| **Final 3-model convex blend** | **OOF 0.96674 → Public/Private LB 0.96910** |
| best single (convnext_atto) | 0.9578 |
| equal-weight blend | 0.9664 |

Blend weights: convnext_atto 0.409 / vit_tiny 0.375 / resnet18 0.216 — all three families earned
weight; blend beat best solo and equal. **CV↔LB gap: +0.0024 in LB's favor** (honest CV).

## Discovery (V2, 8k subsample, 3-fold, 1 epoch)

- tier-2a LR sweep: convnext_atto(3e-4) 0.878 > vit_tiny(3e-4) 0.856 > resnet18(1e-3) 0.798 >
  efficientnet(1e-3) 0.796 > mobilenetv3(1e-3) 0.782. Same pattern as digit-recognizer: CNNs peak
  at 1e-3, ConvNeXt/ViT at 3e-4 (and collapse at 1e-3).
- tier-2b augmentation: **none won for all three** promoted backbones (light/medium hurt at the
  1-epoch discovery scale). Promoted: convnext_atto / vit_tiny / resnet18, all aug=none.

## Full fine-tune (V3, 5-fold, 3 epochs, 128px, determinism preamble)

| config | OOF acc |
|---|---|
| convnext_atto | 0.9578 |
| vit_tiny | 0.9539 |
| resnet18 | 0.9447 |

## Findings distilled to vision_experience.md

- Per-backbone LR ranking reconfirmed on RGB natural images (2nd competition).
- Batch-level GPU augmentation did not help at the short discovery scale — a limitation of the
  no-DataLoader rewrite (batched aug applies one transform per batch); worth revisiting with
  per-sample aug or more discovery epochs.
- 3-family blend again beat best solo; weights spread across families (0.41/0.38/0.22).

## Engineering note (robustness fix this run)

The first attempt hung ~2h on a PyTorch DataLoader worker + CUDA-fork deadlock entering tier-2b.
Fixed by removing DataLoader entirely: images preload once into a uint8 array, and
resize/augment/normalize run in GPU batches. No worker processes = no deadlock, and no repeated
PNG decode. This engine (`competitions_vision/_shared/vp.py`) is now the shared basis for all runs.

## Reproducibility

Layer 1 (seeds/folds/OOF cache/experiment log) + Layer 2 determinism preamble on every training.
Preloaded arrays cached to data/*.npy. Env: torch 2.14+cu130, timm 1.0.28, uv-locked.
