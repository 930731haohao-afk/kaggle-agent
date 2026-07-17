# dogs-vs-cats — Vision Pipeline Run #3 Report (CV-only)

**Competition:** dogs-vs-cats (binary cat/dog) · **Metric:** logloss (minimize)
**Agent:** kaggle-vision-agent (discovery-first) · **Date:** 2026-07-17 · **GPU:** GB10 · **img:** 160

> Third run; first BINARY / logloss / probability-output competition, and first with variable-size
> JPGs (preloaded to a uniform 160px array). All numbers are OOF CV.
> **Submission status: BLOCKED** — the old dogs-vs-cats comp is closed to submissions (HTTP 400) and
> the redux version (`dogs-vs-cats-redux-kernels-edition`) was never joined and is closed to new
> entries (HTTP 403). The prediction file is valid and ready; no leaderboard score is obtainable.

## Headline (OOF)

| | logloss |
|---|---|
| **Final 3-model convex blend** | **0.02914** |
| best single (convnext_atto) | 0.03642 |
| equal-weight blend | 0.03013 |

Blend weights (logloss-minimizing SLSQP + refine): convnext_atto 0.591 / vit_tiny 0.215 /
resnet18 0.195. Blend beat best solo and equal weights on logloss.

## Discovery (V2, 6k subsample, 3-fold, 1 epoch — accuracy proxy for backbone selection)

tier-2a LR: convnext_atto(1e-4) 0.979 > resnet18(1e-3) 0.969 > vit_tiny(1e-4) 0.961 >
mobilenetv3(1e-3) 0.945 > efficientnet(1e-3) 0.942. ConvNeXt/ViT again want low LR (1e-4), collapse
at 1e-3 (convnext 0.536). tier-2b: none for convnext/vit, light (h-flip) for resnet18.

## Full fine-tune (V3, 5-fold, 3 epochs, 160px, determinism)

convnext_atto OOF acc 0.9885 · resnet18 0.9848 · vit_tiny 0.9837 (accuracy; logloss used for blend).

## Notes

- The engine was made metric-aware this run: logloss competitions minimize logloss in the weight
  solve and emit probability submissions (P(positive class)); accuracy competitions keep argmax.
- Reproducibility: Layer-1 + Layer-2 determinism preamble; arrays cached. Same standard as runs #1-2.
