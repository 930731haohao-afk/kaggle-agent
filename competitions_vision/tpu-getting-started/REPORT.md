# tpu-getting-started (Petals to the Metal) — Vision Pipeline Run #4 Report

**Competition:** tpu-getting-started / "Petals to the Metal" — 104-class flower classification
**Agent:** kaggle-vision-agent (discovery-first) · **Date:** 2026-07-20 · **GPU:** GB10 · **img:** 192
**Metric:** macro-F1 (leaderboard); accuracy used as the CV proxy for selection.

> Fourth run; first 104-CLASS fine-grained task and first TFRecord-format data (decoded with the
> pure-python `tfrecord` package, no TensorFlow). Train+val TFRecords merged = 16,465 labeled images;
> 7,382 test images. All numbers are OOF CV.
> **Submission status: SUBMITTED via the notebook route → Public LB 0.93303 (macro-F1).** Petals to
> the Metal is a NOTEBOOK-ONLY competition (direct CSV upload returns HTTP 400). Route: package the
> predictions as a Kaggle dataset → push a notebook that emits them as submission.csv (competition
> attached) → `kaggle competitions submit -k <notebook> -f submission.csv -v <ver>`. Submitted on the
> huangweihaohuang account. CV↔LB: OOF acc 0.93186 vs LB macro-F1 0.93303 — consistent.

## Headline (OOF accuracy)

| | accuracy |
|---|---|
| **Final 3-model convex blend (OOF acc)** | **0.93186** |
| **Kaggle Public LB (macro-F1)** | **0.93303** |
| best single (efficientnet_b0) | 0.90835 |
| equal-weight blend | 0.93015 |

Blend weights: vit_tiny 0.459 / efficientnet_b0 0.288 / mobilenetv3 0.253 — big blend gain (+2.35
over best solo) from decorrelated errors across three families.

## Discovery (V2, 6k subsample, 3-fold, 1 epoch)

tier-2a LR: **efficientnet_b0(1e-3) 0.688 > mobilenetv3(1e-3) 0.686 > vit_tiny(3e-4) 0.421 >
convnext_atto(3e-4) 0.395 > resnet18(1e-3) 0.267.** Notably the backbone ranking FLIPPED vs the
previous 3 comps: EfficientNet/MobileNet lead on fine-grained flowers while convnext_atto (the
digit/cifar winner) fell to 4th — the discovery-first design correctly re-selected backbones per
domain. tier-2b: light aug (h-flip, valid for flowers) helped both CNNs; none for vit_tiny.

## Full fine-tune (V3, 5-fold, 4 epochs, 192px, determinism)

efficientnet_b0 OOF 0.9084 · mobilenetv3 0.9031 · vit_tiny 0.9054.

## Findings distilled to vision_experience.md

- **Backbone ranking is domain-specific** — flowers favor EfficientNet/MobileNet; convnext_atto
  (digit/cifar champion) underperforms on 104-class fine-grained. Validates per-competition discovery.
- **h-flip augmentation helps flowers** (flip-invariant), unlike digits (asymmetric) — the pipeline
  learned the domain-appropriate augmentation from data.
- **TFRecord adapter** added (pure-python, no TF) — the engine now reads CSV-pixel, image-folder,
  and TFRecord formats.
- **Notebook-only competitions** exist — a valid CSV gets HTTP 400; submission requires a Kaggle
  Notebook. Confirm submission type before promising an LB.
