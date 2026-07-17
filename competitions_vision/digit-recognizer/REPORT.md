# digit-recognizer — Vision Pipeline Run #1 Report

**Competition:** digit-recognizer (Kaggle Getting Started — MNIST handwritten digits, 10 classes)
**Agent:** kaggle-vision-agent (discovery-first image pipeline) · **Date:** 2026-07-17 · **GPU:** NVIDIA GB10
**Metric:** categorization accuracy (maximize) · **CV:** StratifiedKFold(5, shuffle, seed=42)

> First end-to-end run of the discovery-first vision pipeline. Every number below is CV (out-of-fold);
> no public kernel or external recipe was consulted — the agent discovered its configuration from the
> data. See `docs/vision/vision-pipeline-discovery-first.drawio.png` for the pipeline.

## 1. Headline result

| | OOF accuracy |
|---|---|
| **Final 3-model convex blend** | **0.99612** |
| best single model (convnext_atto) | 0.99540 |
| equal-weight blend | 0.99605 |

Blend weights (SLSQP-on-logloss + coordinate-refine-on-accuracy): **resnet18 0.427 / convnext_atto 0.339 / vit_tiny 0.233** — all three architecture families earned weight (decorrelated errors), so the blend beat both the best solo and the naive equal blend. Submission: `submissions/v4_blend_submission.csv` (28,000 rows, validated). Not yet uploaded to Kaggle.

## 2. Pipeline stages & evidence

**V2 discovery (14.5 min GPU total; 8k stratified subsample, 3-fold, 1 epoch).**

- *tier-2a — per-backbone LR mini-sweep* (each backbone judged at its OWN best LR, because a single
  fixed LR is an unfair ranker — see §3):

  | backbone | best LR | acc |
  |---|---|---|
  | convnext_atto | 1e-4 | 0.9806 |
  | resnet18 | 1e-3 | 0.9767 |
  | vit_tiny | 1e-4 | 0.9579 |
  | efficientnet_b0 | 1e-3 | 0.9434 |
  | mobilenetv3_large_100 | 1e-3 | 0.9303 |

- *tier-2b — augmentation sweep* (top-3, each at its best LR; digits ⇒ NO horizontal flip):
  convnext_atto **medium** 0.9828, resnet18 **medium** 0.9809, vit_tiny **light** 0.9581.

- *promotion*: 3 survivors across 3 families (convnext_atto / resnet18 / vit_tiny). efficientnet /
  mobilenet dropped (same conv family, strictly worse). vit_tiny kept for family diversity despite a
  lower discovery score.

**V3 full fine-tune (61 min GPU; full 42k, 5-fold, 3 epochs, cosine LR, determinism preamble).**

| config | OOF acc | per-fold |
|---|---|---|
| convnext_atto (1e-4, medium) | 0.99540 | 0.9951 / 0.9948 / 0.9952 / 0.9965 / 0.9954 |
| resnet18 (1e-3, medium) | 0.99531 | 0.9954 / 0.9939 / 0.9954 / 0.9967 / 0.9952 |
| vit_tiny (1e-4, light) | 0.99424 | 0.9945 / 0.9940 / 0.9939 / 0.9938 / 0.9949 |

**V4 ensemble + submission** (convex solve, seconds): result in §1.

## 3. Key findings (distilled to `knowledge/vision_experience.md`)

1. **Per-backbone LR is mandatory.** mobilenetv3 scored 0.374 at lr=3e-4 but 0.930 at its own best
   lr=1e-3 — a single fixed LR measures LR tolerance, not backbone quality. CNNs wanted 1e-3, ViT/
   ConvNeXt wanted 1e-4.
2. **Frozen-probe backbone rankings are untrustworthy** (pre-run derisk: Spearman ρ = −0.30, top-1
   disagrees). Backbone survival is therefore decided by short fine-tunes, not probes — the finding
   that shaped the pipeline design.
3. **Affine augmentation helps digits** (no h-flip); strength is per-family (CNNs: medium, ViT: light).
4. **Discovery-scale gaps shrink at full scale** — vit_tiny trailed by 2.5 pts at 1ep/8k but 0.12 pts
   at 3ep/42k; don't over-prune on discovery scores.
5. **Family diversity pays in the blend** — the transformer member earned 0.23 weight despite being
   the weakest solo.

## 4. Reproducibility

- **Layer 1**: seed=42 everywhere, fixed 5-fold shared across all configs, every experiment logged to
  `experiments.json` (15 entries), OOF/test cached as npz.
- **Layer 2**: all training under the bit-level determinism preamble (`assets/torch_determinism.py`);
  the gate is measured PASS on this GPU (retrain → bit-identical OOF, max|diff|=0;
  `vision/derisk_determinism_gate.py`).
- Environment: torch 2.14+cu130, timm 1.0.28, uv-locked. Reproduce: rerun scripts `v2_*` → `v3_*` →
  `v4_*` in order (each is crash-safe/resumable).

## 5. Honest boundaries

- Scores are local CV; not submitted to Kaggle, so no CV↔LB gap measured yet.
- Determinism guarantee is same-machine/driver/build (floating-point reduction order differs across
  machines).
- 3 epochs / small backbones / single image size (224) — deeper schedules, larger backbones, and a
  resolution sweep are unexplored headroom, deliberately out of scope for a first proof-of-pipeline run.
