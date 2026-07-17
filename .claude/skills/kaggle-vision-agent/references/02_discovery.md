# Stage V2 — Discovery (the heart of the pipeline)

The agent runs ITS OWN cheap experiments to find what works on THIS data. No community recipes are
consulted at this stage — the discoveries themselves become the recipe.

## Before starting: read the local evidence

Read `vision/derisk_results.json` and `knowledge/vision_experience.md` first. **Measured verdict
(digit-recognizer, 2026-07-17): probe-vs-finetune ranking agreement FAILED** — Spearman rho -0.30,
top-1/top-2 disagree. Two consequences, both evidence-backed:

1. **Never drop a backbone on tier-1 probe evidence alone.** Probes answer head/resolution
   questions; backbone survival is decided at tier 2.
2. **Tier-2 backbone comparisons require a per-backbone LR mini-sweep.** A single fixed LR is an
   unfair ranker — at lr=3e-4 mobilenetv3 collapsed to 0.374 while convnext_atto hit 0.976; one-LR
   comparison measures LR tolerance, not backbone quality.

Domain caveat: that evidence is from MNIST-like digits. On a new domain, re-run the cheap
agreement check (the whole 5-backbone x 2-arm experiment costs ~10 min on the GB10 —
`vision/derisk_probe_vs_finetune.py` is the template) and append the result to
`knowledge/vision_experience.md`.

## Tier 0 — embedding cache (run once)

Use the bundled utility (`assets/embed_cache.py`) to extract frozen-backbone embeddings for the
stratified subsample (and later, full train+test for the chosen backbones):

- Menu: 4-6 diverse small backbones across families, e.g. `resnet18`, `efficientnet_b0`,
  `mobilenetv3_large_100`, `vit_tiny_patch16_224`, `convnext_atto`. Diversity of family matters
  more than individual strength — later tiers and the ensemble feed on diversity.
- Cache to `competitions_vision/<name>/data/emb_<backbone>.npy` — atomic writes, extract once.

## Tier 1 — frozen-probe experiments (seconds each)

On the cached embeddings, with the SAME folds everywhere:

1. **Backbone ranking**: linear probe (LogisticRegression, `max_iter=2000`) per backbone, k-fold
   accuracy/metric → rank. This is the probe whose trustworthiness derisk_results.json measures.
2. **Head comparison**: linear vs LightGBM on the best backbone's embeddings. (LightGBM on
   embeddings = the familiar tabular pipeline; it sometimes beats linear on small data.)
3. **Resolution probe**: re-extract the top backbone at 2 sizes (e.g. 160 vs 224) and compare —
   resolution effects ARE visible to frozen probes.

What tier 1 CANNOT see: augmentation, LR, fine-tune dynamics. Do not draw conclusions about them
here; the embeddings come from un-augmented images.

## Tier 2 — cheap fine-tune experiments (minutes each)

Short fine-tunes: unfreeze the last block + head, 1-2 epochs, the subsample, possibly reduced
resolution. Run on the top 2-3 backbones from tier 1. Discover:

1. **Augmentation policy** — compare none / light (flip+crop) / medium (+rotation, color jitter) /
   strong (RandAugment-style). Domain sanity: digits should not be h-flipped; medical images have
   their own invariances. Reason from the data, not from a recipe.
2. **LR regime** — 3-point sweep (e.g. 1e-4 / 3e-4 / 1e-3) at fixed epochs.
3. **Unfreeze depth** — head-only vs last-block vs full, at this budget.

Every arm: same folds, same subsample, logged with `log_experiment_v2` (model=backbone,
params={tier, aug, lr, ...}, score, scheme).

## Exit criterion → Stage V3

Promote a SMALL set (typically 3-6 configs) chosen for **diversity + evidence**: the best config of
each backbone family that survived, with its tier-2-chosen augmentation/LR. Write the promotion
list and the evidence for each into `experiments.json` notes before starting fine-tunes.
