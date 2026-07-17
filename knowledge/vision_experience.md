# Vision experience library ([INT], score-cited)

Last updated: 2026-07-17 (run #1 digit-recognizer complete). House rules identical to
`experience.md`: every entry carries evidence (`experiment, metric A -> B`); negative results are
first-class; `[EXT]` never mixes in here. Consumed by keyword-header matching (same mechanism as
the tabular library).

---

## digits / grayscale small images (run #1: digit-recognizer, full pipeline)

- **Per-backbone LR sweep changes the ranking AND rescues collapses — mandatory before judging any
  backbone.** At each backbone's own best LR (1ep/8k/3-fold): convnext_atto(1e-4) 0.9806 >
  resnet18(1e-3) 0.9767 > vit_tiny(1e-4) 0.9579 > efficientnet_b0(1e-3) 0.9434 >
  mobilenetv3(1e-3) 0.9303 (vs 0.374 at 3e-4). CNNs preferred 1e-3, ViT/ConvNeXt 1e-4. | Evidence:
  digit-recognizer exp #4-8 (tier-2a), v2_tier2a_results.json.
- **Affine augmentation (no h-flip) is a real gain on digits; strength is per-family.** medium
  (rot15/trans10%/scale0.9-1.1) won for both CNNs (convnext 0.9806->0.9828, resnet18
  0.9767->0.9809); light won for the ViT (0.9579->0.9581, marginal). Never h-flip digits (6/9,
  2/5). | Evidence: digit-recognizer exp #9-11 (tier-2b), v2_tier2b_results.json.
- **Discovery-scale gaps shrink dramatically at full scale — don't over-prune on discovery scores.**
  vit_tiny trailed by 2.5pts at 1ep/8k (0.958 vs 0.983) but only 0.12pts at 3ep/42k (0.9942 vs
  0.9954). Family diversity kept it in the pool and it earned 0.23 blend weight. | Evidence:
  digit-recognizer exp #12-14 (V3).
- **3-family convex blend beat the best solo AND equal weights**: blend OOF 0.99612 vs best solo
  0.99540 vs equal 0.99605; weights resnet18 0.427 / convnext 0.339 / vit_tiny 0.233 — all three
  families contributed (decorrelated errors). SLSQP-on-logloss + coordinate-refine-on-accuracy,
  argmax inside the scorer. | Evidence: digit-recognizer exp #15 (V4), v4_results.json.
- **Cost calibration (GB10, full pipeline)**: discovery (V2) = 14.5 min total (tier-2a 623s +
  tier-2b 247s); V3 full fine-tune 3 configs x 5 folds x 3ep = 61 min; V4 convex solve = seconds.
  Whole run #1 ~= 1.5 h GPU. | Evidence: run logs (v2_tier2a/v2_tier2b/v3_finetune.log).

## backbone selection / cheap proxies

- **Frozen-probe backbone ranking is NOT trustworthy as a fine-tune shortlist (digits/grayscale
  28px domain).** Probe rank vs 1-epoch-fine-tune rank over 5 backbones: Spearman rho = **-0.30**,
  top-1 disagrees (probe: mobilenetv3 0.9648; fine-tune winner: convnext_atto 0.9756), top-2
  disagrees as a set. Do not drop backbones on probe evidence alone; probes remain fine for
  head/resolution questions. | Evidence: vision/derisk_results.json (digit-recognizer, n=8000,
  3-fold shared, 2026-07-17), total 598s on GB10.
- **A single fixed-LR short fine-tune is ALSO an unfair ranker — LR sensitivity dominates it.**
  At lr=3e-4 x 1 epoch full-unfreeze: mobilenetv3 collapsed 0.9648(probe) -> **0.3744**,
  efficientnet_b0 degraded 0.9635 -> 0.8769, while resnet18 (0.9665) and convnext_atto (0.9756)
  trained fine. Backbone ranking judgments require a per-backbone LR mini-sweep at tier 2; a
  one-LR comparison measures LR tolerance, not backbone quality. | Evidence: same run.
- **"Stable trainers" exist and are valuable defaults**: resnet18 and convnext_atto were
  well-behaved in both arms (probe 0.9550/0.9570; fine-tune 0.9665/0.9756); convnext_atto won
  fine-tuning outright at atto scale. | Evidence: same run.
- Scope caveat: single domain (MNIST-like digits, far from ImageNet statistics). Re-measure
  probe-vs-finetune agreement per new domain before trusting either direction; append results
  here. | Evidence: n/a (methodological note).

## discovery-cost calibration (GB10)

- Tier-1 frozen probe: ~45-100s per backbone (8k images, 224px, embed+3-fold logreg).
  Tier-2 1-epoch 3-fold fine-tune: ~33-70s per small backbone. A 5-backbone x 2-arm de-risk
  experiment = ~10 min total. Cheap experiments are genuinely cheap here — prefer measuring over
  assuming. | Evidence: vision/derisk_results.json timing_s block.

## RGB natural images (run #2: cifar-10, 32px upscaled to 128)

- **Per-backbone LR ranking reconfirmed on RGB**: convnext_atto(3e-4) > vit_tiny(3e-4) >
  resnet18(1e-3) > efficientnet(1e-3) > mobilenetv3(1e-3); ConvNeXt/ViT collapse at 1e-3
  (convnext 0.635, vit 0.214) where CNNs peak. | Evidence: cifar-10 tier-2a, v-run #2.
- **Batch-level GPU augmentation did NOT help at 1-epoch discovery** (none > light > medium for all
  3 promoted backbones). Partly a weakness of batched aug (one transform per batch vs per-sample);
  revisit with per-sample aug or more discovery epochs before concluding aug is useless on cifar. |
  Evidence: cifar-10 tier-2b (convnext none 0.878 vs medium 0.843).
- **3-family blend beat best solo again**: OOF 0.9578(convnext)→0.96674(blend), weights
  0.41/0.38/0.22; Public/Private LB 0.96910 (CV↔LB +0.0024 in LB favor). | Evidence: cifar-10 V4.
- **DataLoader + CUDA-fork deadlock is a real hazard** — a multi-worker DataLoader hung the run ~2h
  entering tier-2b. Fix: no DataLoader; preload uint8 arrays + GPU-batch resize/aug/norm. | Evidence:
  cifar-10 run #2 engineering note; vp.py.
