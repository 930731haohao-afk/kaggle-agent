# Vision experience library ([INT], score-cited)

Last updated: 2026-07-30 (run #5 siim-isic-melanoma complete). House rules identical to
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

## binary / logloss (run #3: dogs-vs-cats, variable-size JPG -> 160px)

- **Metric-aware blend works**: for logloss, minimize logloss in the SLSQP weight solve and submit
  probabilities (not argmax). Blend logloss 0.02914 < best solo 0.03642 < equal 0.03013; weights
  convnext 0.591/vit 0.215/resnet 0.195. | Evidence: dogs-vs-cats V4, v4_results.json.
- **ConvNeXt/ViT low-LR pattern holds on a 3rd domain** (best 1e-4, collapse at 1e-3: convnext 0.536).
  Now confirmed across digits, cifar, and cats/dogs. | Evidence: dogs-vs-cats tier-2a.
- **Submission access caveat (operational)**: having a competition's DATA does not imply being able
  to SUBMIT. Only competitions with userHasEntered=True and open submissions accept uploads; closed
  comps give 400, un-joined comps give 403. Confirm entry status before promising a leaderboard score.
  | Evidence: dogs-vs-cats 400 / redux 403 (2026-07-17); only digit-recognizer + cifar-10 submittable.

## fine-grained multiclass / TFRecords (run #4: tpu-flowers, 104-class, 192px)

- **Backbone ranking is DOMAIN-SPECIFIC — re-discover per competition.** On 104-class flowers:
  efficientnet_b0(1e-3) 0.688 ~ mobilenetv3(1e-3) 0.686 >> vit_tiny 0.421 > convnext_atto 0.395 >
  resnet18 0.267. convnext_atto (the digit/cifar champion) fell to 4th; EfficientNet/MobileNet won.
  Never assume a backbone that won one domain wins another. | Evidence: tpu-flowers tier-2a, run #4.
- **h-flip augmentation helps flowers** (flip-invariant); light > none > medium for both CNNs.
  Contrast digits (no h-flip). Reason augmentation from the domain, per competition. | Evidence:
  tpu-flowers tier-2b.
- **Big blend gain from cross-family diversity**: OOF 0.9084(best solo)->0.93186(blend), +0.0235;
  weights vit 0.459/eff 0.288/mob 0.253 — all three families essential. | Evidence: tpu-flowers V4.
- **Notebook-only competitions — the working submission route**: direct CSV gets HTTP 400 (e.g.
  Petals to the Metal / TPU comps). Route that WORKS: `kaggle datasets create` the predictions ->
  push a notebook (competition_sources attached) that copies them to /kaggle/working/submission.csv
  -> `kaggle competitions submit -c <comp> -k <owner/notebook> -f submission.csv -v <version>`
  (needs kaggle CLI >= 2.2.2). Confirm submission TYPE before promising an LB. | Evidence:
  tpu-getting-started: CSV 400 then notebook route -> Public LB 0.93303 (2026-07-20).

## dermoscopy / medical imaging, heavy imbalance (run #5: siim-isic-melanoma, 1.77% positive)

- **ViT beat every CNN by a wide margin on dermoscopy, and the discovery-scale gap WIDENED at
  full scale — a direct counterexample to the digit-recognizer prior "gaps shrink at full scale,
  don't over-prune on discovery scores".** At 2ep/fold-0 vit_small(1e-4) 0.8853 vs
  efficientnet_b0(1e-3) 0.8782 — a 0.007 gap that looks like noise. At 10ep/5-fold the same pair
  is 0.92309 vs 0.88447, a 0.039 gap (5.5x wider). The prior is therefore domain-dependent: on
  this domain discovery UNDERSTATED the winner's advantage, so promoting only on discovery rank
  would have been right, and promoting extra CNNs "for family diversity" bought nothing (both
  CNNs took 0 blend weight). Keep running the sweep; just don't assume the direction of the
  discovery-vs-full-scale distortion. | Evidence: siim-isic run2, discovery.json vs
  res_vit_small_256.json / res_effb0_256.json.
- **Per-backbone LR pattern reconfirmed on a 5th domain**: CNNs peak at 1e-3 (resnet50 0.8694,
  efficientnet_b0 0.8782, efficientnet_b3 0.8716), ViT and ConvNeXt at 1e-4 (vit_small 0.8853,
  convnext_tiny 0.8643); both collapse at the other end (convnext_tiny 0.7775 at 1e-3, vit_small
  0.8173 at 1e-3). Now holds across digits, cifar, cats/dogs, flowers and dermoscopy. | Evidence:
  siim-isic run2 discovery sweep (18 configs, 52 min).
- **Input resolution is a better diversity axis than architecture here.** The same vit_small at
  256px and 384px took 0.80 of the convex blend weight between them (0.33 / 0.47); resnet50 and
  efficientnet_b0 took 0.00. Higher resolution also won solo (384px 0.92596 > 256px 0.92309),
  consistent with dermoscopic structures being fine-grained. | Evidence: siim-isic run2
  blend_results.json.
- **Dihedral augmentation is safe and appropriate for dermoscopy** (h-flip + v-flip + transpose =
  all 8 orientations), unlike digits where h-flip is forbidden. 4x flip TTA at inference was used
  throughout. Reason augmentation from the domain, every time. | Evidence: siim-isic run2 vision.py.
- **Cost calibration (GB10, 28,984 train images)**: JPEG decode + centre-crop + resize of 33k
  images (25 GB of JPEG, up to 6000x4000) = **44 s** at 256px with PIL `draft()` DCT downscaling
  and an 18-process pool — the single highest-leverage engineering trick in the run; without
  `draft()` the 6000x4000 photos dominate everything. 5-fold x 10ep fine-tune: resnet18 ~28 min,
  vit_small/efficientnet_b0 ~49 min at 256px, resnet50 ~70 min, vit_small ~89 min at 384px. |
  Evidence: siim-isic run2 logs.
