# Vision experience library ([INT], score-cited)

Last updated: 2026-07-17. House rules identical to `experience.md`: every entry carries evidence
(`experiment, metric A -> B`); negative results are first-class; `[EXT]` never mixes in here.
Consumed by keyword-header matching (same mechanism as the tabular library).

---

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
