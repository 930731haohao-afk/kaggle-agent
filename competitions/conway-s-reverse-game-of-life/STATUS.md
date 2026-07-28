# conway-s-reverse-game-of-life — STATUS

**FINAL (2026-07-28): submission.csv written — OOF MAE 0.110352** (5-fold stratified by delta, seed 42; folds 0.1111/0.1106/0.1103/0.1092/0.1105).
Champion = cnn_v2 (ch96 depth8 residual, lr3e-3, 18ep, 120k synthetic boards/fold, D4 aug + 8x TTA, delta-conditioned) with per-delta thresholds {d1:0.51, d2-5:0.49}; per-delta dirichlet+coord-ascent weight search over {cnn_v1, cnn_v2, lgb_v1} collapsed to pure cnn_v2 at every delta.
Path: all-zero 0.14513 → per-delta heuristic 0.13959 → lgb_v1 0.12252 → cnn_v1 0.11193 → tree search (23 nodes, found ch96+lr3e-3 + per-delta thresh) → cnn_v2 full-grade 0.11039 → thresholded final 0.110352. No LB (closed 2014).

## Competition
Reverse Conway's Game of Life: given a 20x20 board after `delta` (1-5) steps, predict the
board `delta` steps earlier. Metric: MAE over 400 binary cells x 50k test rows = per-cell
error rate. Train 50k rows, test 50k rows, delta uniform.

## Pipeline log (2026-07-28)

### Stage 1 — EDA
- Forward GoL rule verified exactly (dead boundary, 2000/2000 sampled rows reproduce stop
  from start in delta steps) → unlimited synthetic training data possible.
- Start density 0.145 (all deltas), stop density decays with delta (0.137@d1 → 0.120@d5).
- Baselines: all-zero MAE 0.14513; stop-as-start 0.15509 (only wins at delta=1: 0.117).
- Edge rows/cols sparser (0.104 vs 0.148 interior). No empty boards.
- Validation: 5-fold stratified by delta, seed 42 (config.yaml). i.i.d. rows, no leakage.
- Synthetic generator (density U(0.01,0.99) → 5 warmup steps → start → delta steps → stop,
  discard empties) reproduces train marginals almost exactly (density percentiles
  identical to 4 decimals) — validated in scripts/baseline.py output.

### Stage 2 — Features
Grid data: CNN consumes raw stop board + one-hot delta channels (features learned).
LGB member uses per-cell 7x7 stop-board windows + row/col. No tabular FE stage.

### Stage 3 — Linear protocol
- exp #1 baseline all-zero: 0.14513
- exp #2 per-delta (stop@d1, zero@d2-5): 0.13959
- exp #3 refine probe (forward-consistency hill-climb): NEGATIVE — see below
- cnn_v1: delta-conditioned CNN (6ch input: stop + delta one-hot; 8x64 residual convs,
  BN, OneCycle 20 epochs, batch 512, 100k synthetic boards/fold, D4 train-aug + 8x TTA),
  5-fold CV.
- lgb_v1: per-delta per-cell LightGBM (7x7 window + row/col, 150 trees), 5-fold CV,
  OOF MAE 0.12245.
- blend: cnn_v1 + lgb_v1, 1D weight grid on full OOF (MAE@0.5).

### Rejected lever (logged as exp with NEGATIVE note)
Forward-consistency refinement (greedy cell flips minimizing
hamming(evolve(start,delta), stop) + lam*CNN-prior): forward-mismatch improves
0.099→0.068 but MAE worsens at every delta (0.1127→0.1165). Reverse GoL is many-to-one;
per-cell marginals are MAE-optimal, a single consistent preimage is not.

### Stage 4 — Tree search (harness_v3)
Driver: tree_search/run_conway_v3.py, evaluator: tree_search/eval_conway.py.
Fold-0 proxy (12.5k boards = 5M cells; full 5-fold CNN evals ~8.4 min are unaffordable
per node). Root = cached-OOF reuse of cnn_v1 fold-0, digit-verified.
Budget deviation from the 60-node default: TOTAL=40, burst 5, patience 12 — CNN
solo evals cost ~3 min (search grade 10 epochs/50k synth), 60 nodes would not fit the
session budget. Lineages: ARCH / PERDELTA / SEEDBAG / LGB / BLEND (per-delta weights +
per-delta threshold sweep as first-class blend options, per afsis per-target prior).

Results (23 evaluated nodes; tree in experiments_tree_v3.json):
- Root cnn_v1 fold-0 cached reuse digit-verified: 0.112929.
- Best node 7 (8th eval): blend(cnn_v1, lgb_v1) with per-delta weights + per-delta
  threshold sweep → 0.112899. Per-delta LGB weight ≈ 0.006–0.011 (0 at delta=1);
  thresholds {d1:0.49, d2:0.5, d3:0.5, d4:0.49, d5:0.48}.
- Key solo findings (search grade, 10ep/50k synth): ch96+lr3e-3 best CNN
  (0.113895 vs base 0.115752); scale push lr3e-3+14ep/100k on ch64 → 0.113214;
  depth12 consistently hurts at 10-epoch budget; per-delta separate CNNs lose to
  delta-conditioned single model (0.11848 vs 0.11575); kitchen-sink 12-member blends
  (0.11294–0.11298) lose to the lean 3-member blend.
- evals-to-beat-linear: node 5 (6th eval) tied the linear blend concept; node 7 (8th)
  surpassed it on the fold-0 proxy.
- Honest ledger: dedup rejections 6; burst-seed sanity gate never fired; blend cost
  guard n/a (custom per-delta weight search ~70s/kitchen-sink node, logged); driver
  intervention recorded in backtrack_log (budget 40→30, patience 12→5, burst forced at
  n_eval=18 — CNN nodes cost 5-7 min vs ~3 planned). All scores OOF-only, no LB anchor
  (competition closed 2014).

### Stage 5 — Final champion
- cnn_v2 = full-grade rerun of the search discovery: ch96, depth8, lr3e-3, 18 epochs,
  120k synth/fold, 5-fold CV.
- Final submission: per-delta weight + threshold blend of cnn_v1 + cnn_v2 + lgb_v1 on
  full OOF (scripts/final_pipeline.py).
- Outcome: weight search collapsed to pure cnn_v2 at every delta (cnn_v2 dominates both
  other members); per-delta thresholds {d1:0.51, d2-5:0.49} shave 0.11039 → 0.110352.
- submission.csv validated vs sampleSubmission.csv (columns, id order, binary values,
  live-cell rate 0.072). Experiment #8 logged.
