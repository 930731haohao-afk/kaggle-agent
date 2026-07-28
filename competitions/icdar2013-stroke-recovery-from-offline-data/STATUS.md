# ICDAR 2013 Stroke Recovery from Offline Data

## Competition Details
- **URL**: https://www.kaggle.com/competitions/icdar2013-stroke-recovery-from-offline-data
- **Task**: Recover pen stroke trajectories (x, y over time) from offline signature images
- **Metric**: Column-wise RMSE (x and y as single flat vector)
- **Competition Status**: Closed (ICDAR 2013)

## Data
- **Train**: 605 signatures from 120 writers, 148,083 trajectory points
- **Test**: 476 signatures from 80 writers (zero overlap), 122,911 points to predict
- **Images**: 1,081 JPGs (one per signature, variable size)
- **Coordinates**: Per-signature min-max normalized to [0, 1]
- **Avg points/signature**: ~245 (range 52-871)

## Problem Understanding
- Given a static image of a handwritten signature, predict the temporal pen trajectory
- The image shows WHERE the pen was; the challenge is recovering WHEN (temporal ordering)
- Pen trajectories tend to go right-to-left (avg x: 0.682 → 0.530) for these signatures
- LB best: 0.24449, LB average benchmark: 0.25258 — very tight margins (3%)

## Approaches Tried

### 1. Skeleton + Simple Ordering (RMSE ~0.27-0.33)
- Extract skeleton from binarized image
- Order pixels by XY-scan, YX-scan, nearest-neighbor, or graph traversal
- Resample to target length, normalize to [0,1]
- **Result**: All worse than predicting the mean (0.260)
- **Insight**: Simple skeleton ordering is essentially random w.r.t. temporal order

### 2. CDF-Based Prediction (RMSE ~0.249)
- Sort skeleton pixels by x-coordinate
- Map time fraction to x-CDF percentile (monotonically increasing x)
- y from sliding window mean of skeleton y-values
- Correlation-based direction detection (77% accuracy)
- Blend with mean prediction (α=0.6)
- **Result**: 0.249 — slightly beats LB average benchmark (0.253)

### 3. Template Matching (Writer-Held-Out RMSE ~0.249)
- Compute 11 skeleton features (pixel count, aspect ratio, moments, etc.)
- Find k most similar training signatures (excluding same writer)
- Average their trajectories as prediction
- k=50 optimal for writer-held-out evaluation
- **Result**: 0.249 (same writer templates gave 0.199 but that's leakage)

### 4. Ensemble: CDF + Template (Writer-Held-Out RMSE ~0.245) ⭐
- 50% CDF prediction (with correlation direction + α=0.6 blend)
- 50% Template matching (k=50, writer-held-out)
- **Result**: **0.24515** — within 0.001 of LB best (0.24449)

## Final Results

| Experiment | Method | Writer-Held-Out RMSE |
|-----------|--------|---------------------|
| v1 | CDF + mean blend (α=0.6) | 0.24939 |
| v2 | CDF + correlation direction + α=0.5 | 0.25132 |
| v3 | CDF + correlation direction + α=0.6 | 0.24939 |
| v4 | Template k=50 writer-held-out | 0.24909 |
| v5 | **Ensemble CDF(0.5) + Template(0.5)** | **0.24515** |

**Leaderboard comparison**: Best LB = 0.24449, Our best = 0.24515 (within 0.3%)

## Key Insights

1. **Margins are tiny**: Best teams improved only 3% over predicting the mean — this is an extremely hard problem
2. **Ordering is the bottleneck**: The skeleton gives us WHERE, but recovering WHEN is fundamentally ill-posed from a single image
3. **Same-writer leakage**: Template matching with same-writer signatures gives 0.199 but that's not realistic (no writer overlap in train/test)
4. **Direction matters**: Signatures go right-to-left on average; correlation-based detection gets 77% correct
5. **Ensembling helps**: Combining image-derived CDF with data-driven template averages provides complementary signals
6. **Heavy blending toward mean is optimal**: The mean prediction is strong because the ordering problem is so noisy
7. **Competition closed**: API submission rejected with 400 error (ICDAR 2013)

## Files
```
scripts/
├── 00_inspect_data.py          # Initial data inspection
├── 00b_structural_analysis.py  # Train/test structure analysis
├── 00c_occurrence_analysis.py  # Occurrence and writer analysis
├── 01_eda.py                   # EDA with trajectory visualization
├── 02_baseline_skeleton.py     # Skeleton + ordering baselines
├── 03_metric_analysis.py       # Metric computation analysis
├── 03b_metric_verify.py        # Metric verification
├── 04b_fast_solution.py        # Fast NN ordering + smoothing
├── 05_distribution_approach.py # CDF and distribution methods
├── 06_final_solution.py        # CDF with direction + blending
├── 07_learned_direction.py     # Direction estimation methods
├── 08_refined_solution.py      # Comprehensive config search
├── 09_template_matching.py     # Template matching (sig-level LOO)
├── 10_leakage_check.py         # Leakage analysis + writer-held-out
├── 11_ensemble_final.py        # Final ensemble solution
submissions/
├── submission_*_final_ensemble.csv  # Best submission (v5)
└── ...other submissions...
```
