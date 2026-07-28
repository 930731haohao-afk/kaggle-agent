# EMVIC — Eye Movement Verification and Identification

**Last Updated**: 2026-02-22

## Competition Overview
- **URL**: https://www.kaggle.com/c/emvic
- **Problem**: 37-class classification — identify a person from eye movement recordings
- **Metric**: Log Loss (minimize)
- **Dataset**: 652 train, 326 test, 8192 features (4 channels x 2048 timepoints)
- **Deadline**: 2012-04-15 (closed)

## Data Characteristics
- **Features**: 4 eye-tracking channels (lx, ly, rx, ry) x 2048 timepoints per recording
- **Classes**: 37 person IDs (1-indexed), severely imbalanced (2 to 105 samples per class)
- **Challenges**: p >> n (8192 features, 652 samples), extreme class imbalance, small dataset

## EDA Insights
- PCA: 30 components capture 90% variance, 100 components capture 97.4%
- kNN baseline: ~0.457 accuracy (3-NN on 100 PCA components)
- Velocity and gaze speed features show strong per-individual patterns
- Left-right eye coordination (inter-eye distance) varies across individuals

## Feature Engineering
115 statistical features + 100 PCA components = **215 total features**

**Statistical Features (per channel x 4 channels):**
- Basic stats: mean, std, median, min, max, range, percentiles (10/25/75/90)
- Velocity: mean, std, abs_mean, abs_max of first differences
- Acceleration: std, abs_mean of second differences
- Zero-crossing rate of velocity
- Segment stats: mean/std for each quarter of the time series

**Cross-channel Features:**
- Left-right eye differences (mean, std for x and y)
- Inter-eye Euclidean distance
- Gaze speed magnitude (left and right separately): mean, std, max
- Fixation count (low-velocity segments, threshold=2.0)
- Saccade count (high-velocity segments, threshold=20.0)

**PCA Features:**
- 100 PCA components from StandardScaled raw features (97.4% variance)

## Experiment Results

| # | Model | CV Log Loss | CV Std | Notes |
|---|-------|------------|--------|-------|
| 1 | RandomForest | 1.464 | 0.146 | 500 trees, balanced weights |
| 2 | **LightGBM** | **0.902** | 0.133 | Best single model |
| 3 | XGBoost | 1.115 | 0.123 | multi:softprob |
| 4 | CatBoost | 1.045 | 0.095 | MultiClass, balanced |
| 5 | Ensemble (LGB=0.7, XGB=0.1, Cat=0.2) | 0.903 | — | OOF weight optimization |
| 6 | Ensemble (equal avg) | 1.023 | — | Simple average |

**Best model**: LightGBM (CV 0.902), essentially tied with weighted ensemble (0.903)

## Submission
- **File**: `submissions/submission_ensemble3_cv0.902_20260222_130748.csv`
- **Format**: 326 rows x 37 cols (probabilities), no header, no ID
- **Models**: Weighted ensemble (LGB=0.7, XGB=0.1, Cat=0.2) retrained on full data
- **Status**: Competition closed (2012) — API submission returns 400 error

## Validation Strategy
- StratifiedKFold (5 folds, shuffle=True, seed=42)
- Warning: Classes with 2 samples get split unevenly (< n_splits members)
- `labels` parameter required in `log_loss()` to handle missing classes in folds

## Key Learnings
- Statistical feature extraction is effective for time series fingerprinting
- LightGBM dominates on this small dataset (outperforms RF, XGB, CatBoost)
- Ensemble barely helps when one model dominates (RF weighted to 0)
- Class imbalance handling critical: `is_unbalance=True` for LGB, `auto_class_weights="Balanced"` for CatBoost
- Small dataset (652 samples, 37 classes) limits model complexity

## Files
- `config.yaml` — competition metadata
- `experiments.json` — 5 experiment records
- `scripts/eda.py` — EDA analysis
- `scripts/train.py` — full training pipeline (feature engineering + 4 models + ensemble)
- `scripts/submit.py` — submission generation (retrain on full data)
- `data/processed.npz` — preprocessed features (stat + PCA)
- `data/best_weights.npy` — optimal ensemble weights
