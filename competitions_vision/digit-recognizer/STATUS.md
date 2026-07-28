# Digit Recognizer Competition Memory

**Date**: 2026-02-19

## Competition Info
- URL: https://www.kaggle.com/competitions/digit-recognizer
- Problem: Multiclass classification (classify handwritten digits 0-9)
- Metric: Categorization Accuracy
- Train: 42,000 images (28x28 grayscale), Test: 28,000 images
- 784 pixel features (pixel0-pixel783), values 0-255

## Key EDA Findings
- Target well-balanced (1.23x imbalance ratio, min=3,795 for digit 5, max=4,684 for digit 1)
- 76 dead pixels (zero variance, mostly borders), 121 near-constant (std < 1)
- 527 of 784 pixels carry meaningful information
- Most confusable digit pairs by mean image: 4&9 (0.927), 5&8 (0.913), 3&5 (0.902)
- PCA: data is highly compressible — first few components capture bulk of variance
- Train/test pixel distributions nearly identical (corr=0.9999) — no covariate shift
- Zero duplicates in train, zero train-test overlap
- No missing values

## Feature Engineering
- **Track A (tabular)**: 858 features — normalized pixels + image stats (total ink, nonzero count, mean active intensity) + row/column projections (28 each) + quadrant sums + horizontal/vertical symmetry + center of mass + bounding box features (top, bottom, left, right, height, width, aspect ratio)
- **Track B (CNN)**: Raw pixels reshaped to 28x28x1, normalized to 0-1, augmented with random rotation (±15°) and shift (±2px)

## Experiment Results

| # | Model | CV Accuracy | CV Std | Time |
|---|-------|-------------|--------|------|
| 1 | Majority Class | 0.11152 | 0.00000 | 0s |
| 2 | LogisticRegression | 0.92210 | 0.00193 | 67s |
| 3 | LightGBM-default | 0.97102 | 0.00122 | 279s |
| 4 | CNN-2block (5-fold ensemble) | 0.99560 | 0.00043 | 551s |

## Submissions
- `cnn_submission_20260219_141136.csv` → Public LB: **0.99610**

## Model Details
- **Best model**: 2-block CNN (32→64 channels), BatchNorm, Dropout (0.25/0.5), Adam optimizer
- Architecture: Conv3x3→BN→ReLU→Conv3x3→BN→ReLU→MaxPool→Dropout (x2 blocks) → FC(256)→BN→ReLU→Dropout→FC(10)
- Data augmentation: random rotation ±15°, random shift ±2px
- 5-fold ensemble (averaged softmax probabilities)
- Validation: Stratified 5-Fold CV

## Lessons Learned
- CNN vastly outperforms tabular ML for image tasks (99.56% vs 97.10%)
- Simple 2-block CNN with augmentation is sufficient for MNIST-level accuracy
- 5-fold ensemble averaging gives a small but consistent boost
- Public LB (99.61%) slightly exceeded CV (99.56%) — good generalization

## Potential Improvements
- Deeper architecture (ResNet, EfficientNet)
- More aggressive augmentation (elastic deformation, Cutout)
- Test-time augmentation (TTA)
- Ensemble CNN with tabular models
- Learning rate warmup + cosine annealing
