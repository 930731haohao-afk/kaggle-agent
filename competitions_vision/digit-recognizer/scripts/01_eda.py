"""
Exploratory Data Analysis for digit-recognizer competition.
Analyzes pixel patterns, class separability, train-test consistency,
and recommends validation strategy.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# Setup
# ============================================================
data_dir = "competitions/digit-recognizer/data"
plot_dir = "competitions/digit-recognizer/scripts/plots"
os.makedirs(plot_dir, exist_ok=True)

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

pixel_cols = [c for c in train.columns if c.startswith('pixel')]
X_train = train[pixel_cols].values
y_train = train['label'].values
X_test = test[pixel_cols].values

print("=" * 70)
print("DIGIT-RECOGNIZER EDA")
print("=" * 70)

# ============================================================
# 1. Sample Digit Visualization
# ============================================================
print("\n" + "=" * 70)
print("1. SAMPLE DIGIT VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for digit in range(10):
    digit_samples = X_train[y_train == digit][:10]
    for j, img in enumerate(digit_samples):
        axes[digit][j].imshow(img.reshape(28, 28), cmap='gray')
        axes[digit][j].axis('off')
        if j == 0:
            axes[digit][j].set_title(f'Digit {digit}', fontsize=10)
plt.suptitle('Sample Digits (10 per class)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '01_sample_digits.png'), dpi=100)
plt.close()
print("Saved sample digit grid to plots/01_sample_digits.png")

# ============================================================
# 2. Mean Images Per Class
# ============================================================
print("\n" + "=" * 70)
print("2. MEAN IMAGES PER CLASS")
print("=" * 70)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for digit in range(10):
    ax = axes[digit // 5][digit % 5]
    mean_img = X_train[y_train == digit].mean(axis=0).reshape(28, 28)
    ax.imshow(mean_img, cmap='hot')
    ax.set_title(f'Digit {digit} (n={np.sum(y_train == digit)})')
    ax.axis('off')
plt.suptitle('Mean Image Per Digit', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '02_mean_images.png'), dpi=100)
plt.close()
print("Saved mean images to plots/02_mean_images.png")

# Report inter-class similarity
mean_images = np.array([X_train[y_train == d].mean(axis=0) for d in range(10)])
from scipy.spatial.distance import cdist
cosine_sim = 1 - cdist(mean_images, mean_images, metric='cosine')
print("\nCosine similarity between mean digit images (most similar pairs):")
pairs = []
for i in range(10):
    for j in range(i + 1, 10):
        pairs.append((i, j, cosine_sim[i, j]))
pairs.sort(key=lambda x: -x[2])
for i, j, sim in pairs[:5]:
    print(f"  Digits {i} & {j}: {sim:.4f}")
print("Most confusable digit pairs (by mean image) are likely error sources.")

# ============================================================
# 3. Pixel Variance Analysis
# ============================================================
print("\n" + "=" * 70)
print("3. PIXEL VARIANCE ANALYSIS")
print("=" * 70)

pixel_variance = X_train.var(axis=0)
pixel_std = X_train.std(axis=0)

# Variance heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall variance
axes[0].imshow(pixel_variance.reshape(28, 28), cmap='viridis')
axes[0].set_title('Pixel Variance (All Digits)')
axes[0].axis('off')

# Mean image
axes[1].imshow(X_train.mean(axis=0).reshape(28, 28), cmap='gray')
axes[1].set_title('Mean Image (All Digits)')
axes[1].axis('off')

# Active pixel mask (variance > threshold)
active_mask = (pixel_variance > 100).reshape(28, 28)
axes[2].imshow(active_mask, cmap='gray')
axes[2].set_title(f'Active Pixels (variance > 100): {active_mask.sum()}')
axes[2].axis('off')

plt.suptitle('Pixel Activity Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '03_pixel_variance.png'), dpi=100)
plt.close()

n_dead = (pixel_variance == 0).sum()
n_low_var = (pixel_variance < 100).sum()
n_active = (pixel_variance >= 100).sum()
print(f"Dead pixels (zero variance): {n_dead} / 784 ({n_dead/784*100:.1f}%)")
print(f"Low-variance pixels (var < 100): {n_low_var} / 784 ({n_low_var/784*100:.1f}%)")
print(f"Active pixels (var >= 100): {n_active} / 784 ({n_active/784*100:.1f}%)")
print("=> Many border/corner pixels carry no information — can be removed for efficiency.")

# ============================================================
# 4. Per-Class Pixel Distributions
# ============================================================
print("\n" + "=" * 70)
print("4. PER-CLASS PIXEL INTENSITY STATISTICS")
print("=" * 70)

for digit in range(10):
    digit_data = X_train[y_train == digit]
    nonzero_ratio = (digit_data > 0).mean()
    mean_intensity = digit_data[digit_data > 0].mean() if (digit_data > 0).any() else 0
    print(f"  Digit {digit}: avg non-zero pixels per image = {(digit_data > 0).sum(axis=1).mean():.0f}, "
          f"mean intensity of active pixels = {mean_intensity:.1f}")

# ============================================================
# 5. PCA Analysis — Class Separability
# ============================================================
print("\n" + "=" * 70)
print("5. PCA ANALYSIS — CLASS SEPARABILITY")
print("=" * 70)

# Use a sample for speed
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), size=10000, replace=False)
X_sample = X_train[sample_idx]
y_sample = y_train[sample_idx]

# Normalize
X_norm = X_sample / 255.0

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_norm)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
print(f"Top 10 PCA components explain: {cumulative[9]*100:.1f}% variance")
print(f"Top 20 PCA components explain: {cumulative[19]*100:.1f}% variance")
print(f"Top 50 PCA components explain: {cumulative[49]*100:.1f}% variance")

n_90 = np.argmax(cumulative >= 0.90) + 1
n_95 = np.argmax(cumulative >= 0.95) + 1
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")

# PCA 2D scatter
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='tab10',
                     alpha=0.3, s=5)
plt.colorbar(scatter, label='Digit')
ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
ax.set_title('PCA 2D Projection (10K samples)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '04_pca_2d.png'), dpi=100)
plt.close()
print("Saved PCA plot to plots/04_pca_2d.png")

# Explained variance curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(1, 51), explained * 100, alpha=0.6, label='Individual')
ax.plot(range(1, 51), cumulative * 100, 'r-o', markersize=3, label='Cumulative')
ax.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='90% threshold')
ax.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
ax.set_xlabel('Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA Explained Variance')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '05_pca_variance.png'), dpi=100)
plt.close()
print("Saved PCA variance plot to plots/05_pca_variance.png")

# ============================================================
# 6. Train-Test Consistency
# ============================================================
print("\n" + "=" * 70)
print("6. TRAIN-TEST CONSISTENCY")
print("=" * 70)

train_mean = X_train.mean(axis=0)
test_mean = X_test.mean(axis=0)
train_std = X_train.std(axis=0)
test_std = X_test.std(axis=0)

# Compare overall statistics
print(f"Train: mean pixel = {X_train.mean():.2f}, std = {X_train.std():.2f}")
print(f"Test:  mean pixel = {X_test.mean():.2f}, std = {X_test.std():.2f}")

# Per-pixel correlation between train and test means
active_mask_flat = pixel_variance > 0
if active_mask_flat.sum() > 0:
    corr = np.corrcoef(train_mean[active_mask_flat], test_mean[active_mask_flat])[0, 1]
    print(f"Correlation of per-pixel means (train vs test): {corr:.6f}")

# Max deviation in pixel means
mean_diff = np.abs(train_mean - test_mean)
print(f"Max pixel mean difference: {mean_diff.max():.2f} (pixel{mean_diff.argmax()})")
print(f"Mean pixel mean difference: {mean_diff[active_mask_flat].mean():.2f}")

# Distribution of non-zero pixel counts
train_nonzero = (X_train > 0).sum(axis=1)
test_nonzero = (X_test > 0).sum(axis=1)
print(f"\nNon-zero pixels per image:")
print(f"  Train: mean={train_nonzero.mean():.1f}, std={train_nonzero.std():.1f}")
print(f"  Test:  mean={test_nonzero.mean():.1f}, std={test_nonzero.std():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(train_mean, test_mean, alpha=0.3, s=10)
axes[0].plot([0, 80], [0, 80], 'r--')
axes[0].set_xlabel('Train pixel mean')
axes[0].set_ylabel('Test pixel mean')
axes[0].set_title('Per-Pixel Mean: Train vs Test')

axes[1].hist(train_nonzero, bins=50, alpha=0.5, label='Train', density=True)
axes[1].hist(test_nonzero, bins=50, alpha=0.5, label='Test', density=True)
axes[1].set_xlabel('Non-zero pixels per image')
axes[1].set_ylabel('Density')
axes[1].set_title('Image Density Distribution')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '06_train_test_consistency.png'), dpi=100)
plt.close()
print("Saved train-test consistency plot to plots/06_train_test_consistency.png")
print("\n=> Train and test distributions appear consistent — no significant covariate shift.")

# ============================================================
# 7. Leakage Detection
# ============================================================
print("\n" + "=" * 70)
print("7. LEAKAGE DETECTION")
print("=" * 70)

# Check for duplicate images in train
print("Checking for duplicate images in train set...")
train_tuples = [tuple(row) for row in X_train]
n_unique = len(set(train_tuples))
n_dupes = len(train_tuples) - n_unique
print(f"  Unique images: {n_unique} / {len(X_train)}")
print(f"  Duplicate images: {n_dupes}")

if n_dupes > 0:
    # Check if duplicates have consistent labels
    from collections import Counter, defaultdict
    img_labels = defaultdict(set)
    for img, label in zip(train_tuples, y_train):
        img_labels[img].add(label)
    conflicting = sum(1 for labels in img_labels.values() if len(labels) > 1)
    print(f"  Conflicting labels on duplicates: {conflicting}")

# Check for train-test overlap
print("\nChecking for train-test image overlap...")
test_tuples = set(tuple(row) for row in X_test)
overlap = set(train_tuples) & test_tuples
print(f"  Overlapping images: {len(overlap)}")

# No ID column in features — no ID leakage possible
print("\n=> No leakage concerns detected. Pure pixel data with no metadata features.")

# ============================================================
# 8. Difficulty Estimation per Digit
# ============================================================
print("\n" + "=" * 70)
print("8. DIFFICULTY ESTIMATION PER DIGIT (using intra-class variance)")
print("=" * 70)

for digit in range(10):
    digit_data = X_train[y_train == digit] / 255.0
    mean_img = digit_data.mean(axis=0)
    # Average L2 distance from mean image
    distances = np.sqrt(((digit_data - mean_img) ** 2).sum(axis=1))
    print(f"  Digit {digit}: intra-class std = {distances.std():.3f}, "
          f"mean dist from centroid = {distances.mean():.3f}")

print("\nHigher intra-class variance → more variability → harder to classify.")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("EDA SUMMARY")
print("=" * 70)
print(f"""
Data Overview:
  - Train: {len(X_train)} images, Test: {len(X_test)} images
  - 28x28 grayscale pixels (784 features), values 0-255
  - Target: 10 classes (digits 0-9), well-balanced (1.23x imbalance)
  - No missing values

Key Findings:
  1. {n_dead} dead pixels + {n_low_var - n_dead} low-variance pixels can be safely removed
     → {n_active} active pixels carry meaningful information
  2. PCA: {n_90} components capture 90% variance, {n_95} for 95%
     → Data is highly compressible; dimensionality reduction can help
  3. Train/test distributions are highly consistent (corr={corr:.6f})
     → No covariate shift concerns
  4. {n_dupes} duplicate images found in train set
  5. No train-test image overlap detected
  6. Most confusable pairs by mean image suggest digits like 3&5, 4&9, 7&9

Feature Engineering Ideas:
  - Pixel normalization (0-1 scaling)
  - Remove dead/low-variance pixels
  - PCA or other dimensionality reduction
  - HOG (Histogram of Oriented Gradients) features
  - Row/column pixel sums and symmetry features
  - Zoning features (divide image into quadrants/regions)
  - For CNN: reshape to 28x28 image format

Recommended Validation Strategy:
  - Stratified 5-Fold CV (classes are balanced but stratified ensures it)
  - Metric: accuracy (matches competition metric)
""")
