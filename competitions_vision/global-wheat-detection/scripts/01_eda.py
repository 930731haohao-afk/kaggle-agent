"""EDA for global-wheat-detection.

Analyzes bounding box distributions, source-level patterns, box clustering,
and annotation quality for wheat head detection.
"""
import pandas as pd
import numpy as np
import os
import json
from PIL import Image
from collections import Counter

data_dir = "competitions/global-wheat-detection/data"

print("=" * 70)
print("GLOBAL WHEAT DETECTION — EDA")
print("=" * 70)

# ============================================================
# 1. LOAD AND PARSE DATA
# ============================================================
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
train['bbox_parsed'] = train['bbox'].apply(lambda x: json.loads(x))
train['x'] = train['bbox_parsed'].apply(lambda b: b[0])
train['y'] = train['bbox_parsed'].apply(lambda b: b[1])
train['w'] = train['bbox_parsed'].apply(lambda b: b[2])
train['h'] = train['bbox_parsed'].apply(lambda b: b[3])
train['area'] = train['w'] * train['h']
train['cx'] = train['x'] + train['w'] / 2  # center x
train['cy'] = train['y'] + train['h'] / 2  # center y
train['aspect_ratio'] = train['w'] / train['h'].clip(lower=1)

# ============================================================
# 2. SOURCE-LEVEL ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("1. SOURCE-LEVEL ANALYSIS")
print("=" * 70)

for src in sorted(train['source'].unique()):
    s = train[train['source'] == src]
    n_imgs = s['image_id'].nunique()
    n_boxes = len(s)
    bpi = s.groupby('image_id').size()
    print(f"\n--- {src} ---")
    print(f"  Images: {n_imgs}, Boxes: {n_boxes}")
    print(f"  Boxes/img: mean={bpi.mean():.1f}, std={bpi.std():.1f}, "
          f"min={bpi.min()}, max={bpi.max()}")
    print(f"  Box width:  mean={s['w'].mean():.1f}, std={s['w'].std():.1f}")
    print(f"  Box height: mean={s['h'].mean():.1f}, std={s['h'].std():.1f}")
    print(f"  Box area:   mean={s['area'].mean():.0f}, median={s['area'].median():.0f}")
    print(f"  Aspect ratio: mean={s['aspect_ratio'].mean():.2f}")

# ============================================================
# 3. BOX SIZE DISTRIBUTION
# ============================================================
print("\n" + "=" * 70)
print("2. BOX SIZE DISTRIBUTION")
print("=" * 70)

# Size bins
size_bins = [0, 32, 64, 96, 128, 192, 256, 512, 1024]
for dim_name, dim_col in [('width', 'w'), ('height', 'h')]:
    print(f"\n{dim_name} distribution:")
    for i in range(len(size_bins) - 1):
        lo, hi = size_bins[i], size_bins[i+1]
        cnt = ((train[dim_col] >= lo) & (train[dim_col] < hi)).sum()
        pct = cnt / len(train) * 100
        bar = '#' * int(pct / 2)
        print(f"  [{lo:4d}, {hi:4d}): {cnt:6d} ({pct:5.1f}%) {bar}")

# Area distribution
print("\nArea distribution:")
area_bins = [0, 1000, 3000, 5000, 8000, 12000, 20000, 50000, 600000]
for i in range(len(area_bins) - 1):
    lo, hi = area_bins[i], area_bins[i+1]
    cnt = ((train['area'] >= lo) & (train['area'] < hi)).sum()
    pct = cnt / len(train) * 100
    bar = '#' * int(pct / 2)
    print(f"  [{lo:6d}, {hi:6d}): {cnt:6d} ({pct:5.1f}%) {bar}")

# ============================================================
# 4. SPATIAL DISTRIBUTION
# ============================================================
print("\n" + "=" * 70)
print("3. SPATIAL DISTRIBUTION OF BOXES")
print("=" * 70)

# Divide image into 4x4 grid, count box centers
grid_size = 4
grid_counts = np.zeros((grid_size, grid_size), dtype=int)
cell_w = 1024 / grid_size
cell_h = 1024 / grid_size

for _, row in train[['cx', 'cy']].iterrows():
    gi = min(int(row['cx'] / cell_w), grid_size - 1)
    gj = min(int(row['cy'] / cell_h), grid_size - 1)
    grid_counts[gj, gi] += 1

print("Box center density (4x4 grid, % of all boxes):")
total = grid_counts.sum()
for j in range(grid_size):
    row_str = "  "
    for i in range(grid_size):
        pct = grid_counts[j, i] / total * 100
        row_str += f"{pct:5.1f}% "
    print(row_str)

# Edge boxes (touching image boundary)
edge_margin = 5
left_edge = (train['x'] < edge_margin).sum()
right_edge = (train['x'] + train['w'] > 1024 - edge_margin).sum()
top_edge = (train['y'] < edge_margin).sum()
bottom_edge = (train['y'] + train['h'] > 1024 - edge_margin).sum()
total_edge = ((train['x'] < edge_margin) |
              (train['x'] + train['w'] > 1024 - edge_margin) |
              (train['y'] < edge_margin) |
              (train['y'] + train['h'] > 1024 - edge_margin)).sum()
print(f"\nEdge-touching boxes (<{edge_margin}px from boundary): {total_edge} ({total_edge/len(train)*100:.1f}%)")
print(f"  Left: {left_edge}, Right: {right_edge}, Top: {top_edge}, Bottom: {bottom_edge}")

# ============================================================
# 5. BOX OVERLAP ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("4. BOX OVERLAP ANALYSIS (sample)")
print("=" * 70)

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0

# Sample images for overlap analysis
sample_ids = train['image_id'].unique()[:100]
all_ious = []
n_overlapping = 0
n_pairs = 0

for img_id in sample_ids:
    boxes = train[train['image_id'] == img_id][['x', 'y', 'w', 'h']].values
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = compute_iou(boxes[i], boxes[j])
            n_pairs += 1
            if iou > 0:
                all_ious.append(iou)
                n_overlapping += 1

print(f"Analyzed {len(sample_ids)} images, {n_pairs} box pairs")
print(f"Overlapping pairs (IoU > 0): {n_overlapping} ({n_overlapping/max(n_pairs,1)*100:.1f}%)")
if all_ious:
    ious = np.array(all_ious)
    print(f"IoU among overlapping pairs: mean={ious.mean():.3f}, median={np.median(ious):.3f}")
    print(f"  max={ious.max():.3f}")
    high_overlap = (ious > 0.3).sum()
    print(f"  High overlap (IoU > 0.3): {high_overlap}")

# ============================================================
# 6. IMAGES WITHOUT ANNOTATIONS
# ============================================================
print("\n" + "=" * 70)
print("5. IMAGES WITHOUT ANNOTATIONS")
print("=" * 70)

train_dir = os.path.join(data_dir, "train")
all_train_files = set(f.replace('.jpg', '') for f in os.listdir(train_dir) if f.endswith('.jpg'))
annotated_ids = set(train['image_id'].unique())
unannotated = all_train_files - annotated_ids
extra_annotations = annotated_ids - all_train_files

print(f"Images in train dir: {len(all_train_files)}")
print(f"Images with annotations: {len(annotated_ids)}")
print(f"Unannotated images: {len(unannotated)}")
if unannotated:
    print(f"  Examples: {sorted(unannotated)[:10]}")
print(f"Annotations without image: {len(extra_annotations)}")

# ============================================================
# 7. ANCHOR SIZE RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("6. ANCHOR SIZE RECOMMENDATIONS")
print("=" * 70)

# k-means clustering on box sizes for anchor generation
from sklearn.cluster import KMeans

box_dims = train[['w', 'h']].values
for k in [5, 9]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(box_dims)
    centers = km.cluster_centers_
    centers = centers[centers[:, 0].argsort()]
    print(f"\nK-means anchors (k={k}):")
    for i, (aw, ah) in enumerate(centers):
        area = aw * ah
        print(f"  Anchor {i+1}: {aw:.0f} x {ah:.0f} (area={area:.0f})")

# ============================================================
# 8. TRAIN-TEST SOURCE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("7. POTENTIAL DOMAIN SHIFT")
print("=" * 70)

# Check test image visual properties (brightness, color)
test_dir = os.path.join(data_dir, "test")
test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]

print("Test image properties:")
for img_name in test_images:
    img = np.array(Image.open(os.path.join(test_dir, img_name)))
    print(f"  {img_name}: mean_rgb=({img[:,:,0].mean():.0f}, {img[:,:,1].mean():.0f}, "
          f"{img[:,:,2].mean():.0f}), std={img.std():.1f}")

print("\nTrain image properties (sample by source):")
for src in sorted(train['source'].unique()):
    src_ids = train[train['source'] == src]['image_id'].unique()[:3]
    means = []
    for img_id in src_ids:
        img_path = os.path.join(train_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            means.append(img.mean())
    if means:
        print(f"  {src}: mean_pixel={np.mean(means):.0f} (n={len(means)})")

print(f"\n{'='*70}")
print("EDA COMPLETE")
print(f"{'='*70}")
