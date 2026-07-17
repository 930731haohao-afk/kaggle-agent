"""Initial inspection for global-wheat-detection.

Examines data files, image properties, bounding box statistics, and submission format.
"""
import pandas as pd
import numpy as np
import os
from PIL import Image
import json

data_dir = "competitions/global-wheat-detection/data"

print("=" * 70)
print("GLOBAL WHEAT DETECTION — DATA INSPECTION")
print("=" * 70)

# ============================================================
# 1. TRAIN CSV ANALYSIS
# ============================================================
print("\n1. TRAIN CSV ANALYSIS")
print("-" * 40)

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
print(f"Shape: {train.shape}")
print(f"Columns: {list(train.columns)}")
print(f"Dtypes:\n{train.dtypes}")
print(f"\nNull counts:\n{train.isnull().sum()}")

print(f"\nUnique images: {train['image_id'].nunique()}")
print(f"Total bounding boxes: {len(train)}")
print(f"Boxes per image: mean={len(train)/train['image_id'].nunique():.1f}")

# Image dimensions
print(f"\nImage width:  unique={train['width'].unique()}")
print(f"Image height: unique={train['height'].unique()}")

# Source analysis
print(f"\nSources: {train['source'].nunique()} unique")
source_counts = train.groupby('source')['image_id'].nunique()
print("Images per source:")
for src, cnt in source_counts.sort_values(ascending=False).items():
    boxes = len(train[train['source'] == src])
    print(f"  {src}: {cnt} images, {boxes} boxes ({boxes/cnt:.1f} boxes/img)")

# ============================================================
# 2. BOUNDING BOX ANALYSIS
# ============================================================
print("\n2. BOUNDING BOX ANALYSIS")
print("-" * 40)

# Parse bbox strings
train['bbox_parsed'] = train['bbox'].apply(lambda x: json.loads(x))
train['x'] = train['bbox_parsed'].apply(lambda b: b[0])
train['y'] = train['bbox_parsed'].apply(lambda b: b[1])
train['w'] = train['bbox_parsed'].apply(lambda b: b[2])
train['h'] = train['bbox_parsed'].apply(lambda b: b[3])

for col in ['x', 'y', 'w', 'h']:
    vals = train[col]
    print(f"\n{col}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
          f"min={vals.min():.1f}, max={vals.max():.1f}")

# Box area
train['area'] = train['w'] * train['h']
print(f"\nBox area: mean={train['area'].mean():.0f}, median={train['area'].median():.0f}, "
      f"std={train['area'].std():.0f}")
print(f"  min={train['area'].min():.0f}, max={train['area'].max():.0f}")

# Box aspect ratio
train['aspect_ratio'] = train['w'] / train['h'].clip(lower=1)
print(f"\nAspect ratio (w/h): mean={train['aspect_ratio'].mean():.2f}, "
      f"median={train['aspect_ratio'].median():.2f}")

# Boxes per image distribution
boxes_per_img = train.groupby('image_id').size()
print(f"\nBoxes per image distribution:")
print(f"  mean={boxes_per_img.mean():.1f}, std={boxes_per_img.std():.1f}")
print(f"  min={boxes_per_img.min()}, max={boxes_per_img.max()}")
print(f"  25%={boxes_per_img.quantile(0.25):.0f}, 50%={boxes_per_img.median():.0f}, "
      f"75%={boxes_per_img.quantile(0.75):.0f}")

# Check for boxes extending beyond image boundary
oob = ((train['x'] + train['w'] > train['width']) |
       (train['y'] + train['h'] > train['height']) |
       (train['x'] < 0) | (train['y'] < 0))
print(f"\nBoxes out of bounds: {oob.sum()} ({oob.sum()/len(train)*100:.1f}%)")

# Very small boxes
tiny = (train['area'] < 100)
print(f"Very small boxes (area < 100): {tiny.sum()} ({tiny.sum()/len(train)*100:.1f}%)")

# ============================================================
# 3. IMAGE ANALYSIS (sample)
# ============================================================
print("\n3. IMAGE ANALYSIS (sample)")
print("-" * 40)

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

train_images = os.listdir(train_dir)
test_images = os.listdir(test_dir)
print(f"Train images: {len(train_images)}")
print(f"Test images: {len(test_images)}")

# Check a few image properties
sample_imgs = train_images[:10]
for img_name in sample_imgs[:3]:
    img = Image.open(os.path.join(train_dir, img_name))
    print(f"  {img_name}: size={img.size}, mode={img.mode}")

# Check test images
for img_name in test_images[:3]:
    img = Image.open(os.path.join(test_dir, img_name))
    print(f"  {img_name}: size={img.size}, mode={img.mode}")

# ============================================================
# 4. SUBMISSION FORMAT
# ============================================================
print("\n4. SUBMISSION FORMAT")
print("-" * 40)

sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
print(f"Shape: {sub.shape}")
print(f"Columns: {list(sub.columns)}")
print(f"Sample:\n{sub.head()}")
print(f"\nFormat: 'confidence x y w h' per detection, space-separated")
print(f"All test images listed: {sorted(sub['image_id'].tolist())}")

# ============================================================
# 5. TRAIN-TEST OVERLAP CHECK
# ============================================================
print("\n5. TRAIN-TEST OVERLAP CHECK")
print("-" * 40)

train_ids = set(train['image_id'].unique())
test_ids = set(sub['image_id'].unique())
overlap = train_ids & test_ids
print(f"Train image IDs: {len(train_ids)}")
print(f"Test image IDs: {len(test_ids)}")
print(f"Overlap: {len(overlap)}")

print(f"\n{'='*70}")
print("INSPECTION COMPLETE")
print(f"{'='*70}")
