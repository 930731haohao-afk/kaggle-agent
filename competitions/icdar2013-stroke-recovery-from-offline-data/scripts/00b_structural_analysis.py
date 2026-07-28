"""Deeper structural analysis of the ICDAR stroke recovery data."""
import pandas as pd
import numpy as np

data_dir = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"

train = pd.read_csv(f"{data_dir}/train.csv")
test = pd.read_csv(f"{data_dir}/test.csv")
unnorm = pd.read_csv(f"{data_dir}/unnormalized_data.csv")

print("=" * 60)
print("TRAIN STRUCTURE")
print("=" * 60)
print(f"Unique signatures: {train['signature_id'].nunique()} (range: {train['signature_id'].min()}-{train['signature_id'].max()})")
print(f"Unique writers: {train['writer_id'].nunique()} (range: {train['writer_id'].min()}-{train['writer_id'].max()})")
print(f"Unique occurrences: {train['occurrence_id'].nunique()} (range: {train['occurrence_id'].min()}-{train['occurrence_id'].max()})")
print(f"Total points: {len(train)}")

# Points per signature
pts_per_sig = train.groupby('signature_id').size()
print(f"\nPoints per signature:")
print(f"  Mean: {pts_per_sig.mean():.1f}")
print(f"  Median: {pts_per_sig.median():.1f}")
print(f"  Min: {pts_per_sig.min()}")
print(f"  Max: {pts_per_sig.max()}")
print(f"  Std: {pts_per_sig.std():.1f}")

# Signatures per writer
sigs_per_writer = train.groupby('writer_id')['signature_id'].nunique()
print(f"\nSignatures per writer:")
print(f"  Mean: {sigs_per_writer.mean():.1f}")
print(f"  Median: {sigs_per_writer.median():.1f}")
print(f"  Min: {sigs_per_writer.min()}")
print(f"  Max: {sigs_per_writer.max()}")

# Occurrences per signature
occ_per_sig = train.groupby('signature_id')['occurrence_id'].nunique()
print(f"\nOccurrences per signature (=strokes?):")
print(f"  Mean: {occ_per_sig.mean():.1f}")
print(f"  Median: {occ_per_sig.median():.1f}")
print(f"  Min: {occ_per_sig.min()}")
print(f"  Max: {occ_per_sig.max()}")

# Time analysis
print(f"\nTime column:")
print(f"  Range: {train['time'].min()}-{train['time'].max()}")
time_per_sig = train.groupby('signature_id')['time'].max()
print(f"  Max time per signature: mean={time_per_sig.mean():.1f}, median={time_per_sig.median():.1f}, min={time_per_sig.min()}, max={time_per_sig.max()}")

# Check if time resets per occurrence
print(f"\n  Time resets per occurrence?")
sample_sig = train[train['signature_id'] == 1]
for occ in sample_sig['occurrence_id'].unique()[:5]:
    occ_data = sample_sig[sample_sig['occurrence_id'] == occ]
    print(f"    Occurrence {occ}: time {occ_data['time'].min()}-{occ_data['time'].max()}, {len(occ_data)} points")

print("\n" + "=" * 60)
print("TEST STRUCTURE")
print("=" * 60)
print(f"Unique signatures: {test['signature_id'].nunique()} (range: {test['signature_id'].min()}-{test['signature_id'].max()})")
print(f"Unique writers: {test['writer_id'].nunique()} (range: {test['writer_id'].min()}-{test['writer_id'].max()})")
print(f"Unique occurrences: {test['occurrence_id'].nunique()} (range: {test['occurrence_id'].min()}-{test['occurrence_id'].max()})")
print(f"Total points: {len(test)}")

pts_per_sig_test = test.groupby('signature_id').size()
print(f"\nPoints per signature:")
print(f"  Mean: {pts_per_sig_test.mean():.1f}")
print(f"  Median: {pts_per_sig_test.median():.1f}")
print(f"  Min: {pts_per_sig_test.min()}")
print(f"  Max: {pts_per_sig_test.max()}")

# Check overlap
print("\n" + "=" * 60)
print("TRAIN/TEST OVERLAP")
print("=" * 60)
train_sigs = set(train['signature_id'].unique())
test_sigs = set(test['signature_id'].unique())
print(f"Train signatures: {len(train_sigs)} ({min(train_sigs)}-{max(train_sigs)})")
print(f"Test signatures: {len(test_sigs)} ({min(test_sigs)}-{max(test_sigs)})")
print(f"Overlap: {len(train_sigs & test_sigs)}")

train_writers = set(train['writer_id'].unique())
test_writers = set(test['writer_id'].unique())
print(f"Train writers: {len(train_writers)} ({min(train_writers)}-{max(train_writers)})")
print(f"Test writers: {len(test_writers)} ({min(test_writers)}-{max(test_writers)})")
print(f"Writer overlap: {len(train_writers & test_writers)}")

# Images
import os
img_dir = f"{data_dir}/images_stroke/images"
images = sorted(os.listdir(img_dir))
print(f"\nTotal images: {len(images)}")
print(f"Image range: {images[0]} to {images[-1]}")
# Image IDs as numbers
img_ids = [int(f.split('.')[0]) for f in images]
print(f"Image ID range: {min(img_ids)}-{max(img_ids)}")
print(f"Total signatures (train+test): {train['signature_id'].nunique() + test['signature_id'].nunique()}")
print(f"Image count matches total signatures: {len(images) == train['signature_id'].nunique() + test['signature_id'].nunique()}")

# Verify normalization
print("\n" + "=" * 60)
print("NORMALIZATION CHECK")
print("=" * 60)
print(f"Train x range: [{train['x'].min()}, {train['x'].max()}]")
print(f"Train y range: [{train['y'].min()}, {train['y'].max()}]")
print(f"Unnorm x range: [{unnorm['x'].min()}, {unnorm['x'].max()}]")
print(f"Unnorm y range: [{unnorm['y'].min()}, {unnorm['y'].max()}]")

# Check if normalization is per-signature or global
print("\nPer-signature normalization check (first 5 sigs):")
for sig_id in range(1, 6):
    train_sig = train[train['signature_id'] == sig_id]
    unnorm_sig = unnorm[unnorm['signature_id'] == sig_id]
    print(f"  Sig {sig_id}: norm x=[{train_sig['x'].min():.3f}, {train_sig['x'].max():.3f}], "
          f"unnorm x=[{unnorm_sig['x'].min()}, {unnorm_sig['x'].max()}], "
          f"norm y=[{train_sig['y'].min():.3f}, {train_sig['y'].max():.3f}], "
          f"unnorm y=[{unnorm_sig['y'].min()}, {unnorm_sig['y'].max()}]")
