"""Analyze occurrence_id structure (strokes per signature)."""
import pandas as pd
import numpy as np

data_dir = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
train = pd.read_csv(f"{data_dir}/train.csv")
test = pd.read_csv(f"{data_dir}/test.csv")

print("=" * 60)
print("OCCURRENCE ANALYSIS - TRAIN")
print("=" * 60)
occ_per_sig = train.groupby('signature_id')['occurrence_id'].nunique()
print(f"Unique occurrences per signature distribution:")
print(occ_per_sig.value_counts().sort_index())

# Show examples of multi-occurrence signatures
multi_occ = occ_per_sig[occ_per_sig > 1]
if len(multi_occ) > 0:
    print(f"\nSignatures with multiple occurrences: {len(multi_occ)}")
    for sig_id in multi_occ.index[:5]:
        sig_data = train[train['signature_id'] == sig_id]
        for occ in sig_data['occurrence_id'].unique():
            occ_data = sig_data[sig_data['occurrence_id'] == occ]
            print(f"  Sig {sig_id}, Occ {occ}: {len(occ_data)} points, time {occ_data['time'].min()}-{occ_data['time'].max()}")
else:
    print("All signatures have exactly 1 occurrence")

print("\n" + "=" * 60)
print("OCCURRENCE ANALYSIS - TEST")
print("=" * 60)
occ_per_sig_test = test.groupby('signature_id')['occurrence_id'].nunique()
print(f"Unique occurrences per signature distribution:")
print(occ_per_sig_test.value_counts().sort_index())

# Show examples
for sig_id in test['signature_id'].unique()[:5]:
    sig_data = test[test['signature_id'] == sig_id]
    for occ in sig_data['occurrence_id'].unique():
        occ_data = sig_data[sig_data['occurrence_id'] == occ]
        print(f"  Sig {sig_id}, Occ {occ}: {len(occ_data)} points, time {occ_data['time'].min()}-{occ_data['time'].max()}")

# Check occurrence time continuity
print("\n" + "=" * 60)
print("TIME CONTINUITY ACROSS OCCURRENCES (first multi-occ test sig)")
print("=" * 60)
for sig_id in test['signature_id'].unique()[:20]:
    sig_data = test[test['signature_id'] == sig_id]
    n_occ = sig_data['occurrence_id'].nunique()
    if n_occ > 1:
        print(f"\nSig {sig_id} ({n_occ} occurrences):")
        for occ in sorted(sig_data['occurrence_id'].unique()):
            occ_data = sig_data[sig_data['occurrence_id'] == occ]
            print(f"  Occ {occ}: time {occ_data['time'].min()}-{occ_data['time'].max()}, {len(occ_data)} points")
        break

# Writer analysis
print("\n" + "=" * 60)
print("WRITER ANALYSIS")
print("=" * 60)
print(f"\nTrain: {train['writer_id'].nunique()} writers, {train['signature_id'].nunique()} signatures")
print(f"Test: {test['writer_id'].nunique()} writers, {test['signature_id'].nunique()} signatures")

sigs_per_writer_train = train.groupby('writer_id')['signature_id'].nunique()
sigs_per_writer_test = test.groupby('writer_id')['signature_id'].nunique()
print(f"\nSignatures per writer (train): {sigs_per_writer_train.describe()}")
print(f"\nSignatures per writer (test): {sigs_per_writer_test.describe()}")

# Check image sizes
from PIL import Image
import os
img_dir = f"{data_dir}/images_stroke/images"
sizes = []
for img_name in sorted(os.listdir(img_dir))[:20]:
    img = Image.open(f"{img_dir}/{img_name}")
    sizes.append((img_name, img.size))
    print(f"  {img_name}: {img.size[0]}x{img.size[1]}")

# Check a few more
for img_name in ['0606.jpg', '0607.jpg', '1081.jpg']:
    img = Image.open(f"{img_dir}/{img_name}")
    print(f"  {img_name}: {img.size[0]}x{img.size[1]}")
