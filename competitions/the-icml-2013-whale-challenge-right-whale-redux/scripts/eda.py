"""EDA for Right Whale Upcall Detection — audio analysis."""
import os
import numpy as np
import soundfile as sf
import librosa

data_dir = "competitions/the-icml-2013-whale-challenge-right-whale-redux/data"
train_dir = f"{data_dir}/train/train2"

# ============================================================
# 1. DATASET OVERVIEW
# ============================================================
print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

files = os.listdir(train_dir)
positives = [f for f in files if f.endswith("_1.aif")]
negatives = [f for f in files if f.endswith("_0.aif")]
print(f"Total train files: {len(files)}")
print(f"  Positive (whale): {len(positives)} ({100*len(positives)/len(files):.1f}%)")
print(f"  Negative (no whale): {len(negatives)} ({100*len(negatives)/len(files):.1f}%)")
print(f"  Imbalance ratio: {len(negatives)/len(positives):.1f}x")

test_dir = f"{data_dir}/test/test2"
test_files = os.listdir(test_dir)
print(f"\nTest files: {len(test_files)}")

# ============================================================
# 2. AUDIO PROPERTIES
# ============================================================
print("\n" + "=" * 60)
print("2. AUDIO PROPERTIES")
print("=" * 60)

# Sample a few files
sample_pos = sorted(positives)[:5]
sample_neg = sorted(negatives)[:5]

for label, samples in [("POSITIVE", sample_pos), ("NEGATIVE", sample_neg)]:
    print(f"\n  {label} samples:")
    for fname in samples:
        data, sr = sf.read(f"{train_dir}/{fname}")
        print(f"    {fname}: sr={sr}, len={len(data)}, range=[{data.min():.4f}, {data.max():.4f}], rms={np.sqrt(np.mean(data**2)):.6f}")

# ============================================================
# 3. WAVEFORM STATISTICS (positive vs negative)
# ============================================================
print("\n" + "=" * 60)
print("3. WAVEFORM STATISTICS (sample of 500 per class)")
print("=" * 60)

np.random.seed(42)
n_sample = 500

pos_sample = np.random.choice(positives, min(n_sample, len(positives)), replace=False)
neg_sample = np.random.choice(negatives, min(n_sample, len(negatives)), replace=False)

pos_stats = {"rms": [], "max_abs": [], "zcr": [], "energy_low": [], "energy_high": []}
neg_stats = {"rms": [], "max_abs": [], "zcr": [], "energy_low": [], "energy_high": []}

for label, samples, stats in [("positive", pos_sample, pos_stats), ("negative", neg_sample, neg_stats)]:
    for fname in samples:
        data, sr = sf.read(f"{train_dir}/{fname}")
        stats["rms"].append(np.sqrt(np.mean(data**2)))
        stats["max_abs"].append(np.abs(data).max())
        stats["zcr"].append(np.sum(np.diff(np.sign(data)) != 0) / len(data))

        # Frequency domain analysis
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1/sr)
        power = np.abs(fft) ** 2

        # Whale upcalls are in 50-200 Hz range
        low_mask = (freqs >= 50) & (freqs <= 250)
        high_mask = (freqs > 250) & (freqs <= 1000)
        stats["energy_low"].append(power[low_mask].sum())
        stats["energy_high"].append(power[high_mask].sum())

for label, stats in [("POSITIVE", pos_stats), ("NEGATIVE", neg_stats)]:
    print(f"\n  {label}:")
    for key, vals in stats.items():
        vals = np.array(vals)
        print(f"    {key:12s}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
              f"min={vals.min():.6f}, max={vals.max():.6f}")

# Separability
print("\n  SEPARABILITY (mean positive / mean negative):")
for key in pos_stats:
    p_mean = np.mean(pos_stats[key])
    n_mean = np.mean(neg_stats[key])
    ratio = p_mean / n_mean if n_mean > 0 else float("inf")
    print(f"    {key:12s}: ratio = {ratio:.3f}")

# ============================================================
# 4. SPECTROGRAM ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("4. SPECTROGRAM / MEL FEATURES")
print("=" * 60)

# Compute mel spectrogram features for a sample
n_mels = 64
n_fft = 256
hop_length = 64

pos_mels = []
neg_mels = []

for fname in pos_sample[:200]:
    data, sr = sf.read(f"{train_dir}/{fname}")
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    pos_mels.append(mel_db.mean(axis=1))  # average over time

for fname in neg_sample[:200]:
    data, sr = sf.read(f"{train_dir}/{fname}")
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    neg_mels.append(mel_db.mean(axis=1))

pos_mels = np.array(pos_mels)
neg_mels = np.array(neg_mels)

print(f"  Mel spectrogram shape per clip: ({n_mels}, time_frames)")
print(f"  Mean mel profile shape: ({n_mels},)")

# Find most discriminative mel bands
diff = pos_mels.mean(axis=0) - neg_mels.mean(axis=0)
mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
print(f"\n  Top 10 most discriminative mel bands (positive - negative dB):")
top_bands = np.argsort(np.abs(diff))[::-1][:10]
for idx in top_bands:
    print(f"    Band {idx:2d} ({mel_freqs[idx]:6.1f} Hz): diff = {diff[idx]:+.2f} dB")

# ============================================================
# 5. MFCC ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("5. MFCC FEATURES")
print("=" * 60)

n_mfcc = 20
pos_mfccs = []
neg_mfccs = []

for fname in pos_sample[:200]:
    data, sr = sf.read(f"{train_dir}/{fname}")
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    pos_mfccs.append(mfcc.mean(axis=1))

for fname in neg_sample[:200]:
    data, sr = sf.read(f"{train_dir}/{fname}")
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    neg_mfccs.append(mfcc.mean(axis=1))

pos_mfccs = np.array(pos_mfccs)
neg_mfccs = np.array(neg_mfccs)

print(f"  MFCC shape: ({n_mfcc},)")
mfcc_diff = pos_mfccs.mean(axis=0) - neg_mfccs.mean(axis=0)
print(f"\n  MFCC discriminability (positive - negative):")
for i in range(n_mfcc):
    bar = "#" * int(abs(mfcc_diff[i]) * 5)
    sign = "+" if mfcc_diff[i] > 0 else "-"
    print(f"    MFCC {i:2d}: {sign}{abs(mfcc_diff[i]):6.3f} {bar}")

# ============================================================
# 6. TEMPORAL STRUCTURE OF FILENAMES
# ============================================================
print("\n" + "=" * 60)
print("6. TEMPORAL STRUCTURE")
print("=" * 60)

# Filenames contain date/time info
dates = set()
for f in files:
    date = f.split("_")[0]
    dates.add(date)
dates = sorted(dates)
print(f"  Date range: {dates[0]} to {dates[-1]}")
print(f"  Unique dates: {len(dates)}")
for d in dates:
    n_pos = sum(1 for f in positives if f.startswith(d))
    n_neg = sum(1 for f in negatives if f.startswith(d))
    pct = 100 * n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
    print(f"    {d}: {n_pos + n_neg:6d} clips ({n_pos:5d} pos, {pct:.1f}%)")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
