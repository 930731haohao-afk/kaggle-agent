"""EDA for EMVIC — Eye Movement Verification and Identification."""
import numpy as np
import pandas as pd

data_dir = "competitions/emvic/data"

train = pd.read_csv(f"{data_dir}/train.csv")
y = train["class"]
X = train.drop(columns=["class"])

print(f"Train: {X.shape}, {y.nunique()} classes")

# ============================================================
# 1. TARGET ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("1. TARGET DISTRIBUTION")
print("=" * 60)
vc = y.value_counts().sort_index()
for cls, n in vc.items():
    bar = "#" * n
    print(f"  Class {cls:2d}: {n:3d} {bar}")
print(f"\nImbalance ratio: {vc.max()}/{vc.min()} = {vc.max()/vc.min():.1f}x")
print(f"Classes with <=5 samples: {(vc <= 5).sum()}")
print(f"Classes with <=10 samples: {(vc <= 10).sum()}")

# ============================================================
# 2. FEATURE STRUCTURE
# ============================================================
print("\n" + "=" * 60)
print("2. FEATURE STRUCTURE (4 channels x 2048 timepoints)")
print("=" * 60)

channels = {"lx": [], "ly": [], "rx": [], "ry": []}
for col in X.columns:
    for ch in channels:
        if col.startswith(ch) and col[len(ch):].isdigit():
            channels[ch].append(col)

for ch, cols in channels.items():
    vals = X[cols].values
    print(f"\n  {ch} ({len(cols)} features):")
    print(f"    Range: [{vals.min():.1f}, {vals.max():.1f}]")
    print(f"    Mean: {vals.mean():.2f}, Std: {vals.std():.2f}")
    # Per-sample stats
    sample_means = vals.mean(axis=1)
    sample_stds = vals.std(axis=1)
    print(f"    Per-sample mean range: [{sample_means.min():.2f}, {sample_means.max():.2f}]")
    print(f"    Per-sample std range: [{sample_stds.min():.2f}, {sample_stds.max():.2f}]")

# ============================================================
# 3. TEMPORAL PATTERNS
# ============================================================
print("\n" + "=" * 60)
print("3. TEMPORAL PATTERNS")
print("=" * 60)

# Velocity features (difference between consecutive timepoints)
for ch in ["lx", "ly", "rx", "ry"]:
    cols = channels[ch]
    vals = X[cols].values
    velocity = np.diff(vals, axis=1)
    print(f"  {ch} velocity: mean={velocity.mean():.4f}, std={velocity.std():.2f}, max_abs={np.abs(velocity).max():.1f}")

# ============================================================
# 4. CORRELATION BETWEEN CHANNELS
# ============================================================
print("\n" + "=" * 60)
print("4. CHANNEL CORRELATIONS (sample-level means)")
print("=" * 60)

channel_means = {}
for ch, cols in channels.items():
    channel_means[ch] = X[cols].values.mean(axis=1)

channel_df = pd.DataFrame(channel_means)
print(channel_df.corr().round(3))

# ============================================================
# 5. PCA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("5. PCA — VARIANCE EXPLAINED")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Full PCA (limited by n_samples)
pca = PCA(n_components=min(X.shape[0], X.shape[1]))
pca.fit(X_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
for n in [10, 20, 50, 100, 200, 300, 500]:
    if n < len(cumvar):
        print(f"  {n:3d} components: {cumvar[n-1]*100:.1f}% variance")

print(f"  Total components possible: {len(cumvar)}")
print(f"  Components for 90% var: {np.argmax(cumvar >= 0.90) + 1}")
print(f"  Components for 95% var: {np.argmax(cumvar >= 0.95) + 1}")
print(f"  Components for 99% var: {np.argmax(cumvar >= 0.99) + 1}")

# ============================================================
# 6. QUICK SEPARABILITY CHECK
# ============================================================
print("\n" + "=" * 60)
print("6. SEPARABILITY (PCA + kNN)")
print("=" * 60)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

X_pca = PCA(n_components=100).fit_transform(X_scaled)

# kNN on PCA features
knn = KNeighborsClassifier(n_neighbors=3)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_scores = cross_val_score(knn, X_pca, y, cv=skf, scoring="accuracy")
print(f"  kNN (k=3, 100 PCA): accuracy = {knn_scores.mean():.3f} (+/- {knn_scores.std():.3f})")

# Also try with more components
X_pca200 = PCA(n_components=200).fit_transform(X_scaled)
knn_scores2 = cross_val_score(KNeighborsClassifier(n_neighbors=3), X_pca200, y, cv=skf, scoring="accuracy")
print(f"  kNN (k=3, 200 PCA): accuracy = {knn_scores2.mean():.3f} (+/- {knn_scores2.std():.3f})")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
