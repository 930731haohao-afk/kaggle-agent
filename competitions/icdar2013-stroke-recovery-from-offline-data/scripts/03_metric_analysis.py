"""
Understand the metric and compute the average benchmark score on train data.
Also analyze what makes this problem hard.
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology, color

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
IMG_DIR = f"{DATA_DIR}/images_stroke/images"

train = pd.read_csv(f"{DATA_DIR}/train.csv")

# ============================================================
# METRIC ANALYSIS
# ============================================================

print("=" * 60)
print("METRIC COMPUTATION METHODS")
print("=" * 60)

# Method 1: Overall RMSE (sqrt of mean of squared errors across all points)
# For constant prediction of (0.5, 0.5):
const_x = 0.5
const_y = 0.5
sq_err = (train['x'] - const_x)**2 + (train['y'] - const_y)**2
overall_rmse = np.sqrt(np.mean(sq_err))
print(f"Constant (0.5, 0.5) - Overall RMSE: {overall_rmse:.5f}")

# For per-signature mean prediction:
per_sig_preds = train.groupby('signature_id').agg(
    mean_x=('x', 'mean'),
    mean_y=('y', 'mean'),
).reset_index()

merged = train.merge(per_sig_preds, on='signature_id')
sq_err_mean = (merged['x'] - merged['mean_x'])**2 + (merged['y'] - merged['mean_y'])**2
mean_pred_rmse = np.sqrt(np.mean(sq_err_mean))
print(f"Per-sig mean prediction - Overall RMSE: {mean_pred_rmse:.5f}")

# Method 2: Per-signature RMSE, then average
per_sig_rmse = []
for sig_id in train['signature_id'].unique():
    sig = train[train['signature_id'] == sig_id]
    mx, my = sig['x'].mean(), sig['y'].mean()
    sq_err = (sig['x'] - mx)**2 + (sig['y'] - my)**2
    per_sig_rmse.append(np.sqrt(np.mean(sq_err)))
print(f"Per-sig mean prediction - Avg per-sig RMSE: {np.mean(per_sig_rmse):.5f}")

# Method 3: Separate RMSE for x and y, then average
rmse_x = np.sqrt(np.mean((train['x'] - 0.5)**2))
rmse_y = np.sqrt(np.mean((train['y'] - 0.5)**2))
print(f"Constant (0.5, 0.5) - Mean(RMSE_x, RMSE_y): {(rmse_x + rmse_y)/2:.5f}")

# Method 4: RMSE on x, y separately combined
# sqrt(mean((pred_x - true_x)^2)) combined with sqrt(mean((pred_y - true_y)^2))
# Mean of the two
print(f"  RMSE_x (const 0.5): {rmse_x:.5f}")
print(f"  RMSE_y (const 0.5): {rmse_y:.5f}")

# ============================================================
# Average benchmark (skeleton centroid)
# ============================================================
print("\n" + "=" * 60)
print("AVERAGE BENCHMARK (Skeleton Centroid)")
print("=" * 60)

centroid_rmses = []
for sig_id in train['signature_id'].unique()[:100]:
    sig = train[train['signature_id'] == sig_id].sort_values('time')

    # Load and skeletonize
    img = np.array(Image.open(f"{IMG_DIR}/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < 0.85
    skeleton = morphology.skeletonize(binary)
    rows, cols = np.where(skeleton)

    if len(rows) == 0:
        centroid_rmses.append(0.5)
        continue

    # Normalize skeleton centroid
    cx = (np.mean(cols) - cols.min()) / max(cols.max() - cols.min(), 1)
    cy = (np.mean(rows) - rows.min()) / max(rows.max() - rows.min(), 1)

    # RMSE
    sq_err = (sig['x'] - cx)**2 + (sig['y'] - cy)**2
    centroid_rmses.append(np.sqrt(np.mean(sq_err)))

print(f"Skeleton centroid prediction - Avg RMSE: {np.mean(centroid_rmses):.5f}")

# ============================================================
# What if we just predict the perfect mean?
# ============================================================
print("\n" + "=" * 60)
print("ORACLE ANALYSIS")
print("=" * 60)

# If we know the true mean of each trajectory
oracle_per_sig = []
for sig_id in train['signature_id'].unique():
    sig = train[train['signature_id'] == sig_id]
    sq_err = (sig['x'] - sig['x'].mean())**2 + (sig['y'] - sig['y'].mean())**2
    oracle_per_sig.append(np.sqrt(np.mean(sq_err)))
print(f"Oracle (true mean) prediction - Avg RMSE: {np.mean(oracle_per_sig):.5f}")

# If we know the true trajectory (RMSE = 0)
print(f"Oracle (true trajectory) - RMSE: 0.00000")

# What does the 0.245 leaderboard score mean?
# Ratio of best LB to constant prediction
print(f"\nLB best (0.245) / Oracle mean ({np.mean(oracle_per_sig):.3f}) = "
      f"{0.245 / np.mean(oracle_per_sig):.2%}")
print(f"This means even the best solutions are not much better than predicting the mean!")

# ============================================================
# RMSE decomposition: How much comes from ordering vs. position?
# ============================================================
print("\n" + "=" * 60)
print("ERROR DECOMPOSITION")
print("=" * 60)

# For each signature: compute RMSE of skeleton pixels resampled randomly
# vs. skeleton pixels in correct spatial positions but wrong time order
random_rmses = []
for sig_id in train['signature_id'].unique()[:50]:
    sig = train[train['signature_id'] == sig_id].sort_values('time')
    true_x, true_y = sig['x'].values, sig['y'].values
    n = len(sig)

    # Load skeleton
    img = np.array(Image.open(f"{IMG_DIR}/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < 0.85
    skeleton = morphology.skeletonize(binary)
    rows, cols = np.where(skeleton)
    if len(rows) < 2:
        continue

    # Normalize
    nx = (cols - cols.min()) / max(cols.max() - cols.min(), 1)
    ny = (rows - rows.min()) / max(rows.max() - rows.min(), 1)

    # Random ordering (worst case for ordering)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(nx))
    rx, ry = nx[perm], ny[perm]

    # Resample
    from scipy.interpolate import CubicSpline
    t_orig = np.linspace(0, 1, len(rx))
    t_new = np.linspace(0, 1, n)
    cs_x = CubicSpline(t_orig, rx)
    cs_y = CubicSpline(t_orig, ry)
    rx_resampled = np.clip(cs_x(t_new), 0, 1)
    ry_resampled = np.clip(cs_y(t_new), 0, 1)

    rmse = np.sqrt(np.mean((rx_resampled - true_x)**2 + (ry_resampled - true_y)**2))
    random_rmses.append(rmse)

print(f"Random skeleton ordering - Avg RMSE: {np.mean(random_rmses):.5f}")
print(f"(This is the baseline if skeleton ordering is essentially random)")

# Perfect ordering (use DTW to align skeleton to trajectory)
print("\nKey insight: The main challenge is figuring out the temporal ordering")
print("Even with perfect spatial coverage, wrong ordering gives high RMSE")
