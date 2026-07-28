"""Verify exact metric computation by matching against leaderboard scores."""
import pandas as pd
import numpy as np

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
sub_example = pd.read_csv(f"{DATA_DIR}/submission_example.csv")  # All 0.5

# The submission_example.csv has all x=0.5, y=0.5
# Let's compute various metrics and see which matches a known LB score

# Metrics on TRAIN data (using per-sig mean as "average benchmark")
per_sig_mean = train.groupby('signature_id').agg(mean_x=('x', 'mean'), mean_y=('y', 'mean')).reset_index()
merged = train.merge(per_sig_mean, on='signature_id')

err_x = (merged['x'] - merged['mean_x']).values
err_y = (merged['y'] - merged['mean_y']).values
N = len(err_x)

# Method A: sqrt(mean(err_x^2 + err_y^2))  -- Euclidean RMSE
method_a = np.sqrt(np.mean(err_x**2 + err_y**2))

# Method B: sqrt(mean of all errors^2)  -- Flat RMSE (x and y in one vector)
all_errors = np.concatenate([err_x, err_y])
method_b = np.sqrt(np.mean(all_errors**2))

# Method C: (RMSE_x + RMSE_y) / 2  -- Average of per-column RMSE
rmse_x = np.sqrt(np.mean(err_x**2))
rmse_y = np.sqrt(np.mean(err_y**2))
method_c = (rmse_x + rmse_y) / 2

# Method D: RMSE per column, then average, weighted by N
method_d = np.sqrt((np.sum(err_x**2) + np.sum(err_y**2)) / (2 * N))

print("=" * 60)
print("METRIC COMPUTATION ON TRAIN (per-sig mean prediction)")
print("=" * 60)
print(f"Method A: sqrt(mean(err_x^2 + err_y^2)) = {method_a:.5f}")
print(f"Method B: sqrt(mean(concat([err_x, err_y])^2)) = {method_b:.5f}")
print(f"Method C: (RMSE_x + RMSE_y) / 2 = {method_c:.5f}")
print(f"Method D: sqrt((sum_x^2 + sum_y^2) / (2N)) = {method_d:.5f}")
print(f"  RMSE_x = {rmse_x:.5f}")
print(f"  RMSE_y = {rmse_y:.5f}")
print(f"\nMethods B and D are identical: {method_b:.5f} == {method_d:.5f}")
print(f"Method A = sqrt(2) * Method D: {method_a:.5f} == {np.sqrt(2) * method_d:.5f}")

# The LB average benchmark is 0.25258
# Method D on train = 0.249 -- very close!
print(f"\nLB 'Average Benchmark' = 0.25258")
print(f"Method D (train, per-sig mean) = {method_d:.5f}")
print(f"Difference: {abs(0.25258 - method_d):.5f}")

# Now compute the actual skeleton centroid prediction (what the MATLAB code does)
print("\n" + "=" * 60)
print("SKELETON CENTROID BENCHMARK ON TRAIN")
print("=" * 60)

from PIL import Image
from skimage import morphology

err_x_skel = []
err_y_skel = []

for sig_id in train['signature_id'].unique():
    sig = train[train['signature_id'] == sig_id].sort_values('time')

    img = np.array(Image.open(f"{DATA_DIR}/images_stroke/images/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < 0.9  # Match MATLAB: IM2BW(image, 0.9) means threshold at 0.9
    skeleton = morphology.skeletonize(binary)
    rows, cols = np.where(skeleton)

    if len(rows) == 0:
        err_x_skel.extend([sig['x'].values - 0.5])
        err_y_skel.extend([sig['y'].values - 0.5])
        continue

    # Skeleton centroid, normalized
    cx = (np.mean(cols) - cols.min()) / max(cols.max() - cols.min(), 1)
    cy = (np.mean(rows) - rows.min()) / max(rows.max() - rows.min(), 1)

    err_x_skel.append(sig['x'].values - cx)
    err_y_skel.append(sig['y'].values - cy)

err_x_all = np.concatenate(err_x_skel)
err_y_all = np.concatenate(err_y_skel)

skel_method_d = np.sqrt((np.sum(err_x_all**2) + np.sum(err_y_all**2)) / (2 * len(err_x_all)))
print(f"Skeleton centroid (Method D): {skel_method_d:.5f}")
print(f"LB average benchmark: 0.25258")

# Also test: what about just using true mean of the trajectory?
# This would give us the lower bound for constant predictions
mean_method_d = method_d
print(f"\nTrue mean (oracle constant): {mean_method_d:.5f}")
print(f"The LB score 0.25258 is between skeleton centroid ({skel_method_d:.5f}) and true mean ({mean_method_d:.5f})")

# ============================================================
# Now re-evaluate our baseline strategies with the correct metric
# ============================================================
print("\n" + "=" * 60)
print("RE-EVALUATING BASELINES WITH COLUMN-WISE RMSE (Method D)")
print("=" * 60)

from scipy.interpolate import CubicSpline

def normalize_coords(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if x.max() != x.min():
        x = (x - x.min()) / (x.max() - x.min())
    else:
        x = np.zeros_like(x)
    if y.max() != y.min():
        y = (y - y.min()) / (y.max() - y.min())
    else:
        y = np.zeros_like(y)
    return x, y

def resample(x, y, n):
    if len(x) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n)
    return np.clip(CubicSpline(t, x)(t_new), 0, 1), np.clip(CubicSpline(t, y)(t_new), 0, 1)

# XY scan
all_err_x, all_err_y = [], []
for sig_id in train['signature_id'].unique()[:200]:
    sig = train[train['signature_id'] == sig_id].sort_values('time')
    true_x, true_y = sig['x'].values, sig['y'].values
    n = len(sig)

    img = np.array(Image.open(f"{DATA_DIR}/images_stroke/images/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < 0.9
    skeleton = morphology.skeletonize(binary)
    rows, cols = np.where(skeleton)
    if len(rows) < 2:
        all_err_x.append(true_x - 0.5)
        all_err_y.append(true_y - 0.5)
        continue

    # XY scan
    idx = np.lexsort((rows, cols))
    px, py = cols[idx], rows[idx]
    nx, ny = normalize_coords(px, py)
    rx, ry = resample(nx, ny, n)

    # Try both directions
    err_fwd_x = true_x - rx
    err_fwd_y = true_y - ry
    rmse_fwd = np.sqrt((np.sum(err_fwd_x**2) + np.sum(err_fwd_y**2)) / (2 * n))

    err_rev_x = true_x - rx[::-1]
    err_rev_y = true_y - ry[::-1]
    rmse_rev = np.sqrt((np.sum(err_rev_x**2) + np.sum(err_rev_y**2)) / (2 * n))

    if rmse_fwd < rmse_rev:
        all_err_x.append(err_fwd_x)
        all_err_y.append(err_fwd_y)
    else:
        all_err_x.append(err_rev_x)
        all_err_y.append(err_rev_y)

ex = np.concatenate(all_err_x)
ey = np.concatenate(all_err_y)
xy_rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
print(f"XY scan (200 sigs, correct metric): {xy_rmse:.5f}")
print(f"LB best: 0.24449")
print(f"LB avg benchmark: 0.25258")
