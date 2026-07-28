"""
Fast and practical stroke recovery solution.

Key insight from metric analysis:
- LB best: 0.24449, Average benchmark: 0.25258
- Mean prediction: ~0.249 on train
- The margin is tiny — we need a fast, reliable approach

Strategy:
1. Extract skeleton pixels from image
2. Order by fast nearest-neighbor traversal
3. Smooth heavily (the key insight: heavy smoothing reduces noise from bad ordering)
4. Blend with the mean to reduce variance
5. Try forward/reverse, pick best based on smoothness
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
IMG_DIR = f"{DATA_DIR}/images_stroke/images"

train = pd.read_csv(f"{DATA_DIR}/train.csv")


def load_skeleton(sig_id, threshold=0.9):
    img = np.array(Image.open(f"{IMG_DIR}/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < threshold
    skeleton = morphology.skeletonize(binary)
    rows, cols = np.where(skeleton)
    return rows, cols, img.shape


def fast_nn_order(rows, cols):
    """Fast nearest-neighbor ordering using KD-tree."""
    from scipy.spatial import cKDTree
    n = len(rows)
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        return cols[:1].astype(float), rows[:1].astype(float)

    points = np.column_stack((cols, rows))
    tree = cKDTree(points)

    # Start from leftmost point
    start = np.argmin(cols)

    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        # Query k nearest, find first unvisited
        k = min(20, n)
        dists, idxs = tree.query(points[current], k=k)
        found = False
        for d, idx in zip(dists, idxs):
            if not visited[idx]:
                order.append(idx)
                visited[idx] = True
                current = idx
                found = True
                break
        if not found:
            # All k nearest are visited, find any unvisited
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            # Pick closest unvisited
            dists_all = np.sqrt((points[unvisited, 0] - points[current, 0])**2 +
                                (points[unvisited, 1] - points[current, 1])**2)
            nearest = unvisited[np.argmin(dists_all)]
            order.append(nearest)
            visited[nearest] = True
            current = nearest

    order = np.array(order)
    return cols[order].astype(float), rows[order].astype(float)


def normalize(x, y):
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    x = (x - x.min()) / xr if xr > 0 else np.zeros_like(x)
    y = (y - y.min()) / yr if yr > 0 else np.zeros_like(y)
    return x, y


def resample_smooth(x, y, n, sigma=0):
    if len(x) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    if sigma > 0 and len(x) > 3:
        x = gaussian_filter1d(x.astype(float), sigma=sigma)
        y = gaussian_filter1d(y.astype(float), sigma=sigma)
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n)
    rx = np.clip(CubicSpline(t, x)(t_new), 0, 1)
    ry = np.clip(CubicSpline(t, y)(t_new), 0, 1)
    return rx, ry


def smoothness(x, y):
    if len(x) < 3:
        return 0
    return np.mean(np.diff(x, 2)**2 + np.diff(y, 2)**2)


def column_rmse(px, py, tx, ty):
    ex, ey = px - tx, py - ty
    return np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))


def predict_signature_fast(sig_id, target_len, sigma=5.0, blend=0.0):
    """Predict trajectory using fast NN + smoothing + optional blending."""
    rows, cols, shape = load_skeleton(sig_id)
    if len(rows) < 2:
        return np.full(target_len, 0.5), np.full(target_len, 0.5)

    ox, oy = fast_nn_order(rows, cols)
    nx, ny = normalize(ox, oy)

    # Smooth the ordering (critical for noisy NN paths)
    rx, ry = resample_smooth(nx, ny, target_len, sigma=sigma)

    # Pick direction by smoothness
    s_fwd = smoothness(rx, ry)
    s_rev = smoothness(rx[::-1], ry[::-1])
    if s_rev < s_fwd:
        rx, ry = rx[::-1], ry[::-1]

    # Blend with trajectory mean
    if blend > 0:
        mx, my = np.mean(rx), np.mean(ry)
        rx = (1 - blend) * rx + blend * mx
        ry = (1 - blend) * ry + blend * my

    return rx, ry


# ============================================================
# Evaluate multiple configs
# ============================================================
configs = [
    {'sigma': 0, 'blend': 0, 'name': 'raw'},
    {'sigma': 3, 'blend': 0, 'name': 's3'},
    {'sigma': 5, 'blend': 0, 'name': 's5'},
    {'sigma': 10, 'blend': 0, 'name': 's10'},
    {'sigma': 20, 'blend': 0, 'name': 's20'},
    {'sigma': 50, 'blend': 0, 'name': 's50'},
    {'sigma': 5, 'blend': 0.3, 'name': 's5_b0.3'},
    {'sigma': 5, 'blend': 0.5, 'name': 's5_b0.5'},
    {'sigma': 10, 'blend': 0.3, 'name': 's10_b0.3'},
    {'sigma': 10, 'blend': 0.5, 'name': 's10_b0.5'},
    {'sigma': 20, 'blend': 0.3, 'name': 's20_b0.3'},
    {'sigma': 20, 'blend': 0.5, 'name': 's20_b0.5'},
    {'sigma': 0, 'blend': 1.0, 'name': 'mean_only'},
]

n_eval = 200
sig_ids = sorted(train['signature_id'].unique())[:n_eval]

results = {c['name']: {'ex': [], 'ey': []} for c in configs}

logging.info(f"Evaluating {len(configs)} configs on {n_eval} train signatures...")
t0 = time.time()

for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 50 == 0:
        logging.info(f"  {i+1}/{n_eval} ({time.time()-t0:.0f}s)")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)

    for c in configs:
        px, py = predict_signature_fast(sig_id, n, sigma=c['sigma'], blend=c['blend'])

        # Try fwd/rev, pick best
        rmse_f = column_rmse(px, py, tx, ty)
        rmse_r = column_rmse(px[::-1], py[::-1], tx, ty)

        if rmse_f <= rmse_r:
            results[c['name']]['ex'].append(px - tx)
            results[c['name']]['ey'].append(py - ty)
        else:
            results[c['name']]['ex'].append(px[::-1] - tx)
            results[c['name']]['ey'].append(py[::-1] - ty)

logging.info(f"Done in {time.time()-t0:.1f}s")

# Report
print("\n" + "=" * 60)
print(f"RESULTS ({n_eval} train signatures, column-wise RMSE)")
print("=" * 60)
print(f"{'Config':<20} {'RMSE':>10}")
print("-" * 32)

best_name, best_rmse = None, 1.0
for c in configs:
    ex = np.concatenate(results[c['name']]['ex'])
    ey = np.concatenate(results[c['name']]['ey'])
    rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
    print(f"{c['name']:<20} {rmse:>10.5f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_name = c['name']

print(f"\n>>> Best: {best_name} (RMSE: {best_rmse:.5f})")
print(f">>> LB best: 0.24449")
print(f">>> LB avg: 0.25258")
