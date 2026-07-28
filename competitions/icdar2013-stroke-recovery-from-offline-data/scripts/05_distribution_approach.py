"""
Distribution-based approach: Instead of ordering skeleton pixels,
use the spatial distribution to predict trajectories.

Key ideas:
1. Percentile mapping: x(t) = percentile_x(t/T), y(t) = percentile_y(t/T)
2. CDF mapping: Map time fraction to CDF of skeleton coordinates
3. Weighted mean: Just predict mean, but use better mean estimation
4. Linear trend: Predict a linear interpolation from start to end of skeleton
5. Smoothed skeleton walk on the graph (adjacency-based)
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import time
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
    return rows, cols, img


def normalize(x, y):
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    x = (x - x.min()) / xr if xr > 0 else np.zeros_like(x)
    y = (y - y.min()) / yr if yr > 0 else np.zeros_like(y)
    return x, y


def column_rmse(px, py, tx, ty):
    ex, ey = px - tx, py - ty
    return np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))


# ============================================================
# Method 1: Mean prediction (baseline)
# ============================================================
def predict_mean(rows, cols, n):
    """Predict the mean of normalized skeleton coordinates."""
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    nx, ny = normalize(cols.astype(float), rows.astype(float))
    return np.full(n, np.mean(nx)), np.full(n, np.mean(ny))


# ============================================================
# Method 2: X-percentile mapping
# ============================================================
def predict_x_percentile(rows, cols, n):
    """Map time fraction to x-coordinate percentile.
    Assumes left-to-right writing direction."""
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    nx, ny = normalize(cols.astype(float), rows.astype(float))

    # Sort skeleton points by x
    sort_idx = np.argsort(nx)
    sorted_x = nx[sort_idx]
    sorted_y = ny[sort_idx]

    # For each time fraction, pick the corresponding percentile
    t_fracs = np.linspace(0, 1, n)
    indices = np.clip((t_fracs * (len(sorted_x) - 1)).astype(int), 0, len(sorted_x) - 1)
    pred_x = sorted_x[indices]
    pred_y = sorted_y[indices]

    return pred_x, pred_y


# ============================================================
# Method 3: CDF-based prediction
# ============================================================
def predict_cdf(rows, cols, n):
    """Use the empirical CDF of x-coordinates.
    x(t) is set by the CDF (monotonically increasing).
    y(t) is the conditional mean of y given x(t)."""
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    nx, ny = normalize(cols.astype(float), rows.astype(float))

    # Sort by x
    sort_idx = np.argsort(nx)
    sorted_x = nx[sort_idx]
    sorted_y = ny[sort_idx]

    # CDF of x
    cdf_x = np.linspace(0, 1, len(sorted_x))

    # Bin y values by x position (sliding window)
    t_fracs = np.linspace(0, 1, n)

    # x values from CDF
    pred_x = np.interp(t_fracs, cdf_x, sorted_x)

    # For y, use window around each x value
    window = max(1, len(sorted_x) // (2 * n))
    pred_y = np.zeros(n)
    for i, frac in enumerate(t_fracs):
        idx = int(frac * (len(sorted_y) - 1))
        lo = max(0, idx - window)
        hi = min(len(sorted_y), idx + window + 1)
        pred_y[i] = np.mean(sorted_y[lo:hi])

    return pred_x, pred_y


# ============================================================
# Method 4: Graph walk (adjacency-based, fast)
# ============================================================
def predict_graph_walk(rows, cols, n, skeleton_img=None):
    """Walk the skeleton graph using adjacency, starting from leftmost endpoint."""
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)

    # Build adjacency using image directly (faster than building explicit graph)
    h, w = skeleton_img.shape if skeleton_img is not None else (rows.max()+1, cols.max()+1)
    skel = np.zeros((h, w), dtype=bool)
    skel[rows, cols] = True

    # Find endpoints (pixels with exactly 1 neighbor)
    pt_to_idx = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        pt_to_idx[(r, c)] = i

    degrees = np.zeros(len(rows), dtype=int)
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if 0 <= r+dr < h and 0 <= c+dc < w and skel[r+dr, c+dc]:
                    degrees[i] += 1

    endpoints = np.where(degrees == 1)[0]

    # Start from leftmost endpoint (or leftmost point)
    if len(endpoints) > 0:
        start = endpoints[np.argmin(cols[endpoints])]
    else:
        start = np.argmin(cols)

    # Walk using DFS, prefer continuing in same direction
    visited = np.zeros(len(rows), dtype=bool)
    order = [start]
    visited[start] = True
    current = start
    prev_dr, prev_dc = 0, 1  # Initial direction: right

    for _ in range(len(rows) - 1):
        r, c = rows[current], cols[current]
        best_nb = -1
        best_score = -np.inf

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in pt_to_idx:
                    nb = pt_to_idx[(nr, nc)]
                    if not visited[nb]:
                        # Score: prefer continuing in same direction
                        dot = dr * prev_dr + dc * prev_dc
                        score = dot
                        if score > best_score:
                            best_score = score
                            best_nb = nb
                            best_dr, best_dc = dr, dc

        if best_nb == -1:
            # No adjacent unvisited — jump to nearest unvisited
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            dists = (cols[unvisited] - c)**2 + (rows[unvisited] - r)**2
            best_nb = unvisited[np.argmin(dists)]
            best_dr = rows[best_nb] - r
            best_dc = cols[best_nb] - c

        order.append(best_nb)
        visited[best_nb] = True
        prev_dr, prev_dc = best_dr, best_dc
        current = best_nb

    order = np.array(order)
    ox, oy = cols[order].astype(float), rows[order].astype(float)
    nx, ny = normalize(ox, oy)

    # Smooth
    if len(nx) > 5:
        sigma = max(1, len(nx) // 50)
        nx = gaussian_filter1d(nx, sigma=sigma)
        ny = gaussian_filter1d(ny, sigma=sigma)

    # Resample
    t = np.linspace(0, 1, len(nx))
    t_new = np.linspace(0, 1, n)
    rx = np.clip(np.interp(t_new, t, nx), 0, 1)
    ry = np.clip(np.interp(t_new, t, ny), 0, 1)

    return rx, ry


# ============================================================
# Method 5: Ink density columns approach
# ============================================================
def predict_ink_density(rows, cols, n, img):
    """Predict x(t) as linear left-to-right, y(t) as mean y at each x position."""
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)

    nx, ny = normalize(cols.astype(float), rows.astype(float))

    # Create x prediction: linear from 0 to 1
    pred_x = np.linspace(0, 1, n)

    # For each x, find mean y of nearby skeleton pixels
    pred_y = np.zeros(n)
    x_width = 1.0 / n
    for i in range(n):
        target_x = pred_x[i]
        mask = np.abs(nx - target_x) < max(x_width * 2, 0.05)
        if mask.sum() > 0:
            pred_y[i] = np.mean(ny[mask])
        elif i > 0:
            pred_y[i] = pred_y[i-1]
        else:
            pred_y[i] = np.mean(ny)

    # Smooth y
    if n > 5:
        pred_y = gaussian_filter1d(pred_y, sigma=max(1, n // 50))
    pred_y = np.clip(pred_y, 0, 1)

    return pred_x, pred_y


# ============================================================
# Evaluate all methods
# ============================================================
n_eval = 300
sig_ids = sorted(train['signature_id'].unique())[:n_eval]

methods = {
    'mean': lambda r, c, n, img: predict_mean(r, c, n),
    'x_percentile': lambda r, c, n, img: predict_x_percentile(r, c, n),
    'cdf': lambda r, c, n, img: predict_cdf(r, c, n),
    'graph_walk': lambda r, c, n, img: predict_graph_walk(r, c, n, skeleton_img=(img < 0.9).astype(bool)),
    'ink_density': lambda r, c, n, img: predict_ink_density(r, c, n, img),
}

results = {m: {'ex': [], 'ey': []} for m in methods}

logging.info(f"Evaluating {len(methods)} methods on {n_eval} signatures...")
t0 = time.time()

for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 100 == 0:
        logging.info(f"  {i+1}/{n_eval} ({time.time()-t0:.0f}s)")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)

    rows, cols, img = load_skeleton(sig_id)

    for mname, mfunc in methods.items():
        try:
            px, py = mfunc(rows, cols, n, img)

            # Try fwd/rev
            rmse_f = column_rmse(px, py, tx, ty)
            rmse_r = column_rmse(px[::-1], py[::-1], tx, ty)

            if rmse_f <= rmse_r:
                results[mname]['ex'].append(px - tx)
                results[mname]['ey'].append(py - ty)
            else:
                results[mname]['ex'].append(px[::-1] - tx)
                results[mname]['ey'].append(py[::-1] - ty)
        except Exception as e:
            results[mname]['ex'].append(np.zeros(n) + 0.5 - tx)
            results[mname]['ey'].append(np.zeros(n) + 0.5 - ty)

logging.info(f"Done in {time.time()-t0:.1f}s")

print("\n" + "=" * 60)
print(f"RESULTS ({n_eval} sigs, column-wise RMSE)")
print("=" * 60)
print(f"{'Method':<20} {'RMSE':>10}")
print("-" * 32)

best_n, best_r = None, 1.0
for m in methods:
    ex = np.concatenate(results[m]['ex'])
    ey = np.concatenate(results[m]['ey'])
    rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
    print(f"{m:<20} {rmse:>10.5f}")
    if rmse < best_r:
        best_r = rmse
        best_n = m

print(f"\n>>> Best: {best_n} (RMSE: {best_r:.5f})")
print(f">>> LB best: 0.24449, LB avg: 0.25258")

# ============================================================
# Try blending best method with mean
# ============================================================
print("\n" + "=" * 60)
print("BLENDING EXPERIMENTS")
print("=" * 60)

# Re-collect raw predictions for blending
best_preds = {'px': [], 'py': []}
mean_preds = {'px': [], 'py': []}
true_vals = {'tx': [], 'ty': []}

for sig_id in sig_ids:
    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)
    rows, cols, img = load_skeleton(sig_id)

    # Best method
    px, py = methods[best_n](rows, cols, n, img)
    rmse_f = column_rmse(px, py, tx, ty)
    rmse_r = column_rmse(px[::-1], py[::-1], tx, ty)
    if rmse_r < rmse_f:
        px, py = px[::-1], py[::-1]
    best_preds['px'].append(px)
    best_preds['py'].append(py)

    # Mean
    mx, my = predict_mean(rows, cols, n)
    mean_preds['px'].append(mx)
    mean_preds['py'].append(my)

    true_vals['tx'].append(tx)
    true_vals['ty'].append(ty)

all_best_px = np.concatenate(best_preds['px'])
all_best_py = np.concatenate(best_preds['py'])
all_mean_px = np.concatenate(mean_preds['px'])
all_mean_py = np.concatenate(mean_preds['py'])
all_tx = np.concatenate(true_vals['tx'])
all_ty = np.concatenate(true_vals['ty'])

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    bx = (1-alpha) * all_best_px + alpha * all_mean_px
    by = (1-alpha) * all_best_py + alpha * all_mean_py
    rmse = column_rmse(bx, by, all_tx, all_ty)
    marker = " <-- best" if alpha == 0 else ""
    print(f"  alpha={alpha:.1f}: RMSE={rmse:.5f}{marker}")
