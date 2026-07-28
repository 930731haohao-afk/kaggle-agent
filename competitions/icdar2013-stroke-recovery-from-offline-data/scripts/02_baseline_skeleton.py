"""
Baseline: Skeleton extraction + multiple ordering strategies.
Evaluate on train data where we have ground truth.
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology, color
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import lil_matrix
import time
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
IMG_DIR = f"{DATA_DIR}/images_stroke/images"

train = pd.read_csv(f"{DATA_DIR}/train.csv")

# ============================================================
# Helper functions
# ============================================================

def load_and_skeletonize(sig_id, threshold=0.85):
    """Load image, binarize, skeletonize. Return skeleton pixel coords."""
    img_path = f"{IMG_DIR}/{sig_id:04d}.jpg"
    img = np.array(Image.open(img_path).convert('L')) / 255.0
    binary = img < threshold
    skeleton = morphology.skeletonize(binary)
    # Get skeleton pixel coordinates (row, col)
    rows, cols = np.where(skeleton)
    return rows, cols, skeleton, img.shape


def normalize_coords(x, y):
    """Normalize to [0, 1] per-signature."""
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


def resample_to_length(x, y, target_len):
    """Resample using cubic spline to target length."""
    n = len(x)
    if n < 2:
        return np.full(target_len, x[0] if n > 0 else 0.5), \
               np.full(target_len, y[0] if n > 0 else 0.5)
    t_orig = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, target_len)
    # Use cubic spline interpolation
    cs_x = CubicSpline(t_orig, x)
    cs_y = CubicSpline(t_orig, y)
    return cs_x(t_new), cs_y(t_new)


def compute_rmse(pred_x, pred_y, true_x, true_y):
    """Compute RMSE between predicted and true trajectories."""
    pred_x, pred_y = np.array(pred_x), np.array(pred_y)
    true_x, true_y = np.array(true_x), np.array(true_y)
    return np.sqrt(np.mean((pred_x - true_x)**2 + (pred_y - true_y)**2))


# ============================================================
# Ordering strategies
# ============================================================

def order_xy(rows, cols):
    """Scan column-first (x), then row (y). Matches MATLAB xy_benchmark."""
    # Sort by col (x) first, then row (y)
    idx = np.lexsort((rows, cols))
    return cols[idx], rows[idx]


def order_yx(rows, cols):
    """Scan row-first (y), then col (x). Matches MATLAB yx_benchmark."""
    idx = np.lexsort((cols, rows))
    return cols[idx], rows[idx]


def order_nearest_neighbor(rows, cols, start_idx=None):
    """Nearest-neighbor traversal starting from leftmost point."""
    n = len(rows)
    if n == 0:
        return np.array([]), np.array([])

    points = np.column_stack((cols, rows))
    visited = np.zeros(n, dtype=bool)
    order = []

    if start_idx is None:
        # Start from leftmost (smallest x), then topmost
        start_idx = np.lexsort((rows, cols))[0]

    current = start_idx
    visited[current] = True
    order.append(current)

    for _ in range(n - 1):
        # Find nearest unvisited
        dists = cdist(points[current:current+1], points)[0]
        dists[visited] = np.inf
        nearest = np.argmin(dists)
        visited[nearest] = True
        order.append(nearest)
        current = nearest

    order = np.array(order)
    return cols[order], rows[order]


def order_nn_best(rows, cols):
    """Try NN from multiple starting points, pick the smoothest path."""
    n = len(rows)
    if n == 0:
        return np.array([]), np.array([])

    # Find candidate start points: corners and extremes
    candidates = set()
    candidates.add(np.lexsort((rows, cols))[0])   # leftmost
    candidates.add(np.lexsort((rows, cols))[-1])   # rightmost
    candidates.add(np.lexsort((cols, rows))[0])    # topmost
    candidates.add(np.lexsort((cols, rows))[-1])   # bottommost

    best_x, best_y = None, None
    best_smoothness = np.inf

    for start in candidates:
        x, y = order_nearest_neighbor(rows, cols, start_idx=start)
        # Smoothness = sum of squared second derivatives
        if len(x) > 2:
            d2x = np.diff(x, n=2)
            d2y = np.diff(y, n=2)
            smoothness = np.mean(d2x**2 + d2y**2)
        else:
            smoothness = 0
        if smoothness < best_smoothness:
            best_smoothness = smoothness
            best_x, best_y = x, y

    return best_x, best_y


def build_skeleton_graph(skeleton):
    """Build adjacency graph from skeleton image (8-connected)."""
    rows, cols = np.where(skeleton)
    n = len(rows)
    if n == 0:
        return rows, cols, None, {}

    # Create point-to-index mapping
    point_map = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        point_map[(r, c)] = i

    # Build adjacency
    adj = lil_matrix((n, n), dtype=float)
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in point_map:
                    j = point_map[(nr, nc)]
                    dist = np.sqrt(dr**2 + dc**2)
                    adj[i, j] = dist

    return rows, cols, adj.tocsr(), point_map


def find_endpoints(adj, n):
    """Find endpoints (degree-1 nodes) in the skeleton graph."""
    degrees = np.diff(adj.indptr)
    endpoints = np.where(degrees == 1)[0]
    return endpoints


def order_graph_dfs(skeleton):
    """Order skeleton pixels using DFS on the skeleton graph."""
    rows, cols, adj, point_map = build_skeleton_graph(skeleton)
    n = len(rows)
    if n == 0:
        return np.array([]), np.array([])

    # Find endpoints
    endpoints = find_endpoints(adj, n)

    # Start DFS from an endpoint (prefer leftmost endpoint)
    if len(endpoints) > 0:
        # Pick the leftmost endpoint
        ep_cols = cols[endpoints]
        start = endpoints[np.argmin(ep_cols)]
    else:
        start = np.argmin(cols)

    # DFS
    visited = np.zeros(n, dtype=bool)
    order = []
    stack = [start]

    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        order.append(node)

        # Get neighbors sorted by distance to current trajectory direction
        neighbors = adj[node].nonzero()[1]
        unvisited = [nb for nb in neighbors if not visited[nb]]

        # Push in reverse order so closest/most-aligned is processed first
        if len(order) > 1:
            prev = order[-2]
            dx = cols[node] - cols[prev]
            dy = rows[node] - rows[prev]
            # Sort by alignment with current direction
            def alignment(nb):
                ndx = cols[nb] - cols[node]
                ndy = rows[nb] - rows[node]
                dot = dx * ndx + dy * ndy
                return -dot
            unvisited.sort(key=alignment)

        stack.extend(reversed(unvisited))

    order = np.array(order)
    return cols[order], rows[order]


def order_graph_longest_path(skeleton):
    """Find the two most distant endpoints, then trace the path between them."""
    rows, cols, adj, point_map = build_skeleton_graph(skeleton)
    n = len(rows)
    if n == 0:
        return np.array([]), np.array([])
    if n > 5000:
        # Too large for all-pairs shortest path — fall back to NN
        return order_nn_best(rows, cols)

    # Find endpoints
    endpoints = find_endpoints(adj, n)
    if len(endpoints) < 2:
        endpoints = np.arange(min(n, 10))

    # Find the pair of endpoints with the longest shortest path
    # Use BFS from each endpoint
    best_dist = 0
    best_start, best_end = 0, 0

    for ep in endpoints[:20]:  # Limit search
        dist_from_ep = shortest_path(adj, indices=ep, directed=False)
        # Only consider other endpoints
        for ep2 in endpoints:
            if ep2 != ep and dist_from_ep[ep2] > best_dist and not np.isinf(dist_from_ep[ep2]):
                best_dist = dist_from_ep[ep2]
                best_start = ep
                best_end = ep2

    if best_dist == 0:
        return order_nn_best(rows, cols)

    # Trace path from best_start to best_end using BFS
    from collections import deque
    dist_matrix = shortest_path(adj, indices=best_start, directed=False)

    # Reconstruct path (greedy: from end, go to neighbor closest to start)
    path = [best_end]
    visited = {best_end}
    current = best_end

    while current != best_start:
        neighbors = adj[current].nonzero()[1]
        unvisited_neighbors = [nb for nb in neighbors if nb not in visited]
        if not unvisited_neighbors:
            break
        # Pick neighbor with smallest distance to start
        best_nb = min(unvisited_neighbors, key=lambda nb: dist_matrix[nb])
        path.append(best_nb)
        visited.add(best_nb)
        current = best_nb

    path = path[::-1]  # Reverse to go start→end

    # The path may not cover all skeleton pixels (branches)
    # Add remaining pixels using NN from the path
    path_set = set(path)
    remaining = [i for i in range(n) if i not in path_set]

    if remaining:
        # Insert remaining points into the closest position in the path
        for r_idx in remaining:
            rx, ry = cols[r_idx], rows[r_idx]
            # Find closest point in path
            path_points = np.column_stack((cols[path], rows[path]))
            dists = np.sqrt((path_points[:, 0] - rx)**2 + (path_points[:, 1] - ry)**2)
            insert_pos = np.argmin(dists) + 1
            path.insert(insert_pos, r_idx)

    order = np.array(path)
    return cols[order], rows[order]


# ============================================================
# Evaluate strategies on train data
# ============================================================

strategies = {
    'xy_scan': order_xy,
    'yx_scan': order_yx,
    'nn_best': order_nn_best,
    'graph_dfs': None,  # needs skeleton
    'graph_longest': None,  # needs skeleton
}

results = {name: [] for name in strategies}
n_eval = min(100, train['signature_id'].nunique())  # Evaluate on subset
sig_ids = sorted(train['signature_id'].unique())[:n_eval]

logging.info(f"Evaluating {len(strategies)} strategies on {n_eval} signatures...")
start_time = time.time()

for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 20 == 0:
        logging.info(f"  Processing {i+1}/{n_eval}...")

    # Get ground truth
    gt = train[train['signature_id'] == sig_id].sort_values('time')
    true_x = gt['x'].values
    true_y = gt['y'].values
    target_len = len(gt)

    # Extract skeleton
    rows, cols, skeleton, img_shape = load_and_skeletonize(sig_id)

    if len(rows) < 2:
        for name in strategies:
            results[name].append(0.5)  # Fallback
        continue

    # Evaluate each strategy
    for name in strategies:
        try:
            if name == 'graph_dfs':
                px, py = order_graph_dfs(skeleton)
            elif name == 'graph_longest':
                px, py = order_graph_longest_path(skeleton)
            else:
                px, py = strategies[name](rows, cols)

            # Normalize
            nx, ny = normalize_coords(px, py)
            # Resample
            rx, ry = resample_to_length(nx, ny, target_len)
            # Clip to [0,1]
            rx = np.clip(rx, 0, 1)
            ry = np.clip(ry, 0, 1)

            # Compute RMSE (also try reversed)
            rmse_fwd = compute_rmse(rx, ry, true_x, true_y)
            rmse_rev = compute_rmse(rx[::-1], ry[::-1], true_x, true_y)
            rmse = min(rmse_fwd, rmse_rev)

            results[name].append(rmse)
        except Exception as e:
            logging.warning(f"  Sig {sig_id}, {name}: {e}")
            results[name].append(0.5)

elapsed = time.time() - start_time
logging.info(f"Evaluation done in {elapsed:.1f}s")

# ============================================================
# Report results
# ============================================================

print("\n" + "=" * 60)
print("BASELINE EVALUATION RESULTS")
print(f"Evaluated on {n_eval} train signatures")
print("=" * 60)

for name in strategies:
    scores = results[name]
    print(f"\n{name}:")
    print(f"  Mean RMSE: {np.mean(scores):.4f}")
    print(f"  Median RMSE: {np.median(scores):.4f}")
    print(f"  Std RMSE: {np.std(scores):.4f}")
    print(f"  Min/Max: {np.min(scores):.4f} / {np.max(scores):.4f}")

# Best strategy
best_name = min(strategies.keys(), key=lambda n: np.mean(results[n]))
print(f"\n>>> Best strategy: {best_name} (Mean RMSE: {np.mean(results[best_name]):.4f})")

# Save results
results_summary = {}
for name in strategies:
    results_summary[name] = {
        'mean_rmse': float(np.mean(results[name])),
        'median_rmse': float(np.median(results[name])),
        'std_rmse': float(np.std(results[name])),
    }

with open(f"{DATA_DIR}/../scripts/baseline_results.json", 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\nResults saved to baseline_results.json")
