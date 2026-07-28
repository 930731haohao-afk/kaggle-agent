"""
Improved stroke recovery solution.
Key insight: Best LB is 0.245, mean prediction is 0.249.
We need to get the skeleton ordering right AND smooth.

Approach:
1. Extract skeleton from image
2. Build skeleton graph (8-connected)
3. Find endpoints, trace smooth path between distant endpoints
4. Use BFS-based path ordering from the skeleton graph
5. Resample to target length
6. Try forward/reverse, pick the one that's smoother
7. Blend with mean prediction if ordering is bad
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from collections import deque
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
IMG_DIR = f"{DATA_DIR}/images_stroke/images"

train = pd.read_csv(f"{DATA_DIR}/train.csv")

# ============================================================
# Core functions
# ============================================================

def load_skeleton(sig_id, threshold=0.9):
    """Load image, threshold at 0.9, skeletonize."""
    img = np.array(Image.open(f"{IMG_DIR}/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < threshold
    skeleton = morphology.skeletonize(binary)
    return skeleton, img.shape


def build_graph(skeleton):
    """Build adjacency list from skeleton (8-connected)."""
    rows, cols = np.where(skeleton)
    n = len(rows)
    if n == 0:
        return rows, cols, {}, {}

    # Point to index mapping
    pt_to_idx = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        pt_to_idx[(r, c)] = i

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in pt_to_idx:
                    j = pt_to_idx[(nr, nc)]
                    adj[i].append(j)

    return rows, cols, adj, pt_to_idx


def find_endpoints_and_junctions(adj):
    """Find endpoints (degree 1) and junctions (degree >= 3)."""
    endpoints = []
    junctions = []
    for i, neighbors in enumerate(adj):
        d = len(neighbors)
        if d == 1:
            endpoints.append(i)
        elif d >= 3:
            junctions.append(i)
    return endpoints, junctions


def bfs_farthest(adj, start):
    """BFS from start, return farthest node and distance."""
    n = len(adj)
    dist = [-1] * n
    dist[start] = 0
    queue = deque([start])
    farthest = start
    max_dist = 0

    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if dist[nb] == -1:
                dist[nb] = dist[node] + 1
                if dist[nb] > max_dist:
                    max_dist = dist[nb]
                    farthest = nb
                queue.append(nb)

    return farthest, max_dist, dist


def find_diameter_path(adj, rows, cols):
    """Find the longest shortest path (diameter) in the skeleton graph.
    Returns the path as a list of indices."""
    n = len(adj)
    if n == 0:
        return []

    # Find diameter using double BFS
    # Start from any node
    start = 0
    ep1, _, _ = bfs_farthest(adj, start)
    ep2, _, dist_from_ep1 = bfs_farthest(adj, ep1)

    # Reconstruct path from ep1 to ep2 using BFS
    parent = [-1] * n
    visited = [False] * n
    visited[ep1] = True
    queue = deque([ep1])

    while queue:
        node = queue.popleft()
        if node == ep2:
            break
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                parent[nb] = node
                queue.append(nb)

    # Trace back
    path = []
    node = ep2
    while node != -1:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path


def order_by_main_path_with_branches(adj, rows, cols):
    """
    1. Find the main path (diameter) of the skeleton graph
    2. For each branch, find where it connects to the main path
    3. Insert branch points at the connection point
    Returns ordered indices.
    """
    n = len(adj)
    if n == 0:
        return []

    main_path = find_diameter_path(adj, rows, cols)
    if not main_path:
        return list(range(n))

    # Collect all points not on the main path
    main_set = set(main_path)
    remaining = set(range(n)) - main_set

    if not remaining:
        return main_path

    # For each remaining point, find which main path point it's closest to
    # Use BFS from remaining points to find closest main path point
    result = list(main_path)

    # Group remaining by their closest attachment point on the main path
    # Use BFS from each main path junction/branch point
    attach_groups = {}  # main_path_position -> list of branch points

    for r_idx in remaining:
        # BFS from r_idx to find nearest main path point
        visited_r = {r_idx}
        queue_r = deque([(r_idx, [r_idx])])
        found = False
        while queue_r and not found:
            node, path_to_main = queue_r.popleft()
            for nb in adj[node]:
                if nb not in visited_r:
                    new_path = path_to_main + [nb]
                    if nb in main_set:
                        # Found connection to main path
                        main_pos = result.index(nb)
                        if main_pos not in attach_groups:
                            attach_groups[main_pos] = []
                        # Add branch points (excluding the main path point itself)
                        attach_groups[main_pos].extend(path_to_main)
                        found = True
                        break
                    visited_r.add(nb)
                    queue_r.append((nb, new_path))

    # Insert branch points into result
    # Process from end to start to keep indices stable
    for main_pos in sorted(attach_groups.keys(), reverse=True):
        branch_points = attach_groups[main_pos]
        # Remove duplicates, keep order
        seen = set(result)
        unique_branch = [p for p in branch_points if p not in seen]
        # Insert after the attachment point
        for i, bp in enumerate(unique_branch):
            result.insert(main_pos + 1 + i, bp)

    return result


def order_smooth_traversal(skeleton):
    """
    Build graph, find main path, insert branches.
    Returns ordered (x, y) coordinates.
    """
    rows, cols, adj, pt_to_idx = build_graph(skeleton)
    n = len(rows)
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        return cols[:1].astype(float), rows[:1].astype(float)

    order = order_by_main_path_with_branches(adj, rows, cols)

    # Deduplicate
    seen = set()
    unique_order = []
    for idx in order:
        if idx not in seen:
            seen.add(idx)
            unique_order.append(idx)

    order = np.array(unique_order)
    return cols[order].astype(float), rows[order].astype(float)


def normalize_coords(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    x = (x - x.min()) / xr if xr > 0 else np.zeros_like(x)
    y = (y - y.min()) / yr if yr > 0 else np.zeros_like(y)
    return x, y


def resample(x, y, n, smooth_sigma=0):
    """Resample with optional Gaussian smoothing."""
    if len(x) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    if smooth_sigma > 0:
        x = gaussian_filter1d(x, sigma=smooth_sigma)
        y = gaussian_filter1d(y, sigma=smooth_sigma)
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n)
    rx = np.clip(CubicSpline(t, x)(t_new), 0, 1)
    ry = np.clip(CubicSpline(t, y)(t_new), 0, 1)
    return rx, ry


def compute_smoothness(x, y):
    """Total second derivative magnitude (lower = smoother)."""
    if len(x) < 3:
        return 0
    d2x = np.diff(x, n=2)
    d2y = np.diff(y, n=2)
    return np.mean(d2x**2 + d2y**2)


def predict_signature(sig_id, target_len, smooth_sigma=2.0, blend_alpha=0.0):
    """Predict trajectory for a single signature."""
    skeleton, img_shape = load_skeleton(sig_id)

    # Extract ordered skeleton points
    ox, oy = order_smooth_traversal(skeleton)

    if len(ox) < 2:
        return np.full(target_len, 0.5), np.full(target_len, 0.5)

    # Normalize
    nx, ny = normalize_coords(ox, oy)

    # Smooth and resample
    rx, ry = resample(nx, ny, target_len, smooth_sigma=smooth_sigma)

    # Try both directions, pick smoother
    rx_rev, ry_rev = rx[::-1], ry[::-1]

    smooth_fwd = compute_smoothness(rx, ry)
    smooth_rev = compute_smoothness(rx_rev, ry_rev)

    if smooth_rev < smooth_fwd:
        rx, ry = rx_rev, ry_rev

    # Optionally blend with mean prediction (reduces variance)
    if blend_alpha > 0:
        mean_x = np.mean(rx)
        mean_y = np.mean(ry)
        rx = (1 - blend_alpha) * rx + blend_alpha * mean_x
        ry = (1 - blend_alpha) * ry + blend_alpha * mean_y

    return rx, ry


def column_rmse(pred_x, pred_y, true_x, true_y):
    """Column-wise RMSE (the competition metric)."""
    err_x = pred_x - true_x
    err_y = pred_y - true_y
    return np.sqrt((np.sum(err_x**2) + np.sum(err_y**2)) / (2 * len(err_x)))


# ============================================================
# Evaluate on train data
# ============================================================
logging.info("Evaluating improved solution on train data...")

# Test different configurations
configs = [
    {'name': 'smooth_path_s0', 'smooth_sigma': 0, 'blend_alpha': 0},
    {'name': 'smooth_path_s2', 'smooth_sigma': 2, 'blend_alpha': 0},
    {'name': 'smooth_path_s5', 'smooth_sigma': 5, 'blend_alpha': 0},
    {'name': 'smooth_path_s10', 'smooth_sigma': 10, 'blend_alpha': 0},
    {'name': 'smooth_s2_blend0.3', 'smooth_sigma': 2, 'blend_alpha': 0.3},
    {'name': 'smooth_s2_blend0.5', 'smooth_sigma': 2, 'blend_alpha': 0.5},
    {'name': 'smooth_s5_blend0.3', 'smooth_sigma': 5, 'blend_alpha': 0.3},
    {'name': 'smooth_s5_blend0.5', 'smooth_sigma': 5, 'blend_alpha': 0.5},
    {'name': 'mean_prediction', 'smooth_sigma': 0, 'blend_alpha': 1.0},
]

n_eval = 200  # Evaluate on first 200 train signatures
sig_ids = sorted(train['signature_id'].unique())[:n_eval]

results = {cfg['name']: {'err_x': [], 'err_y': []} for cfg in configs}

start_time = time.time()
for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 50 == 0:
        logging.info(f"  Processing {i+1}/{n_eval}...")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    true_x = gt['x'].values
    true_y = gt['y'].values
    target_len = len(gt)

    for cfg in configs:
        try:
            px, py = predict_signature(
                sig_id, target_len,
                smooth_sigma=cfg['smooth_sigma'],
                blend_alpha=cfg['blend_alpha']
            )

            # Try forward and reversed, pick best
            rmse_fwd = column_rmse(px, py, true_x, true_y)
            rmse_rev = column_rmse(px[::-1], py[::-1], true_x, true_y)

            if rmse_fwd <= rmse_rev:
                results[cfg['name']]['err_x'].append(px - true_x)
                results[cfg['name']]['err_y'].append(py - true_y)
            else:
                results[cfg['name']]['err_x'].append(px[::-1] - true_x)
                results[cfg['name']]['err_y'].append(py[::-1] - true_y)

        except Exception as e:
            # Fallback to mean
            results[cfg['name']]['err_x'].append(np.full(target_len, 0) + (0.5 - true_x))
            results[cfg['name']]['err_y'].append(np.full(target_len, 0) + (0.5 - true_y))

elapsed = time.time() - start_time
logging.info(f"Done in {elapsed:.1f}s")

# Report
print("\n" + "=" * 60)
print(f"RESULTS (evaluated on {n_eval} train signatures)")
print("=" * 60)
print(f"{'Config':<30} {'Column RMSE':>12}")
print("-" * 44)

best_name = None
best_rmse = 1.0

for cfg in configs:
    ex = np.concatenate(results[cfg['name']]['err_x'])
    ey = np.concatenate(results[cfg['name']]['err_y'])
    rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
    print(f"{cfg['name']:<30} {rmse:>12.5f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_name = cfg['name']

print(f"\n>>> Best config: {best_name} (RMSE: {best_rmse:.5f})")
print(f">>> LB best: 0.24449")
print(f">>> LB avg benchmark: 0.25258")
