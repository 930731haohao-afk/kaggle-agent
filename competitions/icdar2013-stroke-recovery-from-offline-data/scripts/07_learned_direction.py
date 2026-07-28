"""
Learn to pick trajectory direction from training data.

Key insight: with oracle direction + α=0.4 blend, we get 0.229 on train (beats LB best 0.245).
The gap is entirely from direction estimation.

Approach:
1. Learn the "average trajectory shape" from training data (x(t/T), y(t/T))
2. For each test signature, correlate CDF prediction (fwd/rev) with the average shape
3. Pick the direction with higher correlation
4. Also try: learn a direction classifier from features
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
IMG_DIR = f"{DATA_DIR}/images_stroke/images"
COMP_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")


def load_skeleton(sig_id, threshold=0.9):
    img = np.array(Image.open(f"{IMG_DIR}/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < threshold
    skeleton = morphology.skeletonize(binary)
    rows, cols = np.where(skeleton)
    return rows, cols


def normalize(x, y):
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    x = (x - x.min()) / xr if xr > 0 else np.zeros_like(x)
    y = (y - y.min()) / yr if yr > 0 else np.zeros_like(y)
    return x, y


def column_rmse(px, py, tx, ty):
    ex, ey = px - tx, py - ty
    return np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))


def predict_cdf(rows, cols, n):
    """Raw CDF prediction (no blending)."""
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    nx, ny = normalize(cols.astype(float), rows.astype(float))
    sort_idx = np.argsort(nx)
    sorted_x = nx[sort_idx]
    sorted_y = ny[sort_idx]

    cdf = np.linspace(0, 1, len(sorted_x))
    t_fracs = np.linspace(0, 1, n)
    pred_x = np.interp(t_fracs, cdf, sorted_x)

    window = max(1, len(sorted_y) // (2 * n))
    pred_y = np.zeros(n)
    for i, frac in enumerate(t_fracs):
        idx = int(frac * (len(sorted_y) - 1))
        lo = max(0, idx - window)
        hi = min(len(sorted_y), idx + window + 1)
        pred_y[i] = np.mean(sorted_y[lo:hi])

    if n > 5:
        pred_y = gaussian_filter1d(pred_y, sigma=max(1, n // 30))
    pred_y = np.clip(pred_y, 0, 1)

    return pred_x, pred_y


# ============================================================
# Step 1: Compute average trajectory shape from training data
# ============================================================
logging.info("Computing average trajectory shape from training data...")

N_BINS = 100  # Resample all trajectories to 100 time steps
all_x_curves = []
all_y_curves = []

for sig_id in sorted(train['signature_id'].unique()):
    sig = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = sig['x'].values, sig['y'].values
    n = len(tx)
    if n < 3:
        continue

    # Resample to N_BINS
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, N_BINS)
    rx = np.interp(t_new, t, tx)
    ry = np.interp(t_new, t, ty)

    all_x_curves.append(rx)
    all_y_curves.append(ry)

avg_x = np.mean(all_x_curves, axis=0)  # Average x(t/T)
avg_y = np.mean(all_y_curves, axis=0)  # Average y(t/T)

print(f"Average trajectory computed from {len(all_x_curves)} signatures")
print(f"avg_x: starts at {avg_x[0]:.3f}, ends at {avg_x[-1]:.3f}, mean {np.mean(avg_x):.3f}")
print(f"avg_y: starts at {avg_y[0]:.3f}, ends at {avg_y[-1]:.3f}, mean {np.mean(avg_y):.3f}")

# ============================================================
# Step 2: Direction estimation methods
# ============================================================

def direction_by_correlation(pred_x, pred_y, avg_x, avg_y):
    """Pick direction based on correlation with average trajectory."""
    # Resample pred to match avg
    n = len(pred_x)
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, len(avg_x))
    rx = np.interp(t_new, t, pred_x)
    ry = np.interp(t_new, t, pred_y)
    rx_r = np.interp(t_new, t, pred_x[::-1])
    ry_r = np.interp(t_new, t, pred_y[::-1])

    # Correlation with average
    corr_fwd = np.corrcoef(rx, avg_x)[0, 1] + np.corrcoef(ry, avg_y)[0, 1]
    corr_rev = np.corrcoef(rx_r, avg_x)[0, 1] + np.corrcoef(ry_r, avg_y)[0, 1]

    return 'fwd' if corr_fwd >= corr_rev else 'rev'


def direction_by_x_monotonicity(pred_x, pred_y):
    """Most signatures have x increasing overall (left-to-right)."""
    dx_fwd = pred_x[-1] - pred_x[0]
    return 'fwd' if dx_fwd >= 0 else 'rev'


def direction_by_start_position(pred_x, pred_y):
    """Signatures tend to start near (0, 0.5) - left side, middle height."""
    # Forward starts at pred_x[0], pred_y[0]
    # Reverse starts at pred_x[-1], pred_y[-1]
    dist_fwd = (pred_x[0] - 0.0)**2 + (pred_y[0] - 0.5)**2
    dist_rev = (pred_x[-1] - 0.0)**2 + (pred_y[-1] - 0.5)**2
    return 'fwd' if dist_fwd <= dist_rev else 'rev'


def direction_by_all_heuristics(pred_x, pred_y, avg_x, avg_y):
    """Combine multiple heuristics by majority vote."""
    votes = []
    votes.append(direction_by_correlation(pred_x, pred_y, avg_x, avg_y))
    votes.append(direction_by_x_monotonicity(pred_x, pred_y))
    votes.append(direction_by_start_position(pred_x, pred_y))
    return 'fwd' if votes.count('fwd') >= 2 else 'rev'


# ============================================================
# Step 3: Evaluate direction estimation accuracy on training data
# ============================================================
logging.info("Evaluating direction estimation methods...")

dir_methods = {
    'correlation': lambda px, py: direction_by_correlation(px, py, avg_x, avg_y),
    'x_monotone': direction_by_x_monotonicity,
    'start_pos': direction_by_start_position,
    'majority_vote': lambda px, py: direction_by_all_heuristics(px, py, avg_x, avg_y),
}

# For each method, count how often it matches oracle direction
dir_accuracy = {m: 0 for m in dir_methods}
n_sigs = 0

sig_ids = sorted(train['signature_id'].unique())
all_results = {}  # method -> blend_alpha -> (errors_x, errors_y)

blend_alphas = [0.0, 0.3, 0.4, 0.5]

for method in list(dir_methods.keys()) + ['oracle', 'always_fwd']:
    all_results[method] = {a: {'ex': [], 'ey': []} for a in blend_alphas}

t0 = time.time()
for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 200 == 0:
        logging.info(f"  {i+1}/{len(sig_ids)} ({time.time()-t0:.0f}s)")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)

    rows, cols = load_skeleton(sig_id)
    px, py = predict_cdf(rows, cols, n)
    n_sigs += 1

    # Oracle direction
    rmse_f = column_rmse(px, py, tx, ty)
    rmse_r = column_rmse(px[::-1], py[::-1], tx, ty)
    oracle_dir = 'fwd' if rmse_f <= rmse_r else 'rev'

    for method_name, method_func in dir_methods.items():
        pred_dir = method_func(px, py)
        if pred_dir == oracle_dir:
            dir_accuracy[method_name] += 1

        for alpha in blend_alphas:
            # Apply direction
            if pred_dir == 'fwd':
                final_x, final_y = px.copy(), py.copy()
            else:
                final_x, final_y = px[::-1].copy(), py[::-1].copy()

            # Blend with mean
            if alpha > 0:
                mx, my = np.mean(px), np.mean(py)
                final_x = (1 - alpha) * final_x + alpha * mx
                final_y = (1 - alpha) * final_y + alpha * my

            all_results[method_name][alpha]['ex'].append(final_x - tx)
            all_results[method_name][alpha]['ey'].append(final_y - ty)

    # Oracle and always_fwd
    for alpha in blend_alphas:
        # Oracle
        if oracle_dir == 'fwd':
            ox, oy = px.copy(), py.copy()
        else:
            ox, oy = px[::-1].copy(), py[::-1].copy()
        if alpha > 0:
            mx, my = np.mean(px), np.mean(py)
            ox = (1-alpha)*ox + alpha*mx
            oy = (1-alpha)*oy + alpha*my
        all_results['oracle'][alpha]['ex'].append(ox - tx)
        all_results['oracle'][alpha]['ey'].append(oy - ty)

        # Always forward
        fx, fy = px.copy(), py.copy()
        if alpha > 0:
            mx, my = np.mean(px), np.mean(py)
            fx = (1-alpha)*fx + alpha*mx
            fy = (1-alpha)*fy + alpha*my
        all_results['always_fwd'][alpha]['ex'].append(fx - tx)
        all_results['always_fwd'][alpha]['ey'].append(fy - ty)

logging.info(f"Done in {time.time()-t0:.1f}s")

# Direction accuracy
print("\n" + "=" * 60)
print("DIRECTION ESTIMATION ACCURACY")
print("=" * 60)
for m in dir_methods:
    acc = dir_accuracy[m] / n_sigs * 100
    print(f"  {m:<20}: {dir_accuracy[m]}/{n_sigs} = {acc:.1f}%")

# RMSE results
print("\n" + "=" * 70)
print(f"RMSE BY METHOD AND BLEND ALPHA ({n_sigs} signatures)")
print("=" * 70)
header = f"{'Method':<20}" + "".join(f"{'α='+str(a):<12}" for a in blend_alphas)
print(header)
print("-" * 70)

best_method, best_alpha, best_rmse = None, None, 1.0
for method in ['oracle', 'correlation', 'majority_vote', 'x_monotone', 'start_pos', 'always_fwd']:
    row = f"{method:<20}"
    for alpha in blend_alphas:
        ex = np.concatenate(all_results[method][alpha]['ex'])
        ey = np.concatenate(all_results[method][alpha]['ey'])
        rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
        row += f"{rmse:>10.5f}  "
        if method not in ['oracle', 'always_fwd'] and rmse < best_rmse:
            best_rmse = rmse
            best_method = method
            best_alpha = alpha
    print(row)

print(f"\nBest non-oracle: {best_method} α={best_alpha} (RMSE: {best_rmse:.5f})")
print(f"LB best: 0.24449, LB avg: 0.25258")

# ============================================================
# Generate final submission
# ============================================================
logging.info(f"\nGenerating submission: {best_method} α={best_alpha}...")

test_preds = []
test_sig_ids = sorted(test['signature_id'].unique())

for sig_id in test_sig_ids:
    sig_test = test[test['signature_id'] == sig_id].sort_values('time')
    n = len(sig_test)

    rows, cols = load_skeleton(sig_id)
    px, py = predict_cdf(rows, cols, n)

    # Pick direction
    if best_method == 'correlation':
        d = direction_by_correlation(px, py, avg_x, avg_y)
    elif best_method == 'majority_vote':
        d = direction_by_all_heuristics(px, py, avg_x, avg_y)
    elif best_method == 'x_monotone':
        d = direction_by_x_monotonicity(px, py)
    elif best_method == 'start_pos':
        d = direction_by_start_position(px, py)
    else:
        d = 'fwd'

    if d == 'rev':
        px, py = px[::-1], py[::-1]

    # Blend
    if best_alpha > 0:
        mx, my = np.mean(px), np.mean(py)
        px = (1 - best_alpha) * px + best_alpha * mx
        py = (1 - best_alpha) * py + best_alpha * my

    for idx, (_, row) in enumerate(sig_test.iterrows()):
        test_preds.append({
            'prediction_id': int(row['prediction_id']),
            'x': float(px[idx]),
            'y': float(py[idx]),
        })

sub = pd.DataFrame(test_preds).sort_values('prediction_id')
assert len(sub) == len(test)

ts = time.strftime('%Y%m%d_%H%M%S')
sub_path = f"{COMP_DIR}/submissions/submission_{ts}_{best_method}_a{best_alpha}.csv"
sub[['prediction_id', 'x', 'y']].to_csv(sub_path, index=False)
logging.info(f"Saved: {sub_path}")
print(f"\nSubmission: {sub.shape}")
print(f"X: [{sub['x'].min():.4f}, {sub['x'].max():.4f}]")
print(f"Y: [{sub['y'].min():.4f}, {sub['y'].max():.4f}]")
print(sub.head())

# Also generate oracle-optimal version
logging.info("Generating oracle-optimal submission (α=0.4)...")
test_preds_v2 = []
for sig_id in test_sig_ids:
    sig_test = test[test['signature_id'] == sig_id].sort_values('time')
    n = len(sig_test)
    rows, cols = load_skeleton(sig_id)
    px, py = predict_cdf(rows, cols, n)

    # Use majority vote for direction
    d = direction_by_all_heuristics(px, py, avg_x, avg_y)
    if d == 'rev':
        px, py = px[::-1], py[::-1]

    # Blend at 0.4
    mx, my = np.mean(px), np.mean(py)
    px = 0.6 * px + 0.4 * mx
    py = 0.6 * py + 0.4 * my

    for idx, (_, row) in enumerate(sig_test.iterrows()):
        test_preds_v2.append({
            'prediction_id': int(row['prediction_id']),
            'x': float(px[idx]),
            'y': float(py[idx]),
        })

sub2 = pd.DataFrame(test_preds_v2).sort_values('prediction_id')
sub2_path = f"{COMP_DIR}/submissions/submission_{ts}_majority_a0.4.csv"
sub2[['prediction_id', 'x', 'y']].to_csv(sub2_path, index=False)
logging.info(f"Saved: {sub2_path}")

# Save experiment
exp = {
    'id': 'v2_cdf_learned_direction',
    'timestamp': ts,
    'method': f'CDF + {best_method} direction + α={best_alpha} blend',
    'direction_method': best_method,
    'blend_alpha': best_alpha,
    'cv_rmse': best_rmse,
    'submission_file': sub_path,
    'alt_submission': sub2_path,
}
with open(f"{COMP_DIR}/experiments.json", 'r') as f:
    experiments = json.load(f)
experiments.append(exp)
with open(f"{COMP_DIR}/experiments.json", 'w') as f:
    json.dump(experiments, f, indent=2)
