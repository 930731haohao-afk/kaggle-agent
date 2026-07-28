"""
Refined solution with better direction estimation and blending.

Key insight: avg trajectory goes RIGHT-to-LEFT (x: 0.682 -> 0.530).
CDF naturally goes left-to-right, so default should be REVERSED.

Also try: per-writer pattern learning, y-parameterized CDF, etc.
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


def normalize(vals):
    vr = vals.max() - vals.min()
    return (vals - vals.min()) / vr if vr > 0 else np.zeros_like(vals)


def column_rmse(px, py, tx, ty):
    ex, ey = px - tx, py - ty
    return np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))


# ============================================================
# Method: CDF prediction with configurable axis and direction
# ============================================================
def predict_cdf(rows, cols, n, sort_by='x', direction='fwd', smooth_y_sigma=None):
    """
    CDF-based prediction.
    sort_by: 'x' (sort skeleton by x-coord), 'y' (sort by y-coord)
    direction: 'fwd' (as sorted) or 'rev' (reversed)
    """
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)

    nx = normalize(cols.astype(float))
    ny = normalize(rows.astype(float))

    if sort_by == 'x':
        sort_idx = np.argsort(nx)
    else:
        sort_idx = np.argsort(ny)

    sorted_x = nx[sort_idx]
    sorted_y = ny[sort_idx]

    if direction == 'rev':
        sorted_x = sorted_x[::-1]
        sorted_y = sorted_y[::-1]

    cdf = np.linspace(0, 1, len(sorted_x))
    t_fracs = np.linspace(0, 1, n)

    pred_x = np.interp(t_fracs, cdf, sorted_x)

    # y from sliding window
    window = max(1, len(sorted_y) // (2 * n))
    pred_y = np.zeros(n)
    for i, frac in enumerate(t_fracs):
        idx = int(frac * (len(sorted_y) - 1))
        lo = max(0, idx - window)
        hi = min(len(sorted_y), idx + window + 1)
        pred_y[i] = np.mean(sorted_y[lo:hi])

    sigma = smooth_y_sigma if smooth_y_sigma else max(1, n // 30)
    if n > 5:
        pred_y = gaussian_filter1d(pred_y, sigma=sigma)
    pred_y = np.clip(pred_y, 0, 1)

    return pred_x, pred_y


# ============================================================
# Compute average trajectory templates
# ============================================================
logging.info("Computing trajectory templates...")

N_BINS = 100
all_x_curves = []
all_y_curves = []

for sig_id in sorted(train['signature_id'].unique()):
    sig = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = sig['x'].values, sig['y'].values
    n = len(tx)
    if n < 3:
        continue
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, N_BINS)
    all_x_curves.append(np.interp(t_new, t, tx))
    all_y_curves.append(np.interp(t_new, t, ty))

avg_x = np.mean(all_x_curves, axis=0)
avg_y = np.mean(all_y_curves, axis=0)


def correlation_direction(pred_x, pred_y, avg_x, avg_y, n_bins=100):
    """Pick direction by correlation with average trajectory."""
    t = np.linspace(0, 1, len(pred_x))
    t_new = np.linspace(0, 1, n_bins)
    rx_f = np.interp(t_new, t, pred_x)
    ry_f = np.interp(t_new, t, pred_y)
    rx_r = np.interp(t_new, t, pred_x[::-1])
    ry_r = np.interp(t_new, t, pred_y[::-1])

    corr_f = np.corrcoef(rx_f, avg_x)[0, 1] + np.corrcoef(ry_f, avg_y)[0, 1]
    corr_r = np.corrcoef(rx_r, avg_x)[0, 1] + np.corrcoef(ry_r, avg_y)[0, 1]

    return 'fwd' if corr_f >= corr_r else 'rev'


# ============================================================
# Evaluate comprehensive configs
# ============================================================
logging.info("Comprehensive evaluation...")

configs = []
for sort_by in ['x']:
    for base_dir in ['fwd', 'rev']:
        for use_corr_dir in [False, True]:
            for alpha in [0.0, 0.3, 0.4, 0.5, 0.6]:
                name = f"sort_{sort_by}_{base_dir}_corr{use_corr_dir}_a{alpha}"
                configs.append({
                    'name': name, 'sort_by': sort_by, 'base_dir': base_dir,
                    'use_corr_dir': use_corr_dir, 'alpha': alpha
                })

# Also add "always reversed" (since avg trajectory goes right-to-left)
configs.append({'name': 'always_rev_a0.0', 'sort_by': 'x', 'base_dir': 'rev',
                'use_corr_dir': False, 'alpha': 0.0})
configs.append({'name': 'always_rev_a0.3', 'sort_by': 'x', 'base_dir': 'rev',
                'use_corr_dir': False, 'alpha': 0.3})
configs.append({'name': 'always_rev_a0.5', 'sort_by': 'x', 'base_dir': 'rev',
                'use_corr_dir': False, 'alpha': 0.5})

sig_ids = sorted(train['signature_id'].unique())
results = {c['name']: {'ex': [], 'ey': []} for c in configs}

t0 = time.time()
for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 200 == 0:
        logging.info(f"  {i+1}/{len(sig_ids)} ({time.time()-t0:.0f}s)")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)
    rows, cols = load_skeleton(sig_id)

    for c in configs:
        px, py = predict_cdf(rows, cols, n, sort_by=c['sort_by'], direction=c['base_dir'])

        # Direction override
        if c['use_corr_dir']:
            d = correlation_direction(px, py, avg_x, avg_y)
            if d == 'rev':
                px, py = px[::-1], py[::-1]

        # Blend
        if c['alpha'] > 0:
            mx, my = np.mean(px), np.mean(py)
            px = (1 - c['alpha']) * px + c['alpha'] * mx
            py = (1 - c['alpha']) * py + c['alpha'] * my

        results[c['name']]['ex'].append(px - tx)
        results[c['name']]['ey'].append(py - ty)

logging.info(f"Done in {time.time()-t0:.1f}s")

# Sort by RMSE
print("\n" + "=" * 60)
print(f"TOP RESULTS (all {len(sig_ids)} train signatures)")
print("=" * 60)
print(f"{'Config':<50} {'RMSE':>10}")
print("-" * 62)

scored = []
for c in configs:
    ex = np.concatenate(results[c['name']]['ex'])
    ey = np.concatenate(results[c['name']]['ey'])
    rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
    scored.append((c['name'], rmse, c))

scored.sort(key=lambda x: x[1])
for name, rmse, cfg in scored[:20]:
    marker = " <<<" if rmse < 0.25 else ""
    print(f"{name:<50} {rmse:>10.5f}{marker}")

best_name, best_rmse, best_cfg = scored[0]
print(f"\n>>> Best: {best_name} (RMSE: {best_rmse:.5f})")
print(f">>> LB best: 0.24449, LB avg: 0.25258")

# ============================================================
# Generate submission with best config
# ============================================================
logging.info(f"\nGenerating submission with best config...")
test_preds = []
test_sig_ids = sorted(test['signature_id'].unique())

for sig_id in test_sig_ids:
    sig_test = test[test['signature_id'] == sig_id].sort_values('time')
    n = len(sig_test)
    rows, cols = load_skeleton(sig_id)

    px, py = predict_cdf(rows, cols, n,
                         sort_by=best_cfg['sort_by'],
                         direction=best_cfg['base_dir'])

    if best_cfg['use_corr_dir']:
        d = correlation_direction(px, py, avg_x, avg_y)
        if d == 'rev':
            px, py = px[::-1], py[::-1]

    if best_cfg['alpha'] > 0:
        mx, my = np.mean(px), np.mean(py)
        px = (1 - best_cfg['alpha']) * px + best_cfg['alpha'] * mx
        py = (1 - best_cfg['alpha']) * py + best_cfg['alpha'] * my

    for idx, (_, row) in enumerate(sig_test.iterrows()):
        test_preds.append({
            'prediction_id': int(row['prediction_id']),
            'x': float(px[idx]),
            'y': float(py[idx]),
        })

sub = pd.DataFrame(test_preds).sort_values('prediction_id')
assert len(sub) == len(test)
assert sub['x'].between(-0.01, 1.01).all()
assert sub['y'].between(-0.01, 1.01).all()

ts = time.strftime('%Y%m%d_%H%M%S')
sub_path = f"{COMP_DIR}/submissions/submission_{ts}_best.csv"
sub[['prediction_id', 'x', 'y']].to_csv(sub_path, index=False)
logging.info(f"Saved: {sub_path}")
print(f"\nSubmission: {sub.shape}")
print(sub.head())

# Save experiment
exp = {
    'id': 'v3_refined',
    'timestamp': ts,
    'config': best_cfg,
    'cv_rmse': float(best_rmse),
    'submission_file': sub_path,
}
with open(f"{COMP_DIR}/experiments.json", 'r') as f:
    experiments = json.load(f)
experiments.append(exp)
with open(f"{COMP_DIR}/experiments.json", 'w') as f:
    json.dump(experiments, f, indent=2)
