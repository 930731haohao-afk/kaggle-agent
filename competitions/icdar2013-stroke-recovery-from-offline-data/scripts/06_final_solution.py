"""
Final stroke recovery solution.

Best approach: CDF-based prediction blended with mean.
Evaluate with:
1. Oracle direction picking (upper bound)
2. Always-forward (left-to-right)
3. Smoothness-based direction picking

Then generate submission on test data.
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.ndimage import gaussian_filter1d
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


def predict_cdf_blended(rows, cols, n, blend_alpha=0.5):
    """CDF-based prediction blended with mean.

    x(t) from CDF of skeleton x-coordinates (monotone increasing).
    y(t) from conditional mean of y given x position.
    Then blend with the signature mean.
    """
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)

    nx, ny = normalize(cols.astype(float), rows.astype(float))

    # Sort by x
    sort_idx = np.argsort(nx)
    sorted_x = nx[sort_idx]
    sorted_y = ny[sort_idx]

    # CDF of x
    cdf = np.linspace(0, 1, len(sorted_x))
    t_fracs = np.linspace(0, 1, n)

    # x from CDF
    pred_x = np.interp(t_fracs, cdf, sorted_x)

    # y from sliding window mean
    window = max(1, len(sorted_y) // (2 * n))
    pred_y = np.zeros(n)
    for i, frac in enumerate(t_fracs):
        idx = int(frac * (len(sorted_y) - 1))
        lo = max(0, idx - window)
        hi = min(len(sorted_y), idx + window + 1)
        pred_y[i] = np.mean(sorted_y[lo:hi])

    # Smooth y
    if n > 5:
        pred_y = gaussian_filter1d(pred_y, sigma=max(1, n // 30))
    pred_y = np.clip(pred_y, 0, 1)

    # Blend with mean
    if blend_alpha > 0:
        mean_x = np.mean(nx)
        mean_y = np.mean(ny)
        pred_x = (1 - blend_alpha) * pred_x + blend_alpha * mean_x
        pred_y = (1 - blend_alpha) * pred_y + blend_alpha * mean_y

    return pred_x, pred_y


def smoothness_score(x, y):
    """Lower = smoother path."""
    if len(x) < 3:
        return 0
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2)
    # Penalize speed variance (smooth paths have constant speed)
    return np.std(speed) + 0.1 * np.mean(np.abs(np.diff(dx)) + np.abs(np.diff(dy)))


# ============================================================
# Full evaluation on ALL train data
# ============================================================
logging.info("Full evaluation on all train signatures...")

blend_alphas = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]

# Store predictions for blending experiments
all_predictions = {}  # blend -> list of (pred_x, pred_y, true_x, true_y)

sig_ids = sorted(train['signature_id'].unique())
n_sigs = len(sig_ids)

for alpha in blend_alphas:
    all_predictions[alpha] = {'px': [], 'py': [], 'tx': [], 'ty': []}

t0 = time.time()
for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 100 == 0:
        logging.info(f"  {i+1}/{n_sigs} ({time.time()-t0:.0f}s)")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)

    rows, cols = load_skeleton(sig_id)

    for alpha in blend_alphas:
        px, py = predict_cdf_blended(rows, cols, n, blend_alpha=alpha)

        all_predictions[alpha]['px'].append(px)
        all_predictions[alpha]['py'].append(py)
        all_predictions[alpha]['tx'].append(tx)
        all_predictions[alpha]['ty'].append(ty)

logging.info(f"Done in {time.time()-t0:.1f}s")

# ============================================================
# Compute RMSE for each scenario
# ============================================================
print("\n" + "=" * 70)
print(f"FULL TRAIN EVALUATION ({n_sigs} signatures)")
print("=" * 70)
print(f"{'Blend α':<10} {'Oracle':>10} {'Forward':>10} {'Smoothness':>10}")
print("-" * 42)

best_alpha_oracle = None
best_rmse_oracle = 1.0
best_alpha_fwd = None
best_rmse_fwd = 1.0

for alpha in blend_alphas:
    all_ex_oracle, all_ey_oracle = [], []
    all_ex_fwd, all_ey_fwd = [], []
    all_ex_smooth, all_ey_smooth = [], []

    for j in range(n_sigs):
        px = all_predictions[alpha]['px'][j]
        py = all_predictions[alpha]['py'][j]
        tx = all_predictions[alpha]['tx'][j]
        ty = all_predictions[alpha]['ty'][j]

        # Forward (always left-to-right)
        all_ex_fwd.append(px - tx)
        all_ey_fwd.append(py - ty)

        # Oracle (try both, pick best)
        rmse_f = column_rmse(px, py, tx, ty)
        rmse_r = column_rmse(px[::-1], py[::-1], tx, ty)
        if rmse_f <= rmse_r:
            all_ex_oracle.append(px - tx)
            all_ey_oracle.append(py - ty)
        else:
            all_ex_oracle.append(px[::-1] - tx)
            all_ey_oracle.append(py[::-1] - ty)

        # Smoothness-based
        s_f = smoothness_score(px, py)
        s_r = smoothness_score(px[::-1], py[::-1])
        if s_f <= s_r:
            all_ex_smooth.append(px - tx)
            all_ey_smooth.append(py - ty)
        else:
            all_ex_smooth.append(px[::-1] - tx)
            all_ey_smooth.append(py[::-1] - ty)

    ex_o, ey_o = np.concatenate(all_ex_oracle), np.concatenate(all_ey_oracle)
    rmse_oracle = np.sqrt((np.sum(ex_o**2) + np.sum(ey_o**2)) / (2 * len(ex_o)))

    ex_f, ey_f = np.concatenate(all_ex_fwd), np.concatenate(all_ey_fwd)
    rmse_fwd = np.sqrt((np.sum(ex_f**2) + np.sum(ey_f**2)) / (2 * len(ex_f)))

    ex_s, ey_s = np.concatenate(all_ex_smooth), np.concatenate(all_ey_smooth)
    rmse_smooth = np.sqrt((np.sum(ex_s**2) + np.sum(ey_s**2)) / (2 * len(ex_s)))

    print(f"{alpha:<10.1f} {rmse_oracle:>10.5f} {rmse_fwd:>10.5f} {rmse_smooth:>10.5f}")

    if rmse_oracle < best_rmse_oracle:
        best_rmse_oracle = rmse_oracle
        best_alpha_oracle = alpha
    if rmse_fwd < best_rmse_fwd:
        best_rmse_fwd = rmse_fwd
        best_alpha_fwd = alpha

print(f"\nBest oracle: α={best_alpha_oracle} (RMSE: {best_rmse_oracle:.5f})")
print(f"Best forward: α={best_alpha_fwd} (RMSE: {best_rmse_fwd:.5f})")
print(f"LB best: 0.24449, LB avg: 0.25258")

# ============================================================
# Generate test predictions with best config
# ============================================================
logging.info(f"\nGenerating test predictions with α={best_alpha_fwd}...")

test_preds = []
test_sig_ids = sorted(test['signature_id'].unique())

for sig_id in test_sig_ids:
    sig_test = test[test['signature_id'] == sig_id].sort_values('time')
    n = len(sig_test)

    rows, cols = load_skeleton(sig_id)
    px, py = predict_cdf_blended(rows, cols, n, blend_alpha=best_alpha_fwd)

    for idx, (_, row) in enumerate(sig_test.iterrows()):
        test_preds.append({
            'prediction_id': int(row['prediction_id']),
            'x': float(px[idx]),
            'y': float(py[idx]),
        })

sub = pd.DataFrame(test_preds)
sub = sub.sort_values('prediction_id')

# Validate
assert len(sub) == len(test), f"Expected {len(test)} rows, got {len(sub)}"
assert sub['x'].between(0, 1).all(), "x values out of range"
assert sub['y'].between(0, 1).all(), "y values out of range"

# Save
ts = time.strftime('%Y%m%d_%H%M%S')
sub_path = f"{COMP_DIR}/submissions/submission_{ts}_cdf_blend{best_alpha_fwd:.1f}.csv"
sub[['prediction_id', 'x', 'y']].to_csv(sub_path, index=False)
logging.info(f"Saved submission to {sub_path}")
print(f"\nSubmission shape: {sub.shape}")
print(f"X range: [{sub['x'].min():.4f}, {sub['x'].max():.4f}]")
print(f"Y range: [{sub['y'].min():.4f}, {sub['y'].max():.4f}]")
print(sub.head(10))

# Also save with oracle-optimal alpha
logging.info(f"Generating oracle-optimal submission with α={best_alpha_oracle}...")
test_preds_oracle = []
for sig_id in test_sig_ids:
    sig_test = test[test['signature_id'] == sig_id].sort_values('time')
    n = len(sig_test)
    rows, cols = load_skeleton(sig_id)
    px, py = predict_cdf_blended(rows, cols, n, blend_alpha=best_alpha_oracle)
    for idx, (_, row) in enumerate(sig_test.iterrows()):
        test_preds_oracle.append({
            'prediction_id': int(row['prediction_id']),
            'x': float(px[idx]),
            'y': float(py[idx]),
        })

sub_oracle = pd.DataFrame(test_preds_oracle).sort_values('prediction_id')
sub_oracle_path = f"{COMP_DIR}/submissions/submission_{ts}_cdf_blend{best_alpha_oracle:.1f}_oracle_opt.csv"
sub_oracle[['prediction_id', 'x', 'y']].to_csv(sub_oracle_path, index=False)
logging.info(f"Saved oracle-optimized submission to {sub_oracle_path}")

# Save experiment results
experiment = {
    'id': 'v1_cdf_blended',
    'timestamp': ts,
    'method': 'CDF-based prediction blended with mean',
    'best_alpha_forward': best_alpha_fwd,
    'best_alpha_oracle': best_alpha_oracle,
    'cv_rmse_forward': best_rmse_fwd,
    'cv_rmse_oracle': best_rmse_oracle,
    'submission_file': sub_path,
    'notes': 'CDF maps time fraction to x-percentile, y from sliding window mean, blended with skeleton mean',
}
exp_path = f"{COMP_DIR}/experiments.json"
with open(exp_path, 'r') as f:
    experiments = json.load(f)
experiments.append(experiment)
with open(exp_path, 'w') as f:
    json.dump(experiments, f, indent=2)
logging.info(f"Experiment logged to {exp_path}")
