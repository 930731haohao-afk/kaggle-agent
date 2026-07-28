"""
Template matching approach:
For each test signature, find similar training signatures and use their
trajectories as templates. Also try ensembling CDF with template matching.

Features for similarity:
- Skeleton pixel count
- Image aspect ratio
- Skeleton spatial moments (mean, std, skew of x and y)
- Skeleton endpoints count
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
IMG_DIR = f"{DATA_DIR}/images_stroke/images"
COMP_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

N_BINS = 100  # Resample trajectories to this many points


def load_skeleton(sig_id, threshold=0.9):
    img = np.array(Image.open(f"{IMG_DIR}/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < threshold
    skeleton = morphology.skeletonize(binary)
    rows, cols = np.where(skeleton)
    return rows, cols, img.shape[:2]


def normalize(vals):
    vr = vals.max() - vals.min()
    return (vals - vals.min()) / vr if vr > 0 else np.zeros_like(vals)


def compute_features(rows, cols, img_shape):
    """Compute features from skeleton for similarity matching."""
    feats = {}
    h, w = img_shape

    if len(rows) < 2:
        return {k: 0 for k in ['n_pixels', 'aspect_ratio', 'x_mean', 'x_std', 'x_skew',
                                'y_mean', 'y_std', 'y_skew', 'density', 'x_range_ratio',
                                'y_range_ratio']}

    nx = normalize(cols.astype(float))
    ny = normalize(rows.astype(float))

    feats['n_pixels'] = len(rows)
    feats['aspect_ratio'] = w / max(h, 1)
    feats['x_mean'] = np.mean(nx)
    feats['x_std'] = np.std(nx)
    feats['x_skew'] = float(pd.Series(nx).skew())
    feats['y_mean'] = np.mean(ny)
    feats['y_std'] = np.std(ny)
    feats['y_skew'] = float(pd.Series(ny).skew())
    feats['density'] = len(rows) / max(h * w, 1) * 10000
    feats['x_range_ratio'] = (cols.max() - cols.min()) / max(w, 1)
    feats['y_range_ratio'] = (rows.max() - rows.min()) / max(h, 1)

    return feats


def column_rmse(px, py, tx, ty):
    ex, ey = px - tx, py - ty
    return np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))


# ============================================================
# Step 1: Compute features and trajectories for all train signatures
# ============================================================
logging.info("Computing features for all train signatures...")

train_features = []
train_trajectories = {}  # sig_id -> (x_curve, y_curve) resampled to N_BINS

for sig_id in sorted(train['signature_id'].unique()):
    rows, cols, shape = load_skeleton(sig_id)
    feats = compute_features(rows, cols, shape)
    feats['signature_id'] = sig_id
    train_features.append(feats)

    # Get trajectory, resample to N_BINS
    sig = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = sig['x'].values, sig['y'].values
    t = np.linspace(0, 1, len(tx))
    t_new = np.linspace(0, 1, N_BINS)
    train_trajectories[sig_id] = (np.interp(t_new, t, tx), np.interp(t_new, t, ty))

train_feat_df = pd.DataFrame(train_features)
feature_cols = [c for c in train_feat_df.columns if c != 'signature_id']

# Normalize features
feat_means = train_feat_df[feature_cols].mean()
feat_stds = train_feat_df[feature_cols].std().replace(0, 1)
train_feat_norm = (train_feat_df[feature_cols] - feat_means) / feat_stds

logging.info(f"Train features: {train_feat_df.shape}")

# ============================================================
# Step 2: CDF prediction helper
# ============================================================
def predict_cdf_with_direction(rows, cols, n, avg_x, avg_y):
    """CDF prediction with correlation-based direction and α=0.6 blend."""
    if len(rows) < 2:
        return np.full(n, 0.5), np.full(n, 0.5)

    nx = normalize(cols.astype(float))
    ny = normalize(rows.astype(float))

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

    # Direction by correlation
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, len(avg_x))
    rx_f = np.interp(t_new, t, pred_x)
    ry_f = np.interp(t_new, t, pred_y)
    rx_r = np.interp(t_new, t, pred_x[::-1])
    ry_r = np.interp(t_new, t, pred_y[::-1])

    corr_f = np.corrcoef(rx_f, avg_x)[0, 1] + np.corrcoef(ry_f, avg_y)[0, 1]
    corr_r = np.corrcoef(rx_r, avg_x)[0, 1] + np.corrcoef(ry_r, avg_y)[0, 1]

    if corr_r > corr_f:
        pred_x, pred_y = pred_x[::-1], pred_y[::-1]

    return pred_x, pred_y


# ============================================================
# Step 3: Template matching prediction
# ============================================================
def predict_template(rows, cols, img_shape, n, k=5):
    """Find k most similar training signatures, average their trajectories."""
    feats = compute_features(rows, cols, img_shape)
    feat_vals = np.array([(feats[c] - feat_means[c]) / feat_stds[c] for c in feature_cols])

    # Find k nearest train signatures
    dists = cdist(feat_vals.reshape(1, -1), train_feat_norm.values)[0]
    nearest_idx = np.argsort(dists)[:k]
    nearest_sigs = train_feat_df.iloc[nearest_idx]['signature_id'].values

    # Average their trajectories (resampled to N_BINS, then resample to n)
    avg_x_curve = np.zeros(N_BINS)
    avg_y_curve = np.zeros(N_BINS)
    for sid in nearest_sigs:
        tx_c, ty_c = train_trajectories[sid]
        avg_x_curve += tx_c
        avg_y_curve += ty_c
    avg_x_curve /= k
    avg_y_curve /= k

    # Resample to target length
    t = np.linspace(0, 1, N_BINS)
    t_new = np.linspace(0, 1, n)
    pred_x = np.interp(t_new, t, avg_x_curve)
    pred_y = np.interp(t_new, t, avg_y_curve)

    return pred_x, pred_y


# ============================================================
# Step 4: Evaluate on train data (leave-one-out style for template)
# ============================================================
logging.info("Evaluating methods on all train data...")

# Compute average trajectory for CDF direction
avg_x_all = np.mean([train_trajectories[sid][0] for sid in train_trajectories], axis=0)
avg_y_all = np.mean([train_trajectories[sid][1] for sid in train_trajectories], axis=0)

sig_ids = sorted(train['signature_id'].unique())
n_sigs = len(sig_ids)

# Configs to try
methods = {
    'cdf_corr_a0.6': {},
    'cdf_corr_a0.5': {},
    'template_k5': {},
    'template_k10': {},
    'template_k20': {},
    'ensemble_cdf0.5_tmpl0.5_k5': {},
    'ensemble_cdf0.6_tmpl0.4_k5': {},
    'ensemble_cdf0.7_tmpl0.3_k5': {},
    'ensemble_cdf0.5_tmpl0.5_k10': {},
    'ensemble_cdf0.6_tmpl0.4_k10': {},
    'ensemble_cdf0.4_tmpl0.3_mean0.3_k5': {},
}

results = {m: {'ex': [], 'ey': []} for m in methods}

t0 = time.time()
for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 200 == 0:
        logging.info(f"  {i+1}/{n_sigs} ({time.time()-t0:.0f}s)")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)

    rows, cols, shape = load_skeleton(sig_id)

    # CDF predictions at different blending levels
    px_cdf_raw, py_cdf_raw = predict_cdf_with_direction(rows, cols, n, avg_x_all, avg_y_all)
    mx, my = np.mean(px_cdf_raw), np.mean(py_cdf_raw)

    px_cdf_a06 = 0.4 * px_cdf_raw + 0.6 * mx
    py_cdf_a06 = 0.4 * py_cdf_raw + 0.6 * my
    results['cdf_corr_a0.6']['ex'].append(px_cdf_a06 - tx)
    results['cdf_corr_a0.6']['ey'].append(py_cdf_a06 - ty)

    px_cdf_a05 = 0.5 * px_cdf_raw + 0.5 * mx
    py_cdf_a05 = 0.5 * py_cdf_raw + 0.5 * my
    results['cdf_corr_a0.5']['ex'].append(px_cdf_a05 - tx)
    results['cdf_corr_a0.5']['ey'].append(py_cdf_a05 - ty)

    # Template predictions (leave-one-out: exclude current signature)
    # Temporarily remove current from training features
    mask = train_feat_df['signature_id'] != sig_id
    temp_feat_norm = train_feat_norm[mask].reset_index(drop=True)
    temp_feat_df = train_feat_df[mask].reset_index(drop=True)

    feats = compute_features(rows, cols, shape)
    feat_vals = np.array([(feats[c] - feat_means[c]) / feat_stds[c] for c in feature_cols])
    dists = cdist(feat_vals.reshape(1, -1), temp_feat_norm.values)[0]

    for k in [5, 10, 20]:
        nearest_idx = np.argsort(dists)[:k]
        nearest_sigs = temp_feat_df.iloc[nearest_idx]['signature_id'].values

        avg_xc = np.zeros(N_BINS)
        avg_yc = np.zeros(N_BINS)
        for sid in nearest_sigs:
            txc, tyc = train_trajectories[sid]
            avg_xc += txc
            avg_yc += tyc
        avg_xc /= k
        avg_yc /= k

        t = np.linspace(0, 1, N_BINS)
        t_new = np.linspace(0, 1, n)
        px_tmpl = np.interp(t_new, t, avg_xc)
        py_tmpl = np.interp(t_new, t, avg_yc)

        results[f'template_k{k}']['ex'].append(px_tmpl - tx)
        results[f'template_k{k}']['ey'].append(py_tmpl - ty)

    # Ensembles: CDF + Template
    # Use template k=5 and k=10
    for k in [5, 10]:
        nearest_idx = np.argsort(dists)[:k]
        nearest_sigs = temp_feat_df.iloc[nearest_idx]['signature_id'].values
        avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs], axis=0)
        avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs], axis=0)
        t = np.linspace(0, 1, N_BINS)
        t_new = np.linspace(0, 1, n)
        px_tmpl = np.interp(t_new, t, avg_xc)
        py_tmpl = np.interp(t_new, t, avg_yc)

        for cdf_w, tmpl_w in [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]:
            name = f'ensemble_cdf{cdf_w}_tmpl{tmpl_w}_k{k}'
            if name in results:
                px_ens = cdf_w * px_cdf_a05 + tmpl_w * px_tmpl
                py_ens = cdf_w * py_cdf_a05 + tmpl_w * py_tmpl
                results[name]['ex'].append(px_ens - tx)
                results[name]['ey'].append(py_ens - ty)

    # CDF + Template + Mean
    nearest_idx = np.argsort(dists)[:5]
    nearest_sigs = temp_feat_df.iloc[nearest_idx]['signature_id'].values
    avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs], axis=0)
    avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs], axis=0)
    t = np.linspace(0, 1, N_BINS)
    t_new = np.linspace(0, 1, n)
    px_tmpl = np.interp(t_new, t, avg_xc)
    py_tmpl = np.interp(t_new, t, avg_yc)
    px_ens3 = 0.4 * px_cdf_raw + 0.3 * px_tmpl + 0.3 * mx
    py_ens3 = 0.4 * py_cdf_raw + 0.3 * py_tmpl + 0.3 * my
    results['ensemble_cdf0.4_tmpl0.3_mean0.3_k5']['ex'].append(px_ens3 - tx)
    results['ensemble_cdf0.4_tmpl0.3_mean0.3_k5']['ey'].append(py_ens3 - ty)

logging.info(f"Done in {time.time()-t0:.1f}s")

# Report
print("\n" + "=" * 60)
print(f"RESULTS (all {n_sigs} train signatures)")
print("=" * 60)
scored = []
for m in methods:
    ex = np.concatenate(results[m]['ex'])
    ey = np.concatenate(results[m]['ey'])
    rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
    scored.append((m, rmse))
scored.sort(key=lambda x: x[1])

print(f"{'Method':<45} {'RMSE':>10}")
print("-" * 57)
for name, rmse in scored:
    marker = " <<<" if rmse < 0.249 else ""
    print(f"{name:<45} {rmse:>10.5f}{marker}")

best_name, best_rmse = scored[0]
print(f"\n>>> Best: {best_name} (RMSE: {best_rmse:.5f})")
print(f">>> LB best: 0.24449, LB avg: 0.25258")

# ============================================================
# Generate submission with best method
# ============================================================
logging.info("Generating final submission...")

# Parse best config
test_preds = []
test_sig_ids = sorted(test['signature_id'].unique())

for sig_id in test_sig_ids:
    sig_test = test[test['signature_id'] == sig_id].sort_values('time')
    n = len(sig_test)
    rows, cols, shape = load_skeleton(sig_id)

    # CDF prediction
    px_cdf, py_cdf = predict_cdf_with_direction(rows, cols, n, avg_x_all, avg_y_all)
    mx, my = np.mean(px_cdf), np.mean(py_cdf)

    # Template prediction
    feats = compute_features(rows, cols, shape)
    feat_vals = np.array([(feats[c] - feat_means[c]) / feat_stds[c] for c in feature_cols])
    dists = cdist(feat_vals.reshape(1, -1), train_feat_norm.values)[0]
    nearest_idx = np.argsort(dists)[:5]
    nearest_sigs = train_feat_df.iloc[nearest_idx]['signature_id'].values
    avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs], axis=0)
    avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs], axis=0)
    t = np.linspace(0, 1, N_BINS)
    t_new = np.linspace(0, 1, n)
    px_tmpl = np.interp(t_new, t, avg_xc)
    py_tmpl = np.interp(t_new, t, avg_yc)

    # Use best method
    if 'ensemble' in best_name:
        # Parse weights from name
        if best_name == 'ensemble_cdf0.4_tmpl0.3_mean0.3_k5':
            px = 0.4 * px_cdf + 0.3 * px_tmpl + 0.3 * mx
            py = 0.4 * py_cdf + 0.3 * py_tmpl + 0.3 * my
        elif 'cdf0.5_tmpl0.5' in best_name:
            px_cdf_b = 0.5 * px_cdf + 0.5 * mx
            py_cdf_b = 0.5 * py_cdf + 0.5 * my
            px = 0.5 * px_cdf_b + 0.5 * px_tmpl
            py = 0.5 * py_cdf_b + 0.5 * py_tmpl
        elif 'cdf0.6_tmpl0.4' in best_name:
            px_cdf_b = 0.5 * px_cdf + 0.5 * mx
            py_cdf_b = 0.5 * py_cdf + 0.5 * my
            px = 0.6 * px_cdf_b + 0.4 * px_tmpl
            py = 0.6 * py_cdf_b + 0.4 * py_tmpl
        elif 'cdf0.7_tmpl0.3' in best_name:
            px_cdf_b = 0.5 * px_cdf + 0.5 * mx
            py_cdf_b = 0.5 * py_cdf + 0.5 * my
            px = 0.7 * px_cdf_b + 0.3 * px_tmpl
            py = 0.7 * py_cdf_b + 0.3 * py_tmpl
        else:
            px = 0.4 * px_cdf + 0.6 * mx
            py = 0.4 * py_cdf + 0.6 * my
    elif 'template' in best_name:
        px, py = px_tmpl, py_tmpl
    else:
        # CDF with blending
        if 'a0.6' in best_name:
            px = 0.4 * px_cdf + 0.6 * mx
            py = 0.4 * py_cdf + 0.6 * my
        else:
            px = 0.5 * px_cdf + 0.5 * mx
            py = 0.5 * py_cdf + 0.5 * my

    for idx, (_, row) in enumerate(sig_test.iterrows()):
        test_preds.append({
            'prediction_id': int(row['prediction_id']),
            'x': float(np.clip(px[idx], 0, 1)),
            'y': float(np.clip(py[idx], 0, 1)),
        })

sub = pd.DataFrame(test_preds).sort_values('prediction_id')
assert len(sub) == len(test)

ts = time.strftime('%Y%m%d_%H%M%S')
sub_path = f"{COMP_DIR}/submissions/submission_{ts}_final.csv"
sub[['prediction_id', 'x', 'y']].to_csv(sub_path, index=False)
logging.info(f"Saved: {sub_path}")
print(f"\nSubmission: {sub.shape}")
print(f"X: [{sub['x'].min():.4f}, {sub['x'].max():.4f}]")
print(f"Y: [{sub['y'].min():.4f}, {sub['y'].max():.4f}]")

# Save experiment
exp = {
    'id': 'v4_template_ensemble',
    'timestamp': ts,
    'method': best_name,
    'cv_rmse': float(best_rmse),
    'submission_file': sub_path,
}
with open(f"{COMP_DIR}/experiments.json", 'r') as f:
    experiments = json.load(f)
experiments.append(exp)
with open(f"{COMP_DIR}/experiments.json", 'w') as f:
    json.dump(experiments, f, indent=2)
