"""
Final ensemble: CDF + writer-held-out template matching.
Also try richer image features for better template matching.
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
IMG_DIR = f"{DATA_DIR}/images_stroke/images"
COMP_DIR = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

N_BINS = 100


def load_skeleton(sig_id, threshold=0.9):
    img = np.array(Image.open(f"{IMG_DIR}/{sig_id:04d}.jpg").convert('L')) / 255.0
    binary = img < threshold
    skeleton = morphology.skeletonize(binary)
    rows, cols = np.where(skeleton)
    return rows, cols, img.shape[:2], img


def normalize(vals):
    vr = vals.max() - vals.min()
    return (vals - vals.min()) / vr if vr > 0 else np.zeros_like(vals)


def compute_rich_features(rows, cols, img_shape, img):
    """Richer features including image-level statistics."""
    h, w = img_shape
    feats = {}
    if len(rows) < 2:
        return {k: 0.0 for k in ['n_pixels', 'aspect_ratio', 'x_mean', 'x_std', 'x_skew', 'x_kurt',
                                   'y_mean', 'y_std', 'y_skew', 'y_kurt', 'density',
                                   'x_range_ratio', 'y_range_ratio',
                                   'img_mean', 'img_std', 'ink_fraction',
                                   'n_components', 'skel_length_ratio',
                                   'x_q25', 'x_q75', 'y_q25', 'y_q75',
                                   'xy_corr']}

    nx = normalize(cols.astype(float))
    ny = normalize(rows.astype(float))

    feats['n_pixels'] = float(len(rows))
    feats['aspect_ratio'] = float(w / max(h, 1))
    feats['x_mean'] = float(np.mean(nx))
    feats['x_std'] = float(np.std(nx))
    feats['x_skew'] = float(skew(nx))
    feats['x_kurt'] = float(kurtosis(nx))
    feats['y_mean'] = float(np.mean(ny))
    feats['y_std'] = float(np.std(ny))
    feats['y_skew'] = float(skew(ny))
    feats['y_kurt'] = float(kurtosis(ny))
    feats['density'] = float(len(rows) / max(h * w, 1) * 10000)
    feats['x_range_ratio'] = float((cols.max() - cols.min()) / max(w, 1))
    feats['y_range_ratio'] = float((rows.max() - rows.min()) / max(h, 1))

    # Image-level features
    feats['img_mean'] = float(np.mean(img))
    feats['img_std'] = float(np.std(img))
    feats['ink_fraction'] = float(np.mean(img < 0.9))

    # Skeleton topology (approximate)
    skel_img = np.zeros(img_shape, dtype=bool)
    skel_img[rows, cols] = True
    from scipy.ndimage import label as ndlabel
    n_components_val, _ = ndlabel(skel_img)
    feats['n_components'] = float(n_components_val.max())
    feats['skel_length_ratio'] = float(len(rows) / max(np.sqrt(h*w), 1))

    # Quantiles
    feats['x_q25'] = float(np.percentile(nx, 25))
    feats['x_q75'] = float(np.percentile(nx, 75))
    feats['y_q25'] = float(np.percentile(ny, 25))
    feats['y_q75'] = float(np.percentile(ny, 75))

    # Correlation
    feats['xy_corr'] = float(np.corrcoef(nx, ny)[0, 1]) if len(nx) > 2 else 0.0

    return feats


def column_rmse(px, py, tx, ty):
    ex, ey = px - tx, py - ty
    return np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))


def predict_cdf_with_direction(rows, cols, n, avg_x, avg_y):
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
# Build features and trajectories
# ============================================================
logging.info("Building rich features...")

train_features = []
train_trajectories = {}
sig_to_writer = {}

for sig_id in sorted(train['signature_id'].unique()):
    rows, cols, shape, img = load_skeleton(sig_id)
    feats = compute_rich_features(rows, cols, shape, img)
    feats['signature_id'] = sig_id
    writer_id = train[train['signature_id'] == sig_id]['writer_id'].iloc[0]
    feats['writer_id'] = writer_id
    sig_to_writer[sig_id] = writer_id
    train_features.append(feats)

    sig = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = sig['x'].values, sig['y'].values
    t = np.linspace(0, 1, len(tx))
    t_new = np.linspace(0, 1, N_BINS)
    train_trajectories[sig_id] = (np.interp(t_new, t, tx), np.interp(t_new, t, ty))

train_feat_df = pd.DataFrame(train_features)
feature_cols = [c for c in train_feat_df.columns if c not in ['signature_id', 'writer_id']]
feat_means = train_feat_df[feature_cols].mean()
feat_stds = train_feat_df[feature_cols].std().replace(0, 1)
train_feat_norm = (train_feat_df[feature_cols] - feat_means) / feat_stds

# Average trajectory
avg_x_all = np.mean([train_trajectories[sid][0] for sid in train_trajectories], axis=0)
avg_y_all = np.mean([train_trajectories[sid][1] for sid in train_trajectories], axis=0)

logging.info(f"Features: {len(feature_cols)} columns")

# ============================================================
# Writer-held-out evaluation
# ============================================================
logging.info("Writer-held-out evaluation with ensembles...")

methods = {
    'cdf_corr_a0.6': {},
    'cdf_corr_a0.5': {},
    'tmpl_k50_writer': {},
    'tmpl_k100_writer': {},
    'tmpl_k30_writer': {},
    'rich_tmpl_k50_writer': {},
    'ens_cdf0.4_tmpl50_0.6': {},
    'ens_cdf0.5_tmpl50_0.5': {},
    'ens_cdf0.3_tmpl50_0.7': {},
    'ens_cdf0.4_tmpl100_0.6': {},
    'ens_cdf0.5_tmpl100_0.5': {},
    'global_mean_traj': {},
}

results = {m: {'ex': [], 'ey': []} for m in methods}
sig_ids = sorted(train['signature_id'].unique())

# Simple features for comparison
simple_feature_cols = ['n_pixels', 'aspect_ratio', 'x_mean', 'x_std', 'x_skew',
                       'y_mean', 'y_std', 'y_skew', 'density', 'x_range_ratio', 'y_range_ratio']
train_simple_norm = (train_feat_df[simple_feature_cols] - train_feat_df[simple_feature_cols].mean()) / train_feat_df[simple_feature_cols].std().replace(0, 1)

t0 = time.time()
for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 200 == 0:
        logging.info(f"  {i+1}/{len(sig_ids)} ({time.time()-t0:.0f}s)")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)
    writer_id = sig_to_writer[sig_id]

    rows, cols, shape, img = load_skeleton(sig_id)

    # CDF predictions
    px_cdf, py_cdf = predict_cdf_with_direction(rows, cols, n, avg_x_all, avg_y_all)
    mx, my = np.mean(px_cdf), np.mean(py_cdf)

    px_cdf_a06 = 0.4 * px_cdf + 0.6 * mx
    py_cdf_a06 = 0.4 * py_cdf + 0.6 * my
    results['cdf_corr_a0.6']['ex'].append(px_cdf_a06 - tx)
    results['cdf_corr_a0.6']['ey'].append(py_cdf_a06 - ty)

    px_cdf_a05 = 0.5 * px_cdf + 0.5 * mx
    py_cdf_a05 = 0.5 * py_cdf + 0.5 * my
    results['cdf_corr_a0.5']['ex'].append(px_cdf_a05 - tx)
    results['cdf_corr_a0.5']['ey'].append(py_cdf_a05 - ty)

    # Writer-held-out template matching (simple features)
    mask_w = train_feat_df['writer_id'] != writer_id
    temp_df_w = train_feat_df[mask_w].reset_index(drop=True)
    temp_simple_norm_w = train_simple_norm[mask_w].reset_index(drop=True)

    feats_simple = np.array([(compute_rich_features(rows, cols, shape, img)[c] - train_feat_df[simple_feature_cols].mean()[c]) / train_feat_df[simple_feature_cols].std().replace(0, 1)[c] for c in simple_feature_cols])
    dists_simple = cdist(feats_simple.reshape(1, -1), temp_simple_norm_w.values)[0]

    for k in [30, 50, 100]:
        nearest_k = np.argsort(dists_simple)[:k]
        nearest_sigs = temp_df_w.iloc[nearest_k]['signature_id'].values
        avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs], axis=0)
        avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs], axis=0)
        t_orig = np.linspace(0, 1, N_BINS)
        t_new = np.linspace(0, 1, n)
        px_tmpl = np.interp(t_new, t_orig, avg_xc)
        py_tmpl = np.interp(t_new, t_orig, avg_yc)
        results[f'tmpl_k{k}_writer']['ex'].append(px_tmpl - tx)
        results[f'tmpl_k{k}_writer']['ey'].append(py_tmpl - ty)

    # Rich features template
    temp_rich_norm_w = train_feat_norm[mask_w].reset_index(drop=True)
    feats_rich = np.array([(compute_rich_features(rows, cols, shape, img)[c] - feat_means[c]) / feat_stds[c] for c in feature_cols])
    dists_rich = cdist(feats_rich.reshape(1, -1), temp_rich_norm_w.values)[0]
    nearest_50 = np.argsort(dists_rich)[:50]
    nearest_sigs_rich = temp_df_w.iloc[nearest_50]['signature_id'].values
    avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs_rich], axis=0)
    avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs_rich], axis=0)
    px_tmpl_rich = np.interp(t_new, t_orig, avg_xc)
    py_tmpl_rich = np.interp(t_new, t_orig, avg_yc)
    results['rich_tmpl_k50_writer']['ex'].append(px_tmpl_rich - tx)
    results['rich_tmpl_k50_writer']['ey'].append(py_tmpl_rich - ty)

    # Ensembles: CDF (with direction + blend) + Template (writer-held-out)
    # Template k=50
    nearest_50_s = np.argsort(dists_simple)[:50]
    nearest_sigs_50 = temp_df_w.iloc[nearest_50_s]['signature_id'].values
    avg_xc_50 = np.mean([train_trajectories[sid][0] for sid in nearest_sigs_50], axis=0)
    avg_yc_50 = np.mean([train_trajectories[sid][1] for sid in nearest_sigs_50], axis=0)
    px_tmpl_50 = np.interp(t_new, t_orig, avg_xc_50)
    py_tmpl_50 = np.interp(t_new, t_orig, avg_yc_50)

    # Template k=100
    nearest_100_s = np.argsort(dists_simple)[:100]
    nearest_sigs_100 = temp_df_w.iloc[nearest_100_s]['signature_id'].values
    avg_xc_100 = np.mean([train_trajectories[sid][0] for sid in nearest_sigs_100], axis=0)
    avg_yc_100 = np.mean([train_trajectories[sid][1] for sid in nearest_sigs_100], axis=0)
    px_tmpl_100 = np.interp(t_new, t_orig, avg_xc_100)
    py_tmpl_100 = np.interp(t_new, t_orig, avg_yc_100)

    for cdf_w, tmpl_w in [(0.4, 0.6), (0.5, 0.5), (0.3, 0.7)]:
        name = f'ens_cdf{cdf_w}_tmpl50_{tmpl_w}'
        px_e = cdf_w * px_cdf_a06 + tmpl_w * px_tmpl_50
        py_e = cdf_w * py_cdf_a06 + tmpl_w * py_tmpl_50
        results[name]['ex'].append(px_e - tx)
        results[name]['ey'].append(py_e - ty)

    for cdf_w, tmpl_w in [(0.4, 0.6), (0.5, 0.5)]:
        name = f'ens_cdf{cdf_w}_tmpl100_{tmpl_w}'
        px_e = cdf_w * px_cdf_a06 + tmpl_w * px_tmpl_100
        py_e = cdf_w * py_cdf_a06 + tmpl_w * py_tmpl_100
        results[name]['ex'].append(px_e - tx)
        results[name]['ey'].append(py_e - ty)

    # Global mean trajectory
    px_gm = np.interp(t_new, t_orig, avg_x_all)
    py_gm = np.interp(t_new, t_orig, avg_y_all)
    results['global_mean_traj']['ex'].append(px_gm - tx)
    results['global_mean_traj']['ey'].append(py_gm - ty)

logging.info(f"Done in {time.time()-t0:.1f}s")

# Report
print("\n" + "=" * 60)
print(f"WRITER-HELD-OUT RESULTS ({len(sig_ids)} signatures)")
print("=" * 60)

scored = []
for m in methods:
    ex = np.concatenate(results[m]['ex'])
    ey = np.concatenate(results[m]['ey'])
    rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
    scored.append((m, rmse))
scored.sort(key=lambda x: x[1])

print(f"{'Method':<35} {'RMSE':>10}")
print("-" * 47)
for name, rmse in scored:
    marker = " <<<" if rmse < 0.249 else ""
    print(f"{name:<35} {rmse:>10.5f}{marker}")

best_name, best_rmse = scored[0]
print(f"\n>>> Best: {best_name} (RMSE: {best_rmse:.5f})")
print(f">>> LB best: 0.24449, LB avg: 0.25258")

# ============================================================
# Generate final submission with best config
# ============================================================
logging.info(f"\nGenerating final submission with {best_name}...")

test_preds = []
test_sig_ids = sorted(test['signature_id'].unique())

for sig_id in test_sig_ids:
    sig_test = test[test['signature_id'] == sig_id].sort_values('time')
    n = len(sig_test)
    rows, cols, shape, img = load_skeleton(sig_id)

    # CDF
    px_cdf, py_cdf = predict_cdf_with_direction(rows, cols, n, avg_x_all, avg_y_all)
    mx, my = np.mean(px_cdf), np.mean(py_cdf)
    px_cdf_b = 0.4 * px_cdf + 0.6 * mx
    py_cdf_b = 0.4 * py_cdf + 0.6 * my

    # Template (use all train data)
    feats_s = compute_rich_features(rows, cols, shape, img)
    feat_vals_s = np.array([(feats_s[c] - train_feat_df[simple_feature_cols].mean()[c]) / train_feat_df[simple_feature_cols].std().replace(0, 1)[c] for c in simple_feature_cols])
    dists_s = cdist(feat_vals_s.reshape(1, -1), train_simple_norm.values)[0]

    k = 50
    nearest_k = np.argsort(dists_s)[:k]
    nearest_sigs = train_feat_df.iloc[nearest_k]['signature_id'].values
    avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs], axis=0)
    avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs], axis=0)
    t_orig = np.linspace(0, 1, N_BINS)
    t_new = np.linspace(0, 1, n)
    px_tmpl = np.interp(t_new, t_orig, avg_xc)
    py_tmpl = np.interp(t_new, t_orig, avg_yc)

    # Apply best ensemble weights
    if 'ens' in best_name:
        parts = best_name.split('_')
        cdf_w = float(parts[1].replace('cdf', ''))
        tmpl_w = float(parts[3])
        if 'tmpl100' in best_name:
            nearest_100 = np.argsort(dists_s)[:100]
            nearest_sigs_100 = train_feat_df.iloc[nearest_100]['signature_id'].values
            avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs_100], axis=0)
            avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs_100], axis=0)
            px_tmpl = np.interp(t_new, t_orig, avg_xc)
            py_tmpl = np.interp(t_new, t_orig, avg_yc)
        px = cdf_w * px_cdf_b + tmpl_w * px_tmpl
        py = cdf_w * py_cdf_b + tmpl_w * py_tmpl
    elif 'tmpl' in best_name:
        px, py = px_tmpl, py_tmpl
    elif 'cdf' in best_name:
        if 'a0.6' in best_name:
            px, py = px_cdf_b, py_cdf_b
        else:
            px = 0.5 * px_cdf + 0.5 * mx
            py = 0.5 * py_cdf + 0.5 * my
    else:
        px = np.interp(t_new, t_orig, avg_x_all)
        py = np.interp(t_new, t_orig, avg_y_all)

    for idx, (_, row) in enumerate(sig_test.iterrows()):
        test_preds.append({
            'prediction_id': int(row['prediction_id']),
            'x': float(np.clip(px[idx], 0, 1)),
            'y': float(np.clip(py[idx], 0, 1)),
        })

sub = pd.DataFrame(test_preds).sort_values('prediction_id')
assert len(sub) == len(test)

ts = time.strftime('%Y%m%d_%H%M%S')
sub_path = f"{COMP_DIR}/submissions/submission_{ts}_final_ensemble.csv"
sub[['prediction_id', 'x', 'y']].to_csv(sub_path, index=False)
logging.info(f"Saved: {sub_path}")
print(f"\nSubmission: {sub.shape}")
print(f"X: [{sub['x'].min():.4f}, {sub['x'].max():.4f}]")
print(f"Y: [{sub['y'].min():.4f}, {sub['y'].max():.4f}]")
print(sub.head())

# Save experiment
exp = {
    'id': 'v5_ensemble_final',
    'timestamp': ts,
    'method': best_name,
    'cv_rmse': float(best_rmse),
    'evaluation': 'writer-held-out',
    'submission_file': sub_path,
}
with open(f"{COMP_DIR}/experiments.json", 'r') as f:
    experiments = json.load(f)
experiments.append(exp)
with open(f"{COMP_DIR}/experiments.json", 'w') as f:
    json.dump(experiments, f, indent=2)
