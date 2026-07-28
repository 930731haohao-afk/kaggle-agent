"""
Check if template matching is leaking via same-writer signatures.
Evaluate with writer-level leave-out (no same-writer templates).
"""
import pandas as pd
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.spatial.distance import cdist
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

N_BINS = 100


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


# Build feature matrix and trajectories
logging.info("Building feature matrix...")

train_features = []
train_trajectories = {}
sig_to_writer = {}

for sig_id in sorted(train['signature_id'].unique()):
    rows, cols, shape = load_skeleton(sig_id)
    feats = compute_features(rows, cols, shape)
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

# ============================================================
# Check: how many templates come from the same writer?
# ============================================================
print("=" * 60)
print("LEAKAGE ANALYSIS: Same-writer templates")
print("=" * 60)

same_writer_count = 0
total_templates = 0

for sig_id in sorted(train['signature_id'].unique())[:100]:
    writer_id = sig_to_writer[sig_id]
    mask = train_feat_df['signature_id'] != sig_id
    temp_df = train_feat_df[mask].reset_index(drop=True)
    temp_norm = train_feat_norm[mask].reset_index(drop=True)

    feats = compute_features(*load_skeleton(sig_id))
    feat_vals = np.array([(feats[c] - feat_means[c]) / feat_stds[c] for c in feature_cols])
    dists = cdist(feat_vals.reshape(1, -1), temp_norm.values)[0]

    nearest_idx = np.argsort(dists)[:5]
    nearest_sigs = temp_df.iloc[nearest_idx]['signature_id'].values

    for sid in nearest_sigs:
        total_templates += 1
        if sig_to_writer[sid] == writer_id:
            same_writer_count += 1

print(f"Same-writer templates: {same_writer_count}/{total_templates} = {same_writer_count/total_templates*100:.1f}%")
print(f"Expected by chance: {5/605*100:.1f}% (each writer has ~5 sigs out of 605)")

# ============================================================
# Writer-held-out evaluation
# ============================================================
logging.info("Writer-held-out evaluation...")

results = {
    'template_k5_sig_loo': {'ex': [], 'ey': []},
    'template_k5_writer_loo': {'ex': [], 'ey': []},
    'template_k10_writer_loo': {'ex': [], 'ey': []},
    'template_k20_writer_loo': {'ex': [], 'ey': []},
    'template_k50_writer_loo': {'ex': [], 'ey': []},
    'global_mean': {'ex': [], 'ey': []},
}

# Precompute global mean trajectory
global_mean_x = np.mean([train_trajectories[sid][0] for sid in train_trajectories], axis=0)
global_mean_y = np.mean([train_trajectories[sid][1] for sid in train_trajectories], axis=0)

sig_ids = sorted(train['signature_id'].unique())
t0 = time.time()

for i, sig_id in enumerate(sig_ids):
    if (i + 1) % 200 == 0:
        logging.info(f"  {i+1}/{len(sig_ids)} ({time.time()-t0:.0f}s)")

    gt = train[train['signature_id'] == sig_id].sort_values('time')
    tx, ty = gt['x'].values, gt['y'].values
    n = len(gt)
    writer_id = sig_to_writer[sig_id]

    feats = compute_features(*load_skeleton(sig_id))
    feat_vals = np.array([(feats[c] - feat_means[c]) / feat_stds[c] for c in feature_cols])

    # Sig-level LOO (same as before)
    mask_sig = train_feat_df['signature_id'] != sig_id
    temp_norm_sig = train_feat_norm[mask_sig].reset_index(drop=True)
    temp_df_sig = train_feat_df[mask_sig].reset_index(drop=True)
    dists_sig = cdist(feat_vals.reshape(1, -1), temp_norm_sig.values)[0]

    nearest_5 = np.argsort(dists_sig)[:5]
    nearest_sigs = temp_df_sig.iloc[nearest_5]['signature_id'].values
    avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs], axis=0)
    avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs], axis=0)
    t_orig = np.linspace(0, 1, N_BINS)
    t_new = np.linspace(0, 1, n)
    px = np.interp(t_new, t_orig, avg_xc)
    py = np.interp(t_new, t_orig, avg_yc)
    results['template_k5_sig_loo']['ex'].append(px - tx)
    results['template_k5_sig_loo']['ey'].append(py - ty)

    # Writer-level LOO
    mask_writer = train_feat_df['writer_id'] != writer_id
    temp_norm_w = train_feat_norm[mask_writer].reset_index(drop=True)
    temp_df_w = train_feat_df[mask_writer].reset_index(drop=True)
    dists_w = cdist(feat_vals.reshape(1, -1), temp_norm_w.values)[0]

    for k in [5, 10, 20, 50]:
        nearest_k = np.argsort(dists_w)[:k]
        nearest_sigs_w = temp_df_w.iloc[nearest_k]['signature_id'].values
        avg_xc = np.mean([train_trajectories[sid][0] for sid in nearest_sigs_w], axis=0)
        avg_yc = np.mean([train_trajectories[sid][1] for sid in nearest_sigs_w], axis=0)
        px = np.interp(t_new, t_orig, avg_xc)
        py = np.interp(t_new, t_orig, avg_yc)
        results[f'template_k{k}_writer_loo']['ex'].append(px - tx)
        results[f'template_k{k}_writer_loo']['ey'].append(py - ty)

    # Global mean
    px_gm = np.interp(t_new, t_orig, global_mean_x)
    py_gm = np.interp(t_new, t_orig, global_mean_y)
    results['global_mean']['ex'].append(px_gm - tx)
    results['global_mean']['ey'].append(py_gm - ty)

logging.info(f"Done in {time.time()-t0:.1f}s")

# Report
print("\n" + "=" * 60)
print("WRITER-HELD-OUT EVALUATION")
print("=" * 60)
print(f"{'Method':<35} {'RMSE':>10}")
print("-" * 47)

scored = []
for m in results:
    ex = np.concatenate(results[m]['ex'])
    ey = np.concatenate(results[m]['ey'])
    rmse = np.sqrt((np.sum(ex**2) + np.sum(ey**2)) / (2 * len(ex)))
    scored.append((m, rmse))
scored.sort(key=lambda x: x[1])

for name, rmse in scored:
    print(f"{name:<35} {rmse:>10.5f}")

best_name, best_rmse = scored[0]
print(f"\n>>> Best writer-held-out: {best_name} (RMSE: {best_rmse:.5f})")
print(f">>> LB best: 0.24449, LB avg: 0.25258")
print(f"\nNote: Test set has ZERO writer overlap with train.")
print(f"Writer-held-out is the realistic evaluation for this competition.")
