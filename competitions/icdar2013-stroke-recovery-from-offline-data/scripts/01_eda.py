"""EDA: Visualize signature trajectories and their corresponding images."""
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_dir = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/data"
out_dir = "/home/tjyen/ai_agents/kaggle/competitions/icdar2013-stroke-recovery-from-offline-data/scripts"

train = pd.read_csv(f"{data_dir}/train.csv")

# === Plot trajectory vs image for a few train signatures ===
fig, axes = plt.subplots(4, 4, figsize=(20, 20))

sample_sigs = [1, 10, 50, 100]
for row, sig_id in enumerate(sample_sigs):
    sig_data = train[train['signature_id'] == sig_id]

    # Image
    img_path = f"{data_dir}/images_stroke/images/{sig_id:04d}.jpg"
    img = Image.open(img_path)

    # Col 0: Original image
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f"Sig {sig_id}: Image ({img.size[0]}x{img.size[1]})")
    axes[row, 0].axis('off')

    # Col 1: Trajectory (x,y) scatter colored by time
    scatter = axes[row, 1].scatter(sig_data['x'], sig_data['y'],
                                     c=sig_data['time'], cmap='viridis', s=2)
    axes[row, 1].set_xlim(-0.05, 1.05)
    axes[row, 1].set_ylim(-0.05, 1.05)
    axes[row, 1].invert_yaxis()  # Images have y=0 at top
    axes[row, 1].set_title(f"Trajectory (colored by time, {len(sig_data)} pts)")
    axes[row, 1].set_aspect('equal')
    plt.colorbar(scatter, ax=axes[row, 1], label='time')

    # Col 2: X over time
    axes[row, 2].plot(sig_data['time'], sig_data['x'], 'b-', linewidth=0.5)
    axes[row, 2].set_title(f"x(t)")
    axes[row, 2].set_xlabel('time')
    axes[row, 2].set_ylabel('x')

    # Col 3: Y over time
    axes[row, 3].plot(sig_data['time'], sig_data['y'], 'r-', linewidth=0.5)
    axes[row, 3].set_title(f"y(t)")
    axes[row, 3].set_xlabel('time')
    axes[row, 3].set_ylabel('y')

plt.tight_layout()
plt.savefig(f"{out_dir}/eda_trajectories.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved eda_trajectories.png")

# === Trajectory statistics ===
print("\n=== TRAJECTORY STATISTICS ===")
sig_stats = train.groupby('signature_id').agg(
    n_points=('time', 'count'),
    x_range=('x', lambda x: x.max() - x.min()),
    y_range=('y', lambda x: x.max() - x.min()),
    x_std=('x', 'std'),
    y_std=('y', 'std'),
).reset_index()

print(f"\nPoints per signature: {sig_stats['n_points'].describe()}")
print(f"\nX range (always 1.0 due to normalization): {sig_stats['x_range'].describe()}")
print(f"\nX std: {sig_stats['x_std'].describe()}")
print(f"\nY std: {sig_stats['y_std'].describe()}")

# === Velocity analysis ===
print("\n=== VELOCITY ANALYSIS ===")
velocities = []
for sig_id in train['signature_id'].unique()[:100]:
    sig_data = train[train['signature_id'] == sig_id].sort_values('time')
    dx = sig_data['x'].diff().dropna()
    dy = sig_data['y'].diff().dropna()
    speed = np.sqrt(dx**2 + dy**2)
    velocities.append({
        'signature_id': sig_id,
        'mean_speed': speed.mean(),
        'max_speed': speed.max(),
        'std_speed': speed.std(),
    })
vel_df = pd.DataFrame(velocities)
print(f"Mean speed per step: {vel_df['mean_speed'].describe()}")
print(f"Max speed per step: {vel_df['max_speed'].describe()}")

# === Check if y coordinate is inverted relative to image ===
print("\n=== COORDINATE SYSTEM CHECK ===")
sig_data = train[train['signature_id'] == 1].sort_values('time')
print(f"Sig 1: First point ({sig_data['x'].iloc[0]:.3f}, {sig_data['y'].iloc[0]:.3f})")
print(f"Sig 1: Last point ({sig_data['x'].iloc[-1]:.3f}, {sig_data['y'].iloc[-1]:.3f})")
print(f"Sig 1: x starts high (right side?), y starts low-middle")
print("Note: In image coordinates, y=0 is top. Check if trajectory y matches image y.")

# === Skeleton extraction test ===
print("\n=== SKELETON EXTRACTION TEST ===")
from skimage import morphology, io, color
from skimage.filters import threshold_otsu

img = io.imread(f"{data_dir}/images_stroke/images/0001.jpg")
gray = color.rgb2gray(img)
thresh = threshold_otsu(gray)
binary = gray < thresh  # ink is dark
skeleton = morphology.skeletonize(binary)
print(f"Image 0001: shape={img.shape}, skeleton pixels={skeleton.sum()}")
print(f"Sig 1 has {len(train[train['signature_id'] == 1])} trajectory points")
print(f"Ratio: {skeleton.sum() / len(train[train['signature_id'] == 1]):.1f}x more skeleton pixels")
