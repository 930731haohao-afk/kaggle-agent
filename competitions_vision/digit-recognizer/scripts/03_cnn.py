"""
Track B: CNN model for digit-recognizer.
PyTorch CNN with data augmentation, stratified 5-fold CV.
Target: 99%+ accuracy.
"""
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ============================================================
# Config
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

N_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# Load Data
# ============================================================
data_dir = "competitions/digit-recognizer/data"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

pixel_cols = [c for c in train_df.columns if c.startswith('pixel')]
X = train_df[pixel_cols].values.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
y = train_df['label'].values
X_test = test_df[pixel_cols].values.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0

print(f"Train: {X.shape}, Test: {X_test.shape}")

# ============================================================
# CNN Architecture
# ============================================================
class DigitCNN(nn.Module):
    """
    CNN for MNIST-like digit classification.
    Architecture: 2 conv blocks + dropout + 2 FC layers.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 28x28 -> 14x14
            nn.Dropout2d(0.25),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 14x14 -> 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
# Data Augmentation (applied on CPU tensors)
# ============================================================
def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply random affine-like augmentation to a batch of images."""
    n = images.size(0)
    augmented = images.clone()

    for i in range(n):
        img = augmented[i]  # (1, 28, 28)

        # Random rotation (-15 to +15 degrees)
        if np.random.rand() < 0.5:
            import torchvision.transforms.functional as TF
            angle = np.random.uniform(-15, 15)
            img = TF.rotate(img, angle)

        # Random shift (-2 to +2 pixels)
        if np.random.rand() < 0.5:
            shift_h = np.random.randint(-2, 3)
            shift_w = np.random.randint(-2, 3)
            img = torch.roll(img, shifts=(shift_h, shift_w), dims=(1, 2))

        augmented[i] = img

    return augmented

# ============================================================
# Training Loop
# ============================================================
def train_one_fold(X_train, y_train, X_val, y_val, fold_num):
    """Train CNN for one fold and return validation accuracy."""
    model = DigitCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    X_tr_t = torch.tensor(X_train)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        # Shuffle training data
        perm = torch.randperm(len(X_tr_t))
        X_tr_t = X_tr_t[perm]
        y_tr_t = y_tr_t[perm]

        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(X_tr_t), BATCH_SIZE):
            batch_x = X_tr_t[i:i+BATCH_SIZE]
            batch_y = y_tr_t[i:i+BATCH_SIZE]

            # Apply augmentation
            batch_x = augment_batch(batch_x)
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, len(X_val_t), BATCH_SIZE):
                batch_x = X_val_t[i:i+BATCH_SIZE]
                output = model(batch_x)
                val_preds.append(output.argmax(dim=1).cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_acc = accuracy_score(y_val, val_preds)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{EPOCHS}: loss={epoch_loss/n_batches:.4f}, "
                  f"val_acc={val_acc:.5f}, best={best_val_acc:.5f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if patience_counter >= 7:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model state
    model.load_state_dict(best_state)
    return model, best_val_acc

# ============================================================
# Cross-Validation
# ============================================================
print("\n" + "=" * 70)
print("CNN 5-FOLD CROSS-VALIDATION")
print("=" * 70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_scores = []
fold_models = []
t0 = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model, val_acc = train_one_fold(X_tr, y_tr, X_val, y_val, fold+1)
    cv_scores.append(val_acc)
    fold_models.append(model)
    print(f"  Fold {fold+1} best accuracy: {val_acc:.5f}")

total_time = time.time() - t0

print("\n" + "=" * 70)
print("CNN RESULTS")
print("=" * 70)
print(f"CV Scores: {[f'{s:.5f}' for s in cv_scores]}")
print(f"CV Mean:   {np.mean(cv_scores):.5f} +/- {np.std(cv_scores):.5f}")
print(f"Total time: {total_time:.1f}s")

# ============================================================
# Generate Test Predictions (ensemble of fold models)
# ============================================================
print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS (ensemble of 5 fold models)")
print("=" * 70)

X_test_t = torch.tensor(X_test).to(DEVICE)
all_preds = np.zeros((len(X_test), 10))

for i, model in enumerate(fold_models):
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        fold_preds = []
        for j in range(0, len(X_test_t), BATCH_SIZE):
            batch_x = X_test_t[j:j+BATCH_SIZE]
            output = model(batch_x)
            fold_preds.append(torch.softmax(output, dim=1).cpu().numpy())
        fold_preds = np.concatenate(fold_preds)
        all_preds += fold_preds
    print(f"  Model {i+1} predictions done")

# Average predictions
all_preds /= len(fold_models)
test_labels = all_preds.argmax(axis=1)

# Save submission
sub_dir = "competitions/digit-recognizer/submissions"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_file = os.path.join(sub_dir, f"cnn_submission_{timestamp}.csv")
submission = pd.DataFrame({
    'ImageId': range(1, len(test_labels) + 1),
    'Label': test_labels
})
submission.to_csv(sub_file, index=False)
print(f"\nSubmission saved to {sub_file}")
print(f"Shape: {submission.shape}")
print(f"Label distribution:\n{submission['Label'].value_counts().sort_index()}")

# ============================================================
# Log Experiment
# ============================================================
exp_file = "competitions/digit-recognizer/experiments.json"
with open(exp_file, 'r') as f:
    experiments = json.load(f)

experiments.append({
    "experiment_id": len(experiments) + 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "modeling",
    "model": "CNN-2block",
    "features": "raw_pixels_normalized",
    "params": {
        "conv_blocks": 2,
        "channels": [32, 64],
        "fc_units": 256,
        "dropout": [0.25, 0.5],
        "batch_norm": True,
        "augmentation": "rotation_15deg_shift_2px",
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "scheduler": "ReduceLROnPlateau"
    },
    "cv_strategy": f"{N_FOLDS}-fold-stratified",
    "cv_scores": [round(s, 5) for s in cv_scores],
    "cv_mean": round(float(np.mean(cv_scores)), 5),
    "cv_std": round(float(np.std(cv_scores)), 5),
    "training_time_sec": round(total_time, 1),
    "submission_file": sub_file,
    "notes": "2-block CNN with BN, dropout, augmentation (rotation+shift), 5-fold ensemble"
})

with open(exp_file, 'w') as f:
    json.dump(experiments, f, indent=2)
print(f"Experiment logged to {exp_file}")
