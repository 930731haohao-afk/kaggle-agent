"""
Dogs vs Cats — ResNet-50 Transfer Learning
- ImageNet pretrained ResNet-50, fine-tuned
- Single model, 90/10 stratified split
- 10 epochs with AdamW + cosine annealing
- Submission: probability of dog (label=1)
"""

import os
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================
COMP_DIR = Path("/home/tjyen/ai_agents/kaggle/competitions/dogs-vs-cats")
DATA_DIR = COMP_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test1"

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Dataset
# ============================================================
class DogsVsCatsDataset(Dataset):
    def __init__(self, file_paths: list[str], labels: np.ndarray | None,
                 transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img


# ============================================================
# Training & evaluation
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += preds.eq(labels.long()).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()
        correct += preds.eq(labels.long()).sum().item()
        total += labels.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    logloss = log_loss(all_labels, all_probs)
    return total_loss / total, acc, logloss, np.array(all_probs)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs = []
    for images in loader:
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device)
        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs)
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


# ============================================================
# Main
# ============================================================
def main():
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build file list and labels from train directory
    train_files = sorted(os.listdir(TRAIN_DIR))
    file_paths = [str(TRAIN_DIR / f) for f in train_files]
    labels = np.array([1 if f.startswith("dog.") else 0 for f in train_files])
    logger.info(f"Total train: {len(file_paths)} (cats: {(labels==0).sum()}, dogs: {(labels==1).sum()})")

    # 90/10 stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.1, stratify=labels, random_state=42)
    logger.info(f"Train split: {len(train_paths)}, Val split: {len(val_paths)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Dataloaders
    train_dataset = DogsVsCatsDataset(train_paths, train_labels, train_transform)
    val_dataset = DogsVsCatsDataset(val_paths, val_labels, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model: ResNet-50 pretrained, replace final FC for binary classification
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    best_val_logloss = float('inf')
    best_model_state = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_logloss, val_probs = evaluate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} / {train_acc:.4f} | "
            f"Val: {val_loss:.4f} / {val_acc:.4f} / LogLoss: {val_logloss:.5f} | "
            f"LR: {lr:.6f} | Best LL: {best_val_logloss:.5f} | {elapsed:.0f}s"
        )

    train_time = time.time() - start_time
    logger.info(f"\nTraining complete in {train_time:.0f}s")
    logger.info(f"Best Val LogLoss: {best_val_logloss:.5f}, Acc: {best_val_acc:.4f}")

    # Reload best model
    model.load_state_dict(best_model_state)
    model.to(DEVICE)

    # Predict test set
    logger.info("\nPredicting test set...")
    sub_df = pd.read_csv(DATA_DIR / "sampleSubmission.csv")
    test_ids = sub_df["id"].values
    test_paths = [str(TEST_DIR / f"{tid}.jpg") for tid in test_ids]

    test_dataset = DogsVsCatsDataset(test_paths, None, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_probs = predict(model, test_loader, DEVICE)

    # Clip probabilities to avoid extreme log loss
    test_probs = np.clip(test_probs, 0.005, 0.995)

    # Save submission
    submission = pd.DataFrame({"id": test_ids, "label": test_probs})
    sub_path = COMP_DIR / "submissions" / "submission_resnet50.csv"
    submission.to_csv(sub_path, index=False)
    logger.info(f"Submission saved: {sub_path}")
    logger.info(f"Shape: {submission.shape}")
    logger.info(f"Prob stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")
    logger.info(f"Predicted dogs: {(test_probs > 0.5).sum()}, cats: {(test_probs <= 0.5).sum()}")

    # Save model
    model_path = COMP_DIR / "scripts" / "resnet50_best.pt"
    torch.save(best_model_state, model_path)

    # Log experiment
    exp_path = COMP_DIR / "experiments.json"
    with open(exp_path) as f:
        experiments = json.load(f)
    experiments["experiments"].append({
        "id": len(experiments["experiments"]) + 1,
        "name": "ResNet-50 Transfer Learning",
        "model": "ResNet-50 (ImageNet V2 pretrained)",
        "augmentation": "RandomResizedCrop(224) + HFlip + ColorJitter + Normalize",
        "params": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW + CosineAnnealing",
            "image_size": IMAGE_SIZE, "train_split": 0.9,
        },
        "val_log_loss": float(best_val_logloss),
        "val_accuracy": float(best_val_acc),
        "training_time_seconds": train_time,
        "submission_file": str(sub_path),
    })
    with open(exp_path, "w") as f:
        json.dump(experiments, f, indent=2)
    logger.info("Experiment logged.")


if __name__ == "__main__":
    main()
