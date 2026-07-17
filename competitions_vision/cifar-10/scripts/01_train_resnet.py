"""
CIFAR-10 ResNet-18 Training Script
- Single model (90/10 stratified split)
- 100 epochs, SGD + cosine annealing
- Data augmentation: RandomCrop(4), HFlip, ColorJitter, Cutout(16)
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
from torchvision import transforms
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================
COMP_DIR = Path("/home/tjyen/ai_agents/kaggle/competitions/cifar-10")
DATA_DIR = COMP_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

EPOCHS = 100
BATCH_SIZE = 128
LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# Cutout augmentation
# ============================================================
class Cutout:
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).unsqueeze(0).expand_as(img)
        return img * mask


# ============================================================
# Dataset
# ============================================================
class CIFAR10Dataset(Dataset):
    def __init__(self, image_ids: np.ndarray, labels: np.ndarray | None,
                 image_dir: Path, transform=None):
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_dir / f"{img_id}.png"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img


# ============================================================
# ResNet-18 for CIFAR-10 (32x32 input)
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ============================================================
# Training & evaluation
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs = []
    for images in loader:
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())
    return torch.cat(all_probs)


# ============================================================
# Main
# ============================================================
def main():
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load labels
    labels_df = pd.read_csv(DATA_DIR / "trainLabels.csv")
    all_ids = labels_df["id"].values
    all_labels = labels_df["label"].map(CLASS_TO_IDX).values

    # 90/10 stratified split
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        all_ids, all_labels, test_size=0.1, stratify=all_labels, random_state=42)
    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(n_holes=1, length=16),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # Dataloaders
    train_dataset = CIFAR10Dataset(train_ids, train_labels, TRAIN_DIR, train_transform)
    val_dataset = CIFAR10Dataset(val_ids, val_labels, TRAIN_DIR, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = ResNetCIFAR(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    best_val_acc = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train: {train_loss:.4f} / {train_acc:.4f} | "
                f"Val: {val_loss:.4f} / {val_acc:.4f} | "
                f"LR: {lr:.6f} | Best: {best_val_acc:.4f} | {elapsed:.0f}s"
            )

    train_time = time.time() - start_time
    logger.info(f"\nTraining complete in {train_time:.0f}s")
    logger.info(f"Best Val Accuracy: {best_val_acc:.5f}")

    # Reload best model
    model.load_state_dict(best_model_state)
    model.to(DEVICE)

    # Per-class accuracy on validation set
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    logger.info("\nPer-class accuracy:")
    for i, cls in enumerate(CLASSES):
        mask = all_true == i
        cls_acc = (all_preds[mask] == all_true[mask]).mean()
        logger.info(f"  {cls:12s}: {cls_acc:.4f}")

    # Predict test set
    logger.info("\nPredicting test set (300K images)...")
    sub_df = pd.read_csv(DATA_DIR / "sampleSubmission.csv")
    test_ids = sub_df["id"].values

    test_dataset = CIFAR10Dataset(test_ids, None, TEST_DIR, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_probs = predict(model, test_loader, DEVICE)
    test_preds = test_probs.numpy().argmax(axis=1)
    test_labels = [CLASSES[p] for p in test_preds]

    # Save submission
    submission = pd.DataFrame({"id": test_ids, "label": test_labels})
    sub_path = COMP_DIR / "submissions" / "submission_resnet18.csv"
    submission.to_csv(sub_path, index=False)
    logger.info(f"Submission saved: {sub_path}")
    logger.info(f"Shape: {submission.shape}")
    logger.info(f"Label distribution:\n{submission['label'].value_counts().sort_index()}")

    # Save model
    model_path = COMP_DIR / "scripts" / "resnet18_best.pt"
    torch.save(best_model_state, model_path)

    # Log experiment
    exp_path = COMP_DIR / "experiments.json"
    with open(exp_path) as f:
        experiments = json.load(f)
    experiments["experiments"].append({
        "id": len(experiments["experiments"]) + 1,
        "name": "ResNet-18 CIFAR (single model)",
        "model": "ResNet-18 (CIFAR variant, 11M params)",
        "augmentation": "RandomCrop(4) + HFlip + ColorJitter + Cutout(16)",
        "params": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "optimizer": "SGD + CosineAnnealing",
            "train_split": 0.9,
        },
        "val_accuracy": float(best_val_acc),
        "training_time_seconds": train_time,
        "submission_file": str(sub_path),
    })
    with open(exp_path, "w") as f:
        json.dump(experiments, f, indent=2)
    logger.info("Experiment logged.")


if __name__ == "__main__":
    main()
