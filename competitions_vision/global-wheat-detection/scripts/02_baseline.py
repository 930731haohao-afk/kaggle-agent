"""Baseline for global-wheat-detection.

Faster R-CNN with ResNet-50 FPN backbone, pretrained on COCO.
GroupKFold by source for cross-validation.
Trains fold 0 only for quick baseline, evaluates with mAP@0.5.
"""
import pandas as pd
import numpy as np
import os
import json
import time
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
from sklearn.model_selection import GroupKFold

data_dir = "competitions/global-wheat-detection/data"
script_dir = "competitions/global-wheat-detection/scripts"
sub_dir = "competitions/global-wheat-detection/submissions"
os.makedirs(sub_dir, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================
# 1. LOAD AND PARSE DATA
# ============================================================
print("\n1. LOADING DATA...")
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
train_df['bbox_parsed'] = train_df['bbox'].apply(lambda x: json.loads(x))
train_df['x'] = train_df['bbox_parsed'].apply(lambda b: b[0])
train_df['y'] = train_df['bbox_parsed'].apply(lambda b: b[1])
train_df['w'] = train_df['bbox_parsed'].apply(lambda b: b[2])
train_df['h'] = train_df['bbox_parsed'].apply(lambda b: b[3])

# Convert to [x1, y1, x2, y2] format for torchvision
train_df['x1'] = train_df['x']
train_df['y1'] = train_df['y']
train_df['x2'] = train_df['x'] + train_df['w']
train_df['y2'] = train_df['y'] + train_df['h']

# Filter out degenerate boxes (zero area)
before = len(train_df)
train_df = train_df[(train_df['w'] > 0) & (train_df['h'] > 0)].copy()
print(f"Filtered {before - len(train_df)} degenerate boxes, {len(train_df)} remain")

# Group boxes by image
image_ids = train_df['image_id'].unique()
print(f"Training images: {len(image_ids)}")

# Create image-level dataframe with source info
img_df = train_df.groupby('image_id').agg(
    source=('source', 'first'),
    n_boxes=('x', 'count')
).reset_index()

print(f"Images per source: {img_df['source'].value_counts().to_dict()}")

# ============================================================
# 2. DATASET
# ============================================================
class WheatDataset(Dataset):
    def __init__(self, image_ids, df, img_dir, transforms=None):
        self.image_ids = image_ids
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)  # [C, H, W] in [0, 1]

        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x1', 'y1', 'x2', 'y2']].values.astype(np.float32)

        # Clamp boxes to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, 1023)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, 1023)
        boxes[:, 2] = np.clip(boxes[:, 2], 1, 1024)
        boxes[:, 3] = np.clip(boxes[:, 3], 1, 1024)

        # Ensure x2 > x1 and y2 > y1
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.ones(len(boxes), dtype=torch.int64),  # single class
            'image_id': torch.tensor([idx]),
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================================
# 3. MODEL
# ============================================================
def get_model(num_classes=2):  # background + wheat
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ============================================================
# 4. EVALUATION — mAP@0.5
# ============================================================
def compute_iou_matrix(boxes1, boxes2):
    """Compute IoU matrix between two sets of boxes [N, 4] and [M, 4]."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def compute_ap_at_iou(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Compute AP at a single IoU threshold for one image."""
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # Sort by confidence
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = np.zeros(len(pred_boxes))

    for i in range(len(pred_boxes)):
        best_iou = 0
        best_j = -1
        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_iou >= iou_threshold:
            tp[i] = 1
            matched_gt.add(best_j)

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / len(gt_boxes)

    # AP using all-points interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_model(model, dataloader, df, device, score_threshold=0.3):
    """Evaluate model on validation set, return mAP@0.5."""
    model.eval()
    all_aps = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                gt_boxes = targets[i]['boxes'].numpy()

                # Filter by score threshold
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                ap = compute_ap_at_iou(pred_boxes, pred_scores, gt_boxes, 0.5)
                all_aps.append(ap)

    return np.mean(all_aps) if all_aps else 0.0

# ============================================================
# 5. TRAINING — GroupKFold by source, train fold 0 only
# ============================================================
print("\n3. SETTING UP CROSS-VALIDATION...")

gkf = GroupKFold(n_splits=5)
groups = img_df['source'].values

fold = 0
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(img_df, groups=groups)):
    if fold_idx != fold:
        continue

    train_ids = img_df.iloc[train_idx]['image_id'].values
    val_ids = img_df.iloc[val_idx]['image_id'].values

    train_sources = img_df.iloc[train_idx]['source'].unique()
    val_sources = img_df.iloc[val_idx]['source'].unique()
    print(f"\nFold {fold}:")
    print(f"  Train: {len(train_ids)} images from {list(train_sources)}")
    print(f"  Val:   {len(val_ids)} images from {list(val_sources)}")

    train_dataset = WheatDataset(train_ids, train_df, os.path.join(data_dir, "train"))
    val_dataset = WheatDataset(val_ids, train_df, os.path.join(data_dir, "train"))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    # Model
    model = get_model(num_classes=2)
    model.to(DEVICE)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training
    n_epochs = 5
    best_map = 0.0
    print(f"\nTraining for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)}: "
                      f"loss={epoch_loss/n_batches:.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0

        # Evaluate
        val_map = evaluate_model(model, val_loader, train_df, DEVICE, score_threshold=0.3)

        print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, "
              f"val_mAP@0.5={val_map:.4f}, time={elapsed:.0f}s")

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(),
                       os.path.join(script_dir, "best_model_fold0.pth"))
            print(f"  -> New best mAP@0.5: {best_map:.4f}")

    print(f"\nBest validation mAP@0.5 (fold {fold}): {best_map:.4f}")

# ============================================================
# 6. GENERATE TEST PREDICTIONS
# ============================================================
print("\n4. GENERATING TEST PREDICTIONS...")

# Load best model
model = get_model(num_classes=2)
model.load_state_dict(torch.load(os.path.join(script_dir, "best_model_fold0.pth"),
                                  weights_only=True))
model.to(DEVICE)
model.eval()

test_dir = os.path.join(data_dir, "test")
test_images = [f.replace('.jpg', '') for f in os.listdir(test_dir) if f.endswith('.jpg')]
print(f"Test images: {len(test_images)}")

results = []
score_threshold = 0.3

with torch.no_grad():
    for img_id in test_images:
        img = Image.open(os.path.join(test_dir, f"{img_id}.jpg")).convert("RGB")
        img_tensor = F.to_tensor(img).unsqueeze(0).to(DEVICE)

        output = model(img_tensor)[0]
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()

        keep = scores >= score_threshold
        boxes = boxes[keep]
        scores = scores[keep]

        # Format: "confidence x y w h" (convert from x1y1x2y2 to xywh)
        pred_strings = []
        for box, score in zip(boxes, scores):
            x, y = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            pred_strings.append(f"{score:.4f} {x:.0f} {y:.0f} {w:.0f} {h:.0f}")

        pred_string = " ".join(pred_strings) if pred_strings else "0.1 0 0 50 50"
        results.append({'image_id': img_id, 'PredictionString': pred_string})
        print(f"  {img_id}: {len(boxes)} detections")

# Save submission
timestamp = time.strftime("%Y%m%d_%H%M%S")
sub_path = os.path.join(sub_dir, f"frcnn_baseline_{timestamp}.csv")
sub_df = pd.DataFrame(results)
sub_df.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")
print(f"Shape: {sub_df.shape}")

# ============================================================
# 7. SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("BASELINE RESULTS SUMMARY")
print(f"{'='*70}")
print(f"Model: Faster R-CNN (ResNet-50 FPN, COCO pretrained)")
print(f"Training: {n_epochs} epochs, lr=0.005, batch_size=4")
print(f"Validation: GroupKFold by source (fold 0)")
print(f"  Train sources: {list(train_sources)}")
print(f"  Val sources: {list(val_sources)}")
print(f"Best val mAP@0.5: {best_map:.4f}")
print(f"Score threshold: {score_threshold}")
print(f"Submission: {sub_path}")
