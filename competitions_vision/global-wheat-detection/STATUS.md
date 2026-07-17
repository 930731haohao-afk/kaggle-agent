# Global Wheat Detection

**Date**: 2026-02-19

## Competition Info
- URL: https://www.kaggle.com/competitions/global-wheat-detection
- Problem: Object detection (detect wheat heads in field images)
- Metric: mAP @ IoU 0.5
- **Code competition** — requires Kaggle notebook submission
- Train: 3,373 images (1024x1024 RGB), 147,793 bounding boxes
- Test: 10 images (placeholder)
- 7 data sources from different institutions

## Key EDA Findings
- All images 1024x1024 RGB JPG
- Mean 43.8 boxes per image (range 1-116)
- Box sizes: median area ~5,488 px (~74x74), mean 6,843 px
- 7 sources with significant variation:
  - ethz_1: highest density (68.9 boxes/img), smallest boxes (area 4,803)
  - inrae_1: lowest density (21.0 boxes/img), largest boxes (area 15,828)
- 21.4% of boxes touch image boundary (partial wheat heads)
- Low overlap: 3.2% of box pairs overlap, mean IoU 0.10
- 49 unannotated images in train directory
- Spatial distribution: nearly uniform across image
- K-means anchors (k=5): 58x57, 77x100, 102x63, 111x175, 165x95

## Experiment Results

| # | Model | Local mAP@0.5 | Notes |
|---|-------|---------------|-------|
| 1 | Faster R-CNN (ResNet50-FPN) | 0.7965 | 5 epochs, COCO pretrained, GroupKFold fold 0 |

Training progression: 0.7965 → 0.7802 → 0.7930 → 0.7958 → 0.7959
Best at epoch 1 — slight overfitting in later epochs.

## Submissions
- No Kaggle submission yet (code competition — requires notebook)
- Local prediction file: `frcnn_baseline_20260219_163355.csv`

## Model Details
- **Baseline**: Faster R-CNN with ResNet-50 FPN backbone (COCO pretrained)
- SGD optimizer: lr=0.005, momentum=0.9, weight_decay=0.0005
- StepLR scheduler: step_size=3, gamma=0.1
- GroupKFold by source (fold 0: val=arvalis_1, 1055 images)
- Score threshold=0.3 for inference
- 12-38 detections per test image

## Lessons Learned
- COCO-pretrained Faster R-CNN transfers well to wheat detection (0.80 mAP in 1 epoch)
- Best performance at epoch 1 — model may overfit with more training on this small dataset
- Cross-source validation (GroupKFold by source) tests generalization to new domains
- arvalis_1 as holdout is a challenging test (different box size distribution)
- Code competition format prevents direct CSV submission

## Potential Improvements
- Data augmentation: horizontal/vertical flip, random crop, mosaic, mixup
- Test-time augmentation (TTA): multi-scale, flip
- Better backbone: ResNet-101, ResNeXt-101, EfficientNet
- EfficientDet or YOLOv5 as alternative architectures
- Train all 5 folds and ensemble
- Longer training with cosine annealing LR
- Custom anchors from k-means analysis
- Weighted Box Fusion (WBF) for ensemble post-processing
- Pseudo-labeling on test images
- Build Kaggle notebook for actual submission
