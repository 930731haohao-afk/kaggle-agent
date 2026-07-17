# Dogs vs Cats Competition Status

**Competition**: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
**Type**: Binary image classification (cat vs dog)
**Metric**: Log Loss (minimize)
**Status**: Complete (local) — competition closed, cannot submit via API

## Results

| Version | Model | Val Acc | Val LogLoss | Training Time |
|---------|-------|---------|-------------|---------------|
| v1 | ResNet-50 (ImageNet pretrained) | 99.08% | 0.03067 | 22 min |

## Dataset
- **Train**: 25,000 images (12,500 cats + 12,500 dogs), variable resolution
- **Test**: 12,500 images
- **Submission**: id, probability of dog

## v1: ResNet-50 Transfer Learning
- ResNet-50 with ImageNet V2 weights, replaced FC for binary output
- 23.5M parameters
- Input: resize to 256, center crop 224x224
- Augmentation: RandomResizedCrop(224), HFlip, ColorJitter, ImageNet normalize
- AdamW (lr=1e-4, weight_decay=1e-4) + CosineAnnealing
- 90/10 stratified split, 10 epochs, batch size 64
- Best epoch: 4 (val logloss 0.031, val acc 99.08%)
- Prediction clipped to [0.005, 0.995] to avoid extreme log loss

## Notes
- Transfer learning extremely effective: 98.8% accuracy from epoch 1
- Competition is closed — 400/403 errors on submission API
- The "redux" edition (dogs-vs-cats-redux-kernels-edition) also requires web rule acceptance
