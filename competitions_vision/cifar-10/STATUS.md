# CIFAR-10 Competition Status

**Competition**: [CIFAR-10 - Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
**Type**: Image classification (10 classes)
**Metric**: Accuracy (maximize)
**Status**: Complete

## Results

| Version | Model | Val Acc | Public LB | Private LB | Time |
|---------|-------|---------|-----------|------------|------|
| v1 | ResNet-18 (single, 100 epochs) | 0.9602 | 0.95560 | 0.95560 | 34 min |

## Dataset
- **Train**: 50,000 images (32x32 RGB), perfectly balanced (5,000/class)
- **Test**: 300,000 images (32x32 RGB)
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## v1: ResNet-18 (Single Model)
- ResNet-18 adapted for 32x32 (3x3 initial conv, no max pool)
- 11.17M parameters
- Augmentation: RandomCrop(padding=4), HFlip, ColorJitter, Cutout(16)
- SGD (lr=0.1, momentum=0.9, weight_decay=5e-4) + CosineAnnealing
- 90/10 stratified split, 100 epochs, batch size 128
- Per-class accuracy: dog (91.4%) and cat (92.8%) hardest; automobile/frog/truck (98.2%) easiest
