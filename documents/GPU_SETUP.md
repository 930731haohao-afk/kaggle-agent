# GPU Configuration for Kaggle Agent

**Last Updated**: 2026-02-12

## Hardware
- **GPU**: NVIDIA GB10
- **Driver Version**: 580.95.05
- **CUDA**: Available (nvidia-nccl-cu12 installed)

## ML Library Versions
- **LightGBM**: 4.6.0 (GPU support available)
- **XGBoost**: 3.1.3 (GPU support available)
- **CatBoost**: 1.2.8 (GPU support available)

## GPU Parameters for Training

### LightGBM
```python
lgb_params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    # Other params...
}
```

### XGBoost
```python
xgb_params = {
    'tree_method': 'hist',  # Use 'hist' for GPU on XGBoost 2.0+
    'device': 'cuda',        # or 'cuda:0' for specific GPU
    # Other params...
}
```

### CatBoost
```python
cat_params = {
    'task_type': 'GPU',
    'devices': '0',
    # Other params...
}
```

## Performance Notes
- GPU training is most beneficial for large datasets (>100K rows)
- For small datasets (<10K rows), CPU might be faster due to overhead
- LightGBM GPU tends to be fastest for tree-based models
- Always verify GPU usage with `nvidia-smi` during training

## Testing GPU
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check if model is using GPU during training
nvidia-smi pmon
```
