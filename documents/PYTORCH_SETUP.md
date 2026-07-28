# PyTorch ARM64 + CUDA 13.0 Setup

**Last Updated**: 2026-02-12
**System**: ARM64 (aarch64) with NVIDIA GB10 GPU + CUDA 13.0

## Successfully Installed

✅ **PyTorch 2.11.0 nightly with CUDA 13.0 support**
✅ **PyTorch Geometric 2.7.0**
✅ **Transformers 5.1.0 + Datasets + Accelerate**

## Installation Method

Since PyTorch stable releases don't have pre-built CUDA wheels for ARM64, we use **nightly builds** from PyTorch:

```bash
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu130
```

## Key Packages Installed

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.11.0.dev20260211+cu130 | Nightly with CUDA 13.0 |
| torchvision | 0.26.0.dev20260211+cu130 | Computer vision models |
| torchaudio | 2.11.0.dev20260211+cu130 | Audio processing |
| torch-geometric | 2.7.0 | Graph neural networks |
| transformers | 5.1.0 | HuggingFace NLP models |
| datasets | 5.2.1 | HuggingFace datasets |
| accelerate | 2.4.0 | Multi-GPU training |

## GPU Configuration

- **GPU**: NVIDIA GB10
- **CUDA**: 13.0
- **cuDNN**: 9.17.1
- **Memory**: ~120 GB

## Verification

Test GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Test tensor operations on GPU
x = torch.randn(1000, 1000).cuda()
y = x @ x.T
print(f"Result on: {y.device}")
```

## Important Notes

1. **Nightly builds update daily** - Pin specific versions in production
2. **Use `uv pip install`** - Not `uv add` (dependency resolution issues)
3. **Index configuration** - Add to `pyproject.toml`:
   ```toml
   [[tool.uv.index]]
   name = "pytorch-nightly-cu130"
   url = "https://download.pytorch.org/whl/nightly/cu130"
   explicit = true
   ```

## Troubleshooting

**Issue**: `uv run` reverts to CPU version
- **Solution**: Use `.venv/bin/python3` directly or ensure pyproject.toml doesn't pin torch versions

**Issue**: Missing CUDA libraries
- **Solution**: Reinstall with dependencies: `uv pip install --force-reinstall torch --index-url ...`

## Resources

- [PyTorch ARM64 CUDA issue #159779](https://github.com/pytorch/pytorch/issues/159779)
- [PyTorch Nightly cu130 wheels](https://download.pytorch.org/whl/nightly/cu130/torch/)
- [NVIDIA Grace Hopper support discussion](https://discuss.pytorch.org/t/conda-pytorch-cuda-12-for-arm64-and-grace-hopper/203285)
