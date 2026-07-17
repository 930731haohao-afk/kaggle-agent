# Stage V1 — Setup & data

## Environment check (do this first, it's cheap)

```bash
cd ~/ai_agents/kaggle
.venv/bin/python -c "import torch, timm; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

Expected on this machine: CUDA `True`, device `NVIDIA GB10` (arm64 DGX Spark). If torch/timm are
missing, install into the project venv with the CUDA aarch64 index:

```bash
uv pip install --python .venv/bin/python torch torchvision timm \
  --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple
```

GPU scripts run with `.venv/bin/python` (not bare `python3`). Long trainings ALWAYS run in the
background with a log file, and anything that can hang (native fit/forward) needs an OS-level
subprocess timeout — `signal.alarm` cannot interrupt native code.

## Competition workspace

Same layout as the tabular agent: `competitions/<name>/{config.yaml, data/, scripts/, submissions/,
experiments.json}`. Create `config.yaml` with: metric, direction, target format, image format
(csv-pixels | folder), image size, #classes, and the chosen CV scheme.

## Data loading — two common formats

1. **CSV pixels** (digit-recognizer style): each row = flattened grayscale pixels + label.
   Load with pandas, reshape to (N, H, W). Convert to model input via resize + 3-channel repeat +
   ImageNet normalization (see `assets/embed_cache.py` — it handles this).
2. **Image folders / filename-label CSV** (dogs-vs-cats, cifar-10 style): index the files, decode
   with PIL/torchvision at load time. For tier-0 embedding extraction, stream in batches — never
   load the full image set into RAM.

Always inspect before modeling (the EDA habit): class balance, image size distribution, corrupt/
truncated files, duplicate images (hash a sample), and whether test images differ systematically
from train (size, brightness, source).

## CV design for images

- Default: **StratifiedKFold on the label** (5-fold; 3-fold while iterating cheaply).
- **Grouped data → GroupKFold**: multiple images of the same patient/scene/object MUST NOT be split
  across folds (leakage). Look for grouping keys in filenames/metadata before choosing the scheme.
- Duplicates across train/test are a leakage signal worth an explicit note in config.yaml.
- Same rule as tabular: never compare scores across CV schemes; the scheme is part of every
  experiment log entry.

## Subsampling for the discovery stages

Tier 1-2 experiments run on a stratified subsample (e.g. 8-10k images or ~20% of train, whichever
is smaller) with a fixed seed, so every discovery experiment is fast AND comparable. Record the
subsample seed/size in config.yaml; the SAME subsample and folds serve every discovery arm.
