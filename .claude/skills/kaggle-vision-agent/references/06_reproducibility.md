# Stage V* — Reproducibility (applies to every stage)

The vision agent meets the SAME two-layer standard as tabular (`docs/reproducibility.md`):

## Layer 1 (plan standard) — every vision competition

- **Fixed seeds everywhere** (seed=42 convention): numpy, python `random`, torch, fold splits.
- **Fixed folds**: one `StratifiedKFold/GroupKFold(shuffle=True, random_state=42)` per comp,
  shared by every experiment arm; the CV scheme is recorded in config.yaml and in every log entry.
- **Traceable numbers**: every reported score exists in `experiments.json`
  (via `log_experiment_v2`); OOF/test predictions cached as npz (the verify gate's inputs).
- **uv-locked environment**: torch/timm versions pinned via `pyproject.toml`/`uv.lock`;
  GPU scripts run with `.venv/bin/python`.
- **MLflow mirror**: vision runs export to a SEPARATE store (`mlflow_vision.db`) — never into the
  tabular `mlflow.db` (see the data-separation rules in SKILL.md).

## Layer 2 (bit-level determinism gate) — for certifying submissions

The torch analogue of the tabular LightGBM pin (`deterministic/force_row_wise/num_threads`).
Fixed seeds are NOT enough on GPU: cuDNN autotunes kernels per run (timing-dependent, exactly like
LGBM's auto row/col-wise) and several CUDA kernels use non-deterministic atomics.

**The recipe is bundled — use it, don't re-derive it:**

```python
import sys; sys.path.insert(0, ".claude/skills/kaggle-vision-agent/assets")
from torch_determinism import setup_determinism, seeded_loader_kwargs
setup_determinism(seed=42)          # FIRST, before any model/dataloader is built
...
dl = DataLoader(ds, batch_size=64, shuffle=True, **seeded_loader_kwargs(seed=42))
```

What it pins: `CUBLAS_WORKSPACE_CONFIG`, `PYTHONHASHSEED`, python/numpy/torch seeds,
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`,
seeded DataLoader generator + worker_init (num_workers=0 default).

**Measured evidence (GB10, 2026-07-17)**: with this preamble, a resnet18 fine-tune (1 epoch,
1000 imgs) re-run in two separate processes produced **bit-identical validation logits**
(max|diff| = 0.0), and frozen-embedding extraction was likewise bit-identical.
Gate script: `vision/derisk_determinism_gate.py` (GATE: PASS, `vision/derisk_determinism.log`).

**Gate protocol for a submission**: before submitting, retrain the winning config from scratch
(same machine) and byte-compare the regenerated OOF npz against the validated one. Bit-identical →
certified. Not identical → find the nondeterminism source before trusting the submission.

**Honest boundaries** (state them in reports, same as tabular):
- Guarantee scope = same machine / driver / CUDA / torch build, same batch size & worker count.
  Cross-machine bit-equality is not promised (floating-point reduction order differs).
- If an op has no deterministic implementation, `setup_determinism` raises. Fall back to
  `warn_only=True` ONLY with the violation recorded in experiments.json notes — a documented
  exception, never a silent skip.
- Determinism costs some speed (no cudnn autotune). Acceptable for certification runs; discovery
  tiers may run without the full pin, but any number that reaches a report/submission must come
  from a gated run.
