"""Layer-2 determinism preamble for vision training (the torch analogue of the tabular
LightGBM `deterministic/force_row_wise/num_threads` pin — see docs/reproducibility.md).

Fixed seeds alone do NOT give bit-level reproducibility on GPU: cuDNN autotunes kernel
choice per run (timing-dependent, like LGBM's auto row/col-wise), and several CUDA kernels
use non-deterministic atomics. This module pins all of it.

Usage — call ONCE at the top of every training / inference script, BEFORE building any
model/dataloader:

    from torch_determinism import setup_determinism, seeded_loader_kwargs
    setup_determinism(seed=42)
    dl = DataLoader(ds, batch_size=64, shuffle=True, **seeded_loader_kwargs(seed=42))

Scope of the guarantee (honest boundary, same as tabular): bit-identical re-runs on the
SAME machine / driver / CUDA / torch build, same batch size and worker count. Cross-machine
or cross-version bit-equality is NOT promised (floating-point reduction orders differ).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

def setup_determinism(seed: int = 42, *, warn_only: bool = False) -> None:
    """Pin every torch/CUDA nondeterminism source.

    warn_only=False (default) makes torch RAISE on any op that has no deterministic
    implementation — surfacing the violation instead of silently breaking the gate.
    If a needed op genuinely has no deterministic kernel, rerun with warn_only=True and
    RECORD the warning + the op name in experiments.json notes (the gate is then
    documented-as-unattainable for that config, not silently skipped).
    """
    # CUBLAS needs this BEFORE any cuda context work for deterministic GEMM reductions
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                      # seeds CPU and all CUDA devices

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True    # fixed conv algo choice
    torch.backends.cudnn.benchmark = False       # no timing-dependent autotune


def seeded_loader_kwargs(seed: int = 42, num_workers: int = 0) -> dict:
    """DataLoader kwargs that pin shuffle order and worker RNG.

    num_workers=0 by default: single-process loading removes worker scheduling as a
    nondeterminism source. If throughput requires workers, keep the count FIXED and
    recorded — changing it changes batch composition timing-independently but keep it
    pinned anyway for auditability.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    return dict(generator=g, worker_init_fn=_worker_init, num_workers=num_workers)
