"""Shared utilities for conway-s-reverse-game-of-life."""
import importlib.util
import os

import numpy as np
import pandas as pd

COMP_DIR = "competitions/conway-s-reverse-game-of-life"
DATA = f"{COMP_DIR}/data"
CACHE = f"{COMP_DIR}/scripts/cache"
SEED = 42
N_SPLITS = 5

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)


def load_arrays():
    """Return dict with S (train start), P (train stop), delta, T (test stop), tdelta, test_ids."""
    os.makedirs(CACHE, exist_ok=True)
    npz = f"{CACHE}/arrays.npz"
    if os.path.exists(npz):
        return dict(np.load(npz))
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    start_cols = [c for c in train.columns if c.startswith("start")]
    stop_cols = [c for c in train.columns if c.startswith("stop")]
    out = dict(
        S=train[start_cols].to_numpy(np.int8).reshape(-1, 20, 20),
        P=train[stop_cols].to_numpy(np.int8).reshape(-1, 20, 20),
        delta=train["delta"].to_numpy(np.int64),
        T=test[stop_cols].to_numpy(np.int8).reshape(-1, 20, 20),
        tdelta=test["delta"].to_numpy(np.int64),
        test_ids=test["id"].to_numpy(np.int64),
    )
    np.savez_compressed(npz, **out)
    return out


def get_folds(delta: np.ndarray) -> np.ndarray:
    """Stratified-by-delta fold assignment, fixed seed."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(delta), np.int64)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(delta)), delta)):
        folds[va] = f
    return folds


def life_step(b: np.ndarray) -> np.ndarray:
    """One GoL step, dead boundary. b: (..., 20, 20) int8."""
    p = np.pad(b, [(0, 0)] * (b.ndim - 2) + [(1, 1), (1, 1)])
    sl = (slice(None),) * (b.ndim - 2)
    n = (p[sl + (slice(0, -2), slice(0, -2))] + p[sl + (slice(0, -2), slice(1, -1))] +
         p[sl + (slice(0, -2), slice(2, None))] + p[sl + (slice(1, -1), slice(0, -2))] +
         p[sl + (slice(1, -1), slice(2, None))] + p[sl + (slice(2, None), slice(0, -2))] +
         p[sl + (slice(2, None), slice(1, -1))] + p[sl + (slice(2, None), slice(2, None))])
    return ((n == 3) | ((b == 1) & (n == 2))).astype(np.int8)


def gen_synthetic(n: int, seed: int):
    """Generate synthetic (start, stop, delta) matching the competition process:
    random density U(0.01,0.99) board, 5 warmup steps -> start, delta more -> stop.
    Discard boards empty at start or stop. Returns (S, P, delta) arrays."""
    rng = np.random.RandomState(seed)
    outS, outP, outD = [], [], []
    got = 0
    while got < n:
        m = min(4096, int((n - got) * 1.5) + 256)
        p = rng.uniform(0.01, 0.99, size=(m, 1, 1))
        b = (rng.uniform(size=(m, 20, 20)) < p).astype(np.int8)
        for _ in range(5):
            b = life_step(b)
        start = b
        alive = start.sum(axis=(1, 2)) > 0
        d = rng.randint(1, 6, size=m)
        stop = start.copy()
        for step in range(5):
            stop = np.where((d > step)[:, None, None], life_step(stop), stop)
        alive &= stop.sum(axis=(1, 2)) > 0
        outS.append(start[alive]); outP.append(stop[alive]); outD.append(d[alive])
        got += int(alive.sum())
    S = np.concatenate(outS)[:n]
    P = np.concatenate(outP)[:n]
    D = np.concatenate(outD)[:n]
    return S, P, D


def mae(pred_bin: np.ndarray, true_bin: np.ndarray) -> float:
    return float((pred_bin != true_bin).mean())
