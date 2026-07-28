"""Validate synthetic GoL generator against real train distribution.

Competition generation (per description): 20x20 board, cells filled i.i.d. with a
random density, warmed up 5 steps -> "start"; evolved delta more steps -> "stop".
Empty boards discarded.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
DIR = "competitions/conway-s-reverse-game-of-life"


def life_steps(b, n_steps):
    """Vectorized finite-grid GoL for batch (N,20,20) int8."""
    for _ in range(n_steps):
        p = np.pad(b, ((0, 0), (1, 1), (1, 1)))
        n = sum(np.roll(np.roll(p, i, 1), j, 2) for i in (-1, 0, 1) for j in (-1, 0, 1)) - p
        b = (((n == 3) | ((p == 1) & (n == 2))).astype(np.int8))[:, 1:-1, 1:-1]
    return b


def generate(n, delta, rng):
    """Generate n (start, stop) pairs for given delta."""
    starts, stops = [], []
    got = 0
    while got < n:
        m = n - got + max(64, (n - got) // 2)
        dens = rng.uniform(0.01, 0.99, size=(m, 1, 1))
        init = (rng.random((m, 20, 20)) < dens).astype(np.int8)
        start = life_steps(init, 5)
        alive = start.sum(axis=(1, 2)) > 0
        start = start[alive]
        stop = life_steps(start, delta)
        alive2 = stop.sum(axis=(1, 2)) > 0
        start, stop = start[alive2], stop[alive2]
        take = min(n - got, len(start))
        starts.append(start[:take]); stops.append(stop[:take])
        got += take
    return np.concatenate(starts), np.concatenate(stops)


train = pd.read_csv(f"{DIR}/data/train.csv")
sc = [f"start.{i}" for i in range(1, 401)]
pc = [f"stop.{i}" for i in range(1, 401)]

for delta in (1, 3, 5):
    real_s = train.loc[train.delta == delta, sc].values.reshape(-1, 400)
    real_p = train.loc[train.delta == delta, pc].values.reshape(-1, 400)
    syn_s, syn_p = generate(5000, delta, rng)
    syn_s = syn_s.reshape(-1, 400); syn_p = syn_p.reshape(-1, 400)
    print(f"delta={delta}:")
    print(f"  start density  real {real_s.mean():.4f}  syn {syn_s.mean():.4f}")
    print(f"  stop density   real {real_p.mean():.4f}  syn {syn_p.mean():.4f}")
    qs = [10, 25, 50, 75, 90]
    print(f"  start per-board density pct real {np.percentile(real_s.mean(1), qs).round(3)}")
    print(f"                            syn  {np.percentile(syn_s.mean(1), qs).round(3)}")
