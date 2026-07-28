"""EDA for Conway's Reverse Game of Life."""
import numpy as np
import pandas as pd

DIR = "competitions/conway-s-reverse-game-of-life"

train = pd.read_csv(f"{DIR}/data/train.csv")
test = pd.read_csv(f"{DIR}/data/test.csv")

start_cols = [f"start.{i}" for i in range(1, 401)]
stop_cols = [f"stop.{i}" for i in range(1, 401)]

Y = train[start_cols].values.astype(np.int8)   # start boards (target)
X = train[stop_cols].values.astype(np.int8)    # stop boards (features)
Xt = test[stop_cols].values.astype(np.int8)
d = train["delta"].values
dt = test["delta"].values

print("=== delta distribution ===")
print("train:", np.bincount(d)[1:], " test:", np.bincount(dt)[1:])

print("\n=== cell density ===")
print(f"start alive rate: {Y.mean():.4f}")
print(f"stop alive rate (train): {X.mean():.4f}  (test): {Xt.mean():.4f}")
for k in range(1, 6):
    m = d == k
    print(f"delta={k}: start density {Y[m].mean():.4f}, stop density {X[m].mean():.4f}, n={m.sum()}")

print("\n=== baselines (MAE = mean abs error over 400 cells) ===")
print(f"all-zeros: {Y.mean():.5f}")
ident = np.abs(Y - X).mean()
print(f"identity (start=stop): {ident:.5f}")
for k in range(1, 6):
    m = d == k
    print(f"  delta={k}: identity MAE {np.abs(Y[m] - X[m]).mean():.5f}, zeros {Y[m].mean():.5f}")

print("\n=== per-board stats ===")
alive_stop = X.mean(axis=1)
print(f"stop board alive fraction: min {alive_stop.min():.3f} max {alive_stop.max():.3f} mean {alive_stop.mean():.3f}")
print(f"boards with all-dead stop: train {(X.sum(axis=1)==0).sum()}, test {(Xt.sum(axis=1)==0).sum()}")

# verify forward GoL: evolve start by delta -> should equal stop
def life_step(b):
    n = sum(np.roll(np.roll(b, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) - b
    return ((n == 3) | ((b == 1) & (n == 2))).astype(np.int8)

# NOTE: boards are 20x20 finite grid (cells outside are dead) — use padded step
def life_step_finite(b):
    p = np.pad(b, 1)
    n = sum(np.roll(np.roll(p, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) - p
    r = ((n == 3) | ((p == 1) & (n == 2))).astype(np.int8)
    return r[1:-1, 1:-1]

print("\n=== forward-rule verification (finite grid) on 200 boards ===")
ok = 0
for i in range(200):
    b = Y[i].reshape(20, 20)
    for _ in range(d[i]):
        b = life_step_finite(b)
    ok += int((b == X[i].reshape(20, 20)).all())
print(f"finite-grid forward evolution matches stop: {ok}/200")

print("\n=== train/test consistency: stop density by delta ===")
for k in range(1, 6):
    print(f"delta={k}: train {X[d==k].mean():.4f} vs test {Xt[dt==k].mean():.4f}")
