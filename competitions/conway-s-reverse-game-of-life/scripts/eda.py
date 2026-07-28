"""EDA for Conway's Reverse Game of Life.

Checks:
1. delta distribution train/test
2. start/stop board density distributions
3. Verify forward GoL rule (dead boundary) maps start -> stop in exactly delta steps
4. Baseline MAEs: all-zero, stop-as-start (per delta)
5. Spatial density pattern (edge vs center)
"""
import numpy as np
import pandas as pd

DATA = "competitions/conway-s-reverse-game-of-life/data"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

start_cols = [c for c in train.columns if c.startswith("start")]
stop_cols = [c for c in train.columns if c.startswith("stop")]
print("train shape", train.shape, "test shape", test.shape)
print("n start cols", len(start_cols), "n stop cols", len(stop_cols))

print("\ndelta distribution train:")
print(train["delta"].value_counts().sort_index())
print("delta distribution test:")
print(test["delta"].value_counts().sort_index())

S = train[start_cols].to_numpy(np.int8).reshape(-1, 20, 20)
P = train[stop_cols].to_numpy(np.int8).reshape(-1, 20, 20)
T = test[[c for c in test.columns if c.startswith("stop")]].to_numpy(np.int8).reshape(-1, 20, 20)
delta = train["delta"].to_numpy()

print("\nstart density mean %.4f  stop density mean %.4f  test stop density %.4f"
      % (S.mean(), P.mean(), T.mean()))
for d in range(1, 6):
    m = delta == d
    print(f"delta={d}: start density {S[m].mean():.4f}  stop density {P[m].mean():.4f}")


def life_step(b: np.ndarray) -> np.ndarray:
    """One GoL step, dead boundary. b: (N,20,20) int8."""
    p = np.pad(b, ((0, 0), (1, 1), (1, 1)))
    n = (p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:] +
         p[:, 1:-1, :-2] + p[:, 1:-1, 2:] +
         p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:])
    return ((n == 3) | ((b == 1) & (n == 2))).astype(np.int8)


# verify forward rule on 2000 sample rows
rng = np.random.RandomState(42)
idx = rng.choice(len(S), 2000, replace=False)
ok = 0
for i in idx:
    b = S[i:i + 1]
    for _ in range(delta[i]):
        b = life_step(b)
    ok += int((b[0] == P[i]).all())
print(f"\nforward-rule check (dead boundary): {ok}/2000 rows reproduce stop exactly")

# baselines
zero_mae = S.mean()
same_mae = (S != P).mean()
print(f"\nall-zero MAE {zero_mae:.5f}   stop-as-start MAE {same_mae:.5f}")
for d in range(1, 6):
    m = delta == d
    print(f"delta={d}: zero {S[m].mean():.5f}  stop-as-start {(S[m] != P[m]).mean():.5f}")

# spatial pattern
print("\nstart density by row (edge effect):")
print(np.round(S.mean(axis=(0, 2)), 3))

# stop empty boards?
print("\nempty stop boards train:", int((P.sum(axis=(1, 2)) == 0).sum()),
      "test:", int((T.sum(axis=(1, 2)) == 0).sum()))
print("empty start boards train:", int((S.sum(axis=(1, 2)) == 0).sum()))
