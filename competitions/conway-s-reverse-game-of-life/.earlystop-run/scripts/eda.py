"""EDA for Conway's Reverse Game of Life."""
import numpy as np
import pandas as pd

DATA = "competitions/conway-s-reverse-game-of-life/data"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

start_cols = [c for c in train.columns if c.startswith("start.")]
stop_cols = [c for c in train.columns if c.startswith("stop.")]
print("train shape", train.shape, "test shape", test.shape)
print("start cols", len(start_cols), "stop cols", len(stop_cols))
print("nulls train", int(train.isnull().sum().sum()), "test", int(test.isnull().sum().sum()))

print("\ndelta distribution train:\n", train["delta"].value_counts().sort_index())
print("delta distribution test:\n", test["delta"].value_counts().sort_index())

S = train[start_cols].values.astype(np.int8)
P = train[stop_cols].values.astype(np.int8)

print("\noverall start density %.4f" % S.mean())
print("overall stop density  %.4f" % P.mean())
for d in range(1, 6):
    m = train["delta"].values == d
    print(f"delta={d}: n={m.sum()} start_density={S[m].mean():.4f} stop_density={P[m].mean():.4f} "
          f"copy_stop_MAE={np.abs(S[m]-P[m]).mean():.4f} allzero_MAE={S[m].mean():.4f}")

# empty stop boards (board died) — start still nonzero
stop_empty = P.sum(axis=1) == 0
print("\nempty stop boards: %d (%.2f%%)" % (stop_empty.sum(), 100*stop_empty.mean()))
if stop_empty.sum():
    print("start density for empty-stop rows: %.4f" % S[stop_empty].mean())
    print("delta dist of empty-stop:", train.loc[stop_empty, "delta"].value_counts().sort_index().to_dict())

# test stop density by delta (train/test consistency)
Pt = test[[c for c in test.columns if c.startswith("stop.")]].values.astype(np.int8)
for d in range(1, 6):
    m = test["delta"].values == d
    print(f"test delta={d}: stop_density={Pt[m].mean():.4f}")
te_empty = Pt.sum(axis=1) == 0
print("test empty stop boards: %d (%.2f%%)" % (te_empty.sum(), 100*te_empty.mean()))

# same-cell correlation start vs stop per delta
for d in range(1, 6):
    m = train["delta"].values == d
    s, p = S[m].ravel(), P[m].ravel()
    print(f"delta={d}: corr(start,stop same cell)={np.corrcoef(s, p)[0,1]:.4f} "
          f"P(start=1|stop=1)={s[p==1].mean():.4f} P(start=1|stop=0)={s[p==0].mean():.4f}")

# edge vs center density (20x20 boards, toroidal? kaggle used bounded grid)
G = S.reshape(-1, 20, 20)
print("\nstart density: corners %.4f edges %.4f center %.4f" % (
    G[:, [0,0,-1,-1], [0,-1,0,-1]].mean(),
    np.concatenate([G[:,0,:], G[:,-1,:], G[:,:,0], G[:,:,-1]], axis=1).mean(),
    G[:, 5:15, 5:15].mean()))
Gp = P.reshape(-1, 20, 20)
print("stop density : edges %.4f center %.4f" % (
    np.concatenate([Gp[:,0,:], Gp[:,-1,:], Gp[:,:,0], Gp[:,:,-1]], axis=1).mean(),
    Gp[:, 5:15, 5:15].mean()))
