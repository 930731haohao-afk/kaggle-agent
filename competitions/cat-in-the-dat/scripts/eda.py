"""EDA for cat-in-the-dat: all-categorical binary classification, AUC metric."""
import pandas as pd
import numpy as np

COMP = "competitions/cat-in-the-dat"
train = pd.read_csv(f"{COMP}/data/train.csv")
test = pd.read_csv(f"{COMP}/data/test.csv")

print(f"train {train.shape}, test {test.shape}")
print(f"target rate: {train['target'].mean():.5f}  (pos {train['target'].sum()})")
print(f"missing train: {train.isna().sum().sum()}, test: {test.isna().sum().sum()}")
print(f"dup rows (ex id): {train.drop(columns=['id','target']).duplicated().sum()}")

feats = [c for c in train.columns if c not in ("id", "target")]
print(f"\n{'col':8s} {'nuniq_tr':>8s} {'nuniq_te':>8s} {'te_only':>8s} {'te_only_rows':>12s} {'tr_only':>8s}")
for c in feats:
    tr_u = set(train[c].astype(str).unique())
    te_u = set(test[c].astype(str).unique())
    te_only = te_u - tr_u
    te_only_rows = test[c].astype(str).isin(te_only).sum() if te_only else 0
    print(f"{c:8s} {len(tr_u):8d} {len(te_u):8d} {len(te_only):8d} {te_only_rows:12d} {len(tr_u - te_u):8d}")

# target rate per category for low-card cols
print("\n-- target rate by category (low-card) --")
for c in feats:
    if train[c].nunique() <= 12:
        g = train.groupby(c)["target"].agg(["mean", "count"])
        print(f"{c}: " + "  ".join(f"{i}={r['mean']:.3f}(n={int(r['count'])})" for i, r in g.iterrows()))

# ordinal ordering check
print("\n-- ord_3/4/5 monotonicity vs target --")
for c in ["ord_3", "ord_4"]:
    g = train.groupby(c)["target"].mean().sort_index()
    corr = np.corrcoef(np.arange(len(g)), g.values)[0, 1]
    print(f"{c}: sorted-alpha vs target-rate corr {corr:.3f}, nuniq {len(g)}")
g5 = train.groupby("ord_5")["target"].mean().sort_index()
print(f"ord_5: nuniq {len(g5)}, alpha-order corr {np.corrcoef(np.arange(len(g5)), g5.values)[0,1]:.3f}")

# high-card nom rare categories
print("\n-- high-card nominal frequency profile --")
for c in ["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]:
    vc = train[c].value_counts()
    print(f"{c}: nuniq {len(vc)}, min count {vc.min()}, median {int(vc.median())}, max {vc.max()}, "
          f"singletons {(vc==1).sum()}")

# day/month cyclical
print("\n-- day/month target rate --")
print("day:", train.groupby("day")["target"].mean().round(3).to_dict())
print("month:", train.groupby("month")["target"].mean().round(3).to_dict())
