"""EDA for cat-in-the-dat: all-categorical binary classification, AUC metric."""
import pandas as pd
import numpy as np

DATA = "competitions/cat-in-the-dat/data"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

print(f"train shape {train.shape}, test shape {test.shape}")
print(f"target rate {train['target'].mean():.5f}")
print(f"nulls train {train.isnull().sum().sum()}, test {test.isnull().sum().sum()}")
print(f"dup feature rows train: {train.drop(columns=['id','target']).duplicated().sum()}")

feats = [c for c in train.columns if c not in ("id", "target")]
print("\ncol | card_train | card_test | test_unseen | test_unseen_rows | solo_auc(freq-ordered)")
from sklearn.metrics import roc_auc_score
for c in feats:
    tr_u = set(train[c].unique())
    te_u = set(test[c].unique())
    unseen = te_u - tr_u
    unseen_rows = test[c].isin(unseen).sum()
    # solo AUC via target-mean encoding on full train (optimistic, just for signal ranking)
    tm = train.groupby(c)["target"].mean()
    enc = train[c].map(tm)
    auc = roc_auc_score(train["target"], enc)
    print(f"{c} | {len(tr_u)} | {len(te_u)} | {len(unseen)} | {unseen_rows} | {auc:.4f}")

# ordinal orderings
print("\nord_1 values:", train["ord_1"].unique())
print("ord_2 values:", train["ord_2"].unique())
print("ord_3 sample:", sorted(train["ord_3"].unique()))
print("ord_4 sample:", sorted(train["ord_4"].unique()))
print("ord_5 card:", train["ord_5"].nunique(), "sample:", sorted(train["ord_5"].unique())[:10])

# day/month target rate
print("\nday target rates:", train.groupby("day")["target"].mean().round(4).to_dict())
print("month target rates:", train.groupby("month")["target"].mean().round(4).to_dict())
