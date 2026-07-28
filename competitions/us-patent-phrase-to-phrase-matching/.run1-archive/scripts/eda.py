"""Quick EDA for us-patent-phrase-to-phrase-matching (MLE-bench split)."""
import pandas as pd

DATA = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching/data"
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

print("train shape:", train.shape, "test shape:", test.shape)
print("nulls train:", train.isna().sum().sum(), "test:", test.isna().sum().sum())
print("\nscore value counts:\n", train.score.value_counts().sort_index())
print("\nunique anchors train:", train.anchor.nunique(), "test:", test.anchor.nunique())
print("unique targets train:", train.target.nunique(), "test:", test.target.nunique())
print("unique contexts train:", train.context.nunique(), "test:", test.context.nunique())
anchor_overlap = len(set(test.anchor) & set(train.anchor)) / test.anchor.nunique()
target_overlap = len(set(test.target) & set(train.target)) / test.target.nunique()
ctx_overlap = len(set(test.context) & set(train.context)) / test.context.nunique()
print(f"test anchor overlap with train: {anchor_overlap:.3f}")
print(f"test target overlap with train: {target_overlap:.3f}")
print(f"test context overlap with train: {ctx_overlap:.3f}")
pair_dup = pd.merge(test, train, on=["anchor", "target", "context"], how="inner")
print("exact (anchor,target,context) test rows found in train:", len(pair_dup))
print("\nanchor word len: max", train.anchor.str.split().str.len().max(),
      "target word len: max", train.target.str.split().str.len().max())
print("char len anchor max:", train.anchor.str.len().max(),
      "target max:", train.target.str.len().max())
print("\ncontext section (first letter) dist train:\n", train.context.str[0].value_counts())
print("\nrows per anchor: mean %.1f median %d max %d" % (
    train.groupby("anchor").size().mean(),
    train.groupby("anchor").size().median(),
    train.groupby("anchor").size().max()))
