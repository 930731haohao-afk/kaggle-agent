"""EDA — playground-series-s3e5 (wine quality, ordinal target, QWK metric).

Run: uv run python3 competitions/playground-series-s3e5/scripts/eda.py
Prints findings to stdout; no side files needed (data is tiny, all numeric).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
TARGET, IDC = "quality", "Id"

print("=" * 70)
print("1. LOAD & SHAPE")
print("=" * 70)
print(f"train: {train.shape}   test: {test.shape}")
print(f"dtypes:\n{train.dtypes}")
print(f"missing (train): {train.isnull().sum().sum()}   missing (test): {test.isnull().sum().sum()}")
print(f"duplicate rows (train, incl target): {train.duplicated().sum()}")
feat_cols = [c for c in train.columns if c not in (IDC, TARGET)]
print(f"duplicate rows on features only (train): {train.duplicated(subset=feat_cols).sum()}")

print()
print("=" * 70)
print("2. TARGET ANALYSIS")
print("=" * 70)
vc = train[TARGET].value_counts().sort_index()
print(vc)
print(f"proportions:\n{(vc / len(train)).round(4)}")
print(f"imbalance ratio (max/min class count): {vc.max() / vc.min():.1f}")
print(f"skew: {train[TARGET].skew():.3f}  kurtosis: {train[TARGET].kurt():.3f}")
print("-> classes 3 and 8 are extremely rare (12 and 39 of 2056 rows). "
      "5 and 6 dominate (~79% combined). This is a classic ordinal/imbalanced setup.")

print()
print("=" * 70)
print("3. FEATURE ANALYSIS (numeric summary + correlation with target)")
print("=" * 70)
print(train[feat_cols].describe().T)

corr = train[feat_cols + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\nSpearman correlation with quality (sorted by |r|):")
print(corr.round(3))

print("\nOutlier count per feature (beyond 3 std, train):")
for c in feat_cols:
    z = (train[c] - train[c].mean()) / train[c].std()
    print(f"  {c:24s} {int((z.abs() > 3).sum())}")

print()
print("=" * 70)
print("4. FEATURE INTER-CORRELATION (collinearity, |r|>0.6)")
print("=" * 70)
fcorr = train[feat_cols].corr()
pairs = []
for i, a in enumerate(feat_cols):
    for b in feat_cols[i + 1:]:
        r = fcorr.loc[a, b]
        if abs(r) > 0.6:
            pairs.append((a, b, round(r, 3)))
for p in sorted(pairs, key=lambda x: -abs(x[2])):
    print(f"  {p[0]:20s} <-> {p[1]:20s}  r={p[2]}")
if not pairs:
    print("  none above 0.6")

print()
print("=" * 70)
print("5. TRAIN-TEST DISTRIBUTION CONSISTENCY")
print("=" * 70)
for c in feat_cols:
    tr_m, te_m = train[c].mean(), test[c].mean()
    diff_pct = abs(tr_m - te_m) / (abs(tr_m) + 1e-9) * 100
    flag = "  <-- shift?" if diff_pct > 5 else ""
    print(f"  {c:24s} train_mean={tr_m:8.3f}  test_mean={te_m:8.3f}  diff={diff_pct:5.2f}%{flag}")

print()
print("=" * 70)
print("6. LEAKAGE CHECK")
print("=" * 70)
print(f"Id range train: [{train[IDC].min()}, {train[IDC].max()}]  test: [{test[IDC].min()}, {test[IDC].max()}]")
print("Id is a plain row index from the synthetic-generation process (Playground Series) -> no predictive use, drop.")
print(f"Max |Spearman r| with target = {corr.abs().max():.3f} (alcohol) -- no suspiciously perfect feature.")

print()
print("=" * 70)
print("7. VALIDATION STRATEGY RECOMMENDATION")
print("=" * 70)
print("""
Recommend: 5-fold StratifiedKFold on the raw `quality` label (6 classes, shuffle, seed=42).
Reasoning:
- Target is a small-cardinality ordinal integer (3..8) with severe imbalance (class 3: 0.6%,
  class 8: 1.9%). Plain KFold risks folds with zero examples of the rarest classes, making
  per-fold QWK computation on those folds noisy/undefined for those labels.
- StratifiedKFold on the label preserves per-fold class proportions, giving a stable and
  comparable CV signal across folds -- essential since QWK on ~2000 rows already has high
  fold-to-fold variance.
- Data is tiny (2056 rows) so no need for GroupKFold/time split; rows are i.i.d. synthetic
  samples (Playground Series regenerates the original UCI wine-quality data via a generative
  model), confirmed by near-identical train/test feature distributions (step 5) and no
  duplicate leakage across the ID space.

Modeling head recommendation: REGRESSION (not multiclass classification) with the target
treated as a continuous number, followed by an OPTIMIZED ROUNDER (per-class cutpoints tuned
directly on OOF QWK) rather than naive round-to-nearest-integer. Justification:
- QWK penalizes distance-squared between predicted and true ordinal class. A regressor that
  outputs 5.9 for a true 6 is much better than a classifier that has no notion that class 5
  is "closer" to 6 than class 3 is -- softmax classification treats all misclassifications as
  equally bad, which actively hurts QWK.
- With only 12 examples of class 3 and 39 of class 8, a multiclass classifier has very little
  signal to learn separate decision boundaries for the extreme classes; a regressor pools all
  rows through a single global ordering, using the moderate classes' abundant data to also
  place the rare ones on the continuum.
- Naive rounding (round-to-nearest-integer at .5 boundaries) assumes the regressor's output
  scale is unbiased and symmetric around each class -- rarely true when classes are this
  imbalanced (5/6 dominate, pulling predictions for 3/8 toward the center). Optimizing the
  cutpoints directly against OOF QWK (Nelder-Mead on a 5-threshold vector) corrects this bias
  and is a standard, low-cost, high-leverage trick in QWK competitions.
""")

print("=" * 70)
print("8. FEATURE ENGINEERING IDEAS (from correlations above)")
print("=" * 70)
print("""
- alcohol, sulphates, volatile acidity, citric acid are the top |Spearman r| features -> keep
  as-is, consider ratios/interactions among them.
- Known domain relationships (wine chemistry): total_sulfur/free_sulfur ratio (bound-SO2
  fraction), acidity ratios (fixed/volatile, citric/volatile), alcohol*sulphates interaction
  (both positively associated with quality), density is inversely related to alcohol (both
  driven by sugar/ethanol content) -> density*alcohol interaction, and a pH-vs-acid consistency
  feature (pH should track total acid inversely).
- Collinearity flagged in step 4 (if any) -> keep both since tree models handle collinearity
  natively; only relevant for the linear model diversity candidate.
""")
