"""EDA for playground-series-s4e11 (Depression Prediction)."""
import pandas as pd
import numpy as np
import os
from scipy import stats

data_dir = "competitions/playground-series-s4e11/data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=" * 70)
print("PLAYGROUND SERIES S4E11 - EDA")
print("=" * 70)

# ============================================================
# 1. TARGET ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("1. TARGET ANALYSIS")
print("=" * 70)

target = train['Depression']
print(f"Target distribution:")
vc = target.value_counts().sort_index()
for val, count in vc.items():
    print(f"  {val}: {count} ({count/len(train)*100:.1f}%)")
print(f"Imbalance ratio: {vc.max()/vc.min():.2f}")
print(f"Positive rate: {target.mean():.4f}")

# ============================================================
# 2. COLUMN CLASSIFICATION
# ============================================================
print("\n" + "=" * 70)
print("2. COLUMN CLASSIFICATION")
print("=" * 70)

drop_cols = ['id', 'Depression']
feature_cols = [c for c in train.columns if c not in drop_cols]

numerical_cols = []
categorical_cols = []
for col in feature_cols:
    if train[col].dtype in ['int64', 'float64']:
        if train[col].nunique() <= 10:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    else:
        categorical_cols.append(col)

print(f"Numerical ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")

# ============================================================
# 3. MISSING VALUES ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("3. MISSING VALUES ANALYSIS")
print("=" * 70)

for split_name, df in [("Train", train), ("Test", test)]:
    total_null = df.isnull().sum().sum()
    print(f"\n{split_name} total missing: {total_null}")
    for col in df.columns:
        n = df[col].isnull().sum()
        if n > 0:
            print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")

# Check if missing pattern is structured
print("\nMissing pattern analysis:")
has_wp = train['Working Professional or Student']
for status in has_wp.unique():
    mask = train['Working Professional or Student'] == status
    subset = train[mask]
    print(f"\n  '{status}' ({mask.sum()} rows):")
    for col in train.columns:
        n = subset[col].isnull().sum()
        if n > 0:
            print(f"    {col}: {n}/{mask.sum()} ({n/mask.sum()*100:.1f}%)")

# Check if missingness correlates with target
print("\nMissingness vs Target:")
for col in train.columns:
    if train[col].isnull().sum() > 0:
        is_missing = train[col].isnull().astype(int)
        dep_when_missing = train.loc[is_missing == 1, 'Depression'].mean()
        dep_when_present = train.loc[is_missing == 0, 'Depression'].mean()
        print(f"  {col}: missing→dep_rate={dep_when_missing:.4f}, present→dep_rate={dep_when_present:.4f}")

# ============================================================
# 4. NUMERICAL FEATURES
# ============================================================
print("\n" + "=" * 70)
print("4. NUMERICAL FEATURES")
print("=" * 70)

for col in numerical_cols:
    vals = train[col].dropna()
    print(f"\n{col}:")
    print(f"  count={len(vals)}, mean={vals.mean():.2f}, std={vals.std():.2f}")
    print(f"  min={vals.min()}, 25%={vals.quantile(0.25):.2f}, 50%={vals.median():.2f}, "
          f"75%={vals.quantile(0.75):.2f}, max={vals.max()}")
    print(f"  skew={vals.skew():.3f}, kurtosis={vals.kurtosis():.3f}")
    outliers = ((vals - vals.mean()).abs() > 3 * vals.std()).sum()
    print(f"  outliers (>3std): {outliers} ({outliers/len(vals)*100:.2f}%)")

    # Correlation with target
    valid = train[[col, 'Depression']].dropna()
    if len(valid) > 10:
        corr = valid[col].corr(valid['Depression'])
        print(f"  corr with target: {corr:.4f}")

# ============================================================
# 5. CATEGORICAL FEATURES
# ============================================================
print("\n" + "=" * 70)
print("5. CATEGORICAL FEATURES")
print("=" * 70)

for col in categorical_cols:
    vals = train[col].dropna()
    n_unique = vals.nunique()
    print(f"\n{col}: {n_unique} unique values")

    if n_unique <= 20:
        # Show all values with target rate
        grouped = train.groupby(col)['Depression'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        for idx, row in grouped.iterrows():
            print(f"  {idx}: dep_rate={row['mean']:.4f}, count={int(row['count'])}")
    else:
        # Show top 10 + target rate
        top = vals.value_counts().head(10)
        print(f"  Top 10 values:")
        for val, cnt in top.items():
            dep_rate = train.loc[train[col] == val, 'Depression'].mean()
            print(f"    {val}: count={cnt}, dep_rate={dep_rate:.4f}")

        # Rare categories
        rare = (vals.value_counts() / len(vals) < 0.01).sum()
        print(f"  Rare categories (<1%): {rare}/{n_unique}")

# ============================================================
# 6. FEATURE-TARGET RELATIONSHIPS (numerical)
# ============================================================
print("\n" + "=" * 70)
print("6. FEATURE-TARGET CORRELATIONS")
print("=" * 70)

# Numerical correlations with target
print("\nNumerical features correlation with Depression:")
corrs = []
for col in numerical_cols:
    valid = train[[col, 'Depression']].dropna()
    if len(valid) > 10:
        c = valid[col].corr(valid['Depression'])
        corrs.append((col, c, len(valid)))
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for name, c, n in corrs:
    print(f"  {name:<30} corr={c:+.4f} (n={n})")

# Cross-correlation among numerical features
print("\nNumerical cross-correlations (|r| > 0.3):")
for i, col1 in enumerate(numerical_cols):
    for col2 in numerical_cols[i+1:]:
        valid = train[[col1, col2]].dropna()
        if len(valid) > 10:
            c = valid[col1].corr(valid[col2])
            if abs(c) > 0.3:
                print(f"  {col1} vs {col2}: {c:.4f}")

# ============================================================
# 7. TRAIN-TEST CONSISTENCY
# ============================================================
print("\n" + "=" * 70)
print("7. TRAIN-TEST CONSISTENCY")
print("=" * 70)

for col in feature_cols:
    if col in ['Name']:
        continue

    if train[col].dtype in ['int64', 'float64']:
        train_vals = train[col].dropna()
        test_vals = test[col].dropna()
        if len(train_vals) > 0 and len(test_vals) > 0:
            ks_stat, ks_p = stats.ks_2samp(train_vals, test_vals)
            mean_diff = abs(train_vals.mean() - test_vals.mean())
            flag = " *** SHIFT ***" if ks_p < 0.001 and mean_diff > 0.1 * train_vals.std() else ""
            print(f"  {col:<30} KS={ks_stat:.4f} p={ks_p:.4f} "
                  f"train_mean={train_vals.mean():.3f} test_mean={test_vals.mean():.3f}{flag}")
    else:
        train_cats = set(train[col].dropna().unique())
        test_cats = set(test[col].dropna().unique())
        new_in_test = test_cats - train_cats
        missing_in_test = train_cats - test_cats
        flag = ""
        if len(new_in_test) > 0:
            flag += f" NEW_IN_TEST={len(new_in_test)}"
        if len(missing_in_test) > 0:
            flag += f" MISSING_IN_TEST={len(missing_in_test)}"
        print(f"  {col:<30} train={len(train_cats)} test={len(test_cats)}{flag}")

# ============================================================
# 8. LEAKAGE DETECTION
# ============================================================
print("\n" + "=" * 70)
print("8. LEAKAGE DETECTION")
print("=" * 70)

# Check Suicidal Thoughts - could be near-leakage
if 'Have you ever had suicidal thoughts ?' in train.columns:
    col = 'Have you ever had suicidal thoughts ?'
    ct = pd.crosstab(train[col], train['Depression'], normalize='index')
    print(f"\nSuicidal Thoughts vs Depression (row-normalized):")
    print(ct.to_string())

# Check Family History
if 'Family History of Mental Illness' in train.columns:
    col = 'Family History of Mental Illness'
    ct = pd.crosstab(train[col], train['Depression'], normalize='index')
    print(f"\nFamily History vs Depression (row-normalized):")
    print(ct.to_string())

# Check if Name is predictive (should not be)
name_counts = train.groupby('Name')['Depression'].agg(['mean', 'count'])
high_count = name_counts[name_counts['count'] >= 10]
name_target_std = high_count['mean'].std()
print(f"\nName predictiveness: {len(high_count)} names with 10+ occurrences, "
      f"target_rate std={name_target_std:.4f}")
print("  (low std = not predictive, expected for names)")

# ID leakage check
id_corr = train['id'].corr(train['Depression'])
print(f"\nID correlation with target: {id_corr:.6f}")

# ============================================================
# 9. SPECIAL ANALYSIS: Working Professional vs Student
# ============================================================
print("\n" + "=" * 70)
print("9. WORKING PROFESSIONAL vs STUDENT SPLIT")
print("=" * 70)

for status in train['Working Professional or Student'].unique():
    mask = train['Working Professional or Student'] == status
    subset = train[mask]
    print(f"\n--- {status} ({mask.sum()} rows, {mask.sum()/len(train)*100:.1f}%) ---")
    print(f"  Depression rate: {subset['Depression'].mean():.4f}")

    for col in feature_cols:
        if col in ['Working Professional or Student', 'Name']:
            continue
        vals = subset[col].dropna()
        if len(vals) == 0:
            continue
        if vals.dtype in ['int64', 'float64']:
            # Correlation with target within this group
            valid = subset[[col, 'Depression']].dropna()
            if len(valid) > 10:
                c = valid[col].corr(valid['Depression'])
                if abs(c) > 0.1:
                    print(f"  {col}: mean={vals.mean():.2f}, corr_with_dep={c:+.4f}")
        else:
            if vals.nunique() <= 10:
                dep_range = subset.groupby(col)['Depression'].mean()
                spread = dep_range.max() - dep_range.min()
                if spread > 0.05:
                    print(f"  {col}: dep_rate spread={spread:.4f} ({dep_range.idxmin()}={dep_range.min():.3f} "
                          f"to {dep_range.idxmax()}={dep_range.max():.3f})")

# ============================================================
# 10. DUPLICATE CHECK
# ============================================================
print("\n" + "=" * 70)
print("10. DUPLICATE CHECK")
print("=" * 70)

# Check for duplicate rows (excluding id)
feat_cols_no_id = [c for c in train.columns if c != 'id']
n_dup = train.duplicated(subset=feat_cols_no_id).sum()
print(f"Duplicate rows (excl id): {n_dup}")

# Check train-test overlap (excluding id and target)
common_cols = [c for c in train.columns if c in test.columns and c != 'id']
n_overlap = pd.merge(train[common_cols], test[common_cols], how='inner').shape[0]
print(f"Train-test feature overlap rows: {n_overlap}")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
