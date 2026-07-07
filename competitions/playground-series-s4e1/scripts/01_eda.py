"""EDA for playground-series-s4e1 (Bank Churn Prediction)."""
import pandas as pd
import numpy as np
import os
from scipy import stats

data_dir = "competitions/playground-series-s4e1/data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=" * 70)
print("PLAYGROUND SERIES S4E1 - EDA")
print("=" * 70)

# ============================================================
# 1. TARGET ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("1. TARGET ANALYSIS")
print("=" * 70)

target = train['Exited']
print(f"Target distribution:")
vc = target.value_counts().sort_index()
for val, count in vc.items():
    print(f"  {val}: {count} ({count/len(train)*100:.1f}%)")
print(f"Imbalance ratio: {vc.max()/vc.min():.2f}")
print(f"Positive (churn) rate: {target.mean():.4f}")

# ============================================================
# 2. COLUMN CLASSIFICATION
# ============================================================
print("\n" + "=" * 70)
print("2. COLUMN CLASSIFICATION")
print("=" * 70)

drop_cols = ['id', 'Exited', 'CustomerId', 'Surname']
feature_cols = [c for c in train.columns if c not in drop_cols]

numerical_cols = []
categorical_cols = []
binary_cols = []
for col in feature_cols:
    if train[col].dtype in ['int64', 'float64']:
        if train[col].nunique() == 2:
            binary_cols.append(col)
        elif train[col].nunique() <= 10:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    else:
        categorical_cols.append(col)

print(f"Numerical ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")
print(f"Binary ({len(binary_cols)}): {binary_cols}")

# ============================================================
# 3. MISSING VALUES
# ============================================================
print("\n" + "=" * 70)
print("3. MISSING VALUES")
print("=" * 70)

for split_name, df in [("Train", train), ("Test", test)]:
    total_null = df.isnull().sum().sum()
    print(f"{split_name} total missing: {total_null}")

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

    # Target rate by bins
    binned = pd.qcut(train[col], q=5, duplicates='drop')
    grouped = train.groupby(binned)['Exited'].agg(['mean', 'count'])
    print(f"  Churn rate by quintile:")
    for idx, row in grouped.iterrows():
        print(f"    {idx}: churn_rate={row['mean']:.4f} (n={int(row['count'])})")

    corr = train[col].corr(train['Exited'])
    print(f"  corr with target: {corr:.4f}")

# ============================================================
# 5. CATEGORICAL & BINARY FEATURES
# ============================================================
print("\n" + "=" * 70)
print("5. CATEGORICAL & BINARY FEATURES")
print("=" * 70)

for col in categorical_cols + binary_cols:
    vals = train[col].dropna()
    n_unique = vals.nunique()
    print(f"\n{col}: {n_unique} unique values")

    grouped = train.groupby(col)['Exited'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for idx, row in grouped.iterrows():
        print(f"  {idx}: churn_rate={row['mean']:.4f}, count={int(row['count'])}")

# ============================================================
# 6. FEATURE CORRELATIONS
# ============================================================
print("\n" + "=" * 70)
print("6. FEATURE CORRELATIONS")
print("=" * 70)

# Numerical correlations with target
print("\nAll features correlation with Exited:")
all_num = numerical_cols + binary_cols
corrs = []
for col in all_num:
    c = train[col].corr(train['Exited'])
    corrs.append((col, c))
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for name, c in corrs:
    print(f"  {name:<25} corr={c:+.4f}")

# Cross-correlations
print("\nNumerical cross-correlations (|r| > 0.3):")
for i, col1 in enumerate(all_num):
    for col2 in all_num[i+1:]:
        c = train[col1].corr(train[col2])
        if abs(c) > 0.3:
            print(f"  {col1} vs {col2}: {c:.4f}")

# ============================================================
# 7. DETAILED FEATURE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("7. DETAILED FEATURE ANALYSIS")
print("=" * 70)

# Balance analysis
print("\nBalance distribution:")
zero_balance = (train['Balance'] == 0).sum()
print(f"  Zero balance: {zero_balance} ({zero_balance/len(train)*100:.1f}%)")
print(f"  Churn rate: zero_balance={train.loc[train['Balance']==0, 'Exited'].mean():.4f}, "
      f"non_zero={train.loc[train['Balance']>0, 'Exited'].mean():.4f}")

# NumOfProducts breakdown
print("\nNumOfProducts breakdown:")
for val in sorted(train['NumOfProducts'].unique()):
    mask = train['NumOfProducts'] == val
    print(f"  {val} products: n={mask.sum()}, churn_rate={train.loc[mask, 'Exited'].mean():.4f}")

# Age breakdown
print("\nAge breakdown (decades):")
train['age_decade'] = (train['Age'] // 10) * 10
for dec in sorted(train['age_decade'].unique()):
    mask = train['age_decade'] == dec
    print(f"  {int(dec)}s: n={mask.sum()}, churn_rate={train.loc[mask, 'Exited'].mean():.4f}")
train.drop(columns=['age_decade'], inplace=True)

# Geography x Gender interaction
print("\nGeography x Gender churn rates:")
cross = train.groupby(['Geography', 'Gender'])['Exited'].agg(['mean', 'count'])
print(cross.to_string())

# ============================================================
# 8. TRAIN-TEST CONSISTENCY
# ============================================================
print("\n" + "=" * 70)
print("8. TRAIN-TEST CONSISTENCY")
print("=" * 70)

for col in feature_cols:
    if train[col].dtype in ['int64', 'float64']:
        train_vals = train[col].dropna()
        test_vals = test[col].dropna()
        if len(train_vals) > 0 and len(test_vals) > 0:
            ks_stat, ks_p = stats.ks_2samp(train_vals, test_vals)
            mean_diff = abs(train_vals.mean() - test_vals.mean())
            flag = " *** SHIFT ***" if ks_p < 0.001 and mean_diff > 0.1 * train_vals.std() else ""
            print(f"  {col:<25} KS={ks_stat:.4f} p={ks_p:.4f} "
                  f"train_mean={train_vals.mean():.3f} test_mean={test_vals.mean():.3f}{flag}")
    else:
        train_cats = set(train[col].dropna().unique())
        test_cats = set(test[col].dropna().unique())
        new_in_test = test_cats - train_cats
        flag = f" NEW_IN_TEST={len(new_in_test)}" if new_in_test else ""
        print(f"  {col:<25} train={len(train_cats)} test={len(test_cats)}{flag}")

# ============================================================
# 9. LEAKAGE DETECTION
# ============================================================
print("\n" + "=" * 70)
print("9. LEAKAGE DETECTION")
print("=" * 70)

# ID correlation
id_corr = train['id'].corr(train['Exited'])
print(f"ID correlation with target: {id_corr:.6f}")

# CustomerId correlation
cid_corr = train['CustomerId'].corr(train['Exited'])
print(f"CustomerId correlation with target: {cid_corr:.6f}")

# Check Surname predictiveness
surname_stats = train.groupby('Surname')['Exited'].agg(['mean', 'count'])
high_count = surname_stats[surname_stats['count'] >= 20]
print(f"\nSurname predictiveness: {len(high_count)} surnames with 20+ occurrences")
print(f"  Target rate std across surnames: {high_count['mean'].std():.4f}")
print(f"  Overall churn rate: {target.mean():.4f}")
print(f"  Most churny surnames (n>=20):")
top_churn = high_count.nlargest(5, 'mean')
for name, row in top_churn.iterrows():
    print(f"    {name}: churn_rate={row['mean']:.4f}, count={int(row['count'])}")
least_churn = high_count.nsmallest(5, 'mean')
print(f"  Least churny surnames (n>=20):")
for name, row in least_churn.iterrows():
    print(f"    {name}: churn_rate={row['mean']:.4f}, count={int(row['count'])}")

# ============================================================
# 10. DUPLICATE CHECK
# ============================================================
print("\n" + "=" * 70)
print("10. DUPLICATE CHECK")
print("=" * 70)

feat_cols_no_id = [c for c in train.columns if c not in ['id', 'CustomerId']]
n_dup = train.duplicated(subset=feat_cols_no_id).sum()
print(f"Duplicate rows (excl id/CustomerId): {n_dup}")

common_cols = [c for c in feature_cols if c in test.columns]
n_overlap = pd.merge(train[common_cols], test[common_cols], how='inner').shape[0]
print(f"Train-test feature overlap rows: {n_overlap}")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
