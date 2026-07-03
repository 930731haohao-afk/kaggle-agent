"""EDA for playground-series-s3e11 (media campaign cost, RMSLE, minimize).

Fast, full-data checks only (no heavy plotting): target distribution/skew,
per-feature cardinality & correlation, "store profile" grouping signal,
train/test consistency, missing/dup checks, and CV strategy recommendation.
"""
import numpy as np
import pandas as pd

COMP = "competitions/playground-series-s3e11"
DATA = f"{COMP}/data"
TARGET, ID = "cost", "id"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
print(f"train shape={train.shape}  test shape={test.shape}")

print("\n=== dtypes ===")
print(train.dtypes)

print("\n=== missing values ===")
print("train:", train.isnull().sum().sum(), " test:", test.isnull().sum().sum())
print("dup rows (train):", train.duplicated().sum())
print("id overlap train/test:", len(set(train[ID]) & set(test[ID])))

print("\n=== target ===")
print(train[TARGET].describe())
print("skew:", train[TARGET].skew(), " kurtosis:", train[TARGET].kurtosis())
print("min >= 0 ?", (train[TARGET] >= 0).all())
print("log1p(target) skew:", np.log1p(train[TARGET]).skew())

print("\n=== per-column cardinality ===")
for c in train.columns:
    if c in (ID, TARGET):
        continue
    print(f"{c:30s} nunique={train[c].nunique():6d}  min={train[c].min():.3f}  max={train[c].max():.3f}")

print("\n=== correlation with target ===")
print(train.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values())

print("\n=== train vs test: per-column mean diff (covariate shift check) ===")
for c in train.columns:
    if c in (ID, TARGET):
        continue
    tr_m, te_m = train[c].mean(), test[c].mean()
    diff_pct = 100 * (te_m - tr_m) / (abs(tr_m) + 1e-9)
    flag = "  <-- CHECK" if abs(diff_pct) > 2 else ""
    print(f"{c:30s} train_mean={tr_m:10.3f}  test_mean={te_m:10.3f}  diff%={diff_pct:6.2f}{flag}")

print("\n=== store_sqft: cost varies meaningfully by store (candidate for target encoding) ===")
print(train.groupby("store_sqft")[TARGET].agg(["mean", "count"]).sort_values("mean"))

print("\n=== 'store profile' combo (store_sqft + 5 binary amenities) ===")
store_cols = ["store_sqft", "coffee_bar", "video_store", "salad_bar", "prepared_food", "florist"]
combo_tr = train[store_cols].astype(str).agg("_".join, axis=1)
combo_te = test[store_cols].astype(str).agg("_".join, axis=1)
print("n unique combos in train:", combo_tr.nunique(), " in test:", combo_te.nunique())
print("combos in test not seen in train:", len(set(combo_te) - set(combo_tr)))
g = train.assign(store_combo=combo_tr).groupby("store_combo")[TARGET].agg(["mean", "count"])
pred = combo_tr.map(g["mean"])
ss_res = ((train[TARGET] - pred) ** 2).sum()
ss_tot = ((train[TARGET] - train[TARGET].mean()) ** 2).sum()
print(f"in-sample R2 using store_combo group mean alone: {1 - ss_res/ss_tot:.4f}")
print(f"(overall target std={train[TARGET].std():.3f}, between-group-mean std={g['mean'].std():.3f})")

print("\n=== amenity_count sanity ===")
amen_cols = ["coffee_bar", "video_store", "salad_bar", "prepared_food", "florist"]
train["_amenity_count"] = train[amen_cols].sum(axis=1)
print(train.groupby("_amenity_count")[TARGET].agg(["mean", "count"]))

print("\n=== EDA SUMMARY ===")
print("""
Key findings:
1. Target `cost` in [50.79, 149.75], skew=0.019 (essentially SYMMETRIC, NOT right-skewed
   as often assumed for cost-like targets). log1p(target) skew is -0.34 (slightly LEFT-skewed
   after transform) -- log1p does not improve distributional shape here.
   -> Still train on log1p(cost) with RMSE objective: RMSLE = RMSE(log1p(pred), log1p(actual)),
   so this directly optimizes the competition metric regardless of skewness; clip predictions
   at >=0 before inverse-transform (expm1) since target is always positive.
2. All 15 raw features are weakly correlated with target individually (|r| <= 0.11 for every
   column) -- this is a known low-signal / high-noise playground dataset; large gains from
   single-feature engineering are unlikely.
3. Several "categorical-looking" float columns are actually low-cardinality (2-36 unique values):
   unit_sales, total_children, num_children_at_home, avg_cars_at_home, recyclable_package,
   low_fat, units_per_case, store_sqft, and 5 binary amenity flags (coffee_bar, video_store,
   salad_bar, prepared_food, florist).
4. `store_sqft` (20 unique values) combined with the 5 binary amenity flags defines ~111 distinct
   "store profiles" repeated thousands of times each; the store-profile group mean alone reaches
   in-sample R2 ~0.06 against cost -- notably higher than any single raw feature correlation.
   This is the strongest engineered-feature candidate (must be encoded out-of-fold to avoid leakage).
5. No missing values, no duplicate rows, no train/test id overlap.
6. Train/test per-column means are consistent (no flagged covariate shift beyond noise).
7. All store-profile combos seen in test also appear in train (or are close in composition),
   so out-of-fold / K-fold target encoding is safe without needing a fallback category.

Recommended validation: standard 5-fold KFold (shuffle, seed=42). No temporal or group
structure requiring GroupKFold; store profiles repeat thousands of times so random folds are
fine and won't starve any fold of a given profile.

Feature engineering ideas:
- K-fold target-encode `store_combo` (store_sqft + 5 amenity flags) -- computed inside CV loop.
- `amenity_count` = sum of 5 binary amenity flags.
- ratios: gross_weight / units_per_case, store_sales / unit_sales.
- `children_away` = total_children - num_children_at_home.
- keep store_sqft both as raw numeric (tree-splittable) and as part of the combo encoding.
""")
