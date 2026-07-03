"""EDA for playground-series-s3e14 (Wild Blueberry Yield, regression, MAE).

Prints findings to stdout: target distribution/discreteness, feature
correlations, train/test consistency, missing values, duplicate rows.
"""
import numpy as np
import pandas as pd

COMP = "competitions/playground-series-s3e14"
DATA = f"{COMP}/data"
TARGET, ID = "yield", "id"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

print(f"Train shape: {train.shape}   Test shape: {test.shape}")
print(f"Columns: {train.columns.tolist()}")
print(f"\nMissing values (train): {train.isnull().sum().sum()}  (test): {test.isnull().sum().sum()}")
print(f"Duplicate rows (train, excl id): {train.drop(columns=[ID]).duplicated().sum()}")

print("\n=== Target distribution ===")
print(train[TARGET].describe())
print(f"skew={train[TARGET].skew():.4f}  kurtosis={train[TARGET].kurtosis():.4f}")
print(f"unique yield values: {train[TARGET].nunique()} of {len(train)} rows "
      f"({100*train[TARGET].nunique()/len(train):.1f}% unique) -> repeated but NOT a coarse integer grid "
      f"(unlike s3e16 Age 1-29). Rounding to nearest integer is meaningless here; "
      f"nearest-observed-value snapping is the only discrete-grid idea worth testing empirically.")

print("\n=== Per-feature cardinality (low-cardinality env vars are effectively categorical) ===")
for c in ["clonesize", "honeybee", "bumbles", "andrena", "osmia", "RainingDays", "AverageRainingDays"]:
    print(f"  {c}: {train[c].nunique()} unique -> {sorted(train[c].unique())[:8]}")

print("\n=== Correlation with target (sorted) ===")
print(train.drop(columns=[ID]).corr(numeric_only=True)[TARGET].sort_values())

print("\n=== Train vs test mean shift (%) ===")
for c in train.columns:
    if c in (ID, TARGET):
        continue
    tr_m, te_m = train[c].mean(), test[c].mean()
    pct = 100 * (te_m - tr_m) / (abs(tr_m) + 1e-9)
    print(f"  {c}: train={tr_m:.3f} test={te_m:.3f} shift={pct:.2f}%")
print("-> all shifts < 1% => no covariate shift, standard KFold is safe.")

print("\n=== Upper/Lower T-range collinearity check ===")
temp_cols = ["MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange",
             "MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange"]
print(train[temp_cols].corr().round(3))
print("-> temp range columns are near-perfectly collinear with each other (season blocks); "
      "raw corr with yield is tiny (~-0.02) individually, likely because they take only a handful "
      "of discrete season-block values (near constant blocks) rather than being continuous drivers.")

print("\n=== Fruit-biology features (near-deterministic proxies for yield) ===")
print(train[["fruitset", "fruitmass", "seeds", TARGET]].corr())
print("-> fruitset/fruitmass/seeds correlate 0.83-0.89 with yield: dominant predictive block.")

print("\n=== Pollinator features ===")
poll = ["honeybee", "bumbles", "andrena", "osmia"]
print(train[poll + [TARGET]].corr()[TARGET])
print("-> weak individually; a summed 'total_pollinators' index may capture combined pollination pressure.")

print("\n=== EDA Summary ===")
print("""
Data Overview:
- Train: 15289 rows x 18 cols (16 features + id + yield). Test: 10194 rows x 17 cols. No missing values.
- Target 'yield': continuous, mean~6025, std~1337, range [1945.5, 8969.4], skew ~ -0.16 (roughly symmetric).
- 776 distinct yield values out of 15289 rows (~5% unique) -- repeats come from shared combinations of the
  low-cardinality env variables (clonesize, RainingDays, pollinator densities take only 6-16 distinct values
  each) crossed with continuous fruitset/fruitmass/seeds, not a coarse rounding grid. Snapping predictions to
  the nearest *observed train yield value* is worth testing as a cheap post-processing experiment, but there
  is no simple "round to nearest integer/step" grid like s3e16's Age.

Key Findings:
1. fruitset, fruitmass, seeds are the dominant predictive block (r=0.83-0.89 with yield) -- essentially
   biological outcome measures that summarize pollination success; ratios/products of these should be strong.
2. clonesize, RainingDays/AverageRainingDays negatively correlate with yield (r=-0.38 to -0.48) and are
   low-cardinality (6-8 distinct values) -- effectively categorical/ordinal "regime" variables.
3. Temperature range columns (Max/Min/Average of Upper/Lower TRange) are heavily collinear with each other
   (near-duplicate season-block encodings) and individually weak (|r|~0.02); likely redundant -- good
   candidates for a single condensed feature (e.g. temp range midpoint / spread) rather than 6 raw columns.
4. Pollinator density columns (honeybee, bumbles, andrena, osmia) are individually weak (|r| 0.07-0.20) but a
   combined index may proxy total pollination pressure.
5. No missing values anywhere; only 7 exact duplicate rows in train (harmless, left as-is).
6. Train/test feature means differ by <1% on every column -> no covariate shift.

Concerns:
- No categorical encoding needed (all numeric); low risk of leakage since fruitset/fruitmass/seeds are
  legitimate provided features (not derived from yield after the fact).
- Temp-range collinearity could destabilize linear models but tree models (LGB/XGB/CatBoost) are robust to it.

Feature Engineering Ideas:
- Interaction/product terms among fruitset, fruitmass, seeds (e.g. fruitset*seeds, fruitmass*seeds).
- Condensed temperature feature: upper-lower range spread and average-of-averages, to de-duplicate the 6
  collinear temp columns.
- total_pollinators = honeybee+bumbles+andrena+osmia (and ratio of honeybee to total).
- rain_intensity = RainingDays * AverageRainingDays.
- clonesize as an ordinal/log feature (low cardinality, monotonic negative relationship).

Recommended Validation Strategy:
- Standard 5-fold KFold (shuffle, seed=42). Target is continuous with no groups/time structure and
  train/test distributions match closely, so plain i.i.d. KFold is appropriate (no stratification needed
  since yield is not a small set of classes, unlike s3e16's discrete Age).
""")
