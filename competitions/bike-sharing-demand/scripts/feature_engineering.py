"""Feature engineering for Bike Sharing Demand."""
import pandas as pd
import numpy as np

data_dir = "competitions/bike-sharing-demand/data"

train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["datetime"])
test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["datetime"])

print(f"Raw train shape: {train.shape}")
print(f"Raw test shape: {test.shape}")

# Preserve target and ID
train_datetime = train["datetime"]
test_datetime = test["datetime"]
target = train["count"]
log_target = np.log1p(target)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering consistently to train and test."""
    out = pd.DataFrame()

    # --- Datetime features ---
    dt = df["datetime"]
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    out["day"] = dt.dt.day
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # --- Original features (keep useful ones) ---
    out["season"] = df["season"]
    out["holiday"] = df["holiday"]
    out["workingday"] = df["workingday"]
    out["weather"] = df["weather"].replace(4, 3)  # Merge weather 4 into 3
    out["temp"] = df["temp"]
    out["humidity"] = df["humidity"]
    out["windspeed"] = df["windspeed"]

    # --- Engineered features ---
    # Hour x workingday interaction (encodes commute vs leisure patterns)
    out["hour_workingday"] = out["hour"] * 100 + out["workingday"]

    # Rush hour flag: peak commute hours on workdays
    out["rush_hour"] = (
        (out["workingday"] == 1)
        & (out["hour"].isin([7, 8, 9, 16, 17, 18]))
    ).astype(int)

    # Windspeed zero flag (likely missing data)
    out["windspeed_zero"] = (df["windspeed"] == 0).astype(int)

    return out


# Apply to both sets
X_train = engineer_features(train)
X_test = engineer_features(test)

print(f"\nEngineered train shape: {X_train.shape}")
print(f"Engineered test shape: {X_test.shape}")
print(f"\nFeatures ({X_train.shape[1]}):")
for col in X_train.columns:
    print(f"  {col}: dtype={X_train[col].dtype}, range=[{X_train[col].min()}, {X_train[col].max()}]")

# --- Validation ---
print("\n=== VALIDATION ===")
# Shape check
assert X_train.shape[1] == X_test.shape[1], "Column mismatch!"
print(f"Column count match: {X_train.shape[1]} == {X_test.shape[1]}")

# NaN check
train_nans = X_train.isnull().sum().sum()
test_nans = X_test.isnull().sum().sum()
print(f"NaN in train: {train_nans}, test: {test_nans}")

# Column match
assert list(X_train.columns) == list(X_test.columns), "Column order mismatch!"
print("Column order match: OK")

# --- Save processed data ---
X_train["log_count"] = log_target.values
X_train["count"] = target.values
X_train["datetime"] = train_datetime.values

X_test["datetime"] = test_datetime.values

X_train.to_parquet(f"{data_dir}/train_processed.parquet", index=False)
X_test.to_parquet(f"{data_dir}/test_processed.parquet", index=False)
print(f"\nSaved train_processed.parquet ({X_train.shape})")
print(f"Saved test_processed.parquet ({X_test.shape})")

# --- Quick Feature Importance ---
print("\n=== QUICK FEATURE IMPORTANCE (LightGBM) ===")
import lightgbm as lgb

feature_cols = [c for c in X_train.columns if c not in ["datetime", "count", "log_count"]]
dtrain = lgb.Dataset(X_train[feature_cols], label=log_target)

params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": 42,
}
model = lgb.train(params, dtrain, num_boost_round=300)

importance = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols)
importance = importance.sort_values(ascending=False)

print("\nFeature importance (gain):")
for feat, imp in importance.items():
    bar = "#" * int(imp / importance.max() * 40)
    print(f"  {feat:20s}: {imp:10.1f} {bar}")

zero_imp = importance[importance == 0]
if len(zero_imp) > 0:
    print(f"\nZero importance features: {list(zero_imp.index)}")
else:
    print("\nNo zero-importance features.")
