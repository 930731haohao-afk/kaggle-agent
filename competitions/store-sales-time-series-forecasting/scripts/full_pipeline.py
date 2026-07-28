"""
Full Pipeline for Store Sales - Time Series Forecasting
=========================================================
Regression (RMSLE), 1782 series (54 stores x 33 families).
Train: 2013-2017-08-15 (3M rows), Test: 2017-08-16 to 2017-08-31 (16 days).
Uses supplementary: stores, oil, holidays, transactions.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

COMPETITION_DIR = "competitions/store-sales-time-series-forecasting"
data_dir = os.path.join(COMPETITION_DIR, "data")
TARGET = "sales"
ID = "id"
SEED = 42

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=['date'])
test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=['date'])
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
oil = pd.read_csv(os.path.join(data_dir, "oil.csv"), parse_dates=['date'])
holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"), parse_dates=['date'])
transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), parse_dates=['date'])

test_ids = test[ID]
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Train dates: {train['date'].min().date()} to {train['date'].max().date()}")
print(f"Test dates:  {test['date'].min().date()} to {test['date'].max().date()}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))


# ============================================================
# EDA HIGHLIGHTS
# ============================================================
section("EDA HIGHLIGHTS")

print(f"Stores: {train['store_nbr'].nunique()}, Families: {train['family'].nunique()}")
print(f"Series: {train['store_nbr'].nunique()} x {train['family'].nunique()} = {train['store_nbr'].nunique() * train['family'].nunique()}")
print(f"Target: mean={train[TARGET].mean():.1f}, median={train[TARGET].median():.1f}, zeros={( train[TARGET]==0).mean()*100:.1f}%")

# Oil price info
print(f"\nOil prices: {oil['dcoilwtico'].describe().to_dict()}")

# Top families by sales
top_fam = train.groupby('family')[TARGET].mean().sort_values(ascending=False).head(10)
print(f"\nTop 10 families by avg sales:\n{top_fam.to_string()}")

# Recent trend (use last year)
recent = train[train['date'] >= '2017-01-01']
monthly = recent.groupby(recent['date'].dt.month)[TARGET].mean()
print(f"\n2017 monthly avg sales:\n{monthly.to_string()}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
section("FEATURE ENGINEERING")

# Use only recent training data for efficiency (last ~2 years)
# But keep enough for lag features
train = train[train['date'] >= '2015-01-01'].copy()
print(f"Using train from 2015+: {train.shape}")

n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)

# 1. Merge store info
combined = combined.merge(stores, on='store_nbr', how='left')

# 2. Oil prices (forward-fill missing, then merge)
oil = oil.set_index('date').resample('D').last().reset_index()
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
combined = combined.merge(oil, on='date', how='left')
combined['dcoilwtico'] = combined['dcoilwtico'].ffill().bfill()

# 3. Holidays processing
# National holidays
nat_holidays = holidays[
    (holidays['locale'] == 'National') &
    (holidays['transferred'] == False) &
    (holidays['type'] != 'Work Day')
]['date'].unique()
combined['is_national_holiday'] = combined['date'].isin(nat_holidays).astype(int)

# Any holiday/event
all_holiday_dates = holidays[holidays['type'] != 'Work Day']['date'].unique()
combined['is_any_holiday'] = combined['date'].isin(all_holiday_dates).astype(int)

# 4. Date features
combined['year'] = combined['date'].dt.year
combined['month'] = combined['date'].dt.month
combined['day'] = combined['date'].dt.day
combined['dayofweek'] = combined['date'].dt.dayofweek
combined['dayofyear'] = combined['date'].dt.dayofyear
combined['weekofyear'] = combined['date'].dt.isocalendar().week.astype(int)
combined['is_weekend'] = (combined['dayofweek'] >= 5).astype(int)
combined['quarter'] = combined['date'].dt.quarter
combined['is_month_start'] = combined['date'].dt.is_month_start.astype(int)
combined['is_month_end'] = combined['date'].dt.is_month_end.astype(int)

# 5. Cyclical date features
combined['month_sin'] = np.sin(2 * np.pi * combined['month'] / 12)
combined['month_cos'] = np.cos(2 * np.pi * combined['month'] / 12)
combined['dow_sin'] = np.sin(2 * np.pi * combined['dayofweek'] / 7)
combined['dow_cos'] = np.cos(2 * np.pi * combined['dayofweek'] / 7)

# 6. Days since start (trend)
min_date = combined['date'].min()
combined['days_since_start'] = (combined['date'] - min_date).dt.days

# 7. Encode categoricals
le_family = LabelEncoder()
combined['family_enc'] = le_family.fit_transform(combined['family'])
le_city = LabelEncoder()
combined['city_enc'] = le_city.fit_transform(combined['city'])
le_state = LabelEncoder()
combined['state_enc'] = le_state.fit_transform(combined['state'])
le_type = LabelEncoder()
combined['type_enc'] = le_type.fit_transform(combined['type'])

# 8. Lag-based features (computed from train portion only)
# Per store-family: historical averages
train_part = combined.iloc[:n_train]

# Monthly mean sales per store-family
series_month = train_part.groupby(['store_nbr', 'family_enc', 'month'])[TARGET].mean().reset_index()
series_month.columns = ['store_nbr', 'family_enc', 'month', 'sf_month_mean']
combined = combined.merge(series_month, on=['store_nbr', 'family_enc', 'month'], how='left')

# DOW mean per store-family
series_dow = train_part.groupby(['store_nbr', 'family_enc', 'dayofweek'])[TARGET].mean().reset_index()
series_dow.columns = ['store_nbr', 'family_enc', 'dayofweek', 'sf_dow_mean']
combined = combined.merge(series_dow, on=['store_nbr', 'family_enc', 'dayofweek'], how='left')

# Overall store-family mean
series_mean = train_part.groupby(['store_nbr', 'family_enc'])[TARGET].mean().reset_index()
series_mean.columns = ['store_nbr', 'family_enc', 'sf_mean']
combined = combined.merge(series_mean, on=['store_nbr', 'family_enc'], how='left')

# Store-level mean
store_mean = train_part.groupby('store_nbr')[TARGET].mean().reset_index()
store_mean.columns = ['store_nbr', 'store_mean']
combined = combined.merge(store_mean, on='store_nbr', how='left')

# Family-level mean
family_mean = train_part.groupby('family_enc')[TARGET].mean().reset_index()
family_mean.columns = ['family_enc', 'family_mean']
combined = combined.merge(family_mean, on='family_enc', how='left')

# 9. Recent lag sales (last 7, 14, 28 days averages per store-family)
# Compute from the last available training data
last_train_date = train_part['date'].max()
for lag_days in [7, 14, 28]:
    cutoff = last_train_date - pd.Timedelta(days=lag_days)
    recent_data = train_part[train_part['date'] > cutoff]
    lag_mean = recent_data.groupby(['store_nbr', 'family_enc'])[TARGET].mean().reset_index()
    lag_mean.columns = ['store_nbr', 'family_enc', f'sf_lag{lag_days}_mean']
    combined = combined.merge(lag_mean, on=['store_nbr', 'family_enc'], how='left')

# Drop non-feature columns
drop_cols = [TARGET, ID, 'date', 'family', 'city', 'state', 'type']
feature_cols = [c for c in combined.columns if c not in drop_cols]

print(f"Features: {len(feature_cols)}")

# Split back
X = combined.iloc[:n_train][feature_cols].copy()
y = combined.iloc[:n_train][TARGET].copy()
X_test = combined.iloc[n_train:][feature_cols].copy()

print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"NaN in train: {X.isnull().sum().sum()}, NaN in test: {X_test.isnull().sum().sum()}")

# Fill any remaining NaN
for col in X.columns:
    if X[col].isnull().sum() > 0:
        med = X[col].median()
        X[col] = X[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

print(f"After fillna - NaN in train: {X.isnull().sum().sum()}, NaN in test: {X_test.isnull().sum().sum()}")


# ============================================================
# MODELING
# ============================================================
section("MODELING")

# Use log1p target for RMSLE optimization
y_log = np.log1p(y)


def log_experiment(model_name, params, scores, notes=""):
    exp_file = os.path.join(COMPETITION_DIR, "experiments.json")
    with open(exp_file, "r") as f:
        experiments = json.load(f)
    exp_id = len(experiments) + 1
    experiment = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "n_features": X.shape[1],
        "params": {k: str(v) for k, v in params.items()},
        "cv_strategy": "time-based-split",
        "cv_scores": [round(s, 6) for s in scores],
        "cv_mean": round(float(np.mean(scores)), 6),
        "cv_std": round(float(np.std(scores)), 6),
        "eval_metric": "RMSLE",
        "notes": notes
    }
    experiments.append(experiment)
    with open(exp_file, "w") as f:
        json.dump(experiments, f, indent=2)
    return exp_id


def cv_model_ts(model_fn, model_name, params, notes=""):
    """Time-series CV: split by date boundaries."""
    train_dates = combined.iloc[:n_train]['date']

    # 3 splits: train→2016, val=2016H2 | train→2017-04, val=2017-04-to-06 | train→2017-06, val=2017-06-to-08
    splits = [
        ('2016-01-01', '2016-07-01', '2016-12-31'),
        ('2016-07-01', '2017-01-01', '2017-04-30'),
        ('2017-01-01', '2017-05-01', '2017-08-15'),
    ]

    scores = []
    test_preds_all = np.zeros(len(X_test))
    n_models = 0

    for split_start, val_start, val_end in splits:
        train_mask = (train_dates >= split_start) & (train_dates < val_start)
        val_mask = (train_dates >= val_start) & (train_dates <= val_end)

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        X_tr = X[train_mask]
        y_tr = y_log[train_mask]
        X_val = X[val_mask]
        y_val_orig = y[val_mask]

        model = model_fn(params)
        model.fit(X_tr, y_tr)

        # Predict in log space, convert back
        val_preds_log = model.predict(X_val)
        val_preds = np.expm1(np.maximum(val_preds_log, 0))
        val_preds = np.maximum(val_preds, 0)
        score = rmsle(y_val_orig.values, val_preds)
        scores.append(score)

        test_preds_log = model.predict(X_test)
        test_preds = np.expm1(np.maximum(test_preds_log, 0))
        test_preds = np.maximum(test_preds, 0)
        test_preds_all += test_preds
        n_models += 1
        print(f"  Split train<{val_start} val=[{val_start},{val_end}]: RMSLE={score:.6f}")

    test_preds_all /= n_models
    mean_score = np.mean(scores)
    exp_id = log_experiment(model_name, params, scores, notes)
    print(f"  Mean: {mean_score:.6f} (+/- {np.std(scores):.6f}) [Experiment #{exp_id}]\n")
    return mean_score, test_preds_all


# Also train a final model on ALL training data for submission
def train_final_model(model_fn, params):
    model = model_fn(params)
    model.fit(X, y_log)
    preds_log = model.predict(X_test)
    preds = np.expm1(np.maximum(preds_log, 0))
    return np.maximum(preds, 0)


# MODEL 1: LightGBM
print("--- LightGBM ---")
lgbm_params = {
    "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63,
    "max_depth": -1, "min_child_samples": 50, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1
}
lgbm_score, lgbm_test = cv_model_ts(
    lambda p: lgb.LGBMRegressor(verbosity=-1, random_state=SEED, n_jobs=-1, **p),
    "LightGBM", lgbm_params, "LightGBM with time features and store-family aggregates"
)

# MODEL 2: XGBoost
print("--- XGBoost ---")
xgb_params = {
    "n_estimators": 800, "learning_rate": 0.05, "max_depth": 8,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0
}
xgb_score, xgb_test = cv_model_ts(
    lambda p: xgb.XGBRegressor(**p, random_state=SEED, verbosity=0, n_jobs=-1),
    "XGBoost", xgb_params, "XGBoost with time features"
)

# MODEL 3: LightGBM (trained on all data)
print("--- LightGBM Full ---")
lgbm_full_test = train_final_model(
    lambda p: lgb.LGBMRegressor(verbosity=-1, random_state=SEED, n_jobs=-1, **p),
    lgbm_params
)
print(f"  LightGBM trained on full data. Pred range: [{lgbm_full_test.min():.1f}, {lgbm_full_test.max():.1f}]\n")


# ============================================================
# ENSEMBLE
# ============================================================
section("ENSEMBLE")

# Weighted average: give more weight to full-data model + LightGBM CV
ensemble_test = 0.3 * lgbm_test + 0.2 * xgb_test + 0.5 * lgbm_full_test
ensemble_test = np.maximum(ensemble_test, 0)

print(f"Ensemble prediction stats:")
print(f"  Min: {ensemble_test.min():.1f}, Max: {ensemble_test.max():.1f}")
print(f"  Mean: {ensemble_test.mean():.1f}, Median: {np.median(ensemble_test):.1f}")
print(f"  Zeros: {(ensemble_test == 0).sum()} ({(ensemble_test == 0).mean()*100:.1f}%)")


# ============================================================
# LEADERBOARD
# ============================================================
section("EXPERIMENT LEADERBOARD")

with open(os.path.join(COMPETITION_DIR, "experiments.json"), "r") as f:
    experiments = json.load(f)

sorted_exps = sorted(experiments, key=lambda e: e["cv_mean"])
print(f"{'#':<4} {'ID':<5} {'Model':<25} {'CV RMSLE':<12} {'CV Std':<10}")
print("-" * 58)
for rank, exp in enumerate(sorted_exps, 1):
    print(f"{rank:<4} {exp['experiment_id']:<5} {exp['model'][:24]:<25} "
          f"{exp['cv_mean']:<12.6f} {exp['cv_std']:<10.6f}")

best = sorted_exps[0]
print(f"\nBest: {best['model']} with RMSLE={best['cv_mean']:.6f}")


# ============================================================
# SUBMISSION
# ============================================================
section("SUBMISSION")

submission = pd.DataFrame({
    "id": test_ids.values,
    "sales": ensemble_test
})

# Validation
shape_ok = submission.shape == sample_sub.shape
cols_ok = list(submission.columns) == list(sample_sub.columns)
nan_count = submission.isnull().sum().sum()
id_match = (submission["id"].values == sample_sub["id"].values).all()
values_ok = (submission["sales"] >= 0).all()

print(f"  Shape: {'PASS' if shape_ok else 'FAIL'} ({submission.shape})")
print(f"  Columns: {'PASS' if cols_ok else 'FAIL'}")
print(f"  NaN: {'PASS' if nan_count == 0 else 'FAIL'}")
print(f"  IDs: {'PASS' if id_match else 'FAIL'}")
print(f"  Values >= 0: {'PASS' if values_ok else 'FAIL'}")

all_pass = shape_ok and cols_ok and nan_count == 0 and id_match and values_ok

if all_pass:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_cv = best['cv_mean']
    filename = f"submission_ensemble_{best_cv:.4f}_{timestamp}.csv"
    output_dir = os.path.join(COMPETITION_DIR, "submissions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")
else:
    print("\nValidation FAILED.")
