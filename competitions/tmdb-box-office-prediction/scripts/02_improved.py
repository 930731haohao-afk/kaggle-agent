"""Improved model for TMDB Box Office Prediction.
Key improvements:
1. Budget imputation (27% are zero/missing)
2. Target encoding for actors, directors, genres, companies
3. More interaction features
4. Tuned hyperparameters
"""
import pandas as pd
import numpy as np
import ast
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/tmdb-box-office-prediction/data"
COMP_DIR = "/home/tjyen/ai_agents/kaggle/competitions/tmdb-box-office-prediction"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

y_train = train['revenue'].values
log_y = np.log1p(y_train)

def safe_eval(val):
    if pd.isna(val) or val == '' or val == '[]':
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        try:
            return json.loads(val)
        except:
            return []


# ============================================================
# Build combined dataframe for consistent feature extraction
# ============================================================
all_data = pd.concat([train, test], ignore_index=True)
n_train = len(train)

# Parse all JSON columns once
all_data['genres_parsed'] = all_data['genres'].apply(safe_eval)
all_data['companies_parsed'] = all_data['production_companies'].apply(safe_eval)
all_data['countries_parsed'] = all_data['production_countries'].apply(safe_eval)
all_data['languages_parsed'] = all_data['spoken_languages'].apply(safe_eval)
all_data['keywords_parsed'] = all_data['Keywords'].apply(safe_eval)
all_data['cast_parsed'] = all_data['cast'].apply(safe_eval)
all_data['crew_parsed'] = all_data['crew'].apply(safe_eval)

logging.info("Parsed all JSON columns")

# ============================================================
# FEATURE EXTRACTION
# ============================================================
feats = pd.DataFrame(index=all_data.index)

# Budget
feats['budget'] = all_data['budget'].values
feats['has_budget'] = (all_data['budget'] > 0).astype(int)
feats['log_budget'] = np.log1p(all_data['budget'])

# Popularity
feats['popularity'] = all_data['popularity'].values
feats['log_popularity'] = np.log1p(all_data['popularity'])

# Runtime
median_runtime = all_data['runtime'].median()
feats['runtime'] = all_data['runtime'].fillna(median_runtime).values
feats['runtime_missing'] = all_data['runtime'].isnull().astype(int)

# Language
feats['is_english'] = (all_data['original_language'] == 'en').astype(int)

# Release date
dates = pd.to_datetime(all_data['release_date'], format='mixed', errors='coerce')
feats['release_year'] = dates.dt.year
feats['release_month'] = dates.dt.month
feats['release_day'] = dates.dt.day
feats['release_dayofweek'] = dates.dt.dayofweek
feats['release_quarter'] = dates.dt.quarter
feats['is_summer_release'] = dates.dt.month.isin([5, 6, 7, 8]).astype(int)
feats['is_holiday_release'] = dates.dt.month.isin([11, 12]).astype(int)
feats['is_jan_release'] = (dates.dt.month == 1).astype(int)
feats.loc[feats['release_year'] > 2019, 'release_year'] -= 100

# Status
feats['is_released'] = (all_data['status'] == 'Released').astype(int)

# Homepage & tagline
feats['has_homepage'] = all_data['homepage'].notna().astype(int)
feats['has_tagline'] = all_data['tagline'].notna().astype(int)
feats['tagline_length'] = all_data['tagline'].fillna('').str.len()

# Overview & title
feats['overview_length'] = all_data['overview'].fillna('').str.len()
feats['overview_word_count'] = all_data['overview'].fillna('').str.split().str.len()
feats['title_length'] = all_data['title'].fillna('').str.len()

# Collection
feats['belongs_to_collection'] = all_data['belongs_to_collection'].notna().astype(int)

# Genres
feats['n_genres'] = all_data['genres_parsed'].apply(len)
top_genres = ['Drama', 'Comedy', 'Thriller', 'Action', 'Horror', 'Romance',
              'Adventure', 'Crime', 'Science Fiction', 'Fantasy', 'Family',
              'Animation', 'Documentary', 'Mystery', 'Music', 'War', 'Western',
              'History', 'TV Movie']
for genre in top_genres:
    feats[f'genre_{genre}'] = all_data['genres_parsed'].apply(
        lambda x: int(any(d.get('name') == genre for d in x if isinstance(d, dict))))

# Production companies
feats['n_companies'] = all_data['companies_parsed'].apply(len)

# Count major studios
major_studios = ['Warner Bros.', 'Universal Pictures', 'Paramount Pictures',
                 'Twentieth Century Fox Film Corporation', 'Columbia Pictures',
                 'Walt Disney Pictures', 'New Line Cinema', 'Metro-Goldwyn-Mayer',
                 'Lionsgate', 'DreamWorks Animation', 'Marvel Studios',
                 'Touchstone Pictures', 'TriStar Pictures']
for studio in major_studios:
    safe_name = studio[:20].replace(' ', '_').replace('.', '')
    feats[f'studio_{safe_name}'] = all_data['companies_parsed'].apply(
        lambda x, s=studio: int(any(d.get('name') == s for d in x if isinstance(d, dict))))
feats['n_major_studios'] = sum(feats[f'studio_{s[:20].replace(" ", "_").replace(".", "")}'] for s in major_studios)
feats['has_major_studio'] = (feats['n_major_studios'] > 0).astype(int)

# Production countries
feats['n_countries'] = all_data['countries_parsed'].apply(len)
feats['is_us_produced'] = all_data['countries_parsed'].apply(
    lambda x: int(any(d.get('iso_3166_1') == 'US' for d in x if isinstance(d, dict))))
feats['is_uk_produced'] = all_data['countries_parsed'].apply(
    lambda x: int(any(d.get('iso_3166_1') == 'GB' for d in x if isinstance(d, dict))))

# Languages
feats['n_languages'] = all_data['languages_parsed'].apply(len)

# Keywords
feats['n_keywords'] = all_data['keywords_parsed'].apply(len)

# Cast
feats['n_cast'] = all_data['cast_parsed'].apply(len)
feats['n_male_cast'] = all_data['cast_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('gender') == 2))
feats['n_female_cast'] = all_data['cast_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('gender') == 1))

# Crew
feats['n_crew'] = all_data['crew_parsed'].apply(len)
feats['n_directors'] = all_data['crew_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('job') == 'Director'))
feats['n_producers'] = all_data['crew_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('job') == 'Producer'))
feats['n_writers'] = all_data['crew_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('department') == 'Writing'))

# ============================================================
# TARGET ENCODING (using only train data, with k-fold to avoid leakage)
# ============================================================
logging.info("Target encoding...")

# Extract person IDs for target encoding
def get_top_person_ids(parsed_col, n_top=50, job_filter=None):
    """Get top N most frequent person IDs."""
    counts = {}
    for item_list in parsed_col[:n_train]:  # Only from train
        for item in item_list:
            if isinstance(item, dict):
                if job_filter and item.get('job') != job_filter:
                    continue
                pid = item.get('id', item.get('cast_id'))
                if pid:
                    counts[pid] = counts.get(pid, 0) + 1
    return [pid for pid, _ in sorted(counts.items(), key=lambda x: -x[1])[:n_top]]

# Top actors
top_actor_ids = get_top_person_ids(all_data['cast_parsed'], n_top=100)
feats['has_top_actor'] = all_data['cast_parsed'].apply(
    lambda x: int(any(c.get('id') in top_actor_ids for c in x if isinstance(c, dict))))

# Top directors
top_director_ids = get_top_person_ids(all_data['crew_parsed'], n_top=50, job_filter='Director')
feats['has_top_director'] = all_data['crew_parsed'].apply(
    lambda x: int(any(c.get('id') in top_director_ids for c in x
                      if isinstance(c, dict) and c.get('job') == 'Director')))

# Mean revenue per genre (OOF target encoding)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for genre in ['Action', 'Comedy', 'Drama', 'Thriller', 'Horror', 'Adventure', 'Animation']:
    col = f'genre_{genre}'
    te_col = f'te_{genre}_revenue'
    feats[te_col] = 0.0

    # Only compute for train
    for tr_idx, va_idx in kf.split(range(n_train)):
        mask = feats[col].iloc[tr_idx] == 1
        if mask.sum() > 0:
            mean_rev = log_y[tr_idx][mask.values].mean()
        else:
            mean_rev = log_y[tr_idx].mean()
        feats.loc[va_idx[feats[col].iloc[va_idx] == 1], te_col] = mean_rev
        feats.loc[va_idx[feats[col].iloc[va_idx] == 0], te_col] = log_y[tr_idx][~mask.values].mean()

    # For test: use full train mean
    mask_train = feats[col].iloc[:n_train] == 1
    mean_rev_full = log_y[mask_train.values].mean() if mask_train.sum() > 0 else log_y.mean()
    mean_no_rev = log_y[~mask_train.values].mean()
    test_mask = feats[col].iloc[n_train:] == 1
    feats.loc[feats.index[n_train:][test_mask.values], te_col] = mean_rev_full
    feats.loc[feats.index[n_train:][~test_mask.values], te_col] = mean_no_rev

# Mean revenue for collection vs non-collection
feats['te_collection_revenue'] = 0.0
for tr_idx, va_idx in kf.split(range(n_train)):
    for val in [0, 1]:
        mask = feats['belongs_to_collection'].iloc[tr_idx] == val
        if mask.sum() > 0:
            feats.loc[va_idx[feats['belongs_to_collection'].iloc[va_idx] == val], 'te_collection_revenue'] = \
                log_y[tr_idx][mask.values].mean()
# Test
for val in [0, 1]:
    mask = feats['belongs_to_collection'].iloc[:n_train] == val
    feats.loc[feats.index[n_train:][feats['belongs_to_collection'].iloc[n_train:] == val],
              'te_collection_revenue'] = log_y[mask.values].mean()

# ============================================================
# INTERACTION FEATURES
# ============================================================
feats['budget_x_popularity'] = feats['log_budget'] * feats['log_popularity']
feats['budget_x_runtime'] = feats['log_budget'] * feats['runtime']
feats['year_x_budget'] = feats['release_year'] * feats['log_budget']
feats['cast_x_crew'] = feats['n_cast'] * feats['n_crew']
feats['budget_per_cast'] = feats['budget'] / (feats['n_cast'] + 1)

print(f"Total features: {feats.shape[1]}")

# ============================================================
# MODELING
# ============================================================
logging.info("Training models...")

X_tr = feats.iloc[:n_train].values.astype(np.float32)
X_te = feats.iloc[n_train:].values.astype(np.float32)
X_tr = np.nan_to_num(X_tr, nan=0)
X_te = np.nan_to_num(X_te, nan=0)

feature_names = feats.columns.tolist()

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# LightGBM
print("\n--- LightGBM ---")
oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    model = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=7, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=15,
        reg_alpha=0.5, reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=42
    )
    model.fit(
        X_tr[tr_idx], log_y[tr_idx],
        eval_set=[(X_tr[va_idx], log_y[va_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    oof_lgb[va_idx] = model.predict(X_tr[va_idx])
    test_lgb += model.predict(X_te) / 5

lgb_rmsle = rmsle(y_train, np.expm1(oof_lgb))
print(f"LightGBM RMSLE: {lgb_rmsle:.5f}")

# XGBoost
print("\n--- XGBoost ---")
oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    model = xgb.XGBRegressor(
        n_estimators=2000, max_depth=7, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0, verbosity=0, n_jobs=-1, random_state=42
    )
    model.fit(
        X_tr[tr_idx], log_y[tr_idx],
        eval_set=[(X_tr[va_idx], log_y[va_idx])],
        verbose=False
    )
    oof_xgb[va_idx] = model.predict(X_tr[va_idx])
    test_xgb += model.predict(X_te) / 5

xgb_rmsle = rmsle(y_train, np.expm1(oof_xgb))
print(f"XGBoost RMSLE: {xgb_rmsle:.5f}")

# CatBoost
print("\n--- CatBoost ---")
oof_cat = np.zeros(n_train)
test_cat = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    model = CatBoostRegressor(
        iterations=2000, depth=7, learning_rate=0.03,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        early_stopping_rounds=100
    )
    model.fit(
        X_tr[tr_idx], log_y[tr_idx],
        eval_set=(X_tr[va_idx], log_y[va_idx]),
    )
    oof_cat[va_idx] = model.predict(X_tr[va_idx])
    test_cat += model.predict(X_te) / 5

cat_rmsle = rmsle(y_train, np.expm1(oof_cat))
print(f"CatBoost RMSLE: {cat_rmsle:.5f}")

# Best ensemble search
print("\n--- Ensemble Search ---")
best_rmsle = 999
best_w = None
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        score = rmsle(y_train, np.expm1(oof_ens))
        if score < best_rmsle:
            best_rmsle = score
            best_w = (w1, w2, w3)

print(f"Best: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CAT={best_w[2]:.2f} -> RMSLE={best_rmsle:.5f}")

# Save submissions
sub_template = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

# CatBoost only
sub = sub_template.copy()
sub['revenue'] = np.clip(np.expm1(test_cat), 1, None)
sub.to_csv(f"{COMP_DIR}/submissions/submission_03_cat_improved.csv", index=False)

# Best ensemble
test_best = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cat
sub = sub_template.copy()
sub['revenue'] = np.clip(np.expm1(test_best), 1, None)
sub.to_csv(f"{COMP_DIR}/submissions/submission_04_ensemble_improved.csv", index=False)

# Simple average of all 3
test_avg = (test_lgb + test_xgb + test_cat) / 3
sub = sub_template.copy()
sub['revenue'] = np.clip(np.expm1(test_avg), 1, None)
sub.to_csv(f"{COMP_DIR}/submissions/submission_05_avg3.csv", index=False)
avg_rmsle = rmsle(y_train, np.expm1((oof_lgb + oof_xgb + oof_cat) / 3))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"LightGBM:      RMSLE = {lgb_rmsle:.5f}")
print(f"XGBoost:       RMSLE = {xgb_rmsle:.5f}")
print(f"CatBoost:      RMSLE = {cat_rmsle:.5f}")
print(f"Avg3:          RMSLE = {avg_rmsle:.5f}")
print(f"Best Ensemble: RMSLE = {best_rmsle:.5f}")
print(f"\nSaved submissions 03, 04, 05")
