"""Improved model with budget imputation and more features.
Key: 27% of movies have budget=0, which is really missing data.
Strategy: Predict budget from other features, then use imputed budget.
Also: add more person-level features, interaction terms.
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


all_data = pd.concat([train, test], ignore_index=True)
n_train = len(train)

# Parse JSON columns
all_data['genres_parsed'] = all_data['genres'].apply(safe_eval)
all_data['companies_parsed'] = all_data['production_companies'].apply(safe_eval)
all_data['countries_parsed'] = all_data['production_countries'].apply(safe_eval)
all_data['languages_parsed'] = all_data['spoken_languages'].apply(safe_eval)
all_data['keywords_parsed'] = all_data['Keywords'].apply(safe_eval)
all_data['cast_parsed'] = all_data['cast'].apply(safe_eval)
all_data['crew_parsed'] = all_data['crew'].apply(safe_eval)

logging.info("Parsed all JSON columns")

# ============================================================
# FEATURE EXTRACTION (no budget features yet)
# ============================================================
feats = pd.DataFrame(index=all_data.index)

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
genre_names = ['Drama', 'Comedy', 'Thriller', 'Action', 'Horror', 'Romance',
               'Adventure', 'Crime', 'Science Fiction', 'Fantasy', 'Family',
               'Animation', 'Documentary', 'Mystery', 'Music', 'War', 'Western',
               'History', 'TV Movie']
for genre in genre_names:
    feats[f'genre_{genre}'] = all_data['genres_parsed'].apply(
        lambda x: int(any(d.get('name') == genre for d in x if isinstance(d, dict))))

# Production companies
feats['n_companies'] = all_data['companies_parsed'].apply(len)
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

# Countries
feats['n_countries'] = all_data['countries_parsed'].apply(len)
feats['is_us_produced'] = all_data['countries_parsed'].apply(
    lambda x: int(any(d.get('iso_3166_1') == 'US' for d in x if isinstance(d, dict))))

# Languages
feats['n_languages'] = all_data['languages_parsed'].apply(len)

# Keywords
feats['n_keywords'] = all_data['keywords_parsed'].apply(len)

# Cast & crew
feats['n_cast'] = all_data['cast_parsed'].apply(len)
feats['n_male_cast'] = all_data['cast_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('gender') == 2))
feats['n_female_cast'] = all_data['cast_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('gender') == 1))
feats['n_crew'] = all_data['crew_parsed'].apply(len)
feats['n_directors'] = all_data['crew_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('job') == 'Director'))
feats['n_producers'] = all_data['crew_parsed'].apply(
    lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('job') == 'Producer'))

# Top actors/directors
top_actor_ids = set()
actor_counts = {}
for cast_list in all_data['cast_parsed'][:n_train]:
    for c in cast_list:
        if isinstance(c, dict):
            pid = c.get('id')
            if pid:
                actor_counts[pid] = actor_counts.get(pid, 0) + 1
top_actor_ids = set(pid for pid, cnt in sorted(actor_counts.items(), key=lambda x: -x[1])[:100])
feats['has_top100_actor'] = all_data['cast_parsed'].apply(
    lambda x: int(any(c.get('id') in top_actor_ids for c in x if isinstance(c, dict))))

director_counts = {}
for crew_list in all_data['crew_parsed'][:n_train]:
    for c in crew_list:
        if isinstance(c, dict) and c.get('job') == 'Director':
            pid = c.get('id')
            if pid:
                director_counts[pid] = director_counts.get(pid, 0) + 1
top_director_ids = set(pid for pid, cnt in sorted(director_counts.items(), key=lambda x: -x[1])[:50])
feats['has_top50_director'] = all_data['crew_parsed'].apply(
    lambda x: int(any(c.get('id') in top_director_ids for c in x
                      if isinstance(c, dict) and c.get('job') == 'Director')))

# ============================================================
# BUDGET IMPUTATION
# ============================================================
logging.info("Budget imputation...")

original_budget = all_data['budget'].values.copy()
has_budget = original_budget > 0

# Use non-budget features to predict budget
budget_features = feats.values.astype(np.float32)
budget_features = np.nan_to_num(budget_features, nan=0)

# Train budget model on movies with known budget
train_has_budget = has_budget[:n_train]
if train_has_budget.sum() > 100:
    log_budget = np.log1p(original_budget[:n_train])
    budget_model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1
    )
    budget_model.fit(
        budget_features[:n_train][train_has_budget],
        log_budget[train_has_budget]
    )
    predicted_budget_all = budget_model.predict(budget_features)
    imputed_budget = original_budget.copy().astype(float)
    imputed_budget[~has_budget] = np.expm1(predicted_budget_all[~has_budget])
    imputed_budget = np.clip(imputed_budget, 0, None)

    n_imputed_train = (~has_budget[:n_train]).sum()
    n_imputed_test = (~has_budget[n_train:]).sum()
    print(f"Imputed budget for {n_imputed_train} train + {n_imputed_test} test movies")
    print(f"Mean imputed budget: {imputed_budget[~has_budget].mean():,.0f}")
else:
    imputed_budget = original_budget.astype(float)

# Add budget features
feats['budget_original'] = original_budget
feats['has_budget'] = has_budget.astype(int)
feats['log_budget_original'] = np.log1p(original_budget)
feats['budget_imputed'] = imputed_budget
feats['log_budget_imputed'] = np.log1p(imputed_budget)

# Interactions with imputed budget
feats['budget_x_popularity'] = feats['log_budget_imputed'] * feats['log_popularity']
feats['budget_x_runtime'] = feats['log_budget_imputed'] * feats['runtime']
feats['budget_per_cast'] = imputed_budget / (feats['n_cast'] + 1)

print(f"\nTotal features: {feats.shape[1]}")

# ============================================================
# MODELING
# ============================================================
logging.info("Training models...")

X_tr = feats.iloc[:n_train].values.astype(np.float32)
X_te = feats.iloc[n_train:].values.astype(np.float32)
X_tr = np.nan_to_num(X_tr, nan=0)
X_te = np.nan_to_num(X_te, nan=0)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# LightGBM
oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    m = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=7, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=15,
        reg_alpha=0.5, reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=42
    )
    m.fit(X_tr[tr_idx], log_y[tr_idx],
          eval_set=[(X_tr[va_idx], log_y[va_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[va_idx] = m.predict(X_tr[va_idx])
    test_lgb += m.predict(X_te) / 5
lgb_score = rmsle(y_train, np.expm1(oof_lgb))
print(f"LightGBM: {lgb_score:.5f}")

# XGBoost
oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    m = xgb.XGBRegressor(
        n_estimators=2000, max_depth=7, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0, verbosity=0, n_jobs=-1, random_state=42
    )
    m.fit(X_tr[tr_idx], log_y[tr_idx],
          eval_set=[(X_tr[va_idx], log_y[va_idx])], verbose=False)
    oof_xgb[va_idx] = m.predict(X_tr[va_idx])
    test_xgb += m.predict(X_te) / 5
xgb_score = rmsle(y_train, np.expm1(oof_xgb))
print(f"XGBoost: {xgb_score:.5f}")

# CatBoost
oof_cat = np.zeros(n_train)
test_cat = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    m = CatBoostRegressor(
        iterations=2000, depth=7, learning_rate=0.03,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        early_stopping_rounds=100
    )
    m.fit(X_tr[tr_idx], log_y[tr_idx],
          eval_set=(X_tr[va_idx], log_y[va_idx]))
    oof_cat[va_idx] = m.predict(X_tr[va_idx])
    test_cat += m.predict(X_te) / 5
cat_score = rmsle(y_train, np.expm1(oof_cat))
print(f"CatBoost: {cat_score:.5f}")

# Ridge regression (simple model for diversity)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

oof_ridge = np.zeros(n_train)
test_ridge = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr[tr_idx])
    Xva = sc.transform(X_tr[va_idx])
    Xte = sc.transform(X_te)
    m = Ridge(alpha=10)
    m.fit(Xtr, log_y[tr_idx])
    oof_ridge[va_idx] = m.predict(Xva)
    test_ridge += m.predict(Xte) / 5
ridge_score = rmsle(y_train, np.expm1(oof_ridge))
print(f"Ridge: {ridge_score:.5f}")

# Ensemble search (4 models)
best_rmsle_val = 999
best_w = None
for w1 in np.arange(0.0, 0.51, 0.05):
    for w2 in np.arange(0.0, 0.51, 0.05):
        for w3 in np.arange(0.0, 1.01 - w1 - w2, 0.05):
            w4 = 1.0 - w1 - w2 - w3
            if w4 < -0.01 or w4 > 1.01:
                continue
            oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat + w4 * oof_ridge
            score = rmsle(y_train, np.expm1(oof_ens))
            if score < best_rmsle_val:
                best_rmsle_val = score
                best_w = (w1, w2, w3, w4)

print(f"\nBest ensemble: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CAT={best_w[2]:.2f}, Ridge={best_w[3]:.2f}")
print(f"Best ensemble RMSLE: {best_rmsle_val:.5f}")

# Save submissions
sub_template = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

# CatBoost
sub = sub_template.copy()
sub['revenue'] = np.clip(np.expm1(test_cat), 1, None)
sub.to_csv(f"{COMP_DIR}/submissions/submission_06_cat_budget_imputed.csv", index=False)

# Best ensemble
test_best = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cat + best_w[3] * test_ridge
sub = sub_template.copy()
sub['revenue'] = np.clip(np.expm1(test_best), 1, None)
sub.to_csv(f"{COMP_DIR}/submissions/submission_07_ens4_budget_imputed.csv", index=False)

# Simple avg LGB+CAT
test_lc = (test_lgb + test_cat) / 2
oof_lc = (oof_lgb + oof_cat) / 2
lc_score = rmsle(y_train, np.expm1(oof_lc))
sub = sub_template.copy()
sub['revenue'] = np.clip(np.expm1(test_lc), 1, None)
sub.to_csv(f"{COMP_DIR}/submissions/submission_08_avg_lgb_cat.csv", index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"LightGBM:      {lgb_score:.5f}")
print(f"XGBoost:       {xgb_score:.5f}")
print(f"CatBoost:      {cat_score:.5f}")
print(f"Ridge:         {ridge_score:.5f}")
print(f"Avg LGB+CAT:   {lc_score:.5f}")
print(f"Best Ens:      {best_rmsle_val:.5f}")
