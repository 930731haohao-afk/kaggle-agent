"""EDA + Feature Engineering + Baseline for TMDB Box Office Prediction.
JSON columns need parsing. Revenue is log-normal -> use log1p transform.
"""
import pandas as pd
import numpy as np
import ast
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

DATA_DIR = "/home/tjyen/ai_agents/kaggle/competitions/tmdb-box-office-prediction/data"
COMP_DIR = "/home/tjyen/ai_agents/kaggle/competitions/tmdb-box-office-prediction"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

print("=" * 60)
print("EDA: TARGET ANALYSIS")
print("=" * 60)
rev = train['revenue']
log_rev = np.log1p(rev)
print(f"Revenue: min={rev.min():,.0f}, max={rev.max():,.0f}")
print(f"Log1p revenue: mean={log_rev.mean():.2f}, std={log_rev.std():.2f}")

# Budget analysis
print("\n" + "=" * 60)
print("EDA: BUDGET ANALYSIS")
print("=" * 60)
print(f"Budget zeros: {(train['budget'] == 0).sum()} / {len(train)} ({(train['budget']==0).mean()*100:.1f}%)")
print(f"Budget non-zero mean: {train.loc[train['budget']>0, 'budget'].mean():,.0f}")
has_budget = train['budget'] > 0
print(f"Corr(log_budget, log_revenue) where budget>0: {np.corrcoef(np.log1p(train.loc[has_budget, 'budget']), np.log1p(train.loc[has_budget, 'revenue']))[0,1]:.3f}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

def safe_eval(val):
    """Safely evaluate JSON-like strings."""
    if pd.isna(val) or val == '' or val == '[]':
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        try:
            return json.loads(val)
        except:
            return []

def extract_features(df, top_companies_list=None):
    """Extract features from raw TMDB data."""
    feats = pd.DataFrame(index=df.index)

    # Budget
    feats['budget'] = df['budget'].values
    feats['has_budget'] = (df['budget'] > 0).astype(int)
    feats['log_budget'] = np.log1p(df['budget'])

    # Popularity
    feats['popularity'] = df['popularity'].values
    feats['log_popularity'] = np.log1p(df['popularity'])

    # Runtime
    feats['runtime'] = df['runtime'].fillna(df['runtime'].median()).values
    feats['runtime_missing'] = df['runtime'].isnull().astype(int)

    # Original language
    feats['is_english'] = (df['original_language'] == 'en').astype(int)

    # Release date features
    dates = pd.to_datetime(df['release_date'], format='mixed', errors='coerce')
    feats['release_year'] = dates.dt.year
    feats['release_month'] = dates.dt.month
    feats['release_day'] = dates.dt.day
    feats['release_dayofweek'] = dates.dt.dayofweek
    feats['release_quarter'] = dates.dt.quarter
    feats['is_summer_release'] = dates.dt.month.isin([5, 6, 7, 8]).astype(int)
    feats['is_holiday_release'] = dates.dt.month.isin([11, 12]).astype(int)
    feats.loc[feats['release_year'] > 2019, 'release_year'] -= 100

    # Status
    feats['is_released'] = (df['status'] == 'Released').astype(int)

    # Homepage
    feats['has_homepage'] = df['homepage'].notna().astype(int)

    # Tagline
    feats['has_tagline'] = df['tagline'].notna().astype(int)
    feats['tagline_length'] = df['tagline'].fillna('').str.len()

    # Overview
    feats['overview_length'] = df['overview'].fillna('').str.len()
    feats['overview_word_count'] = df['overview'].fillna('').str.split().str.len()

    # Title
    feats['title_length'] = df['title'].fillna('').str.len()

    # Collection
    feats['belongs_to_collection'] = df['belongs_to_collection'].notna().astype(int)

    # Genres
    genres_list = df['genres'].apply(safe_eval)
    feats['n_genres'] = genres_list.apply(len)
    top_genres = ['Drama', 'Comedy', 'Thriller', 'Action', 'Horror', 'Romance',
                  'Adventure', 'Crime', 'Science Fiction', 'Fantasy', 'Family',
                  'Animation', 'Documentary', 'Mystery']
    for genre in top_genres:
        feats[f'genre_{genre}'] = genres_list.apply(
            lambda x: int(any(d.get('name') == genre for d in x if isinstance(d, dict))))

    # Production companies
    companies = df['production_companies'].apply(safe_eval)
    feats['n_companies'] = companies.apply(len)
    if top_companies_list is not None:
        for name in top_companies_list:
            safe_name = name[:20].replace(' ', '_')
            feats[f'company_{safe_name}'] = companies.apply(
                lambda x, n=name: int(any(d.get('name') == n for d in x if isinstance(d, dict))))

    # Production countries
    countries = df['production_countries'].apply(safe_eval)
    feats['n_countries'] = countries.apply(len)
    feats['is_us_produced'] = countries.apply(
        lambda x: int(any(d.get('iso_3166_1') == 'US' for d in x if isinstance(d, dict))))

    # Spoken languages
    languages = df['spoken_languages'].apply(safe_eval)
    feats['n_languages'] = languages.apply(len)

    # Keywords
    keywords = df['Keywords'].apply(safe_eval)
    feats['n_keywords'] = keywords.apply(len)

    # Cast
    cast = df['cast'].apply(safe_eval)
    feats['n_cast'] = cast.apply(len)
    feats['n_male_cast'] = cast.apply(lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('gender') == 2))
    feats['n_female_cast'] = cast.apply(lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('gender') == 1))

    # Crew
    crew = df['crew'].apply(safe_eval)
    feats['n_crew'] = crew.apply(len)
    feats['n_directors'] = crew.apply(
        lambda x: sum(1 for c in x if isinstance(c, dict) and c.get('job') == 'Director'))

    return feats

# Get top companies from train
logging.info("Finding top production companies from train...")
train_companies = train['production_companies'].apply(safe_eval)
company_counts = {}
for comp_list in train_companies:
    for c in comp_list:
        if isinstance(c, dict):
            name = c.get('name', '')
            company_counts[name] = company_counts.get(name, 0) + 1
top_companies_list = [name for name, _ in sorted(company_counts.items(), key=lambda x: -x[1])[:20]]
print(f"Top companies: {top_companies_list[:5]}...")

logging.info("Extracting train features...")
X_train = extract_features(train, top_companies_list)
logging.info("Extracting test features...")
X_test = extract_features(test, top_companies_list)

y_train = train['revenue'].values
log_y_train = np.log1p(y_train)

print(f"\nFeatures: {X_train.shape[1]}")
print(f"Feature names: {X_train.columns.tolist()[:20]}...")

# Check feature correlations with target
print("\n" + "=" * 60)
print("TOP FEATURE CORRELATIONS WITH LOG(REVENUE)")
print("=" * 60)
corrs = X_train.corrwith(pd.Series(log_y_train, index=X_train.index))
corrs_sorted = corrs.abs().sort_values(ascending=False)
for feat in corrs_sorted.head(20).index:
    print(f"  {feat:35s} r={corrs[feat]:+.3f}")

# ============================================================
# MODELING
# ============================================================
print("\n" + "=" * 60)
print("MODELING")
print("=" * 60)

feature_cols = X_train.columns.tolist()
X_tr = X_train[feature_cols].values
X_te = X_test[feature_cols].values

# Fill NaN
X_tr = np.nan_to_num(X_tr, nan=0)
X_te = np.nan_to_num(X_te, nan=0)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# LightGBM on log(revenue)
logging.info("Training LightGBM...")
oof_lgb = np.zeros(len(train))
test_lgb = np.zeros(len(test))

lgb_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr[tr_idx], log_y_train[tr_idx],
        eval_set=[(X_tr[va_idx], log_y_train[va_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_lgb[va_idx] = model.predict(X_tr[va_idx])
    test_lgb += model.predict(X_te) / 5
    fold_rmsle = rmsle(y_train[va_idx], np.expm1(oof_lgb[va_idx]))
    print(f"  Fold {fold+1}: RMSLE = {fold_rmsle:.5f}, best_iter = {model.best_iteration_}")

lgb_rmsle = rmsle(y_train, np.expm1(oof_lgb))
print(f"\nLightGBM 5-fold CV RMSLE: {lgb_rmsle:.5f}")

# Feature importance
importances = model.feature_importances_
imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)
print("\nTop 15 features:")
for _, row in imp_df.head(15).iterrows():
    print(f"  {row['feature']:35s} imp={row['importance']:5.0f}")

# XGBoost
logging.info("Training XGBoost...")
import xgboost as xgb

oof_xgb = np.zeros(len(train))
test_xgb = np.zeros(len(test))

xgb_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbosity': 0,
    'n_jobs': -1,
    'random_state': 42,
}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr[tr_idx], log_y_train[tr_idx],
        eval_set=[(X_tr[va_idx], log_y_train[va_idx])],
        verbose=False
    )
    oof_xgb[va_idx] = model_xgb.predict(X_tr[va_idx])
    test_xgb += model_xgb.predict(X_te) / 5

xgb_rmsle = rmsle(y_train, np.expm1(oof_xgb))
print(f"XGBoost 5-fold CV RMSLE: {xgb_rmsle:.5f}")

# CatBoost
logging.info("Training CatBoost...")
from catboost import CatBoostRegressor

oof_cat = np.zeros(len(train))
test_cat = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
    model_cat = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        early_stopping_rounds=50
    )
    model_cat.fit(
        X_tr[tr_idx], log_y_train[tr_idx],
        eval_set=(X_tr[va_idx], log_y_train[va_idx]),
    )
    oof_cat[va_idx] = model_cat.predict(X_tr[va_idx])
    test_cat += model_cat.predict(X_te) / 5

cat_rmsle = rmsle(y_train, np.expm1(oof_cat))
print(f"CatBoost 5-fold CV RMSLE: {cat_rmsle:.5f}")

# Ensemble
logging.info("Ensembling...")
for w_lgb in np.arange(0.2, 0.8, 0.1):
    for w_xgb in np.arange(0.1, 0.8 - w_lgb + 0.01, 0.1):
        w_cat = 1.0 - w_lgb - w_xgb
        if w_cat < 0.05:
            continue
        oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
        ens_rmsle = rmsle(y_train, np.expm1(oof_ens))
# Find best blend
best_rmsle = 999
best_w = None
for w_lgb in np.arange(0.0, 1.01, 0.05):
    for w_xgb in np.arange(0.0, 1.01 - w_lgb, 0.05):
        w_cat = 1.0 - w_lgb - w_xgb
        if w_cat < -0.01:
            continue
        oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
        ens_rmsle_val = rmsle(y_train, np.expm1(oof_ens))
        if ens_rmsle_val < best_rmsle:
            best_rmsle = ens_rmsle_val
            best_w = (w_lgb, w_xgb, w_cat)

print(f"\nBest ensemble: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CAT={best_w[2]:.2f}")
print(f"Best ensemble RMSLE: {best_rmsle:.5f}")

# Generate submissions
sub_template = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

# LightGBM submission
sub1 = sub_template.copy()
sub1['revenue'] = np.clip(np.expm1(test_lgb), 1, None)
sub1.to_csv(f"{COMP_DIR}/submissions/submission_01_lgb.csv", index=False)

# Best ensemble submission
test_ens = best_w[0] * test_lgb + best_w[1] * test_xgb + best_w[2] * test_cat
sub2 = sub_template.copy()
sub2['revenue'] = np.clip(np.expm1(test_ens), 1, None)
sub2.to_csv(f"{COMP_DIR}/submissions/submission_02_ensemble.csv", index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"LightGBM RMSLE:    {lgb_rmsle:.5f}")
print(f"XGBoost RMSLE:     {xgb_rmsle:.5f}")
print(f"CatBoost RMSLE:    {cat_rmsle:.5f}")
print(f"Best Ensemble:     {best_rmsle:.5f} (LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CAT={best_w[2]:.2f})")
print(f"\nSaved submission_01_lgb.csv and submission_02_ensemble.csv")
