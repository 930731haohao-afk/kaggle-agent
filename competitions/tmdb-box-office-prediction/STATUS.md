# TMDB Box Office Prediction

## Competition Details
- **URL**: https://www.kaggle.com/competitions/tmdb-box-office-prediction
- **Task**: Predict movie box office revenue from TMDB metadata
- **Metric**: RMSLE (Root Mean Squared Logarithmic Error)
- **Teams**: 1,395
- **Reward**: Knowledge
- **Deadline**: 2019-05-30

## Data
- **Train**: 3,000 movies × 23 columns
- **Test**: 4,398 movies × 22 columns
- **Target**: Revenue (1 to 1.5B, highly skewed → log-normal)
- **Key features**: Budget (27% missing!), popularity, cast/crew JSON, genres JSON, release date
- **Many JSON columns**: genres, cast, crew, production_companies, keywords, spoken_languages

## Key Insights

1. **Budget imputation is critical**: 27% of movies have budget=0 (missing). Imputing with a LightGBM model improved LB from 2.011 to 1.973
2. **CatBoost dominates**: Consistently beats LightGBM and XGBoost on this small dataset
3. **Ensemble helps**: 4-model blend (LGB+XGB+CAT+Ridge) beats best single model by 0.01 on LB
4. **log1p transform essential**: Revenue spans 1 to 1.5B — log transform normalizes the distribution
5. **Top features**: log_popularity (r=0.54), log_budget (r=0.50), crew/cast counts, has_tagline
6. **Major studios matter**: Presence of Warner Bros., Universal, etc. is a strong predictor
7. **Good CV-LB alignment**: CV and LB track consistently (~2% gap)

## Approaches Tried

### 1. CatBoost Baseline (CV 2.056, LB 2.019)
- 68 features: budget, popularity, date, genres, studios, cast/crew counts
- Budget zeros treated as valid zero value
- CatBoost best among LGB/XGB/CAT

### 2. Improved Features (CV 2.051, LB 2.011)
- 87 features: added target encoding, major studios, top actors/directors
- Marginal improvement from more features

### 3. Budget Imputation + 4-Model Ensemble (CV 2.022, LB 1.973)
- Imputed budget=0 using LightGBM trained on movies with known budgets
- 4-model ensemble: LGB(10%) + XGB(20%) + CatBoost(55%) + Ridge(15%)
- **Best submission**

## Results

| Submission | Method | CV RMSLE | Public LB | Private LB |
|-----------|--------|----------|-----------|------------|
| v1 | CatBoost baseline | 2.056 | 2.019 | 2.019 |
| v2 | CatBoost + improved feats | 2.051 | 2.011 | 2.011 |
| v3 | Avg3 (LGB+XGB+CAT) | 2.074 | 2.011 | 2.011 |
| **v4** | **4-model ens + budget imputation** | **2.022** | **1.973** | **1.973** |
| v5 | CatBoost + budget imputation | 2.030 | 1.984 | 1.984 |

**Best LB: 1.973** (v4: 4-model ensemble with budget imputation)

## Files
```
scripts/
├── 00_inspect_data.py           # Data inspection
├── 01_eda_and_baseline.py       # EDA + feature engineering + LGB/XGB/CAT baselines
├── 02_improved.py               # Improved features + target encoding
└── 03_budget_imputation.py      # Budget imputation + 4-model ensemble
submissions/
├── submission_07_ens4_budget_imputed.csv  # Best (LB 1.973)
└── ... (6 other submissions)
```
