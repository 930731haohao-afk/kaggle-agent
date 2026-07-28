# Titanic Competition Status

**Competition**: https://www.kaggle.com/competitions/titanic
**Problem**: Binary classification (predict survival)
**Metric**: Accuracy
**Train**: 891 rows, **Test**: 418 rows
**Best LB Score**: 0.77033

## Key EDA Findings
- Sex is the strongest predictor (female ~74% survival, male ~19%)
- Pclass highly predictive (1st class ~63%, 3rd class ~24%)
- Age matters (children have higher survival)
- Fare correlates with survival (higher fare = higher survival)
- Family size has non-linear effect (solo and large families do worse)
- Cabin 77% missing — used as has_cabin flag + deck letter
- Title extracted from Name is very informative
- No significant train-test distribution shift

## Feature Engineering

### Baseline Features (19 features)
- Title, FamilySize, IsAlone, AgeBin, IsChild, FareLog, FareBin, HasCabin, Deck_freq, TicketFreq, FarePerPerson
- Age imputed by Title + Pclass median
- Categoricals encoded via frequency encoding (fit on train only)

### Adaptive Search Features (+5 → 21 features)
From Kaggle Discussion board research (top 3% solution techniques):
- **Family_Survival_Rate**: Mean survival rate per surname group (from training data)
- **Ticket_Survival_Rate**: Mean survival rate per ticket group
- **Combined_Survival_Rate**: Average of family + ticket rates
- **DeckGroup / DeckGroup_freq**: Grouped deck letters (ABC, DE, FG, U)
- **TicketPrefix_freq**: Frequency of ticket prefix patterns
- Leakage-safe CV: survival rates recomputed within each fold using only training fold data

### Push-to-80 Features (+10 → 31 features)
- Title indicator variables (IsMaster, IsMrs, IsMiss, IsMr)
- Fare-Pclass interactions (Fare_x_Pclass, FareLog_x_Pclass, FarePclassDeviation)
- Sex_x_Pclass interaction
- FamilySizeBin, IsElderly
- **Result**: Too many features on 891 rows — CV and LB both degraded ("more features = better" anti-pattern)

## Experiment Results

### Phase 1: Baseline Models (2026-02-10)

| # | Model | CV Mean | CV Std | Features |
|---|-------|---------|--------|----------|
| 1 | MajorityClass | 0.6162 | 0.0000 | 19 |
| 2 | LogisticRegression | 0.8227 | 0.0127 | 19 |
| 3 | RandomForest | 0.8417 | 0.0166 | 19 |
| 4 | LightGBM-default | 0.8361 | 0.0159 | 19 |
| 5 | LightGBM-tuned | 0.8361 | 0.0192 | 19 |
| 6 | XGBoost | 0.8440 | 0.0067 | 19 |

### Phase 2: Self-Improvement — Best-of-N (2026-03-18)

| # | Model | CV Mean | CV Std | Features |
|---|-------|---------|--------|----------|
| 7 | CatBoost | 0.8339 | 0.0210 | 19 |
| 8 | XGBoost-FeatureSelected (top 10) | 0.8440 | 0.0170 | 19 |
| 9 | LightGBM-HeavyReg | 0.8395 | 0.0241 | 19 |
| 10 | **SoftVotingEnsemble (LR+RF+XGB+LGB)** | **0.8451** | 0.0198 | 19 |
| 11 | GradientBoosting-conservative | 0.8361 | 0.0146 | 19 |

### Phase 3: Adaptive Search — Survival Rate Features (2026-03-18)

| # | Model | CV Mean | CV Std | Features |
|---|-------|---------|--------|----------|
| 12 | XGBoost-AdaptiveSearch | 0.8361 | 0.0142 | 21 |
| 13 | Ensemble-AdaptiveSearch | 0.8361 | 0.0184 | 21 |
| 14 | GBM-AdaptiveSearch | 0.8316 | 0.0180 | 21 |

### Phase 4: Push-to-80 — All 5 Improvements (2026-03-18)

| # | Model | CV Mean | CV Std | Features |
|---|-------|---------|--------|----------|
| 15 | XGBoost-PushTo80 | 0.8350 | 0.0174 | 31 |
| 16 | Ensemble6-PushTo80 (LR+RF+XGB+LGB+SVM+KNN) | 0.8272 | 0.0121 | 31 |
| 17 | Ensemble6-ThreshOpt (threshold=0.42) | 0.8317 | 0.0190 | 31 |
| 18 | GBM-PushTo80 | 0.8328 | 0.0076 | 31 |

## Submissions

| Date | File | Description | Public LB |
|------|------|-------------|-----------|
| 2026-02-10 | submission_xgboost_0.844_*.csv | XGBoost baseline (19 features) | 0.75358 |
| 2026-03-18 | submission_ensemble_0.845_*.csv | Soft voting (LR+RF+XGB+LGB, 19 features) | 0.76794 |
| 2026-03-18 | submission_ensemble_adaptive_*.csv | Ensemble + survival rate features (21 features) | **0.77033** |
| 2026-03-18 | submission_xgb_push80_*.csv | XGBoost + all 5 improvements (31 features) | 0.75598 |
| 2026-03-18 | submission_ens6_thresh42_*.csv | 6-model ensemble, threshold=0.42 (31 features) | 0.76555 |

## Lessons Learned

### CV-LB Gap
- CV consistently overestimates LB by ~7-9% on this dataset (891 rows)
- Higher CV doesn't always mean higher LB — generalization matters more
- Soft voting ensemble had highest CV (0.845) but not highest LB (0.768)
- Adaptive search ensemble had lower CV (0.836) but best LB (0.770)

### Feature Engineering Insights
- **Survival rate features** (family + ticket) were the single biggest LB improvement (+0.002 over ensemble)
- Combined_Survival_Rate ranked #2-3 in feature importance across all models
- Going from 21 → 31 features **hurt** both CV and LB — classic overfitting on small dataset
- Sweet spot for Titanic is ~19-21 carefully chosen features

### Self-Improvement Strategies Applied
1. **Best-of-N**: Tested 5 diverse models, identified SoftVotingEnsemble as best
2. **Verifiable Rewards**: Tracked CV delta after every experiment to detect plateau
3. **Reflexion**: Applied after push-to-80 failure — identified "more features" anti-pattern
4. **Adaptive Search**: Mined Kaggle Discussion boards via web search, found survival rate technique from top 3% solution
5. **Experience Library**: Confirmed known patterns (CV-LB gap, small dataset overfitting)

### Anti-Patterns Confirmed
- "More features = better" — Adding 10 features to 891-row dataset degraded performance
- Threshold optimization on CV doesn't transfer well to LB on small datasets
- 100% Titanic scores use external historical data, not genuine ML

## Next Steps (if resuming)
- Try feature selection (top 15-18 features from the 21-feature set)
- Stacking with out-of-fold predictions (careful of overfitting)
- Different CV strategy (repeated stratified k-fold for more stable estimates)
- Target 0.78+ would require novel approaches beyond standard tabular ML
