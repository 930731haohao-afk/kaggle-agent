# Spaceship Titanic Competition Status

**Competition**: https://www.kaggle.com/competitions/spaceship-titanic
**Problem**: Binary classification (predict which passengers were transported to alternate dimension)
**Metric**: Accuracy
**Train**: 8,693 rows, **Test**: 4,277 rows
**Best LB Score**: 0.80430

## Key EDA Findings
- Target nearly balanced: 50.4% transported, 49.6% not
- CryoSleep is strongest predictor: 82% transport rate vs 33% when awake
- CryoSleep passengers spend $0 on all amenities (consistent)
- HomePlanet matters: Europa 66%, Mars 52%, Earth 42% transport rate
- Destination: 55 Cancri e highest (61%), TRAPPIST-1e lowest (47%)
- PassengerId encodes travel groups ("GGGG_PP" format)
- Cabin encodes deck/number/side ("D/N/S" format)
- 5 spending features highly right-skewed (median=0, max up to ~30K)

## Feature Engineering

### Phase 1: Baseline Features (32 features)
- Group features: GroupSize, IsAlone (from PassengerId parsing)
- Cabin features: Deck, CabinNum, Side (from Cabin parsing)
- Spending: TotalSpending, HasSpent, log-transforms, luxury vs basic ratio
- Age: IsChild, IsTeenager, AgeBin
- Categoricals: label-encoded HomePlanet, Destination, Deck, Side

### Phase 2: Enhanced Features (43 features, +11)
- **Group-level stats**: GroupSpend_mean/std/max, GroupAge_mean/std
- **Interaction features**: Cryo_x_Planet, Age_x_Cryo, Deck_x_Side
- **Frequency encoding**: Deck_freq
- **Spatial**: CabinNumBin (quantile-binned cabin number)
- **Spending patterns**: SpendingPerGroupMember, NumServicesUsed, SpendingAnomaly

### Phase 3: Adaptive Search Features (50 features, +7) — FAILED
- **Group_Transport_Rate**: Mean transport rate per group (from training data)
- **Surname_Transport_Rate**: Mean transport rate per surname
- **Combined_Transport_Rate**: Average of group + surname rates
- **SpendingCluster**: K-Means clustering on log-spending (5 clusters)
- **Frequency encodings**: HomePlanet_freq, Destination_freq, Side_freq
- **Smart CryoSleep imputation**: Infer CryoSleep from spending patterns
- **Result**: Transport rates caused massive CV drop with leakage-safe CV (0.814 → 0.748). LB also degraded (0.763). Unlike Titanic, group membership is not strongly predictive of transport outcome.

## Experiment Results

### Phase 1: Baseline Models (2026-02-10, 32 features)

| # | Model | CV Mean | CV Std |
|---|-------|---------|--------|
| 1 | RandomForest | 0.8003 | 0.0073 |
| 2 | ExtraTrees | 0.8009 | 0.0116 |
| 3 | LightGBM | 0.8064 | 0.0051 |
| 4 | XGBoost | 0.8101 | 0.0073 |

### Phase 2: Self-Improvement — Best-of-N + Ensemble (2026-03-18, 43 features)

| # | Model | CV Mean | CV Std |
|---|-------|---------|--------|
| 5 | XGBoost-Enhanced | 0.8138 | 0.0103 |
| 6 | LightGBM-Enhanced | 0.8141 | 0.0059 |
| 7 | CatBoost-Enhanced | 0.8140 | 0.0083 |
| 8 | RandomForest-Enhanced | 0.8029 | 0.0123 |
| 9 | LogisticRegression-Enhanced | 0.7968 | 0.0085 |
| 10 | **SoftVoting-4Model** | **0.8142** | 0.0072 |
| 11 | SoftVoting-6Model | 0.8107 | 0.0079 |

### Phase 3: Adaptive Search — Transport Rates (2026-03-18, 50 features, leakage-safe CV)

| # | Model | CV Mean | CV Std | Notes |
|---|-------|---------|--------|-------|
| 12 | XGBoost-Adaptive | 0.7481 | 0.0088 | Transport rates noisy per fold |
| 13 | LightGBM-Adaptive | 0.7465 | 0.0100 | |
| 14 | CatBoost-Adaptive | 0.7562 | 0.0073 | |
| 15 | SoftVoting-Adaptive | 0.7503 | 0.0082 | |

## Submissions

| Date | File | Description | Public LB |
|------|------|-------------|-----------|
| 2026-02-10 | submission_ensemble_0.8101_*.csv | Weighted ensemble (RF+ET+LGB+XGB), 32 features | 0.79869 |
| 2026-03-18 | submission_selfimprove_t50_*.csv | 4-model soft voting (XGB+LGB+CB+RF), 43 features, t=0.50 | **0.80430** |
| 2026-03-18 | submission_selfimprove_*.csv | Same model, threshold=0.44 | 0.80360 |
| 2026-03-18 | submission_adaptive_*.csv | 4-model + transport rates, 50 features | 0.76315 |

## Lessons Learned

### What Worked
- **Enhanced feature engineering** (43 features): Group spending stats, interaction features, spatial features improved LB from 0.799 → 0.804
- **CatBoost**: Matched XGBoost/LightGBM — no clear single model winner among gradient boosters
- **4-model soft voting ensemble** (XGB+LGB+CB+RF): Best overall approach
- **Default threshold (0.50)**: Threshold optimization on CV didn't transfer to LB (same as Titanic)

### What Failed
- **Group/Surname transport rates**: Unlike Titanic, group membership doesn't strongly predict transport outcome. Adding transport rates caused LB to drop from 0.804 → 0.763
- **6-model ensemble (adding SVM+LR)**: Worse than 4-model — weaker models diluted the ensemble
- **Spending clustering**: K-Means clusters added noise rather than signal
- **Smart CryoSleep imputation**: Marginal impact at best

### Key Differences from Titanic
- Titanic: family surname strongly predicts survival ("women and children first" → family correlation)
- Spaceship Titanic: "Transported" is less correlated within groups — CryoSleep and spending patterns are the main drivers, not group identity
- Techniques that transferred well: ensembling, group spending stats, interaction features
- Techniques that did NOT transfer: survival/transport rate features, threshold optimization

### Self-Improvement Strategies Applied
1. **Experience Library**: Applied Titanic lessons (ensembles, survival rates, interactions)
2. **Best-of-N**: Tested 5 model candidates, identified LightGBM as single-best, ensemble as overall best
3. **Verifiable Rewards**: Tracked CV delta after every experiment
4. **Reflexion**: Applied after transport rate failure — identified domain difference from Titanic
5. **Adaptive Search**: Searched Discussion boards and blogs for tips; spending patterns and group features helped, transport rates did not

## Next Steps (if resuming)
- Try HistGradientBoosting (native NaN support, no need for imputation)
- Feature selection: test dropping weakest features from the 43-feature set
- Optuna hyperparameter tuning for XGBoost/LightGBM
- Try target encoding for Deck (high signal categorical)
- Stacking with out-of-fold predictions

---

## 5-Stage Ablation Ladder (2026-07-20, huangweihaohuang account)

Ran the faithful Stage 1→5 ladder via `competitions/run_stage_ladder.py` with bespoke
per-comp features (`scripts/features_ladder.py`: Group/GroupSize/Alone from PassengerId,
Cabin deck/num/side, spending aggregates + NoSpend, Name→FamilySize, CryoSleep/VIP tri-state).
Shared folds, seed=42, decisions on post-processed OOF.

| Stage | 1 baseline | 2 skill | 3 +linear-iter | 4 +tree-search | 5 +idea-inject |
|---|---|---|---|---|---|
| CV accuracy | 0.8012 | 0.8075 | 0.8121 | **0.8143** | 0.8143 |

- **Champion submitted → Public LB 0.80383** (`ladder_stage5_accuracy_0.81433.csv`), ≈ tie with
  the old bespoke 0.8043. Ladder is monotonic; gains concentrate at S2→S3.
- **Stage 5 = Stage 4** → idea injection null (constructive no-op), consistent with the project's
  second-pillar finding.
- CV 0.8143 → LB 0.8038: the known ~0.01 CV-vs-LB optimism on this comp (not a bug).
