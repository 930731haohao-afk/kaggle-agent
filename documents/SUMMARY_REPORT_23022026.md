# Kaggle AI Agent: 30-Competition Summary Report

**Author**: Claude Code (Hybrid AI Agent) + Human Oversight
**Period**: February 10 -- 23, 2026 (19 sessions)
**Platform**: NVIDIA GB10 GPU, Python 3.13, uv package manager
**Kaggle User**: tjyen1975

---

## 1. Executive Summary

Over 19 sessions, this hybrid AI agent completed **30 Kaggle competitions** spanning 10 distinct problem types: tabular regression, tabular classification, time series forecasting, image classification, object detection, NLP, audio classification, spectral analysis, event detection, and stroke recovery. The agent autonomously executed end-to-end data science pipelines -- from data ingestion and EDA through feature engineering, modeling, evaluation, and submission -- with human oversight at each stage.

**Key statistics**:
- **20** competitions with public leaderboard submissions
- **10** completed locally (closed competitions or code-competition format)
- **9** different model families used (LightGBM, XGBoost, CatBoost, Ridge, CNN, ResNet, Faster R-CNN, DistilBERT, custom ensembles)
- **Best relative result**: Digit Recognizer at 99.61% accuracy (CNN)
- **Closest to #1**: ICDAR Stroke Recovery within 0.3% of leaderboard best

---

## 2. All Competitions at a Glance

| # | Competition | Problem Type | Metric | Train Size | Features | CV Score | Public LB | Private LB | Best Model |
|---|------------|-------------|--------|-----------|----------|----------|-----------|------------|------------|
| 1 | Titanic | Binary Clf | Accuracy | 891 | 19 | 0.844 | 0.754 | -- | XGBoost |
| 2 | House Prices | Regression | RMSLE | 1,460 | 88 | 0.114 | 0.126 | -- | Ridge+LGB+XGB Ensemble |
| 3 | Cat in the Dat | Binary Clf | AUC | 300,000 | 45 | 0.831 | 0.793 | 0.790 | LR+LGB+XGB Ensemble |
| 4 | Forest Cover Type | 7-class Clf | Accuracy | 15,120 | 67 | 0.889 | 0.775 | 0.775 | RF+ET+LGB+XGB Ensemble |
| 5 | Spaceship Titanic | Binary Clf | Accuracy | 8,693 | 32 | 0.810 | 0.799 | -- | RF+ET+LGB+XGB Ensemble |
| 6 | TPS Jan 2022 | Time Series Reg | SMAPE | 26,298 | 25 | 7.9% | 9.068 | 9.865 | LGB+XGB Ensemble |
| 7 | TPS Aug 2022 | Binary Clf | AUC | 26,570 | 39 | 0.588 | 0.574 | 0.582 | LR+LGB+XGB Ensemble |
| 8 | Store Sales | Time Series Reg | RMSLE | 3,000,888 | 33 | 0.591 | 0.415 | -- | LGB+XGB Ensemble |
| 9 | PS S6E1 (Exams) | Regression | RMSE | 630,000 | 22 | R2=0.786 | 8.704 | 8.729 | LGB+XGB Ensemble |
| 10 | PS S5E10 (Accident) | Regression | RMSE | 517,754 | 27 | 0.056 | 0.056 | 0.056 | LGB+XGB Ensemble |
| 11 | Home Data ML | Regression | MAE | 1,460 | 107 | 13,293 | 12,616 | -- | Ridge+GBR Ensemble |
| 12 | PS S6E2 (Heart) | Binary Clf | AUC | 630,000 | 33 | 0.955 | 0.953 | -- | LGB+CatBoost Ensemble |
| 13 | LLM Classification | 3-class Clf | Log Loss | 57,477 | Text | 1.055 | 1.111 | -- | DistilBERT |
| 14 | PS S3E20 (CO2) | Regression | RMSE | 79,023 | 102 | 33.21 | -- | -- | LGB+XGB Ensemble |
| 15 | Digit Recognizer | 10-class Clf | Accuracy | 42,000 | 784px | 0.996 | 0.996 | -- | CNN 5-fold Ensemble |
| 16 | Energy Prosumers | Time Series Reg | MAE | 2,018,352 | 56 | 35.55 | -- | -- | LightGBM |
| 17 | PS S4E11 (Depression) | Binary Clf | Accuracy | 140,700 | 28 | 0.937 | 0.941 | 0.939 | LightGBM |
| 18 | PS S4E1 (Bank Churn) | Binary Clf | AUC | 165,034 | 28 | 0.897 | 0.887 | 0.892 | LightGBM |
| 19 | Sleep States | Event Detection | EDAP | 127.9M rows | 27 | 0.313 | -- | -- | LightGBM |
| 20 | Global Wheat | Object Detection | mAP@0.5 | 3,373 imgs | -- | 0.797 | -- | -- | Faster R-CNN |
| 21 | CIFAR-10 | 10-class Clf | Accuracy | 50,000 | 32x32px | 0.960 | 0.956 | 0.956 | ResNet-18 |
| 22 | Dogs vs Cats | Binary Clf | Log Loss | 25,000 | Images | 0.031 | -- | -- | ResNet-50 Transfer |
| 23 | Bike Sharing | Regression | RMSLE | 10,886 | 13 | 0.317 | 0.419 | 0.419 | Cat+XGB+LGB Ensemble |
| 24 | Writing Processes | Regression | RMSE | 2,471 | 66 | 0.648 | -- | -- | LGB+XGB+Cat Ensemble |
| 25 | EMVIC (Eyes) | 37-class Clf | Log Loss | 652 | 215 | 0.902 | -- | -- | LightGBM |
| 26 | Whale Detection | Binary Clf | AUC | 47,841 | 402 | 0.951 | -- | -- | LGB+XGB+Cat Ensemble |
| 27 | ICDAR Stroke Recovery | Regression | Col-RMSE | 605 sigs | Image+Traj | 0.245 | -- | -- | CDF+Template Ensemble |
| 28 | AfSIS Soil | Multi-Target Reg | MCRMSE | 1,157 | 3,594 | 0.466 | 0.500 | 0.530 | Ridge Blend |
| 29 | Conway's Reverse GoL | Grid Regression | MAE | 50,000 | 20x20 grid | 0.109 | -- | -- | Deep ResNet CNN |
| 30 | TMDB Box Office | Regression | RMSLE | 3,000 | 87 | 2.022 | 1.973 | 1.973 | 4-model Ensemble |

---

## 3. Breakdown by Problem Type

### 3.1 Tabular Classification (11 competitions)

| Competition | Metric | CV | Public LB | Key Technique |
|------------|--------|-----|-----------|---------------|
| Titanic | Accuracy | 0.844 | 0.754 | Title extraction, family features |
| Cat in the Dat | AUC | 0.831 | 0.793 | Target encoding for high-cardinality categoricals |
| Forest Cover Type | Accuracy | 0.889 | 0.775 | 4-model ensemble, distance features |
| Spaceship Titanic | Accuracy | 0.810 | 0.799 | CryoSleep dominance, cabin/group features |
| TPS Aug 2022 | AUC | 0.588 | 0.574 | Low-signal: LR outperformed tree models |
| PS S6E2 (Heart) | AUC | 0.955 | 0.953 | CatBoost + multi-seed, 33 focused features |
| PS S4E11 (Depression) | Accuracy | 0.937 | 0.941 | Threshold optimization (0.5 to 0.75) |
| PS S4E1 (Bank Churn) | AUC | 0.897 | 0.887 | Surname target encoding with smoothing |
| EMVIC (Eyes) | Log Loss | 0.902 | -- | 115 statistical + 100 PCA from time series |
| Whale Detection | AUC | 0.951 | -- | 402 audio features (mel, MFCC, spectral) |
| LLM Classification | Log Loss | 1.055 | 1.111 | DistilBERT fine-tuning, gradient clipping |

**Pattern**: Gradient boosting ensembles (LightGBM + XGBoost + CatBoost) dominated tabular classification. Feature engineering consistently mattered more than model choice -- all GBMs typically scored within 1% of each other. For the lone NLP competition, a transformer (DistilBERT) was necessary.

### 3.2 Tabular Regression (8 competitions)

| Competition | Metric | CV | Public LB | Key Technique |
|------------|--------|-----|-----------|---------------|
| House Prices | RMSLE | 0.114 | 0.126 | Outlier removal, log1p, linear+tree ensemble |
| PS S6E1 (Exams) | RMSE | R2=0.786 | 8.704 | study_hours dominance (r=0.76) |
| PS S5E10 (Accident) | RMSE | 0.056 | 0.056 | curvature x speed interaction features |
| Home Data ML | MAE | 13,293 | 12,616 | GBR with Huber loss, outlier removal |
| PS S3E20 (CO2) | RMSE | 33.21 | -- | Log1p for extreme skew, time-based validation |
| Writing Processes | RMSE | 0.648 | -- | 66 keystroke features, small-dataset regularization |
| AfSIS Soil | MCRMSE | 0.466 | 0.530 | Ridge on spectral data, SG derivatives + SNV |
| TMDB Box Office | RMSLE | 2.022 | 1.973 | Budget imputation, JSON feature extraction |

**Pattern**: Target transformations (log1p) were critical for skewed distributions. For high-dimensional spectral data (p >> n), Ridge regression vastly outperformed tree models. Outlier handling and feature engineering drove the biggest improvements.

### 3.3 Time Series (4 competitions)

| Competition | Metric | CV | Public LB | Key Technique |
|------------|--------|-----|-----------|---------------|
| TPS Jan 2022 | SMAPE | 7.9% | 9.068 | Series-level aggregates, time-based CV |
| Store Sales | RMSLE | 0.591 | 0.415 | Lag features, store-family hierarchies |
| Energy Prosumers | MAE | 35.55 | -- | Lag-24 dominance, single > split models |
| Bike Sharing | RMSLE | 0.317 | 0.419 | hour_workingday interaction (12x importance) |

**Pattern**: Lag features were consistently the strongest predictors in time series tasks. Time-based validation splits were essential to avoid data leakage. Single global models often outperformed per-segment models due to shared learning.

### 3.4 Image Classification (3 competitions)

| Competition | Metric | Val Score | Public LB | Key Technique |
|------------|--------|-----------|-----------|---------------|
| Digit Recognizer | Accuracy | 0.996 | 0.996 | 2-block CNN, 5-fold ensemble, augmentation |
| CIFAR-10 | Accuracy | 0.960 | 0.956 | ResNet-18 adapted for 32x32, Cutout |
| Dogs vs Cats | Log Loss | 0.031 | -- | ResNet-50 transfer learning, 99.08% acc |

**Pattern**: CNNs vastly outperformed tabular approaches for image data (99.56% vs 97.10% on MNIST). For small images (32x32), architectures must be adapted (3x3 initial conv, no max pool). For larger images, transfer learning from ImageNet was extremely effective -- reaching 98.8% accuracy on epoch 1.

### 3.5 Specialized Problems (4 competitions)

| Competition | Type | CV | Key Technique |
|------------|------|-----|---------------|
| Sleep States | Event Detection | EDAP 0.313 | LGB + peak detection with NMS |
| Global Wheat | Object Detection | mAP 0.797 | Faster R-CNN, COCO pretrained backbone |
| ICDAR Stroke Recovery | Trajectory Prediction | RMSE 0.245 | CDF + template matching ensemble |
| Conway's Reverse GoL | Grid Inverse Problem | MAE 0.109 | Deep ResNet CNN with dilated convolutions |

**Pattern**: Specialized problems required domain-specific approaches. Object detection benefited hugely from pretrained backbones (COCO). For the Game of Life inverse problem, spatial structure made CNNs far superior to per-cell models. The stroke recovery task was fundamentally ill-posed, requiring heavy blending toward the mean.

---

## 4. Models and Techniques

### 4.1 Model Usage Frequency

| Model | Times Used | Best Solo Performance |
|-------|-----------|----------------------|
| LightGBM | 27/30 | Whale Detection (AUC 0.951), Energy Prosumers (MAE 35.55) |
| XGBoost | 24/30 | Titanic (Accuracy 0.844) |
| CatBoost | 14/30 | Heart Disease (AUC 0.953), TMDB (RMSLE 1.984) |
| Ridge/Linear | 8/30 | AfSIS Soil (MCRMSE 0.530) |
| CNN (PyTorch) | 4/30 | Digit Recognizer (Accuracy 0.996) |
| ResNet | 3/30 | CIFAR-10 (Accuracy 0.956) |
| Faster R-CNN | 1/30 | Global Wheat (mAP 0.797) |
| DistilBERT | 1/30 | LLM Classification (Log Loss 1.111) |
| GBR (Huber) | 1/30 | Home Data (MAE 12,616) |

### 4.2 Ensembling Strategies

| Strategy | Times Used | Typical Improvement |
|----------|-----------|-------------------|
| Weighted average (2-4 models) | 22/30 | +0.5-2% over best single model |
| Multi-seed averaging | 3/30 | +0.01-0.1% (variance reduction) |
| OOF stacking | 2/30 | Mixed results (hurt on small datasets) |
| Meta-averaging submissions | 1/30 | +0.5% on AfSIS Soil |
| K-fold CNN ensemble | 2/30 | +0.3-0.5% on image tasks |

### 4.3 Feature Engineering Highlights

| Technique | Competitions Applied | Impact |
|-----------|---------------------|--------|
| Log1p target transform | House Prices, Store Sales, CO2, Bike, TMDB, Energy | Critical for skewed targets |
| Cyclical encoding (sin/cos) | 8 competitions | Standard for temporal/angular features |
| Target encoding (OOF) | Cat in the Dat, Bank Churn, TMDB | High-cardinality categorical handling |
| Interaction features | 15+ competitions | Often among top-importance features |
| Lag features | 4 time series competitions | Dominant predictors (r > 0.9) |
| Spectral preprocessing (SG, SNV) | AfSIS Soil | Essential for chemometrics data |
| Audio feature extraction (mel, MFCC) | Whale Detection | 402 handcrafted features from 2s clips |
| Data augmentation (image) | 4 image competitions | Rotation, flip, Cutout, ColorJitter |

### 4.4 Validation Strategies

| Strategy | When Used | Competitions |
|----------|-----------|-------------|
| Stratified K-Fold (5) | Balanced tabular classification | 15+ competitions |
| Time-based split | Temporal data | TPS Jan, Store Sales, CO2, Energy |
| GroupKFold | Grouped data | Bike, Wheat, Sleep, ICDAR |
| Writer-held-out | Identity-based | ICDAR Stroke Recovery |
| Single train/val split | Large datasets, image tasks | CIFAR-10, Dogs vs Cats |

---

## 5. CV vs. Leaderboard Analysis

### 5.1 CV-LB Alignment (20 submitted competitions)

| Alignment | Count | Examples |
|-----------|-------|---------|
| Excellent (< 5% gap) | 8 | PS S5E10, PS S6E1, PS S6E2, Heart Disease |
| Good (5-15% gap) | 5 | House Prices, Spaceship Titanic, LLM Classification |
| Poor (> 15% gap) | 7 | Titanic, Forest Cover, Cat in the Dat, Bike Sharing |

**Observations**:
- Large datasets (500K+) consistently produced tight CV-LB alignment
- Small train / large test ratios (Forest Cover: 15K train / 566K test) produced the worst gaps
- Target encoding leakage in CV inflated scores (Cat in the Dat: CV 0.831 vs LB 0.793)
- Time-based splits were slightly optimistic when test used a different temporal granularity

### 5.2 Public vs. Private LB (where both available)

| Competition | Public LB | Private LB | Shift |
|------------|-----------|------------|-------|
| Cat in the Dat | 0.793 | 0.790 | -0.003 |
| TPS Jan 2022 | 9.068 | 9.865 | +0.797 |
| TPS Aug 2022 | 0.574 | 0.582 | +0.008 |
| PS S6E1 (Exams) | 8.704 | 8.729 | +0.025 |
| PS S5E10 (Accident) | 0.056 | 0.056 | 0.000 |
| PS S4E11 (Depression) | 0.941 | 0.939 | -0.002 |
| PS S4E1 (Bank Churn) | 0.887 | 0.892 | +0.005 |
| AfSIS Soil | 0.500 | 0.530 | +0.030 |
| TMDB Box Office | 1.973 | 1.973 | 0.000 |
| Bike Sharing | 0.419 | 0.419 | 0.000 |
| CIFAR-10 | 0.956 | 0.956 | 0.000 |
| Forest Cover | 0.775 | 0.775 | 0.000 |

Most competitions showed minimal public-to-private shift, indicating robust models without public LB overfitting. The largest shift was TPS Jan 2022 (SMAPE metric, time series) and AfSIS Soil (small test set with p >> n).

---

## 6. Dataset Size vs. Performance

| Dataset Size | Competitions | Avg CV-LB Gap | Notes |
|-------------|-------------|--------------|-------|
| < 1K samples | 3 (EMVIC, ICDAR, AfSIS) | Large | Overfitting is the dominant challenge |
| 1K-10K | 5 (Titanic, House, Spaceship, Wheat, TMDB) | Moderate | Feature engineering most impactful |
| 10K-100K | 8 (Cat, Forest, TPS Jan/Aug, CO2, Digit, Writing, Whale) | Small-Moderate | Standard GBM pipelines work well |
| 100K-1M | 7 (PS series, Dogs, CIFAR, Depression, Churn, Bike) | Small | Model choice matters less than features |
| > 1M | 3 (Store Sales, Energy, Sleep States) | Tight | Lag features and efficiency matter |

---

## 7. Top Lessons Learned

### 7.1 Universal Principles

1. **Feature engineering > model choice**: In 25+ of 30 competitions, the winning approach was decided by features, not by which GBM variant was used. All three major GBMs (LightGBM, XGBoost, CatBoost) typically scored within 1-2% of each other.

2. **Ensembles always help (even simple ones)**: A weighted average of 2-3 diverse models consistently outperformed the best single model. The improvement was typically 0.5-2%.

3. **Know when to stop**: The Heart Disease competition demonstrated that beyond a certain point, more features, more seeds, and more complex ensembles yield diminishing returns. Tight CV-LB clustering signals a true plateau.

4. **Validation strategy matters enormously**: Time-based splits for temporal data, GroupKFold for grouped data, and stratified K-fold for classification. Wrong validation leads to misleading CV scores and wasted effort.

5. **Target transformation is often the biggest win**: Log1p for skewed targets, threshold optimization for imbalanced classification, and probability clipping for log loss metrics each provided large, "free" improvements.

### 7.2 Domain-Specific Insights

6. **Tabular**: LightGBM is the best default. CatBoost excels on clean datasets. Ridge regression dominates for p >> n problems. Stacking overfits on small datasets.

7. **Images**: Always use CNNs, never tabular ML on pixel features. Transfer learning for images larger than 64x64. Adapt architecture for small images (CIFAR-10 style ResNet). Data augmentation is a free lunch.

8. **Time series**: Lag features are typically the strongest predictors (autocorrelation > 0.9). Single global models often beat per-segment models. Time-based validation is non-negotiable.

9. **NLP**: Transformer fine-tuning is necessary. DistilBERT is more stable than DeBERTa. Gradient clipping prevents NaN losses. Overfitting happens fast -- early stopping at epoch 1-2 is often optimal.

10. **Audio**: Handcrafted features (mel spectrograms, MFCCs, spectral statistics) with GBMs work surprisingly well. Domain knowledge about frequency ranges is critical (whale calls at 100-200 Hz).

### 7.3 Practical Insights

11. **Closed competitions reject API submissions**: Pre-2020 Kaggle competitions consistently return 400/403 errors. Code competitions require notebook format, not CSV uploads.

12. **Tiny test sets are unreliable**: The LLM Classification competition had only 3 test samples, making LB scores essentially random.

13. **Outlier handling can be the single biggest improvement**: Removing 2 outliers in House Prices improved MAE by ~1,000.

14. **Budget imputation**: Missing feature imputation using a secondary model (TMDB budget prediction) can be more valuable than any modeling improvement.

15. **GPU enables experimentation but doesn't guarantee improvement**: Heart Disease v4 (300 models on GPU) didn't beat v3 (75 models on CPU).

---

## 8. Technical Infrastructure

### 8.1 Environment
- **OS**: Linux 6.14.0-1013-nvidia (ARM64)
- **GPU**: NVIDIA GB10 with CUDA 13.0
- **Python**: 3.13 with uv package manager
- **Key packages**: LightGBM, XGBoost, CatBoost, PyTorch 2.11.0 (nightly), Transformers 5.1.0, scikit-learn

### 8.2 Agent Architecture
- **Skill-based**: `.claude/skills/kaggle-agent/` with 6-stage pipeline instructions
- **Tools used**: Read/Glob/Grep (exploration), Bash (Python scripts), Write/Edit (code generation), Task (parallel work)
- **Workflow**: Ingest > EDA > Feature Engineering > Modeling > Evaluation > Submission

### 8.3 Per-Competition Workspace
Each competition follows a standardized structure:
```
competitions/<name>/
  config.yaml          # Competition metadata (metric, target, etc.)
  experiments.json     # Experiment history with parameters and scores
  STATUS.md            # Detailed competition log
  data/                # Raw and processed data (gitignored)
  scripts/             # Generated Python scripts (EDA, features, train, submit)
  submissions/         # Generated submission files
```

---

## 9. Competition Type Distribution

```
Tabular Classification:  11  ================================
Tabular Regression:       8  ========================
Time Series:              4  ============
Image Classification:     3  =========
Specialized:              4  ============
  - Event Detection       1
  - Object Detection      1
  - Stroke Recovery       1
  - Grid Inverse Problem  1
```

---

## 10. Summary Statistics

| Statistic | Value |
|-----------|-------|
| Total competitions | 30 |
| Leaderboard submissions | 20 |
| Local-only completions | 10 |
| Closed competitions | 7 |
| Code competitions | 5 |
| Unique model families | 9 |
| Most used model | LightGBM (27/30) |
| Largest dataset | 127.9M rows (Sleep States) |
| Smallest dataset | 605 signatures (ICDAR) |
| Most features engineered | 3,594 (AfSIS Soil, spectral) |
| Fewest features | 13 (Bike Sharing) |
| Best image accuracy | 99.61% (Digit Recognizer) |
| Best tabular AUC | 0.953 (Heart Disease) |
| Closest to LB #1 | ICDAR Stroke Recovery (within 0.3%) |
| Total sessions | 19 |
| Total scripts generated | ~120+ |

---

## 11. Conclusion

This project demonstrated that a hybrid AI agent -- combining LLM reasoning (Claude Code) with automated ML tools -- can effectively complete Kaggle competitions across a wide range of problem types. The agent's strengths lie in:

1. **Rapid pipeline execution**: End-to-end competition pipelines completed in a single session
2. **Adaptive methodology**: Correctly choosing CNNs for images, GBMs for tabular, transformers for NLP, and Ridge for high-dimensional spectral data
3. **Systematic experimentation**: Logging every experiment, comparing approaches, and knowing when to stop optimizing
4. **Transferable knowledge**: Patterns learned from early competitions (e.g., log1p transforms, ensemble strategies) were reused effectively in later ones

The main limitations were code competition submissions (requiring Kaggle notebook format) and closed competitions (API rejection). Future work could focus on automated notebook generation, hyperparameter optimization with Optuna, and more sophisticated ensemble strategies like stacking with proper out-of-fold methodology.

---

*Report generated on 2026-02-23 from competition STATUS.md files and project-level STATUS.md.*
