# Stage 2: Feature Engineering

## Objective
Create informative features that improve model performance, guided by EDA findings and domain reasoning.

## Steps

### 1. Review EDA Findings
- Re-read the EDA summary and feature engineering ideas from Stage 1
- Review `config.yaml` for problem type and target info
- Check if any features were flagged for removal (leakage, constant, duplicate)

### 2. Propose Feature Strategy
Present a feature engineering plan to the user before writing code. Include:

**Cleaning & Preprocessing:**
- Missing value imputation strategy per column (median, mode, flag, model-based)
- Outlier handling (cap, remove, transform)
- Encoding strategy for categoricals (label, one-hot, target, frequency)

**New Features:**
- **Aggregations**: Group-by statistics (mean, std, count, min, max) when entities have multiple records
- **Interactions**: Multiplication, division, or difference of related numerical features
- **Polynomial**: Squared or cubed terms for features with non-linear relationships
- **Binning**: Discretize continuous features that have step-function relationships with target
- **Date/time**: Day of week, month, year, is_weekend, days_since, cyclical encoding
- **Text** (if applicable): Length, word count, TF-IDF, embeddings
- **Domain-specific**: Features informed by the competition domain (ask user for input)

**Feature Selection (after creation):**
- Remove zero-variance features
- Remove highly correlated feature pairs (keep the more predictive one)
- Importance-based selection from a quick model

### 3. Get User Approval
Wait for user confirmation on the feature strategy before proceeding. The user may:
- Approve as-is
- Add domain-specific feature ideas
- Remove features they consider risky (potential leakage)
- Adjust preprocessing choices

### 4. Implement Feature Engineering
Write a feature engineering script that:
- Loads raw train and test data
- Applies all transformations consistently to both train and test
- Saves processed datasets (e.g., `train_processed.csv`, `test_processed.csv`)
- Reports the final feature count and any features that were dropped

**Critical rules:**
- **Fit on train, transform on both** — Any statistics (mean for imputation, encoder mappings) must be computed on train only and applied to test
- **No target leakage** — Never use test data or target variable in feature computation
- **Handle new categories** — Test set may have categories not seen in training; use a fallback strategy
- **Preserve ID and target columns** — Don't accidentally transform or drop them

Save the script to `competitions/<name>/scripts/feature_engineering.py`

### 5. Validate Features
After running the feature engineering script, check:
- **Shape check**: Train and test have the same columns (minus target)
- **NaN check**: No unexpected NaN values introduced
- **Dtype check**: All features are numeric (or appropriately encoded)
- **Leakage check**: No features have suspiciously high correlation with target
- **Scale check**: Report feature value ranges (some models are sensitive to scale)

### 6. Quick Feature Importance
Train a quick LightGBM/RandomForest model and report:
- Top 20 most important features
- Any features with zero importance (candidates for removal)
- Whether the new features rank higher than the originals

This gives early feedback on whether the feature engineering is helping.

### 7. Report Results
Present to the user:
- Number of features: original vs. after engineering
- Top features by importance
- Any issues found during validation
- Recommendation: proceed to modeling, or iterate on features

## Completion Criteria
- Feature engineering script exists and runs without errors
- Processed train and test datasets are saved
- Features have been validated (no NaN, no leakage, consistent shapes)
- Quick feature importance has been assessed
- User has reviewed and approved the features
