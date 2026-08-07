# Stage 1: Exploratory Data Analysis (EDA)

## Contents
1. [Load Data and Config](#1-load-data-and-config)
2. [Target Analysis](#2-target-analysis)
3. [Feature Analysis](#3-feature-analysis)
4. [Feature-Target Relationships](#4-feature-target-relationships)
5. [Train-Test Consistency](#5-train-test-consistency)
6. [Leakage Detection](#6-leakage-detection)
7. [Validation Strategy Recommendation](#7-validation-strategy-recommendation)
8. [EDA Report](#8-eda-report)

## Objective
Develop a deep understanding of the data to inform feature engineering and modeling decisions.

## Steps

### 1. Load Data and Config
- Read `config.yaml` for competition metadata
- Load train and test datasets
- Identify numerical vs. categorical vs. datetime vs. text columns

### 2. Target Analysis
- **Distribution**: Plot histogram/countplot of the target variable
- **Class balance** (classification): Report class frequencies and imbalance ratio
- **Outliers** (regression): Report percentiles, check for extreme values
- **Log transform candidate?** (regression): Check if target is skewed

### 3. Feature Analysis
For each feature category, analyze:

**Numerical features:**
- Summary statistics (mean, std, min, max, quartiles)
- Distribution shape (skewness, kurtosis)
- Outlier count (beyond 3 standard deviations or IQR method)
- Correlation with target and with other numerical features

**Categorical features:**
- Cardinality (number of unique values)
- Frequency distribution of top categories
- Rare category count (categories with < 1% frequency)
- Target mean/rate per category (to spot predictive categoricals)

**Missing values:**
- Pattern analysis — Are missing values random or systematic?
- Correlation between missingness and target
- Missing in train vs. test — Any columns missing in one but not the other?

### 4. Feature-Target Relationships
- **Numerical vs. target**: Correlation matrix, scatter plots for top correlated features
- **Categorical vs. target**: Grouped statistics (mean/median target per category)
- **Interaction effects**: Check if combinations of features are more predictive

### 5. Train-Test Consistency
- Compare distributions of features between train and test sets
- Flag any features with significant distribution shift (potential leakage or covariate shift)
- Check if test set has categories not seen in training

### 6. Leakage Detection
Watch for:
- Features that are too perfectly correlated with the target (suspiciously high AUC/correlation)
- Time-based features that reveal future information
- ID-like columns that shouldn't be predictive but are
- Features derived from the target

### 7. Validation Strategy Recommendation

Start from the dossier, not from a blank page. Stage 0.5 already derived a `split_policy`;
Stage 1's job is to *verify* it against the data and record the outcome.

**7a. Read the prior.** Open `competitions/<name>/dossier.json` and take `split_policy.scheme`,
`split_policy.forbidden`, and the `test_window.relation` hypothesis the policy rests on.

**7b. Test the prior against EDA findings.** Does an ordering/date column exist, and does the
test window really sit after train (id ranges, date ranges)? Are there repeated entities —
group/user/session ids, or the duplicate feature rows from the duplicate check — that must not
straddle folds? Is the target imbalanced enough to require stratification? Did any feature show
a train/test distribution shift?

Candidate schemes:
- **Standard K-Fold** — Default for i.i.d. tabular data
- **Stratified K-Fold** — For imbalanced classification
- **Group K-Fold** — When data has groups that shouldn't leak across folds (e.g., same user in train and validation)
- **Time-Series Split** — When data has a temporal component
- **Repeated K-Fold** — For small datasets where variance is high

**7c. Decide, and say which source won.** The dossier is a prior, not a conclusion: if the data
contradicts it, the EDA finding wins and that contradiction is itself a headline finding. If the
dossier said `"unknown"`, this step is where it gets resolved. The trap this exists to catch is a
shuffled KFold on a future-window test set — a controlled experiment measured a 4.5× optimism bias from the forbidden split (see the filtered prior library)
(see `references/00_problem_dossier.md`).

**7d. Write the verdict back into `dossier.json` under `"eda_verdict"`.** This is the step
`00_problem_dossier.md` ("Downstream consumption") makes Stage 1 responsible for — without it the
dossier's hypotheses are never marked tested and Stage 2 inherits an unverified prior. (2026-08-03
audit: this step previously stopped at "recommend a scheme", so the contract had no writer.)
Record for each dossier hypothesis — at minimum `test_window` and `external_data.needed` —
CONFIRMED / REFUTED / UNRESOLVED **with the number that decided it** (KS statistic, id-range
overlap, duplicate/conflict counts, imbalance ratio), plus a `validation_decision` field naming
the exact splitter call and seed, e.g.
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Every later stage reuses those folds
byte-identically so scores remain comparable.

### 8. EDA Report
Summarize findings in a structured format:
```
## EDA Summary: <competition-name>

### Data Overview
- Train: X rows, Y columns
- Test: X rows, Y columns
- Target: <type>, <distribution summary>

### Key Findings
1. <finding 1>
2. <finding 2>
...

### Concerns
- <concern 1>
- <concern 2>

### Feature Engineering Ideas
- <idea 1 — based on finding X>
- <idea 2 — based on finding Y>

### Recommended Validation Strategy
- <strategy> because <reason>
```

## Script Approach
- Generate an EDA Python script and save to `competitions/<name>/scripts/eda.py`
- Use bundled template from skill's `assets/templates/eda_template.py` as reference
- Save any generated plots to `competitions/<name>/scripts/plots/` (optional — only if user wants visuals)
- Print all findings to stdout so Claude Code can read and interpret them

## Completion Criteria
- EDA script has been run successfully
- Key findings have been summarized and presented to the user
- Validation strategy has been recommended and approved by the user
- `dossier.json` carries an `eda_verdict` with every hypothesis marked CONFIRMED / REFUTED /
  UNRESOLVED and a `validation_decision` (splitter + seed) — see step 7d
- Feature engineering ideas have been proposed
- User is ready to proceed to feature engineering
