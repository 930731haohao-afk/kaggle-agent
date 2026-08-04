# Stage 2: Feature Engineering

## Objective
Create informative features that improve model performance, guided by EDA findings and domain reasoning.

## Steps

### 1. Review EDA Findings
- Re-read the EDA summary and feature engineering ideas from Stage 1
- Review `config.yaml` for problem type and target info
- Check if any features were flagged for removal (leakage, constant, duplicate)

### 1.5 Dispatch the dossier's injection ideas  (MANDATORY — do this before proposing anything)

Stage 0.5 wrote `competitions/<name>/dossier.json`. Its `injection_ideas` are the **executable**
channel between the judgment layer and this stage; the EDA report's free-text "feature ideas"
bullets are not. This section exists because it was missing: `00_problem_dossier.md` declared
Stage 2 the consumer of `injection_ideas`, this file never mentioned them, and 28 of 32 dossiers
in the repo have no `injection_ledger.json` beneath them — every idea silently dropped, on the
very competition (s3e19) the contract was created for (2026-08-04 architecture gate).

1. **Conform first.** An idea that does not typecheck cannot be dispatched:

   ```bash
   python3 external_data/validate_dossier.py competitions/<name>/dossier.json
   ```

   Non-zero exit means fix the dossier before continuing — do not hand-wave past it. Operator
   names come from `knowledge/injection_operators.md` and nowhere else; inventing a name is how
   an idea becomes unexecutable, and 234 of 282 recorded ideas across this repo do not conform
   while every one of those runs completed anyway.

2. **Dispatch.** Realize the ideas through the typed dispatcher, never by hand:

   ```python
   from external_data.apply import apply_operators
   train, test, plan = apply_operators(
       train, test, dossier["injection_ideas"], target_col=<target>,
       ledger_path="competitions/<name>/injection_ledger.json",
       rules_verdict="competitions/<name>/rules_verdict.json")   # written by Stage 0.5's gate
   ```

3. **Read the ledger back and report it.** Every idea is either in `plan["realized"]` or in
   `plan["unrealized"]` with a reason. State both counts in your Stage 2 report. An idea that
   appears in neither is the failure this whole contract exists to prevent — say so loudly
   rather than proceeding.

4. **Operators that change the data or the target** (`ratio_target`, `log_offset`,
   `join_feature`) become raced lanes at small budget rather than features you fold in here —
   see `references/07_tree_search.md` §v5 arms. Config-only operators (`objective`,
   `blend_member`) land in `plan["node_configs"]` for the search driver to seed; do not apply
   them yourself.

If `dossier.json` is absent, say so explicitly in your report and continue — but a missing
dossier on a benchmark run means Stage 0.5 did not run, which is itself worth stopping for.

### 2. Propose Feature Strategy
Present a feature engineering plan to the user before writing code. Include:

**Cleaning & Preprocessing:**
- Missing value imputation strategy per column (median, mode, flag, model-based)
- Outlier handling (cap, remove, transform)
- Encoding strategy for categoricals (label, one-hot, target, frequency) — **but see the
  target-encoding rule under "Critical rules" below before choosing target encoding**

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
- **Target encoding is FOLD-BOUND, and this stage cannot do it.** Computing an encoder from the
  whole training target and then training on those columns leaks the target into every fold: the
  encoder saw each row's own label. This project measured that trap directly (see
  `knowledge/injection_operators.md`: target encoding "REQUIRES fold-aligned computation"), and
  this file used to license the whole-train form under a generic "fit on train" rule. Stage 2
  runs before the fold definition exists, so it has no way to do it correctly. Therefore:
  **emit target encoding as an `encoding` operator with `scheme: "target"` for the evaluator to
  compute inside each fold — do not materialize target-encoded columns in
  `train_processed.csv`.** The target-free schemes (`count`, `ordinal`, `crosses`, GBDT
  `native`) have no leakage path and may be materialized here.
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

- [ ] `injection_ledger.json` exists and every dossier idea appears in `realized` or
      `unrealized` with a reason (Stage 2 is not complete without it)
- [ ] `validate_dossier.py` exits zero on this competition's dossier
- [ ] No target-encoded column was materialized into the processed files
- [ ] If any external-data operator ran, `rules_verdict.json` exists and reads `permitted` —
      `apply_operators` refuses to fetch without it, and absence of a verdict is not permission
- Feature engineering script exists and runs without errors
- Processed train and test datasets are saved
- Features have been validated (no NaN, no leakage, consistent shapes)
- Quick feature importance has been assessed
- User has reviewed and approved the features
