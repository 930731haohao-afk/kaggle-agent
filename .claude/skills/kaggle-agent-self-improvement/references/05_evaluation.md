# Stage 4: Evaluation & Iteration

## Objective
Analyze model performance deeply, identify improvement opportunities, and execute targeted iterations.

## Steps

### 1. Review Experiment History
Load `experiments.json` and present:
- **Score timeline**: How scores have improved over iterations
- **Best model**: Current best model with its configuration
- **Score plateau**: Are recent iterations showing diminishing returns?

### 2. Error Analysis
For the best model, perform deep error analysis:

**Classification:**
- Confusion matrix — Which classes are confused with each other?
- Per-class precision/recall/F1
- Misclassification analysis — Sample misclassified instances and look for patterns
- Prediction confidence — Are errors high-confidence or low-confidence?
- Threshold optimization — For binary classification, is 0.5 the best threshold?

**Regression:**
- Residual analysis — Plot residuals vs. predicted, residuals vs. features
- Error distribution — Are errors normally distributed or skewed?
- Worst predictions — Which instances have the largest errors? Why?
- Segmented performance — Does the model perform worse on certain subsets?

### 3. Identify Improvement Opportunities
Based on error analysis, propose targeted improvements:

**Feature-based improvements:**
- Features that could help distinguish commonly confused classes
- Interactions between features that correlate with high-error segments
- Missing value patterns that coincide with errors

**Model-based improvements:**
- Underfit indicators → increase model complexity, add features
- Overfit indicators → regularization, feature selection, more data augmentation
- Specific model weaknesses → try a model that handles them better

**Data-based improvements:**
- Class rebalancing (SMOTE, class weights, oversampling)
- Outlier handling for high-error instances
- Data augmentation if applicable

**Post-processing:**
- Threshold tuning (classification)
- Prediction clipping to valid ranges (regression)
- Rounding to known discrete values if applicable
- Calibration (Platt scaling, isotonic regression)

### 4. Execute Iteration
For each proposed improvement:
1. **State the hypothesis** — "Adding feature X should help because..."
2. **Implement the change** — Modify the relevant script
3. **Run the experiment** — Train and evaluate with the same CV strategy
4. **Log the result** — Add to `experiments.json`
5. **Compare** — Did it improve, degrade, or have no effect?
6. **Decide** — Keep the change or revert

**Important**: Only change one thing at a time to understand what's driving improvements.

### 5. Convergence Check
After each iteration round, assess:
- **Improvement rate**: Are we still seeing meaningful gains?
- **Effort vs. reward**: Is the next improvement worth the compute/time?
- **Leaderboard context**: Where would our current score likely place us?

Recommend to the user:
- **Continue iterating** — If clear improvement opportunities remain
- **Move to ensembling** — If individual models have plateaued but diverse models exist
- **Prepare submission** — If satisfied with the score or diminishing returns

### 6. Iteration Log
After each iteration round, append a summary:
```
## Iteration Round N

### Hypothesis
<What we tried and why>

### Result
- Previous best: <score>
- New score: <score>
- Change: <+/- amount>

### Decision
<Keep / Revert / Modify further>

### Next Steps
<What to try next>
```

## Completion Criteria
- Error analysis has been performed on the best model
- At least one iteration has been executed based on the analysis
- All experiments are logged in `experiments.json`
- User has decided whether to continue iterating or proceed to submission
