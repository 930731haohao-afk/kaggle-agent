# Stage 4: Evaluation & Iteration

## Contents
1. [Review Experiment History](#1-review-experiment-history)
2. [Error Analysis](#2-error-analysis)
3. [Identify Improvement Opportunities](#3-identify-improvement-opportunities)
4. [Execute Iteration](#4-execute-iteration)
5. [Reflexion — Structured Self-Critique](#5-reflexion--structured-self-critique)
6. [Convergence Check — Verifiable Rewards](#6-convergence-check--verifiable-rewards)
7. [Adaptive Search — Break Through Plateaus](#7-adaptive-search--break-through-plateaus)
8. [Iteration Log](#8-iteration-log)

## Objective
Analyze model performance deeply, identify improvement opportunities, and execute targeted iterations. Uses **verifiable rewards** (CV scores) as ground truth and **reflexion** to avoid repeating mistakes.

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

### 5. Reflexion — Structured Self-Critique

After each experiment that fails to improve or produces unexpected results, generate a structured reflection:

```
## Reflection — Experiment [N]

### What was tried
[Approach description]

### Result
- CV score: [score] ([+/-] vs previous best)
- Expected: [what you predicted would happen]
- Actual: [what actually happened]

### Why did this succeed/fail?
[Root cause analysis — not just "it didn't work" but WHY]

### What assumptions were wrong?
[List assumptions that turned out to be incorrect]

### What should be tried differently?
[Concrete next actions, informed by this failure]
```

**Anti-patterns to catch through reflection:**
- "More features = better" — On clean/small datasets, more features often hurts
- "Stacking always helps" — On small datasets or p>>n, stacking overfits
- "Diminishing hyperparameter tuning" — After 2-3 rounds, further tuning rarely helps >0.1%
- "Ensembling too early" — Ensemble after individual models are strong, not as a substitute for good features

### 6. Convergence Check — Verifiable Rewards

After each experiment, compute verifiable reward signals:

| Signal | Threshold | Action |
|--------|-----------|--------|
| Score improves | `reward > 0` | Keep changes. Log as new best. Continue direction. |
| Score flat | `abs(reward) < 0.001` | Change approach. Don't repeat same strategy with minor tweaks. |
| Score degrades | `reward < -0.005` | Revert changes. Reflect (Step 5). Try different direction. |
| Stuck 3+ experiments | No improvement in last 3 | **Trigger Adaptive Search** (Step 7). |
| Plateau confirmed | Last 5 within 0.002 range | **Stop iterating.** Move to submission or fundamentally different approach. |

After each iteration round, assess:
- **Improvement rate**: Are we still seeing meaningful gains?
- **Effort vs. reward**: Is the next improvement worth the compute/time?
- **Leaderboard context**: Where would our current score likely place us?

Recommend to the user:
- **Continue iterating** — If clear improvement opportunities remain
- **Move to ensembling** — If individual models have plateaued but diverse models exist
- **Trigger adaptive search** — If stuck for 3+ experiments (see Step 7)
- **Prepare submission** — If plateau confirmed or diminishing returns

### 7. Adaptive Search — Break Through Plateaus

When the verifiable rewards system signals "stuck" (no improvement in 3+ experiments), search for new ideas:

**Search sources (in priority order):**
1. **Past competition experience** — Re-read MEMORY.md and similar competition STATUS.md files with fresh eyes
2. **Kaggle Competition Discussion board** — Competition-specific tips, shared notebooks, feature ideas, and gotchas from other participants. Access via Kaggle API or WebSearch (`site:kaggle.com/competitions/<name>/discussion`). Focus on highly-upvoted posts.
3. **Kaggle API** — Competition metadata and leaderboard context
4. **Context7 / library docs** — Documentation for unfamiliar tools or techniques
5. **WebSearch/WebFetch** — Winning solutions from past similar competitions, blog posts

**Search protocol:**
1. Formulate specific queries (not generic "how to improve score")
2. Extract actionable techniques from results
3. Evaluate any found technique with the same CV strategy (verifiable reward)
4. Reflect on results if the technique doesn't help

### 8. Iteration Log
After each iteration round, append a summary:
```
## Iteration Round N

### Hypothesis
<What we tried and why>

### Result
- Previous best: <score>
- New score: <score>
- Change: <+/- amount>
- Reward signal: <improvement / flat / degradation / plateau>

### Reflection (if score didn't improve)
<Why it failed, what assumptions were wrong>

### Decision
<Keep / Revert / Modify further / Trigger search>

### Next Steps
<What to try next, informed by reflection>
```

## Completion Criteria
- Error analysis has been performed on the best model
- At least one iteration has been executed based on the analysis
- Reflexion has been applied after any failed experiments
- Verifiable rewards signals have been tracked (improvement/flat/degradation/plateau)
- If stuck, adaptive search has been attempted
- All experiments are logged in `experiments.json`
- User has decided whether to continue iterating or proceed to submission
