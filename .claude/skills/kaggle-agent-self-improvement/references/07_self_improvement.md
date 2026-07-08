# Self-Improvement Strategies

## Contents
1. [Overview](#overview)
2. [Strategy 1: Experience Library Lookup](#strategy-1-experience-library-lookup)
3. [Strategy 2: Verifiable Rewards](#strategy-2-verifiable-rewards)
4. [Strategy 3: Reflexion](#strategy-3-reflexion)
5. [Strategy 4: Best-of-N Candidate Generation](#strategy-4-best-of-n-candidate-generation)
6. [Strategy 5: Adaptive Search](#strategy-5-adaptive-search)
7. [Autonomous Iteration Protocol](#autonomous-iteration-protocol)
8. [Memory Updates](#memory-updates)

## Overview

These are **tools in your toolkit**, not mandatory steps. You decide when each strategy adds value based on the situation. The guiding principle: use a strategy when the cost of applying it is low relative to the risk of skipping it.

The core self-improvement loop, when fully engaged:

```
Generate Candidates → Evaluate (CV Score) → Reflect → Search if Stuck → Update Memory → Iterate
```

But you may use any subset of these strategies, or none, depending on the situation.

**Key principle**: Always be transparent. When you decide to use or skip a strategy, briefly state why in your reasoning to the user.

---

## Strategy 1: Experience Library Lookup

**Ask yourself**: *"Have I seen a competition like this before? Could past experience save me time or prevent mistakes?"*

**Use when**: The competition type, metric, or data characteristics overlap with past work, OR the problem domain is unfamiliar and you want to check if there are lessons to transfer.

**Skip when**: The competition is a straightforward repeat of a well-known pattern (e.g., another basic tabular binary classification) and you're confident in the approach.

### How to Execute

1. **Read MEMORY.md** — Check the "Completed Competitions" and "Patterns Learned" sections for relevant experience.

2. **Scan past experiments** — Look at `competitions/*/experiments.json` and `competitions/*/STATUS.md` for competitions with similar characteristics:
   - Same problem type (regression, binary classification, multiclass, time series, image, NLP)
   - Same or similar metric (RMSE, AUC, log loss, accuracy, MAE)
   - Similar data size range (small <5K, medium 5K-100K, large >100K)
   - Similar feature types (numerical, categorical, text, image, time series, spectral)

3. **Extract transferable strategies** — From the most similar competitions, note:
   - Which models worked best (and which failed)
   - Which feature engineering techniques helped most
   - What validation strategy was used
   - Any tricks or gotchas discovered

4. **Present recommendations** — Before proposing a strategy, tell the user:
   ```
   Based on past experience with similar competitions:
   - [Competition X] (same type, similar metric): [Model Y] with [Feature Z] achieved [Score]
   - Recommended starting approach: [approach]
   - Known pitfalls to avoid: [pitfalls]
   ```

### Similarity Matching Rules
- **Exact type match** (regression → regression) is strongest signal
- **Metric match** is second strongest (RMSE competitions share optimization tricks)
- **Data size** matters for model selection (small → Ridge/regularized, large → GBMs/NNs)
- **Feature type** matters for engineering (time series → lags, categorical-heavy → target encoding)

---

## Strategy 2: Verifiable Rewards

**Ask yourself**: *"Am I tracking whether I'm actually improving, or just trying things?"*

**Use when**: Always. This is the one strategy that should be applied after every experiment — it's lightweight (just computing deltas) and prevents wasted effort.

**What**: Use CV scores to objectively evaluate progress, detect plateaus, and inform decisions.

### Score Tracking Protocol

After each experiment, compute and report:

1. **Absolute score**: The CV score for this experiment
2. **Improvement over baseline**: `current_score - baseline_score`
3. **Improvement over previous best**: `current_score - best_score`
4. **Trend**: Are the last 3 experiments improving, flat, or declining?

### Decision Rules

| Situation | Signal | Action |
|-----------|--------|--------|
| First experiment | Any score | Set as baseline. Continue. |
| Score improves | `reward > 0` | Keep changes. Log as new best. Continue direction. |
| Score flat | `abs(reward) < 0.001` | Change approach. Don't repeat same strategy with minor tweaks. |
| Score degrades | `reward < -0.005` | Revert changes. Reflect on why. Try different direction. |
| Stuck 3+ experiments | No improvement in last 3 | **Trigger Adaptive Search** (Strategy 5). |
| Plateau confirmed | Last 5 experiments within 0.002 of each other | **Stop iterating**. Move to submission or ensemble. |

### Plateau Detection

A plateau is confirmed when:
- At least 5 experiments have been run
- The score range of the last 5 experiments is < 0.002 (for normalized metrics) or < 0.5% relative improvement
- Multiple different strategies have been tried (not just hyperparameter tweaks)

When plateau is detected, report:
```
Plateau detected: Last 5 scores [0.856, 0.855, 0.857, 0.856, 0.855] (range: 0.002)
Strategies tried: feature engineering, model tuning, ensemble variations
Recommendation: Stop iterating. Current best (0.857) is likely near optimal for this approach.
Options: (1) Submit best, (2) Try fundamentally different approach (e.g., NN instead of GBM)
```

---

## Strategy 3: Reflexion

**Ask yourself**: *"Did something unexpected happen? Am I at risk of repeating the same mistake?"*

**Use when**: A result surprises you (positive or negative), you've made a similar mistake before, or you're about to try something that resembles a past failure.

**Skip when**: The result was expected and you already understand why. A brief note ("improved as expected, feature X was the driver") is enough.

**What**: Generate a structured self-critique to build understanding and avoid repeating mistakes.

### Reflection Template

After each experiment, write a reflection entry (internally, as part of reasoning):

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

### Lesson for future competitions
[If this is a generalizable lesson, note it for MEMORY.md update]
```

### Reflection Triggers

| Trigger | Reflection Depth |
|---------|-----------------|
| Score improved significantly (>1%) | Brief — note what worked and why |
| Score flat or marginal (<0.5%) | Medium — analyze if the approach was sound but underpowered |
| Score degraded | Deep — full root cause analysis, revert decision |
| Unexpected result (score very different from expectation) | Deep — check for bugs, leakage, data issues |

### Anti-Patterns to Catch

Through reflection, watch for and call out these recurring mistakes:
- **More features = better**: On clean/small datasets, more features often hurts (Heart Disease: 33 > 58 features)
- **Stacking always helps**: On small datasets or p>>n, stacking overfits (AfSIS: CV 0.40 → LB 0.55)
- **Diminishing hyperparameter tuning**: After 2-3 rounds, further tuning rarely helps more than 0.1%
- **Ignoring validation strategy**: A poor CV scheme makes all experiment comparisons unreliable
- **Ensembling too early**: Ensemble after individual models are strong, not as a substitute for good features

---

## Strategy 4: Best-of-N Candidate Generation

**Ask yourself**: *"Am I confident this is the right approach, or should I hedge my bets?"*

**Use when**: Starting a new competition with no strong prior, stuck and need to pivot, or the user asks to "try something different." The overhead of running 3-5 models is low compared to the risk of going deep on the wrong approach.

**Skip when**: Experience library or prior intuition gives strong signal for one approach, or compute budget is tight and you need to be surgical.

**What**: Instead of trying one approach at a time, generate N diverse candidates, evaluate all, then expand the best.

### How to Execute

1. **Generate 3-5 diverse candidates** based on:
   - Experience library suggestions (Strategy 1)
   - Model diversity (at least 2 different model families)
   - Feature diversity (at least 2 different feature sets)

2. **Example candidate set for a tabular regression competition:**
   ```
   Candidate 1: LightGBM + basic features (from experience library)
   Candidate 2: XGBoost + basic features (model diversity)
   Candidate 3: LightGBM + advanced features (feature diversity)
   Candidate 4: Ridge/ElasticNet + basic features (linear model diversity)
   Candidate 5: CatBoost + basic features (handles categoricals natively)
   ```

3. **Evaluate all candidates** with the same CV strategy

4. **Report ranked results:**
   ```
   Candidate Results (5-fold CV):
   1. CatBoost + basic features:     CV = 0.823  ← Best
   2. LightGBM + advanced features:  CV = 0.819
   3. LightGBM + basic features:     CV = 0.815
   4. XGBoost + basic features:      CV = 0.812
   5. Ridge + basic features:        CV = 0.790

   Selected: CatBoost for further iteration. LightGBM + advanced features for ensemble diversity.
   ```

5. **Expand the best**: Take the top 1-2 candidates and iterate (tune hyperparameters, add features, etc.)

### Diversity Requirements

For the candidates to be useful, ensure diversity along at least 2 of these axes:
- **Model family**: tree-based, linear, neural network
- **Feature set**: basic, advanced, selected subset
- **Regularization**: light vs. heavy regularization
- **Preprocessing**: raw vs. scaled vs. transformed target

### When to Use Best-of-N Again

Re-run candidate generation when:
- Stuck for 3+ experiments (combine with Adaptive Search results)
- Switching to a fundamentally different approach
- User requests "try something completely different"

---

## Strategy 5: Adaptive Search

**Ask yourself**: *"Have I run out of ideas from my own reasoning? Would external input help?"*

**Use when**: Stuck for 3+ experiments with no improvement, facing a completely unfamiliar problem domain, or the user asks "what else could we try?" and you genuinely don't know.

**Skip when**: You still have untried ideas from your own reasoning, experience library, or reflexion insights. Searching is slower than thinking — exhaust internal ideas first.

**What**: Search external sources for new ideas when internal strategies are exhausted.

### Search Sources (in priority order)

1. **Past competition experience** (Strategy 1) — re-examine with different lens
2. **Kaggle Competition Discussion board** — The most valuable external source. Competition-specific tips, shared notebooks, feature engineering ideas, and gotchas from other participants. Access via Kaggle API (`kaggle competitions list -s <name>`) or WebSearch (`site:kaggle.com/competitions/<name>/discussion`). Focus on highly-upvoted posts and posts by medal-winning authors.
3. **Kaggle API** — Competition metadata, leaderboard context
4. **Context7 / documentation** — Library docs for unfamiliar tools or techniques
5. **WebSearch/WebFetch** — Winning solutions from past similar competitions, blog posts, research papers

### Search Triggers

| Trigger | Search Action |
|---------|--------------|
| No improvement in 3 experiments | Search for feature engineering ideas for this competition type |
| New competition type never seen before | Search for winning solutions from similar past competitions |
| Large CV-LB gap | Search for validation strategy advice for this data type |
| User asks "what should I try next?" | Comprehensive search across all sources |

### How to Search Effectively

1. **Formulate specific queries** (not generic):
   - Bad: "how to improve kaggle score"
   - Good: "feature engineering for time series energy forecasting competition"
   - Good: "winning solution tabular binary classification imbalanced dataset"

2. **Extract actionable techniques** from search results:
   - Specific feature engineering ideas
   - Model architectures or hyperparameter ranges
   - Validation strategies
   - Post-processing tricks

3. **Validate before applying** — Any technique found via search should be:
   - Evaluated with the same CV strategy (verifiable reward)
   - Compared against current best
   - Reflected on if it doesn't help (Strategy 3)

### Search Limitations

- Kaggle discussion pages are JS-rendered and may not load with WebFetch — use API instead
- Filter search results by quality (upvotes, medal status of author)
- Don't blindly copy solutions — adapt techniques to the current competition's specifics

---

## Autonomous Iteration Protocol

**When**: User invokes the skill and doesn't specify a particular stage, or explicitly asks for autonomous iteration.

**What**: Run 3-5 improvement iterations autonomously before reporting back, using all strategies together.

### Loop Structure

```
For each iteration (max 5):
  1. Generate candidates (Best-of-N) or iterate on current best
  2. Evaluate with CV (Verifiable Rewards)
  3. Reflect on results (Reflexion)
  4. If stuck → Search for new ideas (Adaptive Search)
  5. If improved → Update experience (Experience Library)
  6. If plateau → Stop and report
```

### Stopping Criteria

Stop the autonomous loop when ANY of these are met:
- **Plateau**: No improvement in last 3 iterations
- **Max iterations**: 5 iterations reached
- **Significant improvement**: Found an approach that improves >5% over baseline (report this win immediately)
- **Error or uncertainty**: Something unexpected happens that needs user judgment

### Reporting

After the autonomous loop, present a concise summary:
```
Autonomous Iteration Summary (N iterations):

| # | Approach | CV Score | Delta | Status |
|---|----------|----------|-------|--------|
| 1 | LGB baseline | 0.810 | — | Baseline |
| 2 | + target encoding | 0.823 | +0.013 | Improvement |
| 3 | + interactions | 0.828 | +0.005 | Improvement |
| 4 | CatBoost ensemble | 0.831 | +0.003 | Best |
| 5 | + polynomial features | 0.829 | -0.002 | Reverted |

Best: CatBoost ensemble (CV 0.831, +2.6% over baseline)
Recommendation: [Submit / Continue with specific idea / Try fundamentally different approach]
```

### User Override

The user can always:
- Interrupt the loop ("stop", "wait", "let me review")
- Set a different max iteration count
- Specify which strategies to use or skip
- Provide domain knowledge that changes the direction

---

## Memory Updates

After a competition is complete (or a significant insight is confirmed), update MEMORY.md with:
- Competition name, type, and final score
- Key patterns that worked (or didn't)
- Any new lessons that generalize to future competitions
- Only store **verified** insights (confirmed by CV and/or LB scores)

This closes the learning loop: experience from this competition feeds the Experience Library for future ones.
