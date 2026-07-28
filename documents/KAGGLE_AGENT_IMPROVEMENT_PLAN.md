# Kaggle Agent Self-Improvement Plan

**Date**: 2026-02-12
**Based on**: AI Self-Improvement discussion (11022026)
**Current Status**: Basic 6-stage pipeline with manual iteration

---

## Executive Summary

Transform the Kaggle agent from a **guided assistant** to a **self-improving autonomous agent** by implementing:
1. **Verifiable Rewards** (CV scores, LB scores as ground truth)
2. **Experience Library** (learn from past competitions)
3. **Adaptive Retrieval** (search Kaggle discussions, papers, winning solutions)
4. **Reflexion** (structured self-critique after each iteration)
5. **Best-of-N Sampling** (explore multiple approaches in parallel)

---

## Part 1: Core Self-Improvement Loop

### Current State
```
User → Request Stage → Agent Executes → Reports Results → User Decides Next → Repeat
```

### Target State
```
Agent → Generate N Candidates → Verify (CV Score) → Reflect on Results →
    → Search if Stuck → Select Best → Update Memory → Auto-iterate
```

---

## Improvement 1: Implement Verifiable Rewards System

### What
Make CV scores and LB scores the **primary decision signal**, not just reporting metrics.

### How
**File**: `utils/verifiable_rewards.py`

```python
class VerifiableRewardSystem:
    def __init__(self, competition_dir):
        self.baseline_score = None
        self.best_score = None
        self.score_history = []

    def evaluate(self, experiment_result):
        """
        Returns:
        - reward: float (improvement over baseline)
        - is_improvement: bool
        - should_continue: bool (if stuck, trigger search)
        """
        cv_score = experiment_result['cv_score']

        if self.baseline_score is None:
            self.baseline_score = cv_score
            return 0.0, True, False

        reward = cv_score - self.best_score if self.best_score else cv_score - self.baseline_score
        is_improvement = reward > 0

        # Stuck detection: no improvement in last 3 tries
        recent_scores = self.score_history[-3:]
        should_search = len(recent_scores) == 3 and all(s <= self.best_score for s in recent_scores)

        self.score_history.append(cv_score)
        if is_improvement:
            self.best_score = cv_score

        return reward, is_improvement, should_search
```

**Integration**: After each model training, call `evaluate()` to get structured feedback.

### Expected Impact
- Agent knows objectively if it's improving (±5% accuracy in decision making)
- Automatic detection of when stuck → triggers search

---

## Improvement 2: Experience Library (Cross-Competition Learning)

### What
Build a **searchable knowledge base** of successful strategies across all competitions.

### How
**File**: `utils/experience_library.py`

```python
class ExperienceLibrary:
    """
    Stores: {competition_type, task_type, data_characteristics} → {successful_features, best_models, tricks}
    """
    def __init__(self, db_path="competitions/experience_library.json"):
        self.db = self.load_db(db_path)

    def store_success(self, competition_info, approach, cv_score, lb_score=None):
        """Store when CV > threshold or LB improves significantly"""
        if cv_score < self.get_threshold(competition_info['metric']):
            return  # Don't store mediocre results

        entry = {
            "competition": competition_info['name'],
            "task_type": competition_info['task_type'],
            "metric": competition_info['metric'],
            "data_size": competition_info['data_size'],
            "features": approach['features'],
            "models": approach['models'],
            "tricks": approach['tricks'],
            "cv_score": cv_score,
            "lb_score": lb_score,
            "timestamp": datetime.now().isoformat()
        }
        self.db.append(entry)
        self.save_db()

    def retrieve_similar(self, competition_info, top_k=5):
        """RAG: Find most similar past competitions and their successful approaches"""
        # Similarity: task_type match, metric match, data size range
        similar = []
        for entry in self.db:
            similarity = self.compute_similarity(competition_info, entry)
            similar.append((similarity, entry))

        similar.sort(reverse=True, key=lambda x: x[0])
        return [entry for _, entry in similar[:top_k]]
```

**Integration Point**: Stage 0 (Setup) — After reading competition description, query library for similar competitions.

**Example Output**:
```
Found 3 similar competitions:
1. House Prices (regression, RMSLE) → log1p transform + GBR with Huber loss worked well
2. Store Sales (time series, RMSLE) → Year-based CV + lag features critical
3. Home Data (regression, MAE) → Outlier removal was biggest improvement (-1000 MAE)

Recommended starting strategies:
- Apply log1p to skewed target
- Use GBR with Huber loss for robustness
- Check for outliers in training data
```

### Expected Impact
- 30-50% faster to reach competitive baseline (reuse proven strategies)
- Avoid repeating failed experiments
- Automatic transfer learning across competitions

---

## Improvement 3: Adaptive Retrieval (Search When Stuck)

### What
When CV scores plateau or approach fails, **automatically search** Kaggle discussions, papers, and winning solutions for new ideas.

### How
**File**: `utils/adaptive_search.py`

Implement **ReAct-style loop**:
```
Thought: Current approach (LGB + basic features) → CV = 0.78, stuck for 3 iterations
Thought: Need new ideas. Search for similar competition winning solutions.
Action: search_kaggle_discussions(competition_name, query="feature engineering")
Observation: Top discussion mentions "target encoding + interaction features"
Thought: Haven't tried target encoding yet. This is promising.
Action: implement_target_encoding()
```

**Search Strategies**:
1. **Kaggle Discussions** (via Kaggle API or web scraping)
   - Query: "feature engineering", "model selection", "tricks"
   - Filter by upvotes and gold medal authors
2. **Past Competition Solutions** (Kaggle winners' notebooks)
   - Search: `competition:<similar_competition> tag:gold-medal`
3. **ArXiv Papers** (if cutting-edge techniques needed)
   - Query: "time series forecasting 2026", "graph neural networks tabular"
4. **Stack Overflow / GitHub Issues** (for implementation details)

**Trigger Conditions**:
- No CV improvement in last 3 experiments
- New competition type not seen before
- User explicitly asks "what should I try next?"

**Implementation**:
```python
class AdaptiveSearchAgent:
    def should_search(self, score_history, competition_type):
        # Stuck: no improvement in last 3
        if len(score_history) >= 3:
            recent_scores = score_history[-3:]
            if recent_scores[-1] <= min(recent_scores):
                return True, "stuck"

        # New competition type: proactive search
        if competition_type not in self.experience_library.get_known_types():
            return True, "new_domain"

        return False, None

    def search_and_extract(self, competition_info, query_type):
        if query_type == "feature_engineering":
            results = self.search_kaggle_discussions(competition_info['name'], "feature engineering")
            techniques = self.extract_techniques(results)
            return techniques
        elif query_type == "winning_solutions":
            results = self.search_similar_competitions(competition_info)
            strategies = self.extract_strategies(results)
            return strategies
```

### Expected Impact
- Break through plateaus 60-80% of the time
- Discover techniques not in agent's pre-training (e.g., new libraries, 2025+ papers)
- Mimic how expert Kagglers work (read discussions → try new ideas)

---

## Improvement 4: Reflexion (Structured Self-Critique)

### What
After each failed or low-improvement experiment, generate a **structured reflection** that guides future iterations.

### How
**File**: `utils/reflexion.py`

```python
class ReflexionEngine:
    def reflect(self, experiment, score_change, competition_context):
        """
        Generate structured reflection after experiment
        Returns: {critique, hypothesis, next_actions}
        """
        prompt = f"""
You just tried: {experiment['approach']}
CV score change: {score_change:.4f} (baseline: {competition_context['baseline']}, current: {experiment['cv_score']})

Reflect on:
1. Why did this approach succeed/fail?
2. What assumptions might be wrong?
3. What should we try differently next time?

Competition context:
- Metric: {competition_context['metric']}
- Data characteristics: {competition_context['data_summary']}
- Past successful approaches: {competition_context['experience_library_suggestions']}
"""

        reflection = self.llm.generate(prompt)  # LLM-as-judge for qualitative analysis

        # Store reflection in memory
        self.memory.append({
            "experiment_id": experiment['id'],
            "reflection": reflection,
            "timestamp": datetime.now().isoformat()
        })

        return reflection

    def get_relevant_reflections(self, current_situation):
        """Retrieve past reflections similar to current situation"""
        # Simple keyword match or embedding-based retrieval
        relevant = [r for r in self.memory if self.is_relevant(r, current_situation)]
        return relevant[:3]  # Top 3 most relevant
```

**Integration**: Stage 4 (Evaluation) — After reviewing experiment results, generate reflection before proposing next steps.

**Example Reflection**:
```
Experiment 3: Added polynomial features → CV decreased by 0.02

Reflection:
1. Polynomial features likely causing overfitting (model complexity increased 5x)
2. Assumption "more features = better" is wrong for this small dataset (5K rows)
3. Next time: Focus on feature selection, not feature generation
4. Alternative: Try regularization (Ridge, Lasso) or simpler models (LR instead of XGB)

Stored in memory for future reference.
```

### Expected Impact
- Avoid repeating same mistakes (10-20% fewer wasted experiments)
- Build intuition across experiments
- Transparent reasoning (user can review agent's thought process)

---

## Improvement 5: Best-of-N Sampling (Parallel Exploration)

### What
Instead of trying one approach at a time, **generate N candidate strategies in parallel**, evaluate all, select best.

### How
**File**: `utils/best_of_n.py`

```python
class BestOfNSampler:
    def generate_candidates(self, competition_context, n=5):
        """
        Generate N diverse approaches based on:
        - Experience library suggestions
        - Baseline variations
        - Search results (if available)
        """
        candidates = []

        # Candidate 1: Experience library best match
        candidates.append(self.from_experience_library(competition_context))

        # Candidate 2-3: Model diversity (LGB, XGB, CatBoost)
        candidates.append({"model": "lightgbm", "features": "basic"})
        candidates.append({"model": "xgboost", "features": "basic"})

        # Candidate 4: Feature engineering focus
        candidates.append({"model": "lightgbm", "features": "advanced_interactions"})

        # Candidate 5: Ensemble
        candidates.append({"model": "ensemble_lgb_xgb", "features": "basic"})

        return candidates

    def evaluate_all(self, candidates, competition_dir):
        """Run all candidates in parallel, return ranked by CV score"""
        results = []
        for i, candidate in enumerate(candidates):
            score = self.run_experiment(candidate, competition_dir)
            results.append((score, candidate, i))

        results.sort(reverse=True, key=lambda x: x[0])  # Best first
        return results
```

**Integration**: Stage 3 (Modeling) — Instead of sequential experiments, generate 5 candidates and run in parallel.

**Example**:
```
Generated 5 candidate approaches:
1. LightGBM + target encoding (from similar competition)
2. XGBoost + basic features (baseline)
3. CatBoost + basic features (model diversity)
4. LightGBM + polynomial interactions (feature focus)
5. Ensemble (LGB 70% + XGB 30%)

Running all in parallel (estimated 15 minutes)...

Results:
1. Candidate 1 (LGB + target encoding): CV = 0.823 ← Best
2. Candidate 5 (Ensemble): CV = 0.819
3. Candidate 2 (XGB baseline): CV = 0.810
4. Candidate 3 (CatBoost): CV = 0.807
5. Candidate 4 (Polynomial): CV = 0.795

Selected: LGB + target encoding for next iteration.
```

### Expected Impact
- 50-70% faster to find good approach (parallel vs sequential)
- Discover surprising combinations that single-path wouldn't try
- More robust (won't get stuck in local optimum)

---

## Improvement 6: Tree Search for Strategy Exploration

### What
Model the solution space as a **search tree** and systematically explore promising branches.

### How
**File**: `utils/strategy_tree.py`

```
Root: Competition X
├── Branch A: LightGBM baseline
│   ├── A1: + target encoding → CV 0.82 ✓ Expand
│   │   ├── A1.1: + interactions → CV 0.84 ✓✓ Best so far
│   │   └── A1.2: + PCA → CV 0.81 ✗ Prune
│   └── A2: + one-hot encoding → CV 0.78 ✗ Prune
├── Branch B: XGBoost baseline
│   └── B1: + basic features → CV 0.81 ✓ Expand
└── Branch C: Neural Network
    └── C1: + embeddings → CV 0.79 → Deprioritize (tree models better)
```

**Pruning Logic**:
- Prune if CV < 95% of current best
- Expand if CV in top 30% of all tried branches
- Prioritize branches with consistent improvement

**Implementation**:
```python
class StrategyTree:
    def __init__(self, root_config):
        self.root = Node(config=root_config, cv_score=None)
        self.best_node = None

    def expand(self, node):
        """Generate child strategies from this node"""
        children = []

        # Strategy mutations:
        # 1. Add feature engineering
        # 2. Change model
        # 3. Tune hyperparameters
        # 4. Add ensembling

        for mutation in self.get_mutations(node.config):
            child = Node(config=mutation, parent=node)
            children.append(child)

        return children

    def select_next(self):
        """UCB-style selection: balance exploitation and exploration"""
        # Prioritize: high CV + not fully explored + diverse from current best
        candidates = self.get_unexplored_nodes()
        scores = [self.ucb_score(n) for n in candidates]
        return candidates[np.argmax(scores)]
```

### Expected Impact
- Systematic exploration (won't miss obvious combinations)
- Automatic backtracking when stuck
- Visual tree can be shown to user for transparency

---

## Improvement 7: Autonomous Iteration Loop

### What
Agent autonomously runs **3-5 iterations** before asking for user input, instead of stopping after each step.

### How
**New Mode**: `--autonomous` flag in skill invocation

```python
def autonomous_mode(competition_dir, max_iterations=5, time_budget_minutes=60):
    """
    Autonomous improvement loop:
    1. Generate candidates (Best-of-N)
    2. Evaluate with verifiable rewards
    3. Reflect on results
    4. If stuck, trigger adaptive search
    5. Update experience library
    6. Repeat until max_iterations or time_budget
    """

    for i in range(max_iterations):
        # Generate N candidates
        candidates = best_of_n.generate_candidates(competition_context, n=5)

        # Evaluate
        results = best_of_n.evaluate_all(candidates)
        best_score, best_approach = results[0]

        # Verifiable reward check
        reward, is_improvement, should_search = verifiable_rewards.evaluate({
            'cv_score': best_score,
            'approach': best_approach
        })

        # Reflect
        reflection = reflexion.reflect(best_approach, reward, competition_context)

        # If stuck, search
        if should_search:
            search_results = adaptive_search.search_and_extract(competition_info, "feature_engineering")
            # Incorporate search results into next iteration

        # Update experience library
        if is_improvement:
            experience_library.store_success(competition_info, best_approach, best_score)

        # Log
        print(f"Iteration {i+1}: CV = {best_score:.4f}, Reward = {reward:+.4f}")
        print(f"Reflection: {reflection['summary']}")

        # Check stopping criteria
        if reward < 0.001 and i > 2:  # Plateau
            print("Plateau detected. Stopping autonomous loop.")
            break

    # Report to user
    print(f"\nCompleted {i+1} autonomous iterations.")
    print(f"Best CV: {best_score:.4f}")
    print(f"Recommended next steps: {reflexion.get_recommendation()}")
```

### Expected Impact
- User can "set and forget" for initial exploration
- Faster iteration (no wait for user approval between steps)
- Agent learns to self-diagnose when stuck

---

## Implementation Priority

### Phase 1 (High Impact, Low Effort) — 2-3 days
1. ✅ **Verifiable Rewards System** — Simple scoring logic
2. ✅ **Experience Library** — JSON-based storage + retrieval
3. ✅ **Reflexion** — Structured reflection after each experiment

### Phase 2 (High Impact, Medium Effort) — 1 week
4. ✅ **Best-of-N Sampling** — Parallel candidate generation
5. ✅ **Adaptive Search** — Kaggle discussion scraping + search triggers

### Phase 3 (Medium Impact, High Effort) — 2 weeks
6. ⚠️ **Tree Search** — Strategy tree implementation
7. ⚠️ **Autonomous Loop** — End-to-end autonomous mode

---

## Success Metrics

### Agent Performance
- **Time to competitive baseline**: <30 min (currently ~1-2 hours)
- **Improvement rate**: +5-10% CV score over 5 iterations (currently +2-3%)
- **Plateau breaking**: 70% success rate when stuck (currently ~30%)

### User Experience
- **User intervention frequency**: 1-2 times per competition (currently 5-10 times)
- **Transparency**: User can review all reflections and search queries
- **Trust**: Agent explains "why" for every decision

### Learning Efficiency
- **Cross-competition transfer**: Reuse 50% of strategies from similar competitions
- **Failure prevention**: Avoid 80% of previously failed approaches
- **Knowledge accumulation**: Experience library grows to 50+ entries after 20 competitions

---

## Risks & Mitigations

### Risk 1: Agent gets stuck in local optimum
**Mitigation**: Adaptive search + tree search + forced exploration every N iterations

### Risk 2: Search results are noisy (bad Kaggle advice)
**Mitigation**: Filter by author reputation (gold medals), upvotes, and validate with verifiable rewards

### Risk 3: Autonomous mode wastes compute on bad directions
**Mitigation**: Early stopping if reward < threshold, time budget limits, user can abort

### Risk 4: Experience library overfits to past competitions
**Mitigation**: Diversity penalty in retrieval, periodic pruning of outdated strategies

---

## Next Steps

1. **Review this plan** with user — Get approval on priority and scope
2. **Phase 1 implementation** — Start with Verifiable Rewards + Experience Library
3. **Test on 2-3 past competitions** — Validate improvement metrics
4. **Iterate based on results** — Adjust search strategies, reflection prompts
5. **Phase 2 & 3** — Roll out advanced features incrementally

---

## References

- **AI Self-Improvement Discussion** (documents/AI_Self_Improvement_discussion_11022026.md)
- **Current Kaggle Skill** (.claude/skills/kaggle-agent/SKILL.md)
- **ReAct Paper** — Yao et al., 2022
- **Reflexion Paper** — Shinn et al., 2023
- **Self-RAG** — Asai et al., 2023
- **DeepSeek-R1** — RL + Verifiable Rewards
- **AlphaCode** — Best-of-N + Tree Search
