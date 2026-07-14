# Gradient Boosting — Consolidated Study Notes

> Synthesized English study notes from the course **"Mastering Gradient Boosting Algorithms"**
> ([apxml.com](https://apxml.com/courses/mastering-gradient-boosting-algorithms), Chinese edition),
> **Chapters 1–9**, consolidated into a single reference. Each chapter is a clean restatement of the
> source lessons with reconstructed math (the source rendered formulas via KaTeX, scraped as mangled
> text) and cleaned, idiomatic Python. Companion to the distilled cross-competition
> [`experience.md`](../knowledge/experience.md) and external [`idea_bank.md`](../knowledge/idea_bank.md).
>
> Formulas were reconstructed from a mangled scrape — spot-check against the primary XGBoost /
> LightGBM / CatBoost papers before quoting verbatim.

## Table of Contents

- [Chapter 1 — Gradient Boosting Foundations Revisited](#chapter-1--gradient-boosting-foundations-revisited)
  - [1.1 Ensemble Methods: A Recap](#11-ensemble-methods-a-recap)
  - [1.2 Decision Trees as Base Learners](#12-decision-trees-as-base-learners)
  - [1.3 The Additive Modeling Framework](#13-the-additive-modeling-framework)
  - [1.4 Gradient Descent Fundamentals](#14-gradient-descent-fundamentals)
  - [1.5 Introducing the Gradient Boosting Machine (GBM)](#15-introducing-the-gradient-boosting-machine-gbm)
- [Chapter 2 — The Gradient Boosting Algorithm in Depth](#chapter-2--the-gradient-boosting-algorithm-in-depth)
  - [2.1 Functional Gradient Descent](#21-functional-gradient-descent)
  - [2.2 Deriving the Generic GBM Algorithm](#22-deriving-the-generic-gbm-algorithm)
  - [2.3 Common Loss Functions for Regression](#23-common-loss-functions-for-regression)
  - [2.4 Common Loss Functions for Classification](#24-common-loss-functions-for-classification)
  - [2.5 The Role of Shrinkage (Learning Rate)](#25-the-role-of-shrinkage-learning-rate)
  - [2.6 Sampling Methods (Stochastic Gradient Boosting)](#26-sampling-methods-stochastic-gradient-boosting)
  - [2.7 Implementing GBM with Scikit-learn](#27-implementing-gbm-with-scikit-learn)
  - [2.8 Hands-on: Building a Basic GBM Model](#28-hands-on-building-a-basic-gbm-model)
- [Chapter 3 — Regularization in Gradient Boosting](#chapter-3--regularization-in-gradient-boosting)
  - [3.1 Overfitting Challenges in Boosting](#31-overfitting-challenges-in-boosting)
  - [3.2 Tree-Structure Constraints: Depth, Nodes, and Splits](#32-tree-structure-constraints-depth-nodes-and-splits)
  - [3.3 Shrinkage as Implicit Regularization](#33-shrinkage-as-implicit-regularization)
  - [3.4 Data Subsampling (Stochastic Gradient Boosting)](#34-data-subsampling-stochastic-gradient-boosting)
  - [3.5 Regularized Objective Functions (L1/L2)](#35-regularized-objective-functions-l1l2)
  - [3.6 Early Stopping Strategies](#36-early-stopping-strategies)
  - [3.7 Hands-On: Applying Regularization](#37-hands-on-applying-regularization)
- [Chapter 4 — XGBoost: Extreme Gradient Boosting](#chapter-4--xgboost-extreme-gradient-boosting)
  - [4.1 Motivation and enhancements over standard GBM](#41-motivation-and-enhancements-over-standard-gbm)
  - [4.2 The regularized (second-order) learning objective](#42-the-regularized-second-order-learning-objective)
  - [4.3 Exact greedy split finding](#43-exact-greedy-split-finding)
  - [4.4 Approximate split finding (weighted quantile sketch)](#44-approximate-split-finding-weighted-quantile-sketch)
  - [4.5 Sparsity-aware split finding (default directions for missing values)](#45-sparsity-aware-split-finding-default-directions-for-missing-values)
  - [4.6 System optimizations: cache-aware access, blocks, and parallelism](#46-system-optimizations-cache-aware-access-blocks-and-parallelism)
  - [4.7 The XGBoost API and key parameters](#47-the-xgboost-api-and-key-parameters)
  - [4.8 Hands-on: implementing XGBoost](#48-hands-on-implementing-xgboost)
- [Chapter 5 — LightGBM: Light Gradient Boosting Machine](#chapter-5--lightgbm-light-gradient-boosting-machine)
  - [5.1 Motivation: Overcoming XGBoost's Limitations at Scale](#51-motivation-overcoming-xgboosts-limitations-at-scale)
  - [5.2 Gradient-based One-Side Sampling (GOSS)](#52-gradient-based-one-side-sampling-goss)
  - [5.3 Exclusive Feature Bundling (EFB)](#53-exclusive-feature-bundling-efb)
  - [5.4 Histogram-based Split Finding](#54-histogram-based-split-finding)
  - [5.5 Leaf-wise (Best-first) Tree Growth](#55-leaf-wise-best-first-tree-growth)
  - [5.6 Optimized Categorical Feature Handling](#56-optimized-categorical-feature-handling)
  - [5.7 LightGBM API: Parameters & Configuration](#57-lightgbm-api-parameters--configuration)
  - [5.8 Hands-on: Implementing LightGBM](#58-hands-on-implementing-lightgbm)
- [Chapter 6 — CatBoost: Ordered Boosting for Categorical Data](#chapter-6--catboost-ordered-boosting-for-categorical-data)
  - [6.1 The Trouble with Categorical Data](#61-the-trouble-with-categorical-data)
  - [6.2 Ordered Target Statistics (Ordered TS)](#62-ordered-target-statistics-ordered-ts)
  - [6.3 Prediction Shift and Ordered Boosting](#63-prediction-shift-and-ordered-boosting)
  - [6.4 Feature Combinations](#64-feature-combinations)
  - [6.5 Oblivious (Symmetric) Trees](#65-oblivious-symmetric-trees)
  - [6.6 GPU Training Acceleration](#66-gpu-training-acceleration)
  - [6.7 CatBoost API: Parameters and Configuration](#67-catboost-api-parameters-and-configuration)
  - [6.8 Hands-On: Implementing CatBoost](#68-hands-on-implementing-catboost)
- [Chapter 7 — Advanced Topics & Customization](#chapter-7--advanced-topics--customization)
  - [7.1 Understanding Model Interpretability with SHAP](#71-understanding-model-interpretability-with-shap)
  - [7.2 TreeSHAP for Gradient Boosting](#72-treeshap-for-gradient-boosting)
  - [7.3 Global vs. Local Explanations](#73-global-vs-local-explanations)
  - [7.4 Probability Calibration for Classification](#74-probability-calibration-for-classification)
  - [7.5 Implementing Custom Loss Functions](#75-implementing-custom-loss-functions)
  - [7.6 Implementing Custom Evaluation Metrics](#76-implementing-custom-evaluation-metrics)
  - [7.7 Handling Imbalanced Data with Boosting](#77-handling-imbalanced-data-with-boosting)
  - [7.8 Hands-On: Custom Objectives & SHAP](#78-hands-on-custom-objectives--shap)
- [Chapter 8 — Hyperparameter Optimization Strategies](#chapter-8--hyperparameter-optimization-strategies)
  - [8.1 Why Hyperparameter Tuning Matters](#81-why-hyperparameter-tuning-matters)
  - [8.2 Identifying the Critical Boosting Hyperparameters](#82-identifying-the-critical-boosting-hyperparameters)
  - [8.3 Systematic Tuning: Grid Search vs Random Search](#83-systematic-tuning-grid-search-vs-random-search)
  - [8.4 Advanced Tuning: Bayesian Optimization](#84-advanced-tuning-bayesian-optimization)
  - [8.5 HPO Frameworks: Optuna and Hyperopt](#85-hpo-frameworks-optuna-and-hyperopt)
  - [8.6 Coarse-to-Fine Tuning Strategy](#86-coarse-to-fine-tuning-strategy)
  - [8.7 Cross-Validation Strategy for Honest Tuning](#87-cross-validation-strategy-for-honest-tuning)
  - [8.8 Hands-On: Advanced Tuning with Optuna (XGBoost)](#88-hands-on-advanced-tuning-with-optuna-xgboost)
- [Chapter 9 — Gradient Boosting for Specialized Tasks](#chapter-9--gradient-boosting-for-specialized-tasks)
  - [9.1 Learning to Rank (LTR) with Gradient Boosting](#91-learning-to-rank-ltr-with-gradient-boosting)
  - [9.2 Ranking Objective Functions (Pairwise and Listwise)](#92-ranking-objective-functions-pairwise-and-listwise)
  - [9.3 Gradient Boosting for Survival Analysis](#93-gradient-boosting-for-survival-analysis)
  - [9.4 Survival Objective Functions (Cox Proportional Hazards)](#94-survival-objective-functions-cox-proportional-hazards)
  - [9.5 Quantile Regression with Gradient Boosting](#95-quantile-regression-with-gradient-boosting)
  - [9.6 Implementing the Quantile Loss (Custom Objective)](#96-implementing-the-quantile-loss-custom-objective)
  - [9.7 Multi-Output Gradient Boosting](#97-multi-output-gradient-boosting)
  - [9.8 Hands-On: Implementing Ranking with XGBoost](#98-hands-on-implementing-ranking-with-xgboost)

---

## Chapter 1 — Gradient Boosting Foundations Revisited

> Synthesized study notes from the course *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 1. These notes reconstruct the technical content and clean up the KaTeX-mangled math from the source lessons; they are for study reference, not a verbatim transcript.

---

### 1.1 Ensemble Methods: A Recap

**Ensemble methods** combine the predictions of multiple individual models to produce a final prediction that is usually more accurate and more stable than any single model. The core idea is the "wisdom of the crowd": aggregating the opinions (predictions) of several diverse models mitigates the weaknesses of any one of them. The goal is to improve performance by reducing **variance** (sensitivity to small changes in the training data), reducing **bias** (systematic error from simplifying assumptions in the model), or sometimes both.

Three dominant strategies:

**Bagging (Bootstrap Aggregating).** Create multiple subsets of the training data by bootstrap sampling (sampling with replacement), train an independent base model (usually the same type, e.g. a decision tree) on each subset, then aggregate by **averaging** (regression) or **majority vote** (classification). Because each model sees a slightly different version of the data, they learn slightly different patterns, producing diversity. The best-known example is the **Random Forest**, which adds further diversity by choosing a random subset of features at every split. Bagging's main strength is **variance reduction**, making the ensemble less prone to overfitting — especially with complex, deep-tree base learners. Models train **in parallel**, so bagging is computationally efficient on multi-core systems.

**Boosting.** A fundamentally different approach: models are built **sequentially**, and each new model focuses on correcting the errors made by the ensemble so far. The intuitive loop:

1. Train a simple base model on the data.
2. Identify the instances the model handles poorly (large errors / residuals).
3. Train a new base model that pays more attention to those hard instances.
4. Combine the new model with the previous ones (typically a weighted sum).
5. Repeat 2–4 until a set number of iterations, or until performance stops improving.

Early boosting algorithms such as **AdaBoost** did this by explicitly increasing the weights of misclassified instances in later models. **Gradient boosting** — the focus of the course — refines this by fitting each new model to the **residual errors** (for regression) or, more generally, to the **negative gradient of the loss function** with respect to the current ensemble's predictions. This frames boosting as optimization in *function space*: at each step we add a function (model) pointing in the direction of steepest descent of the overall loss. Boosting excels at **reducing bias** and often yields highly accurate models, but training is inherently **serial** (each model depends on the previous one) and, if not properly regularized, boosting is more prone to overfitting.

The general boosting model is an additive model:

$$
F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta \cdot h_m(x)
$$

where $F_0(x)$ is the initial guess (often the mean of the target), $h_m(x)$ is the base learner added at step $m$ (e.g. a decision tree), $M$ is the total number of boosting rounds, and $\eta$ is the **learning rate** (shrinkage factor) that controls each new tree's contribution.

**Stacking (Stacked Generalization).** Train several *different* types of base model (e.g. random forest, SVM, k-NN) on the same data, then train a **meta-model** (blender / second-level model) that uses the base models' predictions as input features and learns how best to combine them. Stacking can beat bagging or boosting alone but needs careful setup — especially cross-validation, so that the base-model predictions used to train the meta-model are out-of-fold — and can be computationally expensive.

**Why focus on boosting?** Its sequential, adaptive nature, combined with the efficiency and regularization innovations in modern implementations (XGBoost, LightGBM, CatBoost), makes gradient boosting one of the most effective and widely used techniques for structured (tabular) data. The parallel-vs-sequential contrast between bagging and boosting is the key mental model going forward.

> **Relevance to our work:** Our Kaggle tabular pipelines rest on this parallel/sequential distinction. Random-forest-style variance reduction and GBM-style bias reduction are complementary; stacking's requirement of *out-of-fold* base predictions is exactly the CV discipline we enforce to avoid leakage in the meta-layer.

---

### 1.2 Decision Trees as Base Learners

Decision trees — specifically **CART (Classification And Regression Trees)** — are the standard building block in gradient boosting. A tree partitions the feature space into rectangular regions and assigns a constant prediction in each leaf.

**Why decision trees:**

- **Capture non-linearity and interactions.** Unlike linear models, trees naturally model non-linear relationships. The sequential nature of splits lets a tree isolate a region jointly defined by conditions on several features (e.g. split on A, then on B), implicitly modeling their interaction.
- **Handle mixed data types.** CART can natively handle numerical and categorical features during splitting. Some boosting libraries originally required preprocessing (e.g. one-hot encoding) for categoricals; LightGBM and CatBoost include specialized, efficient native handling.
- **Computational tractability.** Finding the best split scans features and candidate split points. Exact-greedy, approximate, and **histogram-based** algorithms make training efficient, especially for the relatively simple trees used in boosting; these are also parallelizable to a degree.
- **Interpretability (single tree).** A large ensemble of hundreds/thousands of trees is complex, but a single shallow tree is easy to visualize and inspect — useful during development and debugging.

**The "weak learner" principle.** Boosting combines many **weak learners** (models slightly better than random guessing) into a strong learner. For trees this means **shallow** trees: stumps (a single split, depth 1) or trees of limited depth (e.g. 2–8). Keeping trees weak:

- **Reduces overfitting** — deep trees overfit; simple trees limit the complexity added per iteration.
- **Focuses on residuals** — each new tree corrects the ensemble's current errors; a weak learner captures the signal in those residuals without fitting noise.
- **Enables gradual improvement** — small steps prevent large, unstable jumps and allow fine-tuning over many iterations.

This contrasts sharply with Random Forests, which typically benefit from **deep, fully grown** trees and rely on averaging many de-correlated trees to reduce variance.

Hyperparameters commonly used to keep trees weak: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`.

**How trees are trained inside boosting.** A standard tree is trained to predict the target $y$ directly. In gradient boosting, at iteration $m$ the new tree $h_m(x)$ is trained to predict the **pseudo-residuals** — the negative gradient of the loss with respect to the previous prediction. For squared-error regression loss the pseudo-residual is exactly the ordinary residual:

$$
r_{im} = y_i - F_{m-1}(x_i)
$$

where $F_{m-1}(x)$ is the ensemble's prediction after $m-1$ iterations. For other losses (e.g. log-loss for classification) the pseudo-residual is the negative gradient, steering tree construction toward the direction that most reduces total loss. The tree's usual split criterion (e.g. variance reduction for a regression tree) is then applied with these pseudo-residuals as the target.

A simple regression walkthrough:

1. Start from a constant prediction $F_0(x)$.
2. Compute residuals $r_{i1} = y_i - F_0(x_i)$ for all points $i$.
3. Fit a shallow tree $h_1(x)$ to predict $r_{i1}$; its splits group points with similar residuals.
4. Update: $F_1(x) = F_0(x) + \nu\, h_1(x)$, where $\nu$ is the learning rate.
5. Compute new residuals $r_{i2} = y_i - F_1(x_i)$.
6. Fit $h_2(x)$ to predict $r_{i2}$, and so on.

Each tree explains the error the ensemble has not yet explained; its splits minimize the variance of residuals within the resulting nodes.

**Addressing single-tree limitations.** A lone tree has **high variance** (sensitive to small data changes) and uses **greedy splitting** (locally optimal, no global-optimality guarantee). Gradient boosting mitigates both: variance is reduced by combining many trees (additively, stage-wise) plus shrinkage and subsampling (stochastic gradient boosting); accuracy improves because the sequential, stage-wise fit can discover more complex patterns than a single greedy tree, with regularization controlling complexity.

> **Relevance to our work:** This is the justification for the shallow, strongly regularized trees we favor in tree-search experiments — depth 2–8 stumps/shallow trees as transferable prototypes, rather than deep RF-style trees. Splitting on pseudo-residuals (not the raw target) is why per-iteration behavior depends on the whole ensemble state so far.

---

### 1.3 The Additive Modeling Framework

The additive model formalizes the sequential learning at the heart of boosting. Unlike bagging (independent, parallel learners), an additive model is built **iteratively**, each new component targeting the shortcomings of the current ensemble.

**Sequential nature.** Start with an initial, usually very simple prediction; assess its shortcomings; add a simple model dedicated to compensating for the errors made so far; repeat. The final prediction is the **sum** of all sequentially built components.

**Formalization.** After $M$ stages the additive model is:

$$
F_M(x) = F_0(x) + \sum_{m=1}^{M} \beta_m h_m(x)
$$

- $F_M(x)$ — final ensemble prediction after $M$ steps.
- $F_0(x)$ — initial base model / starting prediction. Often just the mean of the target (regression) or the log-odds (classification) — the best guess before adding any learner.
- $h_m(x)$ — the $m$-th base learner (usually a decision tree), trained specifically to address the errors of $F_{m-1}(x)$.
- $\beta_m$ — coefficient/weight applied to the $m$-th learner, controlling its contribution. In gradient boosting this is closely tied to the **learning rate / shrinkage**, which damps each step's influence to help prevent overfitting.
- $M$ — total number of base learners (boosting stages).

**Iterative procedure.**

1. **Initialize** with $F_0(x)$ — the constant that minimizes the loss on the training data (e.g. the mean for squared-error loss).
2. **For $m = 1$ to $M$:**
   - Compute the errors (residuals or gradients) of the current ensemble $F_{m-1}(x)$ — the "unexplained" part of the target.
   - Train a new base learner $h_m(x)$ to predict those errors.
   - Determine the optimal coefficient $\beta_m$ (typically by minimizing overall loss).
   - Update: $F_m(x) = F_{m-1}(x) + \beta_m h_m(x)$.

**Why it works.** By focusing on the previous model's errors, each new learner attacks whatever the ensemble currently finds hardest, so performance improves progressively and complex patterns a single model would miss can be captured. Gradient boosting is a very successful family operating inside this additive framework, providing a concrete, mathematically grounded way to decide how to train each $h_m(x)$ to best correct $F_{m-1}(x)$ — via gradient descent in function space.

---

### 1.4 Gradient Descent Fundamentals

To decide *what correction* each new component should make, boosting borrows from optimization — most importantly **gradient descent**.

**Objective: minimize loss.** Supervised learning minimizes a loss function $L(y, F(x))$ measuring the gap between the true target $y$ and the model's prediction $F(x)$. Common examples: mean squared error (regression), log-loss (classification). The goal is a model $F(x)$ with the smallest possible loss over the training data.

**Intuition — descending a slope.** Picture the loss as a landscape of hills and valleys where height is the loss for a given model configuration; we want the lowest point. Gradient descent:

1. Start at some point (initial model).
2. Determine the steepest downhill direction.
3. Take a small step in that direction.
4. Repeat until no further descent is possible (a minimum, hopefully global).

Calculus tells us the gradient $\nabla L$ points in the direction of steepest **ascent**, so to go downhill we move along $-\nabla L$.

**Update rule.** For model parameters $\theta$, the basic update at iteration $t$ is:

$$
\theta_{t+1} = \theta_t - \eta\, \nabla L(\theta_t)
$$

- $\theta_t$ — current parameters.
- $\nabla L(\theta_t)$ — gradient of the loss w.r.t. $\theta$, evaluated at $\theta_t$; points toward fastest loss increase.
- $\eta$ — learning rate, a small positive scalar controlling step size.
- $\theta_{t+1}$ — updated parameters.

**Learning rate $\eta$.** Too large → overshoot the minimum, loss oscillates or diverges. Too small → very slow convergence, many iterations. Finding a good value takes experimentation. In gradient boosting this learning rate is called **shrinkage** and plays a dual role: controlling the learning process *and* acting as a form of regularization.

Variants exist: **batch** gradient descent uses the full dataset; **stochastic gradient descent (SGD)** uses a single point; **mini-batch** uses a small subset. The stochastic variants are cheaper and can help escape shallow local minima. Standard GBM computes gradients over the whole dataset unless stochastic subsampling is explicitly used.

**Gradient descent in function space.** Instead of optimizing a fixed parameter vector $\theta$, gradient boosting optimizes in **function space**. Starting from a simple initial model (e.g. the mean), at each iteration $m$ we seek a new function $h_m(x)$ to add to the current ensemble so that overall loss decreases:

$$
F_m(x) = F_{m-1}(x) + \eta\, h_m(x)
$$

It computes the negative gradient of $L(y, F(x))$ with respect to the current prediction $F_{m-1}(x)$, evaluated at each training instance $i$:

$$
r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
$$

These negative gradients $r_{im}$ are the **pseudo-residuals**: the direction, in function space, in which the loss for each point decreases fastest given the current ensemble. The algorithm fits the new base learner $h_m(x)$ to approximate these pseudo-residuals. In effect, gradient descent guides the sequential construction of the ensemble, telling each next tree which errors to focus on correcting.

> **Relevance to our work:** Shrinkage (`learning_rate`) is simultaneously our step size and our primary regularizer — the same knob we tune against overfitting in CV. Viewing boosting as functional gradient descent is the lens that makes the rest of the course's algorithms coherent.

---

### 1.5 Introducing the Gradient Boosting Machine (GBM)

Jerome Friedman's original **Gradient Boosting Machine (GBM)** unifies ensembling, decision-tree base learners, additive modeling, and gradient descent into one concrete framework for iteratively building an additive model.

At its core, GBM is gradient-descent optimization — but in **function space**, not over a fixed parameter set. Given a model $F_{m-1}(x)$ built over $m-1$ iterations, we add a base learner $h_m(x)$ so that $F_m(x) = F_{m-1}(x) + h_m(x)$ minimizes total loss $L(y, F(x))$. The new function is chosen to point along the negative gradient of the loss with respect to the current predictions. For observation $(x_i, y_i)$, the negative gradient (pseudo-residual) is:

$$
r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
$$

A base learner (e.g. a CART regression tree) is then fit on the features $x_i$ to predict these pseudo-residuals — i.e. the new tree learns the errors of $F_{m-1}(x)$ as expressed by the loss gradient.

**The GBM algorithm:**

1. **Initialize** with a constant model minimizing the loss on the training data (the mean for squared-error loss, the log-odds for log-loss):
   $$
   F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma)
   $$

2. **For $m = 1$ to $M$:**

   a. **Compute pseudo-residuals** — the negative gradients $r_{im}$ for all observations $i = 1, \dots, N$ based on the current model $F_{m-1}(x)$.

   b. **Fit a base learner** $h_m(x)$ (e.g. a regression tree) using features $x_i$ and pseudo-residuals $r_{im}$ as the target; the tree approximates the negative gradient.

   c. **Compute the step size (multiplier)** $\gamma_m$ for the new tree, typically via a line search minimizing the loss:
      $$
      \gamma_m = \arg\min_{\gamma} \sum_{i=1}^{N} L\!\left(y_i,\, F_{m-1}(x_i) + \gamma\, h_m(x_i)\right)
      $$
      In practice — especially combined with shrinkage — a fixed learning rate $\eta$ is often used instead of or together with the line search, so the update includes $\eta \cdot h_m(x)$ or $\eta \cdot \gamma_m \cdot h_m(x)$.

   d. **Update the model:**
      $$
      F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)
      \quad\text{(or, more commonly, } F_m(x) = F_{m-1}(x) + \eta\, \gamma_m h_m(x)\text{)}
      $$

3. **Output** the final model $F_M(x)$.

**Flexibility.** GBM's power comes from the free choice of loss function $L$ and base-learner type $h_m$. Regression trees are the standard choice: they naturally predict continuous pseudo-residuals and capture non-linearities and interactions. The tree's structure (splits, depth) is fixed during the fit in step 2b.

Common losses:

- **Squared error** for regression, $L(y, F) = \tfrac{1}{2}(y - F)^2$, whose negative gradient is the ordinary residual $r_{im} = y_i - F_{m-1}(x_i)$.
- **Log-loss (deviance)** for binary classification.

The loss choice directly determines how pseudo-residuals are computed and therefore which errors the model prioritizes correcting.

This basic GBM is powerful and general but has limitations in **computational efficiency, regularization, and handling specific data types** — the motivation for XGBoost, LightGBM, and CatBoost, which build on this core framework with sophisticated regularization, optimized tree-construction algorithms, and specialized data handling (covered in later chapters).

> **Relevance to our work:** This five-step loop is the exact scaffold underneath every boosting library we run. Knowing that the initial $F_0$ is a loss-minimizing constant, and that the update carries both a per-tree multiplier $\gamma_m$ and a global shrinkage $\eta$, clarifies which knobs the modern libraries expose and how our tree-search perturbations map onto them.

---

### Key takeaways

- **Ensembles** trade off bias and variance: bagging (parallel, variance-reducing, e.g. Random Forest) vs. boosting (sequential, bias-reducing) vs. stacking (meta-model over diverse learners, needs out-of-fold predictions).
- **Boosting is sequential and error-correcting**; the additive recursion is $F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$, with final model $F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta\, h_m(x)$.
- **Shallow CART trees are the standard weak learner** — kept intentionally weak (small depth, leaf/split constraints) so boosting improves gradually and avoids fitting noise; this is opposite to Random Forest's deep trees.
- **Trees fit pseudo-residuals, not the raw target.** For squared-error loss the pseudo-residual equals the ordinary residual $y_i - F_{m-1}(x_i)$; in general it is the negative gradient of the loss w.r.t. the current prediction.
- **Gradient descent supplies the correction direction**; the learning rate $\eta$ is both the step size and the primary regularizer (shrinkage).
- **Gradient boosting = functional gradient descent** — each tree approximates $r_{im} = -\left[\partial L / \partial F(x_i)\right]_{F=F_{m-1}}$.
- **Friedman's GBM** = initialize with a loss-minimizing constant → (compute pseudo-residuals → fit tree → line-search step size → update) repeated $M$ times.
- **The choice of loss function drives everything** downstream (pseudo-residuals, which errors get prioritized); GBM's limitations in efficiency, regularization, and categorical/data handling motivate XGBoost, LightGBM, and CatBoost.

---

## Chapter 2 — The Gradient Boosting Algorithm in Depth

> Source: *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 2. Synthesized study notes reconstructed from the scraped course text, with LaTeX cleaned and code idiomatized. This chapter derives GBM as gradient descent in function space, works out the pseudo-residuals for the common regression and classification losses, and covers shrinkage, stochastic subsampling, and the scikit-learn implementation.

### 2.1 Functional Gradient Descent

Standard ML training (linear regression, a fixed-architecture neural net) optimizes a fixed model structure $f(x;\theta)$ by searching for the best **parameters** $\theta$. Gradient descent updates them as

$$\theta_{new} = \theta_{old} - \eta \, \nabla_{\theta} L,$$

where $\nabla_{\theta} L$ is the gradient of the loss with respect to the parameters.

GBM optimizes differently. Rather than tuning parameters inside a fixed structure, it **builds the model itself iteratively**. The object being optimized is the entire ensemble function $F(x)$. We search the space of functions for one that minimizes the total training loss — this is **gradient descent in function space**.

**Optimization in function space.** Picture a high-dimensional space in which each point is a function $F$. We want $F^*$ minimizing the total loss over the dataset $(x_i, y_i)$, $i = 1, \dots, N$:

$$L_{total}(F) = \sum_{i=1}^{N} L(y_i, F(x_i)).$$

We improve the current estimate iteratively. Let $F_{m-1}(x)$ be the ensemble after $m-1$ boosting rounds. We seek a new base learner $h_m(x)$ (usually a decision tree) that, added to the current model, moves us toward lower loss:

$$F_m(x) = F_{m-1}(x) + \eta \, h_m(x),$$

where $\eta$ is the step size (learning rate) and $h_m(x)$ points in a loss-decreasing direction.

**The gradient direction.** Treat each prediction $F(x_i)$ as a "coordinate" of the current point in function space. The gradient of $L_{total}$ with respect to these coordinates has $i$-th component

$$g_{im} = \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}.$$

The vector $(g_{1m}, \dots, g_{Nm})$ points in the direction of steepest **ascent** of the total loss at $F_{m-1}$.

**Pseudo-residuals — the base learner's target.** To descend, move opposite the gradient. Define the negative-gradient component, the **pseudo-residual**, for each sample $i$ at iteration $m$:

$$r_{im} = -g_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}.$$

These $r_{im}$ are the targets the next base learner $h_m(x)$ should approximate. Why "pseudo"? For squared-error loss $L(y, F) = \tfrac{1}{2}(y - F)^2$, the gradient is $\partial L/\partial F = -(y - F)$, so

$$r_{im} = y_i - F_{m-1}(x_i),$$

exactly the ordinary residual. For other losses $r_{im}$ is not the plain residual but still marks the direction/magnitude in which the current model most needs improvement, evaluated pointwise.

**Taking the step.** We cannot literally add the residual vector to $F_{m-1}$; instead we fit a base learner to predict the pseudo-residuals from the input features:

$$h_m = \arg\min_{h} \sum_{i=1}^{N} \left( r_{im} - h(x_i) \right)^2.$$

This generalizes the negative-gradient step across the whole input domain, not just the training points. Then we update the ensemble, scaled by the learning rate:

$$F_m(x) = F_{m-1}(x) + \eta \, h_m(x).$$

This functional-gradient view (Friedman, 2001) is the unifying framework behind XGBoost, LightGBM, and CatBoost.

> **Relevance to our work:** every gradient-boosting library we use — LightGBM included — is running this same loop. Each tree is fit to the current negative gradient; understanding "the model corrects its own errors, one negative-gradient step at a time" is what makes learning-rate / n_estimators / early-stopping behavior intuitive rather than magic.

### 2.2 Deriving the Generic GBM Algorithm

GBM is an **additive model** built sequentially:

$$F(x) = F_M(x) = F_0(x) + \sum_{m=1}^{M} h_m(x),$$

where $F_0(x)$ is the initial constant estimate (mean of $y$ for L2 regression; log-odds for classification) and each $h_m(x)$ is a base learner added at iteration $m$. A per-tree step weight $\beta_m$ is usually absorbed into $h_m$ or handled by a separate learning rate.

At iteration $m$ we have $F_{m-1}$ and want $h_m$ so that $F_m = F_{m-1} + h_m$ lowers

$$\mathcal{L}(F_m) = \sum_{i=1}^{N} L\big(y_i, F_{m-1}(x_i) + h_m(x_i)\big).$$

We pick $h_m$ pointing along the negative gradient of $\mathcal{L}$ w.r.t. the current predictions, i.e. fit $h_m$ to the pseudo-residuals $r_{im} = -g_{im}$.

**Fitting the base learner.** Typically $h_m$ is a shallow CART regression tree trained on $\{(x_i, r_{im})\}$:

$$h_m \approx \arg\min_{h} \sum_{i=1}^{N} \left( r_{im} - h(x_i) \right)^2.$$

**Optimizing the leaf values (line search).** Once the tree structure — the leaf regions $R_{jm}$ — is fixed by fitting the pseudo-residuals, we do not just average the residuals in each leaf. Instead we pick the constant $\gamma_{jm}$ that minimizes the **original** loss over the samples in that leaf:

$$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L\big(y_i, F_{m-1}(x_i) + \gamma\big).$$

This is a per-leaf line search that makes the tree's contribution directly minimize the loss given $F_{m-1}$. For squared error it collapses to the mean of the residuals in the leaf; for absolute error it is the **median**; for log loss it needs its own closed/approximate form.

**Update with shrinkage.** Finally scale the new tree by the learning rate $\nu$ (shrinkage):

$$F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm} \, \mathbb{I}(x \in R_{jm}).$$

$\nu$ (typically 0.01–0.3) damps each tree's influence, forcing slower, more regularized learning.

**The generic algorithm.**

1. **Initialize** with a constant: $F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma)$.
   - L2 regression → $F_0$ is the mean of $y$.
   - Binary log loss → $F_0$ is the log-odds of the overall positive rate.
2. **For** $m = 1, \dots, M$:
   - **a. Pseudo-residuals:** $r_{im} = -\left[ \dfrac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}}$ for each $i$.
   - **b. Fit base learner:** train tree $h_m$ on $\{(x_i, r_{im})\}$, giving leaf regions $R_{jm}$.
   - **c. Optimal leaf values:** $\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$.
   - **d. Update:** $F_m(x) = F_{m-1}(x) + \nu \sum_{j} \gamma_{jm}\,\mathbb{I}(x \in R_{jm})$.
3. **Output** $F_M(x)$ (mapped through sigmoid/softmax for classification probabilities).

The choice of $L$ determines both the pseudo-residual form and the leaf optimization, which is what lets one framework cover regression and classification.

### 2.3 Common Loss Functions for Regression

Each regression loss yields a different pseudo-residual, changing how the model reacts to outliers.

**Squared error (L2 loss).**

$$L(y_i, F(x_i)) = \tfrac{1}{2}\big(y_i - F(x_i)\big)^2,$$

the $\tfrac{1}{2}$ is a convenience for differentiation. Pseudo-residual:

$$r_{im} = y_i - F_{m-1}(x_i),$$

the ordinary residual — each new tree predicts the ensemble's current error. Characteristics: models the conditional **mean**; **sensitive to outliers** (squared errors dominate); smooth and differentiable everywhere.

**Absolute error (L1 loss).**

$$L(y_i, F(x_i)) = |y_i - F(x_i)|, \qquad r_{im} = \operatorname{sign}\!\big(y_i - F_{m-1}(x_i)\big),$$

where $\operatorname{sign}(z)$ is $+1/{-1}/0$. Each tree only learns whether the current prediction is too low ($+1$) or too high ($-1$). Characteristics: models the conditional **median**; **robust to outliers** (error contributes linearly); **non-smooth** — the derivative is discontinuous at zero residual, handled in practice by setting the gradient to 0 there or using a subgradient. The leaf line search here uses the median of residuals.

**Huber loss** — a compromise: quadratic for small errors, linear for large ones, with threshold $\delta$:

$$L_\delta(y_i, F(x_i)) = \begin{cases} \tfrac{1}{2}(y_i - F(x_i))^2 & |y_i - F(x_i)| \le \delta \\[4pt] \delta\big(|y_i - F(x_i)| - \tfrac{1}{2}\delta\big) & \text{otherwise} \end{cases}$$

The $-\tfrac{1}{2}\delta^2$ term makes it continuously differentiable at $|y_i - F(x_i)| = \delta$. Pseudo-residual:

$$r_{im} = \begin{cases} y_i - F_{m-1}(x_i) & |y_i - F_{m-1}(x_i)| \le \delta \\[4pt] \delta \cdot \operatorname{sign}\!\big(y_i - F_{m-1}(x_i)\big) & \text{otherwise} \end{cases}$$

Small errors behave like L2; large errors are clipped to $\pm\delta$ (L1-like). Smaller $\delta$ → more L1-like/robust but slower convergence; larger $\delta$ → more L2-like. $\delta$ effectively defines what counts as an outlier and is usually tuned by cross-validation.

**Choosing:** L2 for clean data / mean prediction (default, efficient); L1 for data with clear outliers (median focus); Huber when you want both robustness and near-minimum smoothness. In scikit-learn this is the `loss` parameter of `GradientBoostingRegressor` (`'squared_error'`, `'absolute_error'`, `'huber'`, `'quantile'`).

> **Relevance to our work:** on Kaggle tabular tasks with heavy-tailed targets, switching the objective (or log-transforming the target) is a cheap, high-leverage move — it changes what the pseudo-residuals emphasize. LightGBM exposes the same choice via `objective` (`regression_l2`, `regression_l1`, `huber`, `quantile`).

### 2.4 Common Loss Functions for Classification

Classification needs losses that compare predicted class probabilities/scores to discrete labels; the negative gradient again defines the pseudo-residuals.

**Binary: log loss (binomial deviance).** For $y \in \{0, 1\}$, the model's raw output $F(x)$ is a logit, mapped to a probability by the sigmoid:

$$p(x) = P(y = 1 \mid x) = \frac{1}{1 + e^{-F(x)}}.$$

The per-observation log loss (negative log-likelihood):

$$L(y, p(x)) = -\big[\, y \log p(x) + (1 - y)\log(1 - p(x)) \,\big].$$

It punishes confident wrong predictions harshly ($-\log p \to \infty$ as $p \to 0$ when $y=1$) and vanishes for confident correct ones. Using the chain rule with $\dfrac{\partial p}{\partial F} = p(1-p)$ and $\dfrac{\partial L}{\partial p} = \dfrac{p - y}{p(1-p)}$, the negative gradient w.r.t. the raw output collapses neatly to

$$r_{im} = y_i - p_{i,m-1},$$

the difference between the true label (0/1) and the current predicted probability. Each new tree predicts the residual of the current probability estimate.

**Multiclass: log loss (multinomial deviance).** For $y \in \{1, \dots, K\}$ with one-hot labels $y_{ik}$, the model outputs $K$ logits $F_1(x), \dots, F_K(x)$, turned into probabilities by softmax:

$$p_k(x) = \frac{e^{F_k(x)}}{\sum_{j=1}^{K} e^{F_j(x)}}, \qquad L = -\sum_{k=1}^{K} y_k \log p_k(x) = -\log p_c(x),$$

with $c$ the true class. The per-class pseudo-residual is again

$$r_{imk} = y_{ik} - p_{ik,\,m-1}.$$

In practice most implementations fit **$K$ separate regression trees per boosting round** (one per class, each targeting its $y_{ik} - p_{ik}$), then combine to update the $K$ score functions.

**Why log loss over exponential?** The `'exponential'` option reproduces AdaBoost, penalizing errors exponentially so a single very-wrong point can dominate the gradient and distort the model. Log loss (deviance) is less sensitive to outliers/mislabeled points, ties directly to information theory (cross-entropy) and probability (log-likelihood), and its smoother gradient is friendlier to the optimizer. It is the default for classification.

> **Relevance to our work:** the elegant $y - p$ pseudo-residual is why log loss is the workhorse objective and why calibrated probabilities come out naturally. For imbalanced Kaggle problems the deviance/cross-entropy path (plus `scale_pos_weight` / class weights in LightGBM) is almost always preferable to exponential loss.

### 2.5 The Role of Shrinkage (Learning Rate)

The naive update $F_m = F_{m-1} + h_m$ adds each tree's full prediction, which can be too aggressive — a tree that fits the current pseudo-residuals perfectly may overfit that stage's specific errors. **Shrinkage** scales each new tree by a factor $\nu$ (or $\eta$), $0 < \nu \le 1$:

$$F_m(x) = F_{m-1}(x) + \nu \, h_m(x).$$

**Regularizing effect.** Reducing each tree's influence slows learning, taking smaller steps in function space. Consequences:

- **Variance reduction:** each tree matters less, so the ensemble is less sensitive to any single tree's training data → better generalization.
- **More trees needed:** smaller contributions mean more boosting rounds (`n_estimators`) are required to reach a given training fit.
- **Better generalization (usually):** gradual refinement avoids rushing to an overfit solution, improving validation/test performance.

Think of $\nu$ as the step size in numerical optimization — small steps take more iterations but avoid overshooting and often settle at a better, more stable minimum.

**The learning-rate / n_estimators trade-off.**

- Very small $\nu$ (e.g. 0.01) needs large $M$ (thousands): slower to train, often generalizes better.
- Large $\nu$ (0.5–1.0) needs few iterations: fast, but much higher overfitting risk unless other regularizers (tree depth, subsampling) are managed.

**Practical notes:** typical range $[0.01, 0.3]$, common starting point 0.1. Learning rate is among the most important hyperparameters, tuned jointly with `n_estimators` (grid/random/Bayesian search, cross-validated). **Early stopping** is routinely used to find the optimal number of iterations for a given learning rate. Shrinkage interacts with other regularizers — a smaller learning rate can tolerate slightly deeper trees or less aggressive subsampling because each tree's impact is already reduced.

> **Relevance to our work:** the standard recipe — set a low learning rate and let early stopping on a validation fold choose `n_estimators` — is exactly how we tune LightGBM. Lock the two together: never grid-search a fixed n_estimators against learning rate independently; instead fix a small `lr` and read the best iteration off the CV curve.

### 2.6 Sampling Methods (Stochastic Gradient Boosting)

Standard GBM can overfit, especially with deep trees or many rounds. **Stochastic Gradient Boosting (SGB)** (Friedman, 2002) injects randomness by sampling the data each iteration — reducing variance to improve generalization and often speeding up training.

**Row subsampling.** Before fitting each tree $h_m$, draw a fraction $\eta_{subsample}$ of the training rows **without replacement** (the `subsample` parameter). Only that subset is used to (1) compute pseudo-residuals from $F_{m-1}$ and (2) fit $h_m$. Rows not chosen are simply not used for that tree.

- **Regularization:** every tree sees a slightly different subset, so the model is less able to memorize noise/outliers → lower variance.
- **Compute:** fitting on fewer rows is faster.
- **Synergy with shrinkage:** subsampling pairs well with a small learning rate; the injected randomness stabilizes learning when per-tree contributions are small.

Common range 0.5–0.8; `subsample = 1.0` recovers deterministic GBM. Too small a fraction hinders learning (more bias, needs more trees).

**Column (feature) subsampling.** Also sample a random subset of features when building each tree — as in random forests — before finding the best split at each node (`max_features` in scikit-learn).

- **Regularization:** stops the model leaning on a few dominant features; encourages tree diversity and robustness.
- **Compute:** shrinks the split search space, speeding training on wide datasets.

`max_features` accepts an int (exact count), a float (fraction of `n_features`), or a string (`'sqrt'`, `'log2'`); `None`/`n_features` disables column subsampling.

**Combining both.** Row and column subsampling are independent and combine into a strong regularizer, e.g. `subsample=0.8` and `max_features=0.8` → each tree uses 80% of rows and considers 80% of features per split. These become tuned hyperparameters alongside learning rate and tree complexity; note the couplings — e.g. lower sampling rates may require more `n_estimators` or a different learning rate.

> **Relevance to our work:** this is the same knob set as LightGBM's `bagging_fraction` (+ `bagging_freq`) and `feature_fraction`. Values around 0.7–0.9 are our sensible defaults for regularizing on noisy tabular data; combined with a low learning rate and early stopping they are the backbone of a robust CV setup.

### 2.7 Implementing GBM with Scikit-learn

Scikit-learn wraps GBM in `GradientBoostingRegressor` and `GradientBoostingClassifier` (binary and multiclass), both following the standard estimator API (`fit`, `predict`, `predict_proba`).

**Theory → hyperparameters:**

- **`loss`** — the objective. Regressor: `'squared_error'` (L2, default), `'absolute_error'` (L1/LAD), `'huber'`, `'quantile'`. Classifier: `'log_loss'`/`'deviance'` (default, probabilistic) or `'exponential'` (reproduces AdaBoost).
- **`n_estimators`** — number of boosting stages $M$ (default 100); more → more complex, can overfit if unbalanced with other regularizers.
- **`learning_rate`** — shrinkage $\nu$ (default 0.1); smaller needs more estimators but generalizes better.
- **`subsample`** — row fraction per tree; `< 1.0` enables SGB (default 1.0).
- **Tree params:** `max_depth` (default 3), `min_samples_split` (2), `min_samples_leaf` (1), `max_features` (column subsampling, default `None`).
- **`init`** — estimator for $F_0(x)$ (default: mean for regression, log-odds for classification).

Regressor example (synthetic noisy sine):

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 1. Synthetic data
rng = np.random.RandomState(0)
X = rng.rand(100, 1) * 10
y = np.sin(X).ravel() + rng.normal(0, 0.5, X.shape[0])  # noisy target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 3. Train
gbr = GradientBoostingRegressor(
    n_estimators=100,      # number of trees (M)
    learning_rate=0.1,     # shrinkage (nu)
    max_depth=3,           # max depth per tree
    subsample=0.8,         # 80% of rows per tree (SGB)
    loss="squared_error",  # L2 (older sklearn: 'ls')
    random_state=42,
)
gbr.fit(X_train, y_train)

# 4-5. Predict and evaluate
y_pred = gbr.predict(X_test)
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
```

Classifier example (synthetic 2-feature data):

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

# 1. Synthetic classification data
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42, class_sep=1.0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 2. Train
gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,           # shallow trees often suit classification
    subsample=0.8,
    loss="log_loss",       # log loss / deviance (older sklearn: 'deviance')
    random_state=42,
)
gbc.fit(X_train, y_train)

# 3. Predict
y_pred = gbc.predict(X_test)
y_pred_proba = gbc.predict_proba(X_test)[:, 1]  # positive-class prob

# 4. Evaluate
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test log loss: {log_loss(y_test, y_pred_proba):.4f}")

# 5. Feature importance
importances = gbc.feature_importances_
importance_df = (
    pd.DataFrame({"feature": [f"f{i}" for i in range(X.shape[1])],
                  "importance": importances})
    .sort_values("importance", ascending=False)
)
print(importance_df)
```

**Feature importance.** Trained models expose `feature_importances_`, computed from the total impurity reduction (e.g. Friedman MSE) each feature brings across all splits in all trees, weighted by affected samples. Useful for quick relevance checks, but it can mislead when features are correlated or on different scales/types — SHAP values (later chapters) are a more reliable alternative.

**Outlook.** Scikit-learn's estimators are a solid, readable baseline and a good way to learn the mechanics, but XGBoost, LightGBM, and CatBoost offer better performance, optimized split-finding, and native handling of categorical/missing/sparse data.

> **Relevance to our work:** treat sklearn GBM as a reference implementation, not a competition tool. Note `loss='log_loss'` replaced the deprecated `'deviance'`, and `'squared_error'` replaced `'ls'` — keep our code on the current names. Impurity-based `feature_importances_` is fine for a sanity check but we default to SHAP for anything we report.

### 2.8 Hands-on: Building a Basic GBM Model

A full workflow on real datasets — California Housing (regression) and Breast Cancer (binary classification).

Setup:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, roc_auc_score)
from sklearn.datasets import fetch_california_housing, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
```

**Regression — California Housing.** `n_estimators` is $M$ in $F_M(x) = \sum_m \gamma_m h_m(x)$; `learning_rate` is $\nu$; `loss='squared_error'` makes the negative gradient the residual $y_i - F_{m-1}(x_i)$; `max_depth` is the main complexity control; `subsample<1.0` enables SGB.

```python
# Load and split
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train
gbr = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    subsample=0.8, loss="squared_error", random_state=42)
gbr.fit(X_train, y_train)

# Predict and evaluate
y_pred_reg = gbr.predict(X_test)
print(f"Test MSE: {mean_squared_error(y_test, y_pred_reg):.4f}")
print(f"Test R^2: {r2_score(y_test, y_pred_reg):.4f}")
```

Increasing `n_estimators` while lowering `learning_rate` usually yields a better model at the cost of longer training.

**Classification — Breast Cancer (stratified split).** `loss='log_loss'` optimizes the logistic loss (negative gradient $y - p$); `'exponential'` would give AdaBoost.

```python
# Load and split (stratified)
cancer = load_breast_cancer()
X_c = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_c = cancer.target
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)

# Train
gbc = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    subsample=0.8, loss="log_loss", random_state=42)
gbc.fit(X_c_train, y_c_train)

# Predict and evaluate
y_pred_class = gbc.predict(X_c_test)
y_pred_proba = gbc.predict_proba(X_c_test)[:, 1]
print(f"Test accuracy: {accuracy_score(y_c_test, y_pred_class):.4f}")
print(f"Test ROC AUC: {roc_auc_score(y_c_test, y_pred_proba):.4f}")
```

**Feature importance plots:**

```python
importance_df_reg = (
    pd.DataFrame({"Feature": X.columns,
                  "Importance": gbr.feature_importances_})
    .sort_values("Importance", ascending=False))

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature",
            data=importance_df_reg.head(10), palette="viridis")
plt.title("Top-10 Feature Importances (GBM Regressor)")
plt.tight_layout()
plt.show()
```

**Discussion.** This covers the core GBM workflow: instantiate, configure the theory-linked hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`), train, evaluate. Scikit-learn's estimators are valuable for understanding but not always best for large/complex data — they lack some advanced regularization, optimized split-finding, and efficient categorical/missing-value handling found in XGBoost, LightGBM, and CatBoost (next chapters).

> **Relevance to our work:** use `stratify=y` on classification splits (as shown) — it is the same reason we use `StratifiedKFold` in CV. And evaluate on the competition's actual metric (ROC AUC here), not just accuracy, since AUC is threshold-independent and matches many Kaggle leaderboards.

### Key takeaways

- **GBM = gradient descent in function space.** Each round fits a base learner to the negative gradient of the loss at the current predictions (the pseudo-residual $r_{im} = -\partial L/\partial F$), then adds it to the ensemble.
- **Pseudo-residual closed forms depend on the loss:** L2 → residual $y - F$; L1 → $\operatorname{sign}(y - F)$; Huber → residual when $|y-F|\le\delta$, else $\delta\cdot\operatorname{sign}$; logistic/deviance → $y - p$ (same shape for multiclass per class).
- **Leaf values come from a line search**, $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i)+\gamma)$ — mean for L2, median for L1 — not just averaged residuals.
- **Initialization** $F_0$ is the loss-minimizing constant: mean for L2, log-odds for log loss.
- **Shrinkage $\nu$ regularizes** by damping each tree; it trades off against `n_estimators` (low $\nu$ ⇒ many trees). Pair low learning rate with early stopping.
- **Stochastic subsampling** (rows via `subsample`, columns via `max_features`) is cheap regularization plus a speedup; ~0.5–0.8 rows, ~0.7–0.9 features are sensible.
- **Log loss beats exponential loss** for classification: robust to mislabels, probabilistic/information-theoretic grounding, smoother gradients.
- **Scikit-learn is the reference baseline** (`GradientBoostingRegressor`/`Classifier`); production/competition work moves to XGBoost/LightGBM/CatBoost. Current loss names: `'squared_error'` (was `'ls'`), `'log_loss'` (was `'deviance'`).

> **Relevance to our work:** everything here maps one-to-one onto our LightGBM practice — `objective` (loss/pseudo-residual choice), `learning_rate` + early stopping (shrinkage/n_estimators trade-off), `bagging_fraction`/`feature_fraction` (stochastic subsampling), and stratified CV on the true competition metric. The functional-gradient mental model is what makes tuning these deliberate rather than trial-and-error.

---

## Chapter 3 — Regularization in Gradient Boosting

> Source: *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 3 "Regularization in Gradient Boosting". Synthesized study notes reconstructing the KaTeX-mangled math into clean LaTeX and translating the lessons into English. Covers why boosting overfits and the full toolbox for controlling variance: tree-structure constraints, shrinkage, stochastic subsampling, the regularized objective with L1/L2 penalties on leaf weights, and early stopping, closing with a hands-on scikit-learn walkthrough.

### 3.1 Overfitting Challenges in Boosting

Gradient boosting builds models **sequentially**: each new base learner (usually a decision tree) $h_m(x)$ is fit to the pseudo-residuals / negative gradient of the loss of the current ensemble. This iterative error-correction is what makes boosting powerful, but it is also exactly what makes it prone to overfitting.

- The algorithm relentlessly minimizes **training** loss. Early rounds capture the dominant signal; later rounds fit residuals that increasingly represent **random noise** in the training sample rather than real structure. With complex base learners (deep trees) and no stopping, the model starts memorizing training peculiarities.
- **Bias–variance view:** boosting aggressively reduces *bias* by building a complex additive function. Left unconstrained, this drives *variance* very high — the model fits training data well but generalizes poorly, because its late-stage "patterns" are noise artifacts.
- The diagnostic signature: as boosting rounds increase, **training error keeps falling** while **validation error reaches a minimum and then rises**.
- Base-learner complexity matters: deep trees can isolate tiny subsets (even single points), fitting noise-specific splits with no generalization value.
- Contrast with **Random Forest (bagging):** it averages many independently trained deep trees to reduce variance. Boosting's sequential dependence has **no built-in variance-averaging mechanism**, so it needs *explicit* regularization to keep variance in check.

The takeaway that motivates the whole chapter: unconstrained boosting will almost inevitably overfit, so the regularization techniques below are essential.

### 3.2 Tree-Structure Constraints: Depth, Nodes, and Splits

Limiting the structure of the individual (weak) trees is a foundational regularizer. In boosting even shallow trees (depth ~4–8) form a strong ensemble, because complexity accumulates over many iterations.

**Maximum depth (`max_depth`)** — length of the longest root-to-leaf path.
- *Effect:* shallow trees capture only simple patterns and low-order feature interactions; deep trees model higher-order interactions.
- *Regularization:* limiting depth blocks highly specific paths tailored to a few samples, pushing the model toward patterns shared across larger subsets.
- *Trade-off:* too shallow → underfitting (high bias); too deep → overfitting (high variance). Typical values 3–10; optimum depends on dataset size, dimensionality, and the true function's complexity. Deeper trees cost more and need more data to train reliably.

**Minimum samples per leaf (`min_samples_leaf`)** — a split is valid only if **both** children retain at least this many training samples.
- Setting it > 1 prevents leaves that correspond to a handful of (possibly outlier) samples, smoothing the prediction function (especially in regression) and lowering variance.
- Too low (e.g. 1) → leaves for single samples, maximal overfitting; too high → over-constrained, underfitting.
- XGBoost uses **`min_child_weight`** instead: the minimum **sum of Hessian weights** in a leaf, not raw sample count — finer control under weighted data or specific objectives.

**Minimum samples to split (`min_samples_split`)** — minimum samples an internal node needs before it may be split.
- Acts *earlier* than `min_samples_leaf`, pruning small branches sooner. Default (2) allows splits on minimal data, raising overfitting risk.
- Rule of thumb: `min_samples_split >= 2 * min_samples_leaf` so any candidate split can actually produce valid leaves. Setting it can also speed up training slightly.

**Maximum leaf nodes (`max_leaf_nodes`)** — cap the total number of terminal nodes.
- Trees grow by maximizing impurity reduction until the leaf budget is hit, so they may grow **asymmetrically** (deeper where impurity drop is large — best-first / leaf-wise growth).
- Limits the number of distinct prediction regions. Often a **more direct complexity control than `max_depth`**; e.g. LightGBM's default leaf-wise growth uses `num_leaves`, and setting a leaf cap can make `max_depth` largely redundant.

**Minimum impurity decrease / minimum split gain (`min_impurity_decrease`, `min_split_gain`, XGBoost's `gamma`)** — a threshold on the improvement a split must provide.
- A form of **pre-pruning**: splits offering only marginal gain (likely fitting noise) are rejected. A positive value yields more conservative growth. Magnitude depends heavily on the loss and data; `gamma` is the minimum loss reduction required to make a further partition.

All of these are hyperparameters, tuned (typically via cross-validation) jointly with learning rate and number of trees.

### 3.3 Shrinkage as Implicit Regularization

The additive update at boosting round $m$ is:

$$F_m(x) = F_{m-1}(x) + \nu \, h_m(x)$$

where $F_{m-1}(x)$ is the ensemble after $m-1$ rounds, $h_m(x)$ is the new base learner fit to the previous stage's residuals/gradient, and $\nu$ (the **shrinkage** / `learning_rate` / `eta`) is a small number in $(0,1]$, e.g. 0.01–0.1.

$\nu$ acts like a learning rate, but in **function space**. Setting $\nu < 1$ deliberately slows learning: each new tree contributes only a *fraction* $\nu$ of its predicted correction rather than fully correcting the previous ensemble's error.

Why this regularizes:
- **Reduces the influence of any single tree**, so a tree that captured noise cannot dominate the final prediction; the model is less sensitive to any one base learner.
- **Requires more trees** $M$ to reach a comparable training fit than $\nu = 1$ would.
- **Better generalization via averaging:** the final $F_M(x)$ aggregates many lightly-weighted, slightly-different views. Combining many perspectives captures the underlying signal while averaging out noise.

Intuitively, low $\nu$ takes smaller, more cautious steps along the optimization path — slower but smoother and more generalizable, whereas high $\nu$ can rapidly cut training error while overshooting or overlearning noise. In the course's illustration, $\nu = 0.1$ converges slower on training but reaches a **lower validation error** than $\nu = 0.8$, which quickly overfits.

This effect is called **implicit** because shrinkage adds **no explicit penalty** to the loss and does **not directly constrain tree structure** — it modifies the boosting *process* itself, favoring solutions built from many cooperating weak learners.

In practice shrinkage is almost always used (values well below 1.0), creating the fundamental **$\nu$ ↔ $M$ trade-off**: pick a small $\nu$ (0.01–0.1), then determine the best $M$ with a validation set, usually via **early stopping** (§3.6). Very small $\nu$ raises compute cost (large $M$) but the generalization gain generally justifies it. Shrinkage is combined with tree constraints and subsampling.

### 3.4 Data Subsampling (Stochastic Gradient Boosting)

Introducing randomness through data sampling during tree construction — **Stochastic Gradient Boosting (SGB)** — borrows from Bagging and SGD. At each iteration $m$, instead of using the full training set to compute pseudo-residuals and fit the new tree, only a **random subset (sampled without replacement)** of rows is used; features can be subsampled too.

**Why it works — variance reduction:** training each tree on slightly different subsets of data/features **decorrelates** the trees. Each tree sees a slightly different view of the distribution and of the previous trees' residuals, so the ensemble is less likely to over-adapt to noise in the full set. Individual trees may be slightly weaker (higher bias), but the ensemble generalizes better.

Two forms:

- **Row subsampling (`subsample` / `bagging_fraction`):** randomly select a fraction of training rows (without replacement) before fitting each tree. Typical values 0.5–0.8. `subsample = 1.0` recovers standard GB. Below 1.0 adds randomness that fights overfitting and speeds up each iteration (less data processed); too low → underfitting or slower convergence.
- **Column (feature) subsampling:** pick a random fraction of features when building each tree or at each split. Hyperparameters:
  - `colsample_bytree` — sampled once per tree.
  - `colsample_bylevel` — sampled per tree level.
  - `colsample_bynode` (XGBoost) / `feature_fraction_bynode` (LightGBM) — sampled per node split.

  Especially effective in **high-dimensional** data with many irrelevant/redundant features; prevents over-reliance on a few highly predictive features. Reduces variance and speeds training. Typical values 0.5–1.0.

**Relation to Random Forest:** RF uses bootstrap (with-replacement) row sampling and random feature selection at each split, building trees **in parallel/independently**. SGB usually samples rows *without* replacement, offers flexible feature sampling (per-tree/level/node), and builds trees **sequentially** to correct prior errors.

**Interaction & tuning:** row/column sampling combine with shrinkage and tree constraints. A lower learning rate usually needs more `n_estimators` and can pair well with lower sampling rates; conversely higher sampling rates may allow a slightly higher learning rate or fewer trees. Optimal `subsample`, `colsample_*` are data-dependent and tuned jointly with `eta`/`learning_rate`, `max_depth`, `n_estimators` via cross-validation and early stopping (grid/random/Bayesian search — Ch. 8).

### 3.5 Regularized Objective Functions (L1/L2)

Beyond controlling structure, process (shrinkage), and data (subsampling), another approach modifies the **objective the algorithm minimizes each step**, adding an explicit complexity penalty. At iteration $t$, combine the loss with a penalty $\Omega(f_t)$ on the new tree:

$$\text{Obj}^{(t)} = \sum_{i=1}^{n} L\big(y_i,\, F_{t-1}(x_i) + f_t(x_i)\big) + \Omega(f_t)$$

Let $T$ be the number of leaves in $f_t$ and $w_j$ the output value (**weight**) of leaf $j$. The two standard penalties (from Lasso/Ridge) apply to the leaf weights:

- **L1 (Lasso):**
$$\Omega(f_t) = \alpha \sum_{j=1}^{T} |w_j|$$
Larger $\alpha$ pushes leaf weights toward zero (some exactly zero), encouraging **sparsity**. In XGBoost: `reg_alpha`.

- **L2 (Ridge):**
$$\Omega(f_t) = \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$
Larger $\lambda$ encourages smaller, more spread-out weights, preventing any single leaf from having an outsized output and making the model less sensitive to individual points in a leaf (the $\tfrac12$ is for differentiation convenience). In XGBoost: `reg_lambda`.

XGBoost's tree penalty is often written more fully as $\Omega(f) = \gamma T + \tfrac{1}{2}\lambda\sum_j w_j^2$ ( + the L1 $\alpha\sum_j|w_j|$ term), where $\gamma$ penalizes the **number of leaves** $T$ and thus prunes splits (this is the `gamma`/min-split-gain of §3.2).

**Integration via 2nd-order Taylor approximation (XGBoost):** rather than handling arbitrary $L$ directly, XGBoost expands the loss around $F_{t-1}(x_i)$:

$$L\big(y_i, F_{t-1}(x_i) + f_t(x_i)\big) \approx L\big(y_i, F_{t-1}(x_i)\big) + g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i)$$

where $g_i = \partial_{F} L$ (gradient) and $h_i = \partial^2_{F} L$ (Hessian), evaluated at $F_{t-1}(x_i)$. Dropping constants, the objective becomes:

$$\text{Obj}^{(t)} \approx \sum_{i=1}^{n}\Big[ g_i f_t(x_i) + \tfrac{1}{2} h_i f_t^2(x_i) \Big] + \alpha \sum_{j=1}^{T} |w_j| + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Regrouping the per-sample sum by leaf (all $i \in I_j$ share $f_t(x_i) = w_j$, where $I_j$ is the index set of samples in leaf $j$):

$$\text{Obj}^{(t)} \approx \sum_{j=1}^{T}\Big[ \big(\textstyle\sum_{i \in I_j} g_i\big) w_j + \tfrac{1}{2}\big(\textstyle\sum_{i \in I_j} h_i + \lambda\big) w_j^2 + \alpha |w_j| \Big]$$

**Effect on tree building:**

- **Optimal leaf weight** — for a fixed structure, with L2 only ($\alpha = 0$) there is a closed form:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$
$\lambda$ sits in the **denominator**, shrinking $w_j^*$ toward zero and damping the tree's influence. With L1 ($\alpha > 0$) the non-differentiable $|w_j|$ requires iterative/approximate solutions, but the principle (penalize large weights) holds.
- **Split-finding gain** — the gain of replacing a leaf with two children explicitly incorporates $\lambda$ and $\alpha$, so splits producing excessive/too-many large leaf weights are penalized. The algorithm favors splits that reduce loss **and** keep leaf weights controlled → better generalization.

**Practical notes:**
- $\alpha$ (`reg_alpha`) and $\lambda$ (`reg_lambda`) strengths are tuned via cross-validation alongside learning rate and depth.
- **L1 vs L2:** L2 is smoother and the usual default, reliably preventing large leaf weights. L1 can zero out leaf weights (feature-selection-like, but less direct than in linear models) and can help with high-dimensional sparse features; often both are combined.
- **Availability:** explicit L1/L2 on leaf weights is a hallmark of **XGBoost**; LightGBM and CatBoost expose `reg_alpha`/`reg_lambda` too (internals differ). Scikit-learn's `GradientBoosting*` relies instead on shrinkage, subsampling, and tree constraints.

### 3.6 Early Stopping Strategies

Boosting's additive process can continue indefinitely, fitting training noise. **Early stopping** finds the optimal number of boosting iterations by monitoring performance on a held-out validation set and stopping when it no longer improves.

How it works:
- **Data split:** at least three sets — **train** (compute gradients, build trees), **validation** (monitor for stopping only), **test** (final unbiased evaluation). The validation set should be representative of deployment data.
- **Monitor:** after each round, evaluate a chosen metric on validation (e.g. log-loss for classification, RMSE for regression).
- **Stop condition:** if the validation score fails to improve for a preset number of consecutive rounds, stop. This "patience" parameter — `early_stopping_rounds` — prevents premature stops from small random fluctuations.
- **Model selection:** typically return the model from the **best-scoring iteration**, not necessarily the last one before stopping.

As regularization, early stopping caps model capacity: adding trees increases capacity, and it halts at the point beyond which added capacity mostly captures noise. In the course's curve the validation minimum is around iteration 80, with a dashed patience window afterward.

Most libraries (XGBoost, LightGBM, CatBoost) support it in `fit` by supplying an eval set, an eval metric, and the patience (`early_stopping_rounds`). Example (older XGBoost sklearn API):

```python
# Example (XGBoost API)
eval_set = [(X_train, y_train), (X_val, y_val)]

model.fit(X_train, y_train,
          eval_set=eval_set,
          eval_metric='logloss',      # or 'rmse', etc.
          early_stopping_rounds=10,   # stop if val log-loss doesn't improve for 10 rounds
          verbose=True)               # see per-round performance
```

This **simplifies tuning**: set `n_estimators` to a comfortably large value and let early stopping find the best stopping point instead of hand-tuning the tree count.

Caveats:
- **Validation set size/quality:** small or unrepresentative validation sets give noisy estimates and suboptimal stops.
- **Metric choice:** align the early-stopping metric with the real modeling objective.
- **Patience:** too low → premature stops from noise; too high → some overfitting before stopping. Usually less critical to tune than `n_estimators` directly.
- **Interaction with learning rate:** smaller learning rates need more rounds, so early stopping will suggest more iterations when $\nu$ is low.

### 3.7 Hands-On: Applying Regularization

Using scikit-learn's `GradientBoostingClassifier` on a synthetic, overfitting-prone dataset to observe each technique's effect. (Note: sklearn's GBM does **not** implement L1/L2 on leaf weights like XGBoost; it regularizes via tree constraints, shrinkage, and subsampling.)

**Setup — synthetic data with label noise (`flip_y=0.1`):**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=10, n_redundant=5,
                           n_clusters_per_class=2, flip_y=0.1,
                           random_state=42)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
```

**Baseline — likely overfitting** (many estimators, relatively deep trees, no explicit constraints):

```python
gbm_baseline = GradientBoostingClassifier(n_estimators=300,
                                          learning_rate=0.1,
                                          max_depth=5,   # relatively deep trees
                                          random_state=42)
gbm_baseline.fit(X_train, y_train)

y_train_pred_baseline = gbm_baseline.predict(X_train)
y_val_pred_baseline   = gbm_baseline.predict(X_val)
y_train_proba_baseline = gbm_baseline.predict_proba(X_train)[:, 1]
y_val_proba_baseline   = gbm_baseline.predict_proba(X_val)[:, 1]
# High train accuracy + low val accuracy = classic overfitting signature.
```

A large gap between train and validation metrics (high train accuracy, lower validation accuracy; low train log-loss, higher validation log-loss) signals overfitting.

**1. Tree constraints (`max_depth`, `min_samples_leaf`):**

```python
gbm_tree_reg = GradientBoostingClassifier(n_estimators=300,
                                          learning_rate=0.1,
                                          max_depth=3,          # shallower trees
                                          min_samples_leaf=10,  # more samples per leaf
                                          random_state=42)
gbm_tree_reg.fit(X_train, y_train)
```
Expect slightly lower **train** performance but improved **validation** performance (a narrower train–val gap = better generalization).

**2. Shrinkage (`learning_rate`)** — lower rate, more estimators to compensate:

```python
gbm_shrinkage = GradientBoostingClassifier(n_estimators=600,   # more estimators
                                           learning_rate=0.05,  # lower learning rate
                                           max_depth=3,
                                           min_samples_leaf=10,
                                           random_state=42)
gbm_shrinkage.fit(X_train, y_train)
```
Lower learning rate usually yields smoother convergence and better validation results, provided `n_estimators` is raised accordingly.

**3. Subsampling (`subsample`, `max_features`)** — stochastic gradient boosting:

```python
gbm_subsample = GradientBoostingClassifier(n_estimators=600,
                                           learning_rate=0.05,
                                           max_depth=3,
                                           min_samples_leaf=10,
                                           subsample=0.7,     # 70% of rows per tree
                                           max_features=0.8,  # 80% of features per split
                                           random_state=42)
gbm_subsample.fit(X_train, y_train)
```
Often improves stability and validation scores, especially with high-variance or correlated features.

**4. Early stopping** — via sklearn's `validation_fraction`, `n_iter_no_change`, `tol`:

```python
gbm_early_stop = GradientBoostingClassifier(n_estimators=1000,        # high potential max
                                            learning_rate=0.05,
                                            max_depth=3,
                                            min_samples_leaf=10,
                                            subsample=0.7,
                                            max_features=0.8,
                                            validation_fraction=0.2,   # 20% of train for internal validation
                                            n_iter_no_change=10,       # stop after 10 non-improving iters
                                            tol=0.0001,
                                            random_state=42)
gbm_early_stop.fit(X_train, y_train)
print(f"Best n_estimators found: {gbm_early_stop.n_estimators_}")

# Alternative: manually plot validation error vs. iterations (no auto early stopping)
gbm_manual_es = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
                                           max_depth=3, random_state=42)
gbm_manual_es.fit(X_train, y_train)

# Staged log-loss = performance after each iteration
staged_val_loss   = [log_loss(y_val,   proba[:, 1]) for proba in gbm_manual_es.staged_predict_proba(X_val)]
staged_train_loss = [log_loss(y_train, proba[:, 1]) for proba in gbm_manual_es.staged_predict_proba(X_train)]
best_iteration = np.argmin(staged_val_loss) + 1   # +1 since counting from 1

print(f"Lowest validation log-loss at iteration: {best_iteration}")
print(f"Validation log-loss at best iteration: {staged_val_loss[best_iteration-1]:.4f}")
```

`staged_predict_proba` exposes per-iteration predictions, letting you locate the exact iteration where validation loss bottoms out and turns upward. Early stopping automates finding a good `n_estimators`, halting once added trees start hurting generalization.

**Comparison table** (fill values from an actual run):

| Regularization method | Val accuracy | Val log-loss | Notes |
|---|---|---|---|
| Baseline (overfitting) | (run) | (run) | high `max_depth`, no explicit constraints |
| Tree constraints | (run) | (run) | `max_depth=3`, `min_samples_leaf=10` |
| + Shrinkage | (run) | (run) | lower `learning_rate=0.05`, more `n_estimators` |
| + Subsampling | (run) | (run) | `subsample=0.7`, `max_features=0.8` |
| + Early stopping (auto) | (run) | (run) | auto-finds optimal `n_estimators` |

Applying regularization generally raises validation accuracy and lowers validation log-loss versus the overfitting baseline, and the **combination** of tree constraints + shrinkage + subsampling + early stopping usually gives the best result.

### Key takeaways

- Boosting reduces bias aggressively but has **no built-in variance control** (unlike bagging's averaging); explicit regularization is mandatory. Watch the train-vs-validation divergence as the overfitting signature.
- **Tree constraints** (`max_depth`, `min_samples_leaf` / `min_child_weight`, `num_leaves`/`max_leaf_nodes`, `gamma`/min-split-gain) limit per-tree complexity so no tree isolates noise.
- **Shrinkage** ($\nu$, `learning_rate`) is implicit regularization: small $\nu$ down-weights each tree and forces the ensemble to average many views — trading off against a larger $M$ (`n_estimators`).
- **Stochastic subsampling** (row `subsample`, column `colsample_*`) decorrelates trees to cut variance and also speeds training.
- The **regularized objective** $\text{Obj}^{(t)} \approx \sum_i[g_i f_t + \tfrac12 h_i f_t^2] + \Omega(f_t)$ with $\Omega(f)=\gamma T + \tfrac12\lambda\sum_j w_j^2 + \alpha\sum_j|w_j|$ penalizes leaf count and leaf-weight magnitude. L2 ($\lambda$/`reg_lambda`) appears in the closed-form optimal leaf weight $w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$, shrinking outputs; L1 ($\alpha$/`reg_alpha`) drives sparsity. Penalties also enter the split-gain computation.
- **Early stopping** operationalizes the `learning_rate` × `n_estimators` trade-off: fix a large `n_estimators`, small learning rate, and let a patience (`early_stopping_rounds` / `n_iter_no_change`) on a validation metric pick the best round. Lower learning rate ⇒ more optimal rounds.
- Best practice is to **combine** these levers and tune them jointly by cross-validation.

> **Relevance to our work:** Our LightGBM tree-search pipeline uses leaf-wise growth, so `num_leaves` (with `min_child_samples`/`min_child_weight`) is the primary complexity knob per §3.2 — more direct than `max_depth`. The `learning_rate` × `n_estimators` trade-off (§3.3, §3.6) is exactly why we hold a small learning rate and let early stopping choose the round count on the OOF/CV fold rather than fixing tree count — keeping this the arbiter of capacity guards against the train-vs-validation divergence described in §3.1.

> **Relevance to our work:** `feature_fraction`/`bagging_fraction` (§3.4) and `reg_alpha`/`reg_lambda` (§3.5) are core search dimensions for our regularization arm; L2 in particular is the safe default leaf-weight regularizer. When comparing candidates, judge them on validation/CV metrics (never training fit) — the §3.7 baseline-vs-regularized comparison is the template for our per-candidate CV gate.

> **Relevance to our work:** Deterministic, reproducible early-stopping requires stable OOF folds and fixed seeds; recall our LightGBM determinism note (`deterministic`/`force_row_wise`/`num_threads`) so the "best iteration" chosen by early stopping is reproducible across runs of the tree search.

---

## Chapter 4 — XGBoost: Extreme Gradient Boosting

> Source: *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 4, plus Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System" (KDD '16, DOI 10.1145/2939672.2939785) and the XGBoost documentation. Synthesized study notes for the Kaggle gradient-boosting knowledge base. Math reconstructed from KaTeX-mangled source text.

XGBoost is a scalable, regularized re-engineering of gradient boosting built by Tianqi Chen. Its edge over textbook GBM comes from three coordinated ideas: (1) a **regularized second-order objective** that folds complexity control directly into what each tree optimizes; (2) **efficient split-finding algorithms** (exact greedy, approximate quantile sketch, sparsity-aware); and (3) **systems engineering** (parallel split search, cache-aware access, out-of-core blocks) that makes it fast on large data.

---

### 4.1 Motivation and enhancements over standard GBM

Standard GBM is powerful but has practical limits, especially on very large datasets:

- **Computational cost.** Enumerating every candidate split for every feature at every node becomes a bottleneck.
- **Heuristic regularization.** Shrinkage and subsampling help against overfitting, but they are bolted on *around* the boosting step rather than tied to the objective that grows each tree.

XGBoost addresses these with deliberate design choices:

**Formalized regularization.** Instead of controlling complexity only through post-hoc constraints (max depth, shrinkage), XGBoost adds penalties resembling $L_1$ (Lasso) and $L_2$ (Ridge) *directly into the objective optimized while building each tree*. Split selection and leaf-value computation therefore explicitly trade loss reduction against model complexity (number of leaves and magnitude of leaf weights) — a more principled overfitting control than depth heuristics alone.

**Advanced split-finding.**
- *Exact greedy* — still greedy, but evaluates candidate splits efficiently.
- *Approximate* — for datasets too large to enumerate all split points, proposes candidates from feature-value quantiles (percentiles) and only evaluates splits at those points, sharply cutting computation.
- *Sparsity-aware* — handles missing values natively by learning a **default direction** per node during training (no pre-imputation), choosing the direction (left/right) that maximizes gain.

**System efficiency.**
- *Parallelization* — multiple CPU cores are used during tree construction, primarily in the split-finding stage.
- *Cache-aware access* — internal data structures and algorithms are laid out to use CPU cache well and minimize memory-access latency; data is stored in blocks aligned to cache lines.
- *Out-of-core computation* — datasets larger than RAM are processed in blocks streamed from disk, scaling to very large data.

Together these make XGBoost markedly faster and more scalable than traditional GBM while its built-in regularization typically improves generalization.

> **Relevance to our work:** XGBoost's regularization-in-the-objective is exactly why it is a strong default on tabular Kaggle data — you get principled complexity control plus built-in missing-value handling out of the box.

---

### 4.2 The regularized (second-order) learning objective

Boosting builds the ensemble additively. At step $t$, a new tree $f_t(x)$ is added to the previous prediction:

$$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta\, f_t(x)$$

where $\eta$ is the learning rate (`eta` / `learning_rate`). XGBoost finds the tree $f_t$ that optimizes:

$$\text{Obj}^{(t)} = \sum_{i=1}^{n} l\!\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \sum_{k=1}^{t} \Omega(f_k)$$

Here $l(\cdot)$ is a differentiable loss (squared error for regression, log loss for classification) and $\Omega(f_k)$ penalizes tree complexity. Since the penalties of $f_1,\dots,f_{t-1}$ are constant at step $t$, the objective for choosing $f_t$ simplifies to $\sum_i l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$.

**Second-order Taylor expansion.** Optimizing a general tree against this directly is hard, so XGBoost expands the loss to second order around the current prediction $\hat{y}_i^{(t-1)}$:

$$l\!\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) \approx l\!\left(y_i, \hat{y}_i^{(t-1)}\right) + g_i\, f_t(x_i) + \tfrac{1}{2} h_i\, f_t^2(x_i)$$

where $g_i$ and $h_i$ are the first- and second-order derivatives (gradient and Hessian) of the loss at the previous prediction:

$$g_i = \left.\frac{\partial\, l(y_i, \hat{y})}{\partial \hat{y}}\right|_{\hat{y}=\hat{y}_i^{(t-1)}}, \qquad h_i = \left.\frac{\partial^2\, l(y_i, \hat{y})}{\partial \hat{y}^2}\right|_{\hat{y}=\hat{y}_i^{(t-1)}}$$

Using the Hessian (not just the gradient, as in classic GBM) gives more information about the loss curvature, typically yielding faster convergence and better accuracy — and it generalizes cleanly to custom losses. Dropping the constant $l(y_i, \hat{y}_i^{(t-1)})$, the step-$t$ objective to minimize becomes:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n} \left[ g_i\, f_t(x_i) + \tfrac{1}{2} h_i\, f_t^2(x_i) \right] + \Omega(f_t)$$

**The regularization term.** For a tree with $T$ leaves and leaf weights $w_j$ (leaf $j$'s prediction score), where $q(x)$ maps an instance to its leaf index:

$$\Omega(f_t) = \gamma T + \tfrac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

- $\gamma T$ — penalty proportional to the number of leaves. A higher $\gamma$ (parameter `gamma`) makes the algorithm more conservative, requiring a larger loss reduction to justify a new leaf (a split). It behaves like pruning folded into the objective — effectively $L_0$ regularization on leaf count.
- $\tfrac{1}{2}\lambda \sum_j w_j^2$ — $L_2$ penalty on leaf weights. A higher $\lambda$ (`reg_lambda`) shrinks weights toward zero, reducing sensitivity to individual observations and any single tree's influence.

XGBoost also supports $L_1$ on weights, $\alpha \sum_j |w_j|$ (`reg_alpha`), which induces sparsity in leaf scores; the full term is $\Omega(f_t) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2 + \alpha \sum_j |w_j|$.

**Optimal leaf weight and structure score.** Group the sum by leaf. Let $I_j = \{i \mid q(x_i)=j\}$, $G_j = \sum_{i\in I_j} g_i$, $H_j = \sum_{i\in I_j} h_i$. Then:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} \left[ G_j w_j + \tfrac{1}{2}(H_j + \lambda) w_j^2 \right] + \gamma T$$

For a **fixed** structure $q$ this is a sum of independent quadratics in $w_j$. Setting $\partial / \partial w_j = G_j + (H_j + \lambda)w_j = 0$ gives the optimal leaf weight:

$$w_j^* = -\frac{G_j}{H_j + \lambda} = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}$$

Substituting back yields the **structure score** — the minimal objective achievable by structure $q$:

$$\tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

This score (lower is better) is what XGBoost uses to evaluate candidate splits during tree construction; the algorithm greedily picks splits that reduce it most.

> **Relevance to our work:** $w_j^* = -G_j/(H_j+\lambda)$ and the structure score are the two formulas to internalize — every leaf value and every split decision in XGBoost derives from them, and they explain precisely what `lambda`, `gamma`, and `min_child_weight` (a floor on $H_j$) are doing.

---

### 4.3 Exact greedy split finding

For each candidate node the algorithm must find the best (feature, split point) — the split that most reduces the objective, i.e. the largest **gain**. The exact greedy method is called *exact* because it evaluates every possible split point of every feature for the instances at the node, and *greedy* because it takes the locally optimal split at each node.

Consider splitting a node's instances $I = I_L \cup I_R$. Comparing the no-split score with the two-child score, and keeping only the improvement in the loss part minus the cost $\gamma$ of adding one leaf, the split gain is:

$$\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

A split is worthwhile only if the gain is positive (more precisely, at least `min_split_loss`, the API name for $\gamma$). The three terms are: score of the left child, score of the right child, minus score of the parent; $\gamma$ is the complexity cost of turning one leaf into two.

**Algorithm (per node):**

1. Initialize `max_gain = 0`.
2. For each feature $k = 1 \dots d$:
   - Let $I$ be the instances at the node; compute totals $G = \sum_{i\in I} g_i$, $H = \sum_{i\in I} h_i$.
   - Sort the instances by feature $k$'s value.
   - Initialize $G_L = 0$, $H_L = 0$.
   - Scan left→right over the sorted instances $i$:
     - $G_L \mathrel{+}= g_i$, $H_L \mathrel{+}= h_i$; then $G_R = G - G_L$, $H_R = H - H_L$.
     - Evaluate the gain of splitting between the current and next instance (i.e. at each distinct sorted value):
       $$\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda} \right] - \gamma$$
     - If gain > `max_gain`, record it and the (feature, split point).
3. If `max_gain > 0` split the node on the best (feature, split point); otherwise make it a leaf.

**Cost.** The bottleneck is sorting the feature values at each node. With $n$ instances and $d$ features: sorting is $O(n \log n)$ per feature, the scan is $O(n)$, so a node costs about $O(d \cdot n \log n)$. This becomes expensive with many instances, many features, or deep trees — which motivates the approximate method.

> **Relevance to our work:** the single-pass left→right accumulation of $(G_L, H_L)$ with $G_R = G - G_L$ is the core trick that makes gain evaluation cheap once sorted; this is the same gain formula our tree-search priors work reasons about.

---

### 4.4 Approximate split finding (weighted quantile sketch)

Exact greedy guarantees the best split among all values of a continuous feature but scales poorly ($O(n\log n)$ sorting per node, plus cache misses from repeatedly touching large sorted data). The **approximate greedy** algorithm reduces the number of candidate split points evaluated.

**Histogram / bucketing idea.** Discretize each continuous feature into a fixed number of buckets, typically by the feature's **quantiles (percentiles)**, and only consider splits *between buckets*. For, say, 256 buckets the algorithm finds ~255 split points dividing the data into roughly equal-sized groups.

**Aggregated statistics.** For each bucket $k$ (index set $I_k$), precompute and store:

$$G_k = \sum_{i\in I_k} g_i, \qquad H_k = \sum_{i\in I_k} h_i$$

**Evaluate candidates at bucket boundaries.** For a split after bucket $j$, prefix-sum the buckets:

$$G_L = \sum_{k=1}^{j} G_k,\quad H_L = \sum_{k=1}^{j} H_k, \qquad G_R = \sum_{k=j+1}^{N_{\text{bins}}} G_k,\quad H_R = \sum_{k=j+1}^{N_{\text{bins}}} H_k$$

then plug into the standard gain formula:

$$\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

and pick the boundary with maximum gain. (The quantiles are Hessian-**weighted** — hence the "weighted quantile sketch" — so each bucket carries roughly equal total $h$, which matches the $\tfrac{1}{2} h_i$ weighting of the second-order loss.)

**Global vs. local proposals.** XGBoost offers two variants for *when* candidate split points (bucket boundaries) are proposed:

- **Global** — propose once at the start of growing a tree (or before training); reuse the same buckets for every node in that tree. Cheaper (bucketing happens once per feature) and lower memory; accuracy can drop if the data distribution shifts across branches.
- **Local** — re-propose candidates after every split, recomputing quantiles from just the instances at that node. More compute and memory, but adapts to local distributions and can be more accurate, especially deep in the tree.

In practice the global variant is usually sufficient and much faster. The **number of buckets acts as a regularizer**: fewer buckets → coarser splits, faster, more regularized; more buckets → finer splits approaching exact greedy at higher cost. XGBoost's `sketch_eps` controls granularity, with roughly `1 / sketch_eps` buckets. This changes split-finding cost from *number of unique values* to *number of buckets* (usually far smaller), enabling much larger datasets.

> **Relevance to our work:** the `hist`/`approx` tree methods are the practical default for large Kaggle datasets; bucket count is a genuine, cheap regularization knob, and this histogram approach is exactly what LightGBM builds its whole design around (Chapter 5).

---

### 4.5 Sparsity-aware split finding (default directions for missing values)

Datasets commonly contain missing values. Rather than requiring imputation or discarding information, XGBoost learns, at every split, a **default direction** for instances whose split feature is missing — folded directly into split finding to maximize gain.

**Mechanism.** When evaluating split candidate $(j, v)$ on feature $j$ at value $v$:

1. **Partition the non-missing instances** into $I_L = \{i : x_{ij} < v\}$ and $I_R = \{i : x_{ij} \ge v\}$, and compute $G_L, H_L, G_R, H_R$ from these only.
2. **Aggregate the missing instances** $I_{\text{missing}} = \{i : x_{ij} \text{ missing}\}$: $G_{\text{missing}} = \sum g_i$, $H_{\text{missing}} = \sum h_i$.
3. **Score "default = left"** (send all missing to the left child):

$$\text{Gain}_{\text{left}} = \frac{1}{2}\left[ \frac{(G_L+G_{\text{missing}})^2}{H_L+H_{\text{missing}}+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R+G_{\text{missing}})^2}{H_L+H_R+H_{\text{missing}}+\lambda} \right] - \gamma$$

4. **Score "default = right"** (send all missing to the right child):

$$\text{Gain}_{\text{right}} = \frac{1}{2}\left[ \frac{G_L^2}{H_L+\lambda} + \frac{(G_R+G_{\text{missing}})^2}{H_R+H_{\text{missing}}+\lambda} - \frac{(G_L+G_R+G_{\text{missing}})^2}{H_L+H_R+H_{\text{missing}}+\lambda} \right] - \gamma$$

5. **Pick the higher-gain direction** as the learned default for this candidate; its gain is the split's gain. Doing this over all features and split points, the node chooses the overall max-gain (feature, value, default direction).

**Advantages.** No pre-imputation; data-driven handling (the model learns the loss-optimal placement rather than a mean/median heuristic); efficient (only the two aggregate directions are tried, not every per-instance assignment); and it naturally fits differing missingness patterns across features.

**At prediction time**, an instance with a missing split feature simply follows the stored default direction for that node.

> **Relevance to our work:** feed NaNs straight into XGBoost (via `DMatrix(..., missing=np.nan)`) and let it learn directions — often better than imputing. This also matters for engineered sparse features (one-hot, counts) where "missing" carries signal.

---

### 4.6 System optimizations: cache-aware access, blocks, and parallelism

Beyond algorithms, much of XGBoost's speed comes from hardware-aware systems design.

**Parallel tree construction.** The most expensive part of building one tree is finding the best split across all features at each node. XGBoost parallelizes the **outer loop over features**: different features are assigned to different threads, each computing its feature's best split and gain, then a synchronization step reduces these to the node's overall best split. Note this is *intra-tree* parallelism (within a single tree). Boosting itself stays **sequential** across trees — each tree depends on the previous tree's residuals — so trees are not built simultaneously.

**Column block data structure.** Data is stored in an in-memory compressed **column (block) layout**: each feature column is pre-sorted by its value with pointers to the corresponding instances (and their $g_i$, $h_i$). This single up-front sort lets each per-feature thread access what it needs largely sequentially during split finding.

**Cache-aware access.** Modern CPUs rely on L1/L2/L3 caches far faster than RAM; a naive implementation accessing gradient/Hessian statistics in the scattered order dictated by feature values suffers cache misses. XGBoost uses **cache-aware prefetching**: small per-thread buffers into which the $g$/$h$ statistics for an upcoming block of split evaluations are prefetched. Because the buffers are small they tend to stay in cache, so the split-gain computations read from cache instead of RAM.

**Benefits of the block structure (summary):**

- Compressed storage → smaller memory footprint.
- Pre-sorted feature values → efficient split finding for both exact and approximate methods.
- Sequential access to $g$/$h$ when scanning a feature → better cache locality.
- Independent blocks → basis for parallelism across features/threads.
- **Out-of-core support** → blocks can be compressed and stored on disk, then loaded on demand (block sharding + compression), letting XGBoost handle datasets larger than RAM.

> **Relevance to our work:** set `nthread`/`n_jobs` to use all cores, and reach for out-of-core / `hist` when data exceeds memory. These systems tricks are why XGBoost trains in minutes where a naive GBM would take hours.

---

### 4.7 The XGBoost API and key parameters

Parameters fall into three groups: **general**, **booster**, and **learning-task**. The native Python API and the Scikit-learn wrapper share functionality but sometimes differ in names (`eta` ↔ `learning_rate`, `lambda` ↔ `reg_lambda`, `alpha` ↔ `reg_alpha`, `num_boost_round` ↔ `n_estimators`, `nthread` ↔ `n_jobs`, `seed` ↔ `random_state`).

**General parameters** — control the overall run. `booster` (default `gbtree`; also `gblinear`, `dart`), `verbosity` (0 silent … 3 debug), `nthread` / `n_jobs` (parallel threads).

**Booster parameters (gbtree)** — control each tree and are the primary overfitting levers (learning rate, tree-structure controls, regularization, subsampling); detailed in the table below.

**Learning-task parameters** — define the objective and evaluation metric:

- `objective` (default `reg:squarederror`): e.g. `reg:squarederror` (regression), `reg:logistic`, `binary:logistic` (probabilities), `binary:logitraw` (pre-logit score), `multi:softmax` (needs `num_class`, outputs class), `multi:softprob` (needs `num_class`, outputs probability vector), `rank:pairwise` (learning-to-rank).
- `eval_metric`: `rmse`, `mae` (regression); `logloss`, `error`, `merror`, `auc` (classification); `map`, `ndcg` (ranking). Multiple metrics may be given; the **last** one drives early stopping.

**Training-control parameters** — `num_boost_round` / `n_estimators` (number of trees; too few underfits, too many overfits — early stopping mitigates) and `early_stopping_rounds` (stop if the validation metric does not improve for that many consecutive rounds; requires an eval set).

#### Key parameter reference

| Parameter (native / sklearn) | Default | Role |
| --- | --- | --- |
| `eta` / `learning_rate` | 0.3 | Shrinkage on each new tree; lower → more trees, less overfitting, needs larger `num_boost_round`. |
| `max_depth` | 6 | Max tree depth; higher → more complex, higher overfit risk & cost. Typical 3–10. |
| `min_child_weight` | 1 | Minimum sum of instance Hessian ($\sum h_i$) required in a child; larger → more conservative. A regularizer (min samples per node for squared-error loss). |
| `gamma` / `min_split_loss` | 0 | Minimum loss reduction (gain) required to split; the $\gamma$ in the objective. Larger → fewer splits. |
| `lambda` / `reg_lambda` | 1 | $L_2$ penalty on leaf weights (Ridge-like); larger → more conservative. |
| `alpha` / `reg_alpha` | 0 | $L_1$ penalty on leaf weights (Lasso-like); induces sparsity, useful in high dimensions. |
| `subsample` | 1 | Fraction of rows sampled per tree (stochastic GB); 0.7–0.8 reduces variance. |
| `colsample_bytree` | 1 | Fraction of features sampled per tree (most commonly tuned column-sampler). |
| `colsample_bylevel` | 1 | Fraction of features sampled per depth level. |
| `colsample_bynode` | 1 | Fraction of features sampled per node/split. |
| `tree_method` | `auto` | Split algorithm: `exact`, `approx` (quantile), `hist` (histogram, fast/low-memory). |
| `scale_pos_weight` | 1 | Balances positive/negative weights for imbalanced classification; typical value = sum(neg)/sum(pos). |
| `num_boost_round` / `n_estimators` | — | Number of boosting rounds (trees). |
| `early_stopping_rounds` | — | Stop after this many rounds without eval-metric improvement (needs an eval set). |

**Practical tuning workflow (from the source):** start from defaults and establish a cross-validated baseline; tune `eta` and `n_estimators` together (lower `eta` → more rounds, use early stopping to find the count); control tree complexity with `max_depth` (biggest effect), then `min_child_weight`, then `gamma`; add randomness via `subsample` and `colsample_bytree`; apply `lambda`/`alpha` for extra regularization; switch `tree_method` from `exact` to `hist`/`approx` on large data for big speedups at little accuracy cost.

> **Relevance to our work:** this table is the working knob set for our XGBoost baselines. The high-leverage first pass is `eta` + early stopping, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`; systematic hyperparameter search (Optuna etc.) comes in Chapter 8.

---

### 4.8 Hands-on: implementing XGBoost

End-to-end binary classification on the Breast Cancer dataset using the native `DMatrix` + `xgb.train` API, plus the Scikit-learn wrapper.

**Setup and imports.**

```python
# uv pip install xgboost pandas numpy scikit-learn matplotlib
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
```

**Data preparation.** XGBoost accepts NumPy/pandas directly, but its optimized `DMatrix` structure is recommended for memory and speed; it also handles missing values natively when you declare a missing indicator (e.g. `missing=np.nan`).

```python
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Target distribution:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# DMatrix pre-processes into XGBoost's internal format.
# For data with missing values, use DMatrix(data, label=..., missing=np.nan).
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
```

**Configure and train.** Parameters map directly to the theory (learning rate/shrinkage, tree-complexity controls, $L_1$/$L_2$ regularization).

```python
params = {
    # Learning task
    "objective": "binary:logistic",   # outputs probabilities
    "booster": "gbtree",
    "eval_metric": ["logloss", "auc"],  # last metric ('auc') drives early stopping

    # Booster
    "eta": 0.1,               # learning rate / shrinkage
    "max_depth": 3,           # limit single-tree complexity
    "subsample": 0.8,         # row sampling
    "colsample_bytree": 0.8,  # column sampling
    "gamma": 0,               # min loss reduction to split
    "lambda": 1,              # L2 penalty (reg_lambda)
    "alpha": 0,               # L1 penalty (reg_alpha)

    "seed": 42,
}

watchlist = [(dtrain, "train"), (dtest, "eval")]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=watchlist,
    early_stopping_rounds=10,  # stop if eval AUC stalls for 10 rounds
    verbose_eval=20,
)
```

Notes: `eval_metric`'s last entry (`auc`) is used for early stopping; `early_stopping_rounds=10` returns the model at the best iteration and guards against overfitting late trees; `subsample`/`colsample_bytree` at 0.8 add the stochasticity of stochastic gradient boosting.

**Predict and evaluate.** With `binary:logistic`, `predict` returns probabilities; threshold at 0.5 for labels. Use `best_iteration` to avoid later, possibly overfit, trees.

```python
y_pred_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
y_pred_labels = (y_pred_proba > 0.5).astype(int)

print(f"Best iteration: {bst.best_iteration}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_labels):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, y_pred_labels, target_names=cancer.target_names))
```

**Feature importance.** Types: `weight` (split count), `gain` (average gain over splits using the feature — usually preferred), `cover` (average number of samples affected).

```python
importance_type = "gain"  # or 'weight', 'cover'
scores = bst.get_score(importance_type=importance_type)
feat_importances = pd.Series(scores).sort_values(ascending=False)

top_n = 15
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(bst, ax=ax, max_num_features=top_n, importance_type=importance_type)
plt.title(f"Top {top_n} feature importances (type={importance_type})")
plt.tight_layout()
plt.show()
```

**Scikit-learn wrapper.** `XGBClassifier` / `XGBRegressor` plug into Pipelines and `GridSearchCV`/`RandomizedSearchCV`. Same parameters, passed at construction; train with `.fit()` on arrays/DataFrames.

```python
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=100,       # ~ num_boost_round
    learning_rate=0.1,      # ~ eta
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,            # L1
    reg_lambda=1,           # L2
    random_state=42,
    early_stopping_rounds=10,  # constructor arg in modern XGBoost
)

xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print(f"Best iteration: {xgb_clf.best_iteration}")
y_pred_proba_skl = xgb_clf.predict_proba(X_test)[:, 1]
y_pred_labels_skl = xgb_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_labels_skl):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba_skl):.4f}")

importances_skl = xgb_clf.feature_importances_
```

> Note on API drift: in the course code `early_stopping_rounds` (and the deprecated `use_label_encoder`) are passed inside `.fit()`. In current XGBoost (2.x) `early_stopping_rounds` is a constructor argument (as shown above) and `use_label_encoder` is gone. The native `DMatrix` + `xgb.train` path often gives slightly better performance and more direct control for very large or highly customized workloads.

> **Relevance to our work:** this is the canonical baseline template for our tabular pipelines — `DMatrix` with `missing=np.nan`, early stopping on a held-out eval set, `gain`-based importance for feature triage, and the sklearn wrapper when we need Pipeline/CV integration. Systematic tuning of these params is Chapter 8.

---

### Key takeaways

- **Regularized second-order objective is the heart of XGBoost.** It minimizes $\tilde{\mathcal{L}}^{(t)} = \sum_i [g_i f_t(x_i) + \tfrac12 h_i f_t^2(x_i)] + \Omega(f_t)$ with $\Omega = \gamma T + \tfrac12\lambda\sum_j w_j^2 (+\alpha\sum_j|w_j|)$, using both gradient $g_i$ and Hessian $h_i$.
- **Everything derives from two formulas:** optimal leaf weight $w_j^* = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}$ and the structure score $-\tfrac12\sum_j \frac{G_j^2}{H_j+\lambda} + \gamma T$, which yields the split gain $\tfrac12[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma$.
- **Split finding scales via approximation.** Exact greedy is $O(d\, n\log n)$ per node; the approximate method buckets features by (Hessian-weighted) quantiles and evaluates only bucket boundaries, with global vs. local proposal variants and bucket count as a regularizer.
- **Sparsity-aware default directions** handle missing values natively by learning left/right placement that maximizes gain — no imputation needed.
- **Systems engineering** (feature-parallel split search, pre-sorted compressed column blocks, cache-aware prefetch, out-of-core sharding) turns the algorithm into a fast, scalable tool.
- **Practical control** comes from `eta` + early stopping, `max_depth`/`min_child_weight`/`gamma`, `subsample`/`colsample_bytree`, and `lambda`/`alpha`, with `tree_method=hist` for large data.

---

## Chapter 5 — LightGBM: Light Gradient Boosting Machine

> Source: "Mastering Gradient Boosting Algorithms" (apxml.com), Chapter 5. Synthesized study notes for the Kaggle GB knowledge base — clean English restatement of the (Chinese, KaTeX-mangled) course text, with reconstructed math and cleaned Python. LightGBM (Ke et al., NeurIPS 2017) is the framework of record here and our project's primary learner.

### 5.1 Motivation: Overcoming XGBoost's Limitations at Scale

XGBoost was a major advance over classic GBM: it folds regularization directly into the objective and adds system-level optimizations (sparsity-aware split finding, parallelism). But as datasets grow in both the number of instances $N$ and the number of features $M$, even XGBoost hits computational walls. The bottleneck is **split finding**.

XGBoost's exact greedy algorithm, for each node and each feature, must:

1. Sort instances by feature value.
2. Scan every candidate split point (between adjacent sorted values).
3. Compute a regularized gain for each candidate.
4. Repeat across all features.

This exhaustive scan costs roughly $O(N \times M)$ per split in the dense case. Even with pre-sorting, caching, or histogram-based approximation, the fundamental need to scan large volumes of data/feature values remains. The cost shows up as:

- **Training time** — repeatedly walking data points and features for every tree dominates when there are millions/billions of rows or tens of thousands of features.
- **Memory footprint** — storing sorted feature values and per-instance gradient statistics can exceed a single machine's RAM.

These limits bite hardest on web-scale, high-dimensional genomic, or heavily engineered/sparse-feature problems. LightGBM was designed from the ground up for efficiency, introducing four key ideas:

- **GOSS** (Gradient-based One-Side Sampling) — cut the number of instances considered in split finding.
- **EFB** (Exclusive Feature Bundling) — cut the effective number of features to scan.
- **Histogram-based algorithm** — discretize feature values into bins to speed split finding and shrink memory.
- **Leaf-wise (best-first) tree growth** — a faster-converging growth strategy that needs careful complexity control.

### 5.2 Gradient-based One-Side Sampling (GOSS)

Much of the per-tree cost comes from evaluating splits over *all* instances. GOSS reduces the instance count intelligently, exploiting the fact that not all training instances contribute equally.

**Key observation.** In gradient boosting each new tree fits the negative gradient of the loss w.r.t. current predictions. An instance's gradient magnitude $|g_i|$ is essentially how *wrong* the current ensemble still is on it. Large-gradient instances are under-trained and information-rich; small-gradient instances are already well-predicted and contribute diminishing returns while still costing compute. Naively dropping the small-gradient ones would, however, distort the data distribution.

**GOSS mechanism.** Per boosting iteration:

1. **Compute gradients** for all instances from the current ensemble.
2. **Sort by $|g_i|$** in descending order.
3. **Keep top instances** — retain the top $a \times 100\%$ (largest gradients) as set $A$. In LightGBM this ratio is `top_rate` (default ≈ 0.2).
4. **Sample the rest** — from the remaining $(1-a)\times 100\%$ (small gradients) randomly sample a fraction $b$, i.e. $b \times (1-a) \times N$ instances, as set $B$. This ratio is `other_rate` (default ≈ 0.1).
5. **Combine and amplify** — train the current tree on $A \cup B$ only. To keep the gradient statistics unbiased, up-weight the sampled small-gradient instances: each $g_i$ (and $h_i$) in $B$ is multiplied by the constant

$$\frac{1-a}{b}.$$

**Why the amplification matters.** Split gain depends on sums of gradients (and Hessians) in candidate child nodes. Sampling set $B$ with probability $b$ shrinks its count; the factor $\tfrac{1-a}{b}$ rescales the sampled sum so that, in expectation, it matches the total of the full $(1-a)\times N$ small-gradient pool it stands in for. Without it, well-predicted instances would be systematically under-counted in gain calculations, biasing splits.

**GOSS vs. stochastic gradient boosting.** Ordinary row subsampling (Friedman's stochastic GB, `subsample` in XGBoost/sklearn) samples instances *uniformly at random*. GOSS instead does *biased* sampling by gradient magnitude — always keeping the hardest instances — and it usually beats naive random sampling, which can accidentally discard too many important high-gradient rows.

**In practice.** GOSS is invoked via `boosting_type='goss'`. `top_rate` and `other_rate` rarely need heavy tuning; raise `top_rate` to keep more hard data (less speedup), raise `other_rate` to use more small-gradient data (slight accuracy gain at speed cost). (Note: with the default `boosting_type='gbdt'` GOSS is *not* active — you select it explicitly with `'goss'`.)

### 5.3 Exclusive Feature Bundling (EFB)

High-dimensional sparse data (e.g. one-hot-encoded categoricals) makes per-node histogram construction over all features very expensive. EFB attacks the *feature* dimension.

**Key observation.** Sparse datasets contain many **mutually exclusive** features — features that are rarely/never nonzero for the *same* instance. One-hot encoding is the classic source: for a "City" column, only one of `is_City_London`, `is_City_Paris`, `is_City_Tokyo` can be 1 per row. EFB bundles such features into a single denser feature, cutting the number of histograms to build and scan while preserving the information needed for split finding.

**Identifying bundles — a graph problem.**

- **Nodes**: one per feature.
- **Edges**: connect two features if they *conflict* (both nonzero on at least some minimum number of instances). A small conflict tolerance is allowed, since perfect exclusivity is too strict and would limit bundling.
- **Graph coloring**: apply a *greedy* graph-coloring algorithm — assign each feature a "color" (bundle ID) so that edge-connected features get different colors, minimizing the number of colors. Same color ⇒ bundled together.

Optimal graph coloring is NP-hard, but the greedy pass (assign each feature the first color not used by a conflicting neighbor) works well and is cheap in practice. (Example from the course: with 5 sparse features, (F1,F2) → bundle 1, (F3,F5) → bundle 2, F4 → bundle 3.)

**Building the bundled feature — value offsetting.** To keep the original features distinguishable inside one bundle, their value ranges are *offset* into disjoint histogram-bin ranges. For exclusive features A (range $[0, k_A]$) and B (range $[0, k_B]$) in the same bundle:

- A's nonzero values map to bins $[1, k_A]$,
- B's nonzero values map to bins $[k_A+1, k_A+k_B]$,
- both features' zeros map to bin $0$.

Then finding the best split on the bundled feature is equivalent to finding it on the original constituents: a split at bin $j$ with $1 \le j \le k_A$ is a split on A, and $k_A+1 \le j \le k_A+k_B$ is a split on B.

**Benefits / trade-off.** Fewer features to histogram ⇒ faster iterations and lower memory. The trade-off: if bundled features aren't perfectly exclusive (conflict tolerance > 0), a little information can be lost — controlled by the allowed conflict rate (`max_conflict_rate`, usually automatic). Unlike PCA (dense linear combinations), EFB is purpose-built for sparse exclusive features and preserves their original split potential; toggle it with `enable_bundle` (default `True`).

### 5.4 Histogram-based Split Finding

Exact split finding sorts and scans all candidate points per feature, per node — $O(\#\text{data} \times \#\text{features})$ per node. LightGBM instead **discretizes** each continuous feature into a fixed number of bins (a histogram).

**Binning.** Before training (or at the start of each tree), each continuous feature is bucketed into at most `max_bin` bins (default 255). E.g. an "Age" feature over [20, 80] with `max_bin=6` → bins `[20-30), [30-40), ..., [70-80]`; age 35 falls in `[30-40)`. Binning is done once for the whole dataset.

**Building histograms and finding splits.** At a node, for each feature LightGBM accumulates, per bin, the **sum of gradients and sum of Hessians** of the node's instances falling in that bin. To find the best split it iterates over bins $1..\texttt{max\_bin}$, at each bin $k$ considering "value $\le$ bin $k$" vs "value $>$ bin $k$", and computes gain instantly from the accumulated sums. This drops the per-feature split scan from $O(\#\text{data})$ to $O(\#\text{bins})$, so per-node cost becomes

$$O(\#\text{bins} \times \#\text{features}) \ll O(\#\text{data} \times \#\text{features}) \quad (\text{when } \#\text{bins} \ll \#\text{data}).$$

**Histogram subtraction trick.** When a node splits, the parent histogram equals the sum of its two children's histograms ($H_p = H_l + H_r$). LightGBM computes the histogram of the *smaller* child directly, then obtains the sibling by subtraction: $H_r = H_p - H_l$. This avoids scanning the larger child's data and greatly speeds up deeper trees.

**Advantages.** Speed (much cheaper split finding); memory efficiency (store per-bin aggregates, $O(\#\text{bins} \times \#\text{features})$, not pre-sorted values $O(\#\text{data} \times \#\text{features})$); cache friendliness (bin-indexed access vs scattered sorted indices).

**Trade-off — `max_bin`.** Binning is approximate: splits can only land on bin edges, not the exact optimal value. Smaller `max_bin` → faster, less memory, coarser (possible accuracy loss, though the binning also acts as regularization and can even help); larger `max_bin` → more accuracy potential, more compute/memory. Default 255 is usually a good balance; typical range 63–255.

```python
import lightgbm as lgb
import numpy as np

# Example data (replace with real data)
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, 1000)
lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'max_bin': 255,  # number of histogram bins
}

gbm = lgb.train(params, lgb_train, num_boost_round=100)
print(f"Model trained with max_bin={params['max_bin']}")
```

Histograms are the core of LightGBM's efficiency and dovetail with GOSS (fewer data points to bin) and EFB (fewer features to bin).

### 5.5 Leaf-wise (Best-first) Tree Growth

Most GBM implementations (including XGBoost's default) grow trees **level-wise (depth-first by level)**: split every node at a depth before descending. This keeps trees balanced and is less prone to immediate overfitting.

LightGBM defaults to **leaf-wise (best-first)** growth: at each step, split the single leaf *anywhere* in the current tree that yields the largest loss reduction, then repeat.

**Comparison.**

- **Level-wise**: best split of root → best splits of both children → best splits of all four grandchildren → ... one full level at a time (grows horizontally).
- **Leaf-wise**: split root → two leaves → evaluate splitting each leaf, split whichever reduces loss most → three leaves → repeat. The tree tends to deepen along one promising path first, producing an *asymmetric* shape early.

**Advantage — faster convergence.** Always splitting the max-loss-reduction leaf reaches a lower loss with fewer splits than level-wise, because it never wastes splits on nodes that only marginally help just to complete a level. For a fixed number of leaves, leaf-wise can be more accurate; on large data this is meaningfully faster. Combined with fast histogram split evaluation, LightGBM builds effective trees quickly.

**Drawback — overfitting risk.** Greedily chasing the deepest-gain path yields deeper, unbalanced trees that can fit training noise, especially on smaller datasets. LightGBM controls this with regularization parameters:

- **`num_leaves`** — the *primary* complexity control under leaf-wise growth; directly caps the number of leaves per tree. Usually more effective than `max_depth`. Keep it well below $2^{\texttt{max\_depth}}$. Default 31 is a reasonable start but often needs tuning.
- **`max_depth`** — a hard depth cap acting as a safety net even when `num_leaves` isn't reached (`-1` = no limit).
- **`min_data_in_leaf`** (a.k.a. `min_child_samples`) — minimum instances required in a leaf; rejects splits that would create tiny leaves, preventing fits to noise in small groups.

### 5.6 Optimized Categorical Feature Handling

One-hot encoding (OHE) explodes dimensionality on high-cardinality categoricals (more memory, slower training); label encoding is compact but imposes an arbitrary numeric order that misleads splits. LightGBM handles categoricals **natively**, often removing the need for manual preprocessing.

**Native categorical split — Fisher-style optimal partition.** Rather than requiring pre-encoding, LightGBM finds the best *partition of categories* by their relation to the target, using an algorithm based on Fisher (1958, "On Grouping for Maximum Homogeneity"):

1. **Collect statistics** — for each category $c$ reaching the node, sum the gradients and Hessians of its instances:

$$G_c = \sum_{i \in c} g_i, \qquad H_c = \sum_{i \in c} h_i.$$

2. **Sort categories** by a value derived from these stats — typically the leaf-output ratio $G_c / H_c$ (optionally regularized). This puts categories with similar target effect next to each other.
3. **Find the best split** along this sorted list, partitioning categories into two subsets (e.g. {A, C} vs {B, D}) to maximize gain — analogous to numeric histogram splits.

For a feature with $k$ categories, this considers only $k-1$ candidate split points after sorting, vs OHE's $k$ new binary features — far more efficient and often more accurate (groups categories by target effect instead of treating each independently or imposing false order).

**Telling LightGBM which columns are categorical.**

```python
import pandas as pd
import lightgbm as lgb

data = {'numeric_feat': [1.2, 3.4, 0.5, 2.1],
        'category_feat': ['A', 'B', 'A', 'C']}
df = pd.DataFrame(data)
df['category_feat'] = df['category_feat'].astype('category')  # pandas 'category' dtype
```

```python
X = df[['numeric_feat', 'category_feat']]
y = [0, 1, 0, 1]

# sklearn API: pass feature names (recommended with pandas)
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X, y, categorical_feature=['category_feat'])

# Native Dataset API: categoricals must be non-negative integers (0,1,2,...)
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded['category_feat'] = encoder.fit_transform(X[['category_feat']])

lgb_data = lgb.Dataset(X_encoded, label=y,
                       feature_name=['numeric_feat', 'category_feat'],
                       categorical_feature=['category_feat'])
```

Two routes: (a) pandas `category` dtype (the sklearn API auto-detects it), or (b) the explicit `categorical_feature` argument (most reliable). With `lgb.Dataset`, categoricals must be encoded as non-negative integers (e.g. via `OrdinalEncoder`) — LightGBM treats these as unordered category IDs, *not* ordinal values.

**Benefits.** Efficiency (no OHE blow-up, big win on high-cardinality); effectiveness (grouping by target effect finds more meaningful splits than OHE or naive label encoding); simplicity (less feature engineering).

**Tuning parameters.**

- `max_cat_to_onehot` (int, default 4) — if unique categories $\le$ this, use OHE instead of native partition (faster for very low cardinality).
- `cat_smooth` (float, default 10.0) — smooths the per-category $G_c/H_c$ statistic toward the global mean (a prior), guarding against overfitting on rare categories.
- `cat_l2` (float, default 10.0) — L2 penalty specific to categorical splits.
- `max_cat_threshold` — caps the number of categories considered when searching for a split.

(CatBoost, Chapter 6, uses more elaborate ordered target statistics + automatic feature combinations specifically to fight target leakage; XGBoost historically needed manual encoding, with only recent experimental categorical support.)

### 5.7 LightGBM API: Parameters & Configuration

LightGBM offers a scikit-learn-style interface (`LGBMClassifier`, `LGBMRegressor`) and a native training API (`lgb.Dataset` / `lgb.train`). Names differ slightly across interfaces (e.g. `n_estimators` vs `num_iterations`) but the semantics match.

**Core boosting**

- `objective` — loss to optimize: `regression` (L2), `regression_l1` (L1), `huber`, `binary` (logloss), `multiclass` (softmax), `lambdarank` (ranking); custom objectives possible.
- `boosting_type` / `boosting` — `gbdt` (standard), `dart` (dropout, more robust but needs more rounds), `goss` (gradient-based one-side sampling).
- `num_iterations` / `n_estimators` — number of boosting rounds; too few underfits, too many overfits. Tune with `learning_rate` + early stopping.
- `learning_rate` / `eta` — shrinkage per tree; lower needs more rounds but generalizes better (typical 0.01–0.3).

**Tree structure**

- `num_leaves` — *the* main complexity knob under leaf-wise growth; caps leaves per tree. Keep $\ll 2^{\texttt{max\_depth}}$.
- `max_depth` — hard depth cap (`-1` = unlimited); safety net when `num_leaves` is large.
- `min_data_in_leaf` / `min_child_samples` — min instances per leaf; larger = more regularization.
- `min_sum_hessian_in_leaf` / `min_child_weight` — min sum of Hessians per leaf; a statistically grounded leaf-size control for non-L2 losses.

**Regularization**

- `lambda_l1` / `reg_alpha` — L1 on leaf outputs (encourages sparsity).
- `lambda_l2` / `reg_lambda` — L2 on leaf outputs (main shrinkage term).
- `min_gain_to_split` / `min_split_gain` — minimum gain to accept a split (prunes weak splits).

**Sampling**

- `feature_fraction` / `colsample_bytree` — fraction of features sampled per tree (complements EFB, which *bundles* rather than samples).
- `bagging_fraction` / `subsample` — fraction of rows sampled (without replacement) per iteration; requires `bagging_freq > 0`.
- `bagging_freq` — do bagging every $k$ iterations (`0` disables).
- `feature_fraction_bynode` — feature fraction considered at each node split (extra per-node randomness).

**Efficiency / algorithm control**

- `boosting_type='goss'` + `top_rate` / `other_rate` — enable and control GOSS.
- `enable_bundle` (default `True`) — toggle EFB.
- `max_bin` — histogram bin count (typical 63–255); smaller = faster + regularizing but coarser.

**Categorical**: `categorical_feature`, `max_cat_threshold`, `cat_smooth`, `cat_l2` (see §5.6).

**Other**: `metric` (`l1`, `l2`, `rmse`, `auc`, `binary_logloss`, `multi_logloss`, ...); `is_unbalance` / `scale_pos_weight` for imbalance; `device_type` (`cpu`/`gpu`); `n_jobs` (`-1` = all cores); `seed` / `random_state` for reproducibility.

#### Key-parameter cheat sheet

| Parameter | Role | Typical / default | Direction |
|---|---|---|---|
| `num_leaves` | main leaf-wise complexity cap | 31 | ↑ more capacity, ↑ overfit risk |
| `learning_rate` | shrinkage per tree | 0.01–0.3 (0.05) | ↓ + more rounds ⇒ better generalization |
| `n_estimators` | number of trees | tune w/ early stopping | ↑ risk overfit; pair with `learning_rate` |
| `max_depth` | hard depth cap | -1 (unlimited) | set to bound deep branches |
| `min_data_in_leaf` | min samples per leaf | 20 | ↑ stronger regularization |
| `feature_fraction` | per-tree feature subsample | 0.7–1.0 | <1.0 regularizes |
| `bagging_fraction` + `bagging_freq` | per-iter row subsample | 0.8, 5 | <1.0 regularizes (freq>0 required) |
| `lambda_l1` / `lambda_l2` | L1 / L2 on leaf outputs | 0.0+ | ↑ shrink leaves |
| `max_bin` | histogram bins | 255 (63–255) | ↓ faster/regularizes, coarser |
| `categorical_feature` | native categorical columns | — | enables Fisher-style splits |

#### XGBoost ↔ LightGBM equivalent parameters

| XGBoost | LightGBM |
|---|---|
| `eta` | `learning_rate` |
| `n_estimators` / `num_round` | `num_iterations` / `n_estimators` |
| `subsample` | `bagging_fraction` (+ `bagging_freq`) |
| `colsample_bytree` | `feature_fraction` |
| `reg_alpha` | `lambda_l1` |
| `reg_lambda` | `lambda_l2` |
| `min_child_weight` | `min_sum_hessian_in_leaf` |
| `gamma` / `min_split_loss` | `min_gain_to_split` |
| `max_depth` (primary control) | `num_leaves` (primary control), `max_depth` (secondary) |
| `max_bin` | `max_bin` |
| `scale_pos_weight` | `scale_pos_weight` (or `is_unbalance`) |

Note the different *primary* complexity control: XGBoost is level-wise and tuned mainly via `max_depth`; LightGBM is leaf-wise and tuned mainly via `num_leaves`.

```python
import lightgbm as lgb

lgbm_clf = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,       # target rounds, used with early stopping
    max_depth=-1,            # rely on num_leaves
    min_child_samples=20,
    subsample=0.8,           # bagging_fraction
    colsample_bytree=0.7,    # feature_fraction
    reg_alpha=0.1,           # lambda_l1
    reg_lambda=0.1,          # lambda_l2
    n_jobs=-1,
    random_state=42,
    # categorical_feature=[0, 3, 5],
    # max_bin=127,
)
# lgbm_clf.fit(X_train, y_train,
#              eval_set=[(X_val, y_val)], eval_metric='auc',
#              callbacks=[lgb.early_stopping(100)])
```

### 5.8 Hands-on: Implementing LightGBM

The exercise builds a synthetic binary-classification dataset (mixed informative, redundant, and simulated categorical features) and contrasts a default model with a configured one.

**Setup and data generation.**

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
import time

n_samples, n_features = 5000, 30
n_informative, n_redundant, n_categorical = 15, 5, 5
random_state = 42

X, y = make_classification(
    n_samples=n_samples, n_features=n_features,
    n_informative=n_informative, n_redundant=n_redundant, n_repeated=0,
    n_classes=2, n_clusters_per_class=2,
    weights=[0.8, 0.2],   # class imbalance
    flip_y=0.05,          # label noise
    class_sep=0.8, random_state=random_state)

feature_names = [f'num_{i}' for i in range(n_features - n_categorical)] + \
                [f'cat_{i}' for i in range(n_categorical)]
X = pd.DataFrame(X, columns=feature_names)

# Simulate categoricals by discretizing the last columns into integer bins
for i in range(n_categorical):
    col = f'cat_{i}'
    X[col] = pd.qcut(X[col], q=5, labels=False, duplicates='drop').astype(int)

categorical_features_names = [c for c in feature_names if c.startswith('cat_')]
for col in categorical_features_names:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state, stratify=y)
```

**Baseline (default params).**

```python
lgbm_default = lgb.LGBMClassifier(random_state=random_state)
lgbm_default.fit(X_train, y_train)

y_pred = lgbm_default.predict(X_test)
y_proba = lgbm_default.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
```

**Configured model** — explicit categoricals, tuned complexity/regularization/subsampling, and early stopping.

```python
lgbm_cfg = lgb.LGBMClassifier(
    objective='binary', metric='auc',
    n_estimators=500, learning_rate=0.05,
    num_leaves=64, max_depth=-1, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=random_state)

lgbm_cfg.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100, verbose=False)],
    categorical_feature=categorical_features_names)  # explicit native handling

print("Best iteration:", lgbm_cfg.best_iteration_)
```

**Observations from the exercise.**

- **Categorical handling** — passing `categorical_feature` makes LightGBM use its Fisher-style optimal split, usually faster *and* more accurate than treating them as continuous or one-hot encoding; the `category` dtype in the input DataFrame is the recommended representation.
- **Training time** — despite more estimators, histogram splitting + GOSS + EFB keep it fast, and early stopping avoids wasted rounds.
- **Performance** — tuning `num_leaves`, `learning_rate` plus regularization/subsampling typically beats defaults; early stopping guards against overfitting as `n_estimators` grows.

**Feature importance (optional).**

```python
importance_df = pd.DataFrame({
    'feature': lgbm_cfg.booster_.feature_name(),
    'importance': lgbm_cfg.feature_importances_,
}).sort_values('importance', ascending=False)
# plot importance_df.head(20) with seaborn/plotly
```

### Key takeaways

- LightGBM targets the two axes XGBoost struggles with at scale — **instances** (via GOSS) and **features** (via EFB) — on top of **histogram** split finding and **leaf-wise** growth.
- **GOSS**: keep the top-$a$ largest-gradient instances, randomly sample fraction $b$ of the rest, and up-weight those samples by $\tfrac{1-a}{b}$ to stay unbiased.
- **EFB**: greedily graph-color mutually-exclusive sparse features into bundles, offsetting their value ranges so bundled splits recover original-feature splits.
- **Histograms**: bin continuous features (`max_bin`), turning $O(\#\text{data})$ split scans into $O(\#\text{bins})$; the parent = children histogram-subtraction trick speeds deep trees.
- **Leaf-wise growth** converges faster but overfits without control — govern it with `num_leaves` (primary), `max_depth`, and `min_data_in_leaf`.
- **Native categoricals** use a Fisher optimal partition (sort categories by $G_c/H_c$, consider $k-1$ splits) — usually better than OHE/label encoding.
- The primary complexity control differs from XGBoost: `num_leaves` here vs `max_depth` there.

> **Relevance to our work:** LightGBM is our primary learner, so this chapter is directly load-bearing.
> **Determinism** — reproducible runs need `seed`/`random_state` fixed *and* the known LightGBM determinism caveats (per our LGBM-determinism memo: set `deterministic=True`, `force_row_wise=True`, fixed `num_threads`, since default multi-threaded histogram building isn't cross-process reproducible — this is exactly what our tree-search reproduction gate depends on).
> **`num_leaves` + `min_data_in_leaf`** — under leaf-wise growth these are our main overfit controls; tune `num_leaves` first (start 31, scale with dataset size, keep $\ll 2^{\text{max\_depth}}$) and use `min_data_in_leaf` to stop leaves fitting noise on small folds.
> **`max_bin`** doubles as a regularizer/speed knob; **`categorical_feature`** lets us skip OHE and its dimensionality blow-up on high-cardinality columns.
> **GOSS** (`boosting_type='goss'`) is an option when training time on large tables dominates; otherwise default `gbdt`. Cross-validation + early stopping remain the backbone of any tuning we do (see Chapter 8).

---

## Chapter 6 — CatBoost: Ordered Boosting for Categorical Data

> Source: "Mastering Gradient Boosting Algorithms" (apxml.com), Chapter 6, eight lessons.
> Synthesized English study notes reconstructed from a scraped Chinese transcript with KaTeX-mangled math cleaned up. Foundational reference throughout: Prokhorenkova et al., *CatBoost: Unbiased Boosting with Categorical Features*, NeurIPS 2018 (arXiv:1706.09516).

CatBoost's whole design answers one question: how do you put categorical features directly into gradient boosting without leaking the target? Its two signature ideas — **Ordered Target Statistics** (for encoding) and **Ordered Boosting** (for training) — both enforce a "use only the past" rule under random permutations, and both are made practical by **oblivious (symmetric) trees**.

---

### 6.1 The Trouble with Categorical Data

Tree models split on numeric thresholds (`feature < threshold`), so categorical features must first become numbers. Every naive encoding has a failure mode:

- **One-hot encoding (OHE).** Fine for low cardinality, but a high-cardinality feature (e.g. `user_id` with thousands of values) explodes into thousands of mostly-zero columns. Consequences: extreme sparsity, slower split search (dimensionality dominates even sparse-optimized libraries), and deep/complex trees because each split isolates only one category — capturing "several categories together" needs many splits.
- **Label / ordinal encoding.** Assigns an arbitrary integer per category (`red→0, green→1, blue→2`). This invents an ordering that usually does not exist: the model may treat `green` as "between" red and blue, or read `blue−red = 2` as twice `green−red = 1`. Splits like `color_encoded < 1.5` become meaningless for nominal features (acceptable only for genuinely ordinal ones like low/medium/high).
- **Target (mean) encoding.** Replace each category with a statistic of the target over rows in that category (e.g. mean target for `city = A`). This directly builds a strong tie between feature and target.

**Target leakage** is the central danger of target encoding: because a row's own target contributed to its encoding, the model sees a direct clue about $y$. Result:

- **Overfitting** — training performance looks great but is spurious; it does not generalize once the leaked signal is absent on unseen data.
- **Prediction shift / unreliable stats** — rare categories with few observations give noisy statistics, and the in-category target distribution can differ between train and future data, so the encoding is miscalibrated.

Holdout sets or smoothing mitigate leakage but add complexity and never fully solve it. Add to this the difficulty of discovering high-order **feature interactions** (e.g. `product_category × store_location`), which naive OHE + trees capture only with very deep trees or laborious manual feature crosses. These problems motivate CatBoost's built-in Ordered TS and automatic feature combinations.

---

### 6.2 Ordered Target Statistics (Ordered TS)

Standard mean encoding leaks because computing a row's encoding uses that row's own target. CatBoost's fix: impose an artificial "time" ordering and, for each row, compute the target statistic using **only the rows that come before it** — never its own target.

Mechanism:

1. **Random permutation.** Shuffle the training indices $\{1,\dots,n\}$ into a permutation $\sigma$. This acts like time: if $k > j$, sample $\sigma(k)$ comes "after" $\sigma(j)$.
2. **Ordered computation.** For sample $x_{\sigma(k)}$ and categorical feature $i$, the target statistic $TS_{k,i}$ uses only preceding samples ($j < k$) that share the same category value. The current target $y_{\sigma(k)}$ is **excluded**.

With prior smoothing:

$$
TS_{k,i} = \frac{\sum_{j=1}^{k-1} \big[x_{\sigma(j),i} = x_{\sigma(k),i}\big]\, y_{\sigma(j)} + a\,p}{\sum_{j=1}^{k-1} \big[x_{\sigma(j),i} = x_{\sigma(k),i}\big] + a}
$$

Equivalently, for the current row $i$ using only its predecessors in the permutation:

$$
\hat{x} = \frac{\sum_{j<i} [x_j = x_i]\, y_j + a\,p}{\sum_{j<i} [x_j = x_i] + a}
$$

Terms:

- $[x_{\sigma(j),i} = x_{\sigma(k),i}]$ — indicator: 1 if the preceding sample shares the same category on feature $i$, else 0.
- $y_{\sigma(j)}$ — the preceding sample's target.
- $\sum_{j=1}^{k-1}$ — the sum stops at $k-1$, guaranteeing only predecessors count.
- $a$ (also written $\alpha$) — a positive smoothing weight.
- $p$ (the **prior**) — usually the global average target.

The $a\,p$ term is a **regularizer**. Early in the permutation ($k$ small) or for rare categories there are few (or zero) matching predecessors, so the raw statistic would be noisy or undefined; smoothing pulls the estimate toward the global mean $p$ when local evidence is scarce.

**Multiple permutations for stability.** A single permutation introduces its own order-dependent bias. CatBoost generates several random permutations and uses different ones for different trees in the ensemble, averaging out order effects. This ties directly into Ordered Boosting (next section).

**Test-time encoding.** New rows cannot be inserted into the training permutation. Instead CatBoost encodes a test category using statistics learned during training — typically all training rows of that category (still with prior + smoothing), or pre-computed stored statistics; the exact route depends on implementation and parameters.

Benefits: sharply reduced leakage (own target excluded), numerically meaningful encodings for high-cardinality features without high-dimensional sparse OHE, and the whole scheme is built into the algorithm so the user skips manual preprocessing.

---

### 6.3 Prediction Shift and Ordered Boosting

Ordered TS fixes leakage in the *encoding*, but a subtler bias — **prediction shift** — survives in the *training loop*. At iteration $m$, the residual (negative gradient) for a sample depends on the current model $F_{m-1}$, built from all prior iterations. If those earlier trees used target statistics that (even via Ordered TS across the ensemble) incorporated sample $i$'s target, then $F_{m-1}$ has already "seen" $y_i$. Computing $i$'s residual from such a model reintroduces bias.

Standard boosting update:

$$
F_m(x) = F_{m-1}(x) + \alpha \cdot h_m(x)
$$

where $h_m$ fits the residual $r_{m-1}$ computed from $F_{m-1}$. The problem: $r_{m-1}$ for $x_i$ uses $F_{m-1}$, which may already depend on $y_i$.

**Ordered Boosting** removes this by, again, only letting "past" samples influence a given sample's residual:

1. **Permutations.** Generate random permutations $\sigma_1,\dots,\sigma_S$ of the training indices.
2. **Permutation-specific model series.** For each $\sigma_s$, maintain a separate model sequence $M_0^s, M_1^s, \dots, M_{m_{max}}^s$.
3. **Ordered residuals.** When building tree $m$ for permutation $\sigma_s$, the residual for the $i$-th sample in that order is computed **only** from $M_{m-1}^s$ trained on the preceding $i-1$ samples. That is, $r_{\sigma_s(i)}$ depends on $M_{m-1}^s(\mathbf{x}_{\sigma_s(j)})$ for $j < i$ only — the model used to score sample $i$ has never trained on sample $i$.
4. **Update.** The weak learner $h_m^s$ is fit on these ordered residuals: $M_m^s = M_{m-1}^s + \alpha\, h_m^s$.
5. **Final model.** Derived by pooling the learning across all permutations (the exact mechanism is intricate).

This simulates inference-time conditions (target unknown) when estimating each sample's gradient, giving **unbiased residuals**. Ordered TS and Ordered Boosting are complementary: Ordered TS ensures a sample's *encoding* uses only earlier samples' targets; Ordered Boosting ensures a sample's *gradient* uses only a model trained on earlier samples. Together they block leakage in both the feature representation and the update step.

**Trade-off.** Maintaining $S$ model series looks expensive, but the oblivious-tree structure (6.5) plus CatBoost's engineering make it practical. The payoff is much lower prediction shift and stronger train→test generalization — most valuable on datasets with strong categorical predictors where leakage otherwise inflates training metrics that never materialize in production. Training can be slower than leakage-blind implementations, but CatBoost's overall optimizations (GPU, efficient categorical handling) keep it competitive or faster in practice.

---

### 6.4 Feature Combinations

Much predictive signal lives in *interactions*: knowing `Browser='Safari'` and `OS='macOS'` together can beat either alone. Manually crafting crosses (a `browser_os` feature) works but scales badly — the number of pairwise, triple, … combinations grows exponentially, and finding the useful ones needs domain expertise.

CatBoost generates and evaluates categorical combinations **automatically and greedily during tree construction**, not by pre-building a giant feature set. When choosing the best split for a node (or, for oblivious trees, a whole level), it considers combinations of:

- one or more categorical features already used in ancestor splits (root → current node path), plus
- a new candidate categorical feature.

So for a new categorical feature $C_{new}$, CatBoost also evaluates splits on $(C_{ancestor1}, C_{new})$, $(C_{ancestor2}, C_{new})$, $(C_{ancestor1}, C_{ancestor2}, C_{new})$, and so on. Each newly formed combination is treated exactly like any categorical feature — encoded with the same **Ordered TS** "only-the-past" rule to avoid leakage — and the algorithm picks whichever split (original or combined) yields the greatest loss reduction.

**Integration with oblivious trees.** Because a symmetric tree uses the *same* split across all nodes at a level, a chosen combination (e.g. `Country × Browser`) applies to the entire depth level. Combinations thus build up progressively with depth: a level-3 combination can involve categoricals used at levels 1 and 2.

**Controlling complexity.** Generation is greedy, not exhaustive — it typically starts with 2-feature combinations and extends a proven combination with another feature at deeper levels. `max_ctr_complexity` caps the number of categorical features fused into one combination (default commonly 4; the transcript's parameter lesson notes default 4, its combinations lesson notes "up to two" as the practical starting point). Higher values capture higher-order interactions at extra compute and overfitting risk. The related `ctr_max_border_count` controls how many splits are considered for combinations involving numeric features. Upside: automated interaction discovery, higher accuracy from complex categorical dependencies, and native integration with Ordered TS and symmetric trees.

---

### 6.5 Oblivious (Symmetric) Trees

CatBoost's base learner is the **oblivious tree** (a.k.a. symmetric tree) — a key differentiator from XGBoost and LightGBM, which build asymmetric trees where different nodes at the same depth may split on different features.

**Structure.** In an oblivious tree, *every node at a given depth uses the identical split condition* (same feature, same threshold/category). The tree is perfectly balanced and symmetric: all root-to-leaf paths have equal length and the split criterion at each level is the same for all samples passing through. Example, depth 2:

- **Level 0 (root):** all points split on, say, `Feature_X < threshold_1`.
- **Level 1:** both children split on the *same* condition, say `Feature_Y > threshold_2`.
- **Level 2 (leaves):** terminal prediction nodes.

The sequence of tested (feature, threshold) pairs is fixed for a given depth, regardless of branch.

**Why impose this?**

- **Compute efficiency.** A uniform per-level condition vectorizes well: instead of per-sample conditional branching, left/right assignment is a single operation over an array/tensor for the whole batch — a big speedup, especially on parallel hardware (GPU).
- **Faster prediction.** A sample's leaf is determined by a fixed sequence of comparisons, so the leaf index is a simple binary string / integer computed directly, with no irregular pointer chasing.
- **Implicit regularization.** The fixed per-level split is structural regularization: the tree can't carve highly specific paths for tiny data subsets, limiting per-learner complexity and curbing overfitting to noise. CatBoost compensates for the lower expressiveness of a single oblivious tree by building more trees and leaning on its feature-combination machinery.

**Synergy.** Combinations can produce a high-dimensional feature space; the simple, regular oblivious structure keeps modeling tractable there, and its built-in regularization manages the added complexity.

**Trade-off.** A single oblivious tree is less expressive than an asymmetric one and may need more depth to capture certain interactions directly. CatBoost overcomes this at the *ensemble* level (Ordered Boosting + automatic combinations) while keeping the symmetric structure's speed and regularization.

---

### 6.6 GPU Training Acceleration

Boosting is compute-heavy — thousands of sequential trees, each scanning many candidate splits over features and samples — and combination generation adds more. CPU parallelism helps, but GPUs (thousands of simple SIMD cores) suit the massive parallelism inside boosting, often turning hours into minutes on datasets with hundreds of thousands to millions of rows.

Where the GPU helps in CatBoost:

- **Histogram building.** Like LightGBM/XGBoost approximate methods, CatBoost can use histogram-based split finding; computing histograms (value distributions, gradient sums, Hessian sums) parallelizes across data points on the GPU.
- **Oblivious trees.** Their structure is especially GPU-friendly: since all nodes at a depth share one split, left/right routing for the whole level is one highly parallel operation — unlike leaf-/level-wise asymmetric growth where nodes at a depth test different features. The predictable structure maps cleanly onto GPU parallelism.
- **Categorical handling.** Ordered TS has sequential dependencies, but its underlying statistic and permutation computations still benefit from GPU acceleration at scale; combination generation and symmetric structure further aid parallel execution.

**Enabling it.** Install the GPU-enabled CatBoost build with a compatible CUDA NVIDIA GPU and set `task_type='GPU'`.

```python
import catboost as cb
from sklearn.model_selection import train_test_split

# X (features), y (target) already loaded as pandas DataFrame / Series
# Identify categorical feature indices
categorical_features_indices = [
    i for i, col in enumerate(X.columns)
    if X[col].dtype == 'object' or X[col].dtype.name == 'category'
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_gpu = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    task_type='GPU',          # enable GPU training
    devices='0',              # optional: GPU device id(s)
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
)

model_gpu.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
    plot=False,               # True to view learning curves interactively
)
# preds_gpu = model_gpu.predict_proba(X_test)[:, 1]
```

Key GPU parameters: `task_type='GPU'` (required to train on GPU) and `devices` (device ids, e.g. `'0'`, `'0:1'`, `'1'`; default device 0 if omitted).

**Caveats.** You need an NVIDIA GPU with enough VRAM (large data/complex models need more). RAM↔VRAM transfer overhead can outweigh gains on very small datasets — GPU pays off most on medium/large data. Some hyperparameters interact differently on GPU, so tune with the same `task_type` you'll deploy. Minor CPU/GPU floating-point differences can slightly change results or convergence, though CatBoost aims for consistency.

---

### 6.7 CatBoost API: Parameters and Configuration

The main interfaces are `CatBoostClassifier` and `CatBoostRegressor`. Many parameters overlap other GBM libraries; some are CatBoost-specific.

**Core training**

- `iterations` (a.k.a. `n_estimators`) — max boosting rounds / trees. Default 1000; tune with early stopping.
- `learning_rate` — step size shrinkage per tree. Smaller (0.01–0.1) usually needs more iterations but generalizes better. Default auto-detected (often ~0.03) from data size and iterations.
- `depth` — oblivious-tree depth; same split applied across a level. Typical 4–10. Default 6.
- `l2_leaf_reg` — L2 penalty on leaf values (XGBoost's `lambda`). Default 3.0.
- `loss_function` — objective. Regression: `RMSE` (default), `MAE`, `Quantile`, `LogLinQuantile`, `Poisson`, `MAPE`. Classification: `Logloss` (binary, default), `MultiClass`, `CrossEntropy`. Custom objectives supported.
- `eval_metric` — metric for evaluation/early stopping: `RMSE`, `MAE`, `Logloss`, `AUC`, `Accuracy`, `F1`, `Precision`, `Recall`, `MultiClass`, `NDCG`, `MAP`, … Defaults to match `loss_function`.
- `random_seed` (`random_state`) — reproducibility of shuffling/sampling.

**Categorical features (CatBoost's specialty)**

- `cat_features` — indices or names of columns to treat as categorical. Do **not** pre-encode these (no OHE/label encoding) — CatBoost applies Ordered TS internally. If `None`, it may auto-detect, but explicit is recommended. Default `None`.
- `one_hot_max_size` — cardinality threshold: features with ≤ this many unique values use OHE instead of target statistics. Small values force Ordered TS onto most categoricals. Default 2.
- `max_ctr_complexity` — max number of categorical features fused into one combination. Default 4; higher = higher-order interactions at more compute/memory.
- `has_time` — set `True` when rows have a real time order (time series) so Ordered TS respects it and avoids look-ahead. Default `False`.
- `simple_ctr`, `combinations_ctr` — fine-grained control of CTR (counter statistic) types for single features and combinations; advanced use.

**Performance / efficiency**

- `task_type` — `'CPU'` (default) or `'GPU'`.
- `devices` — GPU device ids when `task_type='GPU'`.
- `thread_count` — CPU threads (CPU mode); `-1` uses all cores (default).
- `border_count` — number of bins for discretizing numeric features. Higher = finer splits, more memory/time. Typical 32–255. Default 254 (CPU), 128 (GPU).
- `leaf_estimation_method` — `'Newton'` (2nd-order, faster convergence, default) or `'Gradient'` (1st-order, can be more stable for some objectives).

**Regularization / training control**

- `early_stopping_rounds` — stop if `eval_metric` on `eval_set` doesn't improve for this many rounds; auto-finds a good `iterations`. Needs `eval_set`. Default `None` (off).
- `use_best_model` — with early stopping on, restore the best-validation model state. Default `True` when `eval_set` given, else `False`.
- `subsample` — row subsampling fraction per tree (<1.0 adds regularizing randomness). Default 1.0 (CPU), 0.8 (GPU).
- `bootstrap_type` — sampling scheme: `'Bayesian'` (default; exponential random weights, tied to Ordered Boosting), `'Bernoulli'` (standard subsampling, used with `subsample<1.0`), `'MVS'` (minimum variance sampling), `'No'`.
- `colsample_bylevel` — fraction of features sampled per level during split search; less commonly tuned in CatBoost due to oblivious trees. Default 1.0.

**Key-parameter table**

| Parameter | Role | Default |
|---|---|---|
| `iterations` | max trees / boosting rounds | 1000 |
| `learning_rate` | step-size shrinkage | auto (~0.03) |
| `depth` | oblivious-tree depth | 6 |
| `l2_leaf_reg` | L2 penalty on leaf values | 3.0 |
| `loss_function` | training objective | `RMSE` / `Logloss` |
| `eval_metric` | validation / early-stopping metric | matches loss |
| `cat_features` | columns to treat as categorical (no pre-encoding) | `None` |
| `one_hot_max_size` | cardinality ≤ threshold → OHE, else Ordered TS | 2 |
| `max_ctr_complexity` | max categoricals per auto-combination | 4 |
| `has_time` | respect row time order in Ordered TS | `False` |
| `border_count` | numeric discretization bins | 254 CPU / 128 GPU |
| `bootstrap_type` | weight-sampling scheme | `Bayesian` |
| `subsample` | row subsample fraction | 1.0 CPU / 0.8 GPU |
| `task_type` | `CPU` or `GPU` | `CPU` |
| `early_stopping_rounds` | patience for early stop (needs `eval_set`) | `None` |

**Configuration example (with `Pool` and early stopping)**

```python
import pandas as pd
from catboost import CatBoostClassifier, Pool

train_data = pd.DataFrame({
    'num_feature1': [1.2, 3.4, 0.5, 2.1, 4.5, 1.8],
    'num_feature2': [5, 2, 8, 6, 3, 7],
    'cat_feature1': ['A', 'B', 'A', 'C', 'B', 'A'],
    'cat_feature2': ['X', 'Y', 'Y', 'X', 'X', 'Y'],
    'target':       [1, 0, 1, 0, 1, 0],
})
eval_data = pd.DataFrame({
    'num_feature1': [2.5, 0.8, 3.1],
    'num_feature2': [4, 9, 1],
    'cat_feature1': ['B', 'A', 'C'],
    'cat_feature2': ['Y', 'X', 'Y'],
    'target':       [0, 1, 0],
})

categorical_features_indices = [2, 3]  # cat_feature1, cat_feature2

# Pool packages data + labels + categorical metadata efficiently
train_pool = Pool(data=train_data.drop('target', axis=1),
                  label=train_data['target'],
                  cat_features=categorical_features_indices)
eval_pool = Pool(data=eval_data.drop('target', axis=1),
                 label=eval_data['target'],
                 cat_features=categorical_features_indices)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    cat_features=categorical_features_indices,
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100,
    # task_type='GPU',
    # devices='0',
)

model.fit(train_pool, eval_set=eval_pool, plot=False)

print(f"Best score: {model.get_best_score()['validation']['AUC']:.4f}")
print(f"Best iteration: {model.get_best_iteration()}")
```

**Configuration checklist.** Always pass `cat_features` (in raw categorical form) — this unlocks CatBoost's specialized handling. Use `early_stopping_rounds` with an `eval_set` to size `iterations` and curb overfitting. Tune `learning_rate`, `depth`, `l2_leaf_reg` first. Consider `task_type='GPU'` for large data with compatible hardware. Experiment with `one_hot_max_size` / CTR config (`max_ctr_complexity`, …) for high-cardinality features or complex interactions.

---

### 6.8 Hands-On: Implementing CatBoost

Setup — install the libraries:

```bash
pip install catboost pandas scikit-learn plotly
```

The demo uses the **Adult** census income dataset (predict income >$50K), which mixes numeric and categorical columns. The point: **do not** one-hot or label-encode the categoricals — just identify their column indices and hand them to CatBoost.

**Load and prepare data**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier, Pool
import plotly.graph_objects as go

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income',
]
data = pd.read_csv(url, header=None, names=column_names,
                   sep=r',\s*', engine='python', na_values='?')

# CatBoost can handle NaN natively; dropping here just for simplicity
data.dropna(inplace=True)

X = data.drop('income', axis=1)
y = data['income'].apply(lambda v: 1 if v == '>50K' else 0)

# Identify categorical columns by index (non-numeric dtypes)
categorical_features_indices = np.where(X.dtypes != np.number)[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

**Train a baseline classifier** — the crucial argument is `cat_features`:

```python
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
)

model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")
```

`eval_set` lets CatBoost monitor unseen-data performance (via `eval_metric='AUC'`) and apply `early_stopping_rounds` to stop when AUC stalls; `verbose` controls print frequency.

**Using the `Pool` class** — an optimized container bundling data, labels, and categorical metadata; can help with large data or repeated experiments. Results match the plain-DataFrame run:

```python
train_pool = Pool(data=X_train, label=y_train,
                  cat_features=categorical_features_indices)
eval_pool = Pool(data=X_test, label=y_test,
                 cat_features=categorical_features_indices)

model_pooled = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=100,
)
model_pooled.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)

y_pred_proba_pooled = model_pooled.predict_proba(eval_pool)[:, 1]
print(f"AUC (Pool): {roc_auc_score(y_test, y_pred_proba_pooled):.4f}")
```

**Feature importance** — numeric and categorical features are ranked together, with no manual conversion of the categoricals:

```python
feature_importances = model.get_feature_importance(train_pool)
importance_df = (
    pd.DataFrame({'feature': X_train.columns, 'importance': feature_importances})
    .sort_values(by='importance', ascending=False)
)
print(importance_df)

fig = go.Figure(go.Bar(
    x=importance_df['importance'], y=importance_df['feature'],
    orientation='h', marker_color='#228be6',
))
fig.update_layout(
    title='CatBoost Feature Importance',
    xaxis_title='Importance', yaxis_title='Feature',
    yaxis={'categoryorder': 'total ascending'}, height=500,
)
# fig.show()
```

On Adult, both numeric (`capital-gain`, `age`) and categorical (`relationship`, `marital-status`, `occupation`) features rank as top contributors — evidence of CatBoost's integrated handling.

**Takeaways from the practice session:** minimal categorical preprocessing (just name/index them via `cat_features`); a familiar scikit-learn-like API; use `eval_set` + `early_stopping_rounds` for overfitting control and iteration selection; easy feature-importance extraction. Best performance still needs hyperparameter tuning (`learning_rate`, `depth`, `l2_leaf_reg`, and CatBoost-specific knobs like `one_hot_max_size`) — covered systematically in Chapter 8.

---

### Key takeaways

- **Naive categorical encodings all fail somehow:** OHE explodes high cardinality into sparse, deep-tree-inducing columns; label encoding invents false ordinality; mean/target encoding leaks the target and overfits.
- **Ordered TS** encodes each row from *only its predecessors* in a random permutation, excluding its own target, with prior smoothing $\hat{x} = \frac{\sum_{j<i}[x_j=x_i]y_j + a\,p}{\sum_{j<i}[x_j=x_i] + a}$ to stabilize rare categories.
- **Ordered Boosting** applies the same "only-the-past" principle to residuals: sample $i$'s gradient comes from a model $M_{m-1}$ trained on samples excluding $i$, giving unbiased residuals and fixing prediction shift.
- **Feature combinations** are built greedily during tree growth (ancestor categoricals × new one), encoded with Ordered TS, controlled by `max_ctr_complexity`.
- **Oblivious (symmetric) trees** use one split per level → balanced, vectorizable, GPU-friendly, and implicitly regularizing; the ensemble compensates for a single tree's lower expressiveness.
- **In practice:** pass raw categoricals via `cat_features` (never pre-encode), use `Pool`, and always pair `eval_set` with `early_stopping_rounds`. GPU (`task_type='GPU'`) shines on medium/large data.

> **Relevance to our work:** Reach for CatBoost over LightGBM when a competition dataset has **many high-cardinality categorical features** (IDs, locations, product/category codes) or when a strong categorical predictor makes target/mean encoding **leakage-prone** — exactly where Ordered TS + Ordered Boosting deliver a leakage-free encoding out of the box with minimal preprocessing. LightGBM (native categorical splitting) or XGBoost often remain stronger on predominantly numeric data or when you need the extra flexibility of asymmetric trees; treat CatBoost as a distinct ensemble member and blend candidate whose bias/variance profile differs from the level-/leaf-wise learners.

---

## Chapter 7 — Advanced Topics & Customization

> Synthesized study notes from the course *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 7. These notes reconstruct the technical content and clean up the KaTeX-mangled math and code from the source lessons; they are for study reference, not a verbatim transcript. This is the course's longest, most code-heavy chapter, covering model interpretability (SHAP / TreeSHAP), probability calibration, custom objectives and evaluation metrics, and imbalanced-data techniques.

---

### 7.1 Understanding Model Interpretability with SHAP

Gradient boosting models are powerful predictors but effectively operate as **black boxes**: the sequential addition of many deep trees makes it hard to see *why* a particular prediction was made or which features drive overall behaviour. Standard feature-importance measures (split gain, permutation) give a coarse global picture but lack the resolution to explain a single prediction or capture complex feature interactions. This opacity impedes trust, complicates debugging, and blocks use in regulated settings.

**SHAP (SHapley Additive exPlanations)** is a principled solution rooted in cooperative game theory. It treats a prediction as a *game* in which the features cooperate to produce the output, and it uses **Shapley values** to fairly distribute the "payout" (the gap between the prediction and a baseline) among the "players" (the features).

#### Shapley values

The Shapley value $\phi_i$ of feature $i$ is its **average marginal contribution** to the prediction across all possible feature coalitions (subsets). It quantifies the effect of including that feature's value. For a general ML model, computing exact Shapley values is expensive because it requires evaluating the model over all $2^M$ feature subsets, where $M$ is the number of features.

#### The SHAP framework and its guarantees

SHAP provides efficient algorithms to estimate Shapley values, with three theoretical properties:

- **Local accuracy (additivity).** For a given prediction, the SHAP values of all features sum to the difference between the prediction $f(x)$ and the baseline expected prediction $\phi_0 = E[f(x)]$ (usually the mean prediction over the training data):

$$
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i
$$

- **Missingness.** A feature that is genuinely absent from (marginalized out of) a coalition receives a SHAP value of zero.
- **Consistency.** If the model changes so that a feature's marginal contribution increases or stays the same (regardless of other features), its SHAP value does not decrease. This ties feature importance to how much the model actually relies on the feature — a property gain-based importance can violate.

For tree ensembles like gradient boosting, SHAP offers a specialized, efficient algorithm (**TreeSHAP**, next section) that computes *exact* Shapley values far faster than model-agnostic methods.

#### Using SHAP with boosting libraries

The Python `shap` library integrates with XGBoost, LightGBM and CatBoost. The main entry point is `shap.Explainer`, or the tree-optimized `shap.TreeExplainer`. Workflow: (1) fit the model as usual; (2) build an explainer from the model (optionally passing background data to fix the baseline $\phi_0$); (3) compute SHAP values on the instances you want to explain.

```python
import xgboost
import shap

# 1. Train the model (XGBoost example)
model = xgboost.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Create a TreeExplainer (optimized for XGBoost / LightGBM / CatBoost trees)
explainer = shap.TreeExplainer(model)

# 3. Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)
# shap_values is typically a (num_instances, num_features) array;
# for multiclass it is a list of arrays (one per class).

print(f"SHAP values shape: {shap_values.shape}")
print(f"Baseline expected value: {explainer.expected_value}")

# Explain a single prediction (first test instance)
shap_values_single = explainer.shap_values(X_test.iloc[0, :])
```

#### Visualizing SHAP values

- **Force plot (local).** Explains one prediction, showing which features push it away from the baseline. Features that raise the prediction are red, those that lower it are blue; the final prediction is the baseline plus the sum of all contributions.
- **Summary plot (global).** One dot per (feature, instance) SHAP value. Features are ranked by mean absolute SHAP value; horizontal position shows the SHAP value (effect direction and size) and colour encodes the raw feature value (high/low), revealing whether high values push predictions up or down.
- **Dependence plot.** Plots a feature's value (x) against its SHAP value (y), optionally coloured by a second, possibly interacting feature to expose interaction effects.

*Source references: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017, arXiv:1705.07874); Lundberg, Erion & Lee, "Consistent Individualized Feature Attribution for Tree Ensembles" (arXiv:1802.03888) — introduces TreeSHAP.*

---

### 7.2 TreeSHAP for Gradient Boosting

The generic, model-agnostic **KernelExplainer** approximates Shapley values by perturbing inputs and observing output changes. For gradient boosting ensembles — hundreds or thousands of trees, many features — this sampling-and-re-evaluation approach becomes prohibitively slow. **TreeSHAP** (Lundberg et al.) is an algorithm built specifically for tree models (XGBoost, LightGBM, CatBoost) that computes **exact** Shapley values much more efficiently.

#### How TreeSHAP works

Instead of sampling and re-evaluating, TreeSHAP exploits the tree structure to compute conditional expectations $E[f(x) \mid x_S]$ — the model's expected output given only the feature values in subset $S$. The Shapley value is the weighted difference of these conditional expectations as feature $i$ is added to each subset:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!}\,\Big[\,E[f(x) \mid x_{S \cup \{i\}}] - E[f(x) \mid x_S]\,\Big]
$$

where $F$ is the full feature set and $x_S$ the values of the features in $S$. TreeSHAP avoids the exponential $2^{|F|}$ enumeration by using a **polynomial-time** algorithm that pushes all feature subsets down the tree paths simultaneously: at each split node it tracks, per feature, the proportion of subsets following the left vs. right branch, maintaining a weighted average of conditional expectations from root to leaf. This runs on every tree in the ensemble; the per-tree Shapley values are then combined (the ensemble prediction is a sum of tree outputs, often after a link function such as the logistic transform).

#### Benefits for gradient boosting

- **Efficiency.** Orders of magnitude faster than Kernel SHAP on tree ensembles — exact SHAP values for thousands of predictions in seconds/minutes rather than hours/days.
- **Exactness.** Computes theoretically exact Shapley values for tree models (under the standard feature-independence assumption), eliminating the approximation error of Kernel SHAP.
- **Interaction values.** The algorithm extends to efficiently compute **SHAP interaction values** $\phi_{ij}$, quantifying the extra contribution arising from the interaction between a pair of features, averaged over subsets.

#### Implementation

```python
import xgboost
import shap

# X_train / y_train: training data; X_explain: instances to explain
model = xgboost.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)

# Newer API: returns a shap.Explanation object (values, base_values, data, feature_names)
shap_values_obj = explainer(X_explain)

# SHAP values for the first prediction
print(f"SHAP values (instance 0): {shap_values_obj.values[0]}")
print(f"Base value (expected output): {shap_values_obj.base_values[0]}")
print(f"Model prediction (instance 0): {model.predict(X_explain.iloc[[0]])[0]}")
# Verify: sum(shap_values_obj.values[0]) + base_values[0] ≈ prediction

# Visualization (Jupyter):
# shap.initjs()
# shap.force_plot(shap_values_obj.base_values[0], shap_values_obj.values[0], X_explain.iloc[0])
# shap.summary_plot(shap_values_obj, X_explain)
```

#### Caveats

- **Feature-independence assumption.** Like all Shapley-based methods, TreeSHAP implicitly assumes feature independence when computing conditional expectations. With highly correlated features, interpretation requires care, since the algorithm may average over unrealistic feature-value combinations.
- **Cost at scale.** Much faster than alternatives, but millions of instances or tens of thousands of trees can still be time/memory intensive. `shap_values(..., approximate=True)` or `check_additivity=False` trade exactness for speed.
- **Scope.** TreeSHAP is for tree models only; use `DeepExplainer`, `LinearExplainer`, or the model-agnostic `KernelExplainer` for other model types.

*Source references: Lundberg & Lee (NeurIPS 2017); Shapley, "A Value for n-person Games" (1953) — the game-theoretic foundation.*

---

### 7.3 Global vs. Local Explanations

SHAP (and its efficient TreeSHAP variant) supports two complementary levels of explanation, and distinguishing them is key to fully understanding a model.

#### Global explanations — the big picture

Global explanations describe the model's overall behaviour across the whole dataset: *which features are most influential on average?* and *what is the general relationship between a feature and the output?* The standard SHAP-based global importance for feature $j$ is the **mean absolute SHAP value** over all $n$ instances:

$$
\text{GlobalImportance}_j = \frac{1}{n} \sum_{i=1}^{n} |\phi_{ij}|
$$

where $\phi_{ij}$ is the SHAP value of feature $j$ for instance $i$. Higher mean $|\phi|$ = more influential overall. This is a more reliable and **consistent** ranking than gain or split-count importance. Visualized via the **summary plot** (importance + effect direction combined) or a **bar chart** of mean $|\phi|$; the **dependence plot** shows the global shape of a feature's effect.

#### Local explanations — a single prediction

Local explanations answer *why did the model make this specific prediction for this instance?* — e.g. why this customer's loan was rejected, or which factors most raised this user's churn probability. SHAP values are inherently local: $\phi_{ij}$ measures how feature $j$ pushed instance $i$'s prediction away from the baseline. The core SHAP equation links the baseline $E[f(X)]$ to the instance prediction $f(x_i)$:

$$
f(x_i) = E[f(X)] + \sum_{j=1}^{M} \phi_{ij}
$$

The additivity means each feature's positive/negative contribution to a single prediction is directly readable, typically via a **force plot** (red = pushes up, blue = pushes down; block size = magnitude).

#### Complementary perspectives

- **Global** is valuable for understanding the main drivers, comparing models, and guiding feature engineering.
- **Local** is indispensable for debugging surprising predictions, explaining decisions to stakeholders/customers, assessing individual fairness, and building trust in specific outcomes.

---

### 7.4 Probability Calibration for Classification

Gradient boosting classifiers often achieve strong AUC/F1 but their raw output scores are **not necessarily true probabilities**: a score of 0.9 does not mean a 90% objective chance of the positive class. Optimizing log loss tends to push scores toward 0 and 1 to sharpen discrimination, which can distort the probability interpretation. When calibrated probabilities matter — for decisions, risk assessment, thresholding, or ensembling — you should calibrate the outputs.

#### What calibration means and how to check it

A perfectly calibrated binary classifier has the property that among instances predicted with probability $p$, the actual positive fraction is close to $p$. The standard diagnostic is a **reliability diagram (calibration curve)**: bin the predicted probabilities (0–0.1, 0.1–0.2, …), and for each bin plot the mean predicted probability against the observed positive fraction. The diagonal is perfect calibration; bars below it mean over-confidence (predicted > actual frequency), above it mean under-confidence.

Boosting can produce mis-calibrated probabilities because of its additive nature, its focus on correcting errors (which drives extreme scores at high-confidence points), and the specifics of tree construction — so checking (and possibly fixing) calibration of XGBoost/LightGBM/CatBoost is good practice.

#### Two calibration methods

Both are fit **after** the main classifier, on a **separate calibration set** (a held-out split not used to train the base model). Calibrating on the training data gives over-optimistic results.

**Platt scaling** — parametric. Assumes the deviation between scores $s$ and true probabilities can be corrected by fitting a sigmoid. It finds parameters $A, B$ so that:

$$
P_{\text{Platt}}(y = 1 \mid s) = \frac{1}{1 + \exp(A s + B)}
$$

$A$ and $B$ are found by minimizing log loss on the calibration set. Platt scaling works best when the calibration curve is monotonic and S-shaped; it is computationally cheap and needs relatively little data.

**Isotonic regression** — non-parametric. Fits a non-decreasing, piecewise-constant step function mapping scores to observed targets (least-squares best fit preserving input order), using the **Pool Adjacent Violators Algorithm (PAVA)**. It makes fewer assumptions about the bias shape and fits better when the curve is not sigmoidal, but needs more data for stable results and can produce sharp steps in the calibrated probabilities.

#### Implementation in scikit-learn

`CalibratedClassifierCV` wraps training + calibration with `method='sigmoid'` (Platt) or `method='isotonic'`. Internally it cross-validates: for each fold it (1) trains a base classifier on the fold's training part, (2) predicts on the fold's test part, (3) fits a calibrator on those predictions vs. true labels; finally the base model is retrained on all data and the per-fold calibrators are averaged.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Base model
base_model = XGBClassifier(eval_metric="logloss", random_state=42)
base_model.fit(X_train, y_train)
y_uncal = base_model.predict_proba(X_test)[:, 1]

# 2. Calibrate (CV handled internally)
calibrated_sigmoid = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
calibrated_sigmoid.fit(X_train, y_train)
y_sigmoid = calibrated_sigmoid.predict_proba(X_test)[:, 1]

calibrated_isotonic = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
calibrated_isotonic.fit(X_train, y_train)
y_isotonic = calibrated_isotonic.predict_proba(X_test)[:, 1]

# 3. Reliability diagram
fig, ax = plt.subplots(figsize=(8, 8))
CalibrationDisplay.from_predictions(y_test, y_uncal,    n_bins=10, name="Uncalibrated XGBoost", ax=ax, marker="^")
CalibrationDisplay.from_predictions(y_test, y_sigmoid,  n_bins=10, name="Platt scaling",        ax=ax, marker="o")
CalibrationDisplay.from_predictions(y_test, y_isotonic, n_bins=10, name="Isotonic regression",  ax=ax, marker="s")
ax.set_title("Calibration curves (reliability diagram)")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
ax.legend(); plt.show()
```

> Note: in recent scikit-learn versions the first argument to `CalibratedClassifierCV` is `estimator` (the `base_estimator` keyword used in older code is deprecated).

#### Practical considerations

- **Independent data.** Always calibrate on data unseen by the base model; `CalibratedClassifierCV` handles this via CV, otherwise use a dedicated hold-out.
- **Method choice.** With limited calibration data or a known S-shaped bias, prefer Platt scaling; isotonic is more flexible but data-hungry and can be less smooth. Evaluate both with the **Brier score** or the reliability diagram.
- **Effect on other metrics.** Calibration improves probability reliability (Brier score, log loss) and may slightly reduce ranking metrics like AUC — usually a small effect. Calibrate when accurate probabilities are the primary need.

*Source references: Platt (1999); Zadrozny & Elkan (2002, isotonic); Niculescu-Mizil & Caruana, "Predicting good probabilities with supervised learning" (2005).*

---

### 7.5 Implementing Custom Loss Functions

Standard losses (MSE for regression, log loss for classification) do not always match the true business objective — you might want to penalize over-prediction more than under-prediction, or weight certain error types more heavily. XGBoost and LightGBM let you define and optimize your own custom **objective (loss)** function.

#### The role of gradient and Hessian

Recall (Ch. 2) that boosting adds weak learners sequentially, each correcting the ensemble's errors. At boosting iteration $m$ we add $f_m(x)$ scaled by learning rate $\eta$:

$$
F_m(x) = F_{m-1}(x) + \eta\, f_m(x)
$$

Modern libraries fit $f_m$ using a **second-order Taylor expansion** of the loss around the previous prediction $F_{m-1}(x_i)$:

$$
L(y_i, F_{m-1}(x_i) + f_m(x_i)) \approx L(y_i, F_{m-1}(x_i)) + g_i\, f_m(x_i) + \tfrac{1}{2} h_i\, f_m(x_i)^2
$$

where $g_i$ is the **gradient** (first derivative of the loss w.r.t. the prediction) and $h_i$ the **Hessian** (second derivative), both evaluated at $F_{m-1}(x_i)$:

$$
g_i = \left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x_i)=F_{m-1}(x_i)}
\qquad
h_i = \left[\frac{\partial^2 L(y_i, F(x_i))}{\partial F(x_i)^2}\right]_{F(x_i)=F_{m-1}(x_i)}
$$

**The custom-objective contract:** you do **not** supply the loss $L$ itself. You supply a function that, given the current predictions and true labels, returns the per-instance **gradient** $g_i$ and **Hessian** $h_i$. The library uses these internally when computing split gains and leaf values.

#### Worked examples of gradient and Hessian

**MSE**, $L = (y - \hat y)^2$: $g = -2(y - \hat y) = 2(\hat y - y)$ and $h = 2$ (constant curvature).

**Binary logistic**: `preds` are the raw margins $\hat y$ (pre-sigmoid logits), $p = \sigma(\hat y) = 1/(1+e^{-\hat y})$, and $L = -[y\log p + (1-y)\log(1-p)]$. Differentiating w.r.t. the raw score $\hat y$ (not $p$) gives the clean forms:

$$
g = p - y \qquad h = p(1-p)
$$

**Asymmetric MSE** (over-prediction penalized $A$ times more than under-prediction), with $A > 1$:

$$
L(y, \hat y) =
\begin{cases}
(y - \hat y)^2 & \text{if } \hat y \le y \\
A\,(y - \hat y)^2 & \text{if } \hat y > y
\end{cases}
$$

Differentiating w.r.t. $\hat y$:

$$
g =
\begin{cases}
2(\hat y - y) & \hat y \le y \\
2A(\hat y - y) & \hat y > y
\end{cases}
\qquad
h =
\begin{cases}
2 & \hat y \le y \\
2A & \hat y > y
\end{cases}
$$

```python
import numpy as np

def asymmetric_mse_obj(preds, dtrain):
    """Custom objective: asymmetric MSE (over-prediction penalized A× more). XGBoost signature."""
    labels = dtrain.get_label()   # dtrain is an XGBoost DMatrix
    residual = preds - labels
    A = 1.5                       # over-prediction penalty factor
    grad = np.where(preds <= labels, 2.0 * residual, 2.0 * A * residual)
    hess = np.where(preds <= labels, 2.0, 2.0 * A)
    return grad, hess

# LightGBM signature differs: it passes (labels, preds) directly.
def asymmetric_mse_obj_lgb(labels, preds):
    residual = preds - labels
    A = 1.5
    grad = np.where(preds <= labels, 2.0 * residual, 2.0 * A * residual)
    hess = np.where(preds <= labels, 2.0, 2.0 * A)
    return grad, hess
```

#### Wiring into XGBoost

Pass the objective via the `obj` argument of `xgboost.train`; typically also pass a matching custom `feval`.

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
params = {"eta": 0.1, "max_depth": 3}

def asymmetric_mse_eval(preds, dtrain):
    labels = dtrain.get_label()
    A = 1.5
    errors = preds - labels
    loss = np.where(preds <= labels, errors**2, A * (errors**2))
    return "asymMSE", np.mean(loss)   # (name, value)

bst = xgb.train(
    params, dtrain,
    num_boost_round=100,
    obj=asymmetric_mse_obj,        # custom objective
    feval=asymmetric_mse_eval,     # custom eval (optional)
    evals=[(dtrain, "train"), (dvalid, "eval")],
    early_stopping_rounds=10,
    maximize=False,                # minimize asymMSE
)
```

#### Wiring into LightGBM

Pass the objective via `fobj` (and set `objective: None`, `metric: 'None'` in params so the built-ins are disabled). LightGBM's `feval` returns a third element `is_higher_better`.

```python
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
params = {"objective": None, "metric": "None", "learning_rate": 0.1, "num_leaves": 31}

def asymmetric_mse_eval_lgb(labels, preds):
    A = 1.5
    errors = preds - labels
    loss = np.where(preds <= labels, errors**2, A * (errors**2))
    return "asymMSE", np.mean(loss), False   # (name, value, is_higher_better)

gbm = lgb.train(
    params, lgb_train,
    num_boost_round=100,
    valid_sets=lgb_eval,
    fobj=asymmetric_mse_obj_lgb,
    feval=asymmetric_mse_eval_lgb,
    callbacks=[lgb.early_stopping(10, verbose=True)],
)
```

The scikit-learn wrappers (`XGBRegressor`/`XGBClassifier`, `LGBMRegressor`/`LGBMClassifier`) accept the callable directly via the `objective` parameter.

#### Things to get right

- **Mathematical correctness.** Double-check the derivatives — an error here silently optimizes the wrong thing. Verify with simple cases or numerical gradient checks.
- **Non-negative Hessian.** The Hessian is the loss curvature and must be $\ge 0$ (strictly positive is preferred) so each step's objective is convex and well-defined; zero/negative Hessians make the algorithm unstable.
- **Numerical stability.** Avoid divide-by-zero and overflow/underflow; add a small epsilon in denominators where needed.
- **Classification predictions are raw scores.** For classification, `preds` are pre-sigmoid/softmax margins — compute grad/Hessian w.r.t. those raw scores (e.g. logistic: $g = p - y$, $h = p(1-p)$).
- **Keep eval consistent.** Pair the objective with a `feval` measuring the actual loss / business metric you care about.

*Source references: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016); Ke et al., "LightGBM" (NeurIPS 2017).*

---

### 7.6 Implementing Custom Evaluation Metrics

The final measure of success is often a domain-specific KPI or competition metric that the built-in objectives don't express. The evaluation metric guides hyperparameter tuning and early stopping. Unlike a custom **objective** (which must supply gradient info to drive training), a custom **evaluation metric** simply scores predictions vs. labels at each round — it does not affect the gradient, only monitoring and stopping decisions.

#### Why custom metrics

- **Business KPIs** — e.g. minimizing large errors above a threshold, or ranking metrics like MAP@K / NDCG in recommenders.
- **Competition rules** — Kaggle often mandates metrics like **Quadratic Weighted Kappa (QWK)** or custom precision/recall variants; training/early-stopping on the competition metric helps even if a smoother loss is optimized.
- **Complex aspects** — fairness across subgroups, asymmetric error costs.
- **Monitoring vs. optimizing** — optimize log loss (nice math) but monitor F1 (user experience).

#### Structure of a custom metric function

A typical metric receives predictions and the label-carrying data object, computes the score, and returns a tuple: `metric_name` (str), `metric_value` (float), and (in LightGBM/CatBoost) `is_higher_better` (bool). The `is_higher_better` flag is critical for early stopping to work correctly (True for AUC/F1, False for RMSE/MAPE).

#### XGBoost

`xgb.train` takes `feval`; the function receives `preds` and a `DMatrix`. By default XGBoost assumes a custom metric should be **minimized** unless `maximize=True` is passed.

```python
import numpy as np
import xgboost as xgb

def xg_mape(preds: np.ndarray, dtrain: xgb.DMatrix):
    """Custom MAPE for XGBoost (lower is better)."""
    labels = dtrain.get_label()
    epsilon = 1e-6
    safe_labels = np.maximum(np.abs(labels), epsilon)   # guard divide-by-zero
    mape = np.mean(np.abs((labels - preds) / safe_labels))
    return "MAPE", mape

dtrain = xgb.DMatrix(X_train, label=y_train)
deval = xgb.DMatrix(X_eval, label=y_eval)
params = {"objective": "reg:squarederror", "eta": 0.1, "max_depth": 3}

bst = xgb.train(
    params, dtrain,
    num_boost_round=100,
    evals=[(dtrain, "train"), (deval, "eval")],
    feval=xg_mape,
    early_stopping_rounds=10,
    verbose_eval=10,
)
print(f"Best MAPE on validation: {bst.best_score}")
```

#### LightGBM

`feval` receives `preds` and a `Dataset`, and must return the third `is_higher_better` element. You can pass a *list* of metric functions.

```python
import numpy as np
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef

def lgbm_mcc(preds: np.ndarray, train_data: lgb.Dataset):
    """Matthews correlation coefficient (higher is better)."""
    labels = train_data.get_label()
    pred_labels = (preds > 0.5).astype(int)
    mcc = matthews_corrcoef(labels, pred_labels)
    return "MCC", mcc, True   # (name, value, is_higher_better)

params = {"objective": "binary", "metric": "binary_logloss",
          "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9}

bst = lgb.train(
    params, lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=["train", "eval"],
    feval=[lgbm_mcc],
    callbacks=[lgb.early_stopping(stopping_rounds=10, first_metric_only=False, verbose=True),
               lgb.log_evaluation(period=10)],
)
```

LightGBM uses the returned `is_higher_better` flag directly; both the `params['metric']` metrics and any `feval` metrics are monitored for early stopping.

#### CatBoost

CatBoost uses a **class-based** custom metric with three methods:

- `is_max_optimal(self)` — True if higher is better.
- `evaluate(self, approxes, target, weight)` — `approxes` is a list of lists of raw scores (one inner list per class); returns `(sum_metric, sum_weight)`.
- `get_final_error(self, error, weight)` — combines the sums into the final value.

```python
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score

class CatBoostF1Metric:
    def is_max_optimal(self):
        return True   # higher F1 is better

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1               # binary => one list of raw scores
        preds_raw = np.array(approxes[0])
        preds_prob = 1.0 / (1.0 + np.exp(-preds_raw))   # sigmoid on raw logits
        pred_labels = (preds_prob > 0.5).astype(int)
        f1 = f1_score(target, pred_labels)
        count = len(target) if target is not None else 0
        if count == 0:
            return 0.0, 0.0
        return f1 * count, count                # (sum_metric, sum_weight)

    def get_final_error(self, error_sum, weight_sum):
        return 0.0 if weight_sum == 0 else error_sum / weight_sum

model = CatBoostClassifier(
    iterations=100, learning_rate=0.1, loss_function="Logloss",
    custom_metric=[CatBoostF1Metric()], eval_metric="Logloss",
    early_stopping_rounds=10, verbose=10,
)
model.fit(Pool(X_train, label=y_train), eval_set=Pool(X_eval, label=y_eval))
```

#### Considerations

- **Efficiency.** Metrics run every round (and per CV fold) — vectorize with NumPy, avoid redundant work.
- **Correctness.** Unit-test the metric on known inputs before wiring into training; debugging inside the loop is painful.
- **`is_higher_better` / `is_max_optimal`.** Must be right or early stopping goes the wrong way.
- **Prediction format.** `preds`/`approxes` may be raw scores, probabilities, or final predictions depending on library/objective/API — apply sigmoid/softmax as needed.

*Source references: XGBoost "Custom Objective and Evaluation Metric" docs; Hastie, Tibshirani & Friedman, *ESL*; Manning et al. for MAP@K / NDCG.*

---

### 7.7 Handling Imbalanced Data with Boosting

Datasets are rarely perfectly balanced; the class of interest (fraud, disease, failure) is usually the minority. Standard GBMs can be biased toward the majority — achieving high overall accuracy by mostly predicting the majority while failing on the important minority class.

#### Weighting the minority: `scale_pos_weight`

The simplest lever (XGBoost, LightGBM) scales the **gradient and Hessian of positive-class instances** in the objective, so misclassifying the (usually minority) positive class is penalized more heavily and later trees focus on getting them right. A common rule of thumb:

$$
\text{scale\_pos\_weight} = \frac{\#\,\text{negative samples}}{\#\,\text{positive samples}}
$$

e.g. 900 negatives / 100 positives → `scale_pos_weight = 9`.

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
                           n_clusters_per_class=1, weights=[0.95, 0.05], flip_y=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)
spw = neg_count / pos_count

model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          scale_pos_weight=spw, random_state=42)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test),
                            target_names=["majority", "minority"]))
```

The optimal `scale_pos_weight` may differ from the naive ratio (especially for non-accuracy metrics) and is usually tuned like any other hyperparameter. In **LightGBM**, `is_unbalance=True` is a simpler alternative/complement.

#### Custom objectives for imbalance

Custom objectives (§7.5) give more fundamental control — you can implement a **weighted** cross-entropy, or **Focal Loss**.

**Focal Loss** (from object detection, adaptable to tabular) down-weights easy (usually majority) examples so training focuses on hard (usually minority) ones. Each example's loss is scaled by $(1 - p_t)^\gamma$, where $p_t$ is the predicted probability of the correct class and $\gamma$ is the focusing parameter (larger $\gamma$ = more down-weighting of easy examples); an $\alpha$ term additionally balances the classes. Implementing it means deriving grad/Hessian w.r.t. the raw scores:

```python
import numpy as np

def focal_loss_objective(y_true, y_pred_raw, gamma=2.0, alpha=0.25):
    """Focal-loss objective (binary, simplified). y_pred_raw = raw margins."""
    p = 1.0 / (1.0 + np.exp(-y_pred_raw))                 # sigmoid
    # Loss components (for reference):
    #   y=1:  -alpha       * (1 - p)**gamma * log(p)
    #   y=0:  -(1 - alpha) *  p     **gamma * log(1 - p)
    grad = ...   # first derivative w.r.t. y_pred_raw  (full derivation omitted in source)
    hess = ...   # second derivative w.r.t. y_pred_raw
    return grad, hess

# model = xgb.XGBClassifier(objective=focal_loss_objective, ...)
```

#### Imbalance-sensitive evaluation metrics

Accuracy is misleading under imbalance (a 1%-positive dataset scores 99% accuracy by always predicting the majority). Use metrics that reflect minority performance:

- **Precision** $= TP/(TP+FP)$ — matters when false positives are costly.
- **Recall / sensitivity** $= TP/(TP+FN)$ — matters when false negatives are costly (missed disease).
- **F1** — harmonic mean of precision and recall.
- **AUC-PR** (area under precision–recall curve) — preferred over ROC-AUC for highly imbalanced data, since ROC-AUC can be over-optimistic due to the large number of true negatives.
- **MCC** (Matthews correlation coefficient) — uses all four confusion-matrix quadrants, ranges $-1$ to $+1$; a balanced measure even under imbalance.

Tune hyperparameters and early-stop on AUC-PR / F1 rather than accuracy or log loss.

```python
import lightgbm as lgb

model_lgbm = lgb.LGBMClassifier(objective="binary", metric="auc",
                                is_unbalance=True, random_state=42)
model_lgbm.fit(X_train, y_train,
               eval_set=[(X_test, y_test)],
               eval_metric="auc",
               callbacks=[lgb.early_stopping(10)])
print(classification_report(y_test, model_lgbm.predict(X_test),
                            target_names=["majority", "minority"]))
```

#### Choosing a strategy

No single best method: start simple with `scale_pos_weight` / `is_unbalance`; ensure tuning and early stopping use an imbalance-appropriate metric (AUC-PR, F1); reach for custom objectives (Focal, weighted CE) when you need maximum control; and sampling methods like **SMOTE** (synthetic minority over-sampling) are another complementary axis. Always compare methods under a proper validation scheme with imbalance-sensitive metrics.

*Source references: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017); XGBoost `scale_pos_weight` docs; LightGBM `is_unbalance` docs; Chawla et al., "SMOTE" (2002).*

---

### 7.8 Hands-On: Custom Objectives & SHAP

This capstone exercise combines a custom objective with SHAP analysis: train one model with the standard objective and one with a custom asymmetric objective, then use SHAP to compare how the custom objective changes model behaviour.

#### Scenario: asymmetric-cost regression

Under-predicting demand (lost sales, unhappy customers) is far costlier than over-predicting (excess inventory). Standard MSE treats both equally. Here the loss penalizes **under-prediction** ($\hat y < y$) by a factor $\alpha > 1$ (note this is the *opposite* asymmetry to §7.5's example):

$$
L(y, \hat y) =
\begin{cases}
\alpha\,(y - \hat y)^2 & \text{if } y > \hat y \quad (\text{under-prediction}) \\
(y - \hat y)^2 & \text{if } y \le \hat y \quad (\text{over- or exact prediction})
\end{cases}
$$

Gradient and Hessian w.r.t. $\hat y$. With $w = \alpha$ when $y > \hat y$ and $w = 1$ otherwise:

$$
g = \frac{\partial L}{\partial \hat y} = -2w\,(y - \hat y)
\qquad
h = \frac{\partial^2 L}{\partial \hat y^2} = 2w
$$

#### Implementing the objective (factory pattern)

```python
import numpy as np

def asymmetric_mse_objective(alpha):
    """Factory: custom asymmetric MSE penalizing under-prediction (y_true > y_pred) by alpha."""
    def objective_function(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        residual = y_true - y_pred
        weight = np.where(residual > 0, alpha, 1.0)   # residual > 0  <=>  under-prediction
        grad = -2.0 * weight * residual               # g = -2w(y - y_hat)
        hess = 2.0 * weight                           # h = 2w
        return grad, hess
    return objective_function

# Under-prediction costs 3× more than over-prediction
custom_objective = asymmetric_mse_objective(alpha=3.0)
```

#### Training standard vs. custom

```python
import xgboost as xgb
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
y = y + 5 * np.sin(X["feature_0"])**2 + np.random.normal(0, 10, size=y.shape[0])
y = np.maximum(0, y)   # non-negative target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

common = dict(n_estimators=100, learning_rate=0.1, max_depth=3,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

xgb_std = xgb.XGBRegressor(objective="reg:squarederror", **common).fit(X_train, y_train)
xgb_custom = xgb.XGBRegressor(objective=custom_objective, **common).fit(X_train, y_train)

y_pred_std = xgb_std.predict(X_test)
y_pred_custom = xgb_custom.predict(X_test)

# Asymmetric cost: under-prediction (positive residual) weighted 3×
def asym_cost(err):
    return np.sum(np.maximum(0, err)**2 * 3.0) + np.sum(np.maximum(0, -err)**2)

print(f"Std   MSE: {mean_squared_error(y_test, y_pred_std):.4f}, asym cost: {asym_cost(y_test - y_pred_std):.2f}")
print(f"Custom MSE: {mean_squared_error(y_test, y_pred_custom):.4f}, asym cost: {asym_cost(y_test - y_pred_custom):.2f}")
```

Expected outcome: the custom model has a **slightly higher plain MSE** but a **lower asymmetric cost** — it deliberately reduces costly under-predictions, biasing predictions upward.

#### Explaining with SHAP

```python
import shap
import matplotlib.pyplot as plt

explainer_std = shap.TreeExplainer(xgb_std)
shap_values_std = explainer_std.shap_values(X_test)
explainer_custom = shap.TreeExplainer(xgb_custom)
shap_values_custom = explainer_custom.shap_values(X_test)

# Global: summary plots (compare feature-importance ranking/magnitude)
shap.summary_plot(shap_values_std, X_test, show=False);    plt.title("SHAP importance (standard MSE)");            plt.show()
shap.summary_plot(shap_values_custom, X_test, show=False); plt.title("SHAP importance (asymmetric MSE, alpha=3)"); plt.show()

# Feature-level: dependence plots
shap.dependence_plot("feature_0", shap_values_std, X_test, interaction_index=None, show=False);    plt.show()
shap.dependence_plot("feature_0", shap_values_custom, X_test, interaction_index=None, show=False); plt.show()

# Local: force plots for a single instance
i = 0
shap.force_plot(explainer_std.expected_value,    shap_values_std[i, :],    X_test.iloc[i, :], matplotlib=True, show=False); plt.show()
shap.force_plot(explainer_custom.expected_value, shap_values_custom[i, :], X_test.iloc[i, :], matplotlib=True, show=False); plt.show()
```

The comparison reveals whether optimizing a different objective subtly shifts feature-importance rankings (summary plots), changes a feature's effect shape (dependence plots), or alters per-instance contributions and the baseline expected value (force plots). An actual-vs-predicted scatter (with a $y = x$ reference line) visually confirms the asymmetric model has fewer points far below the diagonal — i.e. fewer severe under-predictions.

#### Takeaway from the exercise

By penalizing under-prediction more heavily, the model's predictions align with the asymmetric cost structure, and SHAP makes the *how* transparent at both global and local levels — pairing custom objectives with interpretability yields models that are both fit-for-purpose and trustworthy.

---

### Key takeaways

- **SHAP unifies interpretability** via Shapley values with local accuracy $f(x) = \phi_0 + \sum_j \phi_j$, missingness, and consistency; it delivers both **global** (mean $|\phi_j|$ over instances) and **local** (per-instance $\phi_{ij}$) explanations from the same values.
- **TreeSHAP** makes exact Shapley values tractable for tree ensembles by pushing feature subsets down tree paths in polynomial time — orders of magnitude faster than Kernel SHAP; watch the feature-independence assumption under correlation.
- **Calibration** (Platt = parametric sigmoid, isotonic = non-parametric PAVA step function) fixes boosting's tendency to output non-probabilistic scores; fit it on **held-out** data (`CalibratedClassifierCV`), check with reliability diagrams / Brier score. Needed when probabilities (not just ranking) matter.
- **Custom objective contract:** supply per-instance gradient $g_i = \partial L/\partial \hat y_i$ and Hessian $h_i = \partial^2 L/\partial \hat y_i^2$, not the loss. Key forms: MSE $(g = 2(\hat y - y), h = 2)$; logistic on raw margins $(g = p - y, h = p(1-p))$; asymmetric/Huber via piecewise weights. Keep $h \ge 0$ and derivatives numerically stable.
- **Custom eval metrics** don't touch the gradient; return `(name, value[, is_higher_better])`. The higher/lower-better flag drives early stopping. XGBoost defaults to minimize (`maximize=True` to flip); LightGBM/CatBoost make direction explicit.
- **Imbalance toolkit:** `scale_pos_weight ≈ neg/pos` (or LightGBM `is_unbalance`), weighted / Focal-Loss custom objectives, sampling (SMOTE), and — crucially — imbalance-sensitive metrics (AUC-PR, F1, MCC) for tuning and early stopping.

> **Relevance to our work:** Custom objectives are our lever for aligning training with the exact competition metric — deriving grad/Hessian for approximations of **QWK** (ordinal targets), **SMAPE** (asymmetric percentage error), or Focal Loss for rare-class comps, then confirming with a matching `feval` and early stopping on the leaderboard metric. **SHAP/TreeSHAP** drives principled **feature pruning** (drop low mean-$|\phi|$ features to cut noise and speed CV) and interaction discovery, and lets us sanity-check that a model relies on sensible signals before trusting it. **Calibration** (isotonic/Platt via `CalibratedClassifierCV`, on out-of-fold predictions) is essential for **log-loss / Brier-scored** competitions and whenever we blend models — well-calibrated probabilities average far more coherently than raw scores.

---

## Chapter 8 — Hyperparameter Optimization Strategies

> Synthesized study notes from the course *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 8. These notes reconstruct the technical content and clean up the KaTeX-mangled math from the source lessons; they are for study reference, not a verbatim transcript.

---

### 8.1 Why Hyperparameter Tuning Matters

The default settings of XGBoost, LightGBM and CatBoost give a *reasonable starting point*, but they rarely produce the best model for a specific dataset. Tuning is not a final cosmetic step — it is often a decisive factor in whether a model succeeds. This matters especially for gradient boosting because the model is built **sequentially**: each new weak learner (usually a tree) corrects the errors of the ensemble so far, and how those trees are built, how much each contributes, and how their complexity is controlled are *all* governed by hyperparameters. Poor choices lead to several undesirable outcomes:

- **Sub-optimal predictive performance.** Wrong values for `learning_rate` (`eta`), `n_estimators`, `max_depth` or the regularization terms (`lambda`, `alpha`) can prevent the model from converging to a good solution. The model may **underfit** (fail to capture the underlying pattern) or **overfit** (memorize noise in the training data and generalize poorly). Tuning searches for the configuration that minimizes the chosen loss on validation data — finding the right point in the **bias–variance trade-off**.
- **Increased overfitting risk.** Boosting models overfit easily when trees are deep or there are many rounds. Regularization-related hyperparameters exist specifically to counter this: tree-complexity params (`max_depth`, `min_child_weight`, `min_split_loss`/`gamma`), subsampling params (`subsample`, `colsample_by*`) and explicit L1/L2 penalties on leaf weights all must be tuned carefully for good generalization. Early stopping controlled by a validation metric is itself a form of tuning that directly limits the number of rounds.
- **Inefficient resource use.** Training is computationally expensive, especially on large data. Hyperparameters affect training time and memory: approximate split-finding (histogram methods in LightGBM, `tree_method='hist'` in XGBoost), feature/data subsampling rates, and tree depth all change training speed substantially. Tuning lets you trade predictive performance against compute cost — often a practical necessity.
- **Sensitivity to data characteristics.** The best settings vary with dataset size, number and type of features (dense, sparse, categorical), noise level and objective. A configuration good for one problem may be poor for another.

**Learning rate example.** A high learning rate can converge fast but may overshoot the optimum or oscillate, hurting generalization. A very low learning rate needs many more boosting rounds (`n_estimators`) — increasing training time — but combined with early stopping usually generalizes better. Tuning finds the right value for the specific problem.

Crucially, **hyperparameters interact**: the best tree depth may depend on the learning rate; the effectiveness of subsampling may change with the number of rounds. This coupling means parameters must be tuned *systematically*, not one at a time in isolation.

---

### 8.2 Identifying the Critical Boosting Hyperparameters

Not all hyperparameters have equal impact. Concentrating effort on the highest-impact ones is essential for efficient optimization.

#### Core boosting parameters

**Number of boosting rounds** (`n_estimators`, `num_boost_round`, `iterations`). The total number of sequential trees. More trees generally means more complexity: too few underfits, too many overfits. Best managed *indirectly via early stopping* — set a large ceiling and let the algorithm stop when validation performance stops improving.

**Learning rate** (`learning_rate`, `eta`). Scales the contribution of each new tree added to the ensemble. Smaller values need more rounds to reach the same training-error reduction but usually generalize better. It acts as a form of regularization by shrinking the step size in function space at each iteration. Typical range **0.01 to 0.3**. There is a direct trade-off: **lowering the learning rate requires raising the number of rounds.** Strongly coupled with `n_estimators`.

#### Tree-structure parameters

**Maximum tree depth** (`max_depth`). Limits how deep each tree may grow. Deeper trees capture more complex feature interactions but overfit more easily. Shallow trees (e.g. depth 4–8) usually strike a good balance. Typical range **3 to 10**, highly data-dependent. In LightGBM, `num_leaves` is often a more direct control than `max_depth` because of its leaf-wise growth.

**Minimum child weight / minimum samples per leaf** (`min_child_weight` [XGBoost], `min_sum_hessian_in_leaf` [LightGBM]; `min_data_in_leaf` [LightGBM/CatBoost], `min_samples_leaf` [sklearn GBM]). Sets a minimum threshold on the sum of instance weights (the Hessian for XGBoost/LightGBM) or the number of samples required in a leaf. Prevents the tree from creating splits that isolate very small groups — a regularization mechanism against fitting noise. Larger values give more conservative trees. Interacts with `max_depth`.

**Minimum split gain** (`gamma` [XGBoost], `min_gain_to_split` [LightGBM], `min_impurity_decrease` [sklearn GBM]). The minimum reduction in the loss required to make a split; splits not meeting it are pruned. A direct regularizer on the splitting process — larger values make the algorithm more conservative.

#### Subsampling parameters (introduce randomness → improve generalization and speed)

**Row subsampling** (`subsample`; alias `bagging_fraction` in LightGBM). The fraction of training rows randomly sampled (without replacement) to build each tree. Values below 1.0 add randomness, reduce variance and help prevent overfitting. Typical range **0.5 to 1.0**.

**Column subsampling** (`colsample_bytree`, `colsample_bylevel`, `colsample_bynode` in XGBoost; alias `feature_fraction` in LightGBM). The fraction of features considered when building each tree, each level, or each split. Especially useful with many features, since it stops the model over-relying on a few dominant features. Regularizes (particularly for high-dimensional data) and speeds up computation. Typical range **0.5 to 1.0**.

#### Algorithm-specific parameters

- **XGBoost:** `reg_alpha` (L1 penalty on leaf weights, can yield sparse weights); `reg_lambda` (L2 penalty on leaf weights, default usually 1, generally more impactful than `reg_alpha` for tree models).
- **LightGBM:** `num_leaves` — the max leaves per tree; because LightGBM grows leaf-wise (splitting the leaf with the largest loss reduction), this is the *primary* complexity control and is usually tuned instead of `max_depth`; high values overfit easily. A common constraint is `num_leaves` $\le 2^{\text{max\_depth}}$. `boosting_type` selects `gbdt` (standard), `dart` (adds dropout, sometimes better at the cost of more tuning) or `goss` (gradient-based one-side sampling).
- **CatBoost:** `cat_features` (explicitly flags categorical columns to enable ordered target statistics — required for effective categorical handling); `l2_leaf_reg` (L2 like XGBoost's `reg_lambda`); `border_count` (bins for numerical feature discretization, affects speed/memory); `one_hot_max_size` (use one-hot encoding for low-cardinality categoricals up to this size).

#### Tuning priority

Tuning everything at once is infeasible. A practical priority order (revisit earlier params after tuning later ones, since they interact):

1. **Core learning parameters** — find a good `learning_rate` + `n_estimators` combination (with early stopping).
2. **Tree complexity** — tune `max_depth` (or `num_leaves` for LightGBM) and `min_child_weight` / `min_data_in_leaf`.
3. **Subsampling** — optimize `subsample` and `colsample_by*`.
4. **Explicit regularization** — fine-tune `gamma`, `reg_alpha`, `reg_lambda` / `l2_leaf_reg`.
5. **Algorithm-specific parameters** — adjust library-specific settings if needed.

---

### 8.3 Systematic Tuning: Grid Search vs Random Search

Grid search and random search are the foundational systematic methods — a big step up from ad-hoc manual tweaking.

#### Grid search

The most intuitive approach:

1. **Define the search space** — for each hyperparameter specify a discrete set of values to evaluate (e.g. `learning_rate` $\in$ [0.01, 0.1, 0.2], `max_depth` $\in$ [3, 5, 7], `n_estimators` $\in$ [100, 200]).
2. **Create the grid** — the Cartesian product of all sets, i.e. every possible combination. The example gives $3 \times 3 \times 2 = 18$ unique combinations.
3. **Evaluate each combination** with cross-validation.
4. **Select the best** by mean CV performance on the chosen metric (AUC, LogLoss, RMSE…).

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, random_state=42)

param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth':     [3, 5, 7],
    'n_estimators':  [100, 200],
    'subsample':     [0.7, 0.9],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print(f"Best params: {grid_search.best_params_}")
print(f"Best AUC:    {grid_search.best_score_:.4f}")
best_xgb_model = grid_search.best_estimator_
```

**Pros:** exhaustive within the grid (guaranteed best of the specified points); simple to understand and implement.
**Cons:** **cost grows exponentially** — with $k$ hyperparameters and $m$ values each you train $m^k$ models (times CV folds); and the optimum may fall *between* grid points, so refining resolution multiplies cost further.

#### Random search

Instead of trying every combination, sample a fixed number of settings from specified distributions:

1. **Define search distributions** — for continuous params (`learning_rate`, `subsample`) use uniform or log-uniform; for discrete params (`max_depth`) provide integer ranges.
2. **Set a budget** — the number of combinations to sample (`n_iter`).
3. **Sample and evaluate** each with cross-validation.
4. **Select the best** mean CV performer.

```python
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification
from scipy.stats import uniform, randint

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, random_state=42)

param_dist = {
    'learning_rate':    uniform(0.01, 0.2),   # sample in [0.01, 0.21)
    'max_depth':        randint(3, 10),        # integers 3..9
    'n_estimators':     randint(100, 500),
    'subsample':        uniform(0.6, 0.4),     # [0.6, 1.0)
    'colsample_bytree': uniform(0.5, 0.5),     # [0.5, 1.0)
}

random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist,
                                   n_iter=50, scoring='roc_auc', cv=5, n_jobs=-1,
                                   verbose=1, random_state=42)
random_search.fit(X, y)

print(f"Best params: {random_search.best_params_}")
print(f"Best AUC:    {random_search.best_score_:.4f}")
```

**Why random search is more efficient in high dimensions:** performance is usually driven by only a few of the hyperparameters. A grid wastes evaluations testing many redundant values of the *unimportant* params, whereas random search samples a **diverse set of values for the important params** for the same budget. This is the classic Bergstra & Bengio (2012) result. Random search also handles continuous parameters naturally and lets you cap cost directly with `n_iter`.
**Cons:** no guarantee of finding the absolute best; results vary run-to-run unless `random_state` is fixed; quality depends on choosing a large enough `n_iter`.

#### Practical search-space guidance

- **`learning_rate`** — log-uniform (e.g. 0.001–0.3), because its effect is multiplicative.
- **`n_estimators`** — integer range (e.g. 100–1000), but strongly tied to learning rate and early stopping; if early stopping is used effectively, tuning `n_estimators` directly matters much less.
- **`max_depth`** — integers e.g. 3–10.
- **Subsampling** (`subsample`, `colsample_*`) — uniform e.g. 0.5–1.0.
- **Regularization** (`lambda`, `alpha`) — log-uniform e.g. 1e-3 to 10.
- **Budget** — for random search start around 20–50 iterations and increase while gains continue.

Both `GridSearchCV` and `RandomizedSearchCV` use CV internally (`cv=`) to reduce the risk of overfitting a single train/test split.

---

### 8.4 Advanced Tuning: Bayesian Optimization

Grid/random search are systematic but blind — grid suffers the curse of dimensionality and evaluates many unpromising regions; random search has no strategy to focus where results are better. **Bayesian optimization** takes a more informed approach: find a good configuration in **as few expensive objective evaluations as possible** (here, each evaluation = train + validate a boosting model).

#### Core idea: informed search

Build a **probabilistic model** of the relationship between hyperparameters and performance (validation accuracy/loss). This **surrogate model** is much cheaper to evaluate than the true objective, and it is used to decide intelligently which hyperparameters to try next — **balancing exploration** (probing uncertain regions) **against exploitation** (sampling near the current best).

#### Two components

**1. Probabilistic surrogate model.** Approximates the true objective $f(x)$, where $x$ is a hyperparameter configuration and $f(x)$ is the resulting performance metric (validation AUC, RMSE…). It is built iteratively from past evaluations $\{(x_i, f(x_i))\}$. A common choice is a **Gaussian Process (GP)**, which places a prior over functions and updates it as evaluations arrive. Critically, a GP gives not just a **mean** prediction $\mu(x)$ for untested configurations but also an **uncertainty** estimate $\sigma(x)$ — and that uncertainty is what guides the search.

**2. Acquisition function.** Uses the surrogate's mean and uncertainty to quantify the "utility" of evaluating the objective at a candidate $x$, trading off exploration vs exploitation. Common choices:

- **Expected Improvement (EI)** — the expected amount by which a point improves on the current best observed value $f(x^+)$. With the improvement defined as $I(x) = \max(0,\, f(x) - f(x^+))$ (for maximization), EI is its expectation under the surrogate's predictive distribution:

$$
\mathrm{EI}(x) = \mathbb{E}\!\left[\max\left(0,\; f(x) - f(x^{+})\right)\right]
$$

  It favors points likely to beat the current best, accounting for both predicted mean and uncertainty.
- **Upper Confidence Bound (UCB)** — picks points with a high optimistic bound, explicitly mixing exploitation (high mean) and exploration (high uncertainty):

$$
\mathrm{UCB}(x) = \mu(x) + \kappa\,\sigma(x)
$$

  where $\kappa$ tunes the exploration–exploitation balance.
- **Probability of Improvement (PI)** — the probability that a point beats the current best.

#### The optimization loop

1. **Initialize** — evaluate $f(x)$ at a few initial points, chosen randomly or by a space-filling design (e.g. Latin hypercube sampling).
2. **Fit surrogate** — fit the probabilistic model (e.g. GP) to all observed $\{(x_i, f(x_i))\}$.
3. **Optimize acquisition** — find $x_{\text{next}}$ that maximizes the acquisition function (cheap relative to the true objective).
4. **Evaluate objective** — train and validate the boosting model at $x_{\text{next}}$ (the expensive step).
5. **Augment data** — add $(x_{\text{next}}, f(x_{\text{next}}))$ to the observations.
6. **Repeat** from step 2 until a stopping criterion (max evaluations, or negligible expected improvement).

The recommendation is the best-performing configuration observed.

**TPE (Tree-structured Parzen Estimator).** An alternative surrogate approach used by Hyperopt and by Optuna's default sampler. Rather than modeling $p(y \mid x)$ directly like a GP, TPE models the densities $p(x \mid y)$ — splitting past trials into "good" ($l(x)$) and "bad" ($g(x)$) groups by their objective values and preferring configurations where the ratio $l(x)/g(x)$ is large (this ratio is equivalent to maximizing EI). TPE scales well to higher dimensions and handles conditional/mixed search spaces naturally.

**Pros:** high sample efficiency (far fewer expensive evaluations than grid/random); actively steers toward promising regions; handles continuous, integer and (with care) categorical hyperparameters.
**Considerations:** fitting the surrogate and optimizing the acquisition adds per-iteration overhead — not worth it if each objective evaluation is very fast (seconds), but a clear win when evaluations take minutes/hours (the typical boosting case). It is also inherently **sequential** (each point depends on all previous results), limiting parallelism vs random search, though batch variants exist.

---

### 8.5 HPO Frameworks: Optuna and Hyperopt

Implementing Bayesian optimization by hand — managing trials, pruning, parallelization — is complex. Dedicated frameworks automate most of it so you focus on defining the search space and objective. Two well-known Python frameworks are **Optuna** and **Hyperopt**, both of which implement TPE.

#### Optuna

A modern, actively developed framework praised for its **define-by-run** API — you build the search space *dynamically inside the objective function*.

- **Define-by-run API.** Use `trial.suggest_float`, `trial.suggest_int`, `trial.suggest_categorical` directly inside the objective; this naturally supports **conditional hyperparameters** (one param's choice affecting another's range/availability).
- **Samplers.** Default is TPE (`TPESampler`); also `RandomSampler`, `GridSampler`, `CmaEsSampler`.
- **Pruning.** Integrates with XGBoost/LightGBM/sklearn to monitor intermediate results (e.g. validation score after some rounds) and **stop unpromising trials early**, saving large amounts of compute.
- **Parallelization.** Easy across processes/machines using a shared storage backend (e.g. a relational DB).
- **Visualization.** Built-in optimization history, parameter relationships, and hyperparameter-importance plots.

```python
import optuna
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# In practice use proper CV or a fixed validation set instead of a single split.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
dtrain = lgb.Dataset(X_train, label=y_train)

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1':         trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2':         trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq':      trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss')

    gbm = lgb.train(
        param, dtrain,
        valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
        callbacks=[pruning_callback, lgb.early_stopping(10, verbose=False)],
    )

    preds = gbm.predict(X_valid)
    pred_labels = (preds > 0.5).astype(int)
    accuracy = accuracy_score(y_valid, pred_labels)
    return 1.0 - accuracy   # Optuna minimizes; lower is better

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)

print("Finished trials:", len(study.trials))
print("Best value:", study.best_trial.value)
print("Best params:", study.best_trial.params)
```

#### Hyperopt

A mature framework, best known for its TPE implementation. Its main difference from Optuna is that the search space is defined **up front** as a nested structure using stochastic-expression functions.

- **Search space:** `hp.choice`, `hp.uniform`, `hp.loguniform`, `hp.quniform` etc.
- **Algorithms:** mainly TPE (`tpe.suggest`) and random search (`rand.suggest`).
- **`fmin`:** the core driver — takes the objective, space, algorithm, `max_evals`, and a `Trials` object.
- **`Trials` object:** stores per-trial params, status and results (for analysis/resuming).
- **Parallelization:** e.g. via `SparkTrials` on Apache Spark.

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

space = {
    'max_depth':        hp.quniform('max_depth', 3, 10, 1),   # integer via quniform
    'learning_rate':    hp.loguniform('learning_rate', -5, -1),  # ~0.0067 to 0.36
    'subsample':        hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'gamma':            hp.uniform('gamma', 0.0, 0.5),
    'lambda':           hp.loguniform('lambda', -2, 2),   # L2, e^-2 .. e^2
    'alpha':            hp.loguniform('alpha', -2, 2),    # L1
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 123,
}

def objective(params):
    params['max_depth'] = int(params['max_depth'])   # Hyperopt passes ints as floats
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist,
                      early_stopping_rounds=30, verbose_eval=False)
    preds = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    loss = log_loss(y_valid, preds)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
print("Best parameters:", best)
```

**Choosing:** Optuna's Pythonic define-by-run API, pruning integration, active community and built-in visualizations make it feel more intuitive for most; Hyperopt's explicit up-front space and Spark integration suit some distributed setups. Scikit-Optimize (`skopt`) is another option with an sklearn-compatible API.

---

### 8.6 Coarse-to-Fine Tuning Strategy

Blindly running an automated search over the *full* range of every parameter is expensive and inefficient. A structured multi-stage approach — broad **coarse** search first, then focused **fine** search — allocates compute wisely.

#### Stage 1 — Coarse tuning (broad exploration)

- **Goal:** quickly identify *promising regions* of the space, not the absolute optimum. Sketch the general contour of performance.
- **Method:** **random search** (efficient at finding diverse combinations), or the early phase of Bayesian optimization.
- **Parameters:** focus on the high-impact ones — `learning_rate`, `n_estimators` (better controlled by early stopping — set a large ceiling rather than searching a wide range directly), `max_depth`, sampling params (`subsample`, `colsample_*`), regularization (`lambda`/`alpha`).
- **Ranges:** define **wide** ranges — log scale for `learning_rate` and regularization (e.g. 0.001–0.1), linear for depth and sampling rates.
- **Execution:** limited iterations (e.g. 30–100); a **cheaper validation** scheme (single hold-out or 3-fold CV); optionally a representative **subset of the data** if it is very large.
- **Result:** a coarse sense of which ranges work and which can be excluded (e.g. "learning rate below 0.01 is consistently poor" or "depth above 8 overfits without gain").

#### Stage 2 — Fine tuning (focused optimization)

- **Goal:** precisely locate the best configuration within the promising region found in Stage 1.
- **Method:** **Bayesian optimization** (Optuna/Hyperopt) excels here — its modeling and intelligent point selection are efficient in a restricted space. A focused grid search is also viable if the narrowed space is small enough.
- **Parameters:** the narrowed ranges from Stage 1, plus additional finer-grained params held fixed earlier (e.g. `min_child_weight`, `gamma`).
- **Ranges:** tighter, based on coarse results (e.g. if good rates were 0.01–0.05, use `LogUniform(0.01, 0.05)`).
- **Execution:** larger budget (e.g. 50–200+ iterations); a **more robust** evaluation — full-data k-fold CV (5- or 10-fold); early stopping in every evaluation to set the best `n_estimators` for each tested combination.

#### Integrating early stopping

Because `n_estimators` is tightly coupled with `learning_rate` (lower rate → more trees), it is more efficient **not** to search `n_estimators` over a wide range but to:

1. Set a large ceiling (e.g. 2000).
2. Use the built-in early stopping of XGBoost/LightGBM/CatBoost in every evaluation (within each CV fold).
3. Configure it to monitor a validation metric and stop after N rounds without improvement (e.g. `early_stopping_rounds=50`).

That way the optimal tree count is determined automatically for every combination of the *other* hyperparameters being tested.

**Practical notes:** Optuna fits this well — run a coarse study, analyze it (visualizations), then launch a refined study with a narrowed space. Tuning is not strictly linear: fine results may reveal the coarse search was too narrow, so be ready to revisit assumptions and re-run a broader search.

---

### 8.7 Cross-Validation Strategy for Honest Tuning

Relying on a single train/validation split to evaluate hyperparameters is misleading — the chosen params may **overfit that specific validation set** and generalize poorly. **Cross-validation (CV)** gives a more reliable estimate by evaluating on multiple subsets.

#### CV inside the tuning loop

For **each** hyperparameter combination the search proposes:

1. Split the training data into $K$ folds.
2. Train $K$ times, each on $K-1$ folds and validate on the held-out fold.
3. Compute the metric on each held-out fold.
4. **Aggregate** (usually mean, sometimes with std) the $K$ scores into one stable estimate for that configuration.

The tuning algorithm uses this aggregated score to decide what to try next (Bayesian) or which combination wins (grid).

#### CV strategies

- **Standard K-fold.** Shuffle and split into $K$ equal folds; $K=5$ or $10$ typical. Higher $K$ trains on more data per iteration but costs more.
- **Stratified K-fold** (`StratifiedKFold`). Preserves each class's proportion in every fold — the **recommended default for classification**, especially with imbalanced data, where random folds could distort class distribution and give unreliable estimates.
- **Group K-fold** (`GroupKFold`). When samples are not independent (multiple measurements per patient, images per location, logs per user session), standard K-fold can place the same group in both train and validation, causing **leakage** and over-optimistic estimates. Group K-fold keeps all samples of a group entirely on one side of each split, using an identifier (`patient_id`, `user_id`).
- **Time-series CV.** For temporal data, random shuffling breaks time order and causes **look-ahead bias** (using future to predict past). Preserve order with:
  - **Expanding window / rolling forecast origin** (`TimeSeriesSplit`): train on folds 1..t, validate on t+1; the training window grows over time — simulates periodic retraining as new data arrives.
  - **Sliding window:** fixed-size training window that slides forward. Choose expanding vs sliding by whether older data stays relevant (expanding) or the pattern drifts (sliding).

#### Wiring CV into frameworks

- **Scikit-learn:** pass `cv=5` or a splitter object (`cv=StratifiedKFold(n_splits=5)`, `cv=TimeSeriesSplit(n_splits=5)`) to `GridSearchCV`/`RandomizedSearchCV`.
- **Optuna/Hyperopt:** include the CV loop *inside* the objective function — run K-fold, average the fold scores, and return that average to guide the search.

#### CV with early stopping

Combining early stopping with CV requires care. A common per-fold procedure to evaluate one hyperparameter set:

1. For each of the $K$ folds:
   - Further split the $K-1$ training folds into a sub-train set and an early-stopping validation set.
   - Train on sub-train, using the early-stopping set to find the best number of rounds for that fold.
   - Record the score on the *main held-out fold* (using that best round count), and record the round count used.
2. Average the $K$ fold scores → the score for this hyperparameter set.
3. Optionally average / take the median of the best round counts across folds.

When training the **final** model (after the best hyperparameters are found), fit on the *entire* training set, setting the boosting rounds to the averaged/median count from CV, or determine it with a separate final validation set.

#### Cost and the final model

CV multiplies tuning cost by $K$: 5-fold CV over 100 combinations = **500 model trainings**. To manage it: use fewer folds ($K=3$ or $5$) if too expensive; prefer efficient search (random/Bayesian over exhaustive grid); do a cheap broad search (fewer folds / less data) then a finer search on promising regions. Remember: **CV during tuning is only for evaluating hyperparameter sets.** Once the best set is found, train the final model once on the full training data, and the CV estimate is your expectation of how it will perform on new, unseen data.

---

### 8.8 Hands-On: Advanced Tuning with Optuna (XGBoost)

A full walkthrough tuning an XGBoost classifier on the Wisconsin breast-cancer dataset, maximizing validation **AUC**. Early stopping handles `n_estimators` implicitly.

**Setup.**

```python
# uv add xgboost optuna scikit-learn plotly   (project uses uv, not pip)
import xgboost as xgb
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

**Objective function.** `trial.suggest_*` defines the search space dynamically; `log=True` samples multiplicative-effect params (learning rate, regularization) evenly across orders of magnitude. The `try/except` returns a poor score (0.0) if a parameter combination makes XGBoost error out, so one bad trial does not crash the study.

```python
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'verbosity': 0,
        'nthread': -1,
        'seed': 42,
        # tuned parameters
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),          # rows
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),   # features
        'lambda':           trial.suggest_float('lambda', 1e-8, 10.0, log=True), # L2
        'alpha':            trial.suggest_float('alpha', 1e-8, 10.0, log=True),  # L1
        'gamma':            trial.suggest_float('gamma', 1e-8, 5.0, log=True),   # min split gain
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    evals  = [(dtrain, 'train'), (dval, 'eval')]

    try:
        bst = xgb.train(
            params, dtrain,
            num_boost_round=1000,        # high ceiling; early stopping picks the real count
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        # Tip: persist the best round count for the final model:
        #   trial.set_user_attr('best_iteration', bst.best_iteration)
        preds = bst.predict(dval, iteration_range=(0, bst.best_iteration))
        return roc_auc_score(y_val, preds)   # maximize AUC
    except xgb.core.XGBoostError as e:
        print(f"XGBoostError in trial {trial.number}: {e}")
        return 0.0
```

**Run the study.** Direction is `maximize` because higher AUC is better. More `n_trials` explores more thoroughly at greater cost.

```python
study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')
study.optimize(objective, n_trials=50)   # raise to 100+ for a thorough search
```

**Analyze.**

```python
best_trial = study.best_trial
print(f"Best trial: {best_trial.number}")
print(f"Best AUC:   {best_trial.value:.6f}")
print("Best params:", best_trial.params)

# Visualizations (need plotly):
optuna.visualization.plot_optimization_history(study).show()  # best score vs trial
optuna.visualization.plot_param_importances(study).show()     # which params mattered
```

`plot_optimization_history` shows per-trial AUC plus the running best (rises fast early, then plateaus as Optuna focuses). `plot_param_importances` ranks hyperparameters by influence on AUC (computed via an MDI-based random forest over the trial results); `plot_slice` / `plot_contour` further illuminate individual param–objective relationships.

**Train the final model.** Refit with the best params, using the best iteration count found by early stopping, on the full training data (optionally train+val combined if a separate test set is held out).

```python
best_params = study.best_params
best_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc',
                    'booster': 'gbtree', 'verbosity': 0, 'nthread': -1, 'seed': 42})

# Cleanest: retrieve best_iteration saved via trial.set_user_attr during the study.
# Otherwise, re-run briefly to recover it:
temp_bst = xgb.train(best_params, xgb.DMatrix(X_train, label=y_train),
                     num_boost_round=1000, evals=[(xgb.DMatrix(X_val, label=y_val), 'eval')],
                     early_stopping_rounds=50, verbose_eval=False)
final_num_boost_round = temp_bst.best_iteration

final_model = xgb.train(best_params, xgb.DMatrix(X_train, label=y_train),
                        num_boost_round=final_num_boost_round, verbose_eval=False)
# Evaluate final_model on a separate, unseen test set for an unbiased estimate.
```

> **Self-correction from the source:** the original objective did not save `best_iteration`. The clean fix is `trial.set_user_attr('best_iteration', bst.best_iteration)` inside the objective, then retrieve via `study.best_trial.user_attrs['best_iteration']`. The "re-train briefly" code above is a workaround.

---

### Practical Tuning Recipe / Checklist

An ordered, practical procedure for tuning a boosting model:

1. **Fix the evaluation first.** Choose a CV scheme that matches the data *before* tuning anything: StratifiedKFold for classification, GroupKFold for grouped/non-independent data, TimeSeriesSplit for temporal data. A trustworthy CV is more important than any single model.
2. **Set a sensible baseline.** Train with library defaults + early stopping to get a reference score.
3. **Pin `n_estimators` via early stopping.** Set a high ceiling (e.g. 1000–2000) and use `early_stopping_rounds` in every evaluation; never grid-search the tree count directly.
4. **Coarse search (Stage 1).** Random search or early Bayesian over wide ranges of the high-impact params — `learning_rate` (log), `max_depth`/`num_leaves`, `subsample`, `colsample_*`, regularization (log). Use cheap validation (3-fold or hold-out), 30–100 iterations. Prune/exclude clearly bad regions.
5. **Fine search (Stage 2).** Bayesian optimization (Optuna/Hyperopt TPE) over the narrowed ranges, adding finer params (`min_child_weight`/`min_data_in_leaf`, `gamma`). Larger budget (50–200+), robust 5–10-fold CV, early stopping per fold.
6. **Respect parameter interactions.** After tuning later-priority params, revisit earlier ones (learning rate ↔ depth ↔ subsample all couple); tuning is iterative, not one-pass.
7. **Watch for overfitting the validation set.** Prefer the aggregated CV mean (and check the std across folds) over any single split; be suspicious of a config that wins by a tiny margin.
8. **Refit the final model** on the full training data with the best params and the CV-derived best round count.
9. **Estimate generalization on a truly held-out test set** — never the data used for tuning.

---

### Key takeaways

- Defaults are a *starting point*, not an answer; because boosting is sequential and its hyperparameters interact, tuning is often decisive and must be **systematic**, not one-parameter-at-a-time.
- Prioritize by impact: **`learning_rate` + `n_estimators` together** (rate down ⇒ rounds up, managed by early stopping) → tree complexity (`max_depth`/`num_leaves`, `min_child_weight`/`min_data_in_leaf`) → subsampling (`subsample`, `colsample_*`) → explicit regularization (`reg_lambda`, `reg_alpha`, `gamma`).
- **Random search beats grid search in high dimensions** because only a few params usually matter and random sampling covers those better for the same budget; grid cost is exponential.
- **Bayesian optimization** (GP or TPE surrogate + an acquisition function like Expected Improvement that balances exploration/exploitation) finds good configs in far fewer expensive evaluations — the right tool when each train+validate is slow.
- **Optuna** (define-by-run, pruning, visualizations) and **Hyperopt** (explicit space, TPE, Spark) automate the search loop.
- Use a **coarse-to-fine** strategy to spend compute where it counts, and always evaluate with the **right CV scheme** to avoid leakage and to keep from overfitting a single validation split.
- Always determine `n_estimators` through **early stopping**, and estimate final generalization on a held-out set the tuner never saw.

> **Relevance to our work:** Our automated tree-search harness already embodies the CV-gated tuning philosophy of this chapter — every candidate is scored through a fixed, honest OOF/CV gate rather than a single split, which is exactly how §8.7 says to avoid overfitting the validation set. The three-arm search (baseline / random-ish expansion / LLM-proposed configs) mirrors coarse-to-fine: broad diverse proposals first, then focused refinement of promising regions. Practical carry-overs: (1) keep `n_estimators` controlled by early stopping and store `best_iteration` per fold so the final refit uses a CV-derived round count — matching the §8.8 self-correction note; (2) sample `learning_rate` and the regularization terms on a **log** scale, tree/sampling params linearly; (3) fix determinism (`deterministic`, `force_row_wise`, `num_threads`) so Optuna/TPE trial scores are reproducible across processes — see our LightGBM-determinism memo. Above all, **watch the CV–LB gap**: a hyperparameter set that wins the CV gate by a razor-thin margin is a prime candidate for private-leaderboard regression, so prefer robust configs (low fold-to-fold std) over the single highest-mean trial, and never trust a public-LB bump that the CV gate did not corroborate.

---

## Chapter 9 — Gradient Boosting for Specialized Tasks

> Source: "Mastering Gradient Boosting Algorithms" (apxml.com), Chapter 9. Synthesized English study notes reconstructed from the scraped course text, with the KaTeX-mangled math cleaned into correct LaTeX. Covers four families of non-standard boosting tasks — learning to rank, survival analysis, quantile regression, and multi-output — plus a hands-on XGBoost ranking walkthrough.

Standard ML tasks are classification (assign a label) or regression (predict a continuous value). This chapter covers boosting objectives that fall outside that mold: ordering items (ranking), predicting time-to-event under censoring (survival), predicting distribution quantiles rather than the mean (quantile regression), and predicting several targets jointly (multi-output). The unifying theme is that gradient boosting only needs a differentiable (or approximately differentiable) loss with a gradient and Hessian — so any task expressible that way can reuse the same additive-tree machinery.

### 9.1 Learning to Rank (LTR) with Gradient Boosting

**The task.** Learning to Rank orders a *set* of items by their relevance/preference/importance relative to a specific **query or context** — the core problem in search engine result pages, product recommendation, and document retrieval. Unlike regression or classification, the goal is the *relative order* of items within a query, not an absolute score or label.

**Why GBMs fit LTR.** Tree ensembles model complex non-linear feature interactions that determine relevance, and the boosting framework gives a principled way to optimize ranking quality *even when the evaluation metric itself is non-differentiable*.

**The metric difficulty.** Ranking metrics — Normalized Discounted Cumulative Gain (**NDCG**), Mean Average Precision (**MAP**), Mean Reciprocal Rank (**MRR**) — score the whole ordered list and depend on the *relative order* of items. They are piecewise-constant / non-differentiable, so standard gradient descent cannot optimize them directly. Example: NDCG@k judges the top $k$ items by relevance grade and position; swapping two items far down the list may not change NDCG@k at all, giving a zero gradient even though other swaps would improve the ranking.

**Three approaches.** GBMs optimize surrogate losses that *implicitly* improve the target ranking metric:

- **Pointwise.** Simplest: treat each item independently, reframing LTR as regression (predict a relevance score) or classification (predict a relevance class). Final ranking = sort items by predicted score. Uses standard losses (MSE, LogLoss). *Limitation:* ignores relative order entirely during training — never explicitly learns "for query $q$, item A should outrank item B" — so it is usually sub-optimal for ranking.
- **Pairwise.** The most common and successful GBM approach. Instead of scoring items alone, it looks at *pairs* of items within the same query. Goal: learn a scoring function $f(q,d)$ such that if $d_i$ is more relevant than $d_j$ for query $q$, then $f(q,d_i) > f(q,d_j)$. Training pairs $(d_i, d_j)$ are built per query; the model optimizes a loss on the score difference $f(q,d_i) - f(q,d_j)$, commonly a logistic loss that maximizes the margin between correctly ordered pairs.
- **Listwise.** Directly optimizes a loss defined over the *entire* item list for a query, capturing interdependencies among all items and approximating the target metric directly. Examples: ListNet, ListMLE. Theoretically appealing but more computationally expensive and complex. In practice a well-tuned LambdaMART (technically pairwise gradients, but listwise in behavior) is a strong baseline.

**LambdaRank / LambdaMART intuition.** LambdaRank does not use the standard pairwise-loss gradient; it computes a **"lambda gradient"** that scales each pair's gradient by the change $|\Delta\text{metric}_{ij}|$ in the target ranking metric (e.g. NDCG) that would result from swapping $d_i$ and $d_j$. This injects information about global list structure and the specific metric into a pairwise optimization: pairs whose swap moves the metric a lot (e.g. lifting a highly relevant item above an irrelevant one near the top) receive larger gradients. **LambdaMART** = LambdaRank gradients + MART (Multiple Additive Regression Trees, i.e. gradient boosting). Most modern GBM LTR implementations (XGBoost, LightGBM) are based on LambdaMART.

**Practical implementation.**
- **Data preparation:** LTR needs data grouped by query. Each row = features of a query–item pair + a relevance label + a query identifier (**QID / group**). The library needs the QID to know which items belong to the same list; input is usually sorted by QID.
- **Objectives:** XGBoost — `rank:pairwise` (LambdaRank principle), `rank:ndcg`, `rank:map`. LightGBM — `lambdarank` (efficient LambdaMART; needs a `group` array giving each query's document count). CatBoost — `PairLogit`, `YetiRank` (ordered-boosting listwise method); also needs grouping.
- **Evaluation:** use ranking metrics (`ndcg@k`, `map@k`). Cross-validation must respect query groups (**Group K-Fold**) to avoid leaking documents of one query across folds.

### 9.2 Ranking Objective Functions (Pairwise and Listwise)

Standard regression (MSE) or classification (LogLoss) losses are unsuitable because they score predictions independently and ignore relative order. Ranking objectives split into pairwise and listwise families.

**Pairwise loss.** Turn ranking into pairwise classification. For a query $q$ with documents $d_i, d_j$ and relevance labels $y_i, y_j$ (higher = more relevant), if $y_i > y_j$ the model should give $f(d_i) > f(d_j)$. A common choice is a logistic loss over pairs with $y_i \neq y_j$:

$$\mathcal{L}_{\text{pair}} = \sum_{q} \sum_{y_i > y_j} \log\!\left(1 + e^{-\sigma\left(f(d_i) - f(d_j)\right)}\right)$$

where $f(d_k)$ is the model's score for document $d_k$ and $\sigma$ is a scaling factor (often 1). It penalizes pairs where the less-relevant document scores $\geq$ the more-relevant one, encouraging higher scores for more-relevant items.

**LambdaRank gradient.** Minimizing the pairwise logistic loss does not directly optimize NDCG or MAP. LambdaRank instead defines a surrogate gradient ("lambda value"). For $d_i$ vs $d_j$ with $y_i > y_j$:

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma\left(f(d_i) - f(d_j)\right)}}\,\bigl|\Delta\text{metric}_{ij}\bigr|$$

Here $\frac{-\sigma}{1 + e^{\sigma(f(d_i)-f(d_j))}}$ is the logistic-loss gradient for the pair, and $|\Delta\text{metric}_{ij}|$ is the absolute change in the ranking metric (e.g. NDCG) if $d_i$ and $d_j$ swap positions. The total lambda gradient for document $d_i$ sums over all relevant pairs:

$$\lambda_i = \sum_{j:\, y_i > y_j} \lambda_{ij} \;-\; \sum_{j:\, y_j > y_i} \lambda_{ji}$$

Each new tree is trained to predict these lambda gradients, pushing the model toward configurations that improve the target metric. `rank:pairwise` in XGBoost/LightGBM uses this LambdaRank-inspired weighting.

*Pairwise pros:* simpler to implement/understand, fast, good in practice. *Cons:* does not explicitly model global list structure; relates to the list metric only indirectly; may under-optimize metrics sensitive to absolute top-of-list position.

**Listwise loss.** Directly optimizes a loss over the entire document list, aiming for the score-sorted permutation to match the ideal permutation from the true labels.
- *Direct metric optimization:* attempts differentiable approximations of NDCG/MAP — hard because the metrics are non-differentiable/discontinuous.
- *Probabilistic methods (ListNet, ListMLE):* define a probability distribution over permutations from predicted scores and minimize divergence (e.g. KL) from the label-defined distribution; ListNet uses the Plackett–Luce model.
- *LambdaMART as listwise:* though its gradients $\lambda_i$ come from pairwise comparisons, the *objective being optimized* is listwise (NDCG/MAP). `rank:ndcg` and `rank:map` in XGBoost/LightGBM are LambdaMART-based — you name the list metric and the algorithm uses the corresponding lambda gradients.

*Listwise pros:* better aligned with NDCG/MAP; considers whole-list relationships. *Cons:* costlier, more complex, gains over pairwise not guaranteed (dataset-dependent).

**Choosing.** `rank:pairwise` is a strong, faster baseline, especially when relative order of relevant vs irrelevant matters more than exact top-k position. `rank:ndcg` / `rank:map` are preferred when a specific list metric is the primary criterion and top-of-list quality matters. Try both; the choice depends on data, metric, and compute.

### 9.3 Gradient Boosting for Survival Analysis

**The task.** Survival analysis (time-to-event analysis) predicts *when* a target event occurs — patient survival, component failure, customer churn. The defining difficulty is **censoring**.

**Censoring.** We often do not observe the event for every subject within the study window. Most common is **right-censoring**: the study ends before the subject's event, or the subject drops out / is lost to follow-up. We know the subject survived *at least* to the censoring time but not the actual event time afterward. Standard regression (predict event time) or classification (predict whether it happened) mishandle this; ignoring censoring or treating censoring times as event times biases results. Encoding: observed time $T_{obs}$ and event indicator $\delta$ ($\delta=1$ event, $\delta=0$ censored).

**Survival and hazard functions.** Rather than predicting event time $T$ directly, describe its distribution:

$$S(t) = P(T > t) \qquad \text{(survival function)}$$

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t} \qquad \text{(hazard rate)}$$

related by $S(t) = \exp\!\left(-\int_0^t h(u)\,du\right)$. The hazard is the instantaneous event risk at time $t$ given survival up to $t$.

**Cox proportional hazards for boosting.** Boosting models the (log) hazard rather than the event time directly. The Cox PH model assumes a multiplicative hazard:

$$h(t \mid X_i) = h_0(t)\,\exp(\eta_i), \qquad \eta_i = F(X_i)$$

where $h_0(t)$ is an arbitrary non-negative **baseline hazard** shared by all individuals, and $\eta_i = F(X_i)$ is the log-risk score (linear $\beta^\top X_i$ in the classic model; a boosted ensemble here, allowing non-linear $\eta$). The key **proportional-hazards assumption**: the hazard ratio between any two individuals is constant over time; covariates act multiplicatively through $\exp(\eta_i)$.

The boosting objective is the **negative log Cox partial likelihood**, which handles censored data *without* estimating $h_0(t)$. Let $t_1 < \dots < t_D$ be the distinct event times, $\mathcal{D}_j$ the set with an event at $t_j$, and $\mathcal{R}_j$ the **risk set** (still under observation just before $t_j$):

$$L = \prod_{j=1}^{D} \frac{\prod_{i \in \mathcal{D}_j} \exp(\eta_i)}{\left(\sum_{k \in \mathcal{R}_j} \exp(\eta_k)\right)^{|\mathcal{D}_j|}}$$

$$\text{Loss} = -\log L = -\sum_{j=1}^{D} \left(\sum_{i \in \mathcal{D}_j} \eta_i - |\mathcal{D}_j|\,\log\!\sum_{k \in \mathcal{R}_j} \exp(\eta_k)\right)$$

One can compute this loss's first and second derivatives (gradient and Hessian) w.r.t. $\eta_i = F(X_i)$; these feed the standard boosting split-finding and leaf-value steps. In XGBoost, `objective='survival:cox'` selects the negative log partial likelihood and computes these internally.

**Evaluation.** Accuracy/RMSE do not apply. Use censoring-aware metrics: **Concordance index (C-index)** — the fraction of comparable subject pairs whose risk ordering matches event ordering, analogous to AUC — plus time-dependent AUC and the Brier score.

**Output.** The model outputs the log-hazard ratio $\eta = F(X)$ per subject (higher = higher predicted risk). These rank individuals by risk; estimating $S(t)$ needs an extra baseline-hazard estimation step.

**Caveats.** Cox-based objectives still assume proportional hazards (test with Schoenfeld residuals; consider Accelerated Failure Time models otherwise). Interpretation gives risk scores, not direct times/probabilities. Tree structure captures non-linearities/interactions; SHAP still works; standard regularization (shrinkage, subsampling, tree constraints) guards against overfitting.

### 9.4 Survival Objective Functions (Cox Proportional Hazards)

A closer look at the Cox PH objective used to train time-to-event boosters. Cox PH is a semi-parametric model of the hazard for covariates $X$:

$$h(t \mid X) = h_0(t)\,\exp(X\beta)$$

- $h_0(t)$: baseline hazard (all covariates zero), left unspecified in the standard model.
- $\beta$: coefficient vector; $\exp(X\beta)$ is the **hazard ratio**. E.g. $\exp(\beta_k)=2$ means a one-unit increase in $X_k$ doubles the hazard at any $t$, holding others fixed.

Cox estimates $\beta$ *without* specifying $h_0(t)$ by maximizing a partial likelihood that compares an individual's risk at their event time against everyone still at risk then.

**Adapting to boosting.** The boosted $F(X)$ plays the role of the linear predictor $X\beta$: $h(t\mid X) = h_0(t)\exp(F(X))$. For each individual $i$ we have observed time $T_i$, event indicator $\delta_i$, covariates $X_i$. The risk set $R_i$ at event time $t_i$ contains all $j$ with $T_j \ge t_i$. The partial likelihood and its negative log (the boosting objective) are:

$$L = \prod_{i:\, \delta_i = 1} \frac{\exp(F(X_i))}{\sum_{j \in R_i} \exp(F(X_j))}$$

$$\text{Obj} = -\log L = -\sum_{i:\, \delta_i = 1} \left[F(X_i) - \log\!\sum_{j \in R_i} \exp(F(X_j))\right]$$

**Gradient and Hessian.** Optimization needs first/second derivatives of the negative log partial likelihood w.r.t. each $F(X_k)$. The derivative for observation $k$ involves summing the term $\dfrac{\exp(F(X_k))}{\sum_{j \in R_i} \exp(F(X_j))}$ over all event times $t_i$ whose risk set $R_i$ contains $k$. XGBoost/LightGBM implement these internally; trees are then fit to the negative gradient.

**Library implementation.**
- **XGBoost:** `objective='survival:cox'`. Labels usually encoded as the event time (positive) and censoring time (negative), e.g. event time $t$, censored time $-t$ — check your version's docs. `predict(X)` returns the log relative risk $F(X)$.
- **LightGBM:** `objective='coxph'`; typically two label columns (time, and 0/1 event indicator).
- **CatBoost:** `loss_function='Cox'`; time column + event indicator column.

**Caveats.** Boosting does not remove the need to check proportional hazards (Schoenfeld residuals). **Tied event times** need Breslow or Efron approximations to the partial likelihood — know your library's default. Evaluate with C-index or time-dependent AUC, not accuracy/AUC; many libraries accept these for monitoring / early stopping.

### 9.5 Quantile Regression with Gradient Boosting

**Why.** Standard regression (squared-error loss) estimates the conditional *mean* $E[Y\mid X]$, only part of the picture. Modeling conditional **quantiles** gives the whole conditional distribution or its tails — the 95th percentile of loss in risk management, or the 10th–90th percentile band for resource planning. The 0.5 quantile is the median, 0.25 the first quartile, 0.9 the 90th percentile.

**Quantile (pinball) loss.** For quantile level $\alpha \in (0,1)$:

$$L_\alpha(y, \hat{y}) = \begin{cases} \alpha\,(y - \hat{y}) & \text{if } y - \hat{y} > 0 \\ (1-\alpha)\,(\hat{y} - y) & \text{if } y - \hat{y} \le 0 \end{cases}$$

Asymmetric penalty: an **underestimate** ($y > \hat{y}$) costs $\alpha\,|y - \hat{y}|$; an **overestimate** ($y \le \hat{y}$) costs $(1-\alpha)\,|y - \hat{y}|$.
- $\alpha = 0.5$: penalty $\tfrac{1}{2}|y-\hat{y}|$ — proportional to MAE, so median regression is robust to outliers.
- $\alpha = 0.9$: underestimation weighted 0.9 vs overestimation 0.1, pushing predictions up (upper quantile).
- $\alpha = 0.1$: overestimation penalized more, pushing predictions down (lower quantile).

**Boosting with it.** Just swap in $L_\alpha$ for squared error. Its negative gradient (the pseudo-residual) is a constant depending only on the *sign* of the error:

$$-\frac{\partial L_\alpha(y,\hat{y})}{\partial \hat{y}} = \begin{cases} \alpha & \text{if } y - \hat{y} > 0 \\ -(1-\alpha) & \text{if } y - \hat{y} \le 0 \end{cases}$$

At iteration $m$, tree $h_m(x)$ is fit to the pseudo-residuals computed from the current ensemble $F_{m-1}(x)$:

$$r_{im} = \begin{cases} \alpha & \text{if } y_i > F_{m-1}(x_i) \\ -(1-\alpha) & \text{if } y_i \le F_{m-1}(x_i) \end{cases}$$

The tree partitions the space and fits these constants so future predictions align with the desired quantile.

**Library support.**

```python
from sklearn.ensemble import GradientBoostingRegressor

# 90th percentile
gbr_q90 = GradientBoostingRegressor(loss='quantile', alpha=0.90, n_estimators=100)
# 10th percentile
gbr_q10 = GradientBoostingRegressor(loss='quantile', alpha=0.10, n_estimators=100)
# Median (50th percentile)
gbr_median = GradientBoostingRegressor(loss='quantile', alpha=0.50, n_estimators=100)
```

```python
import lightgbm as lgb
# 75th percentile
lgbm_q75 = lgb.LGBMRegressor(objective='quantile', alpha=0.75, n_estimators=100)
```

```python
from catboost import CatBoostRegressor
# 20th percentile
cat_q20 = CatBoostRegressor(loss_function='Quantile:alpha=0.2', iterations=100)
```

XGBoost supports quantile regression via `objective='reg:quantileerror'`, but recent versions optimize it with an internal approximation and a custom objective may be needed for exact pinball loss — check version docs.

**Prediction intervals.** To predict several quantiles you generally train a **separate model per quantile** $\alpha$. E.g. forecasting electricity demand: fit Q10, Q50, Q90 independently; the Q10–Q90 gap is an 80% prediction interval that lets a grid operator plan for low- and high-demand scenarios.

**Pros:** full conditional distribution; robustness (median regression like MAE); flexibility of boosting. **Caveats:** $k$ quantiles = $k$ models (compute + management cost); independently trained quantiles can **cross** (predicted Q90 < Q80 for some rows — theoretically inconsistent; fixes exist but add complexity); each quantile may need separate hyperparameter tuning.

### 9.6 Implementing the Quantile Loss (Custom Objective)

The pinball loss can be written compactly with an indicator:

$$L_\alpha(y, \hat{y}) = (y - \hat{y})\bigl(\alpha - I(y - \hat{y} < 0)\bigr)$$

**Gradient:**

$$g = \frac{\partial L_\alpha(y,\hat{y})}{\partial \hat{y}} = I(y - \hat{y} < 0) - \alpha = I(\hat{y} > y) - \alpha = \begin{cases} -\alpha & \text{if } \hat{y} \le y \ \text{(under/exact)} \\ 1 - \alpha & \text{if } \hat{y} > y \ \text{(over)} \end{cases}$$

**Hessian problem.** The gradient is a step function, so its derivative is 0 everywhere except at $\hat{y} = y$, where it is undefined (a Dirac delta). Second-order implementations like XGBoost need a well-defined Hessian. Options:
- **Built-in objective:** libraries like LightGBM (`objective='quantile'`) handle the Hessian internally with approximations — preferred when available.
- **Approximate Hessian:** in a custom objective, return a small positive constant for the Hessian (e.g. `1.0`, or `1e-6`). Not the true second derivative, but it gives numerical stability so the algorithm's $g/h$ leaf-value update still works. Works well in practice.
- **Zero Hessian:** some frameworks tolerate it, effectively falling back to first-order steps, but a small positive value is usually more effective where second-order info is expected.

**Custom objective for XGBoost / LightGBM** — returns per-sample gradient and Hessian:

```python
import numpy as np

def quantile_objective(alpha):
    """Custom objective for quantile regression.

    alpha (float): target quantile in (0, 1).
    Returns a callable compatible with XGBoost/LightGBM custom objectives.
    """
    def objective_function(preds, dtrain):
        labels = dtrain.get_label()
        errors = preds - labels  # preds - labels matches the I(preds > y) convention

        # Gradient
        grad = np.where(errors > 0, 1 - alpha, -alpha)

        # Hessian: small positive constant for numerical stability
        hess = np.full_like(preds, 1.0)  # or a smaller value such as 1e-6

        return grad, hess
    return objective_function

# Example usage
# custom_obj = quantile_objective(0.75)   # target 75th percentile
# XGBoost:  model = xgb.train(params, dtrain, num_boost_round=100, obj=custom_obj)
# LightGBM: model = lgb.train(params, dtrain, num_boost_round=100, fobj=custom_obj)
```

**Notes.** Prefer built-in objectives when they exist; if custom, experiment with the Hessian constant (1.0, 0.1, 1e-3, 1e-6) as it affects convergence/stability. Treat $\alpha$ as a hyperparameter; train one model per quantile. Evaluate with the pinball loss itself on validation, or the Winkler score for prediction intervals — not RMSE.

### 9.7 Multi-Output Gradient Boosting

**The task.** Predict several targets from the same inputs: multi-output regression (e.g. temperature, humidity, pressure from sensor readings) or multi-output classification (e.g. contains-cat / contains-dog / outdoors). Standard boosters (XGBoost, LightGBM, CatBoost, sklearn `GradientBoosting*`) optimize a single 1-D target $y$ and cannot directly consume a target *matrix* $Y$.

**Approach 1 — Independent models (one per target).** Train $k$ separate single-output models, model $j$ mapping $X \to Y_j$.
- *Pros:* simple, no library changes; per-output hyperparameter tuning possible.
- *Cons:* ignores correlations between targets (joint modeling could help if outputs are correlated); training/predicting $k$ models is costly for large $k$.

```python
import xgboost as xgb
import numpy as np

# X_train, Y_train (shape: n_samples, n_outputs); X_test
k_outputs = Y_train.shape[1]
models = []
Y_pred_test = np.zeros((X_test.shape[0], k_outputs))

for i in range(k_outputs):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42 + i)
    model.fit(X_train, Y_train[:, i])          # train on the i-th target column
    models.append(model)
    Y_pred_test[:, i] = model.predict(X_test)
# Y_pred_test now holds predictions for all k outputs
```

**Approach 2 — Scikit-learn multi-output wrapper.** `MultiOutputRegressor` / `MultiOutputClassifier` clone the base estimator and fit one per target — internally identical to Approach 1, but tidier and pipeline/CV-friendly. Wrap any sklearn-compatible estimator (XGBoost/LightGBM/CatBoost sklearn APIs).

```python
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

lgbm = lgb.LGBMRegressor(objective='regression_l1', n_estimators=100, random_state=42)
multi_output_model = MultiOutputRegressor(estimator=lgbm, n_jobs=-1)
multi_output_model.fit(X_train, Y_train)       # fits one LGBMRegressor per column of Y_train
Y_pred_test = multi_output_model.predict(X_test)  # shape (n_samples, n_outputs)
# multi_output_model.estimators_  # access individual fitted estimators
```

*Pros:* convenient, integrates with the sklearn ecosystem. *Cons:* same as Approach 1 — independent models, ignores target correlation, high compute.

**Chaining (regressor/classifier chains).** A related idea: feed each model's prediction as an extra feature to the next, so later targets can use earlier ones — captures some dependency at the cost of an imposed target ordering. (sklearn `RegressorChain` / `ClassifierChain`; the course focuses on the independent and wrapper strategies.)

**Approach 3 — Native multi-output (advanced/research).** Modify the algorithm itself: a **multi-output loss** whose gradient covers all outputs, and a **multi-target split criterion** choosing splits by combined error reduction across outputs (potentially capturing correlations). Standard XGBoost/LightGBM/CatBoost do not ship a general native multi-output mode; it would need heavy C++/CUDA customization or a specialized library. For most practitioners, Approaches 1–2 are standard.

**Choosing.** Use the wrapper (Approach 2) for convenience and pipeline integration when outputs are manageable and you don't need per-output tuning. Use manual models (Approach 1) for fine control over each output's training or when not using the sklearn API. If $k$ is very large, watch compute — consider dimensionality reduction on the output space or models purpose-built for multi-label/multi-output tasks.

### 9.8 Hands-On: Implementing Ranking with XGBoost

Goal: train a model that, given a query, scores documents so that sorting by score approximates the true relevance ordering — using `rank:pairwise`, which reduces mis-ordered document pairs within each query group.

**Data structure for LTR:** per-document features; a graded relevance label (e.g. 0 = irrelevant, 1 = somewhat, 2 = highly relevant); and a query id (`qid`) grouping documents from the same query. The ranking objective operates *within* groups.

**Simulate a dataset:**

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # demo only

np.random.seed(42)
n_queries = 10
n_docs_per_query = 15
n_features = 5

X = np.random.rand(n_queries * n_docs_per_query, n_features)
qids = np.repeat(np.arange(n_queries), n_docs_per_query)

# Relevance driven by feature_0 plus noise
base_relevance = X[:, 0] * 2 + np.random.randn(X.shape[0]) * 0.5

# Discretize into graded labels (0/1/2) by within-query quantiles
y = np.zeros_like(base_relevance, dtype=int)
for qid in range(n_queries):
    query_mask = (qids == qid)
    query_relevance = base_relevance[query_mask]
    q_75 = np.percentile(query_relevance, 75)
    q_25 = np.percentile(query_relevance, 25)
    y[query_mask & (query_relevance >= q_75)] = 2  # highly relevant
    y[query_mask & (query_relevance >= q_25) & (query_relevance < q_75)] = 1  # somewhat
    # rest stay 0 (irrelevant)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['qid'] = qids
df['relevance'] = y
```

**Prepare data — group-aware split + group sizes.** XGBoost's ranking objective needs each query group's size, and the split must keep all docs of a `qid` on the same side (`GroupKFold`). Crucially, sort by `qid` so groups are contiguous, then compute group sizes.

```python
gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(df, groups=df['qid']))

X_train, X_test = df.iloc[train_idx].drop(['qid', 'relevance'], axis=1), df.iloc[test_idx].drop(['qid', 'relevance'], axis=1)
y_train, y_test = df.iloc[train_idx]['relevance'], df.iloc[test_idx]['relevance']
qids_train, qids_test = df.iloc[train_idx]['qid'], df.iloc[test_idx]['qid']

# Scale using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sort by qid so groups are contiguous, then count group sizes
train_order = np.argsort(qids_train.values)
X_train_scaled = X_train_scaled[train_order]
y_train = y_train.iloc[train_order]
qids_train = qids_train.iloc[train_order]
group_train = qids_train.value_counts().sort_index().values

test_order = np.argsort(qids_test.values)
X_test_scaled = X_test_scaled[test_order]
y_test = y_test.iloc[test_order]
qids_test = qids_test.iloc[test_order]
group_test = qids_test.value_counts().sort_index().values

# DMatrix + group info (the essential LTR step)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtrain.set_group(group_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)
dtest.set_group(group_test)
```

`set_group` tells XGBoost how many documents belong to each successive query.

**Train:**

```python
params = {
    'objective': 'rank:pairwise',          # pairwise ranking objective
    'eval_metric': ['ndcg@5', 'ndcg@10'],  # NDCG at cutoffs 5 and 10
    'eta': 0.1,                            # learning rate
    'gamma': 1.0,                          # min loss reduction to split
    'min_child_weight': 1,
    'max_depth': 4,
    'seed': 42,
}

evals = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(params, dtrain, num_boost_round=100, evals=evals, verbose_eval=20)
```

Monitor `ndcg@5-test` / `ndcg@10-test` for generalization.

**Predict and evaluate.** The model scores each document (higher = more relevant); within each query, sort by score and compute NDCG@k. The graded gain uses $2^{\text{rel}} - 1$ with a $\log_2$ position discount:

```python
y_pred_scores = bst.predict(dtest)

def calculate_ndcg_at_k(y_true, y_pred_scores, groups, k):
    """Mean NDCG@k over all query groups."""
    ndcg_scores = []
    start_idx = 0
    for group_size in groups:
        end_idx = start_idx + group_size
        group_y_true = y_true[start_idx:end_idx]
        group_y_pred = y_pred_scores[start_idx:end_idx]

        # Sort docs by predicted score (descending) -> DCG@k
        sorted_indices = np.argsort(group_y_pred)[::-1]
        sorted_y_true = group_y_true[sorted_indices]
        actual_k = min(k, group_size)
        dcg = np.sum((2**sorted_y_true[:actual_k] - 1) / np.log2(np.arange(2, actual_k + 2)))

        # Ideal DCG@k (sort by true relevance)
        ideal_sorted_y_true = np.sort(group_y_true)[::-1]
        idcg = np.sum((2**ideal_sorted_y_true[:actual_k] - 1) / np.log2(np.arange(2, actual_k + 2)))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
        start_idx = end_idx
    return np.mean(ndcg_scores)

y_test_ordered = y_test.values
ndcg_at_5 = calculate_ndcg_at_k(y_test_ordered, y_pred_scores, group_test, k=5)
ndcg_at_10 = calculate_ndcg_at_k(y_test_ordered, y_pred_scores, group_test, k=10)
```

The manual NDCG should closely match XGBoost's internally reported final NDCG, confirming the metric implementation.

**Feature importance** still applies to ranking models:

```python
importance = bst.get_score(importance_type='gain')  # 'gain', 'weight', 'cover'
sorted_importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)
# xgb.plot_importance(bst, importance_type='gain', max_num_features=10)
```

Key takeaways from the practice: LTR requires query grouping (`set_group`), a ranking objective (`rank:pairwise`, or try `rank:ndcg` / `rank:map`), group-respecting splits (`GroupKFold`), and ranking-specific evaluation (NDCG).

### Key takeaways

- Gradient boosting extends beyond mean regression/classification to any task with a differentiable-enough loss (gradient + Hessian) — ranking, survival, quantiles, multi-output all reuse the same additive-tree engine.
- **Ranking:** metrics (NDCG/MAP/MRR) are non-differentiable, so GBMs optimize pointwise/pairwise/listwise surrogates. LambdaMART (pairwise logistic gradient scaled by $|\Delta\text{NDCG}|$) is the workhorse; needs query grouping and Group K-Fold. XGBoost `rank:pairwise|ndcg|map`, LightGBM `lambdarank`.
- **Survival:** handle right-censoring via the Cox negative log partial likelihood; $h(t\mid x)=h_0(t)e^{F(x)}$; output is a log-risk score; evaluate with C-index. XGBoost `survival:cox` (signed-time labels), LightGBM `coxph`, CatBoost `Cox`. Also `survival:aft` (accelerated failure time) exists as an alternative.
- **Quantile:** pinball loss $L_\tau(y,\hat y)=\max(\tau(y-\hat y),(\tau-1)(y-\hat y))$ has a sign-only gradient and a degenerate (zero/undefined) Hessian — built-ins approximate it, custom objectives return a small positive constant Hessian. Fit one model per quantile; Q10/Q90 give an 80% prediction interval. Watch for quantile crossing.
- **Multi-output:** default to independent models per target (manual loop or `MultiOutputRegressor`/`Chain`); native multi-output splitting is research-grade and not in mainstream libraries.

> **Relevance to our work:** For prediction-interval competitions (or any metric rewarding calibrated uncertainty, e.g. pinball / Winkler scoring), train separate LightGBM/XGBoost quantile models at the required $\alpha$ levels rather than a single mean model — Q_low/Q_high directly produce the interval, and the median quantile is a robust point estimate.
> **Relevance to our work:** For recommendation-style or search-ranking competitions where the target is an ordering within groups (sessions, users, queries), use `rank:pairwise` / `lambdarank` with correct `qid`/`group` construction and **Group K-Fold** CV — never a random split that leaks documents of a group across folds — and evaluate with the competition's NDCG/MAP@k rather than regression error.
