# Chapter 1 — Gradient Boosting Foundations Revisited

> Synthesized study notes from the course *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 1. These notes reconstruct the technical content and clean up the KaTeX-mangled math from the source lessons; they are for study reference, not a verbatim transcript.

---

## 1.1 Ensemble Methods: A Recap

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

## 1.2 Decision Trees as Base Learners

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

## 1.3 The Additive Modeling Framework

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

## 1.4 Gradient Descent Fundamentals

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

## 1.5 Introducing the Gradient Boosting Machine (GBM)

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

## Key takeaways

- **Ensembles** trade off bias and variance: bagging (parallel, variance-reducing, e.g. Random Forest) vs. boosting (sequential, bias-reducing) vs. stacking (meta-model over diverse learners, needs out-of-fold predictions).
- **Boosting is sequential and error-correcting**; the additive recursion is $F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$, with final model $F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta\, h_m(x)$.
- **Shallow CART trees are the standard weak learner** — kept intentionally weak (small depth, leaf/split constraints) so boosting improves gradually and avoids fitting noise; this is opposite to Random Forest's deep trees.
- **Trees fit pseudo-residuals, not the raw target.** For squared-error loss the pseudo-residual equals the ordinary residual $y_i - F_{m-1}(x_i)$; in general it is the negative gradient of the loss w.r.t. the current prediction.
- **Gradient descent supplies the correction direction**; the learning rate $\eta$ is both the step size and the primary regularizer (shrinkage).
- **Gradient boosting = functional gradient descent** — each tree approximates $r_{im} = -\left[\partial L / \partial F(x_i)\right]_{F=F_{m-1}}$.
- **Friedman's GBM** = initialize with a loss-minimizing constant → (compute pseudo-residuals → fit tree → line-search step size → update) repeated $M$ times.
- **The choice of loss function drives everything** downstream (pseudo-residuals, which errors get prioritized); GBM's limitations in efficiency, regularization, and categorical/data handling motivate XGBoost, LightGBM, and CatBoost.
