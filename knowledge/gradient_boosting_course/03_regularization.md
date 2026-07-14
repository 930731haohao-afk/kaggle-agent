# Chapter 3 — Regularization in Gradient Boosting

> Source: *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 3 "Regularization in Gradient Boosting". Synthesized study notes reconstructing the KaTeX-mangled math into clean LaTeX and translating the lessons into English. Covers why boosting overfits and the full toolbox for controlling variance: tree-structure constraints, shrinkage, stochastic subsampling, the regularized objective with L1/L2 penalties on leaf weights, and early stopping, closing with a hands-on scikit-learn walkthrough.

## 3.1 Overfitting Challenges in Boosting

Gradient boosting builds models **sequentially**: each new base learner (usually a decision tree) $h_m(x)$ is fit to the pseudo-residuals / negative gradient of the loss of the current ensemble. This iterative error-correction is what makes boosting powerful, but it is also exactly what makes it prone to overfitting.

- The algorithm relentlessly minimizes **training** loss. Early rounds capture the dominant signal; later rounds fit residuals that increasingly represent **random noise** in the training sample rather than real structure. With complex base learners (deep trees) and no stopping, the model starts memorizing training peculiarities.
- **Bias–variance view:** boosting aggressively reduces *bias* by building a complex additive function. Left unconstrained, this drives *variance* very high — the model fits training data well but generalizes poorly, because its late-stage "patterns" are noise artifacts.
- The diagnostic signature: as boosting rounds increase, **training error keeps falling** while **validation error reaches a minimum and then rises**.
- Base-learner complexity matters: deep trees can isolate tiny subsets (even single points), fitting noise-specific splits with no generalization value.
- Contrast with **Random Forest (bagging):** it averages many independently trained deep trees to reduce variance. Boosting's sequential dependence has **no built-in variance-averaging mechanism**, so it needs *explicit* regularization to keep variance in check.

The takeaway that motivates the whole chapter: unconstrained boosting will almost inevitably overfit, so the regularization techniques below are essential.

## 3.2 Tree-Structure Constraints: Depth, Nodes, and Splits

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

## 3.3 Shrinkage as Implicit Regularization

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

## 3.4 Data Subsampling (Stochastic Gradient Boosting)

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

## 3.5 Regularized Objective Functions (L1/L2)

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

## 3.6 Early Stopping Strategies

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

## 3.7 Hands-On: Applying Regularization

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

## Key takeaways

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
