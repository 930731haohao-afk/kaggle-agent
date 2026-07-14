# Chapter 2 — The Gradient Boosting Algorithm in Depth

> Source: *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 2. Synthesized study notes reconstructed from the scraped course text, with LaTeX cleaned and code idiomatized. This chapter derives GBM as gradient descent in function space, works out the pseudo-residuals for the common regression and classification losses, and covers shrinkage, stochastic subsampling, and the scikit-learn implementation.

## 2.1 Functional Gradient Descent

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

## 2.2 Deriving the Generic GBM Algorithm

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

## 2.3 Common Loss Functions for Regression

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

## 2.4 Common Loss Functions for Classification

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

## 2.5 The Role of Shrinkage (Learning Rate)

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

## 2.6 Sampling Methods (Stochastic Gradient Boosting)

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

## 2.7 Implementing GBM with Scikit-learn

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

## 2.8 Hands-on: Building a Basic GBM Model

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

## Key takeaways

- **GBM = gradient descent in function space.** Each round fits a base learner to the negative gradient of the loss at the current predictions (the pseudo-residual $r_{im} = -\partial L/\partial F$), then adds it to the ensemble.
- **Pseudo-residual closed forms depend on the loss:** L2 → residual $y - F$; L1 → $\operatorname{sign}(y - F)$; Huber → residual when $|y-F|\le\delta$, else $\delta\cdot\operatorname{sign}$; logistic/deviance → $y - p$ (same shape for multiclass per class).
- **Leaf values come from a line search**, $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i)+\gamma)$ — mean for L2, median for L1 — not just averaged residuals.
- **Initialization** $F_0$ is the loss-minimizing constant: mean for L2, log-odds for log loss.
- **Shrinkage $\nu$ regularizes** by damping each tree; it trades off against `n_estimators` (low $\nu$ ⇒ many trees). Pair low learning rate with early stopping.
- **Stochastic subsampling** (rows via `subsample`, columns via `max_features`) is cheap regularization plus a speedup; ~0.5–0.8 rows, ~0.7–0.9 features are sensible.
- **Log loss beats exponential loss** for classification: robust to mislabels, probabilistic/information-theoretic grounding, smoother gradients.
- **Scikit-learn is the reference baseline** (`GradientBoostingRegressor`/`Classifier`); production/competition work moves to XGBoost/LightGBM/CatBoost. Current loss names: `'squared_error'` (was `'ls'`), `'log_loss'` (was `'deviance'`).

> **Relevance to our work:** everything here maps one-to-one onto our LightGBM practice — `objective` (loss/pseudo-residual choice), `learning_rate` + early stopping (shrinkage/n_estimators trade-off), `bagging_fraction`/`feature_fraction` (stochastic subsampling), and stratified CV on the true competition metric. The functional-gradient mental model is what makes tuning these deliberate rather than trial-and-error.
