# Chapter 4 — XGBoost: Extreme Gradient Boosting

> Source: *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 4, plus Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System" (KDD '16, DOI 10.1145/2939672.2939785) and the XGBoost documentation. Synthesized study notes for the Kaggle gradient-boosting knowledge base. Math reconstructed from KaTeX-mangled source text.

XGBoost is a scalable, regularized re-engineering of gradient boosting built by Tianqi Chen. Its edge over textbook GBM comes from three coordinated ideas: (1) a **regularized second-order objective** that folds complexity control directly into what each tree optimizes; (2) **efficient split-finding algorithms** (exact greedy, approximate quantile sketch, sparsity-aware); and (3) **systems engineering** (parallel split search, cache-aware access, out-of-core blocks) that makes it fast on large data.

---

## 4.1 Motivation and enhancements over standard GBM

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

## 4.2 The regularized (second-order) learning objective

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

## 4.3 Exact greedy split finding

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

## 4.4 Approximate split finding (weighted quantile sketch)

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

## 4.5 Sparsity-aware split finding (default directions for missing values)

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

## 4.6 System optimizations: cache-aware access, blocks, and parallelism

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

## 4.7 The XGBoost API and key parameters

Parameters fall into three groups: **general**, **booster**, and **learning-task**. The native Python API and the Scikit-learn wrapper share functionality but sometimes differ in names (`eta` ↔ `learning_rate`, `lambda` ↔ `reg_lambda`, `alpha` ↔ `reg_alpha`, `num_boost_round` ↔ `n_estimators`, `nthread` ↔ `n_jobs`, `seed` ↔ `random_state`).

**General parameters** — control the overall run. `booster` (default `gbtree`; also `gblinear`, `dart`), `verbosity` (0 silent … 3 debug), `nthread` / `n_jobs` (parallel threads).

**Booster parameters (gbtree)** — control each tree and are the primary overfitting levers (learning rate, tree-structure controls, regularization, subsampling); detailed in the table below.

**Learning-task parameters** — define the objective and evaluation metric:

- `objective` (default `reg:squarederror`): e.g. `reg:squarederror` (regression), `reg:logistic`, `binary:logistic` (probabilities), `binary:logitraw` (pre-logit score), `multi:softmax` (needs `num_class`, outputs class), `multi:softprob` (needs `num_class`, outputs probability vector), `rank:pairwise` (learning-to-rank).
- `eval_metric`: `rmse`, `mae` (regression); `logloss`, `error`, `merror`, `auc` (classification); `map`, `ndcg` (ranking). Multiple metrics may be given; the **last** one drives early stopping.

**Training-control parameters** — `num_boost_round` / `n_estimators` (number of trees; too few underfits, too many overfits — early stopping mitigates) and `early_stopping_rounds` (stop if the validation metric does not improve for that many consecutive rounds; requires an eval set).

### Key parameter reference

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

## 4.8 Hands-on: implementing XGBoost

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

## Key takeaways

- **Regularized second-order objective is the heart of XGBoost.** It minimizes $\tilde{\mathcal{L}}^{(t)} = \sum_i [g_i f_t(x_i) + \tfrac12 h_i f_t^2(x_i)] + \Omega(f_t)$ with $\Omega = \gamma T + \tfrac12\lambda\sum_j w_j^2 (+\alpha\sum_j|w_j|)$, using both gradient $g_i$ and Hessian $h_i$.
- **Everything derives from two formulas:** optimal leaf weight $w_j^* = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}$ and the structure score $-\tfrac12\sum_j \frac{G_j^2}{H_j+\lambda} + \gamma T$, which yields the split gain $\tfrac12[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma$.
- **Split finding scales via approximation.** Exact greedy is $O(d\, n\log n)$ per node; the approximate method buckets features by (Hessian-weighted) quantiles and evaluates only bucket boundaries, with global vs. local proposal variants and bucket count as a regularizer.
- **Sparsity-aware default directions** handle missing values natively by learning left/right placement that maximizes gain — no imputation needed.
- **Systems engineering** (feature-parallel split search, pre-sorted compressed column blocks, cache-aware prefetch, out-of-core sharding) turns the algorithm into a fast, scalable tool.
- **Practical control** comes from `eta` + early stopping, `max_depth`/`min_child_weight`/`gamma`, `subsample`/`colsample_bytree`, and `lambda`/`alpha`, with `tree_method=hist` for large data.
