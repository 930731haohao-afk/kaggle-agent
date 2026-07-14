# Chapter 6 — CatBoost: Ordered Boosting for Categorical Data

> Source: "Mastering Gradient Boosting Algorithms" (apxml.com), Chapter 6, eight lessons.
> Synthesized English study notes reconstructed from a scraped Chinese transcript with KaTeX-mangled math cleaned up. Foundational reference throughout: Prokhorenkova et al., *CatBoost: Unbiased Boosting with Categorical Features*, NeurIPS 2018 (arXiv:1706.09516).

CatBoost's whole design answers one question: how do you put categorical features directly into gradient boosting without leaking the target? Its two signature ideas — **Ordered Target Statistics** (for encoding) and **Ordered Boosting** (for training) — both enforce a "use only the past" rule under random permutations, and both are made practical by **oblivious (symmetric) trees**.

---

## 6.1 The Trouble with Categorical Data

Tree models split on numeric thresholds (`feature < threshold`), so categorical features must first become numbers. Every naive encoding has a failure mode:

- **One-hot encoding (OHE).** Fine for low cardinality, but a high-cardinality feature (e.g. `user_id` with thousands of values) explodes into thousands of mostly-zero columns. Consequences: extreme sparsity, slower split search (dimensionality dominates even sparse-optimized libraries), and deep/complex trees because each split isolates only one category — capturing "several categories together" needs many splits.
- **Label / ordinal encoding.** Assigns an arbitrary integer per category (`red→0, green→1, blue→2`). This invents an ordering that usually does not exist: the model may treat `green` as "between" red and blue, or read `blue−red = 2` as twice `green−red = 1`. Splits like `color_encoded < 1.5` become meaningless for nominal features (acceptable only for genuinely ordinal ones like low/medium/high).
- **Target (mean) encoding.** Replace each category with a statistic of the target over rows in that category (e.g. mean target for `city = A`). This directly builds a strong tie between feature and target.

**Target leakage** is the central danger of target encoding: because a row's own target contributed to its encoding, the model sees a direct clue about $y$. Result:

- **Overfitting** — training performance looks great but is spurious; it does not generalize once the leaked signal is absent on unseen data.
- **Prediction shift / unreliable stats** — rare categories with few observations give noisy statistics, and the in-category target distribution can differ between train and future data, so the encoding is miscalibrated.

Holdout sets or smoothing mitigate leakage but add complexity and never fully solve it. Add to this the difficulty of discovering high-order **feature interactions** (e.g. `product_category × store_location`), which naive OHE + trees capture only with very deep trees or laborious manual feature crosses. These problems motivate CatBoost's built-in Ordered TS and automatic feature combinations.

---

## 6.2 Ordered Target Statistics (Ordered TS)

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

## 6.3 Prediction Shift and Ordered Boosting

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

## 6.4 Feature Combinations

Much predictive signal lives in *interactions*: knowing `Browser='Safari'` and `OS='macOS'` together can beat either alone. Manually crafting crosses (a `browser_os` feature) works but scales badly — the number of pairwise, triple, … combinations grows exponentially, and finding the useful ones needs domain expertise.

CatBoost generates and evaluates categorical combinations **automatically and greedily during tree construction**, not by pre-building a giant feature set. When choosing the best split for a node (or, for oblivious trees, a whole level), it considers combinations of:

- one or more categorical features already used in ancestor splits (root → current node path), plus
- a new candidate categorical feature.

So for a new categorical feature $C_{new}$, CatBoost also evaluates splits on $(C_{ancestor1}, C_{new})$, $(C_{ancestor2}, C_{new})$, $(C_{ancestor1}, C_{ancestor2}, C_{new})$, and so on. Each newly formed combination is treated exactly like any categorical feature — encoded with the same **Ordered TS** "only-the-past" rule to avoid leakage — and the algorithm picks whichever split (original or combined) yields the greatest loss reduction.

**Integration with oblivious trees.** Because a symmetric tree uses the *same* split across all nodes at a level, a chosen combination (e.g. `Country × Browser`) applies to the entire depth level. Combinations thus build up progressively with depth: a level-3 combination can involve categoricals used at levels 1 and 2.

**Controlling complexity.** Generation is greedy, not exhaustive — it typically starts with 2-feature combinations and extends a proven combination with another feature at deeper levels. `max_ctr_complexity` caps the number of categorical features fused into one combination (default commonly 4; the transcript's parameter lesson notes default 4, its combinations lesson notes "up to two" as the practical starting point). Higher values capture higher-order interactions at extra compute and overfitting risk. The related `ctr_max_border_count` controls how many splits are considered for combinations involving numeric features. Upside: automated interaction discovery, higher accuracy from complex categorical dependencies, and native integration with Ordered TS and symmetric trees.

---

## 6.5 Oblivious (Symmetric) Trees

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

## 6.6 GPU Training Acceleration

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

## 6.7 CatBoost API: Parameters and Configuration

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

## 6.8 Hands-On: Implementing CatBoost

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

## Key takeaways

- **Naive categorical encodings all fail somehow:** OHE explodes high cardinality into sparse, deep-tree-inducing columns; label encoding invents false ordinality; mean/target encoding leaks the target and overfits.
- **Ordered TS** encodes each row from *only its predecessors* in a random permutation, excluding its own target, with prior smoothing $\hat{x} = \frac{\sum_{j<i}[x_j=x_i]y_j + a\,p}{\sum_{j<i}[x_j=x_i] + a}$ to stabilize rare categories.
- **Ordered Boosting** applies the same "only-the-past" principle to residuals: sample $i$'s gradient comes from a model $M_{m-1}$ trained on samples excluding $i$, giving unbiased residuals and fixing prediction shift.
- **Feature combinations** are built greedily during tree growth (ancestor categoricals × new one), encoded with Ordered TS, controlled by `max_ctr_complexity`.
- **Oblivious (symmetric) trees** use one split per level → balanced, vectorizable, GPU-friendly, and implicitly regularizing; the ensemble compensates for a single tree's lower expressiveness.
- **In practice:** pass raw categoricals via `cat_features` (never pre-encode), use `Pool`, and always pair `eval_set` with `early_stopping_rounds`. GPU (`task_type='GPU'`) shines on medium/large data.

> **Relevance to our work:** Reach for CatBoost over LightGBM when a competition dataset has **many high-cardinality categorical features** (IDs, locations, product/category codes) or when a strong categorical predictor makes target/mean encoding **leakage-prone** — exactly where Ordered TS + Ordered Boosting deliver a leakage-free encoding out of the box with minimal preprocessing. LightGBM (native categorical splitting) or XGBoost often remain stronger on predominantly numeric data or when you need the extra flexibility of asymmetric trees; treat CatBoost as a distinct ensemble member and blend candidate whose bias/variance profile differs from the level-/leaf-wise learners.
