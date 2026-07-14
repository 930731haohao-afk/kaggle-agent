# Chapter 5 — LightGBM: Light Gradient Boosting Machine

> Source: "Mastering Gradient Boosting Algorithms" (apxml.com), Chapter 5. Synthesized study notes for the Kaggle GB knowledge base — clean English restatement of the (Chinese, KaTeX-mangled) course text, with reconstructed math and cleaned Python. LightGBM (Ke et al., NeurIPS 2017) is the framework of record here and our project's primary learner.

## 5.1 Motivation: Overcoming XGBoost's Limitations at Scale

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

## 5.2 Gradient-based One-Side Sampling (GOSS)

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

## 5.3 Exclusive Feature Bundling (EFB)

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

## 5.4 Histogram-based Split Finding

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

## 5.5 Leaf-wise (Best-first) Tree Growth

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

## 5.6 Optimized Categorical Feature Handling

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

## 5.7 LightGBM API: Parameters & Configuration

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

### Key-parameter cheat sheet

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

### XGBoost ↔ LightGBM equivalent parameters

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

## 5.8 Hands-on: Implementing LightGBM

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

## Key takeaways

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
