# Chapter 9 — Gradient Boosting for Specialized Tasks

> Source: "Mastering Gradient Boosting Algorithms" (apxml.com), Chapter 9. Synthesized English study notes reconstructed from the scraped course text, with the KaTeX-mangled math cleaned into correct LaTeX. Covers four families of non-standard boosting tasks — learning to rank, survival analysis, quantile regression, and multi-output — plus a hands-on XGBoost ranking walkthrough.

Standard ML tasks are classification (assign a label) or regression (predict a continuous value). This chapter covers boosting objectives that fall outside that mold: ordering items (ranking), predicting time-to-event under censoring (survival), predicting distribution quantiles rather than the mean (quantile regression), and predicting several targets jointly (multi-output). The unifying theme is that gradient boosting only needs a differentiable (or approximately differentiable) loss with a gradient and Hessian — so any task expressible that way can reuse the same additive-tree machinery.

## 9.1 Learning to Rank (LTR) with Gradient Boosting

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

## 9.2 Ranking Objective Functions (Pairwise and Listwise)

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

## 9.3 Gradient Boosting for Survival Analysis

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

## 9.4 Survival Objective Functions (Cox Proportional Hazards)

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

## 9.5 Quantile Regression with Gradient Boosting

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

## 9.6 Implementing the Quantile Loss (Custom Objective)

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

## 9.7 Multi-Output Gradient Boosting

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

## 9.8 Hands-On: Implementing Ranking with XGBoost

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

## Key takeaways

- Gradient boosting extends beyond mean regression/classification to any task with a differentiable-enough loss (gradient + Hessian) — ranking, survival, quantiles, multi-output all reuse the same additive-tree engine.
- **Ranking:** metrics (NDCG/MAP/MRR) are non-differentiable, so GBMs optimize pointwise/pairwise/listwise surrogates. LambdaMART (pairwise logistic gradient scaled by $|\Delta\text{NDCG}|$) is the workhorse; needs query grouping and Group K-Fold. XGBoost `rank:pairwise|ndcg|map`, LightGBM `lambdarank`.
- **Survival:** handle right-censoring via the Cox negative log partial likelihood; $h(t\mid x)=h_0(t)e^{F(x)}$; output is a log-risk score; evaluate with C-index. XGBoost `survival:cox` (signed-time labels), LightGBM `coxph`, CatBoost `Cox`. Also `survival:aft` (accelerated failure time) exists as an alternative.
- **Quantile:** pinball loss $L_\tau(y,\hat y)=\max(\tau(y-\hat y),(\tau-1)(y-\hat y))$ has a sign-only gradient and a degenerate (zero/undefined) Hessian — built-ins approximate it, custom objectives return a small positive constant Hessian. Fit one model per quantile; Q10/Q90 give an 80% prediction interval. Watch for quantile crossing.
- **Multi-output:** default to independent models per target (manual loop or `MultiOutputRegressor`/`Chain`); native multi-output splitting is research-grade and not in mainstream libraries.

> **Relevance to our work:** For prediction-interval competitions (or any metric rewarding calibrated uncertainty, e.g. pinball / Winkler scoring), train separate LightGBM/XGBoost quantile models at the required $\alpha$ levels rather than a single mean model — Q_low/Q_high directly produce the interval, and the median quantile is a robust point estimate.
> **Relevance to our work:** For recommendation-style or search-ranking competitions where the target is an ordering within groups (sessions, users, queries), use `rank:pairwise` / `lambdarank` with correct `qid`/`group` construction and **Group K-Fold** CV — never a random split that leaks documents of a group across folds — and evaluate with the competition's NDCG/MAP@k rather than regression error.
