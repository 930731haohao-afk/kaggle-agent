# Chapter 8 — Hyperparameter Optimization Strategies

> Synthesized study notes from the course *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 8. These notes reconstruct the technical content and clean up the KaTeX-mangled math from the source lessons; they are for study reference, not a verbatim transcript.

---

## 8.1 Why Hyperparameter Tuning Matters

The default settings of XGBoost, LightGBM and CatBoost give a *reasonable starting point*, but they rarely produce the best model for a specific dataset. Tuning is not a final cosmetic step — it is often a decisive factor in whether a model succeeds. This matters especially for gradient boosting because the model is built **sequentially**: each new weak learner (usually a tree) corrects the errors of the ensemble so far, and how those trees are built, how much each contributes, and how their complexity is controlled are *all* governed by hyperparameters. Poor choices lead to several undesirable outcomes:

- **Sub-optimal predictive performance.** Wrong values for `learning_rate` (`eta`), `n_estimators`, `max_depth` or the regularization terms (`lambda`, `alpha`) can prevent the model from converging to a good solution. The model may **underfit** (fail to capture the underlying pattern) or **overfit** (memorize noise in the training data and generalize poorly). Tuning searches for the configuration that minimizes the chosen loss on validation data — finding the right point in the **bias–variance trade-off**.
- **Increased overfitting risk.** Boosting models overfit easily when trees are deep or there are many rounds. Regularization-related hyperparameters exist specifically to counter this: tree-complexity params (`max_depth`, `min_child_weight`, `min_split_loss`/`gamma`), subsampling params (`subsample`, `colsample_by*`) and explicit L1/L2 penalties on leaf weights all must be tuned carefully for good generalization. Early stopping controlled by a validation metric is itself a form of tuning that directly limits the number of rounds.
- **Inefficient resource use.** Training is computationally expensive, especially on large data. Hyperparameters affect training time and memory: approximate split-finding (histogram methods in LightGBM, `tree_method='hist'` in XGBoost), feature/data subsampling rates, and tree depth all change training speed substantially. Tuning lets you trade predictive performance against compute cost — often a practical necessity.
- **Sensitivity to data characteristics.** The best settings vary with dataset size, number and type of features (dense, sparse, categorical), noise level and objective. A configuration good for one problem may be poor for another.

**Learning rate example.** A high learning rate can converge fast but may overshoot the optimum or oscillate, hurting generalization. A very low learning rate needs many more boosting rounds (`n_estimators`) — increasing training time — but combined with early stopping usually generalizes better. Tuning finds the right value for the specific problem.

Crucially, **hyperparameters interact**: the best tree depth may depend on the learning rate; the effectiveness of subsampling may change with the number of rounds. This coupling means parameters must be tuned *systematically*, not one at a time in isolation.

---

## 8.2 Identifying the Critical Boosting Hyperparameters

Not all hyperparameters have equal impact. Concentrating effort on the highest-impact ones is essential for efficient optimization.

### Core boosting parameters

**Number of boosting rounds** (`n_estimators`, `num_boost_round`, `iterations`). The total number of sequential trees. More trees generally means more complexity: too few underfits, too many overfits. Best managed *indirectly via early stopping* — set a large ceiling and let the algorithm stop when validation performance stops improving.

**Learning rate** (`learning_rate`, `eta`). Scales the contribution of each new tree added to the ensemble. Smaller values need more rounds to reach the same training-error reduction but usually generalize better. It acts as a form of regularization by shrinking the step size in function space at each iteration. Typical range **0.01 to 0.3**. There is a direct trade-off: **lowering the learning rate requires raising the number of rounds.** Strongly coupled with `n_estimators`.

### Tree-structure parameters

**Maximum tree depth** (`max_depth`). Limits how deep each tree may grow. Deeper trees capture more complex feature interactions but overfit more easily. Shallow trees (e.g. depth 4–8) usually strike a good balance. Typical range **3 to 10**, highly data-dependent. In LightGBM, `num_leaves` is often a more direct control than `max_depth` because of its leaf-wise growth.

**Minimum child weight / minimum samples per leaf** (`min_child_weight` [XGBoost], `min_sum_hessian_in_leaf` [LightGBM]; `min_data_in_leaf` [LightGBM/CatBoost], `min_samples_leaf` [sklearn GBM]). Sets a minimum threshold on the sum of instance weights (the Hessian for XGBoost/LightGBM) or the number of samples required in a leaf. Prevents the tree from creating splits that isolate very small groups — a regularization mechanism against fitting noise. Larger values give more conservative trees. Interacts with `max_depth`.

**Minimum split gain** (`gamma` [XGBoost], `min_gain_to_split` [LightGBM], `min_impurity_decrease` [sklearn GBM]). The minimum reduction in the loss required to make a split; splits not meeting it are pruned. A direct regularizer on the splitting process — larger values make the algorithm more conservative.

### Subsampling parameters (introduce randomness → improve generalization and speed)

**Row subsampling** (`subsample`; alias `bagging_fraction` in LightGBM). The fraction of training rows randomly sampled (without replacement) to build each tree. Values below 1.0 add randomness, reduce variance and help prevent overfitting. Typical range **0.5 to 1.0**.

**Column subsampling** (`colsample_bytree`, `colsample_bylevel`, `colsample_bynode` in XGBoost; alias `feature_fraction` in LightGBM). The fraction of features considered when building each tree, each level, or each split. Especially useful with many features, since it stops the model over-relying on a few dominant features. Regularizes (particularly for high-dimensional data) and speeds up computation. Typical range **0.5 to 1.0**.

### Algorithm-specific parameters

- **XGBoost:** `reg_alpha` (L1 penalty on leaf weights, can yield sparse weights); `reg_lambda` (L2 penalty on leaf weights, default usually 1, generally more impactful than `reg_alpha` for tree models).
- **LightGBM:** `num_leaves` — the max leaves per tree; because LightGBM grows leaf-wise (splitting the leaf with the largest loss reduction), this is the *primary* complexity control and is usually tuned instead of `max_depth`; high values overfit easily. A common constraint is `num_leaves` $\le 2^{\text{max\_depth}}$. `boosting_type` selects `gbdt` (standard), `dart` (adds dropout, sometimes better at the cost of more tuning) or `goss` (gradient-based one-side sampling).
- **CatBoost:** `cat_features` (explicitly flags categorical columns to enable ordered target statistics — required for effective categorical handling); `l2_leaf_reg` (L2 like XGBoost's `reg_lambda`); `border_count` (bins for numerical feature discretization, affects speed/memory); `one_hot_max_size` (use one-hot encoding for low-cardinality categoricals up to this size).

### Tuning priority

Tuning everything at once is infeasible. A practical priority order (revisit earlier params after tuning later ones, since they interact):

1. **Core learning parameters** — find a good `learning_rate` + `n_estimators` combination (with early stopping).
2. **Tree complexity** — tune `max_depth` (or `num_leaves` for LightGBM) and `min_child_weight` / `min_data_in_leaf`.
3. **Subsampling** — optimize `subsample` and `colsample_by*`.
4. **Explicit regularization** — fine-tune `gamma`, `reg_alpha`, `reg_lambda` / `l2_leaf_reg`.
5. **Algorithm-specific parameters** — adjust library-specific settings if needed.

---

## 8.3 Systematic Tuning: Grid Search vs Random Search

Grid search and random search are the foundational systematic methods — a big step up from ad-hoc manual tweaking.

### Grid search

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

### Random search

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

### Practical search-space guidance

- **`learning_rate`** — log-uniform (e.g. 0.001–0.3), because its effect is multiplicative.
- **`n_estimators`** — integer range (e.g. 100–1000), but strongly tied to learning rate and early stopping; if early stopping is used effectively, tuning `n_estimators` directly matters much less.
- **`max_depth`** — integers e.g. 3–10.
- **Subsampling** (`subsample`, `colsample_*`) — uniform e.g. 0.5–1.0.
- **Regularization** (`lambda`, `alpha`) — log-uniform e.g. 1e-3 to 10.
- **Budget** — for random search start around 20–50 iterations and increase while gains continue.

Both `GridSearchCV` and `RandomizedSearchCV` use CV internally (`cv=`) to reduce the risk of overfitting a single train/test split.

---

## 8.4 Advanced Tuning: Bayesian Optimization

Grid/random search are systematic but blind — grid suffers the curse of dimensionality and evaluates many unpromising regions; random search has no strategy to focus where results are better. **Bayesian optimization** takes a more informed approach: find a good configuration in **as few expensive objective evaluations as possible** (here, each evaluation = train + validate a boosting model).

### Core idea: informed search

Build a **probabilistic model** of the relationship between hyperparameters and performance (validation accuracy/loss). This **surrogate model** is much cheaper to evaluate than the true objective, and it is used to decide intelligently which hyperparameters to try next — **balancing exploration** (probing uncertain regions) **against exploitation** (sampling near the current best).

### Two components

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

### The optimization loop

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

## 8.5 HPO Frameworks: Optuna and Hyperopt

Implementing Bayesian optimization by hand — managing trials, pruning, parallelization — is complex. Dedicated frameworks automate most of it so you focus on defining the search space and objective. Two well-known Python frameworks are **Optuna** and **Hyperopt**, both of which implement TPE.

### Optuna

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

### Hyperopt

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

## 8.6 Coarse-to-Fine Tuning Strategy

Blindly running an automated search over the *full* range of every parameter is expensive and inefficient. A structured multi-stage approach — broad **coarse** search first, then focused **fine** search — allocates compute wisely.

### Stage 1 — Coarse tuning (broad exploration)

- **Goal:** quickly identify *promising regions* of the space, not the absolute optimum. Sketch the general contour of performance.
- **Method:** **random search** (efficient at finding diverse combinations), or the early phase of Bayesian optimization.
- **Parameters:** focus on the high-impact ones — `learning_rate`, `n_estimators` (better controlled by early stopping — set a large ceiling rather than searching a wide range directly), `max_depth`, sampling params (`subsample`, `colsample_*`), regularization (`lambda`/`alpha`).
- **Ranges:** define **wide** ranges — log scale for `learning_rate` and regularization (e.g. 0.001–0.1), linear for depth and sampling rates.
- **Execution:** limited iterations (e.g. 30–100); a **cheaper validation** scheme (single hold-out or 3-fold CV); optionally a representative **subset of the data** if it is very large.
- **Result:** a coarse sense of which ranges work and which can be excluded (e.g. "learning rate below 0.01 is consistently poor" or "depth above 8 overfits without gain").

### Stage 2 — Fine tuning (focused optimization)

- **Goal:** precisely locate the best configuration within the promising region found in Stage 1.
- **Method:** **Bayesian optimization** (Optuna/Hyperopt) excels here — its modeling and intelligent point selection are efficient in a restricted space. A focused grid search is also viable if the narrowed space is small enough.
- **Parameters:** the narrowed ranges from Stage 1, plus additional finer-grained params held fixed earlier (e.g. `min_child_weight`, `gamma`).
- **Ranges:** tighter, based on coarse results (e.g. if good rates were 0.01–0.05, use `LogUniform(0.01, 0.05)`).
- **Execution:** larger budget (e.g. 50–200+ iterations); a **more robust** evaluation — full-data k-fold CV (5- or 10-fold); early stopping in every evaluation to set the best `n_estimators` for each tested combination.

### Integrating early stopping

Because `n_estimators` is tightly coupled with `learning_rate` (lower rate → more trees), it is more efficient **not** to search `n_estimators` over a wide range but to:

1. Set a large ceiling (e.g. 2000).
2. Use the built-in early stopping of XGBoost/LightGBM/CatBoost in every evaluation (within each CV fold).
3. Configure it to monitor a validation metric and stop after N rounds without improvement (e.g. `early_stopping_rounds=50`).

That way the optimal tree count is determined automatically for every combination of the *other* hyperparameters being tested.

**Practical notes:** Optuna fits this well — run a coarse study, analyze it (visualizations), then launch a refined study with a narrowed space. Tuning is not strictly linear: fine results may reveal the coarse search was too narrow, so be ready to revisit assumptions and re-run a broader search.

---

## 8.7 Cross-Validation Strategy for Honest Tuning

Relying on a single train/validation split to evaluate hyperparameters is misleading — the chosen params may **overfit that specific validation set** and generalize poorly. **Cross-validation (CV)** gives a more reliable estimate by evaluating on multiple subsets.

### CV inside the tuning loop

For **each** hyperparameter combination the search proposes:

1. Split the training data into $K$ folds.
2. Train $K$ times, each on $K-1$ folds and validate on the held-out fold.
3. Compute the metric on each held-out fold.
4. **Aggregate** (usually mean, sometimes with std) the $K$ scores into one stable estimate for that configuration.

The tuning algorithm uses this aggregated score to decide what to try next (Bayesian) or which combination wins (grid).

### CV strategies

- **Standard K-fold.** Shuffle and split into $K$ equal folds; $K=5$ or $10$ typical. Higher $K$ trains on more data per iteration but costs more.
- **Stratified K-fold** (`StratifiedKFold`). Preserves each class's proportion in every fold — the **recommended default for classification**, especially with imbalanced data, where random folds could distort class distribution and give unreliable estimates.
- **Group K-fold** (`GroupKFold`). When samples are not independent (multiple measurements per patient, images per location, logs per user session), standard K-fold can place the same group in both train and validation, causing **leakage** and over-optimistic estimates. Group K-fold keeps all samples of a group entirely on one side of each split, using an identifier (`patient_id`, `user_id`).
- **Time-series CV.** For temporal data, random shuffling breaks time order and causes **look-ahead bias** (using future to predict past). Preserve order with:
  - **Expanding window / rolling forecast origin** (`TimeSeriesSplit`): train on folds 1..t, validate on t+1; the training window grows over time — simulates periodic retraining as new data arrives.
  - **Sliding window:** fixed-size training window that slides forward. Choose expanding vs sliding by whether older data stays relevant (expanding) or the pattern drifts (sliding).

### Wiring CV into frameworks

- **Scikit-learn:** pass `cv=5` or a splitter object (`cv=StratifiedKFold(n_splits=5)`, `cv=TimeSeriesSplit(n_splits=5)`) to `GridSearchCV`/`RandomizedSearchCV`.
- **Optuna/Hyperopt:** include the CV loop *inside* the objective function — run K-fold, average the fold scores, and return that average to guide the search.

### CV with early stopping

Combining early stopping with CV requires care. A common per-fold procedure to evaluate one hyperparameter set:

1. For each of the $K$ folds:
   - Further split the $K-1$ training folds into a sub-train set and an early-stopping validation set.
   - Train on sub-train, using the early-stopping set to find the best number of rounds for that fold.
   - Record the score on the *main held-out fold* (using that best round count), and record the round count used.
2. Average the $K$ fold scores → the score for this hyperparameter set.
3. Optionally average / take the median of the best round counts across folds.

When training the **final** model (after the best hyperparameters are found), fit on the *entire* training set, setting the boosting rounds to the averaged/median count from CV, or determine it with a separate final validation set.

### Cost and the final model

CV multiplies tuning cost by $K$: 5-fold CV over 100 combinations = **500 model trainings**. To manage it: use fewer folds ($K=3$ or $5$) if too expensive; prefer efficient search (random/Bayesian over exhaustive grid); do a cheap broad search (fewer folds / less data) then a finer search on promising regions. Remember: **CV during tuning is only for evaluating hyperparameter sets.** Once the best set is found, train the final model once on the full training data, and the CV estimate is your expectation of how it will perform on new, unseen data.

---

## 8.8 Hands-On: Advanced Tuning with Optuna (XGBoost)

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

## Practical Tuning Recipe / Checklist

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

## Key takeaways

- Defaults are a *starting point*, not an answer; because boosting is sequential and its hyperparameters interact, tuning is often decisive and must be **systematic**, not one-parameter-at-a-time.
- Prioritize by impact: **`learning_rate` + `n_estimators` together** (rate down ⇒ rounds up, managed by early stopping) → tree complexity (`max_depth`/`num_leaves`, `min_child_weight`/`min_data_in_leaf`) → subsampling (`subsample`, `colsample_*`) → explicit regularization (`reg_lambda`, `reg_alpha`, `gamma`).
- **Random search beats grid search in high dimensions** because only a few params usually matter and random sampling covers those better for the same budget; grid cost is exponential.
- **Bayesian optimization** (GP or TPE surrogate + an acquisition function like Expected Improvement that balances exploration/exploitation) finds good configs in far fewer expensive evaluations — the right tool when each train+validate is slow.
- **Optuna** (define-by-run, pruning, visualizations) and **Hyperopt** (explicit space, TPE, Spark) automate the search loop.
- Use a **coarse-to-fine** strategy to spend compute where it counts, and always evaluate with the **right CV scheme** to avoid leakage and to keep from overfitting a single validation split.
- Always determine `n_estimators` through **early stopping**, and estimate final generalization on a held-out set the tuner never saw.

> **Relevance to our work:** Our automated tree-search harness already embodies the CV-gated tuning philosophy of this chapter — every candidate is scored through a fixed, honest OOF/CV gate rather than a single split, which is exactly how §8.7 says to avoid overfitting the validation set. The three-arm search (baseline / random-ish expansion / LLM-proposed configs) mirrors coarse-to-fine: broad diverse proposals first, then focused refinement of promising regions. Practical carry-overs: (1) keep `n_estimators` controlled by early stopping and store `best_iteration` per fold so the final refit uses a CV-derived round count — matching the §8.8 self-correction note; (2) sample `learning_rate` and the regularization terms on a **log** scale, tree/sampling params linearly; (3) fix determinism (`deterministic`, `force_row_wise`, `num_threads`) so Optuna/TPE trial scores are reproducible across processes — see our LightGBM-determinism memo. Above all, **watch the CV–LB gap**: a hyperparameter set that wins the CV gate by a razor-thin margin is a prime candidate for private-leaderboard regression, so prefer robust configs (low fold-to-fold std) over the single highest-mean trial, and never trust a public-LB bump that the CV gate did not corroborate.
