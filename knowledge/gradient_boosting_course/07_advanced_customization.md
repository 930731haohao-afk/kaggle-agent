# Chapter 7 — Advanced Topics & Customization

> Synthesized study notes from the course *Mastering Gradient Boosting Algorithms* (apxml.com), Chapter 7. These notes reconstruct the technical content and clean up the KaTeX-mangled math and code from the source lessons; they are for study reference, not a verbatim transcript. This is the course's longest, most code-heavy chapter, covering model interpretability (SHAP / TreeSHAP), probability calibration, custom objectives and evaluation metrics, and imbalanced-data techniques.

---

## 7.1 Understanding Model Interpretability with SHAP

Gradient boosting models are powerful predictors but effectively operate as **black boxes**: the sequential addition of many deep trees makes it hard to see *why* a particular prediction was made or which features drive overall behaviour. Standard feature-importance measures (split gain, permutation) give a coarse global picture but lack the resolution to explain a single prediction or capture complex feature interactions. This opacity impedes trust, complicates debugging, and blocks use in regulated settings.

**SHAP (SHapley Additive exPlanations)** is a principled solution rooted in cooperative game theory. It treats a prediction as a *game* in which the features cooperate to produce the output, and it uses **Shapley values** to fairly distribute the "payout" (the gap between the prediction and a baseline) among the "players" (the features).

### Shapley values

The Shapley value $\phi_i$ of feature $i$ is its **average marginal contribution** to the prediction across all possible feature coalitions (subsets). It quantifies the effect of including that feature's value. For a general ML model, computing exact Shapley values is expensive because it requires evaluating the model over all $2^M$ feature subsets, where $M$ is the number of features.

### The SHAP framework and its guarantees

SHAP provides efficient algorithms to estimate Shapley values, with three theoretical properties:

- **Local accuracy (additivity).** For a given prediction, the SHAP values of all features sum to the difference between the prediction $f(x)$ and the baseline expected prediction $\phi_0 = E[f(x)]$ (usually the mean prediction over the training data):

$$
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i
$$

- **Missingness.** A feature that is genuinely absent from (marginalized out of) a coalition receives a SHAP value of zero.
- **Consistency.** If the model changes so that a feature's marginal contribution increases or stays the same (regardless of other features), its SHAP value does not decrease. This ties feature importance to how much the model actually relies on the feature — a property gain-based importance can violate.

For tree ensembles like gradient boosting, SHAP offers a specialized, efficient algorithm (**TreeSHAP**, next section) that computes *exact* Shapley values far faster than model-agnostic methods.

### Using SHAP with boosting libraries

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

### Visualizing SHAP values

- **Force plot (local).** Explains one prediction, showing which features push it away from the baseline. Features that raise the prediction are red, those that lower it are blue; the final prediction is the baseline plus the sum of all contributions.
- **Summary plot (global).** One dot per (feature, instance) SHAP value. Features are ranked by mean absolute SHAP value; horizontal position shows the SHAP value (effect direction and size) and colour encodes the raw feature value (high/low), revealing whether high values push predictions up or down.
- **Dependence plot.** Plots a feature's value (x) against its SHAP value (y), optionally coloured by a second, possibly interacting feature to expose interaction effects.

*Source references: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017, arXiv:1705.07874); Lundberg, Erion & Lee, "Consistent Individualized Feature Attribution for Tree Ensembles" (arXiv:1802.03888) — introduces TreeSHAP.*

---

## 7.2 TreeSHAP for Gradient Boosting

The generic, model-agnostic **KernelExplainer** approximates Shapley values by perturbing inputs and observing output changes. For gradient boosting ensembles — hundreds or thousands of trees, many features — this sampling-and-re-evaluation approach becomes prohibitively slow. **TreeSHAP** (Lundberg et al.) is an algorithm built specifically for tree models (XGBoost, LightGBM, CatBoost) that computes **exact** Shapley values much more efficiently.

### How TreeSHAP works

Instead of sampling and re-evaluating, TreeSHAP exploits the tree structure to compute conditional expectations $E[f(x) \mid x_S]$ — the model's expected output given only the feature values in subset $S$. The Shapley value is the weighted difference of these conditional expectations as feature $i$ is added to each subset:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F| - |S| - 1)!}{|F|!}\,\Big[\,E[f(x) \mid x_{S \cup \{i\}}] - E[f(x) \mid x_S]\,\Big]
$$

where $F$ is the full feature set and $x_S$ the values of the features in $S$. TreeSHAP avoids the exponential $2^{|F|}$ enumeration by using a **polynomial-time** algorithm that pushes all feature subsets down the tree paths simultaneously: at each split node it tracks, per feature, the proportion of subsets following the left vs. right branch, maintaining a weighted average of conditional expectations from root to leaf. This runs on every tree in the ensemble; the per-tree Shapley values are then combined (the ensemble prediction is a sum of tree outputs, often after a link function such as the logistic transform).

### Benefits for gradient boosting

- **Efficiency.** Orders of magnitude faster than Kernel SHAP on tree ensembles — exact SHAP values for thousands of predictions in seconds/minutes rather than hours/days.
- **Exactness.** Computes theoretically exact Shapley values for tree models (under the standard feature-independence assumption), eliminating the approximation error of Kernel SHAP.
- **Interaction values.** The algorithm extends to efficiently compute **SHAP interaction values** $\phi_{ij}$, quantifying the extra contribution arising from the interaction between a pair of features, averaged over subsets.

### Implementation

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

### Caveats

- **Feature-independence assumption.** Like all Shapley-based methods, TreeSHAP implicitly assumes feature independence when computing conditional expectations. With highly correlated features, interpretation requires care, since the algorithm may average over unrealistic feature-value combinations.
- **Cost at scale.** Much faster than alternatives, but millions of instances or tens of thousands of trees can still be time/memory intensive. `shap_values(..., approximate=True)` or `check_additivity=False` trade exactness for speed.
- **Scope.** TreeSHAP is for tree models only; use `DeepExplainer`, `LinearExplainer`, or the model-agnostic `KernelExplainer` for other model types.

*Source references: Lundberg & Lee (NeurIPS 2017); Shapley, "A Value for n-person Games" (1953) — the game-theoretic foundation.*

---

## 7.3 Global vs. Local Explanations

SHAP (and its efficient TreeSHAP variant) supports two complementary levels of explanation, and distinguishing them is key to fully understanding a model.

### Global explanations — the big picture

Global explanations describe the model's overall behaviour across the whole dataset: *which features are most influential on average?* and *what is the general relationship between a feature and the output?* The standard SHAP-based global importance for feature $j$ is the **mean absolute SHAP value** over all $n$ instances:

$$
\text{GlobalImportance}_j = \frac{1}{n} \sum_{i=1}^{n} |\phi_{ij}|
$$

where $\phi_{ij}$ is the SHAP value of feature $j$ for instance $i$. Higher mean $|\phi|$ = more influential overall. This is a more reliable and **consistent** ranking than gain or split-count importance. Visualized via the **summary plot** (importance + effect direction combined) or a **bar chart** of mean $|\phi|$; the **dependence plot** shows the global shape of a feature's effect.

### Local explanations — a single prediction

Local explanations answer *why did the model make this specific prediction for this instance?* — e.g. why this customer's loan was rejected, or which factors most raised this user's churn probability. SHAP values are inherently local: $\phi_{ij}$ measures how feature $j$ pushed instance $i$'s prediction away from the baseline. The core SHAP equation links the baseline $E[f(X)]$ to the instance prediction $f(x_i)$:

$$
f(x_i) = E[f(X)] + \sum_{j=1}^{M} \phi_{ij}
$$

The additivity means each feature's positive/negative contribution to a single prediction is directly readable, typically via a **force plot** (red = pushes up, blue = pushes down; block size = magnitude).

### Complementary perspectives

- **Global** is valuable for understanding the main drivers, comparing models, and guiding feature engineering.
- **Local** is indispensable for debugging surprising predictions, explaining decisions to stakeholders/customers, assessing individual fairness, and building trust in specific outcomes.

---

## 7.4 Probability Calibration for Classification

Gradient boosting classifiers often achieve strong AUC/F1 but their raw output scores are **not necessarily true probabilities**: a score of 0.9 does not mean a 90% objective chance of the positive class. Optimizing log loss tends to push scores toward 0 and 1 to sharpen discrimination, which can distort the probability interpretation. When calibrated probabilities matter — for decisions, risk assessment, thresholding, or ensembling — you should calibrate the outputs.

### What calibration means and how to check it

A perfectly calibrated binary classifier has the property that among instances predicted with probability $p$, the actual positive fraction is close to $p$. The standard diagnostic is a **reliability diagram (calibration curve)**: bin the predicted probabilities (0–0.1, 0.1–0.2, …), and for each bin plot the mean predicted probability against the observed positive fraction. The diagonal is perfect calibration; bars below it mean over-confidence (predicted > actual frequency), above it mean under-confidence.

Boosting can produce mis-calibrated probabilities because of its additive nature, its focus on correcting errors (which drives extreme scores at high-confidence points), and the specifics of tree construction — so checking (and possibly fixing) calibration of XGBoost/LightGBM/CatBoost is good practice.

### Two calibration methods

Both are fit **after** the main classifier, on a **separate calibration set** (a held-out split not used to train the base model). Calibrating on the training data gives over-optimistic results.

**Platt scaling** — parametric. Assumes the deviation between scores $s$ and true probabilities can be corrected by fitting a sigmoid. It finds parameters $A, B$ so that:

$$
P_{\text{Platt}}(y = 1 \mid s) = \frac{1}{1 + \exp(A s + B)}
$$

$A$ and $B$ are found by minimizing log loss on the calibration set. Platt scaling works best when the calibration curve is monotonic and S-shaped; it is computationally cheap and needs relatively little data.

**Isotonic regression** — non-parametric. Fits a non-decreasing, piecewise-constant step function mapping scores to observed targets (least-squares best fit preserving input order), using the **Pool Adjacent Violators Algorithm (PAVA)**. It makes fewer assumptions about the bias shape and fits better when the curve is not sigmoidal, but needs more data for stable results and can produce sharp steps in the calibrated probabilities.

### Implementation in scikit-learn

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

### Practical considerations

- **Independent data.** Always calibrate on data unseen by the base model; `CalibratedClassifierCV` handles this via CV, otherwise use a dedicated hold-out.
- **Method choice.** With limited calibration data or a known S-shaped bias, prefer Platt scaling; isotonic is more flexible but data-hungry and can be less smooth. Evaluate both with the **Brier score** or the reliability diagram.
- **Effect on other metrics.** Calibration improves probability reliability (Brier score, log loss) and may slightly reduce ranking metrics like AUC — usually a small effect. Calibrate when accurate probabilities are the primary need.

*Source references: Platt (1999); Zadrozny & Elkan (2002, isotonic); Niculescu-Mizil & Caruana, "Predicting good probabilities with supervised learning" (2005).*

---

## 7.5 Implementing Custom Loss Functions

Standard losses (MSE for regression, log loss for classification) do not always match the true business objective — you might want to penalize over-prediction more than under-prediction, or weight certain error types more heavily. XGBoost and LightGBM let you define and optimize your own custom **objective (loss)** function.

### The role of gradient and Hessian

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

### Worked examples of gradient and Hessian

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

### Wiring into XGBoost

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

### Wiring into LightGBM

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

### Things to get right

- **Mathematical correctness.** Double-check the derivatives — an error here silently optimizes the wrong thing. Verify with simple cases or numerical gradient checks.
- **Non-negative Hessian.** The Hessian is the loss curvature and must be $\ge 0$ (strictly positive is preferred) so each step's objective is convex and well-defined; zero/negative Hessians make the algorithm unstable.
- **Numerical stability.** Avoid divide-by-zero and overflow/underflow; add a small epsilon in denominators where needed.
- **Classification predictions are raw scores.** For classification, `preds` are pre-sigmoid/softmax margins — compute grad/Hessian w.r.t. those raw scores (e.g. logistic: $g = p - y$, $h = p(1-p)$).
- **Keep eval consistent.** Pair the objective with a `feval` measuring the actual loss / business metric you care about.

*Source references: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016); Ke et al., "LightGBM" (NeurIPS 2017).*

---

## 7.6 Implementing Custom Evaluation Metrics

The final measure of success is often a domain-specific KPI or competition metric that the built-in objectives don't express. The evaluation metric guides hyperparameter tuning and early stopping. Unlike a custom **objective** (which must supply gradient info to drive training), a custom **evaluation metric** simply scores predictions vs. labels at each round — it does not affect the gradient, only monitoring and stopping decisions.

### Why custom metrics

- **Business KPIs** — e.g. minimizing large errors above a threshold, or ranking metrics like MAP@K / NDCG in recommenders.
- **Competition rules** — Kaggle often mandates metrics like **Quadratic Weighted Kappa (QWK)** or custom precision/recall variants; training/early-stopping on the competition metric helps even if a smoother loss is optimized.
- **Complex aspects** — fairness across subgroups, asymmetric error costs.
- **Monitoring vs. optimizing** — optimize log loss (nice math) but monitor F1 (user experience).

### Structure of a custom metric function

A typical metric receives predictions and the label-carrying data object, computes the score, and returns a tuple: `metric_name` (str), `metric_value` (float), and (in LightGBM/CatBoost) `is_higher_better` (bool). The `is_higher_better` flag is critical for early stopping to work correctly (True for AUC/F1, False for RMSE/MAPE).

### XGBoost

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

### LightGBM

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

### CatBoost

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

### Considerations

- **Efficiency.** Metrics run every round (and per CV fold) — vectorize with NumPy, avoid redundant work.
- **Correctness.** Unit-test the metric on known inputs before wiring into training; debugging inside the loop is painful.
- **`is_higher_better` / `is_max_optimal`.** Must be right or early stopping goes the wrong way.
- **Prediction format.** `preds`/`approxes` may be raw scores, probabilities, or final predictions depending on library/objective/API — apply sigmoid/softmax as needed.

*Source references: XGBoost "Custom Objective and Evaluation Metric" docs; Hastie, Tibshirani & Friedman, *ESL*; Manning et al. for MAP@K / NDCG.*

---

## 7.7 Handling Imbalanced Data with Boosting

Datasets are rarely perfectly balanced; the class of interest (fraud, disease, failure) is usually the minority. Standard GBMs can be biased toward the majority — achieving high overall accuracy by mostly predicting the majority while failing on the important minority class.

### Weighting the minority: `scale_pos_weight`

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

### Custom objectives for imbalance

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

### Imbalance-sensitive evaluation metrics

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

### Choosing a strategy

No single best method: start simple with `scale_pos_weight` / `is_unbalance`; ensure tuning and early stopping use an imbalance-appropriate metric (AUC-PR, F1); reach for custom objectives (Focal, weighted CE) when you need maximum control; and sampling methods like **SMOTE** (synthetic minority over-sampling) are another complementary axis. Always compare methods under a proper validation scheme with imbalance-sensitive metrics.

*Source references: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017); XGBoost `scale_pos_weight` docs; LightGBM `is_unbalance` docs; Chawla et al., "SMOTE" (2002).*

---

## 7.8 Hands-On: Custom Objectives & SHAP

This capstone exercise combines a custom objective with SHAP analysis: train one model with the standard objective and one with a custom asymmetric objective, then use SHAP to compare how the custom objective changes model behaviour.

### Scenario: asymmetric-cost regression

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

### Implementing the objective (factory pattern)

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

### Training standard vs. custom

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

### Explaining with SHAP

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

### Takeaway from the exercise

By penalizing under-prediction more heavily, the model's predictions align with the asymmetric cost structure, and SHAP makes the *how* transparent at both global and local levels — pairing custom objectives with interpretability yields models that are both fit-for-purpose and trustworthy.

---

## Key takeaways

- **SHAP unifies interpretability** via Shapley values with local accuracy $f(x) = \phi_0 + \sum_j \phi_j$, missingness, and consistency; it delivers both **global** (mean $|\phi_j|$ over instances) and **local** (per-instance $\phi_{ij}$) explanations from the same values.
- **TreeSHAP** makes exact Shapley values tractable for tree ensembles by pushing feature subsets down tree paths in polynomial time — orders of magnitude faster than Kernel SHAP; watch the feature-independence assumption under correlation.
- **Calibration** (Platt = parametric sigmoid, isotonic = non-parametric PAVA step function) fixes boosting's tendency to output non-probabilistic scores; fit it on **held-out** data (`CalibratedClassifierCV`), check with reliability diagrams / Brier score. Needed when probabilities (not just ranking) matter.
- **Custom objective contract:** supply per-instance gradient $g_i = \partial L/\partial \hat y_i$ and Hessian $h_i = \partial^2 L/\partial \hat y_i^2$, not the loss. Key forms: MSE $(g = 2(\hat y - y), h = 2)$; logistic on raw margins $(g = p - y, h = p(1-p))$; asymmetric/Huber via piecewise weights. Keep $h \ge 0$ and derivatives numerically stable.
- **Custom eval metrics** don't touch the gradient; return `(name, value[, is_higher_better])`. The higher/lower-better flag drives early stopping. XGBoost defaults to minimize (`maximize=True` to flip); LightGBM/CatBoost make direction explicit.
- **Imbalance toolkit:** `scale_pos_weight ≈ neg/pos` (or LightGBM `is_unbalance`), weighted / Focal-Loss custom objectives, sampling (SMOTE), and — crucially — imbalance-sensitive metrics (AUC-PR, F1, MCC) for tuning and early stopping.

> **Relevance to our work:** Custom objectives are our lever for aligning training with the exact competition metric — deriving grad/Hessian for approximations of **QWK** (ordinal targets), **SMAPE** (asymmetric percentage error), or Focal Loss for rare-class comps, then confirming with a matching `feval` and early stopping on the leaderboard metric. **SHAP/TreeSHAP** drives principled **feature pruning** (drop low mean-$|\phi|$ features to cut noise and speed CV) and interaction discovery, and lets us sanity-check that a model relies on sensible signals before trusting it. **Calibration** (isotonic/Platt via `CalibratedClassifierCV`, on out-of-fold predictions) is essential for **log-loss / Brier-scored** competitions and whenever we blend models — well-calibrated probabilities average far more coherently than raw scores.
