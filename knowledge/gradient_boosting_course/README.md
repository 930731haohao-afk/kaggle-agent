# Gradient Boosting — Course Study Notes

Synthesized English study notes from the course **"Mastering Gradient Boosting Algorithms"**
([apxml.com](https://apxml.com/courses/mastering-gradient-boosting-algorithms), Chinese edition),
Chapters 1–9. Each chapter file is a clean restatement of the source lessons with reconstructed
math (the source rendered formulas via KaTeX, which scraped as mangled text) and cleaned, idiomatic
Python. These are reference notes for this project's gradient-boosting work — they sit alongside the
distilled cross-competition [`experience.md`](../experience.md) and external
[`idea_bank.md`](../idea_bank.md).

## Contents

| # | File | Topic | What it covers |
|---|------|-------|----------------|
| 1 | [01_foundations.md](01_foundations.md) | Foundations revisited | Ensembles (bagging vs boosting), decision trees as base learners, the additive model $F_m = F_{m-1} + \nu h_m$, gradient descent in function space, the classic GBM algorithm. |
| 2 | [02_gbm_algorithm_in_depth.md](02_gbm_algorithm_in_depth.md) | GBM in depth | Functional gradient descent, generic GBM derivation, regression losses (L2/L1/Huber) and their pseudo-residuals, classification losses (deviance/exponential), shrinkage, stochastic subsampling, scikit-learn GBM. |
| 3 | [03_regularization.md](03_regularization.md) | Regularization | Overfitting in boosting, tree-structure constraints, shrinkage as implicit regularization, row/column subsampling, the regularized objective $\Omega(f)=\gamma T+\tfrac12\lambda\lVert w\rVert^2+\alpha\lVert w\rVert_1$, early stopping. |
| 4 | [04_xgboost.md](04_xgboost.md) | XGBoost | Second-order Taylor objective, optimal leaf weight $w_j^*=-G_j/(H_j+\lambda)$, exact-greedy vs approximate (weighted quantile sketch) split finding, sparsity-aware default directions, system optimizations, parameter reference. |
| 5 | [05_lightgbm.md](05_lightgbm.md) | LightGBM | GOSS, EFB, histogram-based splits, leaf-wise growth, native categorical handling, parameter reference + XGBoost↔LightGBM mapping. **(project's primary learner)** |
| 6 | [06_catboost.md](06_catboost.md) | CatBoost | Target leakage, Ordered Target Statistics, prediction shift & Ordered Boosting, feature combinations, oblivious trees, GPU training, parameter reference. |
| 7 | [07_advanced_customization.md](07_advanced_customization.md) | Advanced & customization | SHAP / TreeSHAP, global vs local explanations, probability calibration, custom loss (grad + hessian contract), custom eval metrics, imbalanced data. |
| 8 | [08_hyperparameter_optimization.md](08_hyperparameter_optimization.md) | Hyperparameter optimization | Which params matter, grid vs random search, Bayesian optimization (surrogate + acquisition), Optuna/Hyperopt, coarse-to-fine, CV strategy, a practical tuning recipe. |
| 9 | [09_specialized_tasks.md](09_specialized_tasks.md) | Specialized tasks | Learning-to-rank (pairwise/listwise, LambdaMART), survival analysis (Cox PH), quantile regression (pinball loss), multi-output boosting. |

## How these connect to the pipeline

- **Ch. 3, 5, 8** are the most operationally relevant: regularization knobs, LightGBM internals, and
  tuning strategy directly inform the automated tree-search / CV-gated experimentation here.
- **Ch. 7** (custom objectives + SHAP) matters for competition-specific metrics (QWK, SMAPE) and for
  feature pruning.
- **Ch. 9** (quantile / ranking objectives) is a reference for prediction-interval and ranking-style
  competitions.

> Notes are synthesized for study; formulas were reconstructed from the mangled scrape and should be
> spot-checked against primary sources (XGBoost, LightGBM, CatBoost papers) before being quoted verbatim.
