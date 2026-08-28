# External Idea Bank ([EXT])

> **Positioning of this bank**: these are **[EXT]** entries — sourced from **external literature** (recognized papers / DOIs, Kaggle winner or gold-medal write-ups, standard textbooks / official library documentation).
> Strictly separated from the **[INT]** entries in `knowledge/experience.md` (distilled from this project's **empirical runs across the s3/s4/s5 competitions, each with score-delta evidence**):
> - [INT] = "what we have **verified** in those competitions" (with `competition, exp #N, scoreA→scoreB`).
> - [EXT] = "techniques that the **literature/community broadly considers** effective, with clear mechanisms, usable as **priors** for tree-search candidate nodes" (not necessarily tested in this project).
>
> **Purpose**: for **merged injection into `suggest_priors` of the Phase J harness v4**. The current `suggest_priors` (harness_v2/v3) reads only `experience.md`, matching `##`/`###` headers against `comp_meta` (metric/tags/data_type/keywords) by lowercase substring and returning the bullet lines under each header; this bank deliberately follows the same structure — "headers containing matchable keywords (mae/auc/qwk/smape/rmse/small-sample/time series/categorical…) + bulleted bodies" — so v4 can **pool** [EXT] priors with [INT] experience via the **same mechanism** and feed them to the mutation proposer. When injecting, always keep the `[EXT-NN]` tag so the agent can distinguish "literature priors" from "in-competition evidence".
>
> **Citation integrity statement**: **the source of every entry in this bank was actually verified to exist via WebSearch / WebFetch** (2026-07-07). Academic entries were cross-checked against arXiv/DOI/journal pages; Kaggle pages are front-end rendered (WebFetch retrieves only the title), but every URL resolved successfully with a matching title, corroborated by multiple search results. **There are no "source pending" or fabricated entries whatsoever**. Inclusion criterion: better to omit than to fake — only entries with a clear mechanism and a clickable, verifiable source are admitted.
>
> Last updated: 2026-07-07. 24 entries. How to query: first look up by category header (mapped to metric/data type), then check the "Relation to the experience library" field to judge whether an entry **reinforces existing [INT]**, is a **new direction (not covered by the experience library)**, or is **known (be extra careful against [INT] counterexamples)**.

---

## A. Target encoding family (target/mean/count encoding, fold-safe, EB shrinkage)

### [EXT-01] Target encoding + empirical-Bayes smoothing (high-cardinality target/mean encoding with empirical-Bayes smoothing)
- **Source**: Micci-Barreca, D. (2001). "A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems." *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32. DOI: 10.1145/507533.507538. <https://dl.acm.org/doi/10.1145/507533.507538>
- **Mechanism**: replace a categorical column with that category's target mean, shrinking toward the parent-level/global prior via empirical Bayes `λ·cell_mean + (1−λ)·prior`, where `λ` rises with the within-group sample count (the fewer the observations, the heavier the shrinkage), suppressing the variance of rare high-cardinality categories.
- **Applicability**: high-cardinality categorical columns (categorical, high-cardinality); any metric, with the largest gains on string/ID-like columns that tree models struggle to split on their own; small samples need shrinkage even more.
- **Expected cost**: low (pure feature engineering, a single group aggregation).
- **Relation to the experience library**: **reinforces the theoretical basis**. [INT] already lists "fold-safe target encoding" as the **largest single gain** on group-structured data (s3e11 store_te, s3e20 te_locweek), and s3e20 exp #5 empirically tested the empirical-Bayes shrinkage `α·cell + (1−α)·loc` (α≈0.935, −0.159) — this entry is the original literature source of that shrinkage formula.

### [EXT-02] Fold-safe / leave-one-out target encoding + noise (out-of-fold TE, LOO with noise)
- **Source**: Owen Zhang, "Tips for Data Science Competitions" (SlideShare, deck by the #1 Kaggler; leave-one-out averaging + noise + GBM out-of-fold handling of high-cardinality columns). <https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions>; implementation reference: official docs for `category_encoders.LeaveOneOutEncoder` (with the `sigma` noise parameter) <https://contrib.scikit-learn.org/category_encoders/leaveoneout.html>
- **Mechanism**: when computing target means, **exclude the current row/current fold** (leave-one-out or K-fold out-of-fold), and add a small amount of Gaussian noise to the training feature, cutting off the overfitting path of "the target value leaking back into its own feature".
- **Applicability**: high-cardinality categoricals + target encoding; any metric. **The fold split must be exactly identical to the model-evaluation folds**.
- **Expected cost**: low.
- **Relation to the experience library**: **reinforces + (the noise variant is a) new direction**. [INT] uses fold-safe OOF encoding heavily and carries a painful lesson: **the target-encoding OOF folds must be exactly the same as the model training/evaluation folds** — independent folds with a different seed "seemingly avoid leakage" but are in fact another form of leakage (s4e1, source of score inflation). Owen Zhang's **noise** variant has not yet been tested in this project and can serve as a candidate node.

### [EXT-03] CatBoost ordered target statistics (ordered target statistics / ordered boosting)
- **Source**: Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS 2018*, 6639–6649. arXiv:1706.09516. <https://arxiv.org/abs/1706.09516>; NeurIPS paper page <https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html>
- **Mechanism**: builds a "history prefix" via random permutations to compute categorical target statistics (using only rows preceding the given sample), and uses the isomorphic ordered boosting to eliminate target leakage in gradient estimation (prediction shift), with no manual OOF encoding needed.
- **Applicability**: tabular data with categorical columns; pass raw strings directly as `cat_features` (do not label/freq pre-encode); naturally more robust to label noise.
- **Expected cost**: medium (trains slower than LGB; needs `allow_writing_files=False` to avoid sandbox file-write stalls).
- **Relation to the experience library**: **known + reinforces the mechanism**. [INT] has verified that "passing native `cat_features` to CatBoost greatly beats label/freq pre-encoding" (s3e3, 0.7627→0.8143) and that "ordered boosting is more robust to duplicate-row label noise" (s3e9, weight search gave CAT 100% under default params); it also drew the boundary — **on very small data (~1.7k rows) CatBoost is structurally weak and the weight search still gives it 0**. This entry is the original literature for its native mechanism.

### [EXT-04] Frequency / count encoding
- **Source**: official docs for `category_encoders.CountEncoder` (scikit-learn-contrib). <https://contrib.scikit-learn.org/category_encoders/count.html> (WebFetch confirmed: replaces category names with within-group occurrence counts; supports normalize and min-group merging)
- **Mechanism**: replaces a category with its occurrence frequency (or count); the frequency itself often carries signal (rare vs common), leaks no target information, and needs no fold-safe handling.
- **Applicability**: medium-to-high-cardinality categorical columns; **complementary** to target encoding (one encodes frequency, the other the target); any metric.
- **Expected cost**: low.
- **Relation to the experience library**: **new direction / supplement**. [INT] mentioned freq encoding only in s3e3 as CatBoost's **second-best** preprocessing (and it lost to native cat handling), and has never systematically validated "frequency encoding" as an independent feature-engineering lever — a low-cost candidate node, especially in combination with [EXT-01] target encoding.

---

## B. GBDT algorithms and hyperparameter tuning (Optuna)

### [EXT-05] XGBoost scalable gradient tree boosting
- **Source**: Chen, T., Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016*, 785–794. arXiv:1603.02754. <https://arxiv.org/abs/1603.02754>
- **Mechanism**: regularized boosted trees with a second-order Taylor approximation + sparsity-aware splits + weighted quantile sketch; built-in L1/L2 and tree-complexity regularization.
- **Applicability**: general tabular data; all metrics (custom objectives supported).
- **Expected cost**: low—medium (mature, fast).
- **Relation to the experience library**: **known (one of the three base models)**. [INT] rests throughout on the three-model LGB/XGB/CAT blend as its foundation; this entry is the original XGB literature, completing the prior's provenance.

### [EXT-06] LightGBM (GOSS + EFB + leaf-wise growth)
- **Source**: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 2017*. <https://www.semanticscholar.org/paper/LightGBM:-A-Highly-Efficient-Gradient-Boosting-Tree-Ke-Meng/497e4b08279d69513e4d2313a7fd9a55dfb73273>
- **Mechanism**: Gradient-based One-Side Sampling (keeps large-gradient samples) + Exclusive Feature Bundling (bundles mutually exclusive sparse features) + leaf-wise (best-first) growth; much faster and often more accurate than level-wise, but leaf-wise overfits easily and needs regularization via `num_leaves`/`min_child_samples`.
- **Applicability**: general tabular data; all metrics; the default workhorse for medium-to-large data.
- **Expected cost**: low (fastest).
- **Relation to the experience library**: **known (workhorse model)**. [INT] has LGB as the strongest single model in nearly every competition and has repeatedly verified that "small/medium data rewards **shallow, strongly regularized** LGB" (s3e3 num_leaves 7, s3e7 depth 3); this entry is the original literature.

### [EXT-07] Optuna hyperparameter optimization framework
- **Source**: Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD 2019*, 2623–2631. arXiv:1907.10902. <https://arxiv.org/abs/1907.10902>
- **Mechanism**: define-by-run dynamic search spaces + efficient sampling/pruning (default TPE sampler + median pruner), with support for distributed runs and early pruning.
- **Applicability**: any model that needs hyperparameter tuning; pair with "the final metric (or its proxy) as the objective function".
- **Expected cost**: medium (depends on trials × CV folds; a fold-0 proxy can cap the budget).
- **Relation to the experience library**: **known (core-recipe tool)**. [INT]'s durable recipe "**Optuna (fold-0 proxy or full CV) → add to the pool (not replace) → seed bagging**" has been verified across seven competitions: s3e1/s3e3/s3e7/s3e9/s3e11/s3e14/s3e19; this entry is the original Optuna literature.

### [EXT-08] TPE (Tree-structured Parzen Estimator) sampler
- **Source**: Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B. (2011). "Algorithms for Hyper-Parameter Optimization." *NeurIPS 2011*, 2546–2554. <https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>
- **Mechanism**: models two densities, `l(x)` (good outcomes) and `g(x)` (bad outcomes), and picks the next hyperparameter set by Expected Improvement ∝ `l(x)/g(x)` — sequential Bayesian optimization without Gaussian processes, suited to high-dimensional/conditional spaces.
- **Applicability**: same as [EXT-07]; Optuna's default sampler is TPE.
- **Expected cost**: low (the sampler itself has little overhead).
- **Relation to the experience library**: **reinforces the theory**. All of [INT]'s Optuna tuning actually runs TPE (e.g., s3e3 "50 trials TPE 111s"); this entry explains the underlying algorithm and notes that the cost judgment "full CV suffices on small data — no fold-0 proxy needed" has a theoretical basis.

---

## C. Ensembling: stacking / blending / rank-average / seed bagging

### [EXT-09] Stacked generalization (stacking)
- **Source**: Wolpert, D.H. (1992). "Stacked Generalization." *Neural Networks*, 5(2), 241–259. DOI: 10.1016/S0893-6080(05)80023-1. <https://dl.acm.org/doi/10.1016/S0893-6080%2805%2980023-1>
- **Mechanism**: train a meta model (second layer) on the base models' out-of-fold predictions used as **new features**, letting the meta learn how to combine the base models and correct their individual biases.
- **Applicability**: most effective when base models are numerous and heterogeneous; works for both classification and regression.
- **Expected cost**: medium—high (requires rigorous OOF; overfits easily with few bases).
- **Relation to the experience library**: **known + important [INT] counterexample**. [INT] measured that "**with only 3 bases, Ridge/non-negative stacking clearly loses to simplex grid weight search**" (s3e14, Ridge 344.04 vs blend 340.76); so stacking's applicability **presupposes a large base library** (see [EXT-10]) — with few bases this project prefers the weight search of [EXT-10].

### [EXT-10] Ensemble selection / hill-climbing weight search (greedy forward blend)
- **Source**: Caruana, R., Niculescu-Mizil, A., Crew, G., Ksikes, A. (2004). "Ensemble Selection from Libraries of Models." *ICML 2004*. DOI: 10.1145/1015330.1015432. <https://dl.acm.org/doi/10.1145/1015330.1015432>; author PDF <https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf>
- **Mechanism**: **greedy forward selection** from a model library (the same model may be picked repeatedly = effectively integer weights), each step adding the model that maximizes the OOF metric; can optimize any metric directly (AUC, logloss, MAE…) and uses "selection with replacement" to curb overfitting.
- **Applicability**: when multiple base-model OOFs exist and meta overfitting is a concern; any metric.
- **Expected cost**: low (searches only over OOF vectors).
- **Relation to the experience library**: **reinforces — this is the original literature for this project's main ensembling method**. [INT] uses "simplex/grid OOF weight search" throughout, and it often eliminates weak models outright (weight=0: s3e3 CAT, s3e11 XGB, s3e20 all three GBDTs at 0) while beating Ridge/isotonic metas — precisely the spirit of Caruana's ensemble selection. This entry is its academic foundation.

### [EXT-11] Out-of-fold stacking + weighted blending (Kaggle practice recipes)
- **Source**: van Veen, H.J. et al. (MLWave), "Kaggle Ensembling Guide" (2015). Code <https://github.com/MLWave/Kaggle-Ensemble-Guide> (WebFetch confirmed it contains voting/averaging/rankavg/geomean); Kaggle mirror <https://www.kaggle.com/discussions/getting-started/106655>
- **Mechanism**: a community-curated practical ensembling handbook — simple averaging, weighted averaging, multi-layer stacked generalization + OOF, geometric mean, etc.; emphasizes that "OOF computed over more folds is more stable than a single holdout".
- **Applicability**: all metrics and data types; a decision map for "simple averaging first, then stacking as warranted".
- **Expected cost**: low—medium.
- **Relation to the experience library**: **known**. [INT]'s findings that "a blend reliably beats all single models only when the models are of comparable strength" (s3e1) and that "an unsearched 50/50 equal weighting can lose to the best single model" (s3e20 33.21>32.75) both agree with this guide; this entry supplies the community source and a fuller menu of methods.

### [EXT-12] Rank averaging (rank-average ensembling)
- **Source**: same "Kaggle Ensembling Guide" (`kaggle_rankavg.py`). <https://github.com/MLWave/Kaggle-Ensemble-Guide>
- **Mechanism**: convert each model's predictions to **ranks** before averaging — when models are uncalibrated or differ in output scale/distribution, averaging in rank space is more robust than averaging in probability space.
- **Applicability**: **ranking-type metrics (AUC)**, or when members differ greatly in calibration; not advisable for metrics that strictly score probability values (logloss).
- **Expected cost**: low.
- **Relation to the experience library**: **known, and [INT] measured it at noise level (in that competition)**. [INT] compared prob-space vs rank-average in s3e7 (AUC); the difference was only ±0.000002~0.000007 (noise level) — showing rank-average is **not a universal gain**; its value depends on whether members genuinely differ in scale. This entry delimits its applicability conditions.

### [EXT-13] Bagging / seed bagging (bootstrap aggregation, variance reduction via seed bagging)
- **Source**: Breiman, L. (1996). "Bagging Predictors." *Machine Learning*, 24(2), 123–140. DOI: 10.1023/A:1018054314350. <https://doi.org/10.1023/A:1018054314350>
- **Mechanism**: average multiple high-variance models (bootstrap samples / different random seeds), reducing variance while leaving bias almost untouched; gains are largest for unstable learners (deep trees).
- **Applicability**: when models still carry substantial random variance; any metric. **The higher the member variance, the more effective it is**.
- **Expected cost**: low—medium (linear in the number of members).
- **Relation to the experience library**: **reinforces — seed bagging is [INT]'s key residual gain**. [INT] repeatedly verified that "seed bagging with the same hyperparameters but varied `random_state` is the cheapest post-tuning gain" (positive in s3e14/s3e9/s3e19), and found its **boundary**: seed-bagging **may be ineffective** for models where "Optuna already optimized the final metric directly and converged to a shallow, stable solution" (s3e5) — exactly confirming this entry's variance-reduction mechanism: "low member variance means nothing to gain".

---

## D. Validation design: adversarial validation / nested CV / time series splits

### [EXT-14] Adversarial validation (detecting train/test distribution shift)
- **Source**: Zając, Z. "Adversarial validation, part one / part two" (FastML, 2016-05). <http://fastml.com/adversarial-validation-part-one/> (WebFetch confirmed: author Zygmunt Zając, demonstrated on the Santander competition); overview at KDnuggets <https://www.kdnuggets.com/2020/02/adversarial-validation-overview.html>; see also *The Kaggle Book*.
- **Mechanism**: concatenate train/test, label them 0/1, and train a binary classifier to tell them apart. AUC≈0.5 → distributions match; AUC significantly >0.5 → shift exists, and the classifier can score training rows by "test-likeness" to pick the most test-like subset as the validation set, or to find and treat the columns causing the shift.
- **Applicability**: when train/test distributions are suspected to differ, or CV↔LB are decoupled; any metric.
- **Expected cost**: low (one extra classifier to train).
- **Relation to the experience library**: **new direction (not covered by the experience library)**. [INT] repeatedly stresses that "the CV scheme must match the test scenario" (s3e19 TimeSeriesSplit, s3e20 LOYO), but has **never run adversarial validation** to quantify shift — a clean new candidate node, especially suited for the next-stage harness to trigger when the CV↔LB gap is anomalous.

### [EXT-15] Nested cross-validation (avoiding selection bias)
- **Source**: Cawley, G.C., Talbot, N.L.C. (2010). "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation." *JMLR*, 11, 2079–2107. <https://www.jmlr.org/papers/v11/cawley10a.html>; Varma, S., Simon, R. (2006). "Bias in error estimation when using cross-validation for model selection." *BMC Bioinformatics*, 7:91. DOI: 10.1186/1471-2105-7-91. <https://doi.org/10.1186/1471-2105-7-91>
- **Mechanism**: outer loop estimates performance, inner loop selects models/tunes hyperparameters — putting "model selection" inside cross-validation as well, avoiding the optimistic bias of "tuning and reporting on the same CV" (that bias can be sizable).
- **Applicability**: when an **honest estimate** of post-tuning generalization is needed, or when post-processing thresholds risk overfitting the OOF; especially needed for small samples.
- **Expected cost**: medium—high (outer × inner fold counts multiply).
- **Relation to the experience library**: **reinforces — [INT] already applies its spirit**. [INT] used "nested (leave-fold-out) threshold fitting" to diagnose OptimizedRounder's overfitting to the full OOF, and found the risk shrinks as the ensemble pool matures (s3e5 gap +0.0164→+0.0038); it also honestly notes that "fold-5 is both the tuning target and 1/5 of the OOF, so the score is partly optimistic" (s3e19) — this entry is the original theoretical literature for this class of bias.

### [EXT-16] Purged + embargoed time series splits (purged & embargoed K-fold CV)
- **Source**: López de Prado, M. (2018). *Advances in Financial Machine Learning*, Wiley, Chapter 7 (Cross-Validation in Finance). ISBN 978-1-119-48208-6. Free summary: <https://en.wikipedia.org/wiki/Purged_cross-validation>
- **Mechanism**: when sample labels overlap in time, (a) **purge**: remove from the training folds any rows whose label time window overlaps the validation fold; (b) **embargo**: additionally keep a small buffer after the validation fold out of training — together blocking temporal leakage and preventing overestimation from trend extrapolation.
- **Applicability**: time series / panel data, labels spanning time intervals, test set in a future period (SMAPE/time series); stricter than plain TimeSeriesSplit.
- **Expected cost**: medium (requires label time-interval information).
- **Relation to the experience library**: **reinforces + new variant**. [INT] already uses time-aware CV heavily: "test purely in a future period → TimeSeriesSplit (split on unique dates)" (s3e19, random KFold overestimates by 2.4×) and "multi-year spatio-temporal → Leave-One-Year-Out" (s3e20). **The stricter purge/embargo variant is untried in this project** and is a candidate upgrade for time-series competitions.

---

## E. Feature engineering: interactions / aggregations / high-cardinality representations

### [EXT-17] Deep Feature Synthesis / automated aggregation features (aggregation across relations)
- **Source**: Kanter, J.M., Veeramachaneni, K. (2015). "Deep Feature Synthesis: Towards automating data science endeavors." *IEEE DSAA 2015*, 1–10. (Evolved into the open-source Featuretools.) Library <https://pypi.org/project/featuretools/>
- **Mechanism**: automatically stacks aggregation primitives (count/mean/std/max/min/mode…) and transform primitives along entity relations, systematically generating group-statistic/cross-table aggregation features in place of manual enumeration.
- **Applicability**: data with group keys, multi-table / transactional data; mixed numeric + categorical.
- **Expected cost**: medium (feature explosion requires paired feature selection).
- **Relation to the experience library**: **new direction, but must be checked against [INT] counterexamples**. [INT] verified that "group **target** encoding (TE) is the largest gain" (s3e11 store_te), yet also found that "**once TE exists, adding non-target feature means on the same group key → across-the-board regression, revert**" (s3e11 exp #7, the group target signal was already exhausted by TE) and that "coarse-grained group aggregates are redundant for trees that can already split on their own" (s3e1). So automated aggregation must avoid colliding with existing TE keys and be strictly adjudicated by importance/OOF.

### [EXT-18] Entity embeddings (neural embeddings for high-cardinality categoricals)
- **Source**: Guo, C., Berkhahn, F. (2016). "Entity Embeddings of Categorical Variables." arXiv:1604.06737 (3rd-place solution for Rossmann Store Sales). <https://arxiv.org/abs/1604.06737>
- **Mechanism**: a neural network learns each category value as a low-dimensional continuous vector, placing semantically similar categories near each other in embedding space; the embeddings can be fed back to GBDTs or used directly inside the NN, particularly effective and overfitting-resistant for high cardinality.
- **Applicability**: **high-cardinality categoricals** with enough data to train an NN; the numeric embeddings can also be fed back to tree models.
- **Expected cost**: high (requires training an NN and tuning embedding dimensions).
- **Relation to the experience library**: **new direction (NN-based; [INT] is a pure-GBDT foundation)**. [INT] currently handles categoricals entirely with GBDT + target/frequency encoding and has never used learned embeddings; this entry is a "new direction" candidate for high-cardinality categoricals, at higher cost, suited to competitions already squeezed by TE/GBDT that still seek a breakthrough.

---

## F. Post-processing: probability calibration / threshold optimization

### [EXT-19] Probability calibration: Platt scaling (sigmoid calibration)
- **Source**: Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." *Advances in Large Margin Classifiers*, MIT Press. For implementation and CV-based calibration see the scikit-learn `CalibratedClassifierCV` official docs <https://scikit-learn.org/stable/modules/calibration.html> (WebFetch confirmed: sigmoid/isotonic + cross-validation to avoid bias); concept page <https://en.wikipedia.org/wiki/Platt_scaling>
- **Mechanism**: pass the model score `f` through a sigmoid `1/(1+exp(Af+B))` fitted by logistic regression to obtain calibrated probabilities; few parameters, overfitting-resistant, suited to small calibration datasets.
- **Applicability**: **classification + metrics that score probability values** (logloss, Brier); boosted trees, which often show sigmoid-shaped distortion, benefit especially. **The calibrator must be fitted on independent folds/CV**.
- **Expected cost**: low.
- **Relation to the experience library**: **new direction (classification probability calibration) + note the [INT] regression counterexample**. [INT]'s post-processing so far centers on regression/ordinal targets (rounding, snap, OptimizedRounder) and has **never done classification probability calibration**; but there is a related warning: isotonic on an already-converged blend (see [EXT-20]) was strongly negative on **regression MAE** (s3e14). Platt calibration is a candidate for classification + logloss competitions; do not apply it where the regression median is already well calibrated.

### [EXT-20] Probability calibration: isotonic regression
- **Source**: Zadrozny, B., Elkan, C. (2002). "Transforming classifier scores into accurate multiclass probability estimates." *KDD 2002*, 694–699. DOI: 10.1145/775047.775151. <https://dl.acm.org/doi/10.1145/775047.775151>; comparative study Niculescu-Mizil, A., Caruana, R. (2005). "Predicting good probabilities with supervised learning." *ICML 2005*. <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>
- **Mechanism**: fits a **monotone non-decreasing** piecewise-constant function mapping scores to probabilities; nonparametric and more flexible than Platt, but needs more calibration data and overfits easily on little data.
- **Applicability**: classification + probability metrics with ample calibration data; Niculescu-Mizil & Caruana (2005) empirically showed boosted trees/SVMs improve significantly after calibration.
- **Expected cost**: low—medium.
- **Relation to the experience library**: **known counterexample + clarified applicability**. [INT] applied nested isotonic (MAE objective) to an already-converged **GBDT regression blend** and got a **strongly negative** result (s3e14 exp #6, 346.997 vs 340.699, auto-vetoed) — because the blend was already well calibrated in the median sense. This entry clarifies: isotonic's proper battlefield is **classification probabilities** (logloss/Brier), not regression point estimates.

### [EXT-21] OptimizedRounder — QWK threshold optimization (Nelder-Mead threshold search)
- **Source**: post-processing method popularized by the Kaggle "PetFinder.my Adoption Prediction" (2019) competition (regression output + threshold search via `scipy.optimize` (Nelder-Mead) to maximize QWK). Competition page <https://www.kaggle.com/c/petfinder-adoption-prediction>; kernel explaining the QWK metric and thresholds <https://www.kaggle.com/code/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps>
- **Mechanism**: for ordinal targets, use a **regression head** to obtain continuous scores, then gradient-free optimization (Nelder-Mead) to search K−1 thresholds that cut the continuous scores into ordinal classes, directly maximizing QWK on train/OOF, replacing symmetric rounding.
- **Applicability**: **ordinal targets + QWK**; especially critical under class imbalance (the symmetric-rounding assumption collapses).
- **Expected cost**: low (threshold search is fast).
- **Relation to the experience library**: **known — heavily verified by [INT]; this entry adds the external original source**. [INT] lists "regression + OptimizedRounder (tuning thresholds on OOF QWK)" as the **single highest lever on small-data ordinal problems** (s3e5, naive round 0.4719→optimized thresholds 0.5269, +0.055), and further found that "setting the Optuna objective function directly to the post-processed QWK" is even stronger. This entry gives the technique's community origin (PetFinder) and the Nelder-Mead implementation.

---

## G. Feature importance / champion playbooks / general references

### [EXT-22] Permutation feature importance (robust pruning basis)
- **Source**: Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324 (source of permutation importance). <https://link.springer.com/article/10.1023/A:1010933404324>; method overview <https://christophm.github.io/interpretable-ml-book/feature-importance.html>
- **Mechanism**: after training, **randomly shuffle a single feature** and measure how much the validation metric degrades as that feature's importance; does not depend on the model's internal split counts, so it is less misled by high-cardinality/collinear columns than GBDT built-in gain/split importance.
- **Applicability**: feature selection / pruning; especially needed with highly collinear or high-cardinality columns (built-in importance is biased).
- **Expected cost**: low—medium (one validation re-scoring per feature).
- **Relation to the experience library**: **reinforces — [INT] already prunes by importance but with the built-in version**. [INT]'s "too many features regress → trimming is a standard weapon" prunes via importance probes (s3e14 pruned to 21 features at 340.76; s3e7 pruning noise features overtook) — permutation importance is a more bias-resistant pruning basis and can serve as an upgraded variant of the existing pruning step, especially in near-perfectly collinear (r≥0.999) settings like s3e14.

### [EXT-23] Denoising-autoencoder representation learning (denoising autoencoder, championship-grade tabular solution)
- **Source**: Jahrer, M. (2017). "1st place solution — Porto Seguro's Safe Driver Prediction (Representation Learning)." Kaggle write-up. <https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/michael-jahrer-1st-place-with-representation-learn> (WebFetch confirmed the page title "1st place with representation learning")
- **Mechanism**: add noise to numeric features and train a denoising autoencoder to learn a better **representation** (hidden-layer activations), then feed that representation to downstream NNs (the winning solution blended 1×LGB + 5×NN); it won one of the rare tabular competitions where XGBoost did not take first place.
- **Applicability**: purely numeric / already-encoded tables, large datasets, willingness to invest in NNs; usually needs blending with GBDTs.
- **Expected cost**: high (training a DAE + multiple NNs, heavy tuning).
- **Relation to the experience library**: **new direction (representation learning; [INT] is pure GBDT)**. [INT] has no autoencoder/representation-learning entries yet; this is the most famous tabular precedent of "GBDT wasn't enough — representation learning turned the tables", listed as a high-cost, high-risk breakthrough candidate, suited to competitions where both the experience library and the tree-search layer have been squeezed dry and a systematic gain is still sought.

### [EXT-24] The Kaggle Book — comprehensive tabular-competition handbook (cross-index)
- **Source**: Banachewicz, K., Massaron, L. (2022). *The Kaggle Book: Data analysis and machine learning for competitive data science.* Packt. ISBN 9781801812214. <https://www.packtpub.com/en-us/product/the-kaggle-book-9781801812214>
- **Mechanism**: competition methodology systematically organized by two Kaggle Grandmasters — covering validation design, adversarial validation, feature engineering, ensembling/stacking, hyperparameter tuning, and recipes per data type (tabular/image/text), with Master interviews from various competitions.
- **Applicability**: serves as a secondary cross-index and situational-judgment guide for entries such as [EXT-09~14] and [EXT-01~04]; does not replace the original papers.
- **Expected cost**: — (reference material).
- **Relation to the experience library**: **general reinforcement**. The book integrates many of the above [EXT] techniques (adversarial validation, target encoding, ensembling, the denoising-autoencoder case) into actionable workflows, serving as a human-readable reference whenever the harness needs a "which move to use" situational judgment; the difference from [INT] is that [INT] is **this project's measured score deltas** while the book is **community-consensus recipes**.

---

## Appendix: overlap / complementarity with [INT] at a glance (for v4 pool-merge deduplication)

- **Priors already evidenced by [INT] (inject alongside the [INT] evidence to avoid duplicate triggering)**: EXT-01 (EB shrinkage=s3e20), EXT-02 (fold-safe=multiple competitions), EXT-03 (CatBoost native=s3e3/s3e9), EXT-05/06 (LGB/XGB=foundation), EXT-07/08 (Optuna/TPE=core recipe), EXT-10 (weight search=workhorse), EXT-13 (seed bagging=key residual gain), EXT-15 (nested=threshold diagnostics), EXT-16 (time series splits=s3e19/s3e20), EXT-21 (OptimizedRounder=s3e5), EXT-22 (importance pruning=s3e14).
- **[INT] has counterexamples; [EXT] delimits the correct applicability (always inject together with the counterexamples)**: EXT-09 (stacking: with few bases loses to weight search s3e14), EXT-11/12 (equal weights/rank-average not universal s3e20/s3e7), EXT-17 (non-target group aggregates redundant once TE exists s3e11), EXT-20 (isotonic backfires on regression MAE s3e14).
- **Clean new directions not yet covered by the experience library (the most valuable candidate nodes)**: **EXT-14 adversarial validation**, **EXT-04 frequency encoding (independent lever)**, **EXT-16 purge/embargo (time series upgrade)**, **EXT-18 entity embeddings**, **EXT-19 classification probability calibration (Platt)**, **EXT-23 denoising autoencoder**.
