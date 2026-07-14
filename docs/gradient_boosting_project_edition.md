# Gradient Boosting — Summer Project Edition

> **What this is.** A curated cut of the [consolidated course notes](gradient_boosting_course.md)
> that keeps only the concepts this project actually used, and pins each one to the evidence in
> [`knowledge/experience.md`](../knowledge/experience.md) — the distilled, score-backed record of what
> we validated across the Playground-Series runs (s3e1 … s6e2). Every "**In our project**" claim below
> carries a `competition, exp #N, scoreA→scoreB` citation, matching the `experience.md` house rule.
>
> Course-section links point into `gradient_boosting_course.md`; evidence links point into
> `experience.md`. This is a **used-concepts-only** cut — every entry is a technique or model the project
> actually ran. For the full theory (including topics we didn't use), see the consolidated course notes.

## Coverage at a glance

| Course chapter         |   Role    | Where it shows up in the project                                                                        |
| ---------------------- | :-------: | ------------------------------------------------------------------------------------------------------- |
| 1 — Foundations        |  ◑ light  | Additive model & functional gradient descent as the working mental model.                               |
| 2 — GBM in depth       |  ● used   | Loss choice per metric (log1p for RMSLE/skew), shrinkage, subsampling.                                   |
| 3 — Regularization     |  ●● core  | The recurring **"shallow + strongly-regularized"** winner; early stopping; row/col subsampling.         |
| 4 — XGBoost            |  ● used   | Third blend member; rarely the strongest, sometimes weight-searched to 0.                               |
| 5 — LightGBM           |  ●● core  | Primary learner in every run; determinism gate; `num_leaves`/`min_data_in_leaf` control.                |
| 6 — CatBoost           |  ● used   | Blend member; native `cat_features`; structurally weak on tiny data (validated).                        |
| 7 — Advanced           | ● used | Feature-importance pruning; isotonic calibration (tried, rejected); "custom metric" = Optuna-on-final-metric. |
| 8 — Hyperparameter opt |  ●● core  | Optuna is our tuning workhorse — fold-0 proxy, direct-optimize-the-metric, seed bagging.                |

Legend: ●● core lever · ● used · ◑ light.

---

## 1. Models we actually ran

Three GBDT libraries, always blended by an OOF simplex weight search; AutoGluon available but the
tree trio carried the runs.

- **LightGBM — primary learner** (course [Ch. 5](gradient_boosting_course.md#chapter-5--lightgbm-light-gradient-boosting-machine)). Strongest single model in nearly every competition and the base we tune first.
- **XGBoost — blend member** (course [Ch. 4](gradient_boosting_course.md#chapter-4--xgboost-extreme-gradient-boosting)). Useful for diversity, but frequently weight-searched to 0. | Evidence: s3e11 XGB weight 0 twice; s3e19 (extrapolation) XGB weight 0 twice.
- **CatBoost — blend member** (course [Ch. 6](gradient_boosting_course.md#chapter-6--catboost-ordered-boosting-for-categorical-data)). Wins on label-noise data via ordered boosting; structurally weak on very small tabular data. | Evidence: s3e9, exp #1 CatBoost gets 100% blend weight under label noise (12.54287); s3e3 (1,677 rows) CAT 0.762–0.814 vs LGB 0.837, weight 0 even with native `cat_features`.

> **Durable rule we learned:** don't assume a 3-model blend always helps — read each model's OOF first;
> the weight search will (correctly) zero out a model that doesn't earn its place.

---

## 2. Loss functions & the metric-first mindset (Ch. 2)

Course refs: [2.3 regression losses](gradient_boosting_course.md#23-common-loss-functions-for-regression),
[2.4 classification losses](gradient_boosting_course.md#24-common-loss-functions-for-classification).
The single most transferable habit: **optimize the objective that matches the competition metric**, and
handle the metric's quirks in pre/post-processing rather than fighting the loss.

- **RMSLE ⇒ train on `log1p(y)` with an RMSE objective, predict `expm1`, clip ≥ 0.** | Evidence: s3e11, exp #2/#3 (methodology; the transform is the correct-objective prerequisite even when its own delta is tiny, −0.00013).
- **Heavily right-skewed regression targets ⇒ `log1p`.** Linear models can explode here; GBDT stays robust. | Evidence: s3e20, exp #1, Ridge RMSE 23,917.63 vs LGB 32.75 (mean baseline 155.54); s3e19 skew 1.75 → log1p makes the distribution near-symmetric.
- **Ordinal target ⇒ regression head, not multiclass.** QWK penalizes squared distance, so a continuous output borrows signal from middle classes for rare extremes. | Evidence: s3e5, final 0.52687→0.567 achieved with a regression head; a `class_weight=balanced` multiclass member contributed 0 blend weight.

> **In our project:** the loss is chosen by the metric, then the metric's discreteness (rounding, snapping)
> is a *post-processing* decision (see §6), and CV honesty (see §5) matters more than the loss subtlety.

---

## 3. Regularization — our most reliable lever (Ch. 3)

Course refs: [3.2 tree constraints](gradient_boosting_course.md#32-tree-structure-constraints-depth-nodes-and-splits),
[3.3 shrinkage](gradient_boosting_course.md#33-shrinkage-as-implicit-regularization),
[3.4 subsampling](gradient_boosting_course.md#34-data-subsampling-stochastic-gradient-boosting),
[3.6 early stopping](gradient_boosting_course.md#36-early-stopping-strategies).

The recurring finding across small/medium tabular data: **reward regularization, not capacity.** The best
configs are consistently *shallow with strong regularization*.

- **Small data + correlated engineered features ⇒ tighten regularization before adding capacity.** | Evidence: s3e3, exp #2→#3, removing imbalance weighting + `num_leaves 15→7, L1 0.5→1.0, L2 1.0→2.0` gave AUC 0.81901→0.83292 (+0.017).
- **Label-noise ceiling ⇒ penalize capacity, prefer CatBoost's ordered boosting; regularize LGB/XGB hard.** | Evidence: s3e9, exp #2, regularized LGB (`leaves15/depth5/L1=2/L2=4`) 13.21→12.11; defaults badly overfit.
- **Extrapolation rewards simpler trees too.** | Evidence: s3e19, exp #7, tuning drove `num_leaves 63→20, min_child_samples 20→38`.
- **The tuned optimum is repeatedly "shallow + regularized."** | Evidence: s3e7 depth 3, lr 0.068, reg_alpha 2.14; s3e3 exp #4 `num_leaves 7→3`.

> **Boundary condition (validated):** capacity/regularization needs *reverse* with data size — on 360k rows
> the optimum flips to *deeper + higher lr* (s3e11: depth 10, lr 0.082). "Shallow+strong-reg" is a
> small/medium-data prior, not a universal law.

The parameters our tree-search actually tunes: `learning_rate, num_leaves, max_depth, min_child_samples`
(LGB) / `min_data_in_leaf`, `reg_alpha, reg_lambda` (LGB) / `l2_leaf_reg, random_strength` (CatBoost),
`subsample, colsample_bytree`.

---

## 4. LightGBM specifics we relied on (Ch. 5)

Course ref: [Ch. 5](gradient_boosting_course.md#chapter-5--lightgbm-light-gradient-boosting-machine).
We used LightGBM as a well-tuned black box — the internals that mattered operationally:

- **`num_leaves` / `min_data_in_leaf` are the leaf-wise overfit controls** (course [5.5](gradient_boosting_course.md#55-leaf-wise-best-first-tree-growth)). These are the first knobs the search moves.
- **Native categorical handling** (course [5.6](gradient_boosting_course.md#56-optimized-categorical-feature-handling)): pass raw categories via `cat_features`/`categorical_feature` rather than pre-encoding. | Evidence: s3e3, exp #6, native handling lifted CatBoost 0.7627→0.8143 (encoding was part of the earlier gap).
- **Determinism gate — a hard-won engineering fix.** LightGBM's default timing-dependent histogram choice makes cross-process results differ slightly, breaking bit-exact OOF reproduction. Fix: `deterministic=True, force_row_wise=True, num_threads=<fixed>` (and the XGB/CatBoost equivalents + fixed nthread) → `max|dOOF| = 0`. | Evidence: s6e1/s6e2, `eval_s6e1.py`/`eval_s6e2.py` (first seen as max 0.87/sample drift on s6e1).

> LightGBM's speed comes from GOSS, EFB, and histogram binning (course [5.2–5.4](gradient_boosting_course.md#52-gradient-based-one-side-sampling-goss)) — the reason it's our default primary model.

---

## 5. Cross-validation design (Ch. 2/3/8 in the course; our biggest correctness lever)

The course threads CV through several chapters; in practice this is where most of our score *integrity*
lives. Course refs: [3.6 early stopping](gradient_boosting_course.md#36-early-stopping-strategies),
[8.7 CV strategy for tuning](gradient_boosting_course.md#87-cross-validation-strategy-for-honest-tuning).

- **Never compare scores across CV schemes; log the scheme with every experiment.** A "worse" score after switching schemes is often just a more honest one. | Evidence: s3e19, exp #2 (TimeSeriesSplit 10.18) looked worse than baseline (5.32); diagnostic exp #3 proved the same config scored 4.28 under the old optimistic scheme.
- **Future-period test ⇒ `TimeSeriesSplit` on unique dates**, and submit with weights from *that* scheme. Random KFold interpolates and is wildly optimistic. | Evidence: s3e19, exp #2 vs #3, SMAPE 10.175 vs 4.281 — a 2.4× gap from CV scheme alone.
- **Multi-year spatio-temporal ⇒ Leave-One-Year-Out**, which also supports leakage-free target encoding. | Evidence: s3e20, exp #3/#4.
- **Imbalanced ordinal/classification ⇒ StratifiedKFold on the label** so no fold has zero rare-class rows. | Evidence: s3e5 (69.9× imbalance), per-fold std 0.033.
- **No time/group structure ⇒ plain shuffle KFold; don't over-engineer.** | Evidence: s3e1, s3e11, s3e14.
- **Target-encoding folds must be identical to model-evaluation folds.** Using a different seed's independent folds "to avoid leakage" is itself a leakage path. | Evidence: s4e1, old baseline inflation traced to surname TE computed on seed=99 folds ≠ model's seed=42 folds.
- **CV↔LB gap is real and trackable** (~0.003–0.006, LB slightly worse than OOF, stable across submissions), and **tree-search OOF gains do transfer to the real leaderboard**. | Evidence: s3e16, exp #5, tree-search v2 node #15 OOF 1.33812→1.33563 → Private LB 1.34075→1.33859 (both boards improved).

---

## 6. Ensembling & post-processing (Ch. 7 calibration; project-specific blend)

The course's blend/calibration coverage is thin; this is mostly our own validated recipe. Course ref:
[7.4 probability calibration](gradient_boosting_course.md#74-probability-calibration-for-classification).

- **Blend by OOF simplex weight search; equal-weight is often worse than the best single model.** | Evidence: s3e20, exp #2, 50/50 LGB+XGB = 33.21 > LGB solo 32.75.
- **Simplex grid beats learned stacking on ~3 base models.** Ridge stacking and isotonic recalibration both lost. | Evidence: s3e14, exp #3 Ridge stack 344.04 vs simplex 340.76; exp #6 isotonic 346.99 vs blend 340.70 (auto-rejected).
- **Post-processing trio, all validated — and it's the *metric*, not the target's discrete look, that decides:**
  - **Round to integer** when the metric is MAE on integer targets. | Evidence: s3e16, exp #1, OOF MAE 1.35651→1.33885.
  - **`OptimizedRounder`** (cut-points tuned on OOF QWK) for ordinal targets — the single highest-leverage change on small ordinal data. | Evidence: s3e5, exp #2→#3, QWK 0.47191→0.52687 (+0.055).
  - **Snap-to-nearest-observed** for discrete non-integer target grids under MAE. | Evidence: s3e14, exp #2/#3.
  - **Counter-example:** do **not** snap-to-grid under **RMSE** — squared error wants the conditional mean. | Evidence: s5e10, snap 0.056095 > clip 0.056027.
- **`clip` to the training target range as a safety net.** | Evidence: s3e1, s3e11.

---

## 7. Hyperparameter optimization with Optuna (Ch. 8) — our tuning workhorse

Course refs: [8.4 Bayesian optimization](gradient_boosting_course.md#84-advanced-tuning-bayesian-optimization),
[8.5 Optuna/Hyperopt](gradient_boosting_course.md#85-hpo-frameworks-optuna-and-hyperopt),
[8.6 coarse-to-fine](gradient_boosting_course.md#86-coarse-to-fine-tuning-strategy).
This is the most heavily-exercised course chapter in the project.

- **Tune only the strongest single model** — high ROI, and don't re-run the same recipe on a second base model (they converge to similar shallow optima and kill blend diversity). | Evidence: s3e7, exp #3→#4 LGB +0.0004 solo/+0.0005 blend; exp #5, tuning XGB the same way *hurt* the blend (0.899891→0.899722).
- **Budget control via a fold-0 proxy objective**, then re-run full 5-fold only on the winner — the proxy ranking transfers. | Evidence: s3e7, full 5-fold timed out at 25 min; fold-0 proxy 50 trials in 300s got the gain. s3e1 fold-0 proxy 50 trials in 74.3s.
- **Tiny data (~2k rows) ⇒ skip the proxy; full 5-fold Optuna is already cheap** (40 trials in 26–46s), so optimize the *final post-processed metric directly*. | Evidence: s3e5, exp #5 (LGB full CV 40 trials, 26s) — direct QWK-after-rounder objective, 0.52986→0.56293 (+0.033).
- **Directly optimizing the final metric transfers beyond QWK** — same recipe lifted AUC with no discretization trap. | Evidence: s3e3, exp #4, 0.832925→0.837305.
- **After tuning: add the tuned model to the pool (don't replace), then seed-bag.** Replacing a diverse member loses diversity; a fresh `random_state` on the tuned config is the cheapest residual gain. | Evidence: s3e14, exp #4 (replace) 340.95 vs exp #5 (add) 340.63; exp #7 seed-bag → 340.599.
- **Seed bagging is not universal:** it's worthless on models Optuna already drove to a shallow/stable optimum (low variance), but valuable near a label-noise ceiling (high variance). | Evidence: s3e5 exp #6 tuned-LGB seed-bag 0 gain; vs s3e9 exp #6→#8 seed-bag was the *only* working lever (12.07347→12.07003).

> **The transferable pipeline** (validated on s3e7 / s3e14 / s3e1 / s3e11, across RMSE scales and 2k→360k rows):
> **Optuna fold-proxy tune → add to pool (don't replace) → seed-bag.** Treat it as the default opening move
> for medium tabular GBDT blends; and once a metric needs rounding, re-validate every tuning decision on the
> *rounded* score, not the raw one (s3e16 counter-example).

---

## 8. Feature selection

- We used **feature-importance–driven pruning**: trimming near-collinear or noisy features
  reliably helped. | Evidence: s3e14, exp #2→#3, 27→21 features, 341.02→340.76; s3e7, +14 features 0.89788 < baseline, prune → 0.89939.
- **Trees learn multiplicative interactions themselves** — explicit product features usually just add
  collinearity on small noisy data (but re-check per competition; s6e1 was a counter-example). | Evidence: s3e9, exp #3, +3 interactions 12.07347→12.09483, reverted; s6e1 Δ+0.000094 (kept).

---

### Source & maintenance

Course theory: [`gradient_boosting_course.md`](gradient_boosting_course.md) (apxml.com, chapters 1–9).
Project evidence: [`knowledge/experience.md`](../knowledge/experience.md) (distilled, score-cited; last
updated 2026-07-04). When a new run adds or overturns a finding in `experience.md`, update the matching
section here and keep the `competition, exp #N, A→B` citation format.
